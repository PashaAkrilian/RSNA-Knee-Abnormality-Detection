#!/usr/bin/env python3
"""Bangun kernel student-train (5-fold ConvNeXt-T, fp16, resume) untuk akun alver.
Pakai: python3 build_student_train.py name=student-train-a folds=0,1
      python3 build_student_train.py name=student-train-b folds=2,3
      python3 build_student_train.py name=student-train-c folds=4"""
import json, os, sys

args = dict(a.split("=", 1) for a in sys.argv[1:])
NAME = args.get("name", "student-train-a")
FOLDS = [int(x) for x in args.get("folds", "0,1").split(",")]
EPOCHS = int(args.get("epochs", "14"))
SLUG = f"rsna-knee-{NAME}"

MD = {
    "id": f"alverlfarfaraway/{SLUG}",
    "title": f"RSNA Knee {NAME}",
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "category": "video",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
        "alverlfarfaraway/rsna-knee-student-cache256",
    ],
    "keywords": [],
    "kernel_sources": [],
    "docker_image": "registry Kaggle Python",
    "machine_shape": " NvidiaTeslaT4",
}

CELL0 = """# RSNA Knee — student-train (Fase B langkah 2b)
# ConvNeXt-T 2.5D, 5 fold, target konsensus⊕teacher (labels_student_v0), fp16 (T4!), EMA, resume per fold.
"""

CELL1 = r'''
import os, sys, glob, json, time, copy, math, shutil, subprocess
import numpy as np, pandas as pd, torch
t0 = time.time()
def say(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

def find(pats, base="/kaggle/input"):
    for p in pats:
        for f in glob.glob(os.path.join(base, "**", p), recursive=True):
            return f
    return None

COMP = None
for c in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(c, "train.csv")): COMP = c; break
assert COMP
PREP = None
for c in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(c, "labels_student_v0.csv")): PREP = c; break
assert PREP, "prep dataset belum ada"
GOODPJW = None
for c in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(c, "knee.py")): GOODPJW = c; break
assert GOODPJW
sys.path.insert(0, GOODPJW)
say("COMP", COMP); say("PREP", PREP); say("GOODPJW", GOODPJW)

whl_dir = os.path.dirname(find(["pylibjpeg-2.1.0-py3-none-any.whl"]) or "")
if whl_dir:
    for w in sorted(glob.glob(os.path.join(whl_dir, "*.whl"))):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", w], check=False)

from knee import LABELS, KneeNet, StudyDataset, build_study_table  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

TRAIN = pd.read_csv(f"{COMP}/train.csv")
SERIES = pd.read_csv(f"{COMP}/train_series.csv")
lab = pd.read_csv(f"{PREP}/labels_student_v0.csv", index_col=0)
CACHE = f"{PREP}/cache256_train"
if os.path.isdir(CACHE) and len(os.listdir(CACHE)) > 1000:
    meta = pd.read_csv(f"{PREP}/cache256_train_meta.csv")
    say("cache dari dataset", CACHE)
else:
    # ---- bangun cache on-demand (resep prep v3: SIZE 256, MM 0.5, MAX 28, IPP sort) ----
    import cv2, pydicom
    from multiprocessing import Pool
    say("cache dataset absen — build on-demand dari DICOM")
    MM_PER_PX, SIZE, MAX_SLICES = 0.5, 256, 28
    CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
    PLANES = ["Sagittal", "Coronal", "Axial"]
    def _read(path):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
        if slope is not None and inter is not None:
            arr = arr * float(slope) + float(inter)
        iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
        ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
        ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
        inst = float(ds.get("InstanceNumber") or 0)
        frames = list(arr) if arr.ndim == 3 else [arr]
        return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]
    def _canon_slice(img, iop, ps, plane):
        ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2
        if iop is None:
            return img, ps
        r, c = np.array(iop[:3]), np.array(iop[3:])
        _, (cax, csgn), (rax, rsgn) = CANON[plane]
        if abs(r[cax]) < abs(r[rax]):
            img, r, c, ps = img.T, c, r, ps[::-1]
        if r[cax] * csgn < 0:
            img = img[:, ::-1]
        if c[rax] * rsgn < 0:
            img = img[::-1]
        return img, ps
    def _resample(img, ps):
        out = np.zeros((SIZE, SIZE), np.float32)
        h, w = img.shape
        ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
        cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
        nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
        nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
        interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
        res = cv2.resize(crop, (nw, nh), interpolation=interp)
        oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
        out[oy:oy + nh, ox:ox + nw] = res
        return out
    def load_series(sdir, plane_name=None):
        slices, bad = [], 0
        for fn in os.listdir(sdir):
            try:
                slices.extend(_read(os.path.join(sdir, fn)))
            except Exception:
                bad += 1
        info = {"n_raw": len(slices), "n_bad": bad}
        if not slices:
            return np.zeros((1, SIZE, SIZE), np.uint8), info
        iop = next((s[1] for s in slices if s[1] is not None), None)
        if iop is not None:
            normal = np.cross(iop[:3], iop[3:]); plane = int(np.abs(normal).argmax())
        else:
            normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
        info["plane"] = PLANES[plane]
        if normal is not None and all(s[2] is not None for s in slices):
            sgn = CANON[plane][0] * np.sign(normal[plane])
            slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
        else:
            slices.sort(key=lambda s: s[4])
        if len(slices) > MAX_SLICES:
            keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
            slices = [slices[i] for i in keep]
        vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
        lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        info["n"] = len(vol)
        return (vol * 255).round().astype(np.uint8), info
    def _job(a):
        study, series, plane = a
        out = f"{CACHE}/{series}.npy"
        info = {"study": study, "series": series}
        if os.path.exists(out):
            return None
        try:
            vol, extra = load_series(f"{COMP}/train_series/{study}/{series}")
            np.save(out + ".tmp.npy", vol); os.replace(out + ".tmp.npy", out)
            info.update(extra)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"[:200]
        return info
    os.makedirs(CACHE, exist_ok=True)
    tasks = [(a, b, p) for a, b, p in zip(SERIES.StudyInstanceUID, SERIES.SeriesInstanceUID, SERIES.Anatomical_Plane)]
    rows = []
    with Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(_job, tasks, chunksize=8)):
            if r: rows.append(r)
            if i % 1000 == 0:
                say(f"cache {i}/{len(tasks)} files={len(os.listdir(CACHE))}")
                if shutil.disk_usage("/tmp").free < 3 * 2**30:
                    say("DISK HAMPIR HENTI"); break
    df = pd.DataFrame(rows)
    if "error" in df: say("errors:", int(df["error"].notna().sum()))
    meta = df.dropna(subset=["n"]).copy()
    say("meta rows", len(meta), "files", len(os.listdir(CACHE)))
say("lab", lab.shape, "meta", meta.shape, "cache files", len(os.listdir(CACHE)))
say("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available()
'''

CELL2 = r'''
import numpy as np, torch, torch.nn.functional as F, time, os
from sklearn.metrics import roc_auc_score

WORK = "/kaggle/working"
RUNS = f"{WORK}/runs_student"
os.makedirs(RUNS, exist_ok=True)
table = build_study_table(SERIES, dict(zip(meta.series, meta.n.fillna(0))))
gold = lab[lab.is_gold == 1]; weak = lab[lab.is_gold == 0]

def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs

@torch.no_grad()
def predict(model, loader, res):
    model.eval(); out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x, mask, pos, res=res).float().cpu())
    return torch.cat(out).numpy()

def train_fold(k, epochs=14, bs=8, kwin=8, keval=12, res=256, lr=1e-4, lr_head=5e-4, wd=0.02, ema_d=0.998):
    ckpt = f"{RUNS}/fold{k}.pt"
    if os.path.exists(ckpt):
        say(f"fold{k} sudah ada, lewati"); return
    torch.manual_seed(0); np.random.seed(0)
    tr = weak[weak.fold != k]; va = weak[weak.fold == k]
    say(f"fold{k}: train {len(tr)} val {len(va)} gold {len(gold)}")
    dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_dl = torch.utils.data.DataLoader(
        StudyDataset(tr.index, table, CACHE, tr[LABELS].values.astype(np.float32), kwin, train=True),
        batch_size=bs, shuffle=True, drop_last=True, **dl_kw)
    eval_dl = {
        "gold": torch.utils.data.DataLoader(StudyDataset(gold.index, table, CACHE, None, keval), batch_size=bs, shuffle=False, **dl_kw),
        "val": torch.utils.data.DataLoader(StudyDataset(va.index, table, CACHE, None, keval), batch_size=bs, shuffle=False, **dl_kw),
    }
    gold_y = gold[[f"gold_{c}" for c in LABELS]].values
    val_y = (va[LABELS].values > 0.5).astype(int)

    model = KneeNet("convnext_tiny", pretrained=False).cuda().to(memory_format=torch.channels_last)
    # warm-start dari ckpt goodpjw publik (internet OFF: tak bisa download ImageNet timm)
    gw = sorted(glob.glob(f"{GOODPJW}/cnxt_v0_fold*.pt"))
    if gw:
        try:
            st0 = torch.load(gw[0], map_location="cpu", weights_only=False)
            sd = st0.get("model", st0)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            say(f"warm-start {os.path.basename(gw[0])}: missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            say("warm-start gagal (train dari awal):", repr(e)[:150])
    else:
        say("tidak ada ckpt goodpjw — pretrained=False, train dari random init (BURUK)")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    enc = list(model.enc.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": head, "lr": lr_head}], weight_decay=wd)
    steps = epochs * len(train_dl); warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler("cuda")   # fp16 + scaler (T4; jangan bf16)

    state = f"{RUNS}/state_fold{k}.pt"
    step, start_ep, preds = 0, 0, None
    if os.path.exists(state):
        st = torch.load(state, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step, start_ep = st["step"], st["epoch"] + 1
        say(f"resume epoch {st['epoch']}")
    for ep in range(start_ep, epochs):
        model.train(); t1, tot, n = time.time(), 0.0, 0
        for x, mask, pos, y in train_dl:
            x, mask, pos, y = x.cuda(non_blocking=True), mask.cuda(), pos.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x, mask, pos, res=res, train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            d = min(ema_d, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            tot += loss.item(); n += 1
            if n % 100 == 0: say(f"  ep{ep} {n}/{len(train_dl)} loss {tot/n:.4f} {(time.time()-t1)/n:.2f}s/it")
        preds = {s: predict(ema, dl, res) for s, dl in eval_dl.items()}
        g_auc, _ = macro_auc(gold_y, preds["gold"]); v_auc, _ = macro_auc(val_y, preds["val"])
        say(f"ep{ep:2d} loss {tot/n:.4f} time {time.time()-t1:.0f}s | gold {g_auc:.4f} | val {v_auc:.4f}")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": ep},
                   state + ".tmp"); os.replace(state + ".tmp", state)
    if preds is None:
        preds = {s: predict(ema, dl, res) for s, dl in eval_dl.items()}
    torch.save({"model": ema.state_dict(), "args": {"k": k}}, ckpt)
    rows = []
    for s, v in (("val", va), ("gold", gold)):
        df = pd.DataFrame(preds[s], index=v.index, columns=LABELS); df.insert(0, "set", s); rows.append(df)
    pd.concat(rows).to_csv(f"{RUNS}/oof_fold{k}.csv")
    if os.path.exists(state): os.remove(state)
    say(f"fold{k} SELESAI gold {macro_auc(gold_y, preds['gold'])[0]:.4f} val {macro_auc(val_y, preds['val'])[0]:.4f}")

DEADLINE = t0 + 7.2 * 3600
FOLDS = __FOLDS__
for k in FOLDS:
    if time.time() > DEADLINE: say(f"deadline utk fold{k} — stop (fold selesai dipakai)"); break
    train_fold(k, epochs=__EPOCHS__)
say("TRAIN_DONE files:", sorted(os.listdir(RUNS)))
'''

def cell(src, ctype="code"):
    return {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if ctype == "code" else {})}

nb = {"cells": [cell(CELL0, "markdown"), cell(CELL1), cell(CELL2)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
# token replace
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["source"] = [ln.replace("__FOLDS__", repr(FOLDS)).replace("__EPOCHS__", repr(EPOCHS)) for ln in c["source"]]
out = f"kaggle_kernel_{NAME}"
os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/{SLUG}.ipynb", "w"), indent=1)
json.dump(MD, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("ditulis:", out, "| folds", FOLDS, "| epochs", EPOCHS)
