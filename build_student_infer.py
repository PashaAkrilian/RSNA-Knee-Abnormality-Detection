#!/usr/bin/env python3
"""Bangun kernel student-infer (K2): inferensi murid (ConvNeXt fold + DINOv2-S) + submission "murid solo".

Pakai: python3 build_student_infer.py

Desain:
  - Mount: kompetisi (test_series/test_images on-the-fly, decode meniru student-prep), prep, tonylica, goodpjw.
  - Ckpt murid dicari TOLERAN: /kaggle/input/**/fold[0-9].pt & /kaggle/input/**/dino_full.pt
    (dataset folds rsna-knee-student-folds BELUM dimount di metadata — ditambah saat push nanti
    oleh agen lain; kalau belum ada kernel tetap jalan dgn model yang tersedia).
  - Output: student_test_preds.csv (uid + 12 kolom cnxt_* + 12 kolom dino_*) + submission.csv
    (mean rank-pct 50/50 per finding) — blend S6+murid = kerjaan offline lokal, BUKAN di sini.
Sel model (DinoSmall/StudentDino) diimpor dari build_student_dino (sumber tunggal).
"""
import json, os
import build_student_dino as bd

SLUG = "rsna-knee-student-infer"
MD = {
    "id": f"alverlfarfaraway/{SLUG}",
    "title": "RSNA Knee student infer (murid solo)",
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "category": "video",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "alverlfarfaraway/rsna-knee-student-prep",
        "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    ],
    "keywords": [],
    "kernel_sources": [],
    "docker_image": "registry Kaggle Python",
    "machine_shape": " NvidiaTeslaT4",
}

CELL0 = """# RSNA Knee — student-infer (Fase B, langkah 2d)
# Inferensi murid solo: ConvNeXt-T (rata2 fold, k_eval=12, res 256, fov .92) + DINOv2-S student
# (k_eval=12, res 224). Test didekode on-the-fly (geometri identik cache student-prep).
# Output: /kaggle/working/student_test_preds.csv (prediksi mentah per model) + submission.csv
# (mean rank-pct 50/50) supaya BISA LANGSUNG disubmit sbg "murid solo".
# Blend dgn S6 dilakukan offline lokal (modifikasi builder S6) — tidak di sini.
"""

CELL1 = r'''
import os, sys, glob, time, json, subprocess, re
import numpy as np, pandas as pd, torch
t0 = time.time()
def say(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
torch.backends.cudnn.benchmark = True

def find_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"] + sorted(glob.glob("/kaggle/input/*"))
    for b in cands:
        if os.path.exists(b + "/test.csv"): return b
        for sub in sorted(glob.glob(b + "/*")):
            if os.path.isdir(sub) and os.path.exists(sub + "/test.csv"): return sub
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f: return d
    raise RuntimeError("test.csv tidak ketemu di /kaggle/input")

ROOT = find_root()
TSDIR = next((f"{ROOT}/{c}" for c in ("test_series", "test_images") if os.path.isdir(f"{ROOT}/{c}")), None)
assert TSDIR, f"folder test_series/test_images tidak ada di {ROOT}: {sorted(os.listdir(ROOT))[:20]}"
test = pd.read_csv(f"{ROOT}/test.csv"); test["StudyInstanceUID"] = test.StudyInstanceUID.astype(str)
tser = pd.read_csv(f"{ROOT}/test_series.csv")
tser["StudyInstanceUID"] = tser.StudyInstanceUID.astype(str)
tser["SeriesInstanceUID"] = tser.SeriesInstanceUID.astype(str)
say("ROOT", ROOT); say("TSDIR", TSDIR); say("studi", len(test), "series", len(tser))

GOODPJW = next((c for c in sorted(glob.glob("/kaggle/input/*")) if os.path.exists(c + "/knee.py")), None)
assert GOODPJW, "dataset goodpjw (knee.py) tidak ada"
DINO = next((os.path.join(c, "dinov2-small") for c in sorted(glob.glob("/kaggle/input/*"))
             if os.path.exists(os.path.join(c, "dinov2-small", "pytorch_model.bin"))), None)
sys.path.insert(0, GOODPJW)
say("GOODPJW", GOODPJW); say("DINO", DINO)

# pylibjpeg utk decode cepat (wheel dari dataset goodpjw, sama dgn student-prep)
whl_dir = os.path.dirname(next(iter(glob.glob(f"{GOODPJW}/pylibjpeg*.whl")), "") or "")
if whl_dir:
    for w in sorted(glob.glob(os.path.join(whl_dir, "*.whl"))):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", w], check=False)
say("pylibjpeg:", subprocess.run([sys.executable, "-m", "pip", "show", "pylibjpeg"], capture_output=True).returncode == 0)

from knee import LABELS, KneeNet, StudyDataset, build_study_table  # noqa: E402

# ckpt murid — cari toleran di semua mount (folds akan dimount sbg dataset tambahan saat push nanti)
FOLDS = sorted(p for p in glob.glob("/kaggle/input/**/fold*.pt", recursive=True)
               if re.fullmatch(r"fold\d+\.pt", os.path.basename(p)))
DINO_CK = sorted(glob.glob("/kaggle/input/**/dino_full.pt", recursive=True))
say("fold ckpt:", FOLDS); say("dino ckpt:", DINO_CK)
assert FOLDS or DINO_CK, "tak ada checkpoint murid — mount dataset folds / dino"

SERIES_DIR, counts = {}, {}
for st in tser.StudyInstanceUID.unique():
    sd = f"{TSDIR}/{st}"
    if not os.path.isdir(sd): continue
    for sr in os.listdir(sd):
        p = f"{sd}/{sr}"
        if os.path.isdir(p):
            SERIES_DIR[sr] = p
            counts[sr] = len(os.listdir(p))
table = build_study_table(tser, counts)
test_ids = test.StudyInstanceUID.tolist()
say("series terpetakan:", len(SERIES_DIR), "| studi dgn slot valid:", len(table),
    "| studi tanpa series valid:", len([s for s in test_ids if s not in table]))
'''

CELL3 = r'''
import cv2, pydicom, numpy as np

# === salinan fungsi decode student-prep (cell2) — geometri cache IDENTIK dgn training ===
MM_PER_PX = 0.5
SIZE = 256
MAX_SLICES = 28

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
        sgn = CANON[plane][0] * np.sign(normal[plane])   # urut by IPP (bukan filename)
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

# cache volume per-worker (DataLoader fork) — cukup utk beberapa series sekaligus
_volcache = {}
def loader(sr):
    v = _volcache.get(sr)
    if v is None:
        v, info = load_series(SERIES_DIR[sr])
        if len(_volcache) > 256: _volcache.clear()
        _volcache[sr] = v
    return v

say("fungsi decode siap; SERIES_DIR", len(SERIES_DIR))
'''

CELL4 = r'''
import torch
assert torch.cuda.is_available()
CNXT = []
for f in FOLDS:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    m = KneeNet("convnext_tiny", pretrained=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (f, missing[:5], unexpected[:5])
    CNXT.append(m.cuda().eval().to(memory_format=torch.channels_last))
    say("ConvNeXt", os.path.basename(f), "%.1fM param" % (sum(p.numel() for p in m.parameters()) / 1e6))
    del ck, sd
DINO_M = None
if DINO_CK and DINO:
    ck = torch.load(DINO_CK[0], map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    m = StudentDino(DINO)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    DINO_M = m.cuda().eval()
    say("DINO student <-", DINO_CK[0], "%.1fM param" % (sum(p.numel() for p in m.parameters()) / 1e6))
    del ck, sd
assert CNXT or DINO_M is not None
torch.cuda.empty_cache()
say("model siap | convnext folds", len(CNXT), "| dino", DINO_M is not None)
'''

CELL5 = r'''
from torch.utils.data import DataLoader

N, N_T = len(test_ids), len(LABELS)
P = np.zeros((2, N, N_T), np.float32)      # 0 = ConvNeXt (rata2 fold), 1 = DINOv2 student
ok = np.zeros(N, bool)
KEVAL = 12
dl = DataLoader(StudyDataset(test_ids, table, None, None, KEVAL, False, loader=loader),
                batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
DEADLINE = t0 + 8.0 * 3600
i = 0
for x, mask, pos, _ in dl:
    if time.time() > DEADLINE:
        say(f"deadline 8 jam — stop di studi {i}/{N}; sisanya diisi netral"); break
    try:
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if CNXT:
                p = sum(torch.sigmoid(m(x, mask, pos, res=256)) for m in CNXT) / len(CNXT)
                P[0, i] = p.float().cpu().numpy()
            if DINO_M is not None:
                P[1, i] = torch.sigmoid(DINO_M(x, mask, pos, res=224)).float().cpu().numpy()
        ok[i] = True
    except Exception as e:
        say(f"GAGAL studi {i} {test_ids[i][:24]}: {type(e).__name__} {str(e)[:120]}")
    i += 1
    if i % 20 == 0 or i == N:
        say(f"{i}/{N} studi | ok {int(ok.sum())} | {time.time()-t0:.0f}s")
N_DONE = i
say(f"inferensi selesai: {int(ok.sum())}/{N_DONE} studi valid (dari {N})")
'''

CELL6 = r'''
present = ([0] if CNXT else []) + ([1] if DINO_M is not None else [])
full = np.full((2, N, N_T), np.nan, np.float32)
full[:, :N_DONE] = P[:, :N_DONE]
for mi in present:
    good = np.isfinite(full[mi]).all(1)
    fill = full[mi][good].mean(0) if good.any() else np.full(N_T, 0.5, np.float32)
    full[mi][~good] = np.where(np.isfinite(fill), fill, 0.5)
    say(f"model {'cnxt' if mi == 0 else 'dino'}: {int(good.sum())}/{N} studi valid, sisanya diisi rata-rata kolom")

def rankpct(x): return x.argsort(0).argsort(0).astype(np.float64) / max(1, len(x) - 1)
R = np.mean([rankpct(np.clip(full[mi], 0, 1)) for mi in present], axis=0)

out = pd.DataFrame({"StudyInstanceUID": test_ids})
for mi, tag in ((0, "cnxt"), (1, "dino")):
    if mi in present:
        for j, c in enumerate(LABELS):
            out[f"{tag}_{c}"] = full[mi][:, j].astype(np.float32)
out.to_csv("/kaggle/working/student_test_preds.csv", index=False)
say("student_test_preds.csv", out.shape)

sub_cols = ["StudyInstanceUID"] + LABELS
ssub = f"{ROOT}/sample_submission.csv"
if os.path.exists(ssub):
    sub_cols = list(pd.read_csv(ssub, nrows=1).columns)
sub = pd.DataFrame(R.astype(np.float32), columns=LABELS)
sub.insert(0, "StudyInstanceUID", test_ids)
missing_cols = [c for c in sub_cols if c not in sub.columns]
assert not missing_cols, f"kolom submission tak cocok sample_submission: {missing_cols}"
sub = sub[sub_cols]
assert sub["StudyInstanceUID"].tolist() == test_ids and np.isfinite(sub[LABELS].values).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
say("submission.csv", sub.shape, "| model dipakai:", ["cnxt" if m == 0 else "dino" for m in present])
print(sub.head().to_string(index=False))
say("SUBMISSION SIAP (murid solo; blend S6+murid = tahap offline lokal)")
'''


def cell(src, ctype="code"):
    c = {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True)}
    if ctype == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


def main():
    nb = {"cells": [cell(CELL0, "markdown"), cell(CELL1), cell(bd.CELL_MODEL), cell(CELL3),
                    cell(CELL4), cell(CELL5), cell(CELL6)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"<cell{i}>", "exec")   # validasi syntactic
    out = "kaggle_kernel_student_infer"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(f"{out}/{SLUG}.ipynb", "w"), indent=1)
    json.dump(MD, open(f"{out}/kernel-metadata.json", "w"), indent=2)
    print("ditulis:", out, "| cells:", len(nb["cells"]), "| id:", MD["id"])


if __name__ == "__main__":
    main()
