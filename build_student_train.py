#!/usr/bin/env python3
"""Bangun kernel student-train (5-fold ConvNeXt-T, fp16, resume) untuk akun alver."""
import json, os

MD = {
    "id": "alverlfarfaraway/rsna-knee-student-train",
    "title": "RSNA Knee student train",
    "code_file": "rsna-knee-student-train.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "category": "video",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
        "alverlfarfaraway/rsna-knee-student-prep",
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
import os, sys, glob, json, time, copy, math, shutil
import numpy as np, pandas as pd, torch
t0 = time.time()
def say(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

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

from knee import LABELS, KneeNet, StudyDataset, build_study_table  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

TRAIN = pd.read_csv(f"{COMP}/train.csv")
SERIES = pd.read_csv(f"{COMP}/train_series.csv")
lab = pd.read_csv(f"{PREP}/labels_student_v0.csv", index_col=0)
meta = pd.read_csv(f"{PREP}/cache256_train_meta.csv")
CACHE = f"{PREP}/cache256_train"
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

    model = KneeNet("convnext_tiny").cuda().to(memory_format=torch.channels_last)
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
for k in range(5):
    if time.time() > DEADLINE: say(f"deadline utk fold{k} — stop (fold selesai dipakai)"); break
    train_fold(k)
say("TRAIN_DONE files:", sorted(os.listdir(RUNS)))
'''

def cell(src, ctype="code"):
    return {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if ctype == "code" else {})}

nb = {"cells": [cell(CELL0, "markdown"), cell(CELL1), cell(CELL2)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = "kaggle_kernel_student_train"
os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/rsna-knee-student-train.ipynb", "w"), indent=1)
json.dump(MD, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("ditulis:", out)
