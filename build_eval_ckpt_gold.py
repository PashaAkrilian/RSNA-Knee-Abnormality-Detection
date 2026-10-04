#!/usr/bin/env python3
"""Kernel evaluasi: skor checkpoint arm CoAtNet mana pun di 58 gold, memakai harness & cache yg SAMA
dgn latihan (geometri cocok), supaya perbandingan stok-vs-hasil-kita apple-to-apple.

Pemakaian: python3 build_eval_ckpt_gold.py <tag> <ckpt_file> <dataset_ckpt|-> <cache_tag w|x|y> <img> <SLOTS64|SLOTS44> <k_eval> [kernel_ckpt_source]
Contoh stok v8: build_eval_ckpt_gold.py stockv8 raptor_ft_coatnet_v8_full_swa.pt dreaddevelopment/raptor-knee-native384 y 384 SLOTS44 42
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent
tag, ckpt, ds, ctag, img, slots, kev = sys.argv[1:8]
ksrc = sys.argv[8] if len(sys.argv) > 8 else None
SL = {"SLOTS64": "[('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]",
      "SLOTS44": "[('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), ('Coronal', 0, 6), ('Axial', -1, 8)]"}
ke = (REPO / "build_train_coatnet_ke_src.txt").read_text()
def must(t, a, b):
    assert a in t, a[:60]; return t.replace(a, b, 1)
ke = must(ke, "IMG = 336", f"IMG = {img}")
ke = must(ke, "SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]", f"SLOTS = {SL[slots]}")
ke = must(ke, "K_EVAL = 42", f"K_EVAL = {kev}")
CELL = '''\
import os, sys, time, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1')
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
import numpy as np, pandas as pd
_KE_SRC = __KE__
KE = {'__name__': '_ke'}; exec(compile(_KE_SRC, '<ke>', 'exec'), KE)
LAB = KE['LAB']
ROOT = KE['find_test_root']()
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
GOLD = sorted(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
MOUNTED = {}
for d in (glob.glob('/kaggle/input/*/volcache') + glob.glob('/kaggle/input/*/*/volcache') + glob.glob('/kaggle/input/*/*/*/volcache')):
    for f in os.listdir(d):
        if f.endswith('.npz'): MOUNTED[f[:-4]] = os.path.join(d, f)
tlog(f'cache termount: {len(MOUNTED)} | gold: {len(GOLD)}')
wp = KE['find_weight_file']('__CKPT__')
tlog('ckpt: ' + wp)
model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, 'cuda')
preds = []
for sid in GOLD:
    z = np.load(MOUNTED[sid]); vol, mask = z['vol'], z['mask']
    xw = KE['eval_windows'](vol, mask, k=KE['K_EVAL'], res=RES, norm=KE['NORM'])
    preds.append(KE['infer_probs'](model, xw, 'cuda'))
P = np.stack(preds)
out = pd.DataFrame(P, columns=LAB); out.insert(0, 'StudyInstanceUID', GOLD)
out.to_csv('/kaggle/working/eval___TAG___gold_preds.csv', index=False)
Y = train.set_index('StudyInstanceUID').loc[GOLD, LAB].values.astype(float)
def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
a = [auc(Y[:, j], P[:, j]) for j in range(len(LAB))]
for l, v in zip(LAB, a): tlog(f'  {l:18s} {v:.4f}')
tlog(f'MACRO AUC gold-58 [__TAG__ | __CKPT__]: {np.mean(a):.4f}')
'''.replace("__KE__", repr(ke)).replace("__CKPT__", ckpt).replace("__TAG__", tag)
out = REPO / f"kaggle_kernel_eval_{tag}"; out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "source": CELL.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(out / f"eval_{tag}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"dimaspashaakrilian/rsna-knee-eval-{tag}", "title": f"RSNA Knee - Eval {tag}", "code_file": f"eval_{tag}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [x for x in [ds if ds != "-" else None] if x],
        "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-{ctag}{i}" for i in range(3)] + ([ksrc] if ksrc else []),
        "competition_sources": ["rsna-knee-abnormality-detection"]}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", out.name)
