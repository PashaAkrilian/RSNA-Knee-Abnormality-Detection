#!/usr/bin/env python3
"""Evaluasi BANYAK checkpoint c5-geometri sekaligus di 58 gold (cache w: 336px/64 irisan/rentang 2-98%),
pada dua jumlah window (42 & 94). Pemakaian: build_eval_multi.py <tag> <kernel_sumber> <ckpt1,ckpt2,...>"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent
tag, ksrc, ckpts = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
ke = (REPO / "build_train_coatnet_ke_src.txt").read_text()
assert "IMG = 336" in ke and "K_EVAL = 42" in ke
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
Y = train.set_index('StudyInstanceUID').loc[GOLD, LAB].values.astype(float)
def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
def find(fname):
    hits = []
    for pat in ('/kaggle/input/*/' + fname, '/kaggle/input/*/*/' + fname, '/kaggle/input/*/*/*/' + fname, '/kaggle/input/*/*/*/*/' + fname):
        hits += glob.glob(pat)
    hits = sorted(set(hits)); assert len(hits) == 1, (fname, hits); return hits[0]
vols = {sid: np.load(MOUNTED[sid]) for sid in GOLD}
vols = {sid: (z['vol'], z['mask']) for sid, z in vols.items()}
for ck in __CKPTS__:
    model, RES = KE['load_model'](find(ck), 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, 'cuda')
    for k in (42, 94):
        P = np.stack([KE['infer_probs'](model, KE['eval_windows'](vols[s][0], vols[s][1], k=k, res=RES, norm=KE['NORM']), 'cuda') for s in GOLD])
        out = pd.DataFrame(P, columns=LAB); out.insert(0, 'StudyInstanceUID', GOLD)
        out.to_csv(f'/kaggle/working/eval_{ck[:-3]}_k{k}.csv', index=False)
        a = [auc(Y[:, j], P[:, j]) for j in range(len(LAB))]
        tlog(f'MACRO gold-58 [{ck} | k={k}]: {np.mean(a):.4f}')
    del model
tlog('EVAL SELESAI')
'''.replace("__KE__", repr(ke)).replace("__CKPTS__", repr(ckpts))
out = REPO / f"kaggle_kernel_eval_{tag}"; out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "source": CELL.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(out / f"eval_{tag}.ipynb").write_text(json.dumps(nb, indent=1))
(out / "kernel-metadata.json").write_text(json.dumps({"id": f"dimaspashaakrilian/rsna-knee-eval-{tag}", "title": f"RSNA Knee - Eval {tag}",
    "code_file": f"eval_{tag}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-w{i}" for i in range(3)] + [ksrc],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built:", out.name)
