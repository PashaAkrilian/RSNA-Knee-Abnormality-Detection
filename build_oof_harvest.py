#!/usr/bin/env python3
"""Build rsna-knee-oof-harvest-v1 kernel (ipynb + metadata) lalu siap di-push.
Ekstrak _KE_SRC verbatim dari notebook raptor publik supaya identik."""
import ast, json, re
from pathlib import Path

REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")
OUT = REPO / "kaggle_kernel_oof_harvest_v1"
SRC_TXT = Path("/tmp/opencode/nb/raptor45w65/rsna-knee-raptor-v45-w065.txt")

line = [l for l in SRC_TXT.read_text().splitlines() if l.startswith("_KE_SRC")][0]
ke_src = ast.literal_eval(line.split("=", 1)[1].strip())

SETUP = '''\
import os, sys, subprocess, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
tlog('setup start')

# timm versi baru dibutuhkan coatnet_rmlp_2_rw_384; fallback ke wheel lokal kalau perlu
def timm_ok():
    try:
        import timm, torch
        m = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
                              pretrained=False, num_classes=0, in_chans=3)
        del m
        return True
    except Exception as e:
        tlog(f'timm bawaan gagal: {type(e).__name__}: {e}')
        return False

if not timm_ok():
    hits = []
    for d in ('/kaggle/input/knee-mri-fold-weights', '/kaggle/input'):
        hits += __import__('glob').glob(d + '/**/timm-*.whl', recursive=True)
    assert hits, 'wheel timm tidak ketemu'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--no-deps', hits[0]], check=True)
    tlog(f'timm dipasang dr {hits[0]}')
assert timm_ok(), 'timm tetap tidak siap'
tlog('setup OK')
'''

DRIVER = '''\
import glob
import numpy as np
import pandas as pd

KE = _KE_NS
LAB = KE['LAB']
dev = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# --- data train (konvensi path sama spt test_series) ---
ROOT = KE['find_test_root']()
tsdir = ROOT + '/train_series'
assert os.path.isdir(tsdir), f'train_series tidak ada di {ROOT}'
train = pd.read_csv(ROOT + '/train.csv')
gold_ids = train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'].astype(str).tolist()
tser = pd.read_csv(ROOT + '/train_series.csv')
for c in ('StudyInstanceUID', 'SeriesInstanceUID'):
    tser[c] = tser[c].astype(str)
SER = {k: v[['SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive']].to_dict('records')
       for k, v in tser.groupby('StudyInstanceUID')}
tlog(f'root={ROOT} | gold {len(gold_ids)} studi | series {len(tser)} baris')

ARMS = [
    {'file': 'raptor_ft_coatnet_v5_full_swa.pt',
     'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384},
    {'file': 'raptor_ft_coatnet_v4_full.pt',
     'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384},
    {'file': 'raptor_ft_coatnet_v4_full_swa.pt',
     'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384},
]

def macro_auc(y, s):
    aucs = []
    for j in range(y.shape[1]):
        pos, neg = s[y[:, j] == 1, j], s[y[:, j] == 0, j]
        if len(pos) == 0 or len(neg) == 0:
            continue
        import pandas as _pd
        r = _pd.Series(np.concatenate([pos, neg])).rank().values
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs))

y_true = train[train[LAB].notna().all(axis=1)][LAB].values.astype(float)
reader = KE['_make_reader']()
N = len(gold_ids)

for arm in ARMS:
    wp = KE['find_weight_file'](arm['file'])
    model, res = KE['load_model'](wp, arm['arch'], arm['res'], dev)
    tlog(f"[{arm['file']}] loaded | res {res}")
    probs = np.full((N, len(LAB)), 0.5, np.float32)
    fail = 0
    for i, sid in enumerate(gold_ids):
        try:
            vol, mask = KE['build_study'](sid, SER, tsdir, reader)
            xw = KE['eval_windows'](vol, mask, k=KE['K_EVAL'], res=res, norm=KE['NORM'])
            probs[i] = KE['infer_probs'](model, xw, dev)
            del vol, mask, xw
        except Exception as e:
            fail += 1
            tlog(f"  FALLBACK {i} {sid[:20]} ({type(e).__name__}: {e})")
        if (i + 1) % 10 == 0 or i + 1 == N:
            tlog(f"  {i+1}/{N} | fail={fail} | {time.time()-t0:.0f}s")
    tag = arm['file'].replace('raptor_ft_coatnet_', '').replace('.pt', '')
    out = pd.DataFrame(probs, columns=LAB)
    out.insert(0, 'StudyInstanceUID', gold_ids)
    fn = f'/kaggle/working/oof_coatnet_{tag}.csv'
    out.to_csv(fn, index=False)
    auc = macro_auc(y_true, probs)
    tlog(f"[{arm['file']}] SELESAI fail={fail} | macro AUC vs gold = {auc:.4f} -> {fn}")
    del model
    import torch, gc
    if str(dev).startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()

tlog('ALL DONE')
'''

nb = {
    "cells": [
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": SETUP.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["_KE_SRC = " + repr(ke_src) + "\n",
                    "_KE_NS = {'__name__': '_ke'}\n",
                    "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n",
                    "print('_KE loaded:', sorted(k for k in _KE_NS if not k.startswith('__'))[:8])"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": DRIVER.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
}

OUT.mkdir(exist_ok=True)
(OUT / "oof_harvest.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "dimaspashaakrilian/rsna-knee-oof-harvest-v1",
    "title": "RSNA Knee - OOF harvest CoAtNet gold58",
    "code_file": "oof_harvest.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-widedense",
        "mattiaangeli/knee-mri-fold-weights",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
