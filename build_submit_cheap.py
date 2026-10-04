#!/usr/bin/env python3
"""Kernel SUBMISSION utk model murah (CNN kecil, K-fold) dari build_train_cheap_kfold.py -- SOLO (tanpa pipeline publik).

TUJUAN: ukur LB nyata ensemble fold murah (ResNet-50 224px dst) apple-to-apple dgn c5-solo 0,927 / v3-solo 0,932.
Studi test didekode SEKALI (DICOM -> volume 64 irisan 336px, geometri cache `w`: SLOTS64, rentang 2-98%) lalu SEMUA model
fold dijalankan pada window yg sama (harness lama `main()` mendekode ulang per arm: 6x lebih mahal utk 6 fold).
Resize 336->res memakai antialias=True (SAMA dgn latihan). Ensemble = rata-rata RANK persentil lintas model.
Pengaman: studi yg gagal didekode diisi dgn nilai netral (0,5) + dicatat; prediksi mentah per model disimpan.

Pemakaian: python3 build_submit_cheap.py <train_slug> <outtag> <nfold> <res> [k_eval=42] [name=<outtag>]
  contoh : python3 build_submit_cheap.py rsna-knee-cheap-resnet50-s1 cheap_resnet50_s1 6 224
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN_SLUG, OUTTAG, NFOLD, RES = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
K_EVAL = int(sys.argv[5]) if len(sys.argv) > 5 else 42
NAME = sys.argv[6] if len(sys.argv) > 6 else OUTTAG
SLUG = f"rsna-knee-submit-{NAME}".replace("_", "-")
OUT = REPO / f"kaggle_kernel_submit_{NAME.replace('-', '_')}"

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()
OLD = "lo, hi = (int(n * 0.06), int(n * 0.94) - 1)"
assert OLD in ke_src
ke_src = ke_src.replace(OLD, "lo, hi = (int(n * 0.02), int(n * 0.98) - 1)")      # geometri cache 'w' (rentang 2-98%)
assert "IMG = 336" in ke_src and "SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]" in ke_src

MAIN = r'''
import os, sys, time, glob, json, collections
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
KE = {'__name__': '_ke'}
exec(compile(KE_SRC, '<ke>', 'exec'), KE)
LAB = KE['LAB']; N_T = len(LAB)
MEAN, STD = KE['_MEAN'], KE['_STD']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
tlog(f'device={DEV} gpus={torch.cuda.device_count()} | outtag={OUTTAG} nfold={NFOLD} res={RES} k_eval={K_EVAL}')

def _bglob(name):
    return [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/')
            for p in glob.glob(d + name)]
def resize(t, res):
    if t.shape[-1] != res:
        t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False, antialias=True)[0]
    return t
def eval_windows_aa(vol, mask, k, res):
    D = vol.shape[0]
    cs = KE['_eval_centers'](mask, D, k)
    wins = torch.empty((len(cs), 3, res, res), dtype=torch.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        t = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        wins[j] = (resize(t, res) - MEAN) / STD
    return wins

models = []
for f in range(NFOLD):
    hits = sorted(set(os.path.realpath(p) for p in _bglob(f'{OUTTAG}_f{f}.pt')))
    assert len(hits) == 1, f'checkpoint fold {f}: {hits}'
    ck = torch.load(hits[0], map_location='cpu', weights_only=False)
    bb = KE['build_backbone'](ck['arch'], pretrained=False)
    m = KE['RaptorClassifier'](bb, F_dim=bb.num_features)
    m.load_state_dict(ck['model'], strict=True)
    assert int(ck['res']) == RES, f"res checkpoint {ck['res']} != {RES}"
    models.append(m.eval().to(DEV)); del ck
    tlog(f'fold {f}: {hits[0]}')

ROOT = KE['find_test_root']()
tsdir = ROOT + '/test_series'
if not os.path.isdir(tsdir): tsdir = ROOT + '/test_images'
test = pd.read_csv(ROOT + '/test.csv'); test['StudyInstanceUID'] = test['StudyInstanceUID'].astype(str)
test_ids = test['StudyInstanceUID'].tolist()
tser = pd.read_csv(ROOT + '/test_series.csv')
tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str); tser['SeriesInstanceUID'] = tser['SeriesInstanceUID'].astype(str)
SER = {k: v.to_dict('records') for k, v in tser.groupby('StudyInstanceUID')}
sub_cols = ['StudyInstanceUID'] + LAB
ssub = os.path.join(ROOT, 'sample_submission.csv')
if os.path.exists(ssub): sub_cols = list(pd.read_csv(ssub, nrows=1).columns)
reader = KE['_make_reader']()
N = len(test_ids)
tlog(f'studi test={N} | series={len(tser)}')

def prep(sid):
    vol, mask = KE['build_study'](sid, SER, tsdir, reader)
    return eval_windows_aa(vol, mask, K_EVAL, RES)

P = np.full((len(models), N, N_T), np.nan, np.float32)
failed = []
PREFETCH = 6
with ThreadPoolExecutor(max_workers=3) as ex:
    q = collections.deque()
    nxt = 0
    for i in range(N):
        while nxt < N and len(q) < PREFETCH:
            q.append((nxt, ex.submit(prep, test_ids[nxt]))); nxt += 1
        j, fut = q.popleft()
        try:
            xw = fut.result().unsqueeze(0).to(DEV)
            with torch.no_grad():
                for mi, m in enumerate(models):
                    with torch.autocast('cuda', dtype=torch.float16, enabled=(DEV == 'cuda')):
                        P[mi, j] = torch.sigmoid(m(xw).float())[0].cpu().numpy()
            del xw
        except Exception as e:
            failed.append(test_ids[j]); tlog(f'GAGAL studi {j} {test_ids[j][:20]}: {type(e).__name__}: {str(e)[:120]}')
        if (i + 1) % 100 == 0 or i + 1 == N:
            tlog(f'{i+1}/{N} studi | gagal={len(failed)}')

np.savez_compressed('/kaggle/working/cheap_raw.npz', study_uids=np.asarray(test_ids), raw=P)
# nilai netral utk studi gagal (per model): rata-rata kolom studi lain
for mi in range(len(models)):
    bad = ~np.isfinite(P[mi]).all(axis=1)
    if bad.any():
        fill = np.nanmean(np.where(np.isfinite(P[mi]), P[mi], np.nan), axis=0) if (~bad).any() else np.full(N_T, .5, np.float32)
        P[mi][bad] = np.where(np.isfinite(fill), fill, .5)
def rankpct(x): return x.argsort(0).argsort(0).astype(np.float64) / max(1, len(x) - 1)
R = np.mean([rankpct(np.clip(P[mi], 0, 1)) for mi in range(len(models))], axis=0)
sub = pd.DataFrame(R.astype(np.float32), columns=LAB); sub.insert(0, 'StudyInstanceUID', test_ids); sub = sub[sub_cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[LAB].values).all()
sub.to_csv('/kaggle/working/submission.csv', index=False)
tlog(f'submission.csv {sub.shape} | gagal={len(failed)}')
print(sub.head().to_string(index=False))
tlog('SUBMISSION SIAP')
'''

cells = ["KE_SRC = " + repr(ke_src) + f"\nOUTTAG = {OUTTAG!r}\nNFOLD = {NFOLD}\nRES = {RES}\nK_EVAL = {K_EVAL}\n", MAIN]
for c in cells:
    compile(c, "<cell>", "exec")
nb = {"cells": [{"cell_type": "code", "source": c.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": f"dimaspashaakrilian/{SLUG}", "title": f"RSNA Knee - Submit {NAME}", "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "kernel_sources": [f"dimaspashaakrilian/{TRAIN_SLUG}"],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| sumber ckpt:", TRAIN_SLUG)
