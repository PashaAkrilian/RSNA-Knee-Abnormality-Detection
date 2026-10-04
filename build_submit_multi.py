#!/usr/bin/env python3
"""Kernel SUBMISSION SOLO multi-model umum (CoAtNet 384 / CNN murah apa pun) -- tanpa pipeline publik.

Studi test didekode SEKALI (DICOM -> volume 64 irisan 336px; geometri cache lewat `span`), lalu semua model dijalankan
pada window yang sama; model dibagi bergilir ke semua GPU yang terlihat (2 x T4). Resize ke res tiap model dgn
antialias=True (sama dgn latihan model murah; utk upsampling 336->384 identik dgn bilinear biasa).
Ensemble = rata-rata RANK persentil lintas model (sama dgn harness lama `main()`).

Argumen berbentuk kunci=nilai:
  name=      nama kernel (slug rsna-knee-submit-<name>)
  slugs=     slug kernel sumber checkpoint (tanpa pemilik), dipisah koma
  files=     nama berkas checkpoint (satu per model, urutan bebas), dipisah koma
  span=      lo,hi rentang irisan (default 0.06,0.94 = cache 's'; 0.02,0.98 = cache 'w')
  k_eval=    jumlah window per studi (default 42)
  weights=   bobot per model (opsional, dipisah koma; default sama)
Contoh (4 murid fold CoAtNet, cache 's'):
  python3 build_submit_multi.py name=students4 slugs=rsna-knee-kfold-c5-f0,rsna-knee-kfold-c5-f1,rsna-knee-kfold-c5-f2,rsna-knee-kfold-c5-f3 \
      files=kfold_c5_f0.pt,kfold_c5_f1.pt,kfold_c5_f2.pt,kfold_c5_f3.pt span=0.06,0.94
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
A = dict(kv.split("=", 1) for kv in sys.argv[1:])
NAME = A["name"]
SLUGS = [s for s in A["slugs"].split(",") if s]
FILES = [s for s in A["files"].split(",") if s]
LO, HI = [float(x) for x in A.get("span", "0.06,0.94").split(",")]
K_EVAL = int(A.get("k_eval", 42))
WEIGHTS = [float(x) for x in A["weights"].split(",")] if A.get("weights") else [1.0] * len(FILES)
assert len(WEIGHTS) == len(FILES)
SLUG = f"rsna-knee-submit-{NAME}".replace("_", "-")
OUT = REPO / f"kaggle_kernel_submit_{NAME.replace('-', '_')}"

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()
OLD = "lo, hi = (int(n * 0.06), int(n * 0.94) - 1)"
assert OLD in ke_src
ke_src = ke_src.replace(OLD, f"lo, hi = (int(n * {LO}), int(n * {HI}) - 1)")
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
NORMS = {'imagenet': (KE['_MEAN'], KE['_STD']), 'rad': (torch.full((3, 1, 1), 0.5), torch.full((3, 1, 1), 0.5))}
NG = torch.cuda.device_count()
DEVS = [f'cuda:{i}' for i in range(NG)] if NG else ['cpu']
tlog(f'devices={DEVS} | berkas={FILES} | span={SPAN} k_eval={K_EVAL}')

def _bglob(name):
    return [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/')
            for p in glob.glob(d + name)]
def resize(t, res):
    if t.shape[-1] != res:
        t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False, antialias=True)[0]
    return t
def eval_windows_aa(vol, mask, k, res, norm='imagenet'):
    D = vol.shape[0]
    MEAN, STD = NORMS[norm]
    cs = KE['_eval_centers'](mask, D, k)
    wins = torch.empty((len(cs), 3, res, res), dtype=torch.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        t = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        wins[j] = (resize(t, res) - MEAN) / STD
    return wins

models = []     # (model, device, res)
for i, fname in enumerate(FILES):
    hits = sorted(set(os.path.realpath(p) for p in _bglob(fname)))
    assert len(hits) == 1, f'{fname}: {hits}'
    ck = torch.load(hits[0], map_location='cpu', weights_only=False)
    bb = KE['build_backbone'](ck['arch'], pretrained=False)
    m = KE['RaptorClassifier'](bb, F_dim=bb.num_features)
    m.load_state_dict(ck['model'], strict=True)
    dev = DEVS[i % len(DEVS)]
    models.append((m.eval().to(dev), dev, (int(ck['res']), ck.get('norm', 'imagenet')))); del ck
    tlog(f'model {i}: {fname} -> {dev} res/norm={models[-1][2]} | {hits[0]}')
RESES = sorted({r for _, _, r in models})   # pasangan (res, norm)

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
    return {r: eval_windows_aa(vol, mask, K_EVAL, r[0], r[1]) for r in RESES}

P = np.full((len(models), N, N_T), np.nan, np.float32)
failed = []
def run_dev(dev, wins, j):
    with torch.no_grad():
        for mi, (m, d, r) in enumerate(models):
            if d != dev: continue
            xw = wins[r].unsqueeze(0).to(d)
            with torch.autocast('cuda', dtype=torch.float16, enabled=d.startswith('cuda')):
                P[mi, j] = torch.sigmoid(m(xw).float())[0].cpu().numpy()
PREFETCH = 8
with ThreadPoolExecutor(max_workers=3) as ex, ThreadPoolExecutor(max_workers=len(DEVS)) as gx:
    q = collections.deque(); nxt = 0
    for i in range(N):
        while nxt < N and len(q) < PREFETCH:
            q.append((nxt, ex.submit(prep, test_ids[nxt]))); nxt += 1
        j, fut = q.popleft()
        try:
            wins = fut.result()
            list(gx.map(lambda d: run_dev(d, wins, j), DEVS))
            del wins
        except Exception as e:
            failed.append(test_ids[j]); tlog(f'GAGAL studi {j} {test_ids[j][:20]}: {type(e).__name__}: {str(e)[:120]}')
        if (i + 1) % 100 == 0 or i + 1 == N:
            tlog(f'{i+1}/{N} studi | gagal={len(failed)}')

np.savez_compressed('/kaggle/working/multi_raw.npz', study_uids=np.asarray(test_ids), raw=P, files=np.asarray(FILES))
for mi in range(len(models)):
    bad = ~np.isfinite(P[mi]).all(axis=1)
    if bad.any():
        fill = np.nanmean(np.where(np.isfinite(P[mi]), P[mi], np.nan), axis=0) if (~bad).any() else np.full(N_T, .5, np.float32)
        P[mi][bad] = np.where(np.isfinite(fill), fill, .5)
def rankpct(x): return x.argsort(0).argsort(0).astype(np.float64) / max(1, len(x) - 1)
w = np.asarray(WEIGHTS, np.float64); w = w / w.sum()
R = np.tensordot(w, np.stack([rankpct(np.clip(P[mi], 0, 1)) for mi in range(len(models))]), axes=(0, 0))
sub = pd.DataFrame(R.astype(np.float32), columns=LAB); sub.insert(0, 'StudyInstanceUID', test_ids); sub = sub[sub_cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[LAB].values).all()
sub.to_csv('/kaggle/working/submission.csv', index=False)
tlog(f'submission.csv {sub.shape} | gagal={len(failed)} | bobot={dict(zip(FILES, w.round(3)))}')
print(sub.head().to_string(index=False))
tlog('SUBMISSION SIAP')
'''

cells = ["KE_SRC = " + repr(ke_src) + f"\nFILES = {FILES!r}\nWEIGHTS = {WEIGHTS!r}\nSPAN = {(LO, HI)!r}\nK_EVAL = {K_EVAL}\n", MAIN]
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
    "dataset_sources": [], "kernel_sources": [f"dimaspashaakrilian/{s}" for s in dict.fromkeys(SLUGS)],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| model:", len(FILES), "| span:", (LO, HI))
