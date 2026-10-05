#!/usr/bin/env python3
"""K-fold CoAtNet "murid" generasi-2: DUA fold paralel per sesi Kaggle (2 GPU) + init c5/v8/v10 + label guru ATAU fusi.

Pengembangan build_train_coatnet_kfold.py (1 fold/kernel, 1 GPU terpakai). Di sini tiap kernel menjalankan beberapa fold sekaligus
sebagai proses terpisah (CUDA_VISIBLE_DEVICES=0/1) -> separuh kuota per fold. Resep latih SAMA dgn murid putaran-1 (LR 2e-5 bb /
1e-3 head, 3 epoch, 75% data, fold sadar-grup, gold-58 tak pernah ikut latih), hanya:
  - init dapat c5 (336px/64 slot/span 2-98%, cache w), v10 (384px/64 slot/2-98%, cache x), v8 (384px/44 slot/6-94%, cache y):
    geometri cache = geometri inferensi arm publik yg bersangkutan (supaya checkpoint drop-in)
  - label = fusi r1 (guru + murid OOF, gold-58 0,9348) atau guru
  - keluaran bertag:  kfold<tag>_<init>_f<k>.pt  &  oof_<tag>_<init>_f<k>.csv   (tag mis. r2)
Pemakaian: python3 build_train_coatnet_kfold2.py <init=c5|v10|v8> <labels=teacher|r1> <folds=0,1> [nfold=4] [epochs=3] [tag=r2]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
INIT = sys.argv[1]
LABELS = sys.argv[2]
FOLDS = [int(x) for x in sys.argv[3].split(",")]
NFOLD = int(sys.argv[4]) if len(sys.argv) > 4 else 4
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
TAG = sys.argv[6] if len(sys.argv) > 6 else "r2"
GEOM = {
    "c5":  dict(ckpt="raptor_ft_coatnet_v5_full_swa.pt", cache="w", ds="dreaddevelopment/raptor-knee-maxspan"),
    "v10": dict(ckpt="raptor_ft_coatnet_v10_full.pt", cache="x", ds="dreaddevelopment/raptor-knee-native384dense"),
    "v8":  dict(ckpt="raptor_ft_coatnet_v8_full_swa.pt", cache="y", ds="dreaddevelopment/raptor-knee-native384"),
}
LAB = {"teacher": ("report_labels_blended_rawmean_v1.csv", "dimaspashaakrilian/rsna-knee-blended-report-labels-rawmean-v1"),
       "r1": ("fused_labels_r1.csv", "dimaspashaakrilian/rsna-knee-fused-labels-r1")}
assert INIT in GEOM and LABELS in LAB
G = GEOM[INIT]
LFILE, LDS = LAB[LABELS]
NAME = f"kf{TAG}-{INIT}-{'-'.join(map(str, FOLDS))}"
SLUG = f"rsna-knee-{NAME}"
OUT = REPO / f"kaggle_kernel_{NAME.replace('-', '_')}"

src = (REPO / "build_train_coatnet_kfold.py").read_text()
def grab(name):
    mark = f"{name} = '''\\\n"
    a = src.index(mark) + len(mark); b = src.index("'''\n", a)
    return src[a:b]
CACHE, TRAIN = grab("CACHE"), grab("TRAIN")
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

def must(t, a, b):
    assert a in t, f"pola tidak ketemu: {a[:80]!r}"
    return t.replace(a, b, 1)

# ---- TRAIN: label, init, nama keluaran ----
TRAIN = must(TRAIN, "_hits = _bglob('report_labels_blended_rawmean_v1.csv')", f"_hits = _bglob({LFILE!r})")
TRAIN = must(TRAIN, """if INIT == 'c5':
    wp = KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')
    model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, DEV)
else:
    raise SystemExit('INIT=imagenet belum diimplementasikan di builder ini')""",
"""_CK = {'c5': 'raptor_ft_coatnet_v5_full_swa.pt', 'v10': 'raptor_ft_coatnet_v10_full.pt', 'v8': 'raptor_ft_coatnet_v8_full_swa.pt'}
wp = KE['find_weight_file'](_CK[INIT])
model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, DEV)""")
TRAIN = TRAIN.replace("f'/kaggle/working/kfold_{INIT}_f{FOLD}.pt'", "f'/kaggle/working/kfold{TAG}_{INIT}_f{FOLD}.pt'")
TRAIN = TRAIN.replace("f'/kaggle/working/oof_{INIT}_f{FOLD}.csv'", "f'/kaggle/working/oof_{TAG}_{INIT}_f{FOLD}.csv'")
TRAIN = TRAIN.replace("f'/kaggle/working/probe_init_weak_f{FOLD}.csv'", "f'/kaggle/working/probe_{TAG}_{INIT}_weak_f{FOLD}.csv'")
TRAIN = TRAIN.replace("f'/kaggle/working/probe_init_gold_f{FOLD}.csv'", "f'/kaggle/working/probe_{TAG}_{INIT}_gold_f{FOLD}.csv'")
TRAIN = must(TRAIN, "EPOCHS = int(os.environ['EPOCHS']); INIT = os.environ['INIT']", "EPOCHS = int(os.environ['EPOCHS']); INIT = os.environ['INIT']; TAG = os.environ['TAG']")

WORKER_HEAD = '''\
import os, sys, time, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
t0 = time.time()
GT = os.environ.get('GTAG', '?')
def tlog(m): print(f'[g{GT} t={time.time()-t0:.0f}s] {m}', flush=True)
_KE_NS = {'__name__': '_ke'}
exec(compile(open('/kaggle/working/_ke_src.py').read(), '<ke>', 'exec'), _KE_NS)
'''
WORKER = WORKER_HEAD + "\n" + CACHE + "\n" + TRAIN
compile(WORKER, "<worker>", "exec")

PARENT = r'''
import os, sys, time, json, subprocess, glob
import numpy as np, pandas as pd
t0 = time.time()
def tlog(m): print(f'[main t={time.time()-t0:.0f}s] {m}', flush=True)
open('/kaggle/working/_ke_src.py', 'w').write(KE_SRC)
open('/kaggle/working/worker.py', 'w').write(WORKER_SRC)
import torch
NG = max(1, torch.cuda.device_count())
tlog(f'GPU: {NG} | folds {FOLDS} | init={INIT} labels={LABELS} tag={TAG} epochs={EPOCHS}')
procs = {}
for i, f in enumerate(FOLDS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % NG), FOLD=str(f), NFOLD=str(NFOLD), INIT=INIT, EPOCHS=str(EPOCHS), TAG=TAG, GTAG=f'{i%NG}f{f}')
    fh = open(f'/kaggle/working/worker_f{f}.log', 'w')
    procs[f] = (subprocess.Popen([sys.executable, '-u', '/kaggle/working/worker.py'], env=env, stdout=fh, stderr=subprocess.STDOUT), fh)
pos = {f: 0 for f in procs}
def pump():
    for f in procs:
        with open(f'/kaggle/working/worker_f{f}.log') as fh:
            fh.seek(pos[f]); new = fh.read(); pos[f] = fh.tell()
        for line in new.splitlines(): print(line, flush=True)
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(10); pump()
pump()
for f, (p, fh) in procs.items(): tlog(f'fold {f} rc={p.returncode}')
tlog('SELESAI')
'''
cells = ["KE_SRC = " + repr(ke_src) + "\nWORKER_SRC = " + repr(WORKER) + f"\nFOLDS = {FOLDS!r}\nNFOLD = {NFOLD}\nINIT = {INIT!r}\nLABELS = {LABELS!r}\nTAG = {TAG!r}\nEPOCHS = {EPOCHS}\n", PARENT]
for c in cells: compile(c, "<cell>", "exec")
nb = {"cells": [{"cell_type": "code", "source": c.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": f"dimaspashaakrilian/{SLUG}", "title": "RSNA Knee " + NAME.replace("-", " "), "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": list(dict.fromkeys([LDS, G["ds"]])),
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-{G['cache']}{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| init", INIT, "| labels", LABELS, "| folds", FOLDS, "| nfold", NFOLD, "| epochs", EPOCHS)
