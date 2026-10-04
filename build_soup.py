#!/usr/bin/env python3
"""Model soup: rata-rata BOBOT beberapa fine-tune c5 yang sama-sama bersih terhadap gold-58, supaya
keuntungan ensemble (§ensemble: 4 murid fold = 0,9286 vs satu model ~0,922-0,930) didapat dgn biaya
inferensi SATU arm (+0 menit) dan bukan N arm (+22 menit per arm di pipeline publik).

Semua anggota berasal dari init yang sama (raptor_ft_coatnet_v5_full_swa.pt) dan struktur yg sama, jadi
rata-rata bobot masuk akal (basin yg sama). Diuji, bukan diasumsikan: kernel evaluasi menilai tiap soup
di gold-58 dgn harness & cache yg sama.

Anggota (kernel keluaran -> berkas): kfold f0-f3 (kfold_c5_f{k}.pt), v3 (coatnet_rawmean_ft_v3.pt),
round-2 c5 (coatnet_fused_r1_c5.pt). Keluaran: soup_folds4.pt, soup_all6.pt, soup_v3_r2.pt, soup_folds4_v3.pt
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
MEMBERS = {
    "f0": ("dimaspashaakrilian/rsna-knee-kfold-c5-f0", "kfold_c5_f0.pt"),
    "f1": ("dimaspashaakrilian/rsna-knee-kfold-c5-f1", "kfold_c5_f1.pt"),
    "f2": ("dimaspashaakrilian/rsna-knee-kfold-c5-f2", "kfold_c5_f2.pt"),
    "f3": ("dimaspashaakrilian/rsna-knee-kfold-c5-f3", "kfold_c5_f3.pt"),
    "v3": ("dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean", "coatnet_rawmean_ft_v3.pt"),
    "r2": ("dimaspashaakrilian/rsna-knee-train-round2-c5-r1", "coatnet_fused_r1_c5.pt"),
}
SOUPS = {
    "soup_folds4": ["f0", "f1", "f2", "f3"],
    "soup_folds4_v3": ["f0", "f1", "f2", "f3", "v3"],
    "soup_all6": ["f0", "f1", "f2", "f3", "v3", "r2"],
    "soup_v3_r2": ["v3", "r2"],
}
CELL = '''\
import os, glob, time, json
import torch
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
MEMBERS = __MEMBERS__
SOUPS = __SOUPS__
def find(fname):
    hits = []
    for pat in ('/kaggle/input/*/' + fname, '/kaggle/input/*/*/' + fname, '/kaggle/input/*/*/*/' + fname, '/kaggle/input/*/*/*/*/' + fname):
        hits += glob.glob(pat)
    hits = sorted(set(hits))
    assert len(hits) == 1, f'{fname}: ditemukan {hits}'
    return hits[0]
paths = {k: find(v[1]) for k, v in MEMBERS.items()}
for k, p in paths.items(): tlog(f'{k}: {p} ({os.path.getsize(p)/1e6:.0f} MB)')
cache = {}
def load(k):
    ck = torch.load(paths[k], map_location='cpu', weights_only=False)
    return ck
ref = load('f0')
arch, res = ref['arch'], ref['res']
keys = list(ref['model'].keys())
tlog(f'arch={arch} res={res} tensor={len(keys)}')
for name, mem in SOUPS.items():
    acc = {k: None for k in keys}
    n = 0
    for m in mem:
        ck = ref if m == 'f0' else load(m)
        assert ck['arch'] == arch and int(ck['res']) == int(res), f'{m}: arch/res beda'
        sd = ck['model']
        assert list(sd.keys()) == keys, f'{m}: kunci state_dict beda'
        for k in keys:
            t = sd[k]
            if t.is_floating_point():
                acc[k] = t.double().clone() if acc[k] is None else acc[k] + t.double()
            else:
                acc[k] = t.clone() if acc[k] is None else acc[k]    # buffer integer: ambil dari anggota pertama
        n += 1
        del ck
    out = {}
    for k in keys:
        out[k] = (acc[k] / n).to(ref['model'][k].dtype) if ref['model'][k].is_floating_point() else acc[k]
        if out[k].is_floating_point():
            assert bool(torch.isfinite(out[k]).all()), f'nonfinite {k}'
    torch.save({'arch': arch, 'res': res, 'model': out}, f'/kaggle/working/{name}.pt')
    tlog(f'{name}: rata-rata {n} model {mem} -> {name}.pt')
tlog('SOUP SELESAI')
'''.replace("__MEMBERS__", repr(MEMBERS)).replace("__SOUPS__", repr(SOUPS))
out = REPO / "kaggle_kernel_soup"; out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "source": CELL.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(out / "soup.ipynb").write_text(json.dumps(nb, indent=1))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "dimaspashaakrilian/rsna-knee-soup", "title": "RSNA Knee - Soup", "code_file": "soup.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
    "enable_internet": False, "kernel_sources": sorted({v[0] for v in MEMBERS.values()})}, indent=2))
print("built:", out.name, "| anggota:", list(MEMBERS), "| soup:", list(SOUPS))
