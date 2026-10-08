#!/usr/bin/env python3
"""Uji alur di kernel CPU (nol kuota GPU) untuk tambalan arm 'sibling r224' pada skrip inferensi nartaa (sel 4 notebook keluarga).
Memuat skrip yg SUDAH ditambal sbg modul (tanpa main), lalu: (1) load_model r224 (hash, img=224) + gpu_windows(crop=307->224) di CPU,
(2) membandingkan probabilitasnya dgn implementasi mandiri yg sudah tervalidasi (AUC gold 0,9150 di Colab), (3) memastikan jalur induk (r384/crop 320) tak berubah.
Pemakaian: python3 build_smoke_r224_cpu.py   -> kaggle_kernel_smoke_r224_cpu/ (slug rsna-knee-smoke-r224-cpu)"""
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / "kaggle_kernel_oai_fam_r224/rsna-knee-oai-fam-r224.ipynb").read_text())
patched = "".join(nb["cells"][4]["source"])
ke = (REPO / "build_train_coatnet_ke_src.txt").read_text()
CELL = f'''
import os, sys, glob, time, numpy as np, torch, timm, torch.nn.functional as F
t0 = time.time()
def tlog(m): print(f"[t={{time.time()-t0:.0f}}s] {{m}}", flush=True)
SRC = {patched!r}
ns = {{"__name__": "_t"}}
exec(compile(SRC, "nartaa_patched", "exec"), ns)           # definisi saja; main() tak dijalankan
tlog("skrip ditambal dimuat; ARMS = " + str([(a["file"][-22:], a["res"], a.get("crop"), a.get("single"), a.get("w")) for a in ns["ARMS"]]))
KE = {{"__name__": "_ke"}}; exec(compile({ke!r}, "<ke>", "exec"), KE)
# satu studi dari cache z
fs = sorted(glob.glob("/kaggle/input/*/*/volcache/*.npz") + glob.glob("/kaggle/input/*/*/*/volcache/*.npz") + glob.glob("/kaggle/input/*/volcache/*.npz") + glob.glob("/kaggle/input/*/*/*/*/volcache/*.npz"))[:3]
tlog(f"berkas cache ditemukan: {{len(fs)}}")
z = np.load(fs[0]); vol, mask = z["vol"], z["mask"]; tlog(f"vol {{vol.shape}} mask {{int(mask.sum())}}")
K = 8
def indep_wins(vol, mask, crop, res, mirror=False):
    D = vol.shape[0]; cs = np.array([max(1, min(c, D - 2)) for c in KE["_eval_centers"](mask, D, K)])
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)]))
    lo = (384 - crop) // 2; tri = tri[:, :, lo:lo + crop, lo:lo + crop].float() / 255.
    if crop != res: tri = F.interpolate(tri, size=(res, res), mode="bilinear", align_corners=False)
    return (tri - KE["_MEAN"]) / KE["_STD"]
ARMS = ns["ARMS"]
res_summary = {{}}
for ai, arm in enumerate(ARMS):
    path = ns["find_weight_file"](arm["file"]); tlog(f"arm {{ai}}: {{os.path.basename(path)}}")
    model, res = ns["load_model"](path, arm["arch"], arm["res"], "cpu", img=arm.get("img")); tlog(f"  load_model ok; res={{res}}")
    crop = arm.get("crop")
    xw = ns["gpu_windows"](vol, mask, K, res, "cpu", crop=crop)
    tlog(f"  gpu_windows -> {{tuple(xw.shape)}} dtype {{xw.dtype}} finite {{bool(torch.isfinite(xw).all())}}")
    # implementasi mandiri (tervalidasi pada gold di Colab): crop & ukuran jaringan sama
    c_ref = crop or res
    xr = indep_wins(vol, mask, c_ref, res)
    tlog(f"  selisih maks jendela vs implementasi mandiri: {{float((xw - xr).abs().max()):.2e}}")
    with torch.no_grad():
        p = torch.sigmoid(model(xw.unsqueeze(0)))[0].numpy(); q = torch.sigmoid(model(xr.unsqueeze(0)))[0].numpy()
    tlog(f"  prob (12): {{np.round(p, 3).tolist()}}")
    tlog(f"  selisih maks prob patched vs mandiri: {{float(np.abs(p - q).max()):.2e}}")
    res_summary[ai] = float(np.abs(p - q).max())
    del model
tlog("RINGKASAN " + str(res_summary))
assert all(v < 1e-3 for v in res_summary.values()), "jalur ditambal tidak identik dgn implementasi mandiri"
tlog("UJI R224 LOLOS")
'''
nbo = {"cells": [{"cell_type": "code", "source": CELL.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
out = REPO / "kaggle_kernel_smoke_r224_cpu"; out.mkdir(exist_ok=True)
slug = "rsna-knee-smoke-r224-cpu"
(out / f"{slug}.ipynb").write_text(json.dumps(nbo, indent=1))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee - Smoke r224 CPU", "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["nartaa/rsna-knee-publication-swa-weights-20261007"],
    "kernel_sources": ["dimaspashaakrilian/rsna-knee-volcache-z0"],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built:", out.name)
