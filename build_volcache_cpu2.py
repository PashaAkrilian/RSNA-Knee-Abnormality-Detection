#!/usr/bin/env python3
"""Cache DICOM CPU terparameter: <tag> <img> <slots=SLOTS64|SLOTS44> <lo> <hi>

Cache awal (rsna-knee-volcache-s*) = 336px, 64 irisan, rentang 6-94% -- TIDAK cocok dgn geometri arm
publik: c5 'maxspan' memakai rentang 2-98%, v10 384px, v8 384px/44 irisan. Putaran-2 harus dilatih pada
geometri yg sama dgn inferensinya (irisan tepi penting utk meniskus lateral, label terlemah kita).
Keluaran per shard: /kaggle/working/volcache/<uid>.npz, 3 shard per tag. CPU saja (nol kuota GPU/TPU).
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent
tag, img, slots_key, lo, hi = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
SL = {"SLOTS64": "[('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]",
      "SLOTS44": "[('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), ('Coronal', 0, 6), ('Axial', -1, 8)]"}
ke = (REPO / "build_train_coatnet_ke_src.txt").read_text()
def must(t, a, b):
    assert a in t, f"tak ketemu: {a[:60]!r}"; return t.replace(a, b, 1)
ke = must(ke, "IMG = 336", f"IMG = {img}")
ke = must(ke, "SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]", f"SLOTS = {SL[slots_key]}")
ke = must(ke, "lo, hi = (int(n * 0.06), int(n * 0.94) - 1)", f"lo, hi = (int(n * {lo}), int(n * {hi}) - 1)")
CELL = (REPO / "build_volcache_cpu.py").read_text().split("CELL = '''\\\n", 1)[1].split("'''\n", 1)[0]
for shard in range(3):
    out = REPO / f"kaggle_kernel_volcache_{tag}{shard}"; out.mkdir(exist_ok=True)
    src = CELL.replace("__SHARD__", str(shard)).replace("__NSHARD__", "3").replace("__KE_SRC__", repr(ke))
    nb = {"cells": [{"cell_type": "code", "source": src.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    name = f"volcache_{tag}{shard}.ipynb"; (out / name).write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({"id": f"dimaspashaakrilian/rsna-knee-volcache-{tag}{shard}",
        "title": f"RSNA Knee - Volcache {tag}{shard}", "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
    print("built:", out.name)
