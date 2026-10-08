#!/usr/bin/env python3
"""Versi GPU (2xT4, dua proses paralel) dari build_train_tpu_oai_child.py: fine-tune "anak" dari ckpt OAI nartaa pada label fusi r1 + KD ke induk.

Dipakai SETELAH kuota GPU pulih (10 Okt 07:00 WIB). Teks pelatihan diambil dari builder TPU lalu diubah mekanis (XLA -> CUDA fp16 + GradScaler),
supaya logika identik. Dua proses (CUDA_VISIBLE_DEVICES=0/1) menjalankan set varian masing-masing; proses 0 juga menulis parent_soft_k16.csv bila diminta.
Pemakaian: python3 build_train_gpu_oai_child.py "[[[\"c\",0.5,3]],[[\"d\",0.25,4]]]"     # daftar varian per GPU
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "kaggle_kernel_train_gpu_oai_child"
SLUG = "rsna-knee-train-gpu-oai-child"
SETS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [[["c", 0.5, 3]], [["d", 0.25, 4]]]
PARENT_SOFT = (sys.argv[2] if len(sys.argv) > 2 else "0") == "1"

tpu_src = (REPO / "build_train_tpu_oai_child.py").read_text()
m = re.search(r"^TRAIN = '''\\\n(.*?)^'''\n", tpu_src, re.S | re.M)
assert m, "TRAIN tidak ketemu di builder TPU"
TRAIN = m.group(1)
m2 = re.search(r"^SETUP = '''\\\n(.*?)^'''\n", tpu_src, re.S | re.M)
SETUP = m2.group(1)
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()


def sub1(text, old, new):
    assert text.count(old) == 1, f"pola tidak unik/tidak ada: {old[:70]!r} x{text.count(old)}"
    return text.replace(old, new)


T = TRAIN
T = sub1(T, "import torch_xla.core.xla_model as xm\n", "xm = type('XM', (), {'mark_step': staticmethod(lambda: None)})()\n")
T = sub1(T, "DEV = xm.xla_device()", "DEV = torch.device('cuda:0')")
a = T.index("try:\n    with torch.autocast('xla'")
b = T.index("def bn_eval(m):")
T = T[:a] + ("AC = lambda: torch.autocast('cuda', dtype=torch.float16)\n"
             "SCALER = torch.cuda.amp.GradScaler()\n"
             "tlog('autocast cuda fp16 + GradScaler')\n\n") + T[b:]
T = sub1(T, "(loss / 1.0).backward()", "SCALER.scale(loss / ACC_STUDIES).backward()")
T = sub1(T, """                for p_ in (bb_params + head_params):
                    if p_.grad is not None: p_.grad.div_(ACC_STUDIES)
                set_lr(step); opt.step(); xm.mark_step(); opt.zero_grad(set_to_none=True); step += 1""",
         """                set_lr(step); SCALER.step(opt); SCALER.update(); opt.zero_grad(set_to_none=True); step += 1""")
T = sub1(T, "# --- prediksi induk in-sample (K=16 window merata, 1 tampilan) utk diagnosis memorisasi & fusi label r2 ---\ntlog('prediksi induk pada semua studi latih ...')",
         "if os.environ.get('DO_PARENT_SOFT', '0') != '1':\n    tlog('SELESAI (tanpa parent_soft)'); raise SystemExit(0)\ntlog('prediksi induk pada semua studi latih ...')")
T = sub1(T, "VARIANTS = json.loads(os.environ.get('VARIANTS', '[[\"a\", 0.5, 1], [\"b\", 0.0, 2]]'))", "VARIANTS = json.loads(os.environ['VARIANTS'])")

script = SETUP + "\n_KE_SRC = " + repr(ke_src) + "\n_KE_NS = {'__name__': '_ke'}\nexec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\nprint('_KE loaded', flush=True)\n" + T
compile(script, "<gpu-child>", "exec")

LAUNCH = f'''\
import os, sys, json, subprocess, time
SETS = {json.dumps(SETS)}
os.makedirs('/kaggle/working', exist_ok=True)
open('/kaggle/working/child_train.py', 'w').write({script!r})
procs = []
for g, variants in enumerate(SETS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), VARIANTS=json.dumps(variants), DO_PARENT_SOFT=('1' if (g == 0 and {PARENT_SOFT!r}) else '0'))
    lf = open(f'/kaggle/working/child_log{{g}}.txt', 'w')
    procs.append((g, subprocess.Popen([sys.executable, '-u', '/kaggle/working/child_train.py'], env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
pos = [0] * len(procs)
while True:
    alive = False
    for g, p, lf in procs:
        lf.flush()
        with open(f'/kaggle/working/child_log{{g}}.txt', 'r', errors='replace') as fh:
            fh.seek(pos[g]); new = fh.read(); pos[g] = fh.tell()
        for line in new.splitlines():
            print(f'[g{{g}}] ' + line, flush=True)
        if p.poll() is None: alive = True
    if not alive: break
    time.sleep(20)
print('rc:', [p.returncode for _, p, _ in procs])
'''
nb = {"cells": [{"cell_type": "code", "source": LAUNCH.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": f"dimaspashaakrilian/{SLUG}", "title": "RSNA Knee - Train GPU OAI child", "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["dimaspashaakrilian/rsna-knee-fused-labels-r1", "nartaa/rsna-knee-publication-swa-weights-20261007"],
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-z{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT.name, "| sets:", SETS, "| parent_soft:", PARENT_SOFT)
