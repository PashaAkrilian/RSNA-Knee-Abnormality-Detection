#!/usr/bin/env python3
"""Versi GPU (2xT4, dua proses paralel) dari build_train_tpu_oai_child.py: fine-tune "anak" dari ckpt OAI nartaa pada label fusi r1 + KD ke induk.

Dipakai SETELAH kuota GPU pulih (10 Okt 07:00 WIB). Teks pelatihan diambil dari builder TPU lalu diubah mekanis (XLA -> CUDA fp16 + GradScaler),
supaya logika identik. Dua proses (CUDA_VISIBLE_DEVICES=0/1) menjalankan set varian masing-masing; proses 0 juga menulis parent_soft_k16.csv bila diminta.

Mode SMOKE (uji alur di kernel CPU, NOL kuota GPU/TPU): `... smoke` -> 1 proses CPU fp32, 6 studi latih, 2 epoch, K_WIN=4, evaluasi 8 window, 4 gold,
SWA + simpan + pass induk-in-sample (8 studi). Menangkap bug logika (data/label/KD/SWA/simpan) sebelum membakar kuota GPU. Tidak menguji CUDA/AMP.

Pemakaian:
  python3 build_train_gpu_oai_child.py                       # anak c (KD .5) & d (KD .25), 2 GPU
  python3 build_train_gpu_oai_child.py '[[["c",0.5,3]],[["d",0.25,4]]]' 0
  python3 build_train_gpu_oai_child.py - 1 smoke              # uji alur di CPU (kernel rsna-knee-smoke-oai-child-cpu)
  python3 build_train_gpu_oai_child.py - 0 x colab            # skrip mandiri utk VM Colab -> colab/oai_child/child_train_colab.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SMOKE = len(sys.argv) > 3 and sys.argv[3] == "smoke"
OUT = REPO / ("kaggle_kernel_smoke_oai_child_cpu" if SMOKE else "kaggle_kernel_train_gpu_oai_child")
SLUG = "rsna-knee-smoke-oai-child-cpu" if SMOKE else "rsna-knee-train-gpu-oai-child"
_a1 = sys.argv[1] if len(sys.argv) > 1 else "-"
SETS = json.loads(_a1) if _a1 != "-" else ([[["s", 0.5, 9]]] if SMOKE else [[["c", 0.5, 3]], [["d", 0.25, 4]]])
PARENT_SOFT = SMOKE or ((sys.argv[2] if len(sys.argv) > 2 else "0") == "1")

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
T = sub1(T, "DEV = xm.xla_device()", "DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')")
a = T.index("try:\n    with torch.autocast('xla'")
b = T.index("def bn_eval(m):")
T = T[:a] + ("if torch.cuda.is_available():\n"
             "    AC = lambda: torch.autocast('cuda', dtype=torch.float16)\n"
             "    try: SCALER = torch.amp.GradScaler('cuda')\n"
             "    except Exception: SCALER = torch.cuda.amp.GradScaler()\n"
             "    tlog('autocast cuda fp16 + GradScaler')\n"
             "else:\n"
             "    AC = lambda: contextlib.nullcontext()\n"
             "    try: SCALER = torch.amp.GradScaler('cuda', enabled=False)\n"
             "    except Exception: SCALER = torch.cuda.amp.GradScaler(enabled=False)\n"
             "    tlog('MODE CPU fp32 (smoke)')\n\n") + T[b:]
T = sub1(T, "(loss / 1.0).backward()", "SCALER.scale(loss / ACC_STUDIES).backward()")
T = sub1(T, """                for p_ in (bb_params + head_params):
                    if p_.grad is not None: p_.grad.div_(ACC_STUDIES)
                set_lr(step); opt.step(); xm.mark_step(); opt.zero_grad(set_to_none=True); step += 1""",
         """                set_lr(step); SCALER.step(opt); SCALER.update(); opt.zero_grad(set_to_none=True); step += 1""")
T = sub1(T, "# --- prediksi induk in-sample (K=16 window merata, 1 tampilan) utk diagnosis memorisasi & fusi label r2 ---\ntlog('prediksi induk pada semua studi latih ...')",
         "if os.environ.get('DO_PARENT_SOFT', '0') != '1':\n    tlog('SELESAI (tanpa parent_soft)'); raise SystemExit(0)\ntlog('prediksi induk pada semua studi latih ...')")
T = sub1(T, "VARIANTS = json.loads(os.environ.get('VARIANTS', '[[\"a\", 0.5, 1], [\"b\", 0.0, 2]]'))", "VARIANTS = json.loads(os.environ['VARIANTS'])")
# --- kait SMOKE / jumlah window evaluasi (aktif hanya bila env diset; perilaku produksi tak berubah) ---
T = sub1(T, "assert len(MOUNTED) > 4000, 'cache z tidak lengkap'", "assert len(MOUNTED) > (100 if os.environ.get('SMOKE') else 4000), 'cache z tidak lengkap'")
T = sub1(T, "tlog(f'studi latih (non-gold, berlabel fusi, ter-cache): {len(train_ids)}')",
         "tlog(f'studi latih (non-gold, berlabel fusi, ter-cache): {len(train_ids)}')\n"
         "if os.environ.get('SMOKE'): train_ids = train_ids[:int(os.environ['SMOKE'])]; tlog(f'SMOKE: {len(train_ids)} studi latih')")
T = sub1(T, "gids = sorted(GOLD); gids = [g for g in gids if g in MOUNTED]",
         "gids = sorted(GOLD); gids = [g for g in gids if g in MOUNTED]\nif os.environ.get('SMOKE'): gids = gids[:int(os.environ.get('SMOKE_GOLD', '4'))]\n"
         "EVAL_K = int(os.environ.get('EVAL_K', '94')); EVAL_CH = max(1, EVAL_K // 2)")
T = sub1(T, "xw = KE['eval_windows'](vol, mask, k=94, res=RES, norm=KE['NORM'])     # (94,3,384,384) fp32\n        P.append(predict(m, xw, 47))",
         "xw = KE['eval_windows'](vol, mask, k=EVAL_K, res=RES, norm=KE['NORM'])     # (94,3,384,384) fp32\n        P.append(predict(m, xw, EVAL_CH))")
T = sub1(T, "allids = [s for s in ALL_IDS if s in MOUNTED]", "allids = [s for s in ALL_IDS if s in MOUNTED]\nif os.environ.get('SMOKE'): allids = allids[:8]")

COLAB = len(sys.argv) > 4 and sys.argv[4] == "colab"
script = SETUP + "\n_KE_SRC = " + repr(ke_src) + "\n_KE_NS = {'__name__': '_ke'}\nexec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\nprint('_KE loaded', flush=True)\n" + T
if COLAB:    # VM Colab: /kaggle read-only -> pohon tiruan di /content (lihat colab/bootstrap_child.py)
    script = script.replace('/kaggle/input', '/content/kin').replace('/kaggle/working', '/content/kw').replace('/kaggle/tmp', '/content/kw')
    (REPO / 'colab' / 'oai_child').mkdir(exist_ok=True)
    (REPO / 'colab' / 'oai_child' / 'child_train_colab.py').write_text(script)
    print('colab: ditulis colab/oai_child/child_train_colab.py', len(script), 'byte')
compile(script, "<gpu-child>", "exec")

EXTRA_ENV = "dict(SMOKE='6', SMOKE_GOLD='4', EPOCHS='2', K_WIN='4', EVAL_K='8')" if SMOKE else "{}"
LAUNCH = f'''\
import os, sys, json, subprocess, time
SETS = {json.dumps(SETS)}
os.makedirs('/kaggle/working', exist_ok=True)
open('/kaggle/working/child_train.py', 'w').write({script!r})
procs = []
for g, variants in enumerate(SETS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), VARIANTS=json.dumps(variants), DO_PARENT_SOFT=('1' if (g == 0 and {PARENT_SOFT!r}) else '0'), **{EXTRA_ENV})
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
import glob
print('berkas keluaran:', sorted(os.path.basename(f) for f in glob.glob('/kaggle/working/*.pt') + glob.glob('/kaggle/working/*.csv')))
'''
nb = {"cells": [{"cell_type": "code", "source": LAUNCH.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": f"dimaspashaakrilian/{SLUG}", "title": ("RSNA Knee - Smoke OAI child CPU" if SMOKE else "RSNA Knee - Train GPU OAI child"), "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": (not SMOKE), "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["dimaspashaakrilian/rsna-knee-fused-labels-r1", "nartaa/rsna-knee-publication-swa-weights-20261007"],
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-z{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
if not SMOKE:
    meta["machine_shape"] = "NvidiaTeslaT4"
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT.name, "| sets:", SETS, "| parent_soft:", PARENT_SOFT, "| smoke:", SMOKE)
