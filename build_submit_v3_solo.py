#!/usr/bin/env python3
"""Build rsna-knee-submit-v3-solo: submission REAL pakai checkpoint v3 kita SENDIRIAN
(coatnet_rawmean_ft_v3.pt -- CoAtNet fine-tune di atas label blend raw-probability-mean
flight0234+Claude, gold-58 macro=0.9234), tanpa blend/ensemble apa pun.

TUJUAN: perbandingan APPLE-TO-APPLE dgn c5-solo yg sudah pernah disubmit real (26 Agustus,
submission 55795592, public LB = 0.927). Kedua-duanya: 1 arm CoAtNet, harness `_KE_SRC` yg
SAMA PERSIS, K_EVAL=42 default yg SAMA, tanpa blend. Satu-satunya variabel yg beda = bobot
checkpoint (stock v5_full_swa vs hasil fine-tune label-blend kita).

Kenapa ini eksperimen yg dipilih (13 Sep, keputusan mandiri atas arahan user "putuskan sendiri"):
- Submission pipeline PENUH memakan ~9,7 GPU-jam dari kuota (terukur: kuota 7,55h -> 17,27h
  setelah 1 submission + 2 dev-test). Sisa kuota cuma 12,73h sampai refresh 19 Sep.
- Percobaan hybrid (v3 disisipkan ke pipeline 0,941, alpha=0,15) hasilnya NULL (0,941 -> 0,941,
  tidak terbaca di 3 desimal) dan TIDAK BISA didiagnosis krn Kaggle tidak pernah expose log run
  submission sungguhan. Mengulanginya dgn alpha beda = bakar 9,7h lagi utk efek yg kemungkinan
  besar tetap di bawah resolusi leaderboard.
- Eksperimen ini jauh lebih murah (~1,5-2 GPU-jam, 1 arm saja) DAN informasinya jauh lebih
  menentukan: kalau v3-solo > 0,927 maka label engineering kita TERBUKTI menaikkan model di data
  nyata (bukan cuma di 58 studi gold) -> investasi training besar (v4) layak. Kalau <= 0,927,
  seluruh line label-blend perlu dievaluasi ulang SEBELUM menghabiskan 7-9 jam kuota utk training.

K_EVAL sengaja TIDAK diturunkan (tetap 42 default) supaya perbandingan vs c5-solo benar-benar
apple-to-apple -- beda dgn build_submit_v3_hybrid.py yg menurunkannya ke 16 demi hemat waktu.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "kaggle_kernel_submit_v3_solo"
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

# satu-satunya patch: arahkan ARMS ke checkpoint v3 kita (arch/res/K_EVAL semua tetap default,
# identik dgn yg dipakai c5-solo dulu)
old_arms = "ARMS = [{'file': 'raptor_ft_coatnet_v5_full_swa.pt', 'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384, 'w': 1.0}]"
new_arms = "ARMS = [{'file': 'coatnet_rawmean_ft_v3.pt', 'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384, 'w': 1.0}]"
assert old_arms in ke_src, "ARMS line tidak ketemu verbatim -- cek ke_src sebelum patch"
ke_src = ke_src.replace(old_arms, new_arms)
assert "K_EVAL = 42" in ke_src, "K_EVAL default berubah -- perbandingan vs c5-solo tidak lagi apple-to-apple"

SETUP = '''\
import os, sys, subprocess, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)

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
    import glob
    hits = glob.glob('/kaggle/input/**/timm-*.whl', recursive=True)
    assert hits, 'wheel timm tidak ketemu'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--no-deps', hits[0]], check=True)
assert timm_ok(), 'timm tidak siap'
tlog('setup OK')
'''

RUN = '''\
# ===== inferensi test-set sungguhan: checkpoint v3 SENDIRIAN, tanpa blend =====
import shutil
KE = _KE_NS
tlog(f"ARMS (harus 1 arm, v3 saja): {KE['ARMS']}")
assert len(KE['ARMS']) == 1 and KE['ARMS'][0]['file'] == 'coatnet_rawmean_ft_v3.pt', \\
    'ARMS bukan v3 tunggal -- cek patch builder'
assert KE['K_EVAL'] == 42, 'K_EVAL bukan 42 -- perbandingan vs c5-solo tidak apple-to-apple'
KE['main']()
shutil.copy('/kaggle/working/_raptor.csv', '/kaggle/working/submission.csv')
import pandas as pd
sub = pd.read_csv('/kaggle/working/submission.csv')
tlog(f'submission.csv: {sub.shape}')
print(sub.head().to_string(index=False))
tlog('SUBMISSION SIAP')
'''

nb_cells = [
    {"cell_type": "code", "source": SETUP.splitlines(keepends=True)},
    {"cell_type": "code", "source": [
        "_KE_SRC = " + repr(ke_src) + "\n",
        "_KE_NS = {'__name__': '_ke'}\n",
        "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n",
        "print('_KE loaded')"]},
    {"cell_type": "code", "source": RUN.splitlines(keepends=True)},
]
nb = {"cells": [{**c, "metadata": {}, "outputs": [], "execution_count": None}
                for c in nb_cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

OUT.mkdir(exist_ok=True)
(OUT / "submit_v3_solo.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "dimaspashaakrilian/rsna-knee-submit-v3-solo",
    "title": "RSNA Knee - Submit v3 solo",
    "code_file": "submit_v3_solo.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dreaddevelopment/raptor-knee-maxspan",
    ],
    # checkpoint v3 dimount dari output kernel training-nya (pola kernel_sources standar proyek)
    "kernel_sources": [
        "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
