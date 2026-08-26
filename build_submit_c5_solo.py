#!/usr/bin/env python3
"""Build rsna-knee-submit-c5-solo: submission REAL pakai c5 (raptor_ft_coatnet_v5_full_swa.pt)
SENDIRIAN, tanpa blend/ensemble apa pun -- ground-truth langsung skor publik arm kita di
leaderboard, dibanding fork community 0.936 yang selama ini jadi submission tim.

_KE_SRC (verbatim, sama persis dgn kaggle_kernel_train_coatnet_v1/) SUDAH berisi main() --
pipeline inferensi test-set lengkap. ARMS default di dalamnya SUDAH persis 1 arm c5 saja
(tidak perlu di-patch). Kita cuma exec KE lalu panggil main() secara eksplisit (di training
notebook ini SENGAJA tidak auto-run krn __name__ != '__main__' saat di-exec sbg namespace),
lalu salin _raptor.csv -> submission.csv (nama wajib kompetisi).
"""
import json
from pathlib import Path

REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")
OUT = REPO / "kaggle_kernel_submit_c5_solo"
KE_SRC_FILE = REPO / "build_train_coatnet_ke_src.txt"
ke_src = KE_SRC_FILE.read_text()

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
# ===== jalankan inferensi test-set SUNGGUHAN, c5 (v5_full_swa) SENDIRIAN, tanpa blend =====
import shutil
KE = _KE_NS
tlog(f"ARMS (harus 1 arm, c5 saja): {KE['ARMS']}")
assert len(KE['ARMS']) == 1 and KE['ARMS'][0]['file'] == 'raptor_ft_coatnet_v5_full_swa.pt', \\
    'ARMS bukan c5 tunggal -- cek _KE_SRC belum berubah dari yg diharapkan'
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
(OUT / "submit_c5_solo.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "dimaspashaakrilian/rsna-knee-submit-c5-solo",
    "title": "RSNA Knee - Submit c5 solo (no blend)",
    "code_file": "submit_c5_solo.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dreaddevelopment/raptor-knee-maxspan",
        "mattiaangeli/knee-mri-fold-weights",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
