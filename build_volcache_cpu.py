#!/usr/bin/env python3
"""Build 3 kernel CPU `rsna-knee-volcache-s{0,1,2}`: decode DICOM -> volume uint8 -> npz, SEKALI,
lalu dipakai ulang oleh semua kernel GPU via kernel_sources.

MENGAPA: tiap kernel latih GPU selalu membayar ~0,75 jam GPU hanya utk men-decode 4.407 studi
(cache tidak persist lintas kernel). Kuota GPU dipakai bersama proyek lain di akun ini
(casmi26, fase-*), jadi tiap jam berharga. Kernel CPU TIDAK memakan kuota GPU dan boleh jalan
paralel. Dengan ~8 kernel GPU yang direncanakan (K-fold + putaran-2), ini menghemat ~6 jam GPU.

Dibagi 3 shard (ALL_IDS[shard::3]) supaya tiap output jauh di bawah batas 20GB /kaggle/working
dan ketiganya bisa jalan bersamaan. Keluaran: /kaggle/working/volcache/<uid>.npz (vol, mask) --
format IDENTIK dgn yg ditulis cache_one() di kernel training, jadi load_study() tak berubah.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()
NSHARD = 3

CELL = '''\
import os, sys, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
import numpy as np, pandas as pd
SHARD = __SHARD__; NSHARD = __NSHARD__
_KE_SRC = __KE_SRC__
_KE_NS = {'__name__': '_ke'}
exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)
KE = _KE_NS
LAB = KE['LAB']
ROOT = KE['find_test_root']()
tsdir = ROOT + '/train_series'
train = pd.read_csv(ROOT + '/train.csv')
train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv')
for c in ('StudyInstanceUID', 'SeriesInstanceUID'):
    tser[c] = tser[c].astype(str)
SER = {k: v[['SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive']].to_dict('records')
       for k, v in tser.groupby('StudyInstanceUID')}
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in SER]
MY = ALL_IDS[SHARD::NSHARD]
tlog(f'shard {SHARD}/{NSHARD}: {len(MY)} dari {len(ALL_IDS)} studi')

CDIR = '/kaggle/working/volcache'
os.makedirs(CDIR, exist_ok=True)
reader = KE['_make_reader']()

def cache_one(sid):
    fn = f'{CDIR}/{sid}.npz'
    if os.path.exists(fn):
        return 'skip'
    try:
        vol, mask = KE['build_study'](sid, SER, tsdir, reader)
        np.savez_compressed(fn, vol=vol, mask=mask)
        return 'ok'
    except Exception as e:
        return f'fail:{type(e).__name__}'

from concurrent.futures import ThreadPoolExecutor
stat = {}
with ThreadPoolExecutor(8) as ex:
    for i, r in enumerate(ex.map(cache_one, MY)):
        stat[r] = stat.get(r, 0) + 1
        if (i + 1) % 100 == 0:
            tlog(f'{i+1}/{len(MY)} | {stat}')
tlog(f'SELESAI {stat}')
import subprocess
print(subprocess.run(['du', '-sh', CDIR], capture_output=True, text=True).stdout, flush=True)
assert stat.get('fail', 0) == 0 and not any(k.startswith('fail') for k in stat), f'ada kegagalan: {stat}'
'''

for shard in range(NSHARD):
    out = REPO / f"kaggle_kernel_volcache_s{shard}"
    out.mkdir(exist_ok=True)
    src = (CELL.replace("__SHARD__", str(shard)).replace("__NSHARD__", str(NSHARD))
               .replace("__KE_SRC__", repr(ke_src)))
    nb = {"cells": [{"cell_type": "code", "source": src.splitlines(keepends=True),
                     "metadata": {}, "outputs": [], "execution_count": None}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    name = f"volcache_s{shard}.ipynb"
    (out / name).write_text(json.dumps(nb, indent=1))
    meta = {"id": f"dimaspashaakrilian/rsna-knee-volcache-s{shard}",
            "title": f"RSNA Knee - Volcache s{shard}",
            "code_file": name, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False,
            "competition_sources": ["rsna-knee-abnormality-detection"]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("built:", out)
