#!/usr/bin/env python3
"""Kernel sekali-jalan: S6F nartaa (komponen B skripnya, patch S6F dari build_r7murid) pada 58 studi GOLD (split train).
Tujuan: (1) AUC gold-58 S6F + korelasinya dgn R7/murid/Raptor (analisis lokal), (2) detik/studi nyata utk anggaran waktu run.
Keluaran kernel: /kaggle/working/s6f_gold_probs.csv. Akun utama (inferensi saja).
Pakai: python3 build_s6f_gold.py && kaggle kernels push -p kaggle_kernel_s6f-gold"""
import json
from pathlib import Path

from build_r7murid import S6F_BASE, S6F_DS, S6F_PATCH, code, md

ROOT = Path(__file__).parent
GOLD_PATCH = [
    ('    for split in ("test", "train") if MODE == "parity" else ("test",):',
     '    for split in ("test", "train"):   # [gold] biaya studi train juga (split_parts)'),
    ('        B, _, b_info, b_failed = run_b(I, "B", I.test_ids, "test", b_budget, px=(MODE == "parity"))',
     '        _tr = pd.read_csv(I.comp / "train.csv", dtype={"StudyInstanceUID": str})   # [gold] 58 studi berlabel radiolog\n'
     '        GOLD_IDS = _tr.loc[_tr[LAB].notna().all(axis=1), "StudyInstanceUID"].tolist()\n'
     '        log("[gold] %d gold studies" % len(GOLD_IDS))\n'
     '        B, _, b_info, b_failed = run_b(I, "G", GOLD_IDS, "train", b_budget)'),
]


def script():
    s = S6F_BASE.read_text(encoding="utf-8")
    for a, b in S6F_PATCH + GOLD_PATCH:
        assert s.count(a) == 1, ("pola tidak tunggal", a[:70])
        s = s.replace(a, b)
    compile(s, "s6f_gold.py", "exec")
    return "%%writefile /tmp/s6f_gold.py\n" + s


RUN = r"""import os, sys, time, shutil, subprocess, json
from pathlib import Path
t0 = time.time()
rc = subprocess.run([sys.executable, '-u', '/tmp/s6f_gold.py'], env=dict(os.environ, S6F_GUARD_S='5400'), timeout=6600).returncode
p = Path('/kaggle/working/k1/G/G_probs.csv')
if p.is_file():
    shutil.copyfile(p, '/kaggle/working/s6f_gold_probs.csv')
print('S6F_GOLD', json.dumps({'rc': rc, 'ok': p.is_file(), 'seconds': round(time.time() - t0)}), flush=True)
for f in sorted(Path('/kaggle/working/k1/G').glob('part*/log.txt')):
    print('==', f, '\n', f.read_text(errors='replace')[-1500:], flush=True)
"""

if __name__ == "__main__":
    nb = {"cells": [md("# S6F (nartaa, all data) on the 58 gold studies\n\nComponent B of nartaa's public script only; output `s6f_gold_probs.csv`.\n"),
                    code(script()), code(RUN)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = ROOT / "kaggle_kernel_s6f-gold"
    out.mkdir(exist_ok=True)
    (out / "rsna-knee-s6f-gold.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": "dimaspashaakrilian/rsna-knee-s6f-gold", "title": "RSNA Knee s6f-gold", "code_file": "rsna-knee-s6f-gold.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": [S6F_DS, "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ditulis:", out)
