#!/usr/bin/env python3
"""Uji skala v2 da tahap OF (of_stage.OF_STAGE): MESMO código do notebook de produção, com raiz grande simbólica.

Usa estudos de TREINO (symlinks) como se fossem o teste; roda STOP_H horas e mede a vazão agregada (4 processos, potongan kecil,
penjaga disk, thread=1). Tanpa submisi. Pemakaian: python3 build_scale_of2.py name=ofscale3 nproc=4 chunk=8 stop_h=0.5 kill_h=0.8
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from of_stage import OF_STAGE  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "ofscale3")
NPROC = args.get("nproc", "4")
CHUNK = args.get("chunk", "8")
STOP_H = args.get("stop_h", "0.5")
KILL_H = args.get("kill_h", "0.8")

PRE = r'''
import os as _l_os, sys as _l_sys, glob as _l_glob, json as _l_json, time as _l_time
import shutil as _l_shutil, subprocess as _l_sp, textwrap as _l_tw, hashlib as _l_hl, traceback as _l_tb, random as _l_rnd
import numpy as _l_np, pandas as _l_pd, torch as _l_torch
LEG_OUT = {}
_T_ALL = _l_time.time()
COMP = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
def _l_find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(_l_glob.glob(pat)):
            if _l_os.path.exists(_l_os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(f"{name} not found in /kaggle/input")
_l_env = dict(_l_os.environ)
_l_ng = max(1, _l_torch.cuda.device_count())
_l_labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
             'Synovitis', "Baker's", 'Contusion', 'Fracture']
_tr = _l_pd.read_csv(f"{COMP}/train.csv")
_gold = set(_tr[_tr[_l_labels].notna().all(axis=1)]["StudyInstanceUID"])
_l_ids = [s for s in _tr["StudyInstanceUID"].tolist() if s not in _gold][:1322]
_l_rnd.Random(7).shuffle(_l_ids)                       # prioridade aleatória (imita a ordem de produção)
_of_order = list(_l_ids)
_l_data = COMP
_of_series_dir = f"{COMP}/train_series"                 # simula test_series
_of_series_csv = f"{COMP}/train_series.csv"
_of_test_csv = f"{COMP}/test.csv"
st = _l_os.statvfs("/kaggle/working"); print("DISK free GB at start: %.1f" % (st.f_bavail * st.f_frsize / 1e9), flush=True)
'''

POST = r'''
_el = (_l_time.time() - _T_ALL) / 3600
_nd = 0
for _i in range(__NPROC__):
    try:
        _nd += sum(1 for _ in open(f"/kaggle/working/_leg_of_w{_i}.csv")) - 1
    except OSError:
        pass
print("SCALE2_SUMMARY nproc __NPROC__ chunk __CHUNK__ studies_done", _nd, "elapsed_h %.2f" % _el, flush=True)
_inf = max(1e-6, _el - 0.04)
print("PROJECTION: %.1f studies/h aggregate -> 1322 studies need %.1f h" % (_nd / _inf, 1322 / max(1e-6, _nd / _inf)), flush=True)
'''

stage = (OF_STAGE.replace("__WO__", "0.0").replace("__NPROC__", NPROC).replace("__CHUNK__", CHUNK)
         .replace("__STOP_H__", STOP_H).replace("__KILL_H__", KILL_H).replace("__MINCOV__", "0.0").replace("__REUSE__", args.get("reuse","1")))
post = POST.replace("__NPROC__", NPROC).replace("__CHUNK__", CHUNK)
for i, c in enumerate((PRE, stage, post)):
    compile(c, f"cell{i}", "exec")
nb = {
    "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# RSNA Knee - OrthoFoundation scale test v2 (commit-run)\n"]}] +
             [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.lstrip("\n").splitlines(keepends=True)}
              for c in (PRE, stage, post)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": ["heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "dimaspashaakrilian/rsna-knee-ultralytics-cfg"],
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| nproc", NPROC, "chunk", CHUNK, "stop_h", STOP_H, "kill_h", KILL_H)
