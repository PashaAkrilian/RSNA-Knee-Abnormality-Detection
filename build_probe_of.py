#!/usr/bin/env python3
"""Probe kelayakan: jalankan OrthoFoundation fold0 (heliosli) end-to-end di Kaggle (akun utama), commit-run saja.

Langkah: rakit best.pt dari weights.part*, susun folder model (best.pt + anatomy.pt + kneexnet/), cetak konfigurasi
ckpt, lalu jalankan training.infer.run_inference pada (1) stub test 3 studi, (2) 58 studi emas lewat 'akun palsu'
(fake competition root: test.csv/test_series.csv dari studi emas + symlink test_series -> train_series).
Keluaran: of_stub/*, of_gold/* (submission.csv + log). Tidak ada submisi.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "kaggle_kernel_of_probe"
OUT.mkdir(exist_ok=True)

CODE = r'''
import os, sys, json, time, glob, hashlib, shutil, subprocess, traceback
import numpy as np, pandas as pd, torch

COMP = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
def find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)

OF = find("weight_parts.json")
print("OF dir:", OF, sorted(os.listdir(OF)), flush=True)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
try:
    import timm; print("timm", timm.__version__)
except Exception as e:
    print("timm import fail", e)
print(open(f"{OF}/weight_parts.json").read()[:1800], flush=True)
print(open(f"{OF}/receipt.json").read()[:1200], flush=True)

MD = "/kaggle/working/of"
os.makedirs(MD, exist_ok=True)
parts = sorted(glob.glob(f"{OF}/weights.part*"))
t0 = time.time()
h = hashlib.sha256()
with open(f"{MD}/best.pt", "wb") as out:
    for p in parts:
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                out.write(b); h.update(b)
print("best.pt assembled", os.path.getsize(f"{MD}/best.pt"), "sha", h.hexdigest()[:16], "%.0fs" % (time.time() - t0), flush=True)
assert h.hexdigest().startswith("4f424cc7"), "sha mismatch"
if not os.path.exists(f"{MD}/anatomy.pt"):
    shutil.copy2(f"{OF}/anatomy.pt", f"{MD}/anatomy.pt")
if not os.path.exists(f"{MD}/kneexnet"):
    shutil.copytree(f"{OF}/dependencies/kneexnet", f"{MD}/kneexnet")
print("model dir:", os.listdir(MD), flush=True)

# --- konfigurasi ckpt
state = torch.load(f"{MD}/best.pt", map_location="cpu", weights_only=True)
print("state keys:", list(state.keys()), flush=True)
def show(d, name, lim=2500):
    s = json.dumps(d, default=str)[:lim]
    print(name, s, flush=True)
show(state.get("model_config", {}), "model_config")
show(state.get("train_config", {}), "train_config", 3500)
for k in ("anatomy_sha256", "anatomy_asset_sha256", "ligament_atlas_sha256", "precision_policy", "epoch", "views", "labels", "provenance", "optimizer_steps"):
    if k in state:
        print(k, str(state[k])[:1500], flush=True)
del state

# --- pip wheels offline (timm/pydicom versi yg dikirim bersama ckpt)
for w in sorted(glob.glob(f"{OF}/dependencies/wheels/*.whl")):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", w], capture_output=True, text=True)
    print("pip", os.path.basename(w), "rc", r.returncode, (r.stderr or "")[-200:], flush=True)

sys.path.insert(0, f"{OF}/cloud_code")
import importlib
infer = importlib.import_module("training.infer")
print("imported training.infer OK", flush=True)

def run(tag, data):
    out = f"/kaggle/working/of_{tag}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    try:
        frame = infer.run_inference(data, f"{MD}/best.pt", out, anatomy_weights=f"{MD}/anatomy.pt", workers=2)
        dt = time.time() - t0
        print("RUN", tag, "OK studies", len(frame), "sec", round(dt, 1), "sec/study", round(dt / max(1, len(frame)), 2), flush=True)
        print(frame.head(3).to_string()[:800], flush=True)
        return frame
    except Exception as e:
        print("RUN", tag, "FAIL", repr(e)[:600], flush=True)
        traceback.print_exc()
        return None

# 1) stub test (3 studi)
stub = run("stub", COMP)

# 2) emas-58 via fake root
if stub is not None:
    FR = "/kaggle/working/fakecomp"
    os.makedirs(FR, exist_ok=True)
    tr = pd.read_csv(f"{COMP}/train.csv")
    LABEL = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
             'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
    gold = tr[tr[LABEL].notna().all(axis=1)]
    print("gold studies in train.csv:", len(gold), flush=True)
    real_test = pd.read_csv(f"{COMP}/test.csv")
    ft = pd.DataFrame({c: (gold["StudyInstanceUID"].values if c == "StudyInstanceUID" else None) for c in real_test.columns})
    ft.to_csv(f"{FR}/test.csv", index=False)
    ts_cols = list(pd.read_csv(f"{COMP}/test_series.csv", nrows=1).columns)
    trs = pd.read_csv(f"{COMP}/train_series.csv")
    trs = trs[trs["StudyInstanceUID"].isin(gold["StudyInstanceUID"])]
    trs[[c for c in ts_cols if c in trs.columns]].to_csv(f"{FR}/test_series.csv", index=False)
    shutil.copy(f"{COMP}/sample_submission.csv", f"{FR}/sample_submission.csv") if False else None
    ss = pd.DataFrame({"StudyInstanceUID": gold["StudyInstanceUID"].values})
    for c in LABEL:
        ss[c] = 0.5
    ss.to_csv(f"{FR}/sample_submission.csv", index=False)
    if not os.path.exists(f"{FR}/test_series"):
        os.symlink(f"{COMP}/train_series", f"{FR}/test_series")
    gold[["StudyInstanceUID"] + LABEL].to_csv("/kaggle/working/gold_truth.csv", index=False)
    print("fake root:", os.listdir(FR), "| test_series.csv rows", len(trs), "cols", ts_cols, flush=True)
    sec = None
    gf = run("gold", FR)
    if gf is not None:
        from sklearn.metrics import roc_auc_score
        sub = pd.read_csv("/kaggle/working/of_gold/submission.csv").set_index("StudyInstanceUID")
        truth = gold.set_index("StudyInstanceUID")[LABEL]
        sub = sub.loc[truth.index, [c for c in LABEL if c in sub.columns]]
        aucs = {c: roc_auc_score(truth[c], sub[c]) for c in sub.columns}
        print("GOLD per label:", json.dumps({k: round(v, 3) for k, v in aucs.items()}), flush=True)
        print("GOLD MEAN AUC:", round(float(np.mean(list(aucs.values()))), 4), flush=True)
print("PROBE DONE", flush=True)
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# RSNA Knee - OrthoFoundation fold0 feasibility probe\n",
            "Commit-run saja (tanpa submisi): perakitan best.pt, konfigurasi ckpt, run_inference pada stub dan pada 58 studi emas."]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": CODE.lstrip("\n").splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
compile(CODE, "probe_of", "exec")
(OUT / "rsna-knee-of-probe.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": "dimaspashaakrilian/rsna-knee-of-probe",
    "title": "RSNA Knee OF Probe",
    "code_file": "rsna-knee-of-probe.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", OUT)
