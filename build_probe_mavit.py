#!/usr/bin/env python3
"""Bangun kernel probe MaVIT (akun utama): evaluasi GRATIS pada 58 studi emas + uji jalur submit stub.

Varian: hengck23 MaVIT-288 (00000008.pth) dan lixin73 MaVIT-336 fold0 (kode model = KneeModel milik hengck23).
Keluaran (/kaggle/working): gold_<varian>.csv (probabilitas per studi) + gold_truth.csv + probe_summary.json.
Tidak ada submisi; commit-run saja. Pemakaian:  python3 build_probe_mavit.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "kaggle_kernel_mavit_probe"
OUT.mkdir(exist_ok=True)

CODE = r'''
import os, sys, time, json, traceback
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

HK = "/kaggle/input/datasets/hengck23/hengck23-rnsa-knee"
LX = "/kaggle/input/datasets/lixin73/rsna-knee-mavit-336px-v51-weights"
COMP = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
sys.path.append(HK)
from model import KneeModel
from my_fastread_knee import make_knee_dataset_reader

LABEL = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
         'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count(), flush=True)
for p in (HK, LX, COMP):
    print(p, "exists:", os.path.isdir(p), (sorted(os.listdir(p))[:12] if os.path.isdir(p) else ""), flush=True)

STUDY = pd.read_csv(f"{HK}/train_golden58.csv")
SERIES = pd.read_csv(f"{HK}/train_golden58_series.csv")
TRUTH = STUDY[LABEL].values
STUDY_ID = STUDY["StudyInstanceUID"].tolist()
pd.DataFrame(TRUTH, index=STUDY_ID, columns=LABEL).rename_axis("StudyInstanceUID").to_csv("/kaggle/working/gold_truth.csv")
print("gold studies:", len(STUDY_ID), flush=True)


def pick_state_dict(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    info = {"file": os.path.basename(path), "type": type(saved).__name__}
    sd = saved
    if isinstance(saved, dict):
        info["top_keys"] = [str(k) for k in list(saved.keys())[:15]]
        for k in ("model_state_dict", "model", "state_dict", "ema", "ema_state_dict", "net"):
            if k in saved and isinstance(saved[k], dict):
                sd = saved[k]
                info["used_key"] = k
                break
        else:
            info["used_key"] = "<root>"
    return sd, info


def build(path):
    sd, info = pick_state_dict(path)
    best = None
    for strip in ("", "module.", "model.", "_orig_mod."):
        sd2 = {(k[len(strip):] if strip and k.startswith(strip) else k): v for k, v in sd.items()}
        m = KneeModel(pretrained=False)
        r = m.load_state_dict(sd2, strict=False)
        score = len(r.missing_keys) + len(r.unexpected_keys)
        if best is None or score < best[0]:
            best = (score, strip, m, r)
    score, strip, m, r = best
    info.update(strip=strip, n_params_sd=len(sd), missing=len(r.missing_keys), unexpected=len(r.unexpected_keys),
                missing_head=list(r.missing_keys)[:6], unexpected_head=list(r.unexpected_keys)[:6])
    print("CKPT", json.dumps(info), flush=True)
    return m.cuda().eval(), info


def macro(truth, pred):
    aucs = []
    for i in range(truth.shape[1]):
        t, p = truth[:, i], pred[:, i]
        ok = ~np.isnan(p)
        aucs.append(roc_auc_score(t[ok], p[ok]) if 0 < t[ok].sum() < ok.sum() else np.nan)
    return float(np.nanmean(aucs)), aucs


def run(name, ckpt, size, k, crop_mm=130.0, study_ids=None, series=None, dicom_dir=None, truth=None):
    model, info = build(ckpt)
    reader = make_knee_dataset_reader(out_size=size, num_slice=k, percentile=(1.0, 99.0), crop_mm=crop_mm,
                                      io_threads=16, crop_only_if_nonsquare=True)
    ids = study_ids if study_ids is not None else STUDY_ID
    ser = series if series is not None else SERIES
    ddir = dicom_dir if dicom_dir is not None else f"{COMP}/train_series"
    preds, fails, t0 = [], 0, time.time()
    for i, sid in enumerate(ids):
        try:
            r = reader(sid, ser[ser["StudyInstanceUID"] == sid], ddir, "valid")
            vol = torch.from_numpy(r["volume"]).unsqueeze(0).byte().cuda()
            vm = torch.from_numpy(r["view_mask"]).unsqueeze(0).bool().cuda()
            sl = torch.from_numpy(r["slice"]).unsqueeze(0).float().cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model.infer(vol, vm, sl)).float().cpu().numpy()
        except Exception as e:
            fails += 1
            if fails <= 3:
                print("FAIL", name, sid[:20], repr(e)[:300], flush=True)
                traceback.print_exc()
            p = np.full((1, 12), np.nan, np.float32)
        preds.append(p)
    pred = np.concatenate(preds, 0)
    sec = (time.time() - t0) / max(1, len(ids))
    res = {"name": name, "size": size, "k": k, "crop_mm": crop_mm, "sec_per_study": round(sec, 3), "fails": fails}
    if truth is not None:
        m, aucs = macro(truth, pred)
        res.update(mean_auc=round(m, 4), per_label={l: round(a, 3) for l, a in zip(LABEL, aucs)})
    print("RESULT", json.dumps(res), flush=True)
    df = pd.DataFrame(pred, index=ids, columns=LABEL).rename_axis("StudyInstanceUID")
    df.to_csv(f"/kaggle/working/{name}.csv")
    del model
    torch.cuda.empty_cache()
    return res


summary = []
CK288 = f"{HK}/00000008.pth"
CK336 = f"{LX}/mavit_336px_seed48_fold0.pt"

# 1) hengck23 MaVIT-288 (pembanding terukur: gold 0.9202, LB solo 0.935)
summary.append(run("gold_hk288", CK288, 288, 8, truth=TRUTH))
# 2) lixin73 MaVIT-336 fold0 (tanpa kode publik): coba K=8 dan K=12
if os.path.isfile(CK336):
    for k in (8, 12):
        try:
            summary.append(run(f"gold_lx336_k{k}", CK336, 336, k, truth=TRUTH))
        except Exception as e:
            print("VARIANT FAIL", k, repr(e)[:400], flush=True)
            traceback.print_exc()
    # 3) satu varian crop lebih lebar utk 336 (jaga-jaga bila dilatih dgn crop_mm lain)
    try:
        summary.append(run("gold_lx336_k8_crop150", CK336, 336, 8, crop_mm=150.0, truth=TRUTH))
    except Exception as e:
        print("VARIANT FAIL crop150", repr(e)[:400], flush=True)
else:
    print("CK336 tidak ada:", CK336, flush=True)

# 4) uji jalur submit (stub test: sample_submission) -> memastikan baca test_series bekerja di lingkungan ini
try:
    ss = pd.read_csv(f"{COMP}/sample_submission.csv")
    ts = pd.read_csv(f"{COMP}/test_series.csv")
    summary.append(run("stub_hk288", CK288, 288, 8, study_ids=ss["StudyInstanceUID"].tolist(), series=ts,
                       dicom_dir=f"{COMP}/test_series"))
except Exception as e:
    print("STUB FAIL", repr(e)[:400], flush=True)
    traceback.print_exc()

json.dump(summary, open("/kaggle/working/probe_summary.json", "w"), indent=1)
print("SUMMARY", json.dumps([{k: v for k, v in s.items() if k != "per_label"} for s in summary]), flush=True)
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# RSNA Knee - MaVIT probe (gold-58 + stub)\n",
            "Evaluasi gratis (commit-run, tanpa submisi) dua checkpoint MaVIT publik pada 58 studi emas: ",
            "hengck23 MaVIT-288 dan lixin73 MaVIT-336 fold0 (memakai kode `KneeModel` hengck23). ",
            "Kredit: hengck23 (kode, MaViT/MALA), nartaa (fastreader), lixin73 (bobot 336px)."]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": CODE.lstrip("\n").splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(OUT / "rsna-knee-mavit-probe.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "dimaspashaakrilian/rsna-knee-mavit-probe",
    "title": "RSNA Knee MaVIT Probe",
    "code_file": "rsna-knee-mavit-probe.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": ["hengck23/hengck23-rnsa-knee", "lixin73/rsna-knee-mavit-336px-v51-weights"],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", OUT)
