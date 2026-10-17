#!/usr/bin/env python3
"""Penilaian LOKAL slot compact (usul user 17 Okt: cek mutu sebelum menunggu skor Kaggle).
Data: prediksi OUT-OF-FOLD bersih pada ±4.300 studi latih non-gold (tiap studi diprediksi model yg tak melatihnya):
  R6+flip `data/r6oof/`, R7 `data/r7oof/`, murid r2a/r3 `data/student_*/runs_student/oof_fold*.csv` (baris set=val).
Label = konsensus laporan (`labels_student_v0c.csv`, >0,5) -> derau label, tapi n besar (SE beda berpasangan ±0,001).
Raptor & S6F dilatih pada SEMUA data latih -> tak punya OOF bersih -> separuh Raptor dinilai di gold-58 (`gold_eval_full.py`).
Catatan bias: murid r2a dilatih ke 0,5 konsensus + 0,5 OOF-R7 -> sedikit diuntungkan oleh label konsensus.
Pakai: python3 local_eval_oof.py"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']


def compact_oof(d):
    f = sorted(glob.glob(str(ROOT / d / "compact_oof_f[0-3].csv")))
    o = pd.concat([pd.read_csv(x, dtype={"StudyInstanceUID": str}) for x in f])
    return o[o.scored == 1].drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[L].astype(float)


def student_oof(d):
    f = sorted(glob.glob(str(ROOT / d / "runs_student/oof_fold[0-3].csv")))
    if not f:
        return None
    o = pd.concat([pd.read_csv(x, index_col=0) for x in f])
    o = o[o["set"] == "val"][L]
    o.index = o.index.astype(str)
    return o[~o.index.duplicated()].astype(float)


def auc(y, p):
    r = pd.Series(p).rank().values
    y = y == 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def macro(Y, P):
    return float(np.mean([auc(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]))


lab = pd.read_csv(ROOT / "data/labels/labels_student_v0c.csv", index_col=0)
lab.index = lab.index.astype(str)
lab = lab[lab.is_gold == 0]
M = {"R6flip": compact_oof("data/r6oof"), "R7": compact_oof("data/r7oof"), "murid_r2a": student_oof("data/student_r2a")}
r3 = student_oof("data/student_r3")
if r3 is not None:
    M["murid_r3"] = r3
ids = lab.index
for k, v in M.items():
    ids = ids.intersection(v.index)
ids = list(ids)
Y = (lab.loc[ids, L].values > 0.5).astype(int)
R = {k: v.loc[ids].rank(pct=True).values for k, v in M.items()}   # rank per temuan, seperti blend notebook


def mix(**w):
    t = sum(w.values())
    return sum(wt * R[k] for k, wt in w.items()) / t


V = {k: R[k] for k in R}
V.update({"R6 .7 + murid .3 (g6a/g8 compact)": mix(R6flip=.7, murid_r2a=.3), "R7 .7 + murid .3 (r7v6a compact)": mix(R7=.7, murid_r2a=.3),
          "R6 .5 + R7 .5": mix(R6flip=.5, R7=.5), "R6 .35 + R7 .35 + murid .3": mix(R6flip=.35, R7=.35, murid_r2a=.3),
          "R6 .6 + murid .4": mix(R6flip=.6, murid_r2a=.4), "R6 .8 + murid .2": mix(R6flip=.8, murid_r2a=.2)})
if "murid_r3" in R:
    V.update({"R6 .7 + r3 .3": mix(R6flip=.7, murid_r3=.3), "R6 .7 + r2a .15 + r3 .15": mix(R6flip=.7, murid_r2a=.15, murid_r3=.15)})
base = V["R6flip"]
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(ids), len(ids)) for _ in range(200)]
print(f"OOF latih non-gold: {len(ids)} studi | label konsensus>0,5 | bootstrap 200x vs R6flip")
for k, v in V.items():
    d = np.array([macro(Y[i], v[i]) - macro(Y[i], base[i]) for i in idx])
    print(f"  {k:36s} {macro(Y, v):.4f} | vs R6flip {d.mean():+.4f} (P>0 {np.mean(d > 0):.2f})")
