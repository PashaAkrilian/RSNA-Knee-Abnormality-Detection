#!/usr/bin/env python3
"""Evaluasi gold-58 (label radiolog dari GAMBAR): AUC tiap model, korelasi, dan blend resep kita + bootstrap berpasangan.
Masukan lokal: reference_metadata/train.csv (label gold), data/r7oof/compact_gold.csv (R7 4 fold, bersih),
data/teacher_r384_train_preds.csv (Raptor r384, TERKONTAMINASI: dipilih pd gold -> optimistis),
data/gold_eval/student_gold_preds.csv (murid r2a, kolom cnxt_*), data/s6f_gold/s6f_gold_probs.csv (S6F nartaa, bersih).
Catatan: tanpa r224 (tak ada prediksi gold); separuh Raptor = r384 saja. Angka = ARAH (58 studi, SE ±0,01; selisih berpasangan lebih kecil).
Pakai: python3 gold_eval_blend.py [--boot 2000]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 2000


def load(path, prefix=""):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    return d[[prefix + c for c in L]].set_axis(L, axis=1).astype(float)


tr = pd.read_csv(ROOT / "reference_metadata/train.csv", dtype={"StudyInstanceUID": str})
gold = tr[tr[L].notna().all(axis=1)].set_index("StudyInstanceUID")[L]
M = {"R7": load(ROOT / "data/r7oof/compact_gold.csv"), "Raptor384*": load(ROOT / "data/teacher_r384_train_preds.csv"),
     "murid_r2a": load(ROOT / "data/gold_eval/student_gold_preds.csv", "cnxt_"), "S6F": load(ROOT / "data/s6f_gold/s6f_gold_probs.csv")}
ids = gold.index
for k, v in M.items():
    ids = ids.intersection(v.dropna().index)
Y = (gold.loc[ids].values > 0.5).astype(int)
P = {k: v.loc[ids].values for k, v in M.items()}
print(f"studi gold terpakai: {len(ids)} (* = terkontaminasi)")


def rk(a):
    return pd.DataFrame(a).rank(pct=True).values


def macro(y, p):
    out = []
    for j in range(y.shape[1]):
        pos, neg = p[y[:, j] == 1, j], p[y[:, j] == 0, j]
        if len(pos) and len(neg):
            r = pd.Series(np.concatenate([pos, neg])).rank().values
            out.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(out))


R = {k: rk(v) for k, v in P.items()}
for k in P:
    print(f"  {k:12s} AUC {macro(Y, P[k]):.4f}")
print("korelasi Spearman rata-rata per temuan:")
names = list(P)
for i, a in enumerate(names):
    print("  " + a.ljust(12) + " ".join(f"{np.mean([pd.Series(R[a][:, j]).corr(pd.Series(R[b][:, j])) for j in range(12)]):6.3f}" for b in names))


def blend(ws6=0.0, wst=0.0, wr=0.5):
    compact = (1 - ws6) * R["R7"] + ws6 * R["S6F"]
    compact = (1 - wst) * compact + wst * R["murid_r2a"]
    return wr * R["Raptor384*"] + (1 - wr) * compact


V = {"dasar (Raptor .5 + R7 .5)": blend(), "+S6F .35": blend(ws6=0.35), "+S6F .40": blend(ws6=0.40),
     "+murid .25": blend(wst=0.25), "+murid .40": blend(wst=0.40), "+S6F .35 +murid .30": blend(0.35, 0.30),
     "+S6F .35 +murid .40": blend(0.35, 0.40), "+S6F .25 +murid .30": blend(0.25, 0.30)}
base = V["dasar (Raptor .5 + R7 .5)"]
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
print(f"blend (rank per temuan) — AUC gold & selisih vs dasar (bootstrap {B}x: rerata, P(>0)):")
for k, v in V.items():
    d = np.array([macro(Y[i], v[i]) - macro(Y[i], base[i]) for i in idx])
    print(f"  {k:28s} {macro(Y, v):.4f}   delta {d.mean():+.4f}  P(>0)={np.mean(d > 0):.2f}")
