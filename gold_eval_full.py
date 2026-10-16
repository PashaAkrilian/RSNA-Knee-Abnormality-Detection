#!/usr/bin/env python3
"""Evaluasi gold-58 SETIA: komponen dari run pipeline submisi pada 58 studi gold (`build_gold_full.py`; data/gold_full/<run>/).
Komponen: r384 (biasa+cermin) raptor_raw, r224 (biasa+cermin) raptor224_sub (rank; urutan = probabilitas), R7 ours_sub (gold-nf tanpa flip / gold-f flip),
S6F s6f_raw, murid student_test_preds (cnxt_*). Blend direkonstruksi seperti sel blend notebook (rank pct per model per temuan).
Cek: rekonstruksi resep run gold-nf (= r7v3sr) harus sama dgn submission.csv-nya. Lalu varian + bootstrap berpasangan.
Pakai: python3 gold_eval_full.py [--boot 2000]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
G = ROOT / "data/gold_full"
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 2000
tr = pd.read_csv(ROOT / "reference_metadata/train.csv", dtype={"StudyInstanceUID": str})
gold = tr[tr[L].notna().all(axis=1)].set_index("StudyInstanceUID")[L]
ids = list(gold.index)
Y = (gold.loc[ids].values > 0.5).astype(int)


def load(path, prefix=""):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    return d.reindex(ids)[[prefix + c for c in L]].to_numpy(np.float64)


def pct(a):
    return pd.DataFrame(a).rank(pct=True).values


nf, fl = G / "gold-nf", G / "gold-f"
P = {"r384m": load(nf / "raptor_raw.csv"), "r224m": load(nf / "raptor224_sub.csv"), "R7": load(nf / "ours_sub.csv"),
     "S6F": load(nf / "s6f_raw.csv"), "murid": load(nf / "student_test_preds.csv", "cnxt_")}
if (fl / "ours_sub.csv").is_file():
    P["R7flip"] = load(fl / "ours_sub.csv")
for k, v in P.items():
    assert np.isfinite(v).all(), (k, int((~np.isfinite(v)).sum()))
R = {k: pct(v) for k, v in P.items()}


def macro(y, p):
    out = []
    for j in range(y.shape[1]):
        pos, neg = p[y[:, j] == 1, j], p[y[:, j] == 0, j]
        if len(pos) and len(neg):
            r = pd.Series(np.concatenate([pos, neg])).rank().values
            out.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(out))


WC_H = {'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}


def recipe(s6=0.0, s6slot="compact", st=0.0, wc=None, r7="R7", a224=0.35, wr=0.5):
    rr = (1 - a224) * R["r384m"] + a224 * R["r224m"]
    ro = R[r7].copy()
    if s6:
        if s6slot == "compact":
            ro = (1 - s6) * ro + s6 * R["S6F"]
        else:
            rr = (1 - s6) * rr + s6 * R["S6F"]
    if st:
        ro = (1 - st) * ro + st * R["murid"]
    w = np.array([(wc or {}).get(c, wr) for c in L])
    return w * rr + (1 - w) * ro


if __name__ == "__main__":
    print(f"gold {len(ids)} studi | komponen: " + "  ".join(f"{k} {macro(Y, v):.4f}" for k, v in P.items()))
    names = list(P)
    print("Spearman rata-rata:\n" + "\n".join("  " + a.ljust(7) + " ".join(
        f"{np.mean([pd.Series(R[a][:, j]).corr(pd.Series(R[b][:, j])) for j in range(12)]):6.3f}" for b in names) for a in names))
    sub = load(nf / "submission.csv")
    rec = recipe(0.35, "raptor", 0.30)
    print("cek rekonstruksi r7v3sr vs submission.csv gold-nf: maks |selisih| %.2e" % np.abs(rec - sub).max())
    V = {"r7r224m (dasar)": recipe(), "r7v3c (murid .25)": recipe(st=0.25), "r7v3s (S6F .35 compact + murid .30)": recipe(0.35, "compact", 0.30),
         "r7v3sr (S6F .35 Raptor + murid .30)": recipe(0.35, "raptor", 0.30), "r7v3sw (r7v3s + wc heliosli)": recipe(0.35, "compact", 0.30, WC_H),
         "r7v3sr + wc heliosli": recipe(0.35, "raptor", 0.30, WC_H),
         "S6F Raptor .25 + murid .30": recipe(0.25, "raptor", 0.30), "S6F Raptor .45 + murid .30": recipe(0.45, "raptor", 0.30),
         "S6F Raptor .35 + murid .20": recipe(0.35, "raptor", 0.20), "S6F Raptor .35 + murid .40": recipe(0.35, "raptor", 0.40),
         "S6F Raptor .35 + murid .50": recipe(0.35, "raptor", 0.50)}
    if "R7flip" in P:
        V["r7v3sr + flip"] = recipe(0.35, "raptor", 0.30, r7="R7flip")
        V["r7r224m + flip"] = recipe(r7="R7flip")
    base = V["r7r224m (dasar)"]
    ref = V["r7v3sr (S6F .35 Raptor + murid .30)"]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
    print(f"varian: AUC gold | vs dasar (rerata, P>0) | vs r7v3sr (rerata, P>0)   [bootstrap {B}x]")
    for k, v in V.items():
        d0 = np.array([macro(Y[i], v[i]) - macro(Y[i], base[i]) for i in idx])
        d1 = np.array([macro(Y[i], v[i]) - macro(Y[i], ref[i]) for i in idx])
        print(f"  {k:38s} {macro(Y, v):.4f} | {d0.mean():+.4f} {np.mean(d0 > 0):.2f} | {d1.mean():+.4f} {np.mean(d1 > 0):.2f}")
