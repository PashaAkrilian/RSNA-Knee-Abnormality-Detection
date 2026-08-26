#!/usr/bin/env python3
"""Blend OOF v18 (SlotHead+GAP) + v23 (SlotHead+SpatialAttnPool) -- v23 sendirian (0,7967)
sedikit di bawah v18 (0,8068), TAPI per-label mengungkap pola komplementer: v23 JAUH lebih baik
di temuan kecil/terlokalisir (Medial Meniscus +0,103, Lateral Meniscus +0,064 -- persis hipotesis
yg diuji, docs/RESEARCH_NOTES.md §27b/§28), v18 lebih baik di temuan difus (MCL -0,107, Contusion
-0,096, Synovitis -0,049 dari sudut pandang v23). Blend rank-mean menggabungkan kekuatan keduanya
-> OOF TERBAIK BARU keluarga SlotHead (0,8176 @ alpha~0,5-0,65), ROBUST di rentang lebar (semua
alpha 0,30-0,85 > 0,81) -- bukan artefak titik tunggal, pola sama spt blend v18+pilkwang (§22).
"""
import numpy as np
import pandas as pd

v18 = pd.read_csv("blend_results/oof_predictions_v18.csv").set_index("StudyInstanceUID")
v23 = pd.read_csv("blend_results/oof_predictions_v23.csv").set_index("StudyInstanceUID")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def manual_auc(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def rankpct(x):
    return pd.Series(x).rank(pct=True).values

common = v18.index.intersection(v23.index)
print(f"studi overlap: {len(common)}")

print("\n--- per-label: v18(GAP) vs v23(SpatialAttnPool) ---")
rows = []
for c in LABELS:
    a18 = manual_auc(v18.loc[common, f"true_{c}"], v18.loc[common, f"pred_{c}"])
    a23 = manual_auc(v23.loc[common, f"true_{c}"], v23.loc[common, f"pred_{c}"])
    rows.append((c, a18, a23, a23 - a18))
rows.sort(key=lambda r: r[3])
for c, a, b, d in rows:
    print(f"  {c:20s} v18={a:.4f}  v23={b:.4f}  delta={d:+.4f}")

print("\n--- sweep alpha (bobot v18) ---")
best = (0.0, None)
for alpha in np.arange(0, 1.05, 0.05):
    aucs = []
    for c in LABELS:
        y = v18.loc[common, f"true_{c}"]
        p18 = rankpct(v18.loc[common, f"pred_{c}"])
        p23 = rankpct(v23.loc[common, f"pred_{c}"])
        blend = alpha * p18 + (1 - alpha) * p23
        aucs.append(manual_auc(y, blend))
    m = float(np.nanmean(aucs))
    print(f"  alpha(v18)={alpha:.2f}  macro={m:.4f}")
    if m > best[0]:
        best = (m, alpha)

print(f"\nBEST: macro={best[0]:.4f} @ alpha(v18)={best[1]:.2f}")
print(f"v18 alone=0.8068 | v23 alone=0.7967 | blend terbaik={best[0]:.4f} (+{best[0]-0.8068:+.4f} vs v18)")
