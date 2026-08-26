#!/usr/bin/env python3
"""Cek lokal (tanpa GPU/training): apakah blend BEBERAPA tabel label report (bukan cuma
flight0234 sendirian) lebih akurat vs 58 studi gold -- diadopsi dari temuan Discussion topic
737454/notebook starkhushi (docs/RESEARCH_NOTES.md riset baru). Pola sama spt blend OOF kita
(0.8462, tanpa training tambahan) -- sekarang diterapkan di level LABEL, bukan prediksi.
"""
import numpy as np
import pandas as pd

REPO = "/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection"
train = pd.read_csv(f"{REPO}/reference_metadata/train.csv")
LABELS = train.columns[2:14].tolist()
gold_mask = train[LABELS].notna().all(axis=1)
gold = train[gold_mask].set_index("StudyInstanceUID")
print(f"gold studies: {len(gold)}")

def manual_auc(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def score(df, name):
    df = df.set_index("StudyInstanceUID")
    cols = {}
    for c in LABELS:
        for dc in df.columns:
            if dc.lower().replace("_", " ").replace("'", "") == c.lower().replace("'", ""):
                cols[c] = dc; break
    common = gold.index.intersection(df.index)
    aucs = {}
    for c in LABELS:
        if c not in cols:
            continue
        y = gold.loc[common, c].astype(int)
        p = pd.to_numeric(df.loc[common, cols[c]], errors="coerce")
        aucs[c] = manual_auc(y.values, p.values)
    macro = float(np.nanmean(list(aucs.values())))
    print(f"{name:32s} macro={macro:.4f}  coverage={len(common)}/{len(gold)}")
    return macro, aucs, cols, common

DIR = "/tmp/claude-1000/-home-pasha-akrilian-Documents-RSNA-Knee-Abnormality-Detection/67853717-b3a0-436b-b62b-7f201f6d73c4/scratchpad/label_check"
flight = pd.read_csv(f"{DIR}/report_labels_v4hybrid.csv")
steven = pd.read_csv(f"{DIR}/llm_labels_v4_blend.csv")
pilk = pd.read_csv(f"{DIR}/report_labels_v2.csv")

print("--- individual tables ---")
m_f, a_f, c_f, common_f = score(flight, "flight0234 (v4hybrid, kita pakai)")
m_s, a_s, c_s, common_s = score(steven, "stevenleehans v4 blend")
m_p, a_p, c_p, common_p = score(pilk, "pilkwang report_labels_v2")

print("\n--- blends (rata-rata sederhana, semua tabel yang overlap) ---")
def get_matrix(df, cols, index):
    df = df.set_index("StudyInstanceUID")
    out = pd.DataFrame(index=index)
    for c in LABELS:
        if c in cols:
            out[c] = pd.to_numeric(df.loc[index, cols[c]], errors="coerce")
        else:
            out[c] = np.nan
    return out

common_all = gold.index.intersection(flight.set_index("StudyInstanceUID").index) \
    .intersection(steven.set_index("StudyInstanceUID").index) \
    .intersection(pilk.set_index("StudyInstanceUID").index)
print(f"studi overlap 3 tabel: {len(common_all)}/{len(gold)}")

F = get_matrix(flight, c_f, common_all)
S = get_matrix(steven, c_s, common_all)
P = get_matrix(pilk, c_p, common_all)

for name, blend in [
    ("mean(flight, steven)", (F + S) / 2),
    ("mean(flight, pilk)", (F + P) / 2),
    ("mean(steven, pilk)", (S + P) / 2),
    ("mean(flight, steven, pilk)", (F + S + P) / 3),
    ("mean(flight*0.6, steven*0.4)", F * 0.6 + S * 0.4),
    ("mean(flight*0.7, steven*0.3)", F * 0.7 + S * 0.3),
]:
    aucs = {}
    for c in LABELS:
        y = gold.loc[common_all, c].astype(int).values
        p = blend[c].values
        aucs[c] = manual_auc(y, p)
    macro = float(np.nanmean(list(aucs.values())))
    print(f"{name:32s} macro={macro:.4f}")

print("\n--- per-label (flight0234 vs blend flight+steven, N=", len(common_all), ") ---")
blend_fs = (F + S) / 2
rows = []
for c in LABELS:
    y = gold.loc[common_all, c].astype(int).values
    auc_f = manual_auc(y, F[c].values)
    auc_b = manual_auc(y, blend_fs[c].values)
    rows.append((c, auc_f, auc_b, auc_b - auc_f))
rows.sort(key=lambda r: r[3])
for c, af, ab, d in rows:
    print(f"  {c:20s} flight={af:.3f}  blend={ab:.3f}  delta={d:+.3f}")
