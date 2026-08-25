#!/usr/bin/env python3
"""Fusion v2: tabel OOF gold-57 dgn CoAtNet hasil harvest + seleksi bobot ROBUST.
Pelajaran fusion v1: grid halus per-target overfit di fold kecil -> sekarang
kandidat KASAR (blend uniform subset) + shrinkage, dievaluasi nested grouped-CV.
Run: python3 fusion_nested_cv_v2.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")
OOF = Path("/tmp/opencode/oof")
HARV = Path("/tmp/opencode/harvest")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
ARMS = ["pk", "e9", "e11", "v18", "c5", "c4", "c4s"]


def macro_auc(y, s):
    aucs = []
    for j in range(y.shape[1]):
        pos, neg = s[y[:, j] == 1, j], s[y[:, j] == 0, j]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = pd.Series(np.concatenate([pos, neg])).rank().values
        aucs.append((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs))


def load_arms():
    tr = pd.read_csv(REPO / "reference_metadata/train.csv")
    gold = tr[tr[TARGETS].notna().all(axis=1)][["StudyInstanceUID"] + TARGETS].reset_index(drop=True)

    pk = np.load(OOF / "pk/oof.npz", allow_pickle=True)
    pk_df = pd.DataFrame(pk["pred"], columns=[f"pk_{t}" for t in TARGETS])
    pk_df["StudyInstanceUID"] = pk["ids"]

    e9 = pd.read_csv(OOF / "e9/v52_oof.csv")[["StudyInstanceUID"] + TARGETS]
    e9.columns = ["StudyInstanceUID"] + [f"e9_{t}" for t in TARGETS]
    e9f = pd.read_csv(OOF / "e9/v52_oof.csv")[["StudyInstanceUID", "fold"]]

    e11 = pd.read_csv(OOF / "e11/v52_e11_oof.csv")[["StudyInstanceUID"] + TARGETS]
    e11.columns = ["StudyInstanceUID"] + [f"e11_{t}" for t in TARGETS]

    v18 = pd.read_csv(REPO / "blend_results/oof_predictions_v18.csv")
    v18 = v18[["StudyInstanceUID"] + [f"pred_{t}" for t in TARGETS]]
    v18.columns = ["StudyInstanceUID"] + [f"v18_{t}" for t in TARGETS]

    df = gold.merge(pk_df, on="StudyInstanceUID").merge(e9, on="StudyInstanceUID") \
             .merge(e11, on="StudyInstanceUID").merge(v18, on="StudyInstanceUID")
    for tag, fn in [("c5", "oof_coatnet_v5_full_swa.csv"), ("c4", "oof_coatnet_v4_full.csv"),
                    ("c4s", "oof_coatnet_v4_full_swa.csv")]:
        c = pd.read_csv(HARV / fn)
        c.columns = ["StudyInstanceUID"] + [f"{tag}_{t}" for t in TARGETS]
        df = df.merge(c, on="StudyInstanceUID")
    df = df.merge(e9f.rename(columns={"fold": "grp"}), on="StudyInstanceUID", how="left")
    return df


def ranks(x):
    return pd.Series(x).rank().values


def R(df, arm):
    return np.column_stack([ranks(df[f"{arm}_{t}"].values) for t in TARGETS])


def boot_ci(y, s_a, s_b, n=3000, seed=0):
    """P(AUC_a > AUC_b) + CI selisih via bootstrap paired."""
    rng = np.random.default_rng(seed)
    n_st = len(y)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_st, n_st)
        diffs.append(macro_auc(y[idx], s_a[idx]) - macro_auc(y[idx], s_b[idx]))
    diffs = np.array(diffs)
    return float((diffs > 0).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    df = load_arms()
    print(f"studi: {len(df)} | arm: {ARMS}")
    y = df[TARGETS].values.astype(float)

    Rs = {a: R(df, a) for a in ARMS}
    print("\n== sanity macro AUC per arm ==")
    for a in ARMS:
        print(f"  {a:4s}: {macro_auc(y, Rs[a]):.4f}")

    # ---------- kandidat blend KASAR ----------
    CANDS = {
        "uniform7": ["pk", "e9", "e11", "v18", "c5", "c4", "c4s"],
        "uniform6_noV18": ["pk", "e9", "e11", "c5", "c4", "c4s"],
        "uniform4_core": ["pk", "e9", "e11", "c5"],
        "uniform3_coat": ["c5", "c4", "c4s"],
        "c5_only": ["c5"],
        "pk_e9_c5": ["pk", "e9", "c5"],
        "e9_e11_c5": ["e9", "e11", "c5"],
        "pk_c5": ["pk", "c5"],
        "e9_c5": ["e9", "c5"],
        "uniform5_pkE9E11_c5c4s": ["pk", "e9", "e11", "c5", "c4s"],
    }

    def blend(arms):
        return sum(Rs[a] for a in arms) / len(arms)

    print("\n== kandidat blend (full-gold, optimistis) ==")
    full = {}
    for name, arms in CANDS.items():
        full[name] = macro_auc(y, blend(arms))
        print(f"  {name:26s}: {full[name]:.4f}")

    # ---------- nested grouped-CV antar kandidat ----------
    grps = sorted(df["grp"].dropna().unique())
    picks = []
    oof_S = np.zeros_like(y)
    for g_out in grps:
        te = (df["grp"] == g_out).values
        tr_in = ~te
        best, bn = -1, None
        for name, arms in CANDS.items():
            auc = macro_auc(y[tr_in], blend(arms)[tr_in])
            if auc > best:
                best, bn = auc, name
        picks.append(bn)
        oof_S[te] = blend(CANDS[bn])[te]
    from collections import Counter
    print(f"\npilihan inner-fold: {dict(Counter(picks))}")
    nested = macro_auc(y, oof_S)

    # pembanding: paksa uniform6 tanpa tuning
    u6_oof = np.zeros_like(y)
    for g_out in grps:
        te = (df["grp"] == g_out).values
        u6_oof[te] = blend(CANDS["uniform6_noV18"])[te]

    best_single = max(full[f"{a}_only"] if f"{a}_only" in full else macro_auc(y, Rs[a]) for a in ARMS)
    print("\n== HASIL AKHIR (leak-free) ==")
    print(f"  best single arm              : {best_single:.4f}")
    print(f"  uniform6 (no V18) OOF        : {macro_auc(y, u6_oof):.4f}")
    print(f"  nested-CV pilih kandidat     : {nested:.4f}")

    top = max(CANDS, key=lambda n: full[n])
    p, lo, hi = boot_ci(y, blend(CANDS[top]), blend(CANDS["c5_only"]))
    print(f"\nbootstrap {top} vs c5-only: P(menang)={p:.3f}, dCI=[{lo:+.4f},{hi:+.4f}]")
    p2, lo2, hi2 = boot_ci(y, blend(CANDS[top]), Rs["e9"])
    print(f"bootstrap {top} vs e9     : P(menang)={p2:.3f}, dCI=[{lo2:+.4f},{hi2:+.4f}]")

    print("\n== AUC per target: uniform6 vs c5 ==")
    b6 = blend(CANDS["uniform6_noV18"])
    for j, t in enumerate(TARGETS):
        a5 = macro_auc(y[:, [j]], Rs["c5"][:, [j]])
        a6 = macro_auc(y[:, j][:, None], b6[:, [j]])
        print(f"  {t:17s}: c5={a5:.3f} | u6={a6:.3f} ({a6-a5:+.3f})")


if __name__ == "__main__":
    main()
