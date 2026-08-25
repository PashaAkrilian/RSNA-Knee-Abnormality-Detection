#!/usr/bin/env python3
"""Fusion nested-CV di gold-58 antar arm berbasis OOF publik + OOF kita.
Arms: pilkwang 20-model (oof.npz), e9 RadImageNet heads, e11 diverse heads,
v18 (SlotHead DINOv2-base kita). Semua digabung rank-space; bobot per-target
dipilih via grouped-CV (leak-free), dievaluasi out-of-fold.
Run: python3 fusion_nested_cv.py  (numpy+pandas saja)
"""
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")
OOF = Path("/tmp/opencode/oof")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


def macro_auc(y, s):
    """Macro ROC-AUC rank-based (Mann-Whitney). y binary, s score."""
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
    pk_fold = pd.Series(pk["gold_mask"]).astype(int)

    e9 = pd.read_csv(OOF / "e9/v52_oof.csv")[["StudyInstanceUID"] + TARGETS]
    e9.columns = ["StudyInstanceUID"] + [f"e9_{t}" for t in TARGETS]
    e9f = pd.read_csv(OOF / "e9/v52_oof.csv")[["StudyInstanceUID", "fold"]]

    e11 = pd.read_csv(OOF / "e11/v52_e11_oof.csv")[["StudyInstanceUID"] + TARGETS]
    e11.columns = ["StudyInstanceUID"] + [f"e11_{t}" for t in TARGETS]

    v18 = pd.read_csv(REPO / "blend_results/oof_predictions_v18.csv")
    v18 = v18[["StudyInstanceUID"] + [f"pred_{t}" for t in TARGETS]]
    v18.columns = ["StudyInstanceUID"] + [f"v18_{t}" for t in TARGETS]

    df = gold.merge(pk_df, on="StudyInstanceUID", how="inner")
    df = df.merge(e9, on="StudyInstanceUID", how="inner")
    df = df.merge(e11, on="StudyInstanceUID", how="inner")
    df = df.merge(v18, on="StudyInstanceUID", how="inner")
    df = df.merge(e9f.rename(columns={"fold": "grp"}), on="StudyInstanceUID", how="left")
    return df


def ranks(x):
    return pd.Series(x).rank().values


def blend_eval(df, W):
    """W: dict arm->array(12). Blended rank-score per target."""
    S = np.zeros((len(df), 12))
    tot = np.zeros(12)
    for arm, w in W.items():
        R = np.column_stack([ranks(df[f"{arm}_{t}"].values) for t in TARGETS])
        S += R * w[None, :]
        tot += w
    return S


def main():
    df = load_arms()
    n = len(df)
    print(f"studi gold gabungan (inner-join semua arm): {n} "
          f"(v18 kehilangan {58-n} studi dikenal)")
    y = df[TARGETS].values.astype(float)

    ARMS = ["pk", "e9", "e11", "v18"]
    print("\n== sanity macro AUC per arm (gold gabungan) ==")
    for a in ARMS:
        S = np.column_stack([ranks(df[f"{a}_{t}"].values) for t in TARGETS])
        print(f"  {a:4s}: {macro_auc(y, S):.4f}")

    # ---- Grid per-target, nested grouped-CV pakai kolom grp (fold e9) ----
    grps = sorted(df["grp"].dropna().unique())
    steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def grids():
        # semua kombinasi bobot 4-arm utk SATU target (sum bebas, dinormalisasi)
        from itertools import product
        for w in product(steps, repeat=len(ARMS)):
            if sum(w) == 0:
                continue
            yield np.array(w) / sum(w)

    G = list(grids())
    rng = np.random.default_rng(0)
    W_final = {a: np.zeros(12) for a in ARMS}
    oof_S = np.zeros_like(y)
    picked_log = []

    for j, t in enumerate(TARGETS):
        sel = []
        for g_out in grps:
            te = (df["grp"] == g_out).values
            tr = ~te
            best, bw = -1, None
            for w in G:
                S_tr = sum(ranks(df[f"{a}_{t}"].values[tr]) * w[k] for k, a in enumerate(ARMS))
                auc = macro_auc(y[np.ix_(tr, [j])], S_tr[:, None])
                if auc > best:
                    best, bw = auc, w
            sel.append(bw)
            S_te = sum(ranks(df[f"{a}_{t}"].values[te]) * bw[k] for k, a in enumerate(ARMS))
            oof_S[te, j] = S_te
        Wm = np.mean(sel, axis=0)
        for k, a in enumerate(ARMS):
            W_final[a][j] = Wm[k]
        picked_log.append((t, dict(zip(ARMS, np.round(Wm, 3)))))

    print("\n== weight map hasil nested-CV (rata-rata pilihan inner-fold) ==")
    for t, w in picked_log:
        print(f"  {t:17s}: {w}")

    print("\n== HASIL ==")
    base_best = max(macro_auc(y, np.column_stack([ranks(df[f'{a}_{t}'].values) for t in TARGETS])) for a in ARMS)
    blend_oof = macro_auc(y, oof_S)
    print(f"  single-arm terbaik          : {base_best:.4f}")
    print(f"  BLEND nested-CV (leak-free) : {blend_oof:.4f}  ({blend_oof-base_best:+.4f})")

    # pembanding: blend uniform 4-arm
    Su = sum(np.column_stack([ranks(df[f'{a}_{t}'].values) for t in TARGETS]) for a in ARMS) / 4
    print(f"  blend uniform (referensi)   : {macro_auc(y, Su):.4f}")
    # pembanding: pk+e9 saja @0.5/0.5
    Spk9 = (np.column_stack([ranks(df[f'pk_{t}'].values) for t in TARGETS])
            + np.column_stack([ranks(df[f'e9_{t}'].values) for t in TARGETS])) / 2
    print(f"  pk+e9 50:50 (referensi)     : {macro_auc(y, Spk9):.4f}")


if __name__ == "__main__":
    main()
