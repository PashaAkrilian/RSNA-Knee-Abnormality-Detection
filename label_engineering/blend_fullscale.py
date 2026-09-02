#!/usr/bin/env python3
"""Full-scale flight0234 + Claude report-relabel blend.

Step 1: re-validate the rank-mean blend methodology on the 58 gold studies
(ground truth available) to confirm the pilot's 0.9047 result and lock the
optimal alpha (weight on flight0234).

Step 2: apply that same rank-mean blend, at full scale, to all 4,349 non-gold
studies Claude just relabeled (label_engineering/claude_labels_nongold_full.csv)
+ flight0234's labels for the same studies -- producing the new training-label
table to feed into the next CoAtNet fine-tune.

Rank-mean (not raw-probability-mean): per label column, convert each source's
predictions to a percentile rank in [0,1] (pd.Series.rank(pct=True)) before
averaging -- established in this project as superior to raw-prob averaging
(prevents the most-confident source from dominating). See kaggle_kernel_submission_v2
for the same technique applied to model ensembling.
"""
import numpy as np
import pandas as pd

REPO = "/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection"
SCRATCH = "/tmp/claude-1000/-home-pasha-akrilian-Documents-RSNA-Knee-Abnormality-Detection/67853717-b3a0-436b-b62b-7f201f6d73c4/scratchpad"

train = pd.read_csv(f"{REPO}/reference_metadata/train.csv")
LABELS = train.columns[2:14].tolist()
gold_mask = train[LABELS].notna().all(axis=1)
gold = train[gold_mask].set_index("StudyInstanceUID")
print(f"gold studies: {len(gold)}")

flight_raw = pd.read_csv(f"{SCRATCH}/flight0234/report_labels_v4hybrid.csv").set_index("StudyInstanceUID")
pilot = pd.read_csv(f"{REPO}/label_engineering/claude_labels_pilot_58gold.csv").set_index("StudyInstanceUID")


def manual_auc(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def macro_auc(y_df, p_df, index):
    aucs = {}
    for c in LABELS:
        y = y_df.loc[index, c].astype(int).values
        p = p_df.loc[index, c].values
        aucs[c] = manual_auc(y, p)
    return float(np.nanmean(list(aucs.values()))), aucs


# ---- Step 1: re-validate rank-mean blend on gold-58 ----
common = gold.index.intersection(flight_raw.index).intersection(pilot.index)
print(f"gold overlap (flight & pilot & gold): {len(common)}/{len(gold)}")

flight = flight_raw.loc[common, LABELS].apply(pd.to_numeric, errors="coerce")
claude = pilot.loc[common, LABELS].apply(pd.to_numeric, errors="coerce")

flight_rank = flight.rank(pct=True)
claude_rank = claude.rank(pct=True)

m_f, _ = macro_auc(gold, flight, common)
m_c, _ = macro_auc(gold, claude, common)
print(f"flight0234 alone         macro={m_f:.4f}")
print(f"Claude pilot alone       macro={m_c:.4f}")

best_alpha, best_macro = None, -1
print("\n--- alpha sweep (rank-mean blend, weight on flight0234) ---")
for alpha in np.arange(0.0, 1.01, 0.05):
    blend = alpha * flight_rank + (1 - alpha) * claude_rank
    m, _ = macro_auc(gold, blend, common)
    marker = ""
    if m > best_macro:
        best_macro, best_alpha = m, alpha
        marker = "  <-- best so far"
    print(f"  alpha(flight)={alpha:.2f}  macro={m:.4f}{marker}")

print(f"\nBEST: alpha(flight)={best_alpha:.2f}  macro={best_macro:.4f}")

# ---- Step 2: apply at full scale to the 4,349 non-gold studies ----
print("\n--- applying blend at full scale (4,349 non-gold studies) ---")
master = pd.read_csv(f"{REPO}/label_engineering/claude_labels_nongold_full.csv").set_index("StudyInstanceUID")
flight_full = flight_raw.loc[master.index, LABELS].apply(pd.to_numeric, errors="coerce")
claude_full = master[LABELS].apply(pd.to_numeric, errors="coerce")

flight_full_rank = flight_full.rank(pct=True)
claude_full_rank = claude_full.rank(pct=True)

blended_full = best_alpha * flight_full_rank + (1 - best_alpha) * claude_full_rank
blended_full = blended_full.reset_index()

out_path = f"{REPO}/label_engineering/blended_labels_fullscale_train.csv"
blended_full.to_csv(out_path, index=False)
print(f"saved: {out_path}  ({len(blended_full)} rows)")

# sanity checks
label_cols = LABELS
print("out of range:", ((blended_full[label_cols] < 0) | (blended_full[label_cols] > 1)).any().any())
print("nulls:", blended_full[label_cols].isnull().sum().sum())
print("\nper-label mean (blended, full scale):")
print(blended_full[label_cols].mean().round(3))


# ---- Addendum (v3, post-hoc after v2 negative result) ----
# v2 CoAtNet fine-tune trained on this rank-mean blend scored 0.9170 gold-58 -- WORSE than
# flight0234-only (0.9193) and c5 baseline (0.9206), despite rank-mean scoring BETTER as a
# label-matching metric (0.9047 vs 0.8991). Root cause: rank(pct=True) forces every label
# column into an artificially uniform distribution, destroying the true base-rate skew that
# BCE training needs -- good for AUC (rank-only), bad as a literal regression target.
# Fix: raw-probability-mean instead of rank-mean, preserving calibration. See PROGRESS.md §46.
if __name__ == "__main__":
    print("\n\n=== v3 addendum: raw-probability-mean blend (calibration-preserving) ===")
    best_alpha_raw, best_macro_raw = None, -1
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * flight + (1 - alpha) * claude
        m, _ = macro_auc(gold, blend, common)
        if m > best_macro_raw:
            best_macro_raw, best_alpha_raw = m, alpha
    print(f"BEST raw-mean: alpha(flight)={best_alpha_raw:.2f}  macro={best_macro_raw:.4f}")

    blended_full_raw = best_alpha_raw * flight_full + (1 - best_alpha_raw) * claude_full
    blended_full_raw = blended_full_raw.reset_index()
    out_path_raw = f"{REPO}/label_engineering/blended_labels_fullscale_train_rawmean.csv"
    blended_full_raw.to_csv(out_path_raw, index=False)
    print(f"saved: {out_path_raw}  ({len(blended_full_raw)} rows)")
    print("out of range:", ((blended_full_raw[label_cols] < 0) | (blended_full_raw[label_cols] > 1)).any().any())
    print("per-label std (raw-mean, full scale):")
    print(blended_full_raw[label_cols].std().round(3))
