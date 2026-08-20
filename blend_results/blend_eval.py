import numpy as np
import pandas as pd

LABEL_COLS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
              "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

def manual_auc(y_true, y_score):
    """Rank-based AUC (Mann-Whitney U), pure numpy -- sklearn tidak tersedia lokal."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    valid = ~np.isnan(y_true)
    y_true, y_score = y_true[valid], y_score[valid]
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score))
    sorted_scores = y_score[order]
    # tie-averaged ranks
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i:j+1]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc

def macro_auc(true_mat, pred_mat, label_cols=LABEL_COLS):
    scores = []
    for j, c in enumerate(label_cols):
        a = manual_auc(true_mat[:, j], pred_mat[:, j])
        scores.append(a)
    return float(np.nanmean(scores)), scores

def pct_rank(mat):
    """Percentile rank per kolom -- pola sama dgn write_submission komunitas, biar skala 2 sumber
    prediksi sebanding sebelum diblend."""
    out = np.zeros_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        out[:, j] = pd.Series(mat[:, j]).rank(pct=True).values
    return out

# --- load OOF kita sendiri (v18, 5-fold) ---
ours = pd.read_csv("model_outputs/oof_predictions.csv")
ours_ids = ours["StudyInstanceUID"].tolist()
ours_pred = ours[[f"pred_{c}" for c in LABEL_COLS]].values
ours_true = ours[[f"true_{c}" for c in LABEL_COLS]].values

# --- load OOF pilkwang (20-model komunitas) ---
d = np.load("oof.npz", allow_pickle=True)
pk_ids_all = d["ids"]
pk_pred_all = d["pred"]
pk_true_all = d["y_derived"]
pk_gold_mask = d["gold_mask"].astype(bool)
pk_targets = list(d["targets"])
assert pk_targets == LABEL_COLS, f"urutan label pilkwang beda: {pk_targets}"

pk_gold_ids = pk_ids_all[pk_gold_mask]
pk_gold_pred = pk_pred_all[pk_gold_mask]
pk_gold_true = pk_true_all[pk_gold_mask]
pk_df = pd.DataFrame({"StudyInstanceUID": pk_gold_ids})
for j, c in enumerate(LABEL_COLS):
    pk_df[f"pk_pred_{c}"] = pk_gold_pred[:, j]
    pk_df[f"pk_true_{c}"] = pk_gold_true[:, j]

print(f"OURS: {len(ours_ids)} studi | PILKWANG gold: {len(pk_gold_ids)} studi")

# --- gabungkan pada irisan StudyInstanceUID yang sama ---
merged = ours.merge(pk_df, on="StudyInstanceUID", how="inner")
print(f"Irisan studi (dipakai evaluasi & blend): {len(merged)}")

true_mat = merged[[f"true_{c}" for c in LABEL_COLS]].values
ours_mat = merged[[f"pred_{c}" for c in LABEL_COLS]].values
pk_mat = merged[[f"pk_pred_{c}" for c in LABEL_COLS]].values

# sanity: cross-check label sama antara "true_" (kita) vs "pk_true_" (pilkwang) -- keduanya harus
# berasal dari train.csv resmi yang sama, kalau beda ada masalah join/label
pk_true_mat = merged[[f"pk_true_{c}" for c in LABEL_COLS]].values
mismatch = np.nansum(np.abs(np.nan_to_num(true_mat) - np.nan_to_num(pk_true_mat)))
print(f"Sanity cek label (harus 0): total selisih absolut true_ vs pk_true_ = {mismatch}")

auc_ours, per_ours = macro_auc(true_mat, ours_mat)
auc_pk, per_pk = macro_auc(true_mat, pk_mat)
print(f"\nMacro AUC (irisan {len(merged)} studi, leak-free keduanya):")
print(f"  Kita sendiri (v18, 5-fold OOF)      : {auc_ours:.4f}")
print(f"  Komunitas (pilkwang, 20-model OOF)   : {auc_pk:.4f}")

# --- blend rank-space, sweep alpha ---
ours_rank = pct_rank(ours_mat)
pk_rank = pct_rank(pk_mat)

print("\nSweep blend alpha (0=kita saja, 1=pilkwang saja), rank-space:")
best_alpha, best_auc = None, -1
results = []
for alpha in np.arange(0.0, 1.01, 0.05):
    blend = (1 - alpha) * ours_rank + alpha * pk_rank
    auc_b, _ = macro_auc(true_mat, blend)
    results.append((alpha, auc_b))
    marker = ""
    if auc_b > best_auc:
        best_auc, best_alpha = auc_b, alpha
        marker = "  <-- best so far"
    print(f"  alpha={alpha:.2f}  macro_auc={auc_b:.4f}{marker}")

print(f"\nBEST: alpha={best_alpha:.2f}  macro_auc={best_auc:.4f}")

# --- per-label breakdown di alpha terbaik ---
blend_best = (1 - best_alpha) * ours_rank + best_alpha * pk_rank
_, per_best = macro_auc(true_mat, blend_best)
print(f"\nPer-label AUC (kita | pilkwang | blend@{best_alpha:.2f}):")
for c, a1, a2, a3 in zip(LABEL_COLS, per_ours, per_pk, per_best):
    print(f"  {c:20s} {a1:.4f} | {a2:.4f} | {a3:.4f}")
