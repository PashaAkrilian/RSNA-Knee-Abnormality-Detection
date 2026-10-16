#!/usr/bin/env python3
"""Label murid putaran-2: L1 = (1-w) * konsensus + w * OOF-R7 per temuan (STRATEGI_0960 §0.1/§3).

Masukan (lokal):
  data/labels/labels_student_v0.csv     v0 = 0,5 konsensus + 0,5 Raptor in-sample (+ fold, is_gold, gold_*)
  data/teacher_r384_train_preds.csv     Raptor r384 in-sample -> konsensus = 2*v0 - teacher (persis, terklip 0,01..0,99)
  data/r7oof/compact_oof_f{0..3}.csv    keluaran kernel guru OOF R7 (StudyInstanceUID, 12 temuan, scored, fold)
  data/r7oof/compact_gold.csv           (opsional) R7 4 model pada 58 gold -> ukuran mutu guru
  data/student_folds/oof_fold{k}.csv    (opsional) OOF murid r1 (logit) -> pembanding
Keluaran: data/labels/labels_student_v1.csv (skema = v0; baris tanpa OOF-R7 terskor tetap v0).

Pakai: python3 make_labels_v1.py cons=1 out=data/labels/labels_student_v0c.csv   # konsensus saja (murid r2c, tanpa guru)
       python3 make_labels_v1.py [w=0.5] [calib=none|quantile] [oof_dir=data/r7oof] [out=data/labels/labels_student_v1.csv]
       python3 make_labels_v1.py w=0.5 oof_dir2=data/r6oof out=data/labels/labels_student_v3.csv   # 17 Okt: guru = rerata OOF-R7 & OOF-R6+flip
  calib=quantile: OOF-R7 dipetakan ke sebaran konsensus per temuan (urutan R7 tetap) — dipakai bila keluaran R7 bukan probabilitas.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def roc_auc_score(y, p):                                   # Mann–Whitney (lokal tanpa sklearn), sama dgn manual_auc label_engineering
    r = pd.Series(np.asarray(p, dtype=float)).rank().values
    y = np.asarray(y) == 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
W = float(args.get("w", "0.5"))
CALIB = args.get("calib", "none")
OOF_DIR = Path(args.get("oof_dir", ROOT / "data/r7oof"))
OUT = Path(args.get("out", ROOT / "data/labels/labels_student_v1.csv"))
OOF_DIR2 = Path(args["oof_dir2"]) if "oof_dir2" in args else None   # guru ke-2 (R6+flip): guru = rerata keduanya pd baris yg punya keduanya
assert 0.0 <= W <= 1.0 and CALIB in ("none", "quantile")
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']


def macro(y, p):
    a = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a))


v0 = pd.read_csv(ROOT / "data/labels/labels_student_v0.csv", index_col=0)
v0.index = v0.index.astype(str)
t = pd.read_csv(ROOT / "data/teacher_r384_train_preds.csv")
t = t.set_index(t.columns[0])
t.index = t.index.astype(str)
t = t.loc[v0.index, L]
cons = (2 * v0[L] - t).clip(0.01, 0.99)
assert np.allclose(0.5 * cons + 0.5 * t, v0[L], atol=0.011), "v0 bukan 0,5 konsensus + 0,5 Raptor"
if args.get("cons") == "1":                               # murid r2c: label laporan saja -> beragam dari Raptor & R7
    v1 = v0.copy()
    ng = v0.index[v0.is_gold == 0]
    v1.loc[ng, L] = cons.loc[ng, L].values
    v1["teacher_mean"] = np.nan
    assert np.array_equal(v1.loc[v0.is_gold == 1, L].values, v0.loc[v0.is_gold == 1, L].values) and (v1.fold == v0.fold).all()
    v1.to_csv(OUT)
    print(f"ditulis {OUT} | konsensus saja utk {len(ng)} non-gold")
    sys.exit(0)

oof = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in sorted(glob.glob(str(OOF_DIR / "compact_oof_f*.csv")))])
oof = oof[oof["scored"] == 1].drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
gold_rows = v0.index[v0.is_gold == 1]
assert not set(oof.index) & set(gold_rows), "OOF-R7 memuat studi gold"
rows = v0.index[(v0.is_gold == 0) & v0.index.isin(oof.index)]
print(f"OOF-R7 terskor: {len(oof)} studi | dipakai (non-gold di v0): {len(rows)} / {int((v0.is_gold == 0).sum())} | "
      f"rentang {oof[L].values.min():.3f}..{oof[L].values.max():.3f}, rerata {oof[L].values.mean():.3f}")
r7 = oof.loc[rows, L].astype(float)
if OOF_DIR2 is not None:
    o2 = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in sorted(glob.glob(str(OOF_DIR2 / "compact_oof_f*.csv")))])
    o2 = o2[o2["scored"] == 1].drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    assert not set(o2.index) & set(gold_rows), "OOF guru-2 memuat studi gold"
    both = rows[rows.isin(o2.index)]
    yb = (cons.loc[both, L].values > 0.5).astype(int)
    a7, a6 = macro(yb, r7.loc[both].values), macro(yb, o2.loc[both, L].values.astype(float))
    print(f"guru-2 {OOF_DIR2}: {len(both)} baris | AUC vs konsensus>0,5: OOF-R7 {a7:.4f} | OOF-guru2 {a6:.4f}")
    assert a6 - a7 < 0.03, "guru-2 jauh di atas R7 vs label laporan -> curiga bukan OOF (bocor); periksa fungsi fold"
    r7.loc[both] = 0.5 * r7.loc[both].values + 0.5 * o2.loc[both, L].values.astype(float)
if CALIB == "quantile":                                    # urutan R7, sebaran nilai = konsensus (per temuan)
    for c in L:
        q = np.sort(cons.loc[rows, c].values)
        r7[c] = q[(r7[c].rank(method="first").values - 1).astype(int)]

v1 = v0.copy()
v1.loc[rows, L] = ((1 - W) * cons.loc[rows, L] + W * r7).clip(0.0, 1.0).values
v1.loc[rows, "teacher_mean"] = r7.mean(axis=1).values
assert len(v1) == len(v0) and v1.index.equals(v0.index) and (v1.fold == v0.fold).all()
assert np.array_equal(v1.loc[gold_rows, L].values, v0.loc[gold_rows, L].values), "baris gold berubah"
assert ((v1[L] >= 0) & (v1[L] <= 1)).all().all()
v1.to_csv(OUT)
print(f"ditulis {OUT} | w={W} calib={CALIB} | baris diubah {len(rows)}")

# ---- laporan mutu sumber (validator lemah: konsensus>0,5 pada non-gold; emas-58 hanya pemeriksa, SE ±0,01) ----
yw = (cons.loc[rows, L].values > 0.5).astype(int)
print("AUC vs konsensus>0,5 (non-gold %d): OOF-R7 %.4f | Raptor in-sample %.4f | v1 %.4f" % (
    len(rows), macro(yw, r7.values), macro(yw, t.loc[rows].values), macro(yw, v1.loc[rows, L].values)))
stu = [pd.read_csv(f, index_col=0) for f in sorted(glob.glob(str(ROOT / "data/student_folds/oof_fold*.csv")))]
if stu:
    sv = pd.concat([s[s["set"] == "val"] for s in stu])[L]
    sv.index = sv.index.astype(str)
    both = sv.index.intersection(rows)
    yb = (cons.loc[both, L].values > 0.5).astype(int)
    print("  irisan murid-r1 val (%d): murid r1 %.4f | OOF-R7 %.4f" % (len(both), macro(yb, sv.loc[both].values), macro(yb, r7.loc[both].values)))
    print("  Spearman rerata per temuan: R7~murid %.3f | R7~Raptor %.3f | murid~Raptor %.3f" % tuple(
        np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(12)])
        for a, b in ((r7.loc[both].values, sv.loc[both].values), (r7.loc[both].values, t.loc[both].values),
                     (sv.loc[both].values, t.loc[both].values))))
yg = v0.loc[gold_rows, ["gold_" + c for c in L]].values
print("emas-58: konsensus %.4f | Raptor %.4f" % (macro(yg, cons.loc[gold_rows].values), macro(yg, t.loc[gold_rows].values)), end="")
if stu:
    sg = pd.concat([s[s["set"] == "gold"][L] for s in stu]).groupby(level=0).mean()
    sg.index = sg.index.astype(str)
    print(" | murid r1 (rerata fold) %.4f" % macro(yg, sg.loc[gold_rows].values), end="")
gp = OOF_DIR / "compact_gold.csv"
if gp.exists():
    g = pd.read_csv(gp, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    print(" | R7 compact %.4f" % macro(yg, g.loc[gold_rows, L].values), end="")
print()
