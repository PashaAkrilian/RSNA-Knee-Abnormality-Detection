#!/usr/bin/env python3
"""Fusi NOISY-STUDENT: label guru (laporan) + prediksi murid OUT-OF-FOLD -> label training baru.

Input  : oof_<init>_f{0..K-1}.csv dari kernel K-fold (tiap file: baris held-out fold ke-k utk
         studi lemah + 58 gold; kolom StudyInstanceUID, 12 label, is_gold, fold)
         + tabel guru `report_labels_blended_rawmean_v1.csv` (12 label + 12 __conf, 4.349 studi)
Output : fused_labels_r<R>.csv  -- format IDENTIK dgn tabel guru (kernel training bisa langsung
         memakainya hanya dgn mengganti dataset/nama berkas), label = (1-w)*guru + w*murid_OOF,
         __conf dipertahankan dari guru.

Fusi di ruang PROBABILITAS mentah, BUKAN rank -- pelajaran §46: rank memaksa tiap kolom jadi
distribusi uniform & merusak kalibrasi target BCE. Diuji di §32: konstan w=0,5-0,7 memberi gold-58
0,9330-0,9350; varian adaptif-konfidensi (0,9336-0,9342) tidak lebih baik -> pakai konstan.

Pemakaian:
  python3 label_engineering/fuse_noisy_student.py --oof-dir <dir> --teacher <csv> \
      [--init c5] [--folds 4] [--w 0.5] [--round 1] [--out <dir>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def auc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof-dir', required=True)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--init', default='c5')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--w', type=float, default=0.5, help='bobot murid (0=guru murni)')
    ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--out', default=str(REPO / 'label_engineering'))
    a = ap.parse_args()

    train = pd.read_csv(REPO / 'reference_metadata/train.csv', dtype={'StudyInstanceUID': str})
    LAB = train.columns[2:14].tolist()
    gold = train[train[LAB].notna().all(axis=1)].set_index('StudyInstanceUID')

    T = pd.read_csv(a.teacher, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert all(c in T.columns for c in LAB), 'tabel guru tak punya 12 kolom label'
    CONF = [f'{c}__conf' for c in LAB]

    parts = []
    for k in range(a.folds):
        f = Path(a.oof_dir) / f'oof_{a.init}_f{k}.csv'
        assert f.is_file(), f'OOF fold {k} belum ada: {f}'
        parts.append(pd.read_csv(f, dtype={'StudyInstanceUID': str}))
    oof = pd.concat(parts, ignore_index=True)

    weak = oof[oof['is_gold'] == 0]
    assert weak['StudyInstanceUID'].is_unique, 'ada studi lemah yang muncul di >1 fold held-out'
    S = weak.set_index('StudyInstanceUID')[LAB]
    missing = T.index.difference(S.index)
    extra = S.index.difference(T.index)
    print(f'guru: {len(T)} studi | murid OOF: {len(S)} studi | kurang: {len(missing)} | lebih: {len(extra)}')
    assert len(missing) == 0, f'{len(missing)} studi guru tak punya prediksi OOF -> fold belum lengkap'

    # ---- diagnostik independensi guru~murid pada studi lemah ----
    Tw = T.loc[S.index, LAB].astype(float); Sw = S.astype(float)
    cors = [np.corrcoef(Tw[c], Sw[c])[0, 1] for c in LAB]
    print('\nkorelasi guru~murid_OOF per label (studi lemah; makin rendah = makin independen):')
    for c, r in zip(LAB, cors):
        print(f'  {c:20s} {r:.3f}')
    print(f'  RATA-RATA            {np.mean(cors):.3f}')

    # ---- sanity di gold: prediksi gold dirata-rata lintas fold (semuanya bersih) ----
    g = oof[oof['is_gold'] == 1].groupby('StudyInstanceUID')[LAB].mean()
    idx = gold.index.intersection(g.index)
    Yg = gold.loc[idx, LAB].astype(float)
    mg = np.nanmean([auc(Yg[c].values, g.loc[idx, c].values) for c in LAB])
    print(f'\nmurid (rata2 {a.folds} fold) di gold-58: macro AUC = {mg:.4f}   [v3 = 0.9234]')

    # ---- fusi ----
    w = a.w
    F_ = (1 - w) * Tw + w * Sw
    out = T.copy()
    out.loc[F_.index, LAB] = F_.values
    lo, hi = float(np.nanmin(F_.values)), float(np.nanmax(F_.values))
    assert np.isfinite(out[LAB].values).all() and 0.0 <= lo and hi <= 1.0, 'label fusi di luar [0,1]'
    print(f'\nfusi w_murid={w}: rentang label [{lo:.3f}, {hi:.3f}]')
    print('rata-rata label (guru -> fusi):')
    for c in LAB:
        print(f'  {c:20s} {Tw[c].mean():.3f} -> {F_[c].mean():.3f}')

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f'fused_labels_r{a.round}.csv'
    out.reset_index().to_csv(fn, index=False)
    print(f'\nditulis: {fn}  ({len(out)} baris, {len(out.columns)} kolom)')
    assert all(c in out.columns for c in CONF), 'kolom __conf hilang'


if __name__ == '__main__':
    main()
