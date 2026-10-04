#!/usr/bin/env python3
"""Fusi NOISY-STUDENT multi-sumber: label guru (laporan) + prediksi murid OUT-OF-FOLD dari BEBERAPA keluarga model.

Generalisasi fuse_noisy_student.py (satu murid) utk putaran-2: murid CoAtNet (4 fold, putaran-1) + murid CNN murah
(6 fold) + ... Semua OOF tiap sumber BERSIH: tiap studi lemah diprediksi oleh model yg tak pernah melihat label-nya;
gold-58 tak pernah ikut latih di sumber manapun, jadi AUC label-fusi terhadap gold adalah ukuran kualitas label yg jujur
(N=58 -> SE ~0,01; pakai utk memilih bobot kasar, bukan utk menyetel halus).

Fusi di ruang PROBABILITAS mentah (bukan rank; pelajaran §46):  label = (1-w) * guru + w * rata2_berbobot(murid_i)
Sumber murid ditulis `--src nama=dir:pola:folds:bobot` (pola berkas OOF dgn {k} utk nomor fold), contoh
  --src coat=/tmp/oofall:oof_c5_f{k}.csv:4:1.0  --src r50=/tmp/oofr50:oof_cheap_resnet50_s1_f{k}.csv:6:1.0

Pemakaian:
  python3 label_engineering/fuse_multi_student.py --teacher <csv> --src ... [--src ...] [--w 0.5] [--round 2] [--out DIR]
         [--sweep]   # cetak tabel AUC-gold utk beberapa w & komposisi, tanpa menulis berkas
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


def load_src(spec, LAB):
    name, rest = spec.split('=', 1)
    d, pat, nf, wt = rest.rsplit(':', 3)
    parts = []
    for k in range(int(nf)):
        f = Path(d) / pat.format(k=k)
        assert f.is_file(), f'OOF tak ada: {f}'
        parts.append(pd.read_csv(f, dtype={'StudyInstanceUID': str}))
    oof = pd.concat(parts, ignore_index=True)
    weak = oof[oof['is_gold'] == 0]
    assert weak['StudyInstanceUID'].is_unique, f'{name}: studi lemah muncul di >1 fold held-out'
    gold = oof[oof['is_gold'] == 1].groupby('StudyInstanceUID')[LAB].mean()
    return name, weak.set_index('StudyInstanceUID')[LAB].astype(float), gold.astype(float), float(wt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--src', action='append', required=True)
    ap.add_argument('--w', type=float, default=0.5, help='bobot total murid (0 = guru murni)')
    ap.add_argument('--round', type=int, default=2)
    ap.add_argument('--out', default=str(REPO / 'label_engineering'))
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--flight', default='', help='CSV flight0234 (report_labels_v4hybrid.csv) utk guru-gold; kosong = unduh via kaggle')
    a = ap.parse_args()

    train = pd.read_csv(REPO / 'reference_metadata/train.csv', dtype={'StudyInstanceUID': str})
    LAB = train.columns[2:14].tolist()
    gold = train[train[LAB].notna().all(axis=1)].set_index('StudyInstanceUID')
    Yg = gold[LAB].astype(float)
    T = pd.read_csv(a.teacher, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    CONF = [f'{c}__conf' for c in LAB]
    srcs = [load_src(s, LAB) for s in a.src]
    idx = srcs[0][1].index
    for n, S, _, _ in srcs:
        assert set(S.index) == set(idx), f'{n}: himpunan studi lemah berbeda'
    assert len(T.index.difference(idx)) == 0, 'ada studi guru tanpa OOF'
    ids = list(idx)
    Tw = T.loc[ids, LAB].astype(float)

    def macro_gold(P_gold):  # P_gold: DataFrame indexed by gold UID
        g = Yg.index.intersection(P_gold.index)
        return float(np.nanmean([auc(Yg.loc[g, c].values, P_gold.loc[g, c].values) for c in LAB]))

    print(f"{'sumber':12s} gold-58(rata2 fold)  korelasi(guru, OOF) pada studi lemah")
    for n, S, G, w in srcs:
        cs = np.nanmean([np.corrcoef(Tw[c], S.loc[ids, c])[0, 1] for c in LAB])
        print(f'{n:12s} {macro_gold(G):.4f}             {cs:.3f}')
    # prediksi murid gabungan berbobot (studi lemah)
    wts = np.array([w for *_, w in srcs]); wts = wts / wts.sum()
    Sm = sum(wi * S.loc[ids, LAB] for wi, (_, S, _, _) in zip(wts, srcs))
    Gm = sum(wi * G.reindex(Yg.index) for wi, (_, _, G, _) in zip(wts, srcs))
    print(f'\nmurid GABUNGAN (bobot {dict(zip([s[0] for s in srcs], wts.round(3)))}) gold-58 = {macro_gold(Gm):.4f}')

    # ---- kualitas label di gold-58: guru-gold = 0,7*flight0234 + 0,3*Claude-pilot (formula identik dgn guru studi lemah) ----
    import os, subprocess, tempfile
    fl_path = Path(a.flight) if a.flight else Path('')
    if not fl_path.is_file():
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(['kaggle', 'datasets', 'download', 'flight0234/rsna-knee-hybrid-report-labels', '--unzip', '-p', str(tmp)],
                       capture_output=True, env={**os.environ, 'PATH': os.environ['PATH'] + ':' + os.path.expanduser('~/.local/bin')})
        fl_path = tmp / 'report_labels_v4hybrid.csv'
    flight = pd.read_csv(fl_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LAB].apply(pd.to_numeric, errors='coerce')
    claude = pd.read_csv(REPO / 'label_engineering/claude_labels_pilot_58gold.csv', dtype={'StudyInstanceUID': str}) \
        .set_index('StudyInstanceUID')[LAB].apply(pd.to_numeric, errors='coerce')
    ig = Yg.index.intersection(flight.index).intersection(claude.index).intersection(Gm.index)
    Tg = 0.7 * flight.loc[ig] + 0.3 * claude.loc[ig]
    print(f'\nKUALITAS LABEL terhadap gold-58 (n={len(ig)}): guru-gold = {macro_gold(Tg):.4f}')
    print('   w_murid -> AUC(fusi_gold = (1-w)*guru + w*murid_gabungan)')
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        print(f'   {w:.1f}  {macro_gold((1 - w) * Tg + w * Gm.loc[ig]):.4f}')
    rows = []
    for w in ([0.0, 0.3, 0.5, 0.7] if a.sweep else [a.w]):
        F_ = (1 - w) * Tw + w * Sm
        rows.append((w, float(np.nanmin(F_.values)), float(np.nanmax(F_.values)),
                     float(np.nanmean([np.corrcoef(F_[c], Tw[c])[0, 1] for c in LAB]))))
    print('\nw     min    max    korelasi(fusi, guru)')
    for r in rows:
        print(f'{r[0]:.2f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}')
    if a.sweep:
        return
    w = a.w
    F_ = (1 - w) * Tw + w * Sm
    out = T.copy()
    out.loc[F_.index, LAB] = F_.values
    lo, hi = float(np.nanmin(F_.values)), float(np.nanmax(F_.values))
    assert np.isfinite(out[LAB].values).all() and 0.0 <= lo and hi <= 1.0, 'label fusi di luar [0,1]'
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f'fused_labels_r{a.round}.csv'
    out.reset_index().to_csv(fn, index=False)
    assert all(c in out.columns for c in CONF), 'kolom __conf hilang'
    print(f'\nditulis: {fn} ({len(out)} baris) | w_murid={w}')


if __name__ == '__main__':
    main()
