#!/usr/bin/env python3
"""Analisis blend ensemble PUBLIK (hasil panen gold-58) vs model KITA (v3) — semuanya offline.

Menjawab pertanyaan yang selama ini cuma ditebak (dan bikin 2 submission terbuang, §48/§50):
  1. Seberapa kuat ensemble publik per label, dan di mana ia lemah?
  2. Seberapa berkorelasi ia dgn v3 kita? (kalau ~1.0, tidak ada bobot yang bisa menolong)
  3. Berapa bobot blend GLOBAL optimal — dan apakah puncaknya benar-benar di atas publik sendirian?
  4. Apakah bobot PER-LABEL memberi lebih banyak, diukur JUJUR via nested-CV (bukan grid di data
     yang sama — jebakan yang sudah terbukti membalik kesimpulan di §29c)?
  5. Bootstrap: seberapa sering blend mengalahkan publik sendirian?

Input:
  --public  CSV prediksi ensemble publik utk 58 studi gold (output kernel harvest)
  --ours    CSV prediksi model kita (default: prediksi gold v3)
Pemakaian:
  python3 label_engineering/analyze_public_vs_ours.py --public <path> [--ours <path>]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def manual_auc(y, p):
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
    ap.add_argument('--public', required=True)
    ap.add_argument('--ours', default=None)
    args = ap.parse_args()

    train = pd.read_csv(REPO / 'reference_metadata/train.csv', dtype={'StudyInstanceUID': str})
    LAB = train.columns[2:14].tolist()
    gold = train[train[LAB].notna().all(axis=1)].set_index('StudyInstanceUID')

    pub = pd.read_csv(args.public, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ours_path = args.ours or (REPO / 'label_engineering/v3_gold_preds.csv')
    ours = pd.read_csv(ours_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')

    idx = gold.index.intersection(pub.index).intersection(ours.index)
    print(f'studi beririsan: {len(idx)} / gold {len(gold)}')
    assert len(idx) >= 50, 'terlalu sedikit studi beririsan'

    Y = gold.loc[idx, LAB].astype(float)
    P = pub.loc[idx, LAB].astype(float)
    O = ours.loc[idx, LAB].astype(float)

    def macro(df, rows=None):
        rows = idx if rows is None else rows
        return float(np.nanmean([manual_auc(Y.loc[rows, c].values, df.loc[rows, c].values)
                                 for c in LAB]))

    def rk(df):
        return df.rank(pct=True)

    print('\n=== 1. KEKUATAN PER LABEL ===')
    print(f"{'label':20s} {'publik':>8s} {'v3 kita':>8s} {'selisih':>9s}")
    print('-' * 50)
    gaps = {}
    for c in LAB:
        a = manual_auc(Y[c].values, P[c].values)
        b = manual_auc(Y[c].values, O[c].values)
        gaps[c] = b - a
        print(f'{c:20s} {a:8.4f} {b:8.4f} {b - a:+9.4f}')
    mp, mo = macro(P), macro(O)
    print('-' * 50)
    print(f"{'MACRO':20s} {mp:8.4f} {mo:8.4f} {mo - mp:+9.4f}")

    print('\n=== 2. KORELASI publik vs v3 (makin rendah = makin saling melengkapi) ===')
    cors = [np.corrcoef(P[c].values, O[c].values)[0, 1] for c in LAB]
    for c, r in sorted(zip(LAB, cors), key=lambda t: t[1]):
        print(f'  {c:20s} {r:.4f}')
    print(f'  RATA-RATA            {np.mean(cors):.4f}')

    print('\n=== 3. BLEND GLOBAL (rank-mean) ===')
    Pr, Or = rk(P), rk(O)
    best = (-1, None)
    for a in np.arange(0, 1.01, 0.05):
        m = macro((1 - a) * Pr + a * Or)
        if m > best[0]:
            best = (m, a)
        print(f'  alpha_kita={a:.2f} -> {m:.4f}')
    print(f'  TERBAIK alpha={best[1]:.2f} -> {best[0]:.4f} '
          f'(publik sendirian {mp:.4f}, selisih {best[0] - mp:+.4f})')

    print('\n=== 4. BOBOT PER-LABEL, nested-CV 5-fold (jujur, bukan grid di data yg sama) ===')
    rs = np.random.RandomState(2026)
    order = rs.permutation(len(idx))
    folds = [idx[order[i::5]] for i in range(5)]
    oof_per = pd.DataFrame(index=idx, columns=LAB, dtype=float)
    oof_glob = pd.DataFrame(index=idx, columns=LAB, dtype=float)
    for k in range(5):
        te = folds[k]
        tr = idx.difference(te)
        # bobot per-label dipilih HANYA dari fold latih
        for c in LAB:
            bb = (-1, 0.0)
            for a in np.arange(0, 1.01, 0.05):
                s = (1 - a) * Pr.loc[tr, c] + a * Or.loc[tr, c]
                v = manual_auc(Y.loc[tr, c].values, s.values)
                if not np.isnan(v) and v > bb[0]:
                    bb = (v, a)
            oof_per.loc[te, c] = ((1 - bb[1]) * Pr.loc[te, c] + bb[1] * Or.loc[te, c]).values
        # pembanding: bobot global tunggal, juga dipilih dari fold latih saja
        bg = (-1, 0.0)
        for a in np.arange(0, 1.01, 0.05):
            v = macro((1 - a) * Pr + a * Or, rows=tr)
            if v > bg[0]:
                bg = (v, a)
        oof_glob.loc[te] = ((1 - bg[1]) * Pr.loc[te] + bg[1] * Or.loc[te]).values

    m_per, m_glob = macro(oof_per), macro(oof_glob)
    print(f'  OOF bobot PER-LABEL : {m_per:.4f}')
    print(f'  OOF bobot GLOBAL    : {m_glob:.4f}')
    print(f'  publik sendirian    : {mp:.4f}')

    print('\n=== 5. BOOTSTRAP (2000 resample) ===')
    ids = np.array(idx)
    for name, cand in [('blend per-label', oof_per), ('blend global', oof_glob)]:
        wins = 0
        for _ in range(2000):
            s = pd.Index(ids[rs.choice(len(ids), len(ids), replace=True)]).unique()
            if macro(cand, rows=s) > macro(P, rows=s):
                wins += 1
        print(f'  P({name} > publik sendirian) = {wins / 2000:.3f}')

    print('\n=== KESIMPULAN ===')
    gain = max(m_per, m_glob) - mp
    if gain > 0.002:
        print(f'  Blend MENANG (+{gain:.4f} OOF) -> layak disubmit; pakai skema '
              f'{"per-label" if m_per >= m_glob else "global"}.')
    elif gain > 0:
        print(f'  Blend menang TIPIS (+{gain:.4f}) -> kemungkinan besar tak terbaca di LB '
              f'(3 desimal); jangan bakar submission utk ini.')
    else:
        print(f'  Blend TIDAK menang ({gain:+.4f}) -> model kita tersubsumsi ensemble publik; '
              f'butuh model yang BEDA JENIS, bukan bobot lain.')


if __name__ == '__main__':
    main()
