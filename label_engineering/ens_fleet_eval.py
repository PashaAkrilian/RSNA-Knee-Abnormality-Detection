#!/usr/bin/env python3
"""Sumbangan armada model kita terhadap cabang Raptor STOK di gold-58 (bersih): kurva bobot alpha + korelasi rank.

Cabang stok = 4 arm publik (c5 .6 / v10 .1 / c5-reverse .1 / v8 .2) dari panen gold (`raptor_raw.npz`, K=94 window) -- BUKAN
seluruh pipeline publik (gold-58 terkontaminasi utk pipeline penuh, §33). Alpha optimal utk pipeline penuh lebih kecil dari
utk cabang saja (anggota lain sudah menutup sebagian keragaman), jadi pakai kurva ini utk ARAH dan URUTAN, bukan angka bobot final.

Pemakaian:
  python3 label_engineering/ens_fleet_eval.py --raw <raptor_raw.npz> \
      --fleet coat4=<dir>:oof_c5_f{k}.csv:4  --fleet rad6=<dir>:oof_cheap_radresnet50_s1_f{k}.csv:6 [--fleet ...]
Setiap --fleet = nama=dir:pola:jumlah_fold ; OOF gold dirata-rata lintas fold (rank-mean).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--fleet', action='append', required=True)
    a = ap.parse_args()
    tr = pd.read_csv(REPO / 'reference_metadata/train.csv', dtype={'StudyInstanceUID': str})
    LAB = tr.columns[2:14].tolist()
    gold = tr[tr[LAB].notna().all(axis=1)].set_index('StudyInstanceUID'); idx = list(gold.index)
    Y = gold.loc[idx, LAB].values.astype(int)
    M = lambda P: float(np.nanmean([auc(Y[:, j], P[:, j]) for j in range(12)]))
    rk = lambda P: np.column_stack([pd.Series(P[:, j]).rank(pct=True).values for j in range(12)])
    z = np.load(a.raw); u = [str(x) for x in z['study_uids']]; pos = [u.index(i) for i in idx]
    R = z['raw_probabilities'][:, pos, :]
    BR = np.tensordot(np.array([.6, .1, .1, .2]), R, axes=(0, 0))
    print(f'cabang Raptor stok (gold-58): {M(BR):.4f}\n')
    F = {}
    for spec in a.fleet:
        name, rest = spec.split('=', 1)
        d, pat, nf = rest.rsplit(':', 2)
        gs = []
        for k in range(int(nf)):
            o = pd.read_csv(Path(d) / pat.format(k=k), dtype={'StudyInstanceUID': str})
            gs.append(o[o.is_gold == 1].set_index('StudyInstanceUID').loc[idx, LAB].values)
        singles = [M(g) for g in gs]
        F[name] = np.mean([rk(g) for g in gs], 0)
        cor = np.mean([np.corrcoef(rk(BR)[:, j], F[name][:, j])[0, 1] for j in range(12)])
        print(f'{name:10s} {nf} model: tunggal rata2={np.mean(singles):.4f} (min {min(singles):.4f} max {max(singles):.4f}) | '
              f'ensemble={M(F[name]):.4f} | korelasi rank thd cabang stok={cor:.3f}')
    print('\nblend (1-a)*stok + a*armada  [rank-mean]')
    al = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    print(f"{'armada':26s} " + ' '.join(f'a={x:.1f}  ' for x in al))
    names = list(F)
    combos = [[n] for n in names] + ([names] if len(names) > 1 else [])
    for c in combos:
        ex = np.mean([F[n] for n in c], 0)
        print(f"{'+'.join(c):26s} " + ' '.join(f'{M((1 - x) * rk(BR) + x * ex):.4f} ' for x in al))


if __name__ == '__main__':
    main()
