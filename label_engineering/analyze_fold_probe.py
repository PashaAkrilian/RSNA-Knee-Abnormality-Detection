#!/usr/bin/env python3
"""Analisis PROBE kontaminasi dari kernel K-fold: apakah checkpoint c5 (init) sudah "menghafal" label
studi lemah saat pra-latih oleh penulis aslinya?

Logika: c5 dilatih penulisnya pada SEMUA studi lemah dgn label mirip guru kita, tapi TIDAK pada gold-58.
Kalau c5 menghafal, prediksinya pada studi lemah ("dilihat") akan jauh lebih selaras dgn label guru
dibanding pada gold ("tak pernah dilihat") -- untuk label guru yg kualitas/noise-nya sama.
Keduanya diukur terhadap GURU YANG SAMA FORMULANYA:
  - studi lemah : tabel raw-mean flight0234(0,7)+Claude(0,3) -> kolom 12 label di dataset label
  - gold        : 0,7*flight0234 + 0,3*Claude-pilot (formula identik, tabel tersedia utk gold)
Metrik: (a) korelasi Pearson rata2 12 label, (b) AUC terhadap label guru yg dibiner (>0,5).

Ambang keputusan (ditetapkan SEBELUM melihat hasil):
  selisih korelasi (dilihat - tak dilihat) <= 0,05  -> kontaminasi RINGAN, init c5 aman dipakai
  0,05 .. 0,12                                      -> SEDANG, pakai w_murid lebih kecil (0,4)
  > 0,12                                            -> BERAT, ganti init ke ImageNet bersih

Pemakaian: python3 label_engineering/analyze_fold_probe.py <dir berisi probe_init_*_f<k>.csv> [fold=0]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
d = Path(sys.argv[1]); fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0

tr = pd.read_csv(REPO / 'reference_metadata/train.csv', dtype={'StudyInstanceUID': str})
LAB = tr.columns[2:14].tolist()
gold = tr[tr[LAB].notna().all(axis=1)].set_index('StudyInstanceUID')


def auc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = ~np.isnan(y) & ~np.isnan(p); y, p = y[m], p[m]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cors(P, T):
    return np.array([np.corrcoef(P[c], T[c])[0, 1] for c in LAB])


def aucs(P, T):
    return np.array([auc((T[c] > 0.5).astype(int), P[c]) for c in LAB])


# guru studi lemah: dataset label training
T_weak = pd.read_csv(REPO / 'label_engineering/blended_labels_fullscale_train_rawmean.csv',
                     dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LAB]
# guru gold: formula identik (0,7 flight + 0,3 claude-pilot). flight dari dataset publik.
import subprocess, os, tempfile
fl_path = Path(os.environ.get('FLIGHT_CSV', ''))
if not fl_path.is_file():
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(['kaggle', 'datasets', 'download', 'flight0234/rsna-knee-hybrid-report-labels',
                    '--unzip', '-p', str(tmp)], capture_output=True,
                   env={**os.environ, 'PATH': os.environ['PATH'] + ':' + os.path.expanduser('~/.local/bin')})
    fl_path = tmp / 'report_labels_v4hybrid.csv'
flight = pd.read_csv(fl_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LAB] \
    .apply(pd.to_numeric, errors='coerce')
claude = pd.read_csv(REPO / 'label_engineering/claude_labels_pilot_58gold.csv',
                     dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LAB] \
    .apply(pd.to_numeric, errors='coerce')
idx_g = gold.index.intersection(flight.index).intersection(claude.index)
T_gold = (0.7 * flight.loc[idx_g] + 0.3 * claude.loc[idx_g])

pw = pd.read_csv(d / f'probe_init_weak_f{fold}.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
pg = pd.read_csv(d / f'probe_init_gold_f{fold}.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
iw = pw.index.intersection(T_weak.index); ig = pg.index.intersection(T_gold.index)
print(f'probe: {len(iw)} studi lemah (DILIHAT c5) | {len(ig)} gold (TAK pernah dilihat)\n')

c_seen = cors(pw.loc[iw, LAB], T_weak.loc[iw]); c_unseen = cors(pg.loc[ig, LAB], T_gold.loc[ig])
a_seen = aucs(pw.loc[iw, LAB], T_weak.loc[iw]); a_unseen = aucs(pg.loc[ig, LAB], T_gold.loc[ig])
print(f"{'label':18s} {'corr dilihat':>12s} {'corr tak-dilihat':>17s} {'selisih':>8s} | {'AUC dilihat':>11s} {'AUC tak-dilihat':>16s}")
for i, c in enumerate(LAB):
    print(f"{c:18s} {c_seen[i]:12.3f} {c_unseen[i]:17.3f} {c_seen[i]-c_unseen[i]:+8.3f} | "
          f"{a_seen[i]:11.3f} {a_unseen[i]:16.3f}")
dc = np.nanmean(c_seen) - np.nanmean(c_unseen)
da = np.nanmean(a_seen) - np.nanmean(a_unseen)
print(f"\nRATA-RATA corr: dilihat={np.nanmean(c_seen):.3f}  tak-dilihat={np.nanmean(c_unseen):.3f}  selisih={dc:+.3f}")
print(f"RATA-RATA AUC : dilihat={np.nanmean(a_seen):.3f}  tak-dilihat={np.nanmean(a_unseen):.3f}  selisih={da:+.3f}")

# uji signifikansi kasar: bootstrap selisih korelasi rata2 (resample studi)
rs = np.random.RandomState(0)
def boot(Pw, Tw, Pg, Tg, n=400):
    out = []
    for _ in range(n):
        a = rs.choice(len(Pw), len(Pw)); b = rs.choice(len(Pg), len(Pg))
        out.append(np.nanmean(cors(Pw.iloc[a], Tw.iloc[a])) - np.nanmean(cors(Pg.iloc[b], Tg.iloc[b])))
    return np.percentile(out, [5, 95])
lo, hi = boot(pw.loc[iw, LAB], T_weak.loc[iw], pg.loc[ig, LAB], T_gold.loc[ig])
print(f"bootstrap 90% CI selisih corr: [{lo:+.3f}, {hi:+.3f}]")

print('\nKEPUTUSAN (ambang ditetapkan di awal):')
if dc <= 0.05:
    print('  RINGAN  -> init c5 aman utk semua fold; fusi w_murid=0,5')
elif dc <= 0.12:
    print('  SEDANG  -> init c5 boleh, tapi kurangi w_murid ke 0,4')
else:
    print('  BERAT   -> ganti init ke ImageNet bersih (dataset mohamed3abdelrazik/coatnet-...-in12k-ft-in1k)')
