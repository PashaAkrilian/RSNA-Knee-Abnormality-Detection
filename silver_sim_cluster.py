#!/usr/bin/env python3
"""Simulasi PERAK sadar-klaster (15 Okt): papan privat saat blok publik 0,956–0,957 berisi SALINAN resep.

Model (semua angka relatif thd keluarga "R7 dasar" di privat; arah, bukan ramalan):
- G1 = tim publik >=0,958 (tumbuh ke N_G1 di akhir): privat = keunggulan publik dikerutkan (s) + derau independen sig_ind.
- Klaster = keluarga resep publik (ukuran, delta_f): SATU skor privat per keluarga (salinan byte-identik) = delta_f + e_f,
  e_f ~ N(0, sig_fam) = cara split privat memperlakukan galat khas resep itu; + keluarga kecil (variasi sendiri) acak.
- G0 = tim publik <=0,955: privat = (pub - 0,9563) + N(0, sig_g0); sig_g0 kecil krn mayoritas varian keluarga yg sama.
- Kita = keluarga unik: delta_kita + e_kita. Rank privat = jumlah tim di atas kita + 1; perak bila rank <= 5 % x N.
Pakai: python3 silver_sim_cluster.py   (opsional: python3 silver_sim_cluster.py <csv papan publik> utk G0 dari papan nyata)
"""
import csv
import sys

import numpy as np

rng = np.random.default_rng(7)
N_END = 6150
K = int(round(0.05 * N_END))
# keluarga resep publik di klaster (proyeksi 22 Okt): (nama, ukuran, delta privat sejati thd R7 dasar)
FAM = [("R7 dasar", 380, 0.0000), ("R7+r224 (heliosli 0,9573)", 520, 0.0002), ("R7+r224 cermin (varshitha)", 230, 0.0003),
       ("R6+flip+r224 (heliosli 0,9576)", 180, 0.0003), ("r224 cermin + S6F (leoprovorov)", 140, 0.0006)]
N_SMALL, SMALL_MU, SMALL_SD = 150, 0.0002, 0.0006           # variasi kecil sendiri-sendiri (1 tim/keluarga)
CLUSTER_PUB = 0.9563

if len(sys.argv) > 1:
    pub = np.array([float(r["Score"]) for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig"))])
    g0_pub = pub[pub <= 0.955]
else:
    g0_pub = np.clip(rng.normal(0.935, 0.025, 4600), 0.5, 0.955)


def p_silver(delta, n_g1=250, g1_edge=0.0025, s=0.5, sig_ind=0.002, sig_fam=0.0005, sig_g0=0.001, trials=3000):
    wins = 0
    sizes = np.array([f[1] for f in FAM]); dfam = np.array([f[2] for f in FAM])
    for _ in range(trials):
        ours = delta + sig_fam * rng.standard_normal()
        fam = dfam + sig_fam * rng.standard_normal(len(FAM))
        above = int(sizes[fam > ours].sum())
        small = SMALL_MU + SMALL_SD * rng.standard_normal(N_SMALL) + sig_fam * rng.standard_normal(N_SMALL)
        above += int((small > ours).sum())
        g1 = (1 - s) * rng.exponential(g1_edge, n_g1) + 0.0010 * (1 - s) + sig_ind * rng.standard_normal(n_g1)
        above += int((g1 > ours).sum())
        g0 = (g0_pub - CLUSTER_PUB) + sig_g0 * rng.standard_normal(len(g0_pub))
        above += int((g0 > ours).sum())
        wins += (above + 1) <= K
    return wins / trials


if __name__ == "__main__":
    print(f"N akhir {N_END} | perak = rank <= {K} | klaster {sum(f[1] for f in FAM) + N_SMALL} tim | G0 {len(g0_pub)} tim")
    deltas = (0.0, 0.0003, 0.0006, 0.0010, 0.0015, 0.0020)
    print("skenario (n_G1, sig_fam, sig_ind)        " + "  ".join(f"d={d:.4f}" for d in deltas))
    for n_g1 in (200, 250, 300):
        for sig_fam in (0.0003, 0.0006):
            for sig_ind in (0.0015, 0.0025):
                ps = [p_silver(d, n_g1=n_g1, sig_fam=sig_fam, sig_ind=sig_ind) for d in deltas]
                print(f"  G1={n_g1:3d} sig_fam={sig_fam:.4f} sig_ind={sig_ind:.4f}  " + "  ".join(f"  {p:5.2f} " for p in ps))
