#!/usr/bin/env python3
"""Simulasi peluang PERAK (top 5 %) di papan privat dari papan publik (model sederhana & transparan).

Model: privat_i = pub_i - s*(pub_i - MU) + N(0, sig)   (s = regresi ke rata-rata, sig = selisih privat-publik)
Kita dimodelkan sama, dgn skill sejati +delta. Hasil = pecahan percobaan di mana skor privat kita >= batas top-5 % (SILVER).
Catatan: dataran 0,950 berisi ratusan SALINAN notebook yg sama (lonjakan densitas), jadi koreksi Tweedie/empirical-Bayes
tidak valid; regresi linear + noise adalah pendekatan paling jujur. Angka = ARAH, bukan ramalan.

Pemakaian:
  kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p /tmp/lb && unzip -o /tmp/lb/*.zip -d /tmp/lb
  python3 silver_prob_sim.py /tmp/lb/rsna-knee-abnormality-detection-publicleaderboard-*.csv
"""
import csv
import sys

import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
pub = np.array([float(r["Score"]) for r in rows])
n = len(pub)
SILVER = int(round(n * 0.05))
MU = 0.945  # rata-rata tim "serius" (titik regresi)
rng = np.random.default_rng(3)


def p_silver(ours_pub, delta, s, sig, trials=2000):
    base = pub - s * (pub - MU)
    mo = ours_pub - s * (ours_pub - MU) + delta
    wins, cuts = 0, []
    for _ in range(trials):
        priv = base + sig * rng.standard_normal(n)
        cut = np.partition(priv, -SILVER)[-SILVER]
        cuts.append(cut)
        wins += (mo + sig * rng.standard_normal()) >= cut
    return wins / trials, float(np.mean(cuts))


print("tim", n, "| top-5 % =", SILVER)
print("\nA) tim kita (publik 0,950) + perbaikan sejati delta")
print("  s    sig   corte_priv   +0.000 +0.001 +0.002 +0.003 +0.004 +0.006 +0.008")
for s in (0.3, 0.5, 0.7):
    for sig in (0.003, 0.004, 0.005):
        res = [p_silver(0.950, d, s, sig) for d in (0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008)]
        print(f" {s:.1f}  {sig:.3f}   {res[0][1]:.4f}    " + "  ".join(f"{r[0]:5.2f}" for r in res))
print("\nB) tim dgn skor publik X (mis. merger) -> P(perak)  [s=0,5/sig=0,004 | s=0,3/sig=0,003 | s=0,7/sig=0,005]")
for x in (0.950, 0.952, 0.954, 0.956, 0.958, 0.960):
    print(f"  {x:.3f}  {p_silver(x, 0.0, 0.5, 0.004)[0]:.2f} | {p_silver(x, 0.0, 0.3, 0.003)[0]:.2f} | {p_silver(x, 0.0, 0.7, 0.005)[0]:.2f}")
