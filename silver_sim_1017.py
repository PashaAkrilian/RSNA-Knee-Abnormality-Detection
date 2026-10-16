#!/usr/bin/env python3
"""Simulasi perak 17 Okt (lanjutan silver_sim_cluster.py): blok publik 0,958 = klaster salinan R6+flip; delta privat relatif thd klaster goodpjw.
Pakai: python3 silver_sim_1017.py <folder berisi csv papan publik (kaggle competitions leaderboard -d)>"""
import numpy as np, pandas as pd, glob, sys
rng = np.random.default_rng(11)
lb = pd.read_csv(glob.glob(sys.argv[1] + '/*.csv')[0])
pub = lb.Score.values
N_END = 6300; K = int(0.05 * N_END)
g0 = pub[pub <= 0.956]                                   # tim di bawah klaster
def p_silver(delta, n_g1=200, g1_edge=0.002, s=0.5, sig_ind=0.002, sig_fam=0.0005, q=0.4, in_cluster=False, trials=4000):
    # keluarga (ukuran akhir, delta sejati privat vs goodpjw): goodpjw 450, heliosli-blend 200, R6+flip tanpa wc 150, R7 0,957 3 keluarga
    FAM = [(450, 0.0), (200, 0.0), (150, -0.0004), (400, -0.0005), (300, -0.0006), (300, -0.0008)]
    sizes = np.array([f[0] for f in FAM]); dfam = np.array([f[1] for f in FAM])
    w = 0
    for _ in range(trials):
        fam = dfam + sig_fam * rng.standard_normal(len(FAM))
        if in_cluster:
            ours = fam[0]; above = int(q * sizes[0])          # urutan waktu di dalam klaster goodpjw
            above += int(sizes[1:][fam[1:] > ours].sum())
        else:
            ours = delta + sig_fam * rng.standard_normal()
            above = int(sizes[fam > ours].sum())
        small = 0.0 + 0.0007 * rng.standard_normal(200)      # variasi sendiri di 0,958
        above += int((small > ours).sum())
        g1 = (1 - s) * rng.exponential(g1_edge, n_g1) + 0.001 * (1 - s) + sig_ind * rng.standard_normal(n_g1)
        above += int((g1 > ours).sum())
        above += int(((g0 - 0.958) + 0.001 * rng.standard_normal(len(g0)) > ours).sum())
        w += (above + 1) <= K
    return w / trials
print(f'N {N_END} perak rank <= {K} | G0 {len(g0)}')
ds = (0.0, 0.0003, 0.0006, 0.001, 0.0015, 0.002)
print('skenario                         salinan(q=.4)  ' + '  '.join(f'd={d:.4f}' for d in ds))
for n_g1 in (150, 200, 260):
    for sf in (0.0003, 0.0006):
        pc = p_silver(0, n_g1=n_g1, sig_fam=sf, in_cluster=True)
        ps = [p_silver(d, n_g1=n_g1, sig_fam=sf) for d in ds]
        print(f'G1={n_g1:3d} sig_fam={sf:.4f}            {pc:5.2f}        ' + '  '.join(f'{p:7.2f}' for p in ps))
