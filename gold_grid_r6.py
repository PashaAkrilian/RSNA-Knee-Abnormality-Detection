#!/usr/bin/env python3
"""Gold setia 17 Okt: kombinasi compact R6flip / R7 / murid4 + S6F di separuh Raptor vs goodpjw & r7v6a (bootstrap 2000). Pakai: python3 gold_grid_r6.py"""
import sys; sys.argv = ['x']; sys.path.insert(0, '.')
import numpy as np
from gold_eval_full import R, Y, macro, L, WC_H, ids
def mk(r6=0., r7=0., st=0., s6=0., wc=WC_H, a224=0.35):
    rr=(1-a224)*R['r384m']+a224*R['r224m']; rr=(1-s6)*rr+s6*R['S6F']
    tot=r6+r7+st; ro=(r6*R['R6flip']+r7*R['R7']+st*R['murid4'])/tot
    w=np.array([(wc or {}).get(c,.5) for c in L]); return w*rr+(1-w)*ro
V={'goodpjw (R6flip, r224m, wc)':mk(r6=1),'r7v6a':mk(r7=.7,st=.3,s6=.5),
 'R6flip + S6F.5':mk(r6=1,s6=.5),'R6flip.7 + murid.3':mk(r6=.7,st=.3),
 'R6flip.7+murid.3+S6F.5':mk(r6=.7,st=.3,s6=.5),'R6.35 R7.35 murid.3 S6F.5':mk(r6=.35,r7=.35,st=.3,s6=.5),
 'R6.5 R7.2 murid.3 S6F.5':mk(r6=.5,r7=.2,st=.3,s6=.5),'R6.4 R7.2 murid.4 S6F.5':mk(r6=.4,r7=.2,st=.4,s6=.5),
 'R6.6 murid.4 S6F.5':mk(r6=.6,st=.4,s6=.5),'R6.7 murid.3 S6F.35':mk(r6=.7,st=.3,s6=.35)}
base=V['goodpjw (R6flip, r224m, wc)']; ref=V['r7v6a']
rng=np.random.default_rng(0); idx=[rng.integers(0,58,58) for _ in range(2000)]
for k,v in V.items():
    d0=np.array([macro(Y[i],v[i])-macro(Y[i],base[i]) for i in idx]); d1=np.array([macro(Y[i],v[i])-macro(Y[i],ref[i]) for i in idx])
    print(f'{k:30s} {macro(Y,v):.4f} | vs goodpjw {d0.mean():+.4f} P{np.mean(d0>0):.2f} | vs r7v6a {d1.mean():+.4f} P{np.mean(d1>0):.2f}')
