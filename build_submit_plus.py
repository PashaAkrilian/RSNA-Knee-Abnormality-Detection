#!/usr/bin/env python3
"""Kernel SUBMISSION TERINTEGRASI: pipeline publik d4-blend (0,943) + LEG FLEET MILIK KITA, blend rank berbobot global.

Memakai pola yang SAMA dgn "OUR LEG" di notebook publik pjmathematician/rsna-knee-d4-blend (sel 30-32): simpan keluaran publik
-> jalankan leg -> blend di ruang rank dgn SATU bobot global -> pulihkan otomatis ke keluaran publik bila leg gagal/tak lolos
gerbang (cakupan, fraksi-ikat 0,5, kolom konstan). Bagian dalam pipeline publik TIDAK diubah.
Yang diganti: sel 31 (leg) = leg KITA: model checkpoint kita (CoAtNet 384 murid fold / CNN murah / dst), studi test didekode
SEKALI, semua model dijalankan pada window yg sama dan dibagi bergilir ke 2 GPU; sel 32 hanya diubah bobot bawaannya.
Pengaman waktu: leg dimulai hanya bila sisa waktu >= need_h (jam); dibatalkan TOTAL bila anggaran habis (sel 32 lalu memulihkan keluaran publik).

Argumen kunci=nilai (lihat build_submit_multi.py utk slugs/files/span/k_eval/weights):
  name= slugs= files= span= k_eval= weights=   w= (bobot leg, bawaan 0.35)   need_h= (bawaan 1.2)
Contoh:
  python3 build_submit_plus.py name=students4 slugs=rsna-knee-kfold-c5-f0,...,rsna-knee-kfold-c5-f3 \
      files=kfold_c5_f0.pt,...,kfold_c5_f3.pt span=0.06,0.94 w=0.35
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
A = dict(kv.split("=", 1) for kv in sys.argv[1:])
NAME = A.get("name", "plus")
SLUGS = [s for s in A["slugs"].split(",") if s]
FILES = [s for s in A["files"].split(",") if s]
LO, HI = [float(x) for x in A.get("span", "0.06,0.94").split(",")]
K_EVAL = int(A.get("k_eval", 42))
WEIGHTS = [float(x) for x in A["weights"].split(",")] if A.get("weights") else [1.0] * len(FILES)
W = float(A.get("w", 0.35))
NEED_H = float(A.get("need_h", 1.2))
assert len(WEIGHTS) == len(FILES)
SLUG = f"rsna-knee-d4-plus-{NAME}".replace("_", "-")     # sama dgn slug yg diturunkan Kaggle dari judul "RSNA Knee - D4 plus <name>"
OUT = REPO / f"kaggle_kernel_d4plus_{NAME.replace('-', '_')}"

base = json.loads((REPO / "kaggle_base/pj_rsna_knee_d4_blend_20260928.ipynb").read_text())
meta0 = json.loads((REPO / "kaggle_base/pj_rsna_knee_d4_blend_metadata.json").read_text())
cells = base["cells"]
assert len(cells) == 33
assert "OUR LEG — 2/3" in "".join(cells[31]["source"]) and "OUR LEG — 3/3" in "".join(cells[32]["source"])

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()
OLD = "lo, hi = (int(n * 0.06), int(n * 0.94) - 1)"
assert OLD in ke_src
ke_src = ke_src.replace(OLD, f"lo, hi = (int(n * {LO}), int(n * {HI}) - 1)")

LEG = r'''
# ============================ OUR LEG — 2/3 (KITA): fleet model milik kita sendiri =================
import os as _o, time as _t, pathlib as _pl, glob as _g, json as _json, collections as _col, gc as _gc
import numpy as _np, pandas as _pdx, torch as _tc, torch.nn.functional as _F
from concurrent.futures import ThreadPoolExecutor as _TPE

def _proc_age_h():
    up = float(open('/proc/uptime').read().split()[0])
    start = int(open('/proc/self/stat').read().split(')')[-1].split()[19]) / _o.sysconf('SC_CLK_TCK')
    return (up - start) / 3600

_FK_SRC = __KE_SRC__
_FK_FILES = __FILES__
_FK_WEIGHTS = __WEIGHTS__
_FK_K = __KEVAL__
_FK_NEED_H = __NEEDH__
_OV_OURS = _pl.Path('/kaggle/working/ours_fleet/submission.csv'); _OV_OURS2 = _pl.Path('/kaggle/working/ours_core/submission.csv')
_LEFT = 9.0 - _proc_age_h() - 0.33
print(f"[ours 2/3] sisa waktu {_LEFT:.2f} h (butuh >= {_FK_NEED_H}); {len(_FK_FILES)} model: {_FK_FILES}", flush=True)
_fk_models = []
try:
    if _LEFT < _FK_NEED_H:
        raise RuntimeError(f"waktu tak cukup ({_LEFT:.2f} h < {_FK_NEED_H} h)")
    _t0 = _t.time()
    _FK = {'__name__': '_fk'}
    exec(compile(_FK_SRC, '<fk>', 'exec'), _FK)
    _LAB = _FK['LAB']; _NT = len(_LAB)
    _NORMS = {'imagenet': (_FK['_MEAN'], _FK['_STD']), 'rad': (_tc.full((3, 1, 1), 0.5), _tc.full((3, 1, 1), 0.5))}
    _NG = _tc.cuda.device_count()
    _DEVS = [f'cuda:{i}' for i in range(_NG)] if _NG else ['cpu']
    def _fk_glob(name):
        return sorted({_o.path.realpath(p) for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/') for p in _g.glob(d + name)})
    def _fk_resize(t, res):
        if t.shape[-1] != res:
            t = _F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False, antialias=True)[0]
        return t
    def _fk_windows(vol, mask, k, res, norm):
        D = vol.shape[0]; MEAN, STD = _NORMS[norm]
        cs = _FK['_eval_centers'](mask, D, k)
        w = _tc.empty((len(cs), 3, res, res), dtype=_tc.float32)
        for j, c in enumerate(cs):
            c = max(1, min(c, D - 2))
            t = _tc.from_numpy(_np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(_np.float32) / 255.0)
            w[j] = (_fk_resize(t, res) - MEAN) / STD
        return w
    for i, fname in enumerate(_FK_FILES):
        hits = _fk_glob(fname)
        assert len(hits) == 1, f'{fname}: {hits}'
        ck = _tc.load(hits[0], map_location='cpu', weights_only=False)
        bb = _FK['build_backbone'](ck['arch'], pretrained=False)
        m = _FK['RaptorClassifier'](bb, F_dim=bb.num_features); m.load_state_dict(ck['model'], strict=True)
        dev = _DEVS[i % len(_DEVS)]
        _fk_models.append((m.eval().to(dev), dev, (int(ck['res']), ck.get('norm', 'imagenet')))); del ck
        print(f"[ours 2/3] model {i}: {fname} -> {dev} res/norm={_fk_models[-1][2]}", flush=True)
    _RESES = sorted({r for _, _, r in _fk_models})
    _ROOT = _FK['find_test_root']()
    _tsdir = _ROOT + '/test_series'
    if not _o.path.isdir(_tsdir): _tsdir = _ROOT + '/test_images'
    _test = _pdx.read_csv(_ROOT + '/test.csv'); _test['StudyInstanceUID'] = _test['StudyInstanceUID'].astype(str)
    _ids = _test['StudyInstanceUID'].tolist(); _N = len(_ids)
    _tser = _pdx.read_csv(_ROOT + '/test_series.csv')
    _tser['StudyInstanceUID'] = _tser['StudyInstanceUID'].astype(str); _tser['SeriesInstanceUID'] = _tser['SeriesInstanceUID'].astype(str)
    _SER = {k: v.to_dict('records') for k, v in _tser.groupby('StudyInstanceUID')}
    _reader = _FK['_make_reader']()
    def _prep(sid):
        vol, mask = _FK['build_study'](sid, _SER, _tsdir, _reader)
        return {r: _fk_windows(vol, mask, _FK_K, r[0], r[1]) for r in _RESES}
    _P = _np.full((len(_fk_models), _N, _NT), _np.nan, _np.float32)
    _failed = []
    def _run_dev(dev, wins, j):
        with _tc.no_grad():
            for mi, (m, d, r) in enumerate(_fk_models):
                if d != dev: continue
                xw = wins[r].unsqueeze(0).to(d)
                with _tc.autocast('cuda', dtype=_tc.float16, enabled=d.startswith('cuda')):
                    _P[mi, j] = _tc.sigmoid(m(xw).float())[0].cpu().numpy()
    with _TPE(max_workers=3) as _ex, _TPE(max_workers=len(_DEVS)) as _gx:
        _q = _col.deque(); _nx = 0
        for i in range(_N):
            while _nx < _N and len(_q) < 8:
                _q.append((_nx, _ex.submit(_prep, _ids[_nx]))); _nx += 1
            j, fut = _q.popleft()
            if 9.0 - _proc_age_h() - 0.22 < 0:
                raise RuntimeError(f"anggaran waktu habis di studi {i}/{_N} -> leg dibatalkan total")
            try:
                wins = fut.result(); list(_gx.map(lambda d: _run_dev(d, wins, j), _DEVS)); del wins
            except Exception as e:
                _failed.append(_ids[j]); print(f"[ours 2/3] GAGAL studi {j}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            if (i + 1) % 100 == 0 or i + 1 == _N:
                print(f"[ours 2/3] {i+1}/{_N} studi | gagal={len(_failed)} | {(_t.time()-_t0)/60:.1f} mnt", flush=True)
    if len(_failed) > 0.02 * _N:
        raise RuntimeError(f"terlalu banyak studi gagal: {len(_failed)}/{_N}")
    for mi in range(len(_fk_models)):
        bad = ~_np.isfinite(_P[mi]).all(axis=1)
        if bad.any():
            fill = _np.nanmean(_np.where(_np.isfinite(_P[mi]), _P[mi], _np.nan), axis=0)
            _P[mi][bad] = _np.where(_np.isfinite(fill), fill, .5)
    def _rk(x): return x.argsort(0).argsort(0).astype(_np.float64) / max(1, len(x) - 1)
    _w = _np.asarray(_FK_WEIGHTS, _np.float64); _w = _w / _w.sum()
    _R = _np.tensordot(_w, _np.stack([_rk(_np.clip(_P[mi], 0, 1)) for mi in range(len(_fk_models))]), axes=(0, 0))
    _out = _pdx.DataFrame(_R.astype(_np.float32), columns=_LAB); _out.insert(0, 'StudyInstanceUID', _ids)
    _OV_OURS.parent.mkdir(parents=True, exist_ok=True)
    _out.to_csv(_OV_OURS, index=False)
    _json.dump({'n_studies': _N, 'n_covered_any': _N - len(_failed), 'models': _FK_FILES, 'minutes': (_t.time() - _t0) / 60}, open(_OV_OURS.parent / 'receipt.json', 'w'))
    _np.savez_compressed('/kaggle/working/ours_fleet_raw.npz', study_uids=_np.asarray(_ids), raw=_P)
    print(f"[ours 2/3] leg selesai: {_out.shape} | gagal={len(_failed)} | {(_t.time()-_t0)/60:.1f} mnt", flush=True)
except Exception as _e:
    print(f"[ours 2/3] SOFT FAIL: {type(_e).__name__}: {str(_e)[:300]} -> publik saja", flush=True)
    try: _OV_OURS.unlink()
    except Exception: pass
finally:
    for _m in _fk_models:
        try: del _m
        except Exception: pass
    _fk_models = []; _gc.collect()
    try: _tc.cuda.empty_cache()
    except Exception: pass
'''
LEG = (LEG.replace("__KE_SRC__", repr(ke_src)).replace("__FILES__", repr(FILES)).replace("__WEIGHTS__", repr(WEIGHTS))
          .replace("__KEVAL__", str(K_EVAL)).replace("__NEEDH__", str(NEED_H)))
compile(LEG, "<leg>", "exec")
cells[31]["source"] = LEG.splitlines(keepends=True)
cells[31]["outputs"] = []; cells[31]["execution_count"] = None

c32 = "".join(cells[32]["source"])
OLDW = '_OV_W = float(_ov_os.environ.get("RSNA_OURS_W", "0.45"))'
assert OLDW in c32
c32 = c32.replace(OLDW, f'_OV_W = float(_ov_os.environ.get("RSNA_OURS_W", "{W}"))')
compile(c32, "<c32>", "exec")
cells[32]["source"] = c32.splitlines(keepends=True)

OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(base, indent=1))
meta = dict(meta0)
meta.update({"id": f"dimaspashaakrilian/{SLUG}", "title": f"RSNA Knee - D4 plus {NAME}", "code_file": f"{SLUG}.ipynb", "is_private": True})
meta["dataset_sources"] = [d for d in meta0["dataset_sources"] if d]
meta["kernel_sources"] = list(meta0.get("kernel_sources", [])) + [f"dimaspashaakrilian/{s}" for s in dict.fromkeys(SLUGS)]
meta.pop("id_no", None)
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| model:", len(FILES), "| w =", W, "| span", (LO, HI), "| need_h", NEED_H)
print("kernel_sources:", meta["kernel_sources"])
