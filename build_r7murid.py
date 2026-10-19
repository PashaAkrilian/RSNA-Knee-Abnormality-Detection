#!/usr/bin/env python3
"""r7murid: notebook R7 heliosli (Raptor 0,5 + compact R7 0,5 = 0,956; salinan kita 57146631) + kaki ke-3 = murid ConvNeXt-T kita.

- Tahap 0 (baru): murid 4 fold (dataset `dimaspashaakrilian/rsna-knee-student-folds`) jalan sebagai SUBPROSES sebelum Raptor,
  dengan anggaran waktu sendiri -> /kaggle/working/student_test_preds.csv (NaN = studi tak terskor). Jam NB_T0 mulai di tahap 0,
  jadi DEADLINE 8,3 j compact ikut menghitung waktu murid.
- Tahap 3: setelah blend R7, per temuan out = (1-W)*rank(R7) + W*rank(murid); studi tak terskor murid tetap rank R7;
  murid < 97 % terskor -> R7 murni (dicetak MURID_LEG).
- Mode commit (test.csv < 50 studi): murid juga dijalankan pada 58 studi gold -> student_gold_preds.csv + "GOLD58 macro AUC" di log
  (harus ≈ emas per fold 0,919–0,924; jauh di bawah = decode/bobot rusak -> JANGAN submit).
- Murid DINO TIDAK dipakai (emas-58 0,71).

Pakai: python3 build_r7murid.py [w=0.2] [name=r7murid] [budget=4500] [base=r7|v224|v224m|g958] [slot=final|raptor] [flip=0|1]
  base=g958: notebook goodpjw 0,958 v27 tanpa Part B (compact = R6 + flip TTA heliosli, 384m + 224m, wc heliosli; 17 Okt).
  base=v224: basis = notebook varshitha (R7 + Raptor-r224 0,35 di slot Raptor; rank naik 233->156 di blok 0,956) -> nama default r7m224.
  base=v224m: notebook varshitha 14 Okt = v224 + TTA cermin anatomis nartaa utk r224 (rank 357->289 di blok 0,956).
  w=0      : tanpa tahap murid sama sekali (mis. hanya flip=1).
  slot=raptor: murid masuk DI DALAM slot Raptor (rr = (1-w) rr + w murid) -> keseimbangan keluarga Raptor/R7 tetap 0,5/0,5
             (pelajaran 14 Okt: murid r1 mirip Raptor, rho 0,93; sebagai kaki ke-3 di luar slot rank tidak naik).
  flip=1   : compact R7 memakai --flip-tta heliosli (v4c, mirror anatomis = augmentasi latih R7 mirror 0,5) dgn eval-stride 2
             (biaya GPU ≈ stride 1 tanpa flip).
"""
import copy
import re
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_student_infer as bsi

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
W = float(args.get("w", "0.2"))
BASE_ARG = args.get("base", "r7")
assert BASE_ARG in ("r7", "v224", "v224m", "g958"), BASE_ARG
V224 = BASE_ARG in ("v224", "v224m", "g958")  # basis varshitha (slot Raptor = 0,65 r384 + 0,35 r224)
SLOT = args.get("slot", "final")
W2 = float(args.get("w2", "0"))          # 19 Okt: bobot murid di slot Raptor (slot=both) — murid menggantikan sebagian r224
assert 0.0 <= W2 <= 0.5, W2
A224O = float(args.get("a224", "-1"))    # 19 Okt: timpa A224 notebook (default -1 = biarkan)
FLIP = args.get("flip", "0") == "1"
FSTRIDE = int(args.get("fstride", "2"))   # eval-stride compact saat flip=1 (2 = biaya GPU ≈ tanpa flip; 1 = TTA penuh, ±+1,7 j)
assert FSTRIDE in (1, 2, 3)
assert SLOT in ("final", "raptor", "compact", "both")
FOLDS_DS = args.get("folds", "dimaspashaakrilian/rsna-knee-student-folds")   # set fold murid (r1 = default; r2a/r2c = dataset baru)
NAME = args.get("name", "r7m224" if V224 else "r7murid")
S6F = float(args.get("s6f", "0"))   # bobot S6F nartaa (all-data) DI DALAM slot compact, diterapkan sebelum murid; 0 = tanpa tahap S6F
assert 0.0 <= S6F <= 0.6, S6F
S6F_SLOT = args.get("s6fslot", "compact")   # compact = campur ke ro (R7); raptor = campur ke rr (keluarga label Raptor/JEV)
assert S6F_SLOT in ("compact", "raptor"), S6F_SLOT
WC = args.get("wc", "")   # "h" = bobot Raptor per temuan heliosli 16 Okt (notebook 0,958); "h2" = versi 17 Okt (notebook 0,9586)
assert WC in ("", "h", "h2"), WC
WC_LINE = {"h": "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   # [wc=h] heliosli 16 Okt (notebook 0,958)\n",
           "h2": "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}   # [wc=h2] heliosli 17 Okt (notebook 0,9586, probe papan per kelas)\n"}
EXT = args.get("ext", "")                 # 20 Okt: "slug:w" = kaki CSV prediksi 24-kanal (test_pred_2plv4.csv) di blend akhir
EXT_DS = EXT.split(":")[0] if EXT else ""
EW = float(EXT.split(":")[1]) if EXT else 0.0
assert 0.0 <= EW <= 0.4, EW
R7ADD = float(args.get("r7add", "0"))   # 18 Okt (base=g958): R7 heliosli sbg model compact ke-2 SETELAH S6F; bobot R7 di dalam gabungan compact
C2 = args.get("c2", "r7")   # 18 Okt malam: model compact ke-2 (bobot r7add) = R7 heliosli ATAU "bleg" = B-leg publik (R6 fine-tune label soft-v2)
assert C2 in ("r7", "bleg"), C2
BLEG_DS = ["kritikabenjwal/rsna-knee-bleg-s0-f01", "sarthakgupta1801/rsna-knee-bleg-s0-f23"]   # E_prior_s0_f0..f3 (seed 0, 4 fold)
R6STRIDE = args.get("r6stride", "")      # base=g958: eval-stride R6 (manifest = 1; 2 = ±separuh waktu GPU R6) agar R7 sempat jalan
assert 0.0 <= R7ADD <= 0.6 and R6STRIDE in ("", "1", "2", "3") and (BASE_ARG == "g958" or not (R7ADD or R6STRIDE)), (R7ADD, R6STRIDE)
BUDGET = int(args.get("budget", "4500"))
assert 0.0 <= W <= 0.5 and 600 <= BUDGET <= 7200, (W, BUDGET)
_b = {"r7": "r7_heliosli_0957_20261013", "v224": "r7_r224_varshitha_20261013", "v224m": "r7_r224m_varshitha_20261014",
      "g958": "r6_goodpjw_0958_20261016"}[BASE_ARG]   # g958: goodpjw v27 (0,958) tanpa Part B = R6+flip compact, 384m+224m, wc heliosli
BASE = ROOT / "kaggle_base" / f"{_b}.ipynb"
BASE_META = ROOT / "kaggle_base" / f"{_b}.meta.json"
SLUG = "rsna-knee-" + NAME
FOLDS2 = args.get("folds2", "")   # 19 Okt: dataset fold KEDUA (murid gabungan, mis. r4) -> inferensi memakai semua fold[0-9].pt
EXTRA_DS = [FOLDS_DS] + ([FOLDS2] if FOLDS2 else []) + ["goodpjw2008/rsna-knee-2-5d-convnext-reader"]
MB, ME = "# ---- [murid] kaki ke-3", "# ---- [murid] selesai ----"
S6F_DS = "nartaa/rsna-knee-s6fall-weights"
S6F_BASE = ROOT / "kaggle_base" / "s6fall_solo_nartaa_20261015.py"   # skrip publik nartaa (komponen B = S6F all-data, 0,950 solo)

DECODE = bsi.CELL3.split("# cache volume per-worker")[0]      # fungsi decode = salinan student-prep (geometri cache latih)
assert all(f"def {f}(" in DECODE for f in ("_read", "_canon_slice", "_resample", "load_series")) and "say(" not in DECODE

STUDENT = r'''%%writefile /tmp/student_run.py
# Stage 0 (ours): ConvNeXt-T 2.5D student, 4 folds -> /kaggle/working/student_test_preds.csv (NaN = study not scored).
import os, sys, glob, time, json
import numpy as np, pandas as pd, torch
T0 = time.time()
BUDGET = __BUDGET__
def say(*a): print('[murid %6.0fs]' % (time.time() - T0), *a, flush=True)
I = '/kaggle/input'
def mount(slug, required):
    for p in [f'{I}/{slug}', f'{I}/competitions/{slug}'] + sorted(glob.glob(f'{I}/datasets/*/{slug}')) + sorted(glob.glob(f'{I}/*/{slug}')):
        if os.path.exists(os.path.join(p, required)):
            return p
    raise FileNotFoundError(slug + '/' + required)
COMP = mount('rsna-knee-abnormality-detection', 'test.csv')
GOODPJW = mount('rsna-knee-2-5d-convnext-reader', 'knee.py')
FOLD_DIR = mount('__FOLDSLUG__', 'fold0.pt')
sys.path.insert(0, GOODPJW)
from knee import LABELS, KneeNet, StudyDataset, build_study_table
__DECODE__
FOLDS = sorted(glob.glob(FOLD_DIR + '/fold[0-9].pt'))
assert len(FOLDS) >= 2, FOLDS   # >=2: murid r2 dua fold boleh diuji lebih awal
torch.backends.cudnn.benchmark = True
MODELS = []
for f in FOLDS:
    sd = torch.load(f, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    m = KneeNet('convnext_tiny', pretrained=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (f, missing[:5], unexpected[:5])
    MODELS.append(m.cuda().eval().to(memory_format=torch.channels_last))
say('folds', [os.path.basename(f) for f in FOLDS], '| COMP', COMP)


class Safe(torch.utils.data.Dataset):
    # a study whose decode raises is returned empty (ok=0) instead of killing the loader
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        try:
            x, mask, pos, _ = self.base[i]
            return x, mask, pos, 1
        except Exception:
            return torch.zeros(1, dtype=torch.uint8), torch.zeros(6, dtype=torch.bool), torch.zeros(1), 0


def predict(split, ids, series_csv, budget, out_csv):
    ser = pd.read_csv(series_csv)
    for c in ('StudyInstanceUID', 'SeriesInstanceUID'):
        ser[c] = ser[c].astype(str)
    ser = ser[ser.StudyInstanceUID.isin(set(ids))]
    dirs, counts = {}, {}
    for st, sr in zip(ser.StudyInstanceUID, ser.SeriesInstanceUID):
        p = f'{COMP}/{split}/{st}/{sr}'
        if os.path.isdir(p):
            dirs[sr] = p
            counts[sr] = min(len(os.listdir(p)), MAX_SLICES)     # = slice count of the training cache (capped at 28)
    table = build_study_table(ser, counts)
    ds = Safe(StudyDataset(ids, table, None, None, 12, False, loader=lambda sr: load_series(dirs[sr])[0]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    P = np.full((len(ids), len(LABELS)), np.nan, np.float32)
    t1, i = time.time(), -1
    for i, (x, mask, pos, ok) in enumerate(dl):
        if time.time() - t1 > budget:
            say('budget reached at study', i, 'of', len(ids)); break
        if int(ok) == 0 or not bool(mask.any()):
            continue
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
                p = sum(torch.sigmoid(m(x, mask, pos, res=256).float()) for m in MODELS) / len(MODELS)
            P[i] = p[0].cpu().numpy()
        except Exception as e:
            say('study', i, 'failed', type(e).__name__, str(e)[:120])
        if (i + 1) % 200 == 0:
            say(i + 1, '/', len(ids), 'scored', int(np.isfinite(P[:, 0]).sum()))
    df = pd.DataFrame(P, columns=['cnxt_' + c for c in LABELS])
    df.insert(0, 'StudyInstanceUID', ids)
    df.to_csv(out_csv + '.tmp', index=False)
    os.replace(out_csv + '.tmp', out_csv)
    say(out_csv, 'scored', int(np.isfinite(P[:, 0]).sum()), '/', len(ids), '| %.2f s/study' % ((time.time() - t1) / max(1, i + 1)))
    return P


test_ids = pd.read_csv(COMP + '/test.csv')['StudyInstanceUID'].astype(str).tolist()
predict('test_series', test_ids, COMP + '/test_series.csv', BUDGET, '/kaggle/working/student_test_preds.csv')
if len(test_ids) < 50:      # commit run (visible sample test): validate decode + weights on the 58 gold studies
    from sklearn.metrics import roc_auc_score
    tr = pd.read_csv(COMP + '/train.csv')
    tr['StudyInstanceUID'] = tr.StudyInstanceUID.astype(str)
    gold = tr[tr[LABELS].notna().all(axis=1)]
    gids = [u for u in gold.StudyInstanceUID if os.path.isdir(f'{COMP}/train_series/{u}')]
    G = predict('train_series', gids, COMP + '/train_series.csv', 1800, '/kaggle/working/student_gold_preds.csv')
    y = gold.set_index('StudyInstanceUID').loc[gids, LABELS].values.astype(float)
    ok = np.isfinite(G).all(axis=1)
    aucs = {c: round(float(roc_auc_score(y[ok, j], G[ok, j])), 4) for j, c in enumerate(LABELS) if len(set(y[ok, j])) == 2}
    say('GOLD58 macro AUC %.4f over %d studies' % (np.mean(list(aucs.values())), int(ok.sum())), json.dumps(aucs))
say('MURID_DONE')
'''

STAGE0 = r'''# Stage 0 (ours, build_r7murid): our ConvNeXt-T student in its own process, before Raptor, with its own time budget.
# The clock starts here (NB_T0), so the later 8.3 h deadline includes the student's time. Any failure -> R7 only.
import os, sys, time, subprocess
NB_T0 = time.time()
try:
    _rc = subprocess.run([sys.executable, '-u', '/tmp/student_run.py'], timeout=__TIMEOUT__).returncode
    print('MURID_STAGE rc %d seconds %.0f' % (_rc, time.time() - NB_T0), flush=True)
except subprocess.TimeoutExpired:
    print('MURID_STAGE timeout -> R7 only', flush=True)
'''

LOAD = r'''
# ---- [murid] kaki murid (build_r7murid): our ConvNeXt-T student, weight fixed before any run; ranked over its own scored rows;
# a study it did not score keeps the other legs' rank; fewer than 97 % scored -> no student leg (printed)
W_MURID = __W__
murid = {'w': W_MURID, 'slot': '__SLOT__', 'used': False}
_stu_v = _stu_ok = None
_sp = WORK / 'student_test_preds.csv'
if W_MURID > 0 and _sp.is_file():
    _s = pd.read_csv(_sp, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    _cols = ['cnxt_' + c for c in labels]
    if set(_cols) <= set(_s.columns):
        _v = _s.reindex(test_ids)[_cols].to_numpy(dtype=np.float64)
        _ok = np.isfinite(_v).all(axis=1)
        murid['scored'] = int(_ok.sum())
        if _ok.mean() >= 0.97:
            _stu_v, _stu_ok, murid['used'] = _v, _ok, True
print('MURID_LEG', json.dumps(murid), flush=True)
'''

APPLY_FINAL = r'''if _stu_ok is not None:   # [murid] third leg on the final blend
    for j, c in enumerate(labels):
        _rb = pct(out[:, j])
        out[:, j] = np.where(_stu_ok, (1.0 - W_MURID) * _rb + W_MURID * pct_on(_stu_v[:, j], _stu_ok), _rb)
# ---- [murid] selesai ----
'''

EXT_BLEND = r'''if __EW__ > 0:   # [ext] external per-finding predictions (24-channel, different inductive bias)
    import glob as _g
    _f = _g.glob('/kaggle/input/**/test_pred_2plv4.csv', recursive=True)
    if _f:
        _e = pd.read_csv(_f[0], index_col=0)
        _e.index = _e.index.astype(str)
        _e = _e.reindex(test_ids)
        _hit = np.isfinite(_e[labels].values).all(axis=1)
        if _hit.mean() >= 0.97:
            for j, c in enumerate(labels):
                _r = pct(out[:, j])
                _x = pct_on(_e[c].values, _hit)
                out[:, j] = np.where(_hit, (1.0 - __EW__) * _r + __EW__ * _x, _r)
            print('EXT_LEG used=1 w=%.2f hit=%.3f' % (__EW__, _hit.mean()), flush=True)
        else:
            print('EXT_LEG used=0 hit=%.3f' % _hit.mean(), flush=True)
    else:
        print('EXT_LEG used=0 file-missing', flush=True)
'''

IN_LOOP = ("    if _stu_ok is not None:   # [murid] student inside the Raptor slot (Raptor-like, rho 0.93), over its own scored rows\n"
           "        rr = np.where(_stu_ok, (1.0 - W_MURID) * rr + W_MURID * pct_on(_stu_v[:, j], _stu_ok), rr)\n")
IN_LOOP_COMPACT = IN_LOOP.replace("Raptor slot (Raptor-like, rho 0.93)", "compact (R7) slot (student taught by R7 OOF)").replace("rr", "ro")
IN_LOOP2 = IN_LOOP.replace("W_MURID", "__W2__")   # 19 Okt: murid di slot Raptor dgn bobot sendiri (slot=both)


# tahap S6F: skrip nartaa UTUH, hanya komponen B (A = tahap-1 Raptor-384 kita sendiri, sumber byte-identik bcd864cb...);
# guard skrip = sisa waktu run; keluaran B_probs.csv -> /kaggle/working/s6f_raw.csv; gagal apa pun -> tanpa kaki S6F
S6F_PATCH = [
    ('GUARD_MAX_S = 480 * 60', 'GUARD_MAX_S = int(float(os.environ.get("S6F_GUARD_S", "3600")))   # [s6f] sisa waktu notebook induk'),
    ('A_TIMEOUT_S = 9000', 'A_TIMEOUT_S = 0   # [s6f] A tidak dijalankan'),
    ('B_TIMEOUT_S = 14400', 'B_TIMEOUT_S = 600   # [s6f] lantai (hanya peringatan); anggaran B = guard - END_SLACK_S'),
    ('SETUP_SLACK_S = 30 * 60', 'SETUP_SLACK_S = 10 * 60'),
    ('    for p in (WORK / "submission.csv", WORK / ".submission.csv.tmp"):',
     '    for p in ():   # [s6f] submission.csv milik notebook induk, jangan disentuh'),
    ('        A, a_info = run_a(I)', '        A, a_info = None, {"skipped": "[s6f] A = stage-1 Raptor-384 of the host notebook"}'),
    ('        _RECEIPT["blend"] = blend_and_write(I, A, B, b_failed)',
     '        _RECEIPT["blend"] = {"s6f_probs": str(OUT / "B" / "B_probs.csv"), "b_failed": b_failed}   # [s6f] blend di notebook induk'),
    ('"submission_sha256": _RECEIPT["blend"]["submission_sha256"]', '"s6f_probs": _RECEIPT["blend"]["s6f_probs"]'),
]

S6F_RUN = r"""# Stage 2c (ours, build_r7murid s6f): nartaa's public S6 ConvNeXt trained on all data (0.950 alone; forum 749359), component B of
# his published script only (A = our stage-1 Raptor-384, the same scored source).  Own time cap from what is left of the run
# (ends by NB_T0 + 8.0 h); any failure or timeout -> no S6F leg (the blend then uses the other members).
import os, sys, time, json, shutil, subprocess
from pathlib import Path
_s6f_t0 = time.time()
_s6f_left = (NB_T0 + 8.0 * 3600) - time.time()
S6F_STATUS = {'status': 'not started', 'left_s': round(_s6f_left)}
if _s6f_left < 45 * 60:
    S6F_STATUS['status'] = 'skipped: too little time left'
else:
    _cap = int(min(4 * 3600, _s6f_left - 10 * 60))
    S6F_STATUS['cap_s'] = _cap
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/s6f_run.py'], env=dict(os.environ, S6F_GUARD_S=str(_cap)),
                             timeout=_cap + 900).returncode
        _p = Path('/kaggle/working/k1/B/B_probs.csv')
        if _rc == 0 and _p.is_file():
            shutil.copyfile(_p, '/kaggle/working/s6f_raw.csv')
            S6F_STATUS['status'] = 'ok'
        else:
            S6F_STATUS['status'] = 'failed rc %s' % _rc
    except subprocess.TimeoutExpired:
        S6F_STATUS['status'] = 'timeout'
S6F_STATUS['seconds'] = round(time.time() - _s6f_t0)
print('S6F_STAGE', json.dumps(S6F_STATUS), flush=True)
"""

S6F_LOAD = r"""
# ---- [s6f] nartaa's S6 ConvNeXt (all data) inside the compact slot, weight fixed before any run, ranked over its own scored
# rows; fewer than 97 % scored, or a mean rank correlation with R7 below 0.5 (n >= 100) -> no S6F leg (printed)
W_S6F = __WS6F__
s6f = {'w': W_S6F, 'used': False}
_s6f_v = _s6f_ok = None
_s6p = WORK / 's6f_raw.csv'
if W_S6F > 0 and _s6p.is_file():
    _s = pd.read_csv(_s6p, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    if set(labels) <= set(_s.columns):
        _v = _s.reindex(test_ids)[labels].to_numpy(dtype=np.float64)
        _ok = np.isfinite(_v).all(axis=1)
        _both = _ok & ~our_fail
        s6f['scored'] = int(_ok.sum())
        if _both.sum() >= 3:
            _rho = [pd.Series(_v[_both, j]).corr(pd.Series(ours[c].values[_both]), method='spearman') for j, c in enumerate(labels)]
            s6f['rho_r7_mean'] = round(float(np.nanmean(_rho)), 4)
        if _ok.mean() >= 0.97 and (_both.sum() < 100 or s6f.get('rho_r7_mean', 0) >= 0.5):
            _s6f_v, _s6f_ok, s6f['used'] = _v, _ok, True
print('S6F_LEG', json.dumps(s6f), flush=True)
"""
S6F_IN_LOOP = ("    if _s6f_ok is not None:   # [s6f] S6F inside the compact (R7) slot, over its own scored rows (before the student)\n"
               "        ro = np.where(_s6f_ok, (1.0 - W_S6F) * ro + W_S6F * pct_on(_s6f_v[:, j], _s6f_ok), ro)\n")
S6F_IN_LOOP_RAPTOR = re.sub(r"\bro\b", "rr", S6F_IN_LOOP.replace("compact (R7) slot", "Raptor slot (JEV-label family)"))


R7_RUN = r"""# Stage 2d (ours, build_r7murid r7add): heliosli's public R7 compact model (bundle 8c7b666e, its setup + controller cells unchanged,
# own temp dir) after S6F, as a second compact model.  Its controller stops at NB_T0 + 8.3 h like R6's; any failure -> R6 alone.
import shutil, time
_r7t0 = time.time(); R7_OK = False
shutil.copyfile(WORK / 'ours_sub.csv', WORK / 'ours_r6.csv'); R6_PRED = pred
try:
    exec(compile(open('/tmp/r7_stage.py', encoding='utf-8').read(), '/tmp/r7_stage.py', 'exec'), globals())
    shutil.copyfile(WORK / 'ours_sub.csv', WORK / 'ours_r7.csv'); R7_PRED = pred; R7_OK = True
except BaseException as _e:
    print('R7_STAGE_FAILED', repr(_e)[:300], flush=True)
print('R7_STAGE', 'ok' if R7_OK else 'failed', 'seconds %.0f' % (time.time() - _r7t0), flush=True)
"""

R7_MERGE = r"""# [r7add] compact slot = (1 - W_R7) rank(R6) + W_R7 rank(R7), each ranked over its own scored studies (a study only one model scored
# takes that model's rank); R7 failed or < 97 % scored -> R6 alone (printed).  The blend cell below then sees one compact model.
import csv, json, shutil
import numpy as np, pandas as pd
from pathlib import Path
W_R7 = __WR7__
_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
_lbl = [c for c in pd.read_csv(WORK / 'ours_r6.csv', nrows=1).columns if c != 'StudyInstanceUID']
def _scored(pdir):
    s = set()
    for f in Path(pdir).glob('worker*/predictions.csv'):
        with open(f, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                try:
                    if all(0.0 <= float(r[c]) <= 1.0 for c in _lbl):
                        s.add(r['StudyInstanceUID'])
                except (KeyError, ValueError, TypeError):
                    pass
    return s
def _pr(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pd.Series(np.asarray(v, dtype=np.float64)[ok]).rank(pct=True).values
    return r
_s6 = _scored(R6_PRED)
_s7 = _scored(R7_PRED) if R7_OK else set()
R7_LEG = {'w': W_R7, 'r7_ok': bool(R7_OK), 'r6_scored': len(_s6 & set(_ids)), 'r7_scored': len(_s7 & set(_ids)), 'used': False}
if R7_OK and len(_s7 & set(_ids)) >= 0.97 * len(_ids):
    _r6 = pd.read_csv(WORK / 'ours_r6.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').loc[_ids, _lbl]
    _r7 = pd.read_csv(WORK / 'ours_r7.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').loc[_ids, _lbl]
    _ok6, _ok7 = np.array([u in _s6 for u in _ids]), np.array([u in _s7 for u in _ids])
    _m = np.zeros((len(_ids), len(_lbl)))
    for j, c in enumerate(_lbl):
        a_, b_ = _pr(_r6[c].values, _ok6), _pr(_r7[c].values, _ok7)
        _m[:, j] = np.where(_ok6 & _ok7, (1.0 - W_R7) * a_ + W_R7 * b_, np.where(_ok6, a_, np.where(_ok7, b_, 0.5)))
    _mg = pd.DataFrame(_m, columns=_lbl)
    _mg.insert(0, 'StudyInstanceUID', _ids)
    _mg.to_csv(WORK / 'ours_sub.csv', index=False)
    _pm = Path('/kaggle/temp' if Path('/kaggle/temp').is_dir() else '/tmp') / 'probe_merged' / 'worker0'
    _pm.mkdir(parents=True, exist_ok=True)
    _mg[[u in (_s6 | _s7) for u in _ids]].to_csv(_pm / 'predictions.csv', index=False)
    pred = _pm.parent                                   # the blend reads pred/worker*/predictions.csv for "scored"
    R7_LEG['rho_r6_r7_mean'] = round(float(np.mean([pd.Series(_r6[c].values[_ok6 & _ok7]).corr(pd.Series(_r7[c].values[_ok6 & _ok7]), method='spearman') for c in _lbl])), 4) if (_ok6 & _ok7).sum() >= 3 else None
    R7_LEG['used'] = True
else:
    shutil.copyfile(WORK / 'ours_r6.csv', WORK / 'ours_sub.csv')
    pred = R6_PRED
print('R7_LEG', json.dumps(R7_LEG), flush=True)
"""


def bleg_stage_script():
    """B-leg = 4 fold ConvNeXt-S publik (R6 fe055406 di-fine-tune 2 epoch dgn label soft-v2, arm E_prior; OOF lokal 2.651 studi:
    ganti R7 di compact +0,0033 P 1,00; rho~R6 0,81). Jalan lewat setup + controller R6 basis g958 (manifest R6 utk aset/argumen),
    bobot dari dataset B-leg, TANPA flip & TANPA stacker R6 (stacker dipasang pd keluaran R6; bukti OOF/gold B-leg = logit mentah)."""
    b = json.loads((ROOT / "kaggle_base" / "r6_goodpjw_0958_20261016.ipynb").read_text(encoding="utf-8"))["cells"]
    k = [i for i, c in enumerate(b) if src(c).startswith("ARM = ")]
    assert len(k) == 1 and src(b[k[0] + 1]).lstrip().startswith("deadline = T0 + DEADLINE_HOURS"), k
    setup, run = src(b[k[0]]), src(b[k[0] + 1])
    rep = [("ARM = 'E_lrprior'\n", "ARM = 'E_prior'   # [bleg] B-leg fold models carry arm E_prior (inference math is arm-independent)\n"),
           ("FLIP_TTA = True", "FLIP_TTA = False   # [bleg] B-leg evidence (OOF, gold) is without flip;"),
           ("TMP = TMP / 'probe'\n", "TMP = TMP / 'probe_bleg'   # [bleg] own temp dir: R6 used /probe\n"),
           ("for name in manifest['models'] + (manifest.get('models2') or []):\n"
            "    assert hashlib.sha256((M_SRC / name).read_bytes()).hexdigest() == manifest['sha256'][name], name\n"
            "    shutil.copy(M_SRC / name, BUNDLE / name)\n",
            "_bd = [d for d in list(INPUT.glob('rsna-knee-bleg-s0-f*')) + list(INPUT.glob('datasets/*/rsna-knee-bleg-s0-f*')) + list(INPUT.glob('*/rsna-knee-bleg-s0-f*')) if d.is_dir()]\n"
            "_bl = sorted({p.name: p for d in _bd for p in d.rglob('E_prior_s0_f*.pt')}.values(), key=lambda p: p.name)   # [bleg] public B-leg seed-0 folds\n"
            "assert [p.name for p in _bl] == ['E_prior_s0_f%d.pt' % i for i in range(4)], [str(p) for p in _bl]\n"
            "for p in _bl:\n"
            "    shutil.copy(p, BUNDLE / p.name)\n"
            "    print('BLEG_MODEL', p.name, hashlib.sha256((BUNDLE / p.name).read_bytes()).hexdigest()[:16], flush=True)\n"),
           ("'--eval-stride', str(manifest.get('eval_stride', 3))", "'--eval-stride', '2'"),
           ("if manifest.get('fluid_stacker'):\n    # v4b", "if False:   # [bleg] R6's fluid stacker was fit on R6 outputs\n    # v4b"),
           ("if manifest.get('state_stacker'):\n    # v4c", "if False:   # [bleg] R6's state stacker was fit on R6 outputs\n    # v4c"),
           ("print('ASSETS_READY'", "print('BLEG_ASSETS_READY'")]
    for a, c in rep:
        assert setup.count(a) == 1, a[:60]
        setup = setup.replace(a, c)
    s = setup + "\n" + run
    compile(s, "bleg_stage.py", "exec")
    return "%%writefile /tmp/r7_stage.py\n" + s


def r7_stage_script():
    b = json.loads((ROOT / "kaggle_base" / "r7_r224m_varshitha_20261014.ipynb").read_text(encoding="utf-8"))["cells"]
    k = [i for i, c in enumerate(b) if src(c).startswith("ARM = ")]
    assert len(k) == 1 and src(b[k[0] + 1]).lstrip().startswith("deadline = T0 + DEADLINE_HOURS"), k
    setup, run = src(b[k[0]]), src(b[k[0] + 1])
    assert "BUNDLE_ID = '8c7b666e27dc0e8f'" in setup and setup.count("TMP = TMP / 'probe'\n") == 1
    setup = setup.replace("TMP = TMP / 'probe'\n", "TMP = TMP / 'probe_r7'   # [r7add] own temp dir: R6 used /probe\n")
    s = setup + "\n" + run
    compile(s, "r7_stage.py", "exec")
    return "%%writefile /tmp/r7_stage.py\n" + s


def s6f_script():
    s = S6F_BASE.read_text(encoding="utf-8")
    for a, b in S6F_PATCH:
        assert s.count(a) == 1, ("pola S6F tidak tunggal", a)
        s = s.replace(a, b)
    compile(s, "s6f_run.py", "exec")
    return "%%writefile /tmp/s6f_run.py\n" + s


def src(c):
    return "".join(c["source"])


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def build():
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    cells = copy.deepcopy(nb["cells"])
    # sel dicari dari isinya (basis R7 = 7 sel, basis varshitha = 13 sel)
    i_st1 = [k for k, c in enumerate(cells) if "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py']" in src(c)]
    i_bl = [k for k, c in enumerate(cells) if src(c).startswith("# Stage 3: per-finding rank blend")]
    assert len(i_st1) == 1 and len(i_bl) == 1 and i_st1[0] < i_bl[0] == len(cells) - 1, (i_st1, i_bl)
    i_st1, i_bl = i_st1[0], i_bl[0]
    assert any(src(c).startswith("%%writefile /tmp/raptor_run.py") for c in cells[:i_st1])
    assert ("A224" in src(cells[i_bl])) == V224
    c3 = src(cells[i_st1])
    assert c3.count("NB_T0 = time.time()\n") == 1
    c3 = c3.replace("NB_T0 = time.time()\n", "NB_T0 = globals().get('NB_T0') or time.time()   # [murid] Stage 0 started the clock\n")
    c6 = src(cells[i_bl])
    anchor = "sub = pd.DataFrame(out.astype(np.float32), columns=labels)\n"
    rec = "'rank_basis': 'each model over its own non-failed rows'}"
    assert c6.count(anchor) == 1 and c6.count(rec) == 1 and "def pct_on(" in c6 and MB not in c6
    a_out, a_wr = "out = np.zeros((n, len(labels)))\n", "    wr = W_RAPTOR_C.get(c, W_RAPTOR)\n"
    if WC:
        k = [l for l in c6.splitlines(keepends=True) if l.startswith("W_RAPTOR_C = {")]
        assert len(k) == 1, k
        c6 = c6.replace(k[0], WC_LINE[WC])
    if S6F > 0:                                                    # S6F: muat sebelum loop; campur ke ro SEBELUM baris murid
        assert c6.count(a_out) == 1 and c6.count(a_wr) == 1
        c6 = c6.replace(a_out, S6F_LOAD.replace("__WS6F__", repr(S6F)) + a_out).replace(
            a_wr, (S6F_IN_LOOP if S6F_SLOT == "compact" else S6F_IN_LOOP_RAPTOR) + a_wr)
    if EW > 0:
        assert c6.count(anchor) == 1
        c6 = c6.replace(anchor, EXT_BLEND.replace("__EW__", repr(EW)) + anchor)
    load = LOAD.replace("__W__", repr(W)).replace("__SLOT__", SLOT)
    if W > 0 and SLOT == "final":
        c6 = c6.replace(anchor, load + APPLY_FINAL + anchor)
    elif W > 0:                                                    # slot=raptor|compact|both: muat sebelum loop, campur ke rr|ro di dalam loop
        assert c6.count(a_out) == 1 and c6.count(a_wr) == 1
        mix = {"raptor": IN_LOOP, "compact": IN_LOOP_COMPACT, "both": IN_LOOP_COMPACT + IN_LOOP2.replace("__W2__", repr(W2))}[SLOT]
        c6 = c6.replace(a_out, load + a_out).replace(a_wr, mix + a_wr)
    extra_rec = (["'ext': {'ds': %r, 'w': %r}" % (EXT_DS, EW)] if EW > 0 else []) + (["'murid': murid"] if W > 0 else []) + (["'s6f': s6f"] if S6F > 0 else [])
    if extra_rec:
        c6 = c6.replace(rec, rec[:-1] + ", " + ", ".join(extra_rec) + "}")
    if A224O >= 0:                                                 # 19 Okt: ganti porsi r224 di slot Raptor (mis. 0 = murid penuh)
        k = [l for l in c6.splitlines(keepends=True) if l.startswith("A224 = ")]
        assert len(k) == 1, k
        c6 = c6.replace(k[0], f"A224 = {A224O}   # [a224] Raptor-slot share of the 224-crop model (ours, 19 Okt)\n")
    mid = cells[i_st1 + 1:i_bl]
    if FLIP:                                                       # compact R7: flip-TTA heliosli v4c + eval-stride 2
        k_set = [k for k, c in enumerate(mid) if "if manifest.get('flip_tta'):" in src(c)]
        assert len(k_set) == 1, k_set
        c4 = src(mid[k_set[0]])
        for a_, b_ in (("if manifest.get('flip_tta'):", "if True:   # [flip=1] anatomical mirror TTA (trainer v9.4 mirror; R7 trained with mirror 0.5)"),
                       ("'--eval-stride', str(manifest.get('eval_stride', 3))", f"'--eval-stride', '{FSTRIDE}'")):   # flip x stride FSTRIDE
            assert c4.count(a_) == 1, a_
            c4 = c4.replace(a_, b_)
        mid = mid[:k_set[0]] + [code(c4)] + mid[k_set[0] + 1:]
    if R6STRIDE:                                                   # g958: R6 eval-stride (manifest R6 = 1)
        k_r6 = [k for k, c in enumerate(mid) if "BUNDLE_ID = 'fe055406fad6a12f'" in src(c)]
        assert len(k_r6) == 1, k_r6
        a_ = "'--eval-stride', str(manifest.get('eval_stride', 3))"
        assert src(mid[k_r6[0]]).count(a_) == 1
        mid = mid[:k_r6[0]] + [code(src(mid[k_r6[0]]).replace(a_, f"'--eval-stride', '{R6STRIDE}'"))] + mid[k_r6[0] + 1:]
    student = STUDENT.replace("__BUDGET__", str(BUDGET)).replace("__DECODE__", DECODE).replace("__FOLDSLUG__", FOLDS_DS.split("/")[1])
    if FOLDS2:                                                     # 19 Okt: mount kedua; glob fold[0-9].pt di inferensi menangkap keduanya
        student = student.replace("FOLD_DIR = mount('" + FOLDS_DS.split("/")[1] + "', 'fold0.pt')",
                                  "FOLD_DIR = mount('" + FOLDS_DS.split("/")[1] + "', 'fold0.pt')\nFOLD_DIR2 = mount('" + FOLDS2.split("/")[1] + "', 'fold0.pt')")
        assert student.count("FOLDS = sorted(glob.glob(FOLD_DIR + '/fold[0-9].pt'))") == 1
        student = student.replace("FOLDS = sorted(glob.glob(FOLD_DIR + '/fold[0-9].pt'))",
                                  "FOLDS = sorted(glob.glob(FOLD_DIR + '/fold[0-9].pt') + glob.glob(FOLD_DIR2 + '/fold[0-9].pt'))")
    stage0 = STAGE0.replace("__TIMEOUT__", str(BUDGET + 2400))     # + gold check in commit mode (≤1800 s) + model load
    base_txt = ("goodpjw2008's public 0.958 notebook v27 without its Part B reader (Raptor slot = 0.65 r384 + 0.35 r224 mirror; compact = "
                "heliosli's R6 with flip TTA; heliosli's per-finding Raptor weights)" if BASE_ARG == "g958" else
                "varshithakolupuri's R7 + Raptor-r224 public notebook (Raptor slot = 0.65 r384 + 0.35 r224; R7 0.5)" if V224 else
                "heliosli R7 public notebook (Raptor 0.5 + compact R7 0.5, bundle `8c7b666e27dc0e8f`)")
    extra = ([f"our ConvNeXt-T 2.5D student (4 folds, weight {W}, slot {SLOT})"] if W > 0 else []) + (
        [f"R7 compact with heliosli's anatomical flip TTA at eval stride {FSTRIDE}"] if FLIP else []) + (
        [f"nartaa's public S6 ConvNeXt (all data) inside the {S6F_SLOT} slot, weight {S6F}"] if S6F > 0 else []) + (
        [(f"heliosli's R7" if C2 == "r7" else "the public B-leg ConvNeXt-S (R6 fine-tuned on soft-v2 labels; no flip, no stackers, stride 2)")
         + f" as a second compact model after S6F, weight {R7ADD} inside the compact model"] if R7ADD > 0 else []) + (
        [f"R6 eval stride {R6STRIDE}"] if R6STRIDE else [])
    title = (f"# RSNA Knee: R7 variant ({NAME})\n\n{base_txt}, plus: {'; '.join(extra) or 'nothing'}. Built by `build_r7murid.py`.\n")
    stu_cells = [code(student), code(stage0), code(c3)] if W > 0 else [cells[i_st1]]
    if S6F > 0:                                                    # tahap S6F setelah R7 (GPU sudah bebas), sebelum blend
        mid = mid + [code(s6f_script()), code(S6F_RUN)]
    if R7ADD > 0:                                                  # R7 setelah S6F: S6F (terbukti di papan) tak pernah terpotong oleh R7
        r7run = R7_RUN if C2 == "r7" else R7_RUN.replace("heliosli's public R7 compact model (bundle 8c7b666e, its setup + controller cells unchanged,",
                                                         "public B-leg compact model (R6 fine-tuned on soft-v2 labels; R6 setup + controller, arm E_prior,")
        mid = mid + [code(r7_stage_script() if C2 == "r7" else bleg_stage_script()), code(r7run), code(R7_MERGE.replace("__WR7__", repr(R7ADD)))]
    nb["cells"] = [md(title)] + cells[1:i_st1] + stu_cells + mid + [code(c6)]
    for k, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            s = src(c)
            compile(s.split("\n", 1)[1] if s.startswith("%%") else s, f"<cell{k}>", "exec")
    out = ROOT / f"kaggle_kernel_{NAME}"
    out.mkdir(exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(BASE_META.read_text(encoding="utf-8"))
    for k in ("id_no",):
        meta.pop(k, None)
    meta.update(id=f"dimaspashaakrilian/{SLUG}", title="RSNA Knee " + NAME, code_file=f"{SLUG}.ipynb", is_private=True,
                dataset_sources=meta["dataset_sources"] + [d for d in EXTRA_DS if W > 0 and d not in meta["dataset_sources"]]
                + ([S6F_DS] if S6F > 0 and S6F_DS not in meta["dataset_sources"] else [])
                + (BLEG_DS if R7ADD > 0 and C2 == "bleg" else [])
                + ([EXT_DS] if EW > 0 and EXT_DS not in meta["dataset_sources"] else []))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def check_leg():
    """Uji matematika kaki murid di luar Kaggle: tanpa file / lengkap / sebagian NaN / terlalu banyak NaN."""
    labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
              "Baker's", 'Contusion', 'Fracture']
    wt = W if W > 0 else 0.2                                       # w=0 (mis. flip saja): logika tetap diuji dgn 0,2
    leg = LOAD.replace("__W__", repr(wt)).replace("__SLOT__", "final") + APPLY_FINAL
    pct = lambda v: pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

    def pct_on(v, ok):
        r = np.full(len(v), np.nan)
        if ok.any():
            r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
        return r

    rng = np.random.default_rng(0)
    n = 40
    ids = [f"1.2.{i}" for i in range(n)]
    base = rng.random((n, 12))
    with tempfile.TemporaryDirectory() as d:
        def run(stu):
            p = Path(d) / "student_test_preds.csv"
            if p.exists():
                p.unlink()
            if stu is not None:
                stu.to_csv(p, index=False)
            ns = dict(np=np, pd=pd, json=json, pct=pct, pct_on=pct_on, labels=labels, test_ids=ids, WORK=Path(d), out=base.copy())
            exec(compile(leg, "leg", "exec"), ns)
            return ns["out"], ns["murid"]
        o, m = run(None)                                               # tanpa file -> R7 murni
        assert not m["used"] and np.array_equal(o, base)
        stu = pd.DataFrame(rng.random((n, 12)), columns=['cnxt_' + c for c in labels])
        stu.insert(0, "StudyInstanceUID", ids)
        o, m = run(stu.sample(frac=1, random_state=1))                 # lengkap, urutan acak
        exp = np.column_stack([(1 - wt) * pct(base[:, j]) + wt * pct(stu.iloc[:, j + 1]) for j in range(12)])
        assert m["used"] and np.allclose(o, exp) and ((o >= 0) & (o <= 1)).all()
        s2 = stu.copy()
        s2.iloc[3:5, 1:] = np.nan                                      # 2/40 = 5 % tak terskor -> < 97 % -> R7 murni
        o, m = run(s2)
        assert not m["used"] and m["scored"] == 38 and np.array_equal(o, base)
        big = 100                                                      # 1 % tak terskor -> leg dipakai, baris itu = rank R7
        ids_b, base_b = [f"9.{i}" for i in range(big)], rng.random((big, 12))
        s3 = pd.DataFrame(rng.random((big, 12)), columns=['cnxt_' + c for c in labels])
        s3.insert(0, "StudyInstanceUID", ids_b)
        s3.iloc[7, 1:] = np.nan
        p = Path(d) / "student_test_preds.csv"
        s3.to_csv(p, index=False)
        ns = dict(np=np, pd=pd, json=json, pct=pct, pct_on=pct_on, labels=labels, test_ids=ids_b, WORK=Path(d), out=base_b.copy())
        exec(compile(leg, "leg", "exec"), ns)
        assert ns["murid"]["used"] and np.isclose(ns["out"][7, 0], pct(base_b[:, 0])[7])
    # slot=raptor: muat + campur ke rr di dalam loop (baris NaN murid -> rr tetap)
        ns = dict(np=np, pd=pd, json=json, pct=pct, pct_on=pct_on, labels=labels, test_ids=ids_b, WORK=Path(d))
        exec(compile(LOAD.replace("__W__", repr(wt)).replace("__SLOT__", "raptor"), "load", "exec"), ns)
        v = s3[["cnxt_" + c for c in labels]].to_numpy()
        ok = np.isfinite(v).all(axis=1)
        for var, tmpl in (("rr", IN_LOOP), ("ro", IN_LOOP_COMPACT)):     # slot raptor -> rr, slot compact -> ro
            loop = f"for j in range(12):\n    {var} = pct(base_b[:, j])\n" + tmpl + f"    got.append({var})\n"
            ns.update(base_b=base_b, got=[])
            exec(compile(loop, "loop", "exec"), ns)
            for j in range(12):
                want = np.where(ok, (1 - wt) * pct(base_b[:, j]) + wt * pct_on(v[:, j], ok), pct(base_b[:, j]))
                assert np.allclose(ns["got"][j], want), var
    print(f"cek kaki murid OK (w={wt}): tanpa file, lengkap, 5 % NaN -> fallback, 1 % NaN -> baris itu rank R7, slot raptor & compact")


def check_s6f():
    """Uji kaki S6F di luar Kaggle: tanpa file / lengkap / 5 % NaN / korelasi rendah dgn R7; campuran ro sebelum murid."""
    labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
              "Baker's", 'Contusion', 'Fracture']
    ws = S6F if S6F > 0 else 0.35
    pct = lambda v: pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

    def pct_on(v, ok):
        r = np.full(len(v), np.nan)
        if ok.any():
            r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
        return r

    rng = np.random.default_rng(1)
    n = 120
    ids = [f"7.{i}" for i in range(n)]
    r7 = rng.random((n, 12))
    ours = pd.DataFrame(r7, index=ids, columns=labels)
    load = S6F_LOAD.replace("__WS6F__", repr(ws))
    with tempfile.TemporaryDirectory() as d:
        def run(df):
            p = Path(d) / "s6f_raw.csv"
            if p.exists():
                p.unlink()
            if df is not None:
                df.to_csv(p, index=False)
            ns = dict(np=np, pd=pd, json=json, pct=pct, pct_on=pct_on, labels=labels, test_ids=ids, WORK=Path(d), ours=ours,
                      our_fail=np.zeros(n, bool), got=[])
            exec(compile(load, "s6f_load", "exec"), ns)
            exec(compile("for j, c in enumerate(labels):\n    ro = pct_on(ours[c].values, ~our_fail)\n" + S6F_IN_LOOP
                         + "    got.append(ro)\n", "s6f_loop", "exec"), ns)
            return ns
        ns = run(None)                                             # tanpa file -> ro = R7
        assert not ns["s6f"]["used"] and all(np.allclose(g, pct(r7[:, j])) for j, g in enumerate(ns["got"]))
        good = pd.DataFrame(r7 + 0.3 * rng.random((n, 12)), columns=labels)   # berkorelasi dgn R7
        good.insert(0, "StudyInstanceUID", ids)
        ns = run(good.sample(frac=1, random_state=2))              # lengkap, urutan acak -> dipakai
        assert ns["s6f"]["used"] and ns["s6f"]["rho_r7_mean"] > 0.5
        for j in range(12):
            want = (1 - ws) * pct(r7[:, j]) + ws * pct(good.iloc[:, j + 1].values)
            assert np.allclose(ns["got"][j], want)
        g2 = good.copy(); g2.iloc[:6, 1:] = np.nan                 # 6/120 = 5 % tak terskor -> tanpa kaki
        assert not run(g2)["s6f"]["used"]
        ns = run(good)                                             # slot raptor: rumus sama pada rr
        ns.update(rr_got=[])
        exec(compile("for j, c in enumerate(labels):\n    rr = pct_on(ours[c].values, ~our_fail)\n" + S6F_IN_LOOP_RAPTOR
                     + "    rr_got.append(rr)\n", "s6f_loop_rr", "exec"), ns)
        assert all(np.allclose(a, b) for a, b in zip(ns["rr_got"], ns["got"]))
        bad = good.copy(); bad[labels] = rng.random((n, 12))       # tak berkorelasi (rho ~0) -> tanpa kaki
        nb_ = run(bad)
        assert not nb_["s6f"]["used"] and nb_["s6f"]["rho_r7_mean"] < 0.5
    print(f"cek kaki S6F OK (w={ws}): tanpa file, lengkap, 5 % NaN -> tanpa kaki, rho rendah -> tanpa kaki")


def check_r7merge():
    """Uji gabungan R6+R7 di luar Kaggle: R7 lengkap -> rata-rata rank; R7 < 97 % -> R6 saja; studi yg hanya diskor satu model."""
    labels = ['ACL', 'MCL']
    rng = np.random.default_rng(2)
    n = 100
    ids = [f"1.2.5.{i}" for i in range(n)]   # UID ber-titik banyak (angka desimal akan terbaca float)
    with tempfile.TemporaryDirectory() as d:
        D = Path(d)
        (D / "test.csv").write_text("StudyInstanceUID\n" + "\n".join(ids) + "\n")
        def write(name, pdir, rows):
            df = pd.DataFrame(rng.random((n, 2)), columns=labels)
            df.insert(0, "StudyInstanceUID", ids)
            df.to_csv(D / name, index=False)
            (D / pdir / "worker0").mkdir(parents=True, exist_ok=True)
            df.iloc[rows].to_csv(D / pdir / "worker0" / "predictions.csv", index=False)
            return df
        r6 = write("ours_r6.csv", "p6", list(range(n)))
        for rows7, want_used in ((list(range(1, n)), True), (list(range(90)), False)):
            r7 = write("ours_r7.csv", "p7", rows7)
            ns = dict(DATA=D, WORK=D, R6_PRED=D / "p6", R7_PRED=D / "p7", R7_OK=True)
            exec(compile(R7_MERGE.replace("__WR7__", "0.3").replace("Path('/kaggle/temp' if Path('/kaggle/temp').is_dir() else '/tmp')", f"Path('{d}')"), "merge", "exec"), ns)
            got = pd.read_csv(D / "ours_sub.csv")
            assert ns["R7_LEG"]["used"] == want_used, ns["R7_LEG"]
            if want_used:
                ok7 = np.arange(n) >= 1
                a = r6["ACL"].rank(pct=True).values
                b = np.full(n, np.nan); b[ok7] = r7["ACL"].values[ok7].argsort().argsort() + 1.0; b[ok7] /= ok7.sum()
                assert np.isclose(got["ACL"].values[0], a[0]) and np.allclose(got["ACL"].values[1:], 0.7 * a[1:] + 0.3 * b[1:])
            else:
                assert np.allclose(got["ACL"].values, r6["ACL"].values)
    print("cek gabungan R6+R7 OK: lengkap -> rata-rata rank (studi tanpa R7 = rank R6), R7 < 97 % -> R6 saja")


if __name__ == "__main__":
    out = build()
    check_leg()
    check_s6f()
    check_r7merge()
    print("ditulis:", out, "| id dimaspashaakrilian/" + SLUG, "| w", W, "| budget", BUDGET, "s | datasets +", EXTRA_DS)
