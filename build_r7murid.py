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

Pakai: python3 build_r7murid.py [w=0.2] [name=r7murid] [budget=4500] [base=r7|v224]
  base=v224: basis = notebook varshitha (R7 + Raptor-r224 0,35 di slot Raptor; rank naik 233->156 di blok 0,956) -> nama default r7m224.
"""
import copy
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
V224 = args.get("base", "r7") == "v224"
NAME = args.get("name", "r7m224" if V224 else "r7murid")
BUDGET = int(args.get("budget", "4500"))
assert 0.0 <= W <= 0.5 and 600 <= BUDGET <= 7200, (W, BUDGET)
_b = "r7_r224_varshitha_20261013" if V224 else "r7_heliosli_0957_20261013"
BASE = ROOT / "kaggle_base" / f"{_b}.ipynb"
BASE_META = ROOT / "kaggle_base" / f"{_b}.meta.json"
SLUG = "rsna-knee-" + NAME
EXTRA_DS = ["dimaspashaakrilian/rsna-knee-student-folds", "goodpjw2008/rsna-knee-2-5d-convnext-reader"]
MB, ME = "# ---- [murid] kaki ke-3", "# ---- [murid] selesai ----"

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
FOLD_DIR = mount('rsna-knee-student-folds', 'fold0.pt')
sys.path.insert(0, GOODPJW)
from knee import LABELS, KneeNet, StudyDataset, build_study_table
__DECODE__
FOLDS = sorted(glob.glob(FOLD_DIR + '/fold[0-9].pt'))
assert len(FOLDS) >= 4, FOLDS
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

LEG = r'''
# ---- [murid] kaki ke-3 (build_r7murid): our ConvNeXt-T student, weight fixed before any run; ranked over its own scored rows;
# a study it did not score keeps the R7 blend's rank; fewer than 97 % scored -> no student leg (R7 only, printed)
W_MURID = __W__
murid = {'w': W_MURID, 'used': False}
_sp = WORK / 'student_test_preds.csv'
if W_MURID > 0 and _sp.is_file():
    _s = pd.read_csv(_sp, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    _cols = ['cnxt_' + c for c in labels]
    if set(_cols) <= set(_s.columns):
        _v = _s.reindex(test_ids)[_cols].to_numpy(dtype=np.float64)
        _ok = np.isfinite(_v).all(axis=1)
        murid['scored'] = int(_ok.sum())
        if _ok.mean() >= 0.97:
            for j, c in enumerate(labels):
                _rb = pct(out[:, j])
                out[:, j] = np.where(_ok, (1.0 - W_MURID) * _rb + W_MURID * pct_on(_v[:, j], _ok), _rb)
            murid['used'] = True
print('MURID_LEG', json.dumps(murid), flush=True)
# ---- [murid] selesai ----
'''


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
    c6 = c6.replace(anchor, LEG.replace("__W__", repr(W)) + anchor).replace(rec, rec[:-1] + ", 'murid': murid}")
    student = STUDENT.replace("__BUDGET__", str(BUDGET)).replace("__DECODE__", DECODE)
    stage0 = STAGE0.replace("__TIMEOUT__", str(BUDGET + 2400))     # + gold check in commit mode (≤1800 s) + model load
    base_txt = ("varshithakolupuri's R7 + Raptor-r224 public notebook (Raptor slot = 0.65 r384 + 0.35 r224; R7 0.5)" if V224 else
                "heliosli R7 public notebook (Raptor 0.5 + compact R7 0.5, bundle `8c7b666e27dc0e8f`)")
    title = (f"# RSNA Knee: R7 + Raptor + our student ({NAME})\n\n{base_txt} unchanged, plus a third per-finding rank leg: "
             f"our ConvNeXt-T 2.5D student (4 folds, weight {W}). Built by `build_r7murid.py`.\n")
    nb["cells"] = ([md(title)] + cells[1:i_st1] + [code(student), code(stage0), code(c3)] + cells[i_st1 + 1:i_bl] + [code(c6)])
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
                dataset_sources=meta["dataset_sources"] + [d for d in EXTRA_DS if d not in meta["dataset_sources"]])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def check_leg():
    """Uji matematika kaki murid di luar Kaggle: tanpa file / lengkap / sebagian NaN / terlalu banyak NaN."""
    labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
              "Baker's", 'Contusion', 'Fracture']
    leg = LEG.replace("__W__", repr(W))
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
        exp = np.column_stack([(1 - W) * pct(base[:, j]) + W * pct(stu.iloc[:, j + 1]) for j in range(12)])
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
    print(f"cek kaki murid OK (w={W}): tanpa file, lengkap, 5 % NaN -> fallback, 1 % NaN -> baris itu rank R7")


if __name__ == "__main__":
    out = build()
    check_leg()
    print("ditulis:", out, "| id dimaspashaakrilian/" + SLUG, "| w", W, "| budget", BUDGET, "s | datasets +", EXTRA_DS)
