#!/usr/bin/env python3
"""S6+: notebook publico S6 do heliosli (Raptor 0,7 + compact ConvNeXt-S 4 folds 0,3 = 0,954) estendido com pernas OPCIONAIS e deadline-seguras.

  reader=1 : perna goodpjw 2.5D ConvNeXt (3 folds; codigo identico ao Apex) FUNDIDA na perna Raptor com os pesos por-finding do Apex (TARGET_WEIGHTS) -> 'Apex' no lugar do Raptor;
             roda ANTES do estagio compact (timm do container, como no Apex validado).
  of=1     : perna OrthoFoundation-L fold0 (heliosof) com o supervisor v4.1 (of_stage_v4.py) como ULTIMO estagio, com STOP/KILL absolutos desde NB_T0
             (so usa o tempo que sobra; se nao sobrar, a perna e ignorada e o resultado = S6[+reader]).
  Blend final por finding = media ponderada de ranks (percentis) das pernas disponiveis (pesos renormalizados por estudo): S6 original = Raptor .7 / compact .3.
  noraptor=1: variante LIMPA (sem Raptor/OAI): so compact + reader (+OF opcional); pesos wc (compact, padrao .65) e wd (reader, padrao .35).
  lenient=1: o controlador do compact NAO derruba o notebook quando >3 % de fallbacks (cai para o rank Raptor), em vez de ERROR sem pontuacao.

Pemakaian:
  python3 build_s6plus.py name=s6reader reader=1 wr=0.70 wc=0.30 rscale=0.5      # reader conservador (pesos Apex x 0,5)
  python3 build_s6plus.py name=s6plus reader=1 of=1 wr=0.55 wc=0.30 wo=0.15 stop_h=7.9 kill_h=8.25
  python3 build_s6plus.py name=s6plusr pseudo=160 reader=1 of=1 wr=0.55 wc=0.30 wo=0.15 t0=now stop_h=0.4 kill_h=0.6 deadline=1.3   # rehearsal (log legivel)
"""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from of_stage_v4 import DATA_PRESET_PROD, OF4_CELL  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "s6plus")
READER = args.get("reader", "0") == "1"
OF = args.get("of", "0") == "1"
PSEUDO = int(args.get("pseudo", "0"))
NORAPTOR = args.get("noraptor", "0") == "1"
if NORAPTOR:
    READER = True
    WR = float(args.get("wd", "0.35"))                      # peso do reader (na variante limpa 'wr' nao se usa)
    WC = float(args.get("wc", "0.65"))
else:
    WR = float(args.get("wr", "0.70"))
    WC = float(args.get("wc", "0.30"))
WO = float(args.get("wo", "0.15" if OF else "0.0"))
if not OF:
    WO = 0.0
assert abs(WR + WC + WO - 1.0) < 1e-9, (WR, WC, WO)
STOP_H = args.get("stop_h", "7.9")
KILL_H = args.get("kill_h", "8.25")
T0MODE = args.get("t0", "nb")              # nb = NB_T0 do S6 (producao) | now = relogio local da celula OF (rehearsal)
DEADLINE = args.get("deadline", "")        # troca DEADLINE_HOURS do compact (S6 original 8.3)
LENIENT = args.get("lenient", "1") == "1"
RSCALE = float(args.get("rscale", "1.0"))        # escala dos pesos por-finding do reader (Apex TARGET_WEIGHTS); 0.5 = reader conservador
NB = Path(args.get("nb", ROOT / "kaggle_base" / "s6_heliosli_public_20261011.ipynb"))
BASE_APEX = ROOT / "kaggle_base" / "apex_grandmaster_stack_20261008.ipynb"

PSEUDO_CELL = r'''
import os as _po, random as _prnd, shutil as _psh
import pandas as _ppd
_COMP = next(p for p in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]
             if _po.path.exists(p + "/train.csv"))
_PS = "/kaggle/working/pseudo"
_LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
_ptr = _ppd.read_csv(_COMP + "/train.csv")
_pgold = set(_ptr[_ptr[_LAB].notna().all(axis=1)]["StudyInstanceUID"])
_pids = [s for s in _ptr["StudyInstanceUID"].tolist() if s not in _pgold]
_prnd.Random(7).shuffle(_pids)
_pids = _pids[:__PSEUDO__]
_psh.rmtree(_PS, ignore_errors=True)
_po.makedirs(_PS + "/test_series")
for _s in _pids:
    _po.symlink(f"{_COMP}/train_series/{_s}", f"{_PS}/test_series/{_s}")
_ptrs = _ppd.read_csv(_COMP + "/train_series.csv")
_pcols = [c for c in _ppd.read_csv(_COMP + "/test_series.csv", nrows=2).columns if c in _ptrs.columns]
_ptrs[_ptrs["StudyInstanceUID"].isin(set(_pids))][_pcols].to_csv(_PS + "/test_series.csv", index=False)
_ocols = list(_ppd.read_csv(_COMP + "/test.csv", nrows=1).columns)
_ppd.DataFrame({c: (_pids if c == "StudyInstanceUID" else None) for c in _ocols}).to_csv(_PS + "/test.csv", index=False)
_ssn = _ppd.read_csv(_COMP + "/sample_submission.csv", nrows=1)
_ppd.DataFrame({c: (_pids if c == "StudyInstanceUID" else 0.5) for c in _ssn.columns}).to_csv(_PS + "/sample_submission.csv", index=False)   # mesmas colunas, ids do pseudo-teste
print("[PSEUDO] studies", len(_pids), flush=True)
'''

MEMMON_CELL = r'''
# ---- monitor de RAM (log): MemAvailable + maiores processos a cada 120 s (diagnostico de OOM; inofensivo) ----
import threading as _mt, time as _mtime, subprocess as _msp
def _memmon():
    while True:
        try:
            _a = [l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]
            _top = _msp.run("ps -eo rss,comm --sort=-rss | head -4 | tr '\\n' ';'", shell=True, capture_output=True, text=True, timeout=20).stdout
            print(f"[MEMMON] MemAvailable {int(_a) / 1048576:.1f} GB | top rss_kB,comm: {_top}", flush=True)
        except Exception:
            pass
        _mtime.sleep(120)
_mt.Thread(target=_memmon, daemon=True).start()
'''

READER_CELL = r'''
# ---- perna reader (goodpjw 2.5D ConvNeXt MIL, 3 folds): mesmo codigo do Apex (own_src/infer.py), roda ANTES do compact (timm do container) ----
import os as _r_os, sys as _r_sys, glob as _r_glob, subprocess as _r_sp, time as _r_time, traceback as _r_tb
_R_T0 = _r_time.time()
READER_OK = False
try:
    def _r_find(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
            for d in sorted(_r_glob.glob(pat)):
                if _r_os.path.exists(_r_os.path.join(d, name)):
                    return d.rstrip("/")
        raise FileNotFoundError(f"{name} not found in /kaggle/input")
    _r_assets = _r_find("cnxt_v0_fold0.pt")
    _r_data = "/kaggle/working/pseudo" if _r_os.path.exists("/kaggle/working/pseudo/test_series.csv") else _r_find("test_series.csv")
    _r_ck = sorted(_r_glob.glob(_r_assets + "/cnxt_v0_fold*.pt"))
    _r_env = dict(_r_os.environ)
    try:
        _o = _r_sp.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        _used = {int(a.split(",")[0]): int(a.split(",")[1]) for a in _o.strip().splitlines() if "," in a}
        _r_env["CUDA_VISIBLE_DEVICES"] = str(min(_used, key=_used.get)) if _used else "0"
    except Exception:
        pass
    print(f"[READER] {len(_r_ck)} checkpoints | data {_r_data}", flush=True)
    _r_sp.run([_r_sys.executable, "/kaggle/working/own_src/infer.py", _r_data, "test", "/kaggle/working/_own.csv"] + _r_ck, check=True, env=_r_env)
    READER_OK = _r_os.path.isfile("/kaggle/working/_own.csv")
    print(f"[READER] done {_r_time.time() - _R_T0:.0f}s ok={READER_OK}", flush=True)
except Exception as _e:
    _r_tb.print_exc()
    print(f"[READER EXCEPTION] {type(_e).__name__}: {_e} -> perna ignorada (Raptor puro)", flush=True)
'''

BLEND_CELL = r'''
# Stage 3 (S6+): per-finding rank blend; pernas: Raptor (+reader fundido, pesos Apex), compact (ours), OrthoFoundation (opcional).
# Pesos fixos ANTES de qualquer execucao (nada ajustado em rotulos): Raptor __WR__, compact __WC__, OF __WO__ (reader: pesos Apex x __RSCALE__); estudo sem uma perna usa as demais (pesos renormalizados).
import csv, os
import numpy as np
import pandas as pd
W_RAPTOR, W_OURS, W_OF = __WR__, __WC__, __WO__
READER_W = {"Baker's": 0.35, "Contusion": 0.30, "Medial OA": 0.25, "ACL": 0.25, "Medial Meniscus": 0.20, "MCL": 0.20,
            "Lateral Meniscus": 0.08, "Fracture": 0.08, "Lateral OA": 0.08, "PF OA": 0.08, "Effusion": 0.04, "Synovitis": 0.03}
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]
raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})
rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})
for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub')):
    assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'
    assert set(labels) <= set(df.columns), nm + ': columns differ'
ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]
raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]
rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]
n = len(test_ids)
rap_fail = (raw.values == 0.5).all(axis=1)             # Raptor leaves a failed study at 0.5 in every column
scored = set()                                          # ours fills a failed study with the column mean: read what was scored
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                scored.add(r['StudyInstanceUID'])
our_fail = np.array([u not in scored for u in test_ids])
if rap_fail.sum() > 0.03 * n:
    raise RuntimeError('Raptor failed on %d of %d studies' % (rap_fail.sum(), n))

reader = None
if os.path.isfile('/kaggle/working/_own.csv'):
    try:
        _rd = pd.read_csv('/kaggle/working/_own.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)[labels]
        reader = _rd if _rd.notna().all().all() else None
        print('[BLEND] reader leg', 'OK' if reader is not None else 'incomplete -> ignored', flush=True)
    except Exception as _e:
        print('[BLEND] reader leg unreadable:', repr(_e)[:120], flush=True)
ofl = None
if W_OF > 0 and os.path.isfile('/kaggle/working/_leg_of.csv'):
    try:
        ofl = pd.read_csv('/kaggle/working/_leg_of.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)[labels]
        print('[BLEND] OF leg coverage %.3f' % (1.0 - float(ofl.isna().all(axis=1).mean())), flush=True)
    except Exception as _e:
        print('[BLEND] OF leg unreadable:', repr(_e)[:120], flush=True)


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values


out = np.zeros((n, len(labels)))
rho_rr, rho_ro = {}, {}
for j, c in enumerate(labels):
    rr = pct(raw[c].values)
    if reader is not None:                              # Raptor leg = Apex fusion (percentile ranks, per-finding reader weight)
        w = READER_W.get(c, 0.15) * RSCALE
        rr = pct((1.0 - w) * rr + w * pct(reader[c].values))
    ro = pct(ours[c].values)
    num = np.where(rap_fail, 0.0, W_RAPTOR * rr) + np.where(our_fail, 0.0, W_OURS * ro)
    den = np.where(rap_fail, 0.0, W_RAPTOR) + np.where(our_fail, 0.0, W_OURS)
    if ofl is not None:
        ov = ofl[c].values
        ok = ~np.isnan(ov)
        po = np.zeros(n)
        po[ok] = pct(ov[ok])
        num = num + np.where(ok, W_OF * po, 0.0)
        den = den + np.where(ok, W_OF, 0.0)
    b = np.where(den > 0, num / np.maximum(den, 1e-12), 0.5)
    out[:, j] = b
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)
if rho_rr and min(rho_rr.values()) < 0.99:
    raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'of': W_OF}, 'reader': reader is not None, 'of_leg': ofl is not None, 'rows': n,
         'raptor_failed': int(rap_fail.sum()), 'ours_failed': int(our_fail.sum()), 'raptor_seconds': round(RAPTOR_SECONDS, 1),
         'total_seconds': round(time.time() - NB_T0, 1), 'spearman_raptor_raw_vs_sub': rho_rr, 'spearman_raptor_vs_ours': rho_ro}
(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps({k: blend[k] for k in ('weights', 'reader', 'of_leg', 'rows', 'raptor_failed', 'ours_failed', 'total_seconds')}), flush=True)
display(sub.head())
'''


NORAPTOR_CELL = r'''
# variante LIMPA: sem estagio Raptor/OAI (so compact + reader). Define NB_T0 e RAPTOR_SECONDS para o resto do notebook S6.
import os, sys, time, subprocess
NB_T0 = time.time()
RAPTOR_SECONDS = 0.0
'''

BLEND_CLEAN = r'''
# Stage 3 (S6 limpo): per-finding rank blend de compact (ours) + reader (+OF opcional), SEM Raptor/OAI. Pesos fixos: compact __WC__, reader __WR__, OF __WO__.
import csv, os
import numpy as np
import pandas as pd
W_OURS, W_RD, W_OF = __WC__, __WR__, __WO__
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]
assert sorted(ours['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(ours.columns), 'ours: ids/colunas'
ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]
n = len(test_ids)
scored = set()
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                scored.add(r['StudyInstanceUID'])
our_fail = np.array([u not in scored for u in test_ids])
reader = None
if os.path.isfile('/kaggle/working/_own.csv'):
    _rd = pd.read_csv('/kaggle/working/_own.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)[labels]
    reader = _rd if _rd.notna().all().all() else None
assert reader is not None, 'perna reader indisponivel: a variante limpa precisa de compact + reader'
ofl = None
if W_OF > 0 and os.path.isfile('/kaggle/working/_leg_of.csv'):
    ofl = pd.read_csv('/kaggle/working/_leg_of.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)[labels]


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values


out = np.zeros((n, len(labels)))
for j, c in enumerate(labels):
    ro, rd = pct(ours[c].values), pct(reader[c].values)
    num = np.where(our_fail, 0.0, W_OURS * ro) + W_RD * rd
    den = np.where(our_fail, 0.0, W_OURS) + W_RD
    if ofl is not None:
        ov = ofl[c].values
        ok = ~np.isnan(ov)
        po = np.zeros(n)
        po[ok] = pct(ov[ok])
        num = num + np.where(ok, W_OF * po, 0.0)
        den = den + np.where(ok, W_OF, 0.0)
    out[:, j] = num / np.maximum(den, 1e-12)
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
blend = {'variant': 'clean_no_raptor', 'weights': {'compact': W_OURS, 'reader': W_RD, 'of': W_OF}, 'of_leg': ofl is not None, 'rows': n,
         'ours_failed': int(our_fail.sum()), 'total_seconds': round(time.time() - NB_T0, 1)}
(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(blend), flush=True)
display(sub.head())
'''


def fill_of(src):
    t0 = "NB_T0" if T0MODE == "nb" else "_time.time()"
    rep = {"__WD__": "/kaggle/working", "__NPROC__": "4", "__CHUNK__": "8", "__STOP_H__": STOP_H, "__KILL_H__": KILL_H, "__MINCOV__": "0.10",
           "__MAXCHUNKS__": "1", "__RSSMAX__": "7.5", "__INJECT__": "0", "__INJECT_AFTER__": "360", "__POLL__": "10", "__PRINTEVERY__": "300",
           "__MEMPERW__": "6.5", "__MEMFREE__": "7.0", "__MMAP__": "1", "__NOPIN__": "1", "__EXTRAENV__": "{}", "__DISKPER__": "0.42",
           "__DISKBASE__": "1.5", "__FORCEW__": "0", "__SETUP__": "1", "__T0EXPR__": t0,
           "__DATA_PRESET__": "_of_data = str(DATA)"}
    for k, v in rep.items():
        src = src.replace(k, v)
    return src


nb = json.loads(NB.read_text(encoding="utf-8"))
cells = copy.deepcopy(nb["cells"])
assert len(cells) == 7
if PSEUDO and not NORAPTOR:
    s2 = "".join(cells[2]["source"])
    old2 = 'cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",'
    assert s2.count(old2) == 1
    cells[2]["source"] = s2.replace(old2, 'cands = ["/kaggle/working/pseudo", "/kaggle/input/competitions/rsna-knee-abnormality-detection",').splitlines(keepends=True)
s4 = "".join(cells[4]["source"])
old4 = "DATA = find_mount(COMPETITION, 'test.csv')"
assert s4.count(old4) == 1
if PSEUDO:
    s4 = s4.replace(old4, "DATA = Path('/kaggle/working/pseudo') if Path('/kaggle/working/pseudo/test.csv').is_file() else find_mount(COMPETITION, 'test.csv')")
if DEADLINE:
    assert "DEADLINE_HOURS = 8.3" in s4
    s4 = s4.replace("DEADLINE_HOURS = 8.3", f"DEADLINE_HOURS = {DEADLINE}", 1)
cells[4]["source"] = s4.splitlines(keepends=True)
if LENIENT:
    s5 = "".join(cells[5]["source"])
    old5 = "'--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS)], env=env).returncode"
    assert s5.count(old5) == 1
    s5 = s5.replace(old5, "'--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS), '--max-fallback-fraction', '0.6'], env=env).returncode")
    cells[5]["source"] = s5.splitlines(keepends=True)


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.lstrip("\n").splitlines(keepends=True)}


new_cells = cells[:2]
if PSEUDO:
    p = PSEUDO_CELL.replace("__PSEUDO__", str(PSEUDO))
    compile(p, "pseudo", "exec")
    new_cells.append(code(p))
compile(MEMMON_CELL, "memmon", "exec")
new_cells.append(code(MEMMON_CELL))
if NORAPTOR:
    compile(NORAPTOR_CELL, "noraptor", "exec")
    new_cells.append(code(NORAPTOR_CELL))
else:
    new_cells += cells[2:4]                                  # Raptor script + stage 1
if READER:
    apex = json.loads(BASE_APEX.read_text(encoding="utf-8"))["cells"]
    for i in (5, 6, 7, 8):                              # own_src: makedirs + writefile preprocess/knee/infer (codigo do reader, identico ao Apex)
        new_cells.append(copy.deepcopy(apex[i]))
    compile(READER_CELL, "reader", "exec")
    new_cells.append(code(READER_CELL))
new_cells += cells[4:6]                                  # compact setup + run
if OF:
    of_cell = fill_of(OF4_CELL)
    compile(of_cell, "of", "exec")
    inner = re.search(r'_OF_WORKER_SRC = r"""(.*?)"""\n', of_cell, re.S).group(1)
    compile(inner, "of-worker", "exec")
    new_cells.append(code(of_cell))
blend = (BLEND_CLEAN if NORAPTOR else BLEND_CELL).replace("__WR__", repr(WR)).replace("__WC__", repr(WC)).replace("__WO__", repr(WO)).replace("__RSCALE__", repr(RSCALE))
compile(blend, "blend", "exec")
new_cells.append(code(blend))
nb["cells"] = new_cells
slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
ds = ["heliosli/rsna-knee-compact-v1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"] + ([] if NORAPTOR else ["nartaa/rsna-knee-publication-swa-weights-20261007"])
if READER:
    ds.append("goodpjw2008/rsna-knee-2-5d-convnext-reader")
if OF:
    ds.append("dimaspashaakrilian/rsna-knee-ultralytics-cfg")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ds, "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| reader", READER, "| of", OF, "| pesos", (WR, WC, WO), "| pseudo", PSEUDO, "| deadline compact", DEADLINE or "8.3",
      "| OF stop/kill", STOP_H, KILL_H, "t0", T0MODE, "| lenient", LENIENT, "| cells", len(new_cells), "| datasets", ds)
