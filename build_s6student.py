#!/usr/bin/env python3
"""Kloning kernel `kaggle_kernel_s6prb` (S6 = Raptor 0,7 + compact 0,3, PROBIT + SYNRES) -> `kaggle_kernel_s6student`
dengan PENAMBAHAN leg murid (hasil kernel student-infer) ke blend. Id baru: alverlfarfaraway/rsna-knee-s6student.

Leg murid:
  - dataset `alverlfarfaraway/rsna-knee-student-preds` (mount via kernel-metadata) berisi `student_test_preds.csv`
    (StudyInstanceUID + cnxt_<finding> + dino_<finding>; kolom tag yang absen = model itu tak dipakai saat infer);
  - file dicari via glob `/kaggle/input/**/student_test_preds.csv` (override penguji lokal: env `S6S_STU_PAT`);
  - TIDAK ADA / tidak lengkap / tak terbaca -> FALLBACK: submission = S6 biasa + print warning (TIDAK error);
  - per finding: m = 0.5*rankpct(cnxt_c) + 0.5*rankpct(dino_c)  [rankpct = pct() lokal kolom itu; 1 perna hilang -> renormalizado];
  - blend akhir: b_final = (1 - __WMURID__)*rankpct(b_s6) + __WMURID__*rankpct(m), default WMURID = 0.35, lalu clip [0,1];
  - diterapkan SEBELUM residual Synovitis (synres = "pos-blend"), di dalam cell Stage 3 -> tanpa cell baru.

Validasi lokal (jalan otomatis setiap build): compile semua cell; simulasi numpy blend math (rankpct + weighted, dgn
file & tanpa file); jalur fallback tanpa file; isi metadata (id/code_file/dataset_sources).

Pemakaian:
  python3 build_s6student.py                         # default: name=s6student wmurid=0.35
  python3 build_s6student.py name=s6student wmurid=0.30
"""
import contextlib
import copy
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "s6student")
WMURID = float(args.get("wmurid", "0.35"))
NB_SRC = Path(args.get("nb", ROOT / "kaggle_kernel_s6prb" / "rsna-knee-s6prb.ipynb"))
META_SRC = ROOT / "kaggle_kernel_s6prb" / "kernel-metadata.json"
STUDENT_DS = "alverlfarfaraway/rsna-knee-student-preds"
SLUG = "rsna-knee-" + NAME
MARK_BEGIN = "# ---- leg murid"
MARK_END = "# ---- fim leg murid ----"
assert 0.0 <= WMURID <= 1.0, WMURID

MURID_BLOCK = r'''
# ---- leg murid (student preds) [build_s6student] ----
# Murid = media de rank-pct das pernas cnxt/dino POR FINDING; blend final por finding:
#   b_final = (1 - WMURID) * rankpct(b_s6) + WMURID * rankpct(m)   -> clip [0,1].
# Sem student_test_preds.csv (ou incompleto/ilegivel) -> fallback S6 puro: so WARNING, nunca erro.
WMURID = __WMURID__
student, _stu_path = None, None
if WMURID > 0:
    import glob as _s_gb
    _stu_pat = os.environ.get('S6S_STU_PAT', '/kaggle/input/**/student_test_preds.csv')   # S6S_STU_PAT = override de teste local
    try:
        _cand = sorted(_s_gb.glob(_stu_pat, recursive=True))
        if not _cand:
            print('[BLEND] murid: student_test_preds.csv NAO encontrado -> fallback S6 sem murid', flush=True)
        else:
            _sd = pd.read_csv(_cand[0], dtype={'StudyInstanceUID': str})
            _have = [c for c in labels if ('cnxt_' + c) in _sd.columns or ('dino_' + c) in _sd.columns]
            _uid = set(_sd['StudyInstanceUID'])
            _miss = [u for u in test_ids if u not in _uid]
            if not _have or _miss:
                print('[BLEND] murid: arquivo incompleto (%d/%d findings, %d ids faltando) -> fallback S6 sem murid'
                      % (len(_have), len(labels), len(_miss)), flush=True)
            else:
                student = _sd.set_index('StudyInstanceUID').loc[test_ids]
                _stu_path = _cand[0]
                print('[BLEND] murid leg OK:', _stu_path, '| findings', len(_have), '/', len(labels), flush=True)
    except Exception as _e:
        student, _stu_path = None, None
        print('[BLEND] murid ilegivel -> fallback S6 sem murid:', repr(_e)[:120], flush=True)
if student is not None:
    _m_ok = 0
    for j, c in enumerate(labels):
        _legs = []
        for _tag in ('cnxt', 'dino'):
            if (_tag + '_' + c) in student.columns:
                _v = pd.to_numeric(student[_tag + '_' + c], errors='coerce').to_numpy(dtype=np.float64)
                _legs.append(pct(np.where(np.isfinite(_v), _v, 0.5)))       # 0.5 ausente = neutro
        if not _legs:
            print('[BLEND] murid: finding sem coluna cnxt_/dino_ -> mantido S6:', c, flush=True)
            continue
        _m = np.mean(_legs, axis=0)                     # 0.5*rankpct(cnxt) + 0.5*rankpct(dino) (renormalizado se faltar 1 perna)
        out[:, j] = np.clip((1.0 - WMURID) * pct(out[:, j]) + WMURID * pct(_m), 0.0, 1.0)
        _m_ok += 1
    print('[BLEND] leg murid aplicado: %d/%d findings | w=%.2f' % (_m_ok, len(labels), WMURID), flush=True)
# ---- fim leg murid ----
'''


def pct(v):
    """Rank percentil local (mesma def. do cell Stage 3)."""
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.lstrip("\n").splitlines(keepends=True)}


def build():
    nb = json.loads(NB_SRC.read_text(encoding="utf-8"))
    cells = copy.deepcopy(nb["cells"])
    idx = [i for i, c in enumerate(cells)
           if c["cell_type"] == "code" and "# Stage 3" in "".join(c["source"]) and "W_RAPTOR, W_OURS, W_OF" in "".join(c["source"])]
    assert len(idx) == 1, f"cell Stage 3 unico nao encontrado: {idx}"
    i = idx[0]
    src = "".join(cells[i]["source"])
    assert src.count("if SYNRES:") == 1 and src.count("def pct(v):") == 1
    assert "__WMURID__" not in src and MARK_BEGIN not in src

    block = MURID_BLOCK.replace("__WMURID__", repr(WMURID))
    assert "__WMURID__" not in block
    compile(block, "murid", "exec")
    anchor = "if SYNRES:\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, block + "\n" + anchor, 1)

    old_rec = "'of_leg': ofl is not None, 'rows': n,"
    assert src.count(old_rec) == 1
    src = src.replace(old_rec, old_rec + " 'student': student is not None, 'wmurid': WMURID, 'student_file': _stu_path,", 1)
    old_nc = "('weights', 'reader', 'of_leg', 'rows', 'raptor_failed', 'ours_failed', 'total_seconds')"
    assert src.count(old_nc) == 1
    src = src.replace(old_nc, "('weights', 'reader', 'of_leg', 'student', 'wmurid', 'rows', 'raptor_failed', 'ours_failed', 'total_seconds')", 1)
    compile(src, "blend", "exec")
    assert src.count("__WMURID__") == 0
    cells[i]["source"] = src.splitlines(keepends=True)

    nb["cells"] = cells
    out = ROOT / f"kaggle_kernel_{NAME}"
    out.mkdir(exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = json.loads(META_SRC.read_text(encoding="utf-8"))
    meta["id"] = f"alverlfarfaraway/{SLUG}"
    meta["title"] = "RSNA Knee " + NAME
    meta["code_file"] = f"{SLUG}.ipynb"
    ds = list(meta.get("dataset_sources", []))
    if STUDENT_DS not in ds:
        ds.append(STUDENT_DS)
    meta["dataset_sources"] = ds
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, i, len(cells)


def _extract_block(src):
    a, b = src.index(MARK_BEGIN), src.index(MARK_END) + len(MARK_END)
    return src[a:b]


def _ns(test_ids, labels, out):
    return {"np": np, "pd": pd, "os": os, "pct": pct, "test_ids": list(test_ids),
            "labels": list(labels), "out": np.array(out, dtype=np.float64, copy=True)}


def _run(block, ns):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(block, "murid-test", "exec"), ns)
    return buf.getvalue(), ns


def validate(out, ncells):
    nb = json.loads((out / f"{SLUG}.ipynb").read_text(encoding="utf-8"))
    assert len(nb["cells"]) == ncells
    for k, c in enumerate(nb["cells"]):                       # 1) compile todas as cells
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if s.lstrip().startswith("%%"):                   # magic IPython (ex.: %%writefile) -> ignora a 1a linha
                s = "pass\n" + s.lstrip().split("\n", 1)[1]
            compile(s, f"<cell{k}>", "exec")
    src = "".join(nb["cells"][-1]["source"])
    assert MARK_BEGIN in src and MARK_END in src and "__WMURID__" not in src
    assert src.index(MARK_BEGIN) < src.index("if SYNRES:") < src.index("sub = pd.DataFrame")
    block = _extract_block(src)

    labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
              'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
    test_ids = [f"1.2.3.{i}" for i in range(24)]
    rng = np.random.default_rng(7)
    out0 = rng.random((24, len(labels)))

    # 2a) fallback: SEM arquivo -> student None, out intacto, warning impresso, sem excecao
    os.environ.pop("S6S_STU_PAT", None)
    log, ns = _run(block, _ns(test_ids, labels, out0))
    assert ns["student"] is None and ns["_stu_path"] is None, "fallback: student deveria ser None"
    assert np.array_equal(ns["out"], out0), "fallback: out nao pode ser alterado"
    if WMURID > 0:
        assert "fallback S6" in log, log

    # 2b) com arquivo (ordem embaralhada + uid str) -> b_final = 0.65*rankpct(b) + 0.35*rankpct(m)
    def expected(o, stu, w):
        e = np.array(o, dtype=np.float64, copy=True)
        for j, c in enumerate(labels):
            legs = [pct(stu[f"{t}_{c}"].to_numpy(dtype=np.float64)) for t in ("cnxt", "dino") if f"{t}_{c}" in stu.columns]
            m = np.mean(legs, axis=0)
            e[:, j] = np.clip((1.0 - w) * pct(e[:, j]) + w * pct(m), 0.0, 1.0)
        return e

    with _tmpdir() as d:
        stu = pd.DataFrame({"StudyInstanceUID": test_ids})
        for t in ("cnxt", "dino"):
            for c in labels:
                stu[f"{t}_{c}"] = rng.random(len(test_ids)).astype(np.float32)
        p = Path(d) / "student_test_preds.csv"
        stu.sample(frac=1.0, random_state=3).to_csv(p, index=False)      # linhas fora de ordem
        os.environ["S6S_STU_PAT"] = str(p)
        log, ns = _run(block, _ns(test_ids, labels, out0))
        if WMURID == 0.0:                                   # leg desligada: arquivo presente mas nada muda
            assert ns["student"] is None and np.array_equal(ns["out"], out0), "wmurid=0 nao pode alterar out"
            assert "murid leg OK" not in log
        else:
            assert ns["student"] is not None and ns["_stu_path"] == str(p), log
            exp = expected(out0, stu.set_index("StudyInstanceUID").loc[test_ids], WMURID)
            assert np.allclose(ns["out"], exp, atol=1e-12), "blend math (cnxt+dino) diverge"
            assert (ns["out"] >= 0).all() and (ns["out"] <= 1).all()
            assert "murid leg OK" in log and "leg murid aplicado" in log, log

        # 2c) so cnxt (dino ausente) -> m = rankpct(cnxt), 1 perna renormalizada
        stu2 = stu[[c for c in stu.columns if not c.startswith("dino_")]]
        p2 = Path(d) / "only_cnxt.csv"
        stu2.to_csv(p2, index=False)
        os.environ["S6S_STU_PAT"] = str(p2)
        log, ns = _run(block, _ns(test_ids, labels, out0))
        if WMURID > 0:
            assert ns["student"] is not None
            exp2 = expected(out0, stu2.set_index("StudyInstanceUID").loc[test_ids], WMURID)
            assert np.allclose(ns["out"], exp2, atol=1e-12), "blend math (só cnxt) diverge"

        # 2d) arquivo com ids faltando -> fallback (nao erro)
        stu3 = stu.iloc[:-3]
        p3 = Path(d) / "missing_ids.csv"
        stu3.to_csv(p3, index=False)
        os.environ["S6S_STU_PAT"] = str(p3)
        log, ns = _run(block, _ns(test_ids, labels, out0))
        assert ns["student"] is None and np.array_equal(ns["out"], out0), "ids faltando deveria dar fallback"
        if WMURID > 0:
            assert "fallback S6" in log, log
    os.environ.pop("S6S_STU_PAT", None)

    # 3) metadata
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    src_meta = json.loads(META_SRC.read_text(encoding="utf-8"))
    assert meta["id"] == f"alverlfarfaraway/{SLUG}" and meta["code_file"] == f"{SLUG}.ipynb"
    assert meta["competition_sources"] == src_meta["competition_sources"]
    assert meta["dataset_sources"][:len(src_meta["dataset_sources"])] == src_meta["dataset_sources"]
    assert STUDENT_DS in meta["dataset_sources"] and meta["dataset_sources"].count(STUDENT_DS) == 1
    print(f"validasi OK: {ncells} cells compilan | blend math (2 pernas / 1 perna) OK | "
          f"fallback sem arquivo + ids faltando OK | metadata OK (wmurid={WMURID})")


@contextlib.contextmanager
def _tmpdir():
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="s6student_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    out, i_blend, n_cells = build()
    print(f"ditulis: {out} | id alverlfarfaraway/{SLUG} | cell Stage 3 = idx {i_blend} | wmurid={WMURID} | "
          f"datasets + {STUDENT_DS}")
    validate(out, n_cells)
