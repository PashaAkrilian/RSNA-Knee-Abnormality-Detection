#!/usr/bin/env python3
"""Sonda de escala do notebook publico S6 do heliosli (Raptor 0,7 + compact ConvNeXt-S 4 folds 0,3 = 0,954): commit-run com raiz pseudo-oculta.

Contexto (11 Okt 03:00): heliosli RESTAUROU o dataset `heliosli/rsna-knee-compact-v1` (02:50 WIB) e republicou o notebook S6 ("Dataset restored(I'd been told it wasn't
allowed.)"). O usuario ja submeteu uma copia (57051680, 02:56 WIB). Este kernel mede, com LOG LEGIVEL, vazao/RAM/tempo do estagio 2 (compact) em N estudos de TREINO
(sem rotulo ouro) fingindo ser o teste — SEM alterar o codigo do heliosli alem de apontar a raiz de dados para /kaggle/working/pseudo (so no modo pseudo).

Pemakaian: python3 build_s6probe.py name=s6probe pseudo=160 [deadline=1.5]
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "s6probe")
PSEUDO = int(args.get("pseudo", "160"))
DEADLINE = args.get("deadline", "")          # opcional: troca DEADLINE_HOURS (ex.: 1.2 h para a sonda)
NB = Path(args.get("nb", ROOT / "kaggle_base" / "s6_heliosli_public_20261011.ipynb"))
assert NB.is_file(), f"copie o notebook publico S6 para {NB}"

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
_psh.copy(_COMP + "/sample_submission.csv", _PS + "/sample_submission.csv")
print("[PSEUDO] studies", len(_pids), "| series cols", _pcols, "| test cols", _ocols, flush=True)
'''

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = copy.deepcopy(nb["cells"])
assert len(cells) == 7, len(cells)
s2 = "".join(cells[2]["source"])
old2 = 'cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",'
assert s2.count(old2) == 1, s2.count(old2)
cells[2]["source"] = s2.replace(old2, 'cands = ["/kaggle/working/pseudo", "/kaggle/input/competitions/rsna-knee-abnormality-detection",').splitlines(keepends=True)
s4 = "".join(cells[4]["source"])
old4 = "DATA = find_mount(COMPETITION, 'test.csv')"
assert s4.count(old4) == 1, s4.count(old4)
s4 = s4.replace(old4, "DATA = Path('/kaggle/working/pseudo') if Path('/kaggle/working/pseudo/test.csv').is_file() else find_mount(COMPETITION, 'test.csv')")
if DEADLINE:
    assert s4.startswith("ARM = 'E_lrprior'")
    s4 = s4.replace("DEADLINE_HOURS = 8.3", f"DEADLINE_HOURS = {DEADLINE}", 1)
cells[4]["source"] = s4.splitlines(keepends=True)
pc = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": PSEUDO_CELL.replace("__PSEUDO__", str(PSEUDO)).lstrip("\n").splitlines(keepends=True)}
compile(PSEUDO_CELL.replace("__PSEUDO__", str(PSEUDO)), "pseudo", "exec")
nb["cells"] = cells[:2] + [pc] + cells[2:]
slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["heliosli/rsna-knee-compact-v1", "nartaa/rsna-knee-publication-swa-weights-20261007", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| slug", slug, "| pseudo", PSEUDO, "| deadline", DEADLINE or "8.3 (original)")
