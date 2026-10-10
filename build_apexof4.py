#!/usr/bin/env python3
"""Bangun notebook Apex + OrthoFoundation-L fold0 v4: OF JALAN LEBIH DULU (sebelum sel-sel Apex), lalu Apex, lalu blend rank.

Mengapa (10 Okt malam): 57044355 (v3 = OF sesudah Apex) selesai dlm +-1 j 35 mnt dgn 0,950 = Apex murni -> kaki OF tidak jalan di run penilaian
(RAM diduga). v4: proses induk kecil saat OF mulai, antrean potongan dgn klaim, supervisor/restart/daur-ulang pekerja, pemutus waktu (of_stage_v4.py).

Pemakaian (key=value):
  python3 build_apexof4.py name=apexof4 wo=0.30                          # PRODUKSI: bila OF gagal -> keluaran Apex murni
  python3 build_apexof4.py name=apexof4t wo=0.30 loud=1                  # PERCOBAAN: bila OF gagal -> gagal KERAS (status ERROR = bukti)
  python3 build_apexof4.py name=apexof4r pseudo=96 stop_h=0.6 kill_h=0.8 inject=1 loud=1   # REHEARSAL commit-run (log terbaca)
  order=apex_first : DIAGNOSTICO — OF depois do Apex (como em 57044355), mesmo supervisor v4; printevery=60 para log denso.
  pseudo=N : N studi TREN (tanpa emas) disamarkan sbg uji di /kaggle/working/pseudo; sel Apex ditambal utk membaca akar itu (hanya mode pseudo).
"""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from of_stage_v4 import BLEND4, DATA_PRESET_PROD, DATA_PRESET_PSEUDO, OF4_CELL  # noqa: E402

BASE = ROOT / "kaggle_base" / "apex_grandmaster_stack_20261008.ipynb"
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "apexof4")
WO = float(args.get("wo", "0.30"))
PSEUDO = int(args.get("pseudo", "0"))
STOP_H = args.get("stop_h", "6.2")
KILL_H = args.get("kill_h", "6.6")
NPROC = args.get("nproc", "3")
CHUNK = args.get("chunk", "8")
MAXCHUNKS = args.get("maxchunks", "1")
RSSMAX = args.get("rssmax", "8.0")
MMAP = args.get("mmap", "0")
NOPIN = args.get("nopin", "0")
EXTRAENV = args.get("extraenv", "{}")        # dict literal Python, ex.: {\"MALLOC_MMAP_THRESHOLD_\": \"1048576\"}
MEMPERW = args.get("memperw", "8.5")
MEMFREE = args.get("memfree", "9.5")
INJECT = args.get("inject", "0")
LOUD = args.get("loud", "0") not in ("0", "", "false", "False")
MINCOV = args.get("mincov", "0.15")
POLL = args.get("poll", "10")
PRINTEVERY = args.get("printevery", "300")
ORDER = args.get("order", "of_first")        # of_first (producao) | apex_first (diagnostico do handoff com Apex grande)
assert ORDER in ("of_first", "apex_first")

PSEUDO_CELL = r'''
# ---- REHEARSAL: raiz pseudo-oculta (estudos de TREINO sem rotulo ouro fazem o papel do teste) ----
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
print("[PSEUDO] studies", len(_pids), "| series cols", _pcols, "| test cols", _ocols, flush=True)
'''


def fill_of(src):
    rep = {"__WD__": "/kaggle/working", "__NPROC__": NPROC, "__CHUNK__": CHUNK, "__STOP_H__": STOP_H, "__KILL_H__": KILL_H, "__MINCOV__": MINCOV,
           "__MAXCHUNKS__": MAXCHUNKS, "__RSSMAX__": RSSMAX, "__INJECT__": INJECT, "__INJECT_AFTER__": "360", "__POLL__": POLL, "__PRINTEVERY__": PRINTEVERY,
           "__MEMPERW__": MEMPERW, "__MEMFREE__": MEMFREE, "__MMAP__": MMAP, "__NOPIN__": NOPIN, "__EXTRAENV__": EXTRAENV, "__DISKPER__": "0.42", "__DISKBASE__": "1.5", "__FORCEW__": "0", "__SETUP__": "1",
           "__T0EXPR__": "_time.time()", "__DATA_PRESET__": DATA_PRESET_PSEUDO if PSEUDO else DATA_PRESET_PROD}
    for k, v in rep.items():
        src = src.replace(k, v)
    assert "__" not in re.sub(r"__[a-z]+__|__name__|__main__", "", src.replace("__init__", "")), "placeholder nao substituido"
    return src


def src_of(cell):
    return "".join(cell["source"])


base = json.loads(BASE.read_text(encoding="utf-8"))
cells = copy.deepcopy(base["cells"])
md_cells, code_cells = cells[:3], cells[3:]
assert [c["cell_type"] for c in md_cells] == ["markdown"] * 3 and all(c["cell_type"] == "code" for c in code_cells)

if PSEUDO:        # tambal SOMENTE no rehearsal: sel-sel Apex leem a raiz pseudo-oculta
    s4 = src_of(code_cells[1])                                   # cell 4 (Raptor)
    old4 = 'cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",'
    assert s4.count(old4) == 1
    code_cells[1]["source"] = s4.replace(old4, 'cands = ["/kaggle/working/pseudo", "/kaggle/input/competitions/rsna-knee-abnormality-detection",').splitlines(keepends=True)
    s9 = src_of(code_cells[-1])                                  # cell 9 (reader + fusao)
    old9 = '_o_data = _o_find("test_series.csv")'
    assert s9.count(old9) == 1
    code_cells[-1]["source"] = s9.replace(
        old9, '_o_data = "/kaggle/working/pseudo" if _o_os.path.exists("/kaggle/working/pseudo/test_series.csv") else _o_find("test_series.csv")').splitlines(keepends=True)

of_cell = fill_of(OF4_CELL)
blend = BLEND4.replace("__MINCOV__", MINCOV).replace("__WO__", repr(WO)).replace("__LOUD__", repr(LOUD))
extra = [PSEUDO_CELL.replace("__PSEUDO__", str(PSEUDO))] if PSEUDO else []
for i, c in enumerate(extra + [of_cell, blend]):
    compile(c, f"cell{i}", "exec")
inner = re.search(r'_OF_WORKER_SRC = r"""(.*?)"""\n', of_cell, re.S).group(1)
compile(inner, "worker", "exec")

def _code(c):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.lstrip("\n").splitlines(keepends=True)}


new_cells = list(md_cells) + [_code(c) for c in extra]
if ORDER == "of_first":
    new_cells += [_code(of_cell)] + code_cells
else:
    new_cells += code_cells + [_code(of_cell)]
new_cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## Extra leg (OrthoFoundation-L fold0, heliosli) + rank blend\n",
    "O kaki OF roda ANTES dos sel Apex (processo pai pequeno); depois deste ponto Apex ja escreveu submission.csv e o blend ponderado (w=%.2f) e aplicado.\n" % WO,
    "Kredit: heliosli (OrthoFoundation fold0, kode & anatomi), nartaa (fastreader & Raptor), goodpjw (reader), dreaddevelopment/ItsmeSujan (Apex)."]})
new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": blend.lstrip("\n").splitlines(keepends=True)})

nb = copy.deepcopy(base)
nb["cells"] = new_cells
slug = "rsna-knee-" + NAME.replace("_", "-")
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME.replace("_", " "),
    "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "nartaa/rsna-knee-publication-swa-weights-20261007",
                        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "dimaspashaakrilian/rsna-knee-ultralytics-cfg"],
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| slug", slug, "| wo", WO, "| pseudo", PSEUDO, "| stop_h", STOP_H, "kill_h", KILL_H, "| nproc", NPROC, "chunk", CHUNK,
      "| order", ORDER, "| maxchunks", MAXCHUNKS, "| inject", INJECT, "| loud", LOUD, "| cells", len(new_cells))
