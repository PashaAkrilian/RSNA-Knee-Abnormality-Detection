#!/usr/bin/env python3
"""S6-eq: notebook publik #1 heliosli (Raptor 0,5 + compact 0,5, rank sadar-gagal) dirakit ulang sebagai kernel kita,
opsional + Part B reader goodpjw (bobot per temuan 0,03–0,35, tanpa perubahan).

Basis (salinan publik, tidak diubah):
  kaggle_base/heliosli_blend_mine_in_20261011.ipynb   heliosli/blend-mine-in-private-lb-luck-to-all-upvoters (8 sel)
  kaggle_base/goodpjw_s6_reader_lb0954_20261012.ipynb goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954 (sel 10–15 = Part B)

Pemakaian:
  python3 build_s6eq.py name=s6eq  owner=alverlfarfaraway
  python3 build_s6eq.py name=s6eqr owner=alverlfarfaraway reader=1
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "s6eq")
OWNER = args.get("owner", "dimaspashaakrilian")
READER = args.get("reader", "0") == "1"
BASE_EQ = ROOT / "kaggle_base" / "heliosli_blend_mine_in_20261011.ipynb"
BASE_RD = ROOT / "kaggle_base" / "goodpjw_s6_reader_lb0954_20261012.ipynb"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def check_compiles(cells):
    for k, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = src(c)
        if s.lstrip().startswith("%%"):                       # magic sel (%%writefile): lewati baris pertama
            s = "pass\n" + s.lstrip().split("\n", 1)[1]
        compile(s, f"<cell{k}>", "exec")


nb = json.loads(BASE_EQ.read_text(encoding="utf-8"))
cells = copy.deepcopy(nb["cells"])
assert len(cells) == 8, len(cells)
stage3 = src(cells[7])
assert "W_RAPTOR, W_OURS = 0.5, 0.5" in stage3 and "def pct_on" in stage3, "basis bukan versi 0,5/0,5 sadar-gagal"
assert "rsna-knee-compact-r6cnxfsss" in src(cells[2])
for c in cells:                                              # bersihkan keluaran tersimpan
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

datasets = ["heliosli/rsna-knee-compact-r6cnxfsss", "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
if READER:
    rd = json.loads(BASE_RD.read_text(encoding="utf-8"))["cells"]
    assert "Part B" in src(rd[10]) and "OWN_W_PER_TARGET" in src(rd[15]), "struktur notebook goodpjw berubah"
    part_b = copy.deepcopy(rd[10:16])
    for c in part_b:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    cells += part_b
    datasets.append("goodpjw2008/rsna-knee-2-5d-convnext-reader")

check_compiles(cells)
nb["cells"] = cells
slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
meta = {
    "id": f"{OWNER}/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": ["gpu"], "dataset_sources": datasets, "kernel_sources": [], "model_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"], "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| sel", len(cells), "| reader", READER, "| datasets", datasets, "| id", meta["id"])
