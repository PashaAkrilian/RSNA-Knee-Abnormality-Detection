#!/usr/bin/env python3
"""Notebook submisi: pipeline Apex (ckpt OAI nartaa 0,949 + reader goodpjw per-label = LB 0,950) + ANAK fine-tune kita sbg arm tambahan.

Dasar: kaggle_base/apex_grandmaster_stack_20261008.ipynb (sel 4 = skrip inferensi nartaa dgn daftar ARMS berbobot -> rank-mean tertimbang;
sel 9 = fusi per-label dgn reader goodpjw). Anak = ckpt hasil kernel TPU `rsna-knee-train-tpu-oai-child` (format sama dgn ckpt induk), jadi cukup
(1) menambah entri ARMS, (2) melonggarkan penegasan SHA hanya utk ckpt induk, (3) menempelkan kernel TPU sbg kernel_sources.
Argumen key=value:  name=<nama> children=<f1,f2,..> src=<slug1[,slug2]> (kernel TPU dan/atau GPU penghasil anak) ds=<dataset1[,dataset2]> (mis. dimaspashaakrilian/rsna-knee-oai-child-colab-ckpt, hasil Colab) wp=<bobot induk> wc=<bobot tiap anak> r224=1|0 w224=<bobot sibling r224> reader=1|0
Slug hasil: rsna-knee-oai-fam-<name>.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
A = dict(kv.split("=", 1) for kv in sys.argv[1:])
NAME = A.get("name", "fam1")
CHILDREN = [c for c in A.get("children", "oai_child_a_swa.pt,oai_child_b_swa.pt" if "r224" not in A else "").split(",") if c]
SRCS = [x for x in A.get("src", "dimaspashaakrilian/rsna-knee-train-tpu-oai-child").split(",") if x]
DSS = [x for x in A.get("ds", "").split(",") if x]
WP = float(A.get("wp", "0.4" if "r224" not in A else str(1 - float(A.get("w224", "0.25"))))); WC = float(A.get("wc", "0.3"))
READER = A.get("reader", "1") == "1"
R224 = A.get("r224", "0") == "1"; W224 = float(A.get("w224", "0.25"))
SLUG = f"rsna-knee-oai-fam-{NAME}".replace("_", "-")
OUT = REPO / f"kaggle_kernel_oai_fam_{NAME.replace('-', '_')}"

nb = json.loads((REPO / "kaggle_base/apex_grandmaster_stack_20261008.ipynb").read_text())
meta0 = json.loads((REPO / "kaggle_base/apex_grandmaster_stack_metadata.json").read_text())
cells = nb["cells"]

# --- sel 4: ARMS + penegasan SHA hanya utk induk ---
c4 = "".join(cells[4]["source"])
old_arms = '''ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]'''
assert c4.count(old_arms) == 1
arms = [f'    {{"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": {WP}}},']
for f in CHILDREN:
    arms.append(f'    {{"file": "{f}", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": {WC}}},')
if R224:   # sibling efisiensi nartaa (0,945): crop tengah 307 px -> 224, 1 tampilan, backbone img_size=224
    arms.append(f'    {{"file": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 224, "img": 224, "crop": 307, "single": True, "w": {W224}}},')
c4 = c4.replace(old_arms, "ARMS = [   # induk OAI nartaa + anak fine-tune kita (label fusi r1, KD ke induk)\n" + "\n".join(arms) + "\n]")
old_sha = '''    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"'''
assert c4.count(old_sha) == 1
c4 = c4.replace(old_sha, '''    if os.path.basename(pt_path).startswith("raptor_ft_alldata"):   # induk: hash terikat; anak kita: tanpa hash tetap
        assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"''')
if R224:
    from oai_family_r224_patch import patch_r224
    c4 = patch_r224(c4)
cells[4]["source"] = c4.splitlines(keepends=True)

# --- sel 9: bila reader=0 -> lewati fusi reader (submission.csv = rank-mean induk+anak) ---
if not READER:
    cells[9]["source"] = ["print('[fam] reader dilewati; submission.csv = rank-mean induk+anak', flush=True)\n"]
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None

OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = dict(meta0)
meta.update({"id": f"dimaspashaakrilian/{SLUG}", "title": "RSNA Knee oai fam " + NAME.replace("-", " ").replace("_", " "),
             "code_file": f"{SLUG}.ipynb", "is_private": True, "keywords": ["gpu"]})
meta.pop("id_no", None)
meta["kernel_sources"] = SRCS
meta["dataset_sources"] = list(meta0.get("dataset_sources", [])) + DSS
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| arms:", 1 + len(CHILDREN), "| bobot induk/anak:", WP, WC, "| reader:", READER)
