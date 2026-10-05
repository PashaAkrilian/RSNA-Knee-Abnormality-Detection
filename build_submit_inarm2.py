#!/usr/bin/env python3
"""Pipeline publik d4-blend (0,943) dgn ARM RAPTOR DIGANTI versi label-fusi kita (di DALAM stack, bukan leg luar).

Latar (RESEARCH_NOTES §36d): leg luar (murid CoAtNet, RadImageNet, reader ConvNeXt publik) pd bobot 0,30-0,40 -> 0,942 (turun dari
0,943): model lebih lemah dari stack mengencerkan. Yang terbukti lebih baik di LB solo adalah versi label-fusi dari arm Raptor itu sendiri:
c5 stok 0,927 -> r2c5 0,933 (+0,006). Cabang Raptor = 60% keluarga CoAt yg memegang 60-100% bobot akhir per label, jadi arm diganti
langsung di daftar ARMS (cell 27) dgn geometri & bobot yg sama; checkpoint Raptor tidak di-pin hash (hanya arsitektur/resolusi dicek).
  - maxspan-v5 & maxspan-v5-reverse (c5, .60 + .10)  -> coatnet_fused_r1_c5.pt  (r2c5, cache w = geometri arm ini)
  - native384-v8 (.20)                               -> coatnet_fused_r1_v8.pt  (r2v8, cache y = geometri arm ini)   [opsional]
  - native384dense-v10 (.10)                         -> stok
Pemakaian: python3 build_submit_inarm2.py <name> [c5=1] [v8=1]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "inarm2"
A = dict(kv.split("=", 1) for kv in sys.argv[2:])
USE_C5 = A.get("c5", "1") == "1"
USE_V8 = A.get("v8", "1") == "1"
SLUG = f"rsna-knee-inarm2-{NAME}".replace("_", "-")
OUT = REPO / f"kaggle_kernel_inarm2_{NAME.replace('-', '_')}"

nb = json.loads((REPO / "kaggle_base/pj_rsna_knee_d4_blend_20260928.ipynb").read_text())
meta0 = json.loads((REPO / "kaggle_base/pj_rsna_knee_d4_blend_metadata.json").read_text())
cells = nb["cells"]

KSRC, ROOTS = [], []
if USE_C5:
    KSRC.append("dimaspashaakrilian/rsna-knee-train-round2-c5-r1")
    ROOTS += ["/kaggle/input/rsna-knee-train-round2-c5-r1", "/kaggle/input/notebooks/dimaspashaakrilian/rsna-knee-train-round2-c5-r1"]
if USE_V8:
    KSRC.append("dimaspashaakrilian/rsna-knee-train-round2-v8-r1")
    ROOTS += ["/kaggle/input/rsna-knee-train-round2-v8-r1", "/kaggle/input/notebooks/dimaspashaakrilian/rsna-knee-train-round2-v8-r1"]

# cell 2: tambah akar aset (pencari aset menelusuri ASSET_ROOTS saja, kedalaman 6, nama berkas persis, wajib tepat 1 hit)
c2 = "".join(cells[2]["source"])
marker = "\ndef _asset_walk():"
assert c2.count(marker) >= 1 and "ASSET_ROOTS = [" in c2
c2 = c2.replace(marker, f"\nASSET_ROOTS += {ROOTS!r}   # checkpoint label-fusi kita (arm Raptor pengganti)\n" + marker, 1)
cells[2]["source"] = c2.splitlines(keepends=True)

# cell 27: ganti berkas di ARMS (literal di dalam _KE_SRC)
c27 = "".join(cells[27]["source"])
if USE_C5:
    assert c27.count('"file": "raptor_ft_coatnet_v5_full_swa.pt"') == 2
    c27 = c27.replace('"file": "raptor_ft_coatnet_v5_full_swa.pt"', '"file": "coatnet_fused_r1_c5.pt"')
if USE_V8:
    assert c27.count('"file": "raptor_ft_coatnet_v8_full_swa.pt"') == 1
    c27 = c27.replace('"file": "raptor_ft_coatnet_v8_full_swa.pt"', '"file": "coatnet_fused_r1_v8.pt"')
cells[27]["source"] = c27.splitlines(keepends=True)
# cell 31 asli (fleet privat pjmathematician) crash bila aset privat tak ada (_EFF6 None) -> ganti no-op; cell 32 lalu memulihkan keluaran stack
assert "OUR LEG — 2/3" in "".join(cells[31]["source"])
cells[31]["source"] = ["print('[ours 2/3] leg fleet privat dilewati (inarm2: perbaikan ada di dalam ARMS)', flush=True)\n"]
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None

OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = dict(meta0)
meta.update({"id": f"dimaspashaakrilian/{SLUG}", "title": "RSNA Knee inarm2 " + NAME.replace("-", " "), "code_file": f"{SLUG}.ipynb", "is_private": True})
meta["dataset_sources"] = [d for d in meta0["dataset_sources"] if d]
meta["kernel_sources"] = list(meta0.get("kernel_sources", [])) + KSRC
meta.pop("id_no", None)
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| c5->r2c5:", USE_C5, "| v8->r2v8:", USE_V8)
