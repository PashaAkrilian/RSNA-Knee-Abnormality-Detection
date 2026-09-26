#!/usr/bin/env python3
"""Build rsna-knee-submit-ourarms: pipeline publik TERBARU (d4-blend, 0.943) dgn SELURUH arm
CoAtNet diganti versi yang sudah di-fine-tune di label kita.

KENAPA SELURUHNYA, BUKAN SATU (pelajaran dari 2 kegagalan):
- §48 menempel v3 di LUAR hasil akhir (α=0,15)      -> 0,941 (nihil)
- §50 memasukkan v3 ke ARMS dgn w=0,35              -> 0,941 (nihil)
Keduanya hanya mengubah SEBAGIAN kecil suara di dalam cabang CoAtNet, sementara cabang itu sendiri
(bobot 0,60-1,00 per label di blend akhir) komposisinya nyaris tak berubah. Bukti bahwa resep label
kita nyata: v3 solo 0,932 vs c5 solo 0,927 di LB NYATA (§49, +0,005, variabel tunggal = checkpoint).
Maka yang masuk akal: naikkan SEMUA anggota cabang itu sekaligus.

Peta pengganti (stock -> punya kita):
  raptor_ft_coatnet_v5_full_swa.pt -> coatnet_rawmean_ft_v3.pt   (dipakai 2x: maxspan-v5 & -reverse)
  raptor_ft_coatnet_v10_full.pt    -> coatnet_ourlabels_v10.pt
  raptor_ft_coatnet_v8_full_swa.pt -> coatnet_ourlabels_v8.pt
Bobot ARMS TIDAK diubah (0,60/0,10/0,10/0,20) — supaya satu-satunya variabel adalah bobot
checkpoint, persis seperti eksperimen v3-solo yang memberi jawaban bersih.

Pemakaian:
  python3 build_submit_ourarms.py            # pakai semua pengganti yang checkpoint-nya siap
  python3 build_submit_ourarms.py v3         # hanya ganti arm c5 (uji bertahap)
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-1000/-home-pasha-Documents-RSNA-Knee-Abnormality-Detection/"
               "56a96f16-dd4d-450e-802e-b23bc47ce017/scratchpad")
SRC = SCRATCH / "new26/pjmathematician_rsna-knee-d4-blend/rsna-knee-d4-blend.ipynb"
SRC_META = SRC.parent / "kernel-metadata.json"
OUT = REPO / "kaggle_kernel_submit_ourarms"

# stock -> (checkpoint kita, kernel sumbernya)
SWAPS = {
    "v3":  ("raptor_ft_coatnet_v5_full_swa.pt", "coatnet_rawmean_ft_v3.pt",
            "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean"),
    "v10": ("raptor_ft_coatnet_v10_full.pt", "coatnet_ourlabels_v10.pt",
            "dimaspashaakrilian/rsna-knee-train-coatnet-ours-v10"),
    "v8":  ("raptor_ft_coatnet_v8_full_swa.pt", "coatnet_ourlabels_v8.pt",
            "dimaspashaakrilian/rsna-knee-train-coatnet-ours-v8"),
}
want = sys.argv[1:] or list(SWAPS)
for w in want:
    assert w in SWAPS, f"tak dikenal: {w} (pilihan {list(SWAPS)})"

nb = json.loads(SRC.read_text())
meta = json.loads(SRC_META.read_text())

n_file, n_pin = 0, 0
# Pin-bypass generik utk checkpoint KITA. find_weight_file() milik mereka menolak file asing
# ("unpinned Raptor checkpoint") -- ini yg bikin percobaan §50 ERROR di push pertama. Kita sisipkan
# cabang di AWAL fungsi, tetap mewajibkan tepat 1 file ketemu; logika pin utk checkpoint MEREKA
# dibiarkan utuh. Catatan escaping: kode ini ada di DALAM string literal `_KE_SRC`, jadi newline
# ditulis "\\n" dan kutip tunggal "\\'".
OLD_DEF = "def find_weight_file(fname):"
NEW_DEF = (
    "def find_weight_file(fname):\\n"
    "    if fname.startswith((\\'coatnet_ourlabels_\\', \\'coatnet_rawmean_\\')):\\n"
    "        _h = sorted(glob.glob(\\'/kaggle/input/**/\\' + fname, recursive=True))\\n"
    "        if len(_h) != 1:\\n"
    "            raise RuntimeError(f\\'expected one ours checkpoint {fname}, found {_h}\\')\\n"
    "        return _h[0]"
)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    orig = s
    for key in want:
        stock, ours, _ = SWAPS[key]
        needle = f'"file": "{stock}"'
        if needle in s:
            n_file += s.count(needle)
            s = s.replace(needle, f'"file": "{ours}"')
    if OLD_DEF in s and "coatnet_ourlabels_" not in orig:
        s = s.replace(OLD_DEF, NEW_DEF, 1)
        n_pin += 1
    if s != orig:
        cell["source"] = s.splitlines(keepends=True)

assert n_file >= len(want), f"penggantian file checkpoint kurang: {n_file} utk {want}"
assert n_pin == 1, f"pin-bypass harus disisipkan tepat 1x, dapat {n_pin}"
print(f"checkpoint ditukar: {n_file} entri ARMS | pin-bypass disisipkan: {n_pin}")

OUT.mkdir(exist_ok=True)
NAME = "submit_ourarms.ipynb"
(OUT / NAME).write_text(json.dumps(nb, indent=1))

meta = dict(meta)
meta["id"] = "dimaspashaakrilian/rsna-knee-submit-ourarms"
meta["title"] = "RSNA Knee - Submit our arms"
meta["code_file"] = NAME
meta.pop("id_no", None)
meta["is_private"] = True
for k in ("dataset_sources", "kernel_sources", "model_sources"):
    if isinstance(meta.get(k), list):
        meta[k] = [x for x in meta[k] if x]
meta["kernel_sources"] = list(meta.get("kernel_sources", [])) + [SWAPS[k][2] for k in want]
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
print("kernel_sources:", meta["kernel_sources"])
