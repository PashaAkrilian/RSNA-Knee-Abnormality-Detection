#!/usr/bin/env python3
"""Build rsna-knee-submit-v3-inarm: integrasi v3 SECARA BENAR ke pipeline publik 0,941 —
v3 masuk SEBAGAI ANGGOTA arm CoAtNet (di dalam ARMS), bukan ditempel di luar hasil akhir.

LATAR (kenapa beda dari percobaan pertama, build_submit_v3_hybrid.py):
- Percobaan-1 (hybrid, 13 Sep pagi): v3 di-rank-blend di ATAS submission.csv final dgn alpha=0,15.
  Hasil real: 0,941 -> 0,941 (NULL, tak terbaca di 3 desimal). Diagnosis §29a: run cuma pakai
  ~4,8 jam dari batas 9 jam, jadi BUKAN krn kehabisan waktu — bobot efektifnya saja terlalu encer,
  dan v3 di situ "bersaing dgn dirinya sendiri" krn c5 (induk v3) sudah ada di dalam ensemble.
- Bukti BARU yg membenarkan integrasi lebih agresif (§29d + hasil real 13 Sep sore):
  **v3 SOLO = 0,932 di leaderboard nyata vs c5 SOLO = 0,927** — harness/K_EVAL/struktur identik,
  satu-satunya variabel = bobot checkpoint. Jadi v3 TERBUKTI +0,005 lebih baik dari c5 di data
  nyata (bukan cuma +0,0028 di 58 gold), dan c5 adalah arm DOMINAN (w=0,60) di pipeline publik.
- Uji blend lokal (§29b): konfigurasi terbaik = PERTAHANKAN c5, TAMBAH v3 dgn bobot >= c5
  (0,9280) — lebih baik drpd ganti c5 (0,9264) maupun alpha kecil (0,9254).

PERUBAHAN: HANYA daftar ARMS di dalam `_KE_SRC` milik pipeline mereka (literal Python biasa).
Bobot baru (dinormalisasi otomatis oleh harness mereka):
    v3 (kita)            0,35   <- bobot terbesar, sesuai bukti v3 > c5 di LB nyata
    maxspan-v5 (c5)      0,30   (turun dari 0,60, TETAP dipertahankan utk diversity)
    native384dense-v10   0,08   (dari 0,10)
    maxspan-v5-reverse   0,07   (dari 0,10)
    native384-v8         0,20   (tidak berubah)
Tidak ada logika lain yg disentuh — `_coat_substitute()` (resgated blend) & per-label outer
routing mereka bekerja di atas CSV hasil, jadi tidak terpengaruh. Loop mereka
`for a, arm in enumerate(ARMS)` + alokasi `len(ARMS)` menyesuaikan sendiri ke 5 arm.

Biaya: +1 arm CoAtNet (k_eval 62) ~ +25% waktu cabang CoAtNet. Run sebelumnya 4,8 jam dari batas
9 jam, jadi aman. Kuota GPU tersisa 11,49h, pipeline penuh ~9,7h.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-1000/-home-pasha-Documents-RSNA-Knee-Abnormality-Detection/56a96f16-dd4d-450e-802e-b23bc47ce017/scratchpad")
FORK = SCRATCH / "code_research/our_fork"
OUT = REPO / "kaggle_kernel_submit_v3_inarm"

nb = json.loads((FORK / "bend-the-knee-to-the-dinosaurs-7d6471.ipynb").read_text())
meta = json.loads((FORK / "kernel-metadata.json").read_text())

OLD_ARMS = (
    'ARMS = [\\n    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",\\n'
    '     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\\n'
    '     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,\\n'
    '     "reverse": False, "w": 0.60},\\n'
)
NEW_ARMS = (
    'ARMS = [\\n    {"name": "ours-v3-rawmean", "file": "coatnet_rawmean_ft_v3.pt",\\n'
    '     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\\n'
    '     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,\\n'
    '     "reverse": False, "w": 0.35},\\n'
    '    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",\\n'
    '     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\\n'
    '     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,\\n'
    '     "reverse": False, "w": 0.30},\\n'
)

# Pipeline mereka mem-PIN checkpoint yg boleh dimuat (nama + ukuran persis) lewat
# find_weight_file(); checkpoint asing ditolak `unpinned Raptor checkpoint` (ini yg bikin
# dev-test v1 ERROR). Kita daftarkan checkpoint kita dgn cara yg TIDAK melemahkan pengaman
# mereka: cabang khusus utk file KITA (tetap wajib match tepat 1 file + ukuran persis),
# sedangkan logika pin utk checkpoint MEREKA dibiarkan utuh apa adanya.
OLD_GATE = "    if fname not in by_name:\\n        raise RuntimeError(f\\'unpinned Raptor checkpoint: {fname}\\')"
NEW_GATE = (
    "    if fname == \\'coatnet_rawmean_ft_v3.pt\\':\\n"
    "        _ours = sorted(glob.glob(\\'/kaggle/input/**/coatnet_rawmean_ft_v3.pt\\', recursive=True))\\n"
    "        if len(_ours) != 1:\\n"
    "            raise RuntimeError(f\\'expected one ours-v3 checkpoint, found {_ours}\\')\\n"
    "        if os.path.getsize(_ours[0]) != 292874395:\\n"
    "            raise RuntimeError(f\\'ours-v3 checkpoint size mismatch: {os.path.getsize(_ours[0])}\\')\\n"
    "        return _ours[0]\\n"
    + OLD_GATE
)

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if OLD_ARMS not in src:
        continue
    assert OLD_GATE in src, "gate 'unpinned Raptor checkpoint' tidak ketemu verbatim"
    src = src.replace(OLD_GATE, NEW_GATE, 1)
    src = src.replace(OLD_ARMS, NEW_ARMS, 1)
    # turunkan bobot dua arm pelengkap supaya total tetap 1,0 (harness menormalisasi, ini utk kejelasan)
    src = src.replace('"reverse": False, "w": 0.10}', '"reverse": False, "w": 0.08}', 1)
    src = src.replace('"reverse": True, "w": 0.10}', '"reverse": True, "w": 0.07}', 1)
    cell["source"] = src.splitlines(keepends=True)
    patched += 1

assert patched == 1, f"ARMS harus dipatch tepat di 1 cell, dapat {patched}"

OUT.mkdir(exist_ok=True)
nb_name = "submit_v3_inarm.ipynb"
(OUT / nb_name).write_text(json.dumps(nb, indent=1))

meta = dict(meta)
meta["id"] = "dimaspashaakrilian/rsna-knee-submit-v3-inarm"
meta["title"] = "RSNA Knee - Submit v3 inarm"
meta["code_file"] = nb_name
meta.pop("id_no", None)
meta["kernel_sources"] = list(meta.get("kernel_sources", [])) + [
    "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean",
]
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
print("kernel_sources:", meta["kernel_sources"])
