#!/usr/bin/env python3
"""Builder fine-tune CoAtNet TERPARAMETER: terapkan resep label kita (raw-mean blend
flight0234+Claude) ke checkpoint publik MANA PUN di keluarga Raptor, bukan cuma c5.

DASAR (bukti, bukan tebakan):
- v3 = c5 yang di-fine-tune di label kita -> **0,932 vs 0,927 c5 stock di LB NYATA** (§49).
  Jadi resep label kita memberi ~+0,005 per arm, terukur apple-to-apple.
- TAPI mengganti SATU arm saja di ensemble publik tidak menggerakkan skor akhir sama sekali
  (§48 dan §50, dua-duanya 0,941 persis) -- arm CoAtNet memang cuma sebagian dari bobot akhir.
- Simulasi §30a: menumpuk CoAtNet lagi di label yang sama mentok ~0,937.
=> Maka yang masuk akal: angkat SELURUH arm keluarga CoAtNet (c5 sudah = v3; sisanya v8 & v10),
   supaya cabang CoAtNet (bobot 0,60-1,00 per label di blend akhir mereka) naik utuh, bukan
   sepotong.

Tiap varian memakai geometri persis seperti di daftar ARMS pipeline publik, supaya checkpoint
hasilnya tetap drop-in:
  c5  : img 336, SLOTS64, k_eval 62  (sudah dikerjakan -> v3)
  v10 : img 384, SLOTS64, k_eval 62  (native384dense)
  v8  : img 384, SLOTS44, k_eval 42, span 0.06-0.94 (native384)

Pemakaian:  python3 build_train_coatnet_variant.py v8
            python3 build_train_coatnet_variant.py v10
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

SLOTS64 = ("[('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), "
           "('Coronal', 0, 8), ('Axial', -1, 12)]")
SLOTS44 = ("[('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), "
           "('Coronal', 0, 6), ('Axial', -1, 8)]")

VARIANTS = {
    "v10": dict(ckpt="raptor_ft_coatnet_v10_full.pt", img=384, slots=SLOTS64, k_eval=62,
                dataset="dreaddevelopment/raptor-knee-native384dense",
                out="coatnet_ourlabels_v10.pt", slug="rsna-knee-train-coatnet-ours-v10",
                title="RSNA Knee - Train CoatNet ours v10"),
    "v8": dict(ckpt="raptor_ft_coatnet_v8_full_swa.pt", img=384, slots=SLOTS44, k_eval=42,
               dataset="dreaddevelopment/raptor-knee-native384",
               out="coatnet_ourlabels_v8.pt", slug="rsna-knee-train-coatnet-ours-v8",
               title="RSNA Knee - Train CoatNet ours v8"),
}

name = sys.argv[1] if len(sys.argv) > 1 else "v10"
assert name in VARIANTS, f"varian tak dikenal: {name} (pilihan: {list(VARIANTS)})"
V = VARIANTS[name]

# --- ambil builder v3 sebagai basis, lalu ganti hal-hal yang khas varian ---
base = (REPO / "build_train_coatnet_v3_rawmean.py").read_text()
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

# geometri harness disesuaikan varian
old_img = "IMG = 336"
assert old_img in ke_src
ke_src = ke_src.replace(old_img, f"IMG = {V['img']}", 1)
old_slots = ("SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), "
             "('Coronal', 0, 8), ('Axial', -1, 12)]")
assert old_slots in ke_src
ke_src = ke_src.replace(old_slots, f"SLOTS = {V['slots']}", 1)
assert "K_EVAL = 42" in ke_src
ke_src = ke_src.replace("K_EVAL = 42", f"K_EVAL = {V['k_eval']}", 1)

# checkpoint sumber fine-tune
assert "KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')" in base
base = base.replace("KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')",
                    f"KE['find_weight_file']({V['ckpt']!r})", 1)

# nama output + identitas kernel
base = base.replace("coatnet_rawmean_ft_v3.pt", V["out"])
base = base.replace("train_coatnet_v3_rawmean_gold_preds.csv",
                    f"train_{V['slug']}_gold_preds.csv")
base = base.replace('OUT = REPO / "kaggle_kernel_train_coatnet_v3_rawmean"',
                    f'OUT = REPO / "kaggle_kernel_{V["slug"].replace("-", "_")}"')
base = base.replace('(OUT / "train_coatnet_v3_rawmean.ipynb")',
                    f'(OUT / "{V["slug"]}.ipynb")')
base = base.replace('"id": "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean"',
                    f'"id": "dimaspashaakrilian/{V["slug"]}"')
base = base.replace('"title": "RSNA Knee - Train CoatNet v3 rawmean"',
                    f'"title": "{V["title"]}"')
base = base.replace('"code_file": "train_coatnet_v3_rawmean.ipynb"',
                    f'"code_file": "{V["slug"]}.ipynb"')
base = base.replace('REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")',
                    'REPO = Path(__file__).resolve().parent')
# dataset sumber checkpoint varian ini
base = base.replace('"dreaddevelopment/raptor-knee-widedense",',
                    f'"dreaddevelopment/raptor-knee-widedense",\n        "{V["dataset"]}",')
# harness sudah dipatch di atas -> tanam langsung, jangan baca ulang file aslinya
base = base.replace('ke_src = SRC_TXT.read_text()',
                    'ke_src = _KE_SRC_PATCHED')

stamp = (f"\n# --- geometri harness disesuaikan varian {name} "
         f"(img={V['img']}, k_eval={V['k_eval']}) ---\n"
         f"_KE_SRC_PATCHED = {ke_src!r}\n")
marker = 'SRC_TXT = REPO / "build_train_coatnet_ke_src.txt"'
assert marker in base
base = base.replace(marker, marker + "\n" + stamp, 1)

out_py = REPO / f"build_train_coatnet_ours_{name}.py"
out_py.write_text(base)
print("builder ditulis:", out_py)

import subprocess
r = subprocess.run([sys.executable, str(out_py)], capture_output=True, text=True, cwd=REPO)
print(r.stdout.strip() or r.stderr.strip()[-800:])
