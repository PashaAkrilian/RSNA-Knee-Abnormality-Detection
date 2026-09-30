#!/usr/bin/env python3
"""Build rsna-knee-train-round2-<init>-r<R>: putaran-2 NOISY-STUDENT -- fine-tune CoAtNet penuh di
atas label FUSI (guru laporan + murid OOF), bukan label guru saja (RESEARCH_NOTES §32).

Identik dgn resep v3 (bukti +0,005 di LB nyata, §49) KECUALI:
  - label = fused_labels_r<R>.csv  (dibuat scripts label_engineering/fuse_noisy_student.py)
  - cache volume dimount dari 3 kernel CPU (rsna-knee-volcache-s*) -> hemat ~0,75 jam GPU
  - init bisa c5 (default, geometri 336px/64 slot), v10 (384px/64 slot), atau v8 (384px/44 slot):
    tiap arm publik dilatih dgn geometri yang SAMA spt di daftar ARMS pipeline publik supaya
    checkpoint-nya tetap drop-in.

Pemakaian: python3 build_train_coatnet_round2.py <round=1> [init=c5|v10|v8] [epochs=4]
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
INIT = sys.argv[2] if len(sys.argv) > 2 else "c5"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

SLOTS64 = ("[('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), "
           "('Coronal', 0, 8), ('Axial', -1, 12)]")
SLOTS44 = ("[('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), "
           "('Coronal', 0, 6), ('Axial', -1, 8)]")
GEOM = {
    "c5":  dict(ckpt="raptor_ft_coatnet_v5_full_swa.pt", img=336, slots=SLOTS64, k_eval=42,
                ds="dreaddevelopment/raptor-knee-maxspan"),
    "v10": dict(ckpt="raptor_ft_coatnet_v10_full.pt", img=384, slots=SLOTS64, k_eval=62,
                ds="dreaddevelopment/raptor-knee-native384dense"),
    "v8":  dict(ckpt="raptor_ft_coatnet_v8_full_swa.pt", img=384, slots=SLOTS44, k_eval=42,
                ds="dreaddevelopment/raptor-knee-native384"),
}
assert INIT in GEOM, f"init tak dikenal: {INIT}"
G = GEOM[INIT]
TAG = f"round2-{INIT}-r{R}"
SLUG = f"rsna-knee-train-{TAG}"
OUTDIR = f"kaggle_kernel_train_{TAG.replace('-', '_')}"
OUTPT = f"coatnet_fused_r{R}_{INIT}.pt"

base = (REPO / "build_train_coatnet_v3_rawmean.py").read_text()
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

def must_replace(text, old, new, count=1):
    assert old in text, f"tidak ketemu verbatim: {old[:70]!r}"
    return text.replace(old, new, count)

# --- geometri harness sesuai init ---
ke_src = must_replace(ke_src, "IMG = 336", f"IMG = {G['img']}")
ke_src = must_replace(ke_src, "SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), "
                              "('Coronal', 0, 8), ('Axial', -1, 12)]", f"SLOTS = {G['slots']}")
ke_src = must_replace(ke_src, "K_EVAL = 42", f"K_EVAL = {G['k_eval']}")

# --- label fusi ---
base = must_replace(base, "report_labels_blended_rawmean_v1.csv", f"fused_labels_r{R}.csv", 99)
base = must_replace(base, "dimaspashaakrilian/rsna-knee-blended-report-labels-rawmean-v1",
                    f"dimaspashaakrilian/rsna-knee-fused-labels-r{R}")
# --- checkpoint init & dataset-nya ---
base = must_replace(base, "KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')",
                    f"KE['find_weight_file']({G['ckpt']!r})")
base = must_replace(base, '"dreaddevelopment/raptor-knee-widedense",',
                    f'"dreaddevelopment/raptor-knee-widedense",\n        "{G["ds"]}",')
# --- epoch ---
base = must_replace(base, "EPOCHS = int(os.environ.get('EPOCHS', '4'))",
                    f"EPOCHS = int(os.environ.get('EPOCHS', '{EPOCHS}'))")
# --- cache termount dari kernel CPU ---
base = must_replace(base, """CDIR = '/kaggle/tmp/volcache'
os.makedirs(CDIR, exist_ok=True)
reader = KE['_make_reader']()

def cache_one(sid):
    fn = f'{CDIR}/{sid}.npz'
    if os.path.exists(fn):
        return 'skip'""", """CDIR = '/kaggle/tmp/volcache'
os.makedirs(CDIR, exist_ok=True)
import glob as _g
MOUNTED = {}
for _d in _g.glob('/kaggle/input/**/volcache', recursive=True):
    if os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.endswith('.npz'):
                MOUNTED[_f[:-4]] = os.path.join(_d, _f)
tlog(f'cache termount dari kernel CPU: {len(MOUNTED)} studi (sisanya di-decode di sini)')
reader = KE['_make_reader']()

def cache_one(sid):
    fn = f'{CDIR}/{sid}.npz'
    if sid in MOUNTED or os.path.exists(fn):
        return 'skip'""")
base = must_replace(base, """def load_study(sid):
    z = np.load(f'/kaggle/tmp/volcache/{sid}.npz')
    return z['vol'], z['mask']""", """def load_study(sid):
    fn = f'/kaggle/tmp/volcache/{sid}.npz'
    if not os.path.exists(fn):
        fn = MOUNTED[sid]
    z = np.load(fn)
    return z['vol'], z['mask']""")
# --- keluaran & identitas ---
base = base.replace("coatnet_rawmean_ft_v3.pt", OUTPT)
base = base.replace("train_coatnet_v3_rawmean_gold_preds.csv", f"train_{TAG}_gold_preds.csv")
base = must_replace(base, 'OUT = REPO / "kaggle_kernel_train_coatnet_v3_rawmean"', f'OUT = REPO / "{OUTDIR}"')
base = must_replace(base, '(OUT / "train_coatnet_v3_rawmean.ipynb")', f'(OUT / "{SLUG}.ipynb")')
base = must_replace(base, '"id": "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean"',
                    f'"id": "dimaspashaakrilian/{SLUG}"')
base = must_replace(base, '"title": "RSNA Knee - Train CoatNet v3 rawmean"',
                    f'"title": "RSNA Knee - Train {TAG}"')
base = must_replace(base, '"code_file": "train_coatnet_v3_rawmean.ipynb"', f'"code_file": "{SLUG}.ipynb"')
base = must_replace(base, 'REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")',
                    'REPO = Path(__file__).resolve().parent')
base = must_replace(base, '"competition_sources": ["rsna-knee-abnormality-detection"],\n}',
                    '"kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-s{i}" for i in range(3)],\n'
                    '    "competition_sources": ["rsna-knee-abnormality-detection"],\n}')
# harness yang sudah dipatch ditanam langsung (jangan baca ulang berkas aslinya)
base = must_replace(base, "ke_src = SRC_TXT.read_text()", "ke_src = _KE_SRC_PATCHED")
marker = 'SRC_TXT = REPO / "build_train_coatnet_ke_src.txt"'
base = must_replace(base, marker, marker + f"\n_KE_SRC_PATCHED = {ke_src!r}\n")

out_py = REPO / f"build_train_{TAG.replace('-', '_')}.py"
out_py.write_text(base)
r = subprocess.run([sys.executable, str(out_py)], capture_output=True, text=True, cwd=REPO)
print(r.stdout.strip() or r.stderr.strip()[-900:])
# dedupe dataset_sources (init c5 membuat raptor-knee-maxspan muncul 2x -> Kaggle bisa menolak)
_mp = REPO / OUTDIR / "kernel-metadata.json"
_m = json.loads(_mp.read_text())
_m["dataset_sources"] = list(dict.fromkeys(_m["dataset_sources"]))
_mp.write_text(json.dumps(_m, indent=2))
print("builder:", out_py.name, "| slug:", SLUG, "| keluaran:", OUTPT)
