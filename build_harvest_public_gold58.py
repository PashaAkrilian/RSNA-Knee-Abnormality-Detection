#!/usr/bin/env python3
"""Build rsna-knee-harvest-public-gold58: jalankan PIPELINE PUBLIK TERBAIK (d4-blend, 0.943)
di atas 58 studi GOLD (yang labelnya kita tahu), bukan di test set tersembunyi.

KENAPA INI PENTING (akar kegagalan 2 percobaan sebelumnya):
Dua kali kita menyuntikkan v3 ke ensemble publik (α=0,15 di luar §48; w=0,35 di dalam ARMS §50)
dan DUA-DUANYA menghasilkan 0,941 persis. Penyebabnya bukan kode — melainkan kita menentukan
bobot blending secara BUTA: kita tidak pernah punya prediksi ensemble publik di data berlabel,
jadi tidak bisa mengukur (a) seberapa kuat ia per label, (b) seberapa berkorelasi dgn model kita,
(c) bobot blend mana yang sebenarnya optimal. Tiap tebakan memakan ~10 GPU-jam + 1 slot submission.

Setelah panen ini, semua keputusan blending bisa diuji OFFLINE, gratis, berkali-kali, dan cuma
konfigurasi terbaik yang disubmit.

CARA KERJA (nyaris tanpa menyentuh kode mereka):
Ternyata `find_root()` dan `find_test_root()` di notebook publik SUDAH menghormati env var
`RSNA_COMP_ROOT`. Jadi kita cukup:
  1. Bangun direktori kompetisi palsu `/kaggle/working/goldcomp` berisi:
       test.csv             -> 58 UID gold
       test_series.csv      -> baris train_series.csv utk 58 studi itu
       sample_submission.csv-> kolom persis seperti aslinya
       test_series/<uid>    -> SYMLINK ke train_series/<uid> (tidak menyalin 570GB)
  2. Set RSNA_COMP_ROOT ke situ sebelum sel path resolution mereka jalan.
Satu-satunya patch kode: `find_root()` diberi jalan pintas env var, karena `_one_direct()` miliknya
akan melempar error kalau menemukan DUA root valid (punya kita + kompetisi asli).

Output: submission.csv berisi prediksi ensemble publik utk 58 studi gold -> ditarik ke lokal,
diskor, dipakai mengoptimalkan bobot blend (global & per-label) via nested-CV.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-1000/-home-pasha-Documents-RSNA-Knee-Abnormality-Detection/"
               "56a96f16-dd4d-450e-802e-b23bc47ce017/scratchpad")
SRC_NB = SCRATCH / "new26/pjmathematician_rsna-knee-d4-blend/rsna-knee-d4-blend.ipynb"
OUT = REPO / "kaggle_kernel_harvest_public_gold58"

nb = json.loads(SRC_NB.read_text())

SETUP = '''\
# ===== arahkan pipeline publik ini ke 58 studi GOLD (bukan test tersembunyi) =====
import os, shutil, pandas as pd
from pathlib import Path as _P

_REAL = None
for _c in ['/kaggle/input/competitions/rsna-knee-abnormality-detection',
           '/kaggle/input/rsna-knee-abnormality-detection']:
    if (_P(_c) / 'train.csv').is_file():
        _REAL = _P(_c); break
assert _REAL is not None, 'root kompetisi asli tidak ketemu'
print('root asli:', _REAL, flush=True)

_G = _P('/kaggle/working/goldcomp')
if _G.exists():
    shutil.rmtree(_G)
(_G / 'test_series').mkdir(parents=True)

_tr = pd.read_csv(_REAL / 'train.csv', dtype={'StudyInstanceUID': str})
_ss_real = pd.read_csv(_REAL / 'sample_submission.csv', nrows=1)
_SUBCOLS = list(_ss_real.columns)
_LAB = [c for c in _SUBCOLS if c != 'StudyInstanceUID']
_gold = _tr[_tr[_LAB].notna().all(axis=1)]['StudyInstanceUID'].astype(str).tolist()
print('studi gold ditemukan:', len(_gold), flush=True)
assert len(_gold) >= 50, f'jumlah gold tak wajar: {len(_gold)}'

pd.DataFrame({'StudyInstanceUID': _gold}).to_csv(_G / 'test.csv', index=False)

_ts = pd.read_csv(_REAL / 'train_series.csv',
                  dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
_ts[_ts.StudyInstanceUID.isin(_gold)].to_csv(_G / 'test_series.csv', index=False)
print('baris series utk gold:', int(_ts.StudyInstanceUID.isin(_gold).sum()), flush=True)

_ss = pd.DataFrame({'StudyInstanceUID': _gold})
for _c in _LAB:
    _ss[_c] = 0.5
_ss[_SUBCOLS].to_csv(_G / 'sample_submission.csv', index=False)

_n = 0
for _u in _gold:
    _src = _REAL / 'train_series' / _u
    if _src.is_dir():
        os.symlink(_src, _G / 'test_series' / _u); _n += 1
print('symlink series dibuat:', _n, flush=True)
assert _n == len(_gold), 'ada studi gold tanpa direktori series'

os.environ['RSNA_COMP_ROOT'] = str(_G)
print('RSNA_COMP_ROOT =', os.environ['RSNA_COMP_ROOT'], flush=True)
print('submission.csv hasil run ini = prediksi ensemble publik di 58 studi GOLD', flush=True)
'''

# --- patch tunggal: find_root() diberi jalan pintas env var ---
# NB: find_root() ada sbg kode notebook biasa (newline ASLI), bukan di dalam string literal
# seperti ARMS di _KE_SRC -- jadi jangan pakai '\\n' yg ter-eskape di sini.
OLD = "def find_root():\n    return _one_direct('competition', ["
NEW = ("def find_root():\n"
       "    _g = os.environ.get('RSNA_COMP_ROOT')\n"
       "    if _g and (Path(_g) / 'test.csv').is_file():\n"
       "        return Path(_g)\n"
       "    return _one_direct('competition', [")

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    if OLD in s:
        cell["source"] = s.replace(OLD, NEW, 1).splitlines(keepends=True)
        patched += 1
assert patched == 1, f"find_root harus dipatch tepat 1x, dapat {patched}"

nb["cells"] = [{"cell_type": "code", "source": SETUP.splitlines(keepends=True),
                "metadata": {}, "outputs": [], "execution_count": None}] + nb["cells"]

OUT.mkdir(exist_ok=True)
NAME = "harvest_public_gold58.ipynb"
(OUT / NAME).write_text(json.dumps(nb, indent=1))

meta = {
    "id": "dimaspashaakrilian/rsna-knee-harvest-public-gold58",
    "title": "RSNA Knee - Harvest public gold58",
    "code_file": NAME,
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
# bawa semua sumber dataset/model/kernel dari notebook aslinya
src_meta_path = SRC_NB.parent / "kernel-metadata.json"
if src_meta_path.is_file():
    src_meta = json.loads(src_meta_path.read_text())
    for k in ("dataset_sources", "kernel_sources", "model_sources", "docker_image"):
        if k in src_meta:
            v = src_meta[k]
            # metadata aslinya memuat beberapa entri kosong "" yg bikin push ditolak
            meta[k] = [x for x in v if x] if isinstance(v, list) else v
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
print("sumber dibawa:", {k: meta.get(k) for k in ("dataset_sources", "kernel_sources", "model_sources")})
