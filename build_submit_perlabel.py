#!/usr/bin/env python3
"""Build rsna-knee-submit-perlabel: pipeline publik apa adanya + arm v3 kita, di-blend dgn bobot
PER-LABEL hasil PENGUKURAN (bukan satu bobot global yang ditebak).

KENAPA DESAIN INI (akar penyebab 2 hasil null akhirnya ketemu, RESEARCH_NOTES §31):
Nested-CV di 58 gold menunjukkan bobot optimal v3 SANGAT berbeda antar label —
Baker's 0,85 · PF OA 0,80 · Synovitis 0,75 · ACL 0,69 · Fracture 0,67, TAPI
Lateral Meniscus 0,01 · MCL 0,03 · Lateral OA 0,05.
Satu bobot global memaksa kompromi antara label di mana v3 jauh lebih baik dan label di mana v3
merugikan; keduanya saling meniadakan. Itulah kenapa §48 (α=0,15) dan §50 (w=0,35) SAMA-SAMA
menghasilkan 0,941 persis. Dgn bobot per-label, OOF naik 0,9220 -> 0,9296 (bootstrap P=0,989).

Struktur sengaja dibuat SEDERHANA dan tidak menyentuh internal pipeline mereka: biarkan pipeline
publik menghasilkan submission.csv-nya sendiri, jalankan arm kita terpisah, lalu blend per label.
Pola pengaman (self-gating waktu + try/except menyeluruh) disalin dari §48 yang terbukti berjalan
mulus — yang keliru di sana cuma skema bobotnya, bukan mekanikanya.

CATATAN BOBOT: nilai di PERLABEL_W diukur terhadap PROKSI cabang CoAtNet (0,75·c5+0,25·c4).
Setelah kernel panen gold-58 selesai, ganti dgn bobot yang diukur terhadap ensemble publik PENUH —
itu target yang sebenarnya.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-1000/-home-pasha-Documents-RSNA-Knee-Abnormality-Detection/"
               "56a96f16-dd4d-450e-802e-b23bc47ce017/scratchpad")
SRC = SCRATCH / "new26/pjmathematician_rsna-knee-d4-blend/rsna-knee-d4-blend.ipynb"
OUT = REPO / "kaggle_kernel_submit_perlabel"

# bobot arm KITA per label (0 = murni publik, 1 = murni punya kita) -- lihat catatan di docstring
PERLABEL_W = {
    "ACL": 0.69, "MCL": 0.03, "Medial Meniscus": 0.39, "Lateral Meniscus": 0.01,
    "Medial OA": 0.35, "Lateral OA": 0.05, "PF OA": 0.80, "Effusion": 0.39,
    "Synovitis": 0.75, "Baker's": 0.85, "Contusion": 0.54, "Fracture": 0.67,
}

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()
old_arms = ("ARMS = [{'file': 'raptor_ft_coatnet_v5_full_swa.pt', "
            "'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384, 'w': 1.0}]")
assert old_arms in ke_src
ke_src = ke_src.replace(old_arms, ("ARMS = [{'file': 'coatnet_rawmean_ft_v3.pt', "
                                   "'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', "
                                   "'res': 384, 'w': 1.0}]"), 1)
assert "K_EVAL = 42" in ke_src
ke_src = ke_src.replace("K_EVAL = 42", "K_EVAL = 24", 1)   # hemat waktu; arm ini tambahan

CELL = '''\
# ===== arm KITA (v3) + blend PER-LABEL dgn bobot hasil pengukuran (RESEARCH_NOTES §31) =====
# Bobot per label, BUKAN satu bobot global: itu yang bikin §48 (0,15) & §50 (0,35) sama-sama nihil.
import time as _t
try:
    _T0 = globals().get('T0', _t.time())
    _sisa = 9 * 3600 - (_t.time() - _T0)
    _budget = min(3600, _sisa - 1200)
    print(f"[ours] terpakai {(_t.time()-_T0)/3600:.2f}h | jatah arm kita {_budget/60:.0f} menit",
          flush=True)
    if _budget < 600:
        print("[ours] DILEWATI - sisa waktu tidak cukup, submission.csv publik dibiarkan utuh",
              flush=True)
    else:
        import pandas as _pd, numpy as _np, os as _os
        _W = __PERLABEL_W__
        _SRC = __KE_SRC__
        _NS = {'__name__': '_ours'}
        exec(compile(_SRC, '<ours>', 'exec'), _NS)
        _NS['_V3_DEADLINE_ABS'] = _t.time() + _budget
        if _os.path.exists('/kaggle/working/_raptor.csv'):
            _os.replace('/kaggle/working/_raptor.csv', '/kaggle/working/_raptor_public.csv')
        _NS['main']()
        _ours = _pd.read_csv('/kaggle/working/_raptor.csv', dtype={'StudyInstanceUID': str})
        _os.replace('/kaggle/working/_raptor.csv', '/kaggle/working/_ours.csv')
        if _os.path.exists('/kaggle/working/_raptor_public.csv'):
            _os.replace('/kaggle/working/_raptor_public.csv', '/kaggle/working/_raptor.csv')

        _pub = _pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
        _LAB = [c for c in _pub.columns if c != 'StudyInstanceUID']
        _ours = _ours.set_index('StudyInstanceUID').reindex(_pub['StudyInstanceUID']).reset_index()
        assert _ours[_LAB].notna().all().all(), 'arm kita tidak menutupi semua studi'

        _pr = _pub[_LAB].rank(method='average', pct=True)
        _orr = _ours[_LAB].rank(method='average', pct=True)
        _out = _pub.copy()
        for _c in _LAB:
            _w = float(_W.get(_c, 0.0))
            _out[_c] = (1.0 - _w) * _pr[_c] + _w * _orr[_c]
        assert _np.isfinite(_out[_LAB].to_numpy(_np.float64)).all()
        _out.to_csv('/kaggle/working/submission.csv', index=False)
        print('[ours] blend PER-LABEL selesai; bobot dipakai:', _W, flush=True)
        print(_out.head().to_string(index=False), flush=True)
except Exception as _e:
    import traceback as _tb
    print(f"[ours] GAGAL, submission.csv publik dibiarkan utuh: {type(_e).__name__}: {_e}",
          flush=True)
    _tb.print_exc()
'''.replace("__PERLABEL_W__", repr(PERLABEL_W)).replace("__KE_SRC__", repr(ke_src))

nb = json.loads(SRC.read_text())
meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
nb["cells"].append({"cell_type": "code", "source": CELL.splitlines(keepends=True),
                    "metadata": {}, "outputs": [], "execution_count": None})

OUT.mkdir(exist_ok=True)
NAME = "submit_perlabel.ipynb"
(OUT / NAME).write_text(json.dumps(nb, indent=1))

meta = dict(meta)
meta.update(id="dimaspashaakrilian/rsna-knee-submit-perlabel",
            title="RSNA Knee - Submit perlabel", code_file=NAME, is_private=True)
meta.pop("id_no", None)
for k in ("dataset_sources", "kernel_sources", "model_sources"):
    if isinstance(meta.get(k), list):
        meta[k] = [x for x in meta[k] if x]
meta["kernel_sources"] = list(meta.get("kernel_sources", [])) + [
    "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean"]
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
print("bobot per-label:", PERLABEL_W)
