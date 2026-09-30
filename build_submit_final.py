#!/usr/bin/env python3
"""Build rsna-knee-submit-final: pipeline publik terbaik APA ADANYA + N arm milik kita,
di-blend dgn bobot PER-LABEL dari file konfigurasi (hasil pengukuran, bukan tebakan).

Desain (alasan: RESEARCH_NOTES §31) -- bobot GLOBAL saling meniadakan (label di mana arm kita
unggul vs label di mana ia merugikan), itulah sebab §48 & §50 sama-sama 0,941. Bobot per-label
terukur: OOF 0,9220 -> 0,9296 (bootstrap 0,989) thd proksi cabang CoAtNet.

Struktur sengaja SEDERHANA & tidak menyentuh internal pipeline publik: biarkan ia menulis
submission.csv-nya, lalu tiap arm kita jalan TERPISAH lewat harness `_KE_SRC` (geometri sendiri),
lalu blend rank per-label. Pengaman disalin dari §48 (terbukti jalan mulus): gerbang waktu per arm
(pakai global T0 milik notebook), deadline internal di loop studi, try/except menyeluruh per arm --
arm yang gagal/dilewati => bobotnya otomatis kembali ke publik; submission.csv publik tak pernah rusak.

Konfigurasi (JSON):
  {"arms": [{"name": "r2c5", "file": "coatnet_fused_r1_c5.pt",
             "kernel": "dimaspashaakrilian/rsna-knee-train-round2-c5-r1",
             "img": 336, "slots": "SLOTS64", "k_eval": 32, "budget_min": 50}],
   "weights": {"r2c5": {"ACL": 0.7, ... 12 label ...}}}
Pemakaian: python3 build_submit_final.py <config.json> [slug-suffix]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-1000/-home-pasha-Documents-RSNA-Knee-Abnormality-Detection/"
               "56a96f16-dd4d-450e-802e-b23bc47ce017/scratchpad")
CFG = json.loads(Path(sys.argv[1]).read_text())
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "final"
SRC = SCRATCH / "new26/pjmathematician_rsna-knee-d4-blend/rsna-knee-d4-blend.ipynb"
OUT = REPO / f"kaggle_kernel_submit_{SUFFIX}"

SLOTS = {
    "SLOTS64": "[('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), "
               "('Coronal', 0, 8), ('Axial', -1, 12)]",
    "SLOTS44": "[('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), "
               "('Coronal', 0, 6), ('Axial', -1, 8)]",
}
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

base_ke = (REPO / "build_train_coatnet_ke_src.txt").read_text()
OLD_ARMS = ("ARMS = [{'file': 'raptor_ft_coatnet_v5_full_swa.pt', "
            "'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384, 'w': 1.0}]")
assert OLD_ARMS in base_ke and "IMG = 336" in base_ke and "K_EVAL = 42" in base_ke

arm_sources = {}
for arm in CFG["arms"]:
    ke = base_ke.replace(OLD_ARMS, (f"ARMS = [{{'file': {arm['file']!r}, "
                                    "'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', "
                                    "'res': 384, 'w': 1.0}]"), 1)
    ke = ke.replace("IMG = 336", f"IMG = {arm['img']}", 1)
    ke = ke.replace("K_EVAL = 42", f"K_EVAL = {arm['k_eval']}", 1)
    old_slots = ("SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), "
                 "('Coronal', 0, 8), ('Axial', -1, 12)]")
    assert old_slots in base_ke
    ke = ke.replace(old_slots, f"SLOTS = {SLOTS[arm['slots']]}", 1)
    # deadline internal per-studi (degradasi lembut: sisa studi tetap rank netral 0,5)
    old_loop = """        for i, sid in enumerate(test_ids):
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)"""
    assert old_loop in ke
    ke = ke.replace(old_loop, """        for i, sid in enumerate(test_ids):
            if time.time() > globals().get('_ARM_DEADLINE_ABS', float('inf')):
                print(f'[arm] deadline internal di studi {i}/{N}; sisa tetap rank netral', flush=True)
                break
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)""", 1)
    # checkpoint kita TIDAK ada di daftar pin pipeline publik, tapi harness ini sendiri yg memuatnya
    # (bukan find_weight_file milik mereka) -- find_weight_file di ke_src ini mencari global di
    # /kaggle/input, tak ada pin. Tak perlu bypass.
    arm_sources[arm["name"]] = ke

CELL = '''\
# ===== arm-arm KITA + blend PER-LABEL terukur (RESEARCH_NOTES §31, §32) =====
import time as _t, os as _os, json as _json
try:
    import pandas as _pd, numpy as _np
    _T0 = globals().get('T0', _t.time())
    _CFG = __CFG__
    _KE = __KE__
    _W = _CFG['weights']
    _pub = _pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
    _LAB = [c for c in _pub.columns if c != 'StudyInstanceUID']
    _pr = _pub[_LAB].rank(method='average', pct=True)
    _acc = {c: (1.0 * _pr[c]) for c in _LAB}          # mulai dari publik penuh
    _wsum = {c: 0.0 for c in _LAB}
    _ok = []
    for _arm in _CFG['arms']:
        _name = _arm['name']
        try:
            _sisa = 9 * 3600 - (_t.time() - _T0)
            _budget = min(_arm.get('budget_min', 50) * 60, _sisa - 1500)
            print(f"[ours:{_name}] terpakai {(_t.time()-_T0)/3600:.2f}h | jatah {_budget/60:.0f} menit",
                  flush=True)
            if _budget < 600:
                print(f"[ours:{_name}] DILEWATI (waktu tak cukup) -> bobotnya kembali ke publik", flush=True)
                continue
            _ns = {'__name__': '_ours_' + _name}
            exec(compile(_KE[_name], '<ours:' + _name + '>', 'exec'), _ns)
            _ns['_ARM_DEADLINE_ABS'] = _t.time() + _budget
            if _os.path.exists('/kaggle/working/_raptor.csv'):
                _os.replace('/kaggle/working/_raptor.csv', '/kaggle/working/_raptor_keep.csv')
            _ns['main']()
            _o = _pd.read_csv('/kaggle/working/_raptor.csv', dtype={'StudyInstanceUID': str})
            _os.replace('/kaggle/working/_raptor.csv', f'/kaggle/working/_ours_{_name}.csv')
            if _os.path.exists('/kaggle/working/_raptor_keep.csv'):
                _os.replace('/kaggle/working/_raptor_keep.csv', '/kaggle/working/_raptor.csv')
            _o = _o.set_index('StudyInstanceUID').reindex(_pub['StudyInstanceUID']).reset_index()
            assert _o[_LAB].notna().all().all(), 'arm tidak menutupi semua studi'
            _orr = _o[_LAB].rank(method='average', pct=True)
            for _c in _LAB:
                _w = float(_W[_name].get(_c, 0.0))
                _acc[_c] = _acc[_c] + _w * (_orr[_c] - _pr[_c])   # geser dari publik ke arm
                _wsum[_c] += _w
            _ok.append(_name)
            print(f"[ours:{_name}] SELESAI; total terpakai {(_t.time()-_T0)/3600:.2f}h", flush=True)
        except Exception as _e:
            import traceback as _tb
            print(f"[ours:{_name}] GAGAL -> bobotnya kembali ke publik: {type(_e).__name__}: {_e}",
                  flush=True)
            _tb.print_exc()
    if _ok:
        _out = _pub.copy()
        for _c in _LAB:
            _out[_c] = _acc[_c]
        _out[_LAB] = _out[_LAB].rank(method='average', pct=True)
        assert _np.isfinite(_out[_LAB].to_numpy(_np.float64)).all()
        _out.to_csv('/kaggle/working/submission.csv', index=False)
        print('[ours] blend per-label selesai. arm aktif:', _ok,
              '| bobot total per label:', {c: round(_wsum[c], 2) for c in _LAB}, flush=True)
    else:
        print('[ours] tidak ada arm yang berhasil -> submission.csv publik dibiarkan utuh', flush=True)
except Exception as _e:
    import traceback as _tb
    print(f"[ours] GAGAL TOTAL, submission.csv publik dibiarkan utuh: {type(_e).__name__}: {_e}", flush=True)
    _tb.print_exc()
'''.replace("__CFG__", repr(CFG)).replace("__KE__", repr(arm_sources))

nb = json.loads(SRC.read_text())
meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
nb["cells"].append({"cell_type": "code", "source": CELL.splitlines(keepends=True),
                    "metadata": {}, "outputs": [], "execution_count": None})
OUT.mkdir(exist_ok=True)
NAME = f"submit_{SUFFIX}.ipynb"
(OUT / NAME).write_text(json.dumps(nb, indent=1))

meta = dict(meta)
meta.update(id=f"dimaspashaakrilian/rsna-knee-submit-{SUFFIX}",
            title=f"RSNA Knee - Submit {SUFFIX}", code_file=NAME, is_private=True)
meta.pop("id_no", None)
for k in ("dataset_sources", "kernel_sources", "model_sources"):
    if isinstance(meta.get(k), list):
        meta[k] = [x for x in meta[k] if x]
meta["kernel_sources"] = list(dict.fromkeys(
    list(meta.get("kernel_sources", [])) + [a["kernel"] for a in CFG["arms"]]))
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
print("arms:", [a["name"] for a in CFG["arms"]])
print("kernel_sources:", meta["kernel_sources"])
