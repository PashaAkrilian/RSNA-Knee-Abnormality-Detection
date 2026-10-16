#!/usr/bin/env python3
"""Jalankan pipeline SUBMISI apa adanya pada 58 studi GOLD (split train) dgn root kompetisi palsu /tmp/goldcomp.
Tiap tahap (Raptor 384/224 + cermin, R7 ± flip, S6F, murid, blend) memakai jalur test-nya sendiri tanpa ubahan selain
penemu root -> prediksi gold SETIA utk semua komponen (analisis lokal `gold_eval_blend.py`).
Pakai: python3 build_gold_full.py src=r7v3sr name=gold-nf   (src = kernel kaggle_kernel_<src> yg sudah dibangun)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SRC, NAME = args["src"], args["name"]
G = "/tmp/goldcomp"

SETUP = r'''# [gold] fake competition root: the 58 radiologist-labelled train studies as "test"; every stage then runs its own test path
import os, pandas as pd
REAL = next(d for d in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
            if os.path.exists(d + '/train.csv'))
G = '__G__'
os.makedirs(G + '/test_series', exist_ok=True)
_L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
      "Baker's", 'Contusion', 'Fracture']
_tr = pd.read_csv(REAL + '/train.csv', dtype={'StudyInstanceUID': str})
_gold = _tr.loc[_tr[_L].notna().all(axis=1), 'StudyInstanceUID'].tolist()
_ts = pd.read_csv(REAL + '/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
pd.DataFrame({'StudyInstanceUID': _gold}).to_csv(G + '/test.csv', index=False)
_ss = pd.DataFrame({'StudyInstanceUID': _gold})
for _c in _L:
    _ss[_c] = 0.5
_ss.to_csv(G + '/sample_submission.csv', index=False)
_ts[_ts.StudyInstanceUID.isin(set(_gold))].to_csv(G + '/test_series.csv', index=False)
for _s in _gold:
    if not os.path.exists(f'{G}/test_series/{_s}'):
        os.symlink(f'{REAL}/train_series/{_s}', f'{G}/test_series/{_s}')
for _f in ('train.csv', 'train_series.csv', 'train_series'):
    if not os.path.exists(f'{G}/{_f}'):
        os.symlink(f'{REAL}/{_f}', f'{G}/{_f}')
assert len(_gold) == 58 and all(os.path.isdir(f'{G}/test_series/{s}') for s in _gold)
print('GOLDCOMP', len(_gold), 'studies |', len(pd.read_csv(G + '/test_series.csv')), 'series', flush=True)
'''.replace("__G__", G)

PATCH = [  # (pola, ganti, jumlah sel yg harus kena)
    ('    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",',
     f'    cands = ["{G}", "/kaggle/input/competitions/rsna-knee-abnormality-detection",   # [gold]', 2),
    ("COMP = mount('rsna-knee-abnormality-detection', 'test.csv')", f"COMP = '{G}'   # [gold]", (0, 1)),   # murid opsional
    ("DATA = find_mount(COMPETITION, 'test.csv')", f"DATA = Path('{G}')   # [gold]", 1),
    ('I.comp = find_mount(COMP, "test.csv")', f'I.comp = Path("{G}")   # [gold]', (0, 1)),   # S6F opsional
]

if __name__ == "__main__":
    src_dir = ROOT / f"kaggle_kernel_{SRC}"
    nb = json.loads(next(src_dir.glob("*.ipynb")).read_text(encoding="utf-8"))
    hits = {p[0]: 0 for p in PATCH}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        for a, b, _ in PATCH:
            if a in s:
                assert s.count(a) == 1, a
                s = s.replace(a, b)
                hits[a] += 1
        c["source"] = s.splitlines(keepends=True)
    for a, _, want in PATCH:
        assert hits[a] in (want if isinstance(want, tuple) else (want,)), (a, hits[a], want)
    nb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": SETUP.splitlines(keepends=True)})
    out = ROOT / f"kaggle_kernel_{NAME}"
    out.mkdir(exist_ok=True)
    slug = "rsna-knee-" + NAME
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.pop("id_no", None)   # salinan kernel orang lain: id numerik pemilik asal -> 403
    meta.update(id=f"dimaspashaakrilian/{slug}", title="RSNA Knee " + NAME, code_file=f"{slug}.ipynb", is_private=True)   # notebook kompetisi tak boleh publik
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ditulis:", out, "| patch:", {a[:40]: n for a, n in hits.items()})
