#!/usr/bin/env python3
"""Guru OOF compact: jalankan model compact S6 (bundel heliosli 209a32c8, 4 fold) pada studi LATIH yang tidak pernah
dilihat fold itu -> prediksi out-of-fold bersih untuk pseudo-label murid (STRATEGI_0960.md §3).

Dasar (terverifikasi): manifest bundel `folds: 4`, `pool_studies: 4344`, `gold_in_training: 0`; compact_train.py membagi pool
non-gold dengan fold = int(sha1(StudyInstanceUID).hexdigest()[:8], 16) % 4 ; compact_infer.py memuat model lewat
glob(<arm>_s*_f*.pt) di folder bundel -> cukup menyalin SATU file fold.

Mode:
  fold=k  : pseudo-uji = studi non-gold dengan hash % 4 == k (±1.086 studi), hanya model f{k}, eval-stride 3 (default)
  gold=1  : pseudo-uji = 58 studi gold, keempat model, eval-stride 1 (sama dengan saat uji) -> ukuran kualitas guru
Keluaran kernel: /kaggle/working/compact_oof_f{k}.csv atau compact_gold.csv (+ receipt).

Pakai:
  python3 build_compact_oof.py fold=0 owner=nxtirfan [base=r7|s6]   (ulang untuk fold 1..3; default base=r7)
  python3 build_compact_oof.py gold=1 owner=alverlfarfaraway
Basis: kaggle_base/heliosli_blend_mine_in_20261011.ipynb (sel 2 cek bundel, sel 5 setup compact, sel 6 jalankan compact).
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
GOLD = args.get("gold", "0") == "1"
FOLD = None if GOLD else int(args.get("fold", "0"))
assert GOLD or FOLD in (0, 1, 2, 3), FOLD
STRIDE = int(args.get("stride", "1" if GOLD else "3"))
OWNER = args.get("owner", "dimaspashaakrilian")
R7 = args.get("base", "r7") == "r7"          # r7 = bundel R7 8c7b666e (14 Okt, guru lebih kuat); s6 = bundel r6cnxfsss
_sfx = "-r7" if R7 else ""
NAME = args.get("name", f"compact-gold{_sfx}" if GOLD else f"compact-oof{_sfx}-f{FOLD}")
BASE = ROOT / "kaggle_base" / ("r7_heliosli_0957_20261013.ipynb" if R7 else "heliosli_blend_mine_in_20261011.ipynb")
# indeks sel di basis: (jumlah sel, sel setup compact, sel jalankan compact); R7 tak punya sel cek bundel terpisah
NCELL, I_SETUP, I_RUN = (7, 4, 5) if R7 else (8, 5, 6)
C_DS = "heliosli/rsna-knee-compact-r7cnxfsss" if R7 else "heliosli/rsna-knee-compact-r6cnxfsss"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']


def fold_of(uid, k=4):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % k


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.lstrip("\n").splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.lstrip("\n").splitlines(keepends=True)}


PSEUDO = r'''
# pseudo-uji dari studi LATIH: fold hash compact (sha1 % 4) atau 58 gold; struktur sama dengan data uji kompetisi
import os, glob, hashlib, shutil, time
import pandas as pd
NB_T0 = time.time(); RAPTOR_SECONDS = 0.0
GOLD_MODE, FOLD = __GOLD__, __FOLD__
LABELS = __LABELS__
COMP = next(p for p in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]
            if os.path.exists(p + "/train.csv"))
tr = pd.read_csv(COMP + "/train.csv")
is_gold = tr[LABELS].notna().all(axis=1)
have = set(os.listdir(COMP + "/train_series"))
if GOLD_MODE:
    uids = [u for u in tr.loc[is_gold, "StudyInstanceUID"] if u in have]
else:
    uids = [u for u in tr.loc[~is_gold, "StudyInstanceUID"] if u in have
            and int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 == FOLD]
PS = "/kaggle/working/pseudo"
shutil.rmtree(PS, ignore_errors=True); os.makedirs(PS + "/test_series")
for u in uids:
    os.symlink(f"{COMP}/train_series/{u}", f"{PS}/test_series/{u}")
trs = pd.read_csv(COMP + "/train_series.csv")
cols = [c for c in pd.read_csv(COMP + "/test_series.csv", nrows=2).columns if c in trs.columns]
trs[trs["StudyInstanceUID"].isin(set(uids))][cols].to_csv(PS + "/test_series.csv", index=False)
tcols = list(pd.read_csv(COMP + "/test.csv", nrows=1).columns)
pd.DataFrame({c: (uids if c == "StudyInstanceUID" else None) for c in tcols}).to_csv(PS + "/test.csv", index=False)
ss = pd.read_csv(COMP + "/sample_submission.csv", nrows=1)
pd.DataFrame({c: (uids if c == "StudyInstanceUID" else 0.5) for c in ss.columns}).to_csv(PS + "/sample_submission.csv", index=False)
print("PSEUDO_READY", "gold" if GOLD_MODE else "fold %d" % FOLD, "studies", len(uids), flush=True)
'''

FINAL = r'''
# simpan keluaran guru (probabilitas per temuan) + receipt
import json, shutil
import pandas as pd
_out = "/kaggle/working/" + ("compact_gold.csv" if GOLD_MODE else "compact_oof_f%d.csv" % FOLD)
import csv as _csv
_sub = pd.read_csv(WORK / "ours_sub.csv", dtype={"StudyInstanceUID": str})
assert set(_sub["StudyInstanceUID"]) == set(uids), "keluaran tidak mencakup semua studi pseudo"
_scored = set()                                       # studi yang benar-benar diskor pekerja (bukan isian rata-rata fallback)
for _f in pred.glob("worker*/predictions.csv"):
    with open(_f, encoding="utf-8") as _fh:
        for _r in _csv.DictReader(_fh):
            try:
                if all(0.0 <= float(_r[L]) <= 1.0 for L in LABELS):
                    _scored.add(_r["StudyInstanceUID"])
            except (KeyError, ValueError, TypeError):
                pass
_sub["scored"] = _sub["StudyInstanceUID"].isin(_scored).astype(int)
_sub["fold"] = -1 if GOLD_MODE else FOLD
_sub.to_csv(_out, index=False)
_rc = json.loads((WORK / "submission_receipt.json").read_text())
_rc.update({"mode": "gold" if GOLD_MODE else "fold", "fold": FOLD, "eval_stride": __STRIDE__, "studies": len(uids),
            "scored_studies": int(_sub["scored"].sum()), "models": sorted(p.name for p in BUNDLE.glob("*_f*.pt"))})
(WORK / ("receipt_" + _out.split("/")[-1].replace(".csv", ".json"))).write_text(json.dumps(_rc, indent=1))
shutil.rmtree("/kaggle/working/pseudo", ignore_errors=True)        # jangan ikutkan symlink di keluaran
print("TEACHER_DONE", _out, json.dumps({k: _rc.get(k) for k in ("status", "rows", "scored", "fallback_studies", "seconds", "models")}), flush=True)
'''

nb = json.loads(BASE.read_text(encoding="utf-8"))
cells = copy.deepcopy(nb["cells"])
assert len(cells) == NCELL

c2 = None
if not R7:
    c2 = src(cells[2])                                    # cek bundel: Raptor tidak dipakai -> lepas cek bobot nartaa
    old = "_checked_mount('rsna-knee-publication-swa-weights-20261007', 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')\n"
    assert c2.count(old) == 1
    c2 = c2.replace(old, "")

c5 = src(cells[I_SETUP])
rep = {
    "T0 = NB_T0   # [blend] the 8.3 h deadline counts from the notebook start (the Raptor stage ran first)":
        "T0 = NB_T0   # [oof] the deadline counts from the notebook start (no Raptor stage)",
    "DATA = find_mount(COMPETITION, 'test.csv')": "DATA = Path('/kaggle/working/pseudo')",
    "for name in manifest['models'] + (manifest.get('models2') or []):":
        "for name in [m for m in manifest['models'] if GOLD_MODE or m.endswith('_f%d.pt' % FOLD)]:",
    "'--eval-stride', str(manifest.get('eval_stride', 3))": "'--eval-stride', str(__STRIDE__)",
}
for a, b in rep.items():
    assert c5.count(a) == 1, a
    c5 = c5.replace(a, b)
c5 = c5.replace("__STRIDE__", str(STRIDE))

c6 = src(cells[I_RUN])                                    # sudah lenient di basis (--max-fallback-fraction 1.0); dipakai apa adanya
assert c6.count("'--max-fallback-fraction', '1.0'") == 1 and "ours_sub.csv" in c6

pseudo = PSEUDO.replace("__GOLD__", "True" if GOLD else "False").replace("__FOLD__", repr(FOLD)).replace("__LABELS__", repr(LABELS))
final = FINAL.replace("__STRIDE__", str(STRIDE))
title = "# Guru OOF compact — " + ("58 gold, 4 model" if GOLD else f"fold {FOLD}, model f{FOLD} saja") + f", eval-stride {STRIDE}\n" \
        f"Dibangun oleh `build_compact_oof.py` dari notebook publik heliosli ({'R7' if R7 else 'S6'}). Lihat `STRATEGI_0960.md` §3.\n"
new_cells = [md(title), cells[1], code(pseudo)]
body = [s for s in (c2, c5, c6, final) if s is not None]
for s in body:
    compile(s, "cell", "exec")
new_cells += [code(s) for s in body]
compile(pseudo, "pseudo", "exec")
nb["cells"] = new_cells

slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
meta = {
    "id": f"{OWNER}/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": ["gpu"], "dataset_sources": [C_DS, "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
    "kernel_sources": [], "model_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
    "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# uji lokal: pembagian fold pada train.csv lokal (non-gold)
tr = ROOT / "reference_metadata" / "train.csv"
if tr.exists():
    import csv
    rows = list(csv.DictReader(open(tr, encoding="utf-8")))
    nongold = [r["StudyInstanceUID"] for r in rows if not all(r[L] != "" for L in LABELS)]
    gold = [r["StudyInstanceUID"] for r in rows if all(r[L] != "" for L in LABELS)]
    counts = [sum(1 for u in nongold if fold_of(u) == k) for k in range(4)]
    print("cek lokal: non-gold", len(nongold), "| per fold", counts, "| gold", len(gold))
print("ditulis:", out, "| mode", "gold" if GOLD else f"fold {FOLD}", "| stride", STRIDE, "| id", meta["id"])
