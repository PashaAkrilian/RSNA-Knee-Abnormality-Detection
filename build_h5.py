#!/usr/bin/env python3
"""h5: notebook h4 (salinan PERSIS notebook publik heliosli "=0.9586": Raptor 384m + 224 polos, R6 + flip TTA, bobot V2,
Part B reader goodpjw per-temuan 0,003-0,035) + dua kaki kita yang terbukti di g6a/g6c:

- S6F nartaa (ConvNeXt-S all-data) bobot 0,5 DI DALAM separuh Raptor (struktur g6a/g6c; gold 58 +0,002 bersama murid).
- Murid ConvNeXt-T 2.5D kita bobot 0,4 DI DALAM slot compact. Set 8 model yang DIMILIKI akun alver (output kernel):
  r4 fold0-3 (`rsna-knee-student-r4-a/b`) + r2a full-data fold6-9 (`rsna-knee-student-r2a-full/full2`).
  Gold 58 dari OOF latih = 0,9332 (r2a 4 fold akun utama 0,9334; rho 0,990) -> dataset murid akun utama tak diperlukan.
- R6 eval-stride 2 (gold 0,9377 >= stride 1 0,9366, rho 0,998; ±35 % lebih cepat) agar S6F + murid + reader muat < 9 j.
- Part B reader: penjaga waktu 8,2 j -> 7,9 j (reader ±0,5 j harus selesai sebelum batas 9 j).
- Mode commit: cek gold murid DIMATIKAN (hemat kuota GPU alver; jalur decode/bobot yg sama sudah tervalidasi di g17/h4m).

Templat kaki (STUDENT, STAGE0, LOAD, S6F_*) diambil apa adanya dari build_r7murid.py (dipakai g6c/g10a/g12).
Pakai: python3 build_h5.py [name=h5] [wst=0.4] [ws6f=0.5] [wc=v2|v1] [r6stride=2|1] [budget=5400]
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
sys.argv = sys.argv[:1]          # build_r7murid/build_student_dino membaca sys.argv saat import: pakai default mereka
import build_r7murid as b7       # noqa: E402

NAME = ARGS.get("name", "h5")
WST = float(ARGS.get("wst", "0.4"))
WS6F = float(ARGS.get("ws6f", "0.5"))
WCK = ARGS.get("wc", "v2")
R6STRIDE = ARGS.get("r6stride", "2")
BUDGET = int(ARGS.get("budget", "5400"))
assert 0.0 < WST <= 0.5 and 0.0 < WS6F <= 0.6 and WCK in ("v1", "v2") and R6STRIDE in ("1", "2") and 1800 <= BUDGET <= 7200
OWNER = "alverlfarfaraway"
SLUG = "rsna-knee-" + NAME
BASE = ROOT / "kaggle_base" / "h4_heliosli_0958_exact_20261020.ipynb"
BASE_META = ROOT / "kaggle_base" / "h4_heliosli_0958_exact_20261020.meta.json"
STUDENT_KERNELS = [f"{OWNER}/rsna-knee-student-r4-a", f"{OWNER}/rsna-knee-student-r4-b",
                   f"{OWNER}/rsna-knee-student-r2a-full", f"{OWNER}/rsna-knee-student-r2a-full2"]
WC_LINE = {
    "v2": "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}   # [V2] per-finding Raptor weight (ours = 1 - it); every other finding uses W_RAPTOR\n",
    "v1": "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   # [h5 wc=v1] heliosli 16 Okt weights (goodpjw v27 / g6a)\n",
}

# murid: 8 fold dari output kernel alver (runs_student/fold*.pt). Pencarian kedalaman 1-4 TANPA rekursi (pohon DICOM besar).
FOLD_FIND = (
    "_KS = ('rsna-knee-student-r4-a', 'rsna-knee-student-r4-b', 'rsna-knee-student-r2a-full', 'rsna-knee-student-r2a-full2')\n"
    "_cand = []\n"
    "for _pat in ('/*', '/*/*', '/*/*/*', '/*/*/*/*'):\n"
    "    _cand += glob.glob(I + _pat + '/runs_student/fold[0-9].pt')\n"
    "FOLDS = sorted({os.path.basename(p): p for p in _cand if any('/' + k + '/' in p for k in _KS)}.values())   # [h5] r4 f0-3 + r2a-full f6-9\n"
    "FOLD_DIR = os.path.dirname(FOLDS[0]) if FOLDS else ''\n"
    "assert len(FOLDS) == 8, FOLDS\n"
)


def src(c):
    return "".join(c["source"])


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def one(cells, needle):
    k = [i for i, c in enumerate(cells) if needle in src(c)]
    assert len(k) == 1, (needle, k)
    return k[0]


def sub1(s, a, b):
    assert s.count(a) == 1, ("anchor not unique", a[:80], s.count(a))
    return s.replace(a, b)


def build():
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    cells = copy.deepcopy(nb["cells"])
    i_st1 = one(cells, "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py']")
    i_r6 = one(cells, "BUNDLE_ID = 'fe055406fad6a12f'")
    i_run = one(cells, "deadline = T0 + DEADLINE_HOURS * 3600")
    i_bl = one(cells, "# Stage 3: per-finding rank blend")
    i_pb = one(cells, "# Part B (V2): goodpjw2008's 2.5D ConvNeXt MIL reader")
    assert i_st1 < i_r6 < i_run < i_bl < i_pb == len(cells) - 1, (i_st1, i_r6, i_run, i_bl, i_pb)

    # Stage 0 (murid) sebelum Raptor; jam NB_T0 mulai di Stage 0 (deadline R6 8,3 j & guard S6F/Part B ikut menghitungnya)
    student = b7.STUDENT.replace("__BUDGET__", str(BUDGET)).replace("__DECODE__", b7.DECODE)
    student = sub1(student, "FOLD_DIR = mount('__FOLDSLUG__', 'fold0.pt')\n", "")
    student = sub1(student, "FOLDS = sorted(glob.glob(FOLD_DIR + '/fold[0-9].pt'))\nassert len(FOLDS) >= 2, FOLDS   # >=2: murid r2 dua fold boleh diuji lebih awal\n", FOLD_FIND)
    student = sub1(student, "if len(test_ids) < 50:      # commit run", "if False:   # [h5] gold check off (alver GPU quota); was: len(test_ids) < 50 -- commit run")
    stage0 = b7.STAGE0.replace("__TIMEOUT__", str(BUDGET + 900))
    c3 = sub1(src(cells[i_st1]), "NB_T0 = time.time()\n", "NB_T0 = globals().get('NB_T0') or time.time()   # [h5] Stage 0 started the clock\n")

    # R6 eval-stride
    c_r6 = src(cells[i_r6])
    if R6STRIDE != "1":
        c_r6 = sub1(c_r6, "'--eval-stride', str(manifest.get('eval_stride', 3))", f"'--eval-stride', '{R6STRIDE}'")

    # Stage 3: S6F di separuh Raptor + murid di slot compact (sebelum bobot per temuan)
    c6 = src(cells[i_bl])
    a_out, a_wr = "out = np.zeros((n, len(labels)))\n", "    wr = W_RAPTOR_C.get(c, W_RAPTOR)\n"
    wcl = [l for l in c6.splitlines(keepends=True) if l.startswith("W_RAPTOR_C = {")]
    assert len(wcl) == 1, wcl
    c6 = c6.replace(wcl[0], WC_LINE[WCK])
    load_s6f = b7.S6F_LOAD.replace("__WS6F__", repr(WS6F))
    load_stu = b7.LOAD.replace("__W__", repr(WST)).replace("__SLOT__", "compact")
    c6 = sub1(c6, a_out, load_s6f + load_stu + a_out)
    c6 = sub1(c6, a_wr, b7.S6F_IN_LOOP_RAPTOR + b7.IN_LOOP_COMPACT + a_wr)
    rec = "'rank_basis': 'each model over its own non-failed rows'}"
    c6 = sub1(c6, rec, rec[:-1] + ", 'murid': murid, 's6f': s6f, 'h5': {'r6stride': %r, 'wc': %r}}" % (R6STRIDE, WCK))

    # Part B: penjaga waktu lebih ketat
    c_pb = sub1(src(cells[i_pb]), "> 8.2 * 3600:", "> 7.9 * 3600:   # [h5] was 8.2 h: the reader (~0.5 h) must end well before 9 h")

    title = (f"# RSNA Knee: {NAME}\n\nheliosli's public \"=0.9586\" notebook (h4, exact: nartaa 384 mirror + 224 plain, heliosli R6 with flip TTA, "
             f"per-finding weights {WCK.upper()}, goodpjw's reader as Part B), plus: nartaa's public S6 ConvNeXt (all data) at {WS6F} inside "
             f"the Raptor half; our ConvNeXt-T 2.5D student (8 models: r4 folds 0-3 + r2a full-data 6-9) at {WST} inside the compact slot; "
             f"R6 eval stride {R6STRIDE}. Built by `build_h5.py`.\n")
    new = ([md(title)] + cells[1:i_st1] + [code(student), code(stage0), code(c3)] + cells[i_st1 + 1:i_r6] + [code(c_r6)]
           + cells[i_r6 + 1:i_run + 1] + [code(b7.s6f_script()), code(b7.S6F_RUN)] + cells[i_run + 1:i_bl] + [code(c6), code(c_pb)])
    for k, c in enumerate(new):
        if c["cell_type"] == "code":
            s = src(c)
            compile(s.split("\n", 1)[1] if s.startswith("%%") else s, f"<cell{k}>", "exec")
    nb["cells"] = new
    out = ROOT / f"kaggle_kernel_{NAME}"
    out.mkdir(exist_ok=True)
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(BASE_META.read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb", is_private=True, enable_gpu=True, enable_internet=False)
    meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d] + [b7.S6F_DS]
    meta["kernel_sources"] = list(dict.fromkeys(meta.get("kernel_sources", []) + STUDENT_KERNELS))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, len(new)


if __name__ == "__main__":
    o, n = build()
    print("ditulis:", o, "| cells", n, "| id", f"{OWNER}/{SLUG}", "| wst", WST, "| ws6f", WS6F, "| wc", WCK, "| r6stride", R6STRIDE,
          "| budget", BUDGET)
