#!/usr/bin/env python3
"""Bangun notebook submisi: Apex (nartaa r384 + reader goodpjw) + kaki tambahan MaVIT (hengck23) dan/atau OrthoFoundation fold0 (heliosli).

Tiap kaki dijalankan di akhir notebook sebagai subproses terisolasi per GPU; bila satu kaki gagal, kaki itu dilewati dan
keluaran Apex (atau Apex+kaki lain) tetap sah. Rank-blend per temuan: final = (1 - sum(w)) * rank(Apex) + sum(w_i * rank(kaki_i)).
Tidak ada bobot yang di-fit di emas-58.

Pemakaian (key=value):
  python3 build_submit_legs.py name=apexlegs legs=mavit,of wm=0.20 wo=0.25
  python3 build_submit_legs.py name=apexof legs=of wo=0.25
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BASE = ROOT / "kaggle_base" / "apex_grandmaster_stack_20261008.ipynb"
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "apexlegs")
LEGS = [x for x in args.get("legs", "mavit,of").split(",") if x]
WM = float(args.get("wm", "0.20"))
WO = float(args.get("wo", "0.25"))

COMMON = r'''
import os as _l_os, sys as _l_sys, glob as _l_glob, json as _l_json, time as _l_time
import shutil as _l_shutil, subprocess as _l_sp, textwrap as _l_tw, hashlib as _l_hl, traceback as _l_tb
import numpy as _l_np, pandas as _l_pd, torch as _l_torch
import gc as _l_gc
_l_gc.collect()
if _l_torch.cuda.is_available():
    _l_torch.cuda.empty_cache()
_l_pub = "/kaggle/working/submission.csv"
_l_bak = "/kaggle/working/_apex_before_legs.csv"
if not _l_os.path.exists(_l_bak):
    _l_shutil.copy(_l_pub, _l_bak)
LEG_OUT = {}      # name -> (csv path, weight)
def _l_find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(_l_glob.glob(pat)):
            if _l_os.path.exists(_l_os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(f"{name} not found in /kaggle/input")
_l_data = _l_find("test_series.csv")
_l_df0 = _l_pd.read_csv(_l_bak, dtype={"StudyInstanceUID": str})
_l_ids = _l_df0.StudyInstanceUID.tolist()
_l_labels = [c for c in _l_df0.columns if c != "StudyInstanceUID"]
open("/kaggle/working/_leg_ids.txt", "w").write("\n".join(_l_ids))
_l_env = dict(_l_os.environ)
_l_own = "/kaggle/working/_own_env"
if _l_os.path.isdir(_l_own):
    _l_env["PYTHONPATH"] = _l_own + _l_os.pathsep + _l_env.get("PYTHONPATH", "")
_l_ng = max(1, min(2, _l_torch.cuda.device_count()))
print("[LEGS] studies", len(_l_ids), "| gpus", _l_ng, flush=True)

def _l_run_shards(script, extra_args, tag, timeout, env=None):
    procs = []
    for s in range(_l_ng):
        e = dict(env or _l_env)
        e["CUDA_VISIBLE_DEVICES"] = str(s)
        lg = open(f"/kaggle/working/_{tag}_shard{s}.log", "w")
        procs.append((_l_sp.Popen([_l_sys.executable, "-u", script] + [str(a) for a in extra_args] + [str(s), str(_l_ng)],
                                  env=e, stdout=lg, stderr=_l_sp.STDOUT), lg))
    for p, lg in procs:
        p.wait(timeout=timeout)
        lg.close()
    for s in range(_l_ng):
        print(f"--- {tag} shard{s} log tail ---\n" + "".join(open(f"/kaggle/working/_{tag}_shard{s}.log").readlines()[-8:]), flush=True)
    for p, _ in procs:
        assert p.returncode == 0, f"{tag} shard failed rc={p.returncode}"
'''

MAVIT = r'''
# ---------------- leg: MaVIT (hengck23 288px) ----------------
_l_t0 = _l_time.time()
try:
    _hk = _l_find("my_fastread_knee.py")
    _ck = _l_os.path.join(_hk, "00000008.pth")
    assert _l_os.path.isfile(_ck), _ck
    _script = _l_tw.dedent("""
        import os, sys, json, time
        import numpy as np, pandas as pd, torch
        hk, data, ck, ids_file, shard, nsh = sys.argv[1:7]
        shard, nsh = int(shard), int(nsh)
        sys.path.insert(0, hk)
        from model import KneeModel
        from my_fastread_knee import make_knee_dataset_reader
        LABEL = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
                 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
        ids = open(ids_file).read().split("\\n")[shard::nsh]
        SERIES = pd.read_csv(os.path.join(data, "test_series.csv"))
        DDIR = os.path.join(data, "test_series")
        saved = torch.load(ck, map_location="cpu", weights_only=False)
        model = KneeModel(pretrained=False)
        print("load", model.load_state_dict(saved["model_state_dict"], strict=False), flush=True)
        model = model.cuda().eval()
        reader = make_knee_dataset_reader(out_size=288, num_slice=8, percentile=(1.0, 99.0), crop_mm=130.0,
                                          io_threads=12, crop_only_if_nonsquare=True)
        rows, fails, t0 = [], 0, time.time()
        for i, sid in enumerate(ids):
            try:
                r = reader(sid, SERIES[SERIES["StudyInstanceUID"] == sid], DDIR, "valid")
                vol = torch.from_numpy(r["volume"]).unsqueeze(0).byte().cuda()
                vm = torch.from_numpy(r["view_mask"]).unsqueeze(0).bool().cuda()
                sl = torch.from_numpy(r["slice"]).unsqueeze(0).float().cuda()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    p = torch.sigmoid(model.infer(vol, vm, sl)).float().cpu().numpy()[0]
            except Exception as e:
                fails += 1
                if fails <= 3:
                    print("FAIL", sid[:24], repr(e)[:200], flush=True)
                p = np.full(12, np.nan, np.float32)
            rows.append(p)
            if (i + 1) % 100 == 0:
                print("mavit shard", shard, i + 1, "/", len(ids), "%.2fs/study" % ((time.time() - t0) / (i + 1)), flush=True)
        pd.DataFrame(np.array(rows), index=ids, columns=LABEL).rename_axis("StudyInstanceUID").to_csv(f"/kaggle/working/_leg_mavit_s{shard}.csv")
        print("done mavit shard", shard, "fails", fails, "sec/study %.2f" % ((time.time() - t0) / max(1, len(ids))), flush=True)
    """)
    open("/kaggle/working/_leg_mavit.py", "w").write(_script)
    _l_run_shards("/kaggle/working/_leg_mavit.py", [_hk, _l_data, _ck, "/kaggle/working/_leg_ids.txt"], "mavit", 3 * 3600)
    _d = _l_pd.concat([_l_pd.read_csv(f"/kaggle/working/_leg_mavit_s{s}.csv", dtype={"StudyInstanceUID": str}) for s in range(_l_ng)]
                      ).set_index("StudyInstanceUID").reindex(_l_ids)[_l_labels]
    _fail = float(_d.isna().all(axis=1).mean())
    print(f"[MaVIT] failed-study fraction {_fail:.4f} | {_l_time.time() - _l_t0:.0f}s", flush=True)
    assert _fail < 0.05, "too many MaVIT failures"
    _d.to_csv("/kaggle/working/_leg_mavit.csv")
    LEG_OUT["mavit"] = ("/kaggle/working/_leg_mavit.csv", __WM__)
except Exception as _e:
    _l_tb.print_exc()
    print(f"[MaVIT EXCEPTION] {type(_e).__name__}: {_e} -> leg skipped", flush=True)
'''

OF = r'''
# ---------------- leg: OrthoFoundation-L fold0 (heliosli) ----------------
_l_t0 = _l_time.time()
try:
    _OF = _l_find("weight_parts.json")
    _UL = _l_find("assets/bus.jpg")          # resource yaml ultralytics 8.4.166 (dataset sendiri; paket vendored tanpa yaml)
    _MD = "/kaggle/working/of"
    _CC = "/kaggle/working/cloud_code"
    _l_os.makedirs(_MD, exist_ok=True)
    if not _l_os.path.exists(_CC):
        _l_shutil.copytree(f"{_OF}/cloud_code", _CC)
    _l_shutil.copytree(_UL, f"{_CC}/training/vendor_runtime/kneexnet/ultralytics", dirs_exist_ok=True)
    assert _l_os.path.exists(f"{_CC}/training/vendor_runtime/kneexnet/ultralytics/cfg/default.yaml")
    _h = _l_hl.sha256()
    with open(f"{_MD}/best.pt", "wb") as _out:
        for _p in sorted(_l_glob.glob(f"{_OF}/weights.part*")):
            with open(_p, "rb") as _f:
                while True:
                    _b = _f.read(1 << 24)
                    if not _b:
                        break
                    _out.write(_b)
                    _h.update(_b)
    assert _h.hexdigest().startswith("4f424cc7"), "OF best.pt sha mismatch"
    if not _l_os.path.exists(f"{_MD}/anatomy.pt"):
        _l_shutil.copy2(f"{_OF}/anatomy.pt", f"{_MD}/anatomy.pt")
    if not _l_os.path.exists(f"{_MD}/kneexnet"):
        _l_shutil.copytree(f"{_OF}/dependencies/kneexnet", f"{_MD}/kneexnet")
    _env = dict(_l_env)
    _tgt = "/kaggle/working/_of_env"
    for _w in sorted(_l_glob.glob(f"{_OF}/dependencies/wheels/*.whl")):
        _l_sp.run([_l_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _tgt, _w], check=False)
    _env["PYTHONPATH"] = _tgt + _l_os.pathsep + _CC + _l_os.pathsep + _env.get("PYTHONPATH", "")
    print("[OF] assembled", _l_os.path.getsize(f"{_MD}/best.pt"), "bytes |", f"{_l_time.time() - _l_t0:.0f}s", flush=True)
    _script = _l_tw.dedent("""
        import os, sys, time, shutil
        import pandas as pd
        data, model_dir, ids_file, deadline, shard, nsh = sys.argv[1:7]
        shard, nsh, deadline = int(shard), int(nsh), float(deadline)
        from training.infer import run_inference
        ids = open(ids_file).read().split("\\n")[shard::nsh]
        CH = 30
        frames, t0, fails = [], time.time(), 0
        for k in range(0, len(ids), CH):
            if time.time() > deadline:
                print("DEADLINE reached; skipping remaining", len(ids) - k, "studies", flush=True)
                break
            chunk = ids[k:k + CH]
            out = f"/kaggle/working/_leg_of_out{shard}_{k}"
            try:
                fr = run_inference(data, f"{model_dir}/best.pt", out, anatomy_weights=f"{model_dir}/anatomy.pt", workers=2, study_ids=chunk)
                frames.append(fr)
            except Exception as e:
                fails += len(chunk)
                print("CHUNK FAIL", k, repr(e)[:300], flush=True)
            finally:
                shutil.rmtree(out, ignore_errors=True)
            done = min(k + CH, len(ids))
            print("of shard", shard, done, "/", len(ids), "%.1fs/study" % ((time.time() - t0) / done), "fails", fails, flush=True)
        if frames:
            pd.concat(frames).to_csv(f"/kaggle/working/_leg_of_s{shard}.csv", index=False)
        print("done of shard", shard, "frames", len(frames), "fails", fails, flush=True)
    """)
    open("/kaggle/working/_leg_of.py", "w").write(_script)
    _deadline = _l_time.time() + 7.0 * 3600
    _l_run_shards("/kaggle/working/_leg_of.py", [_l_data, _MD, "/kaggle/working/_leg_ids.txt", _deadline], "of", 8 * 3600, env=_env)
    _parts = [f"/kaggle/working/_leg_of_s{s}.csv" for s in range(_l_ng) if _l_os.path.exists(f"/kaggle/working/_leg_of_s{s}.csv")]
    assert _parts, "no OF output"
    _d = _l_pd.concat([_l_pd.read_csv(p, dtype={"StudyInstanceUID": str}) for p in _parts]
                      ).set_index("StudyInstanceUID").reindex(_l_ids)[_l_labels]
    _fail = float(_d.isna().all(axis=1).mean())
    print(f"[OF] failed/missing-study fraction {_fail:.4f} | {_l_time.time() - _l_t0:.0f}s", flush=True)
    assert _fail < 0.10, "too many OF failures"
    _d.to_csv("/kaggle/working/_leg_of.csv")
    LEG_OUT["of"] = ("/kaggle/working/_leg_of.csv", __WO__)
except Exception as _e:
    _l_tb.print_exc()
    print(f"[OF EXCEPTION] {type(_e).__name__}: {_e} -> leg skipped", flush=True)
'''

BLEND = r'''
# ---------------- blend: Apex + successful legs (rank-mean per finding) ----------------
try:
    if not LEG_OUT:
        raise RuntimeError("no leg succeeded")
    _apex = _l_pd.read_csv(_l_bak, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[_l_ids, _l_labels]
    _ra = _apex.rank(method="average", pct=True)
    _wsum = sum(w for _, w in LEG_OUT.values())
    _fused = (1.0 - _wsum) * _ra
    for _name, (_path, _w) in LEG_OUT.items():
        _d = _l_pd.read_csv(_path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[_l_ids, _l_labels]
        _rl = _d.rank(method="average", pct=True).fillna(_ra)
        _fused = _fused + _w * _rl
        if len(_l_ids) >= 20:
            print(f"[BLEND] {_name} w={_w:.2f} mean corr(apex,leg)=" +
                  f"{_l_np.mean([_l_np.corrcoef(_ra[c], _rl[c])[0, 1] for c in _l_labels]):.3f}", flush=True)
    _fused = _fused.rank(method="average", pct=True)
    assert _fused.notna().all().all(), "fused NaN"
    assert (_fused.nunique() > 1).all() or len(_l_ids) < 5, "fused constant"
    _fused.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(_l_pub, index=False)
    print(f"[BLEND SUCCESS] Apex + {list(LEG_OUT)} -> {_l_pub}", flush=True)
except Exception as _e:
    _l_tb.print_exc()
    print(f"[BLEND EXCEPTION] {type(_e).__name__}: {_e} -> restoring the Apex output", flush=True)
    _l_shutil.copy(_l_bak, _l_pub)
print("Final submission:", _l_os.path.getsize(_l_pub), "bytes", flush=True)
'''

cells_src = [COMMON]
if "mavit" in LEGS:
    cells_src.append(MAVIT.replace("__WM__", repr(WM)))
if "of" in LEGS:
    cells_src.append(OF.replace("__WO__", repr(WO)))
cells_src.append(BLEND)
for i, c in enumerate(cells_src):
    compile(c, f"cell{i}", "exec")

# valida sintaks skrip dalam (dedent) tiap kaki
import ast
import re
import textwrap
for c in cells_src:
    for m in re.finditer(r'_l_tw\.dedent\("""(.*?)"""\)', c, re.S):
        inner = textwrap.dedent(ast.literal_eval('"""' + m.group(1) + '"""'))
        compile(inner, "inner", "exec")

nb = copy.deepcopy(json.loads(BASE.read_text(encoding="utf-8")))
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
    "## Extra legs (MaVIT hengck23, OrthoFoundation-L fold0 heliosli) + rank blend\n",
    "Kredit: hengck23 (MaVIT), heliosli (OrthoFoundation fold0, kode & anatomi), nartaa (fastreader & Raptor), goodpjw (reader), serta kredit Apex di atas."]})
for c in cells_src:
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": c.lstrip("\n").splitlines(keepends=True)})

slug = "rsna-knee-" + NAME.replace("_", "-")
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
ds = ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "nartaa/rsna-knee-publication-swa-weights-20261007"]
if "mavit" in LEGS:
    ds.append("hengck23/hengck23-rnsa-knee")
if "of" in LEGS:
    ds.append("heliosli/rsna-orthofoundation-fold0-epoch7-20261004")
    ds.append("dimaspashaakrilian/rsna-knee-ultralytics-cfg")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME.replace("_", " "),
    "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ds, "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| slug", slug, "| legs", LEGS, "| wm", WM, "| wo", WO, "| datasets", ds)
