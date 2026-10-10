#!/usr/bin/env python3
"""Bangun notebook submisi: Apex (nartaa r384 + reader goodpjw) + pembaca ketiga MaVIT (hengck23 / lixin73).

MaVIT = keluarga arsitektur BERBEDA (MaViT/MALA + transformer jendela), solo LB 0,935 (288px, hengck23). Dipasang
sebagai tahap akhir: rank-blend per temuan di atas keluaran Apex; bila tahap MaVIT gagal, keluaran Apex dipulihkan
(aman). Tidak ada bobot yang di-fit di emas-58.

Pemakaian (key=value):
  python3 build_submit_mavit.py name=apexmavit w=0.20 models=hk288
  python3 build_submit_mavit.py name=apexmavit2 w=0.25 models=hk288,lx336k8 tier=1
models: hk288 (hengck23 00000008.pth, 288px K=8) | lx336k8 / lx336k12 (lixin73 fold0, 336px, K=8/12)
tier=1 : bobot per temuan 3 tingkat (kuat/sedang/lemah) relatif thd w (kuat = w*1.25, lemah = w*0.5)
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BASE = ROOT / "kaggle_base" / "apex_grandmaster_stack_20261008.ipynb"

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "apexmavit")
W = float(args.get("w", "0.20"))
MODELS = [m for m in args.get("models", "hk288").split(",") if m]
TIER = args.get("tier", "0") == "1"

CATALOG = {
    "hk288": dict(name="hk288", src="hk", file="00000008.pth", size=288, k=8, crop_mm=130.0),
    "lx336k8": dict(name="lx336k8", src="lx", file="mavit_336px_seed48_fold0.pt", size=336, k=8, crop_mm=130.0),
    "lx336k12": dict(name="lx336k12", src="lx", file="mavit_336px_seed48_fold0.pt", size=336, k=12, crop_mm=130.0),
}
cfg = [CATALOG[m] for m in MODELS]

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
if TIER:
    strong = {"ACL", "MCL", "Medial Meniscus", "Medial OA", "Effusion", "Baker's"}
    weak = {"Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis", "Fracture"}
    weights = {l: round(W * (1.25 if l in strong else 0.5 if l in weak else 1.0), 4) for l in LABELS}
else:
    weights = {l: W for l in LABELS}

STAGE = r'''
# ==============================================================================
# MaVIT third reader (hengck23 / lixin73) -> per-finding rank blend on top of the Apex output
# Independent architecture family (MaViT/MALA + window transformer). Failure => Apex output restored.
# ==============================================================================
import os as _m_os, sys as _m_sys, glob as _m_glob, json as _m_json, time as _m_time
import shutil as _m_shutil, subprocess as _m_sp, textwrap as _m_tw
import numpy as _m_np, pandas as _m_pd, torch as _m_torch

_m_pub = "/kaggle/working/submission.csv"
_m_bak = "/kaggle/working/_apex_before_mavit.csv"
_m_shutil.copy(_m_pub, _m_bak)
_MAV_W = __WEIGHTS__
_MAV_MODELS = __MODELS__
_MAV_TIMEOUT = 3 * 3600
_m_t0 = _m_time.time()

try:
    import gc as _m_gc
    _m_gc.collect()
    if _m_torch.cuda.is_available():
        _m_torch.cuda.empty_cache()

    def _m_find(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
            for d in sorted(_m_glob.glob(pat)):
                if _m_os.path.exists(_m_os.path.join(d, name)):
                    return d.rstrip("/")
        raise FileNotFoundError(f"{name} not found in /kaggle/input")

    _m_dirs = {"hk": _m_find("my_fastread_knee.py")}
    if any(m["src"] == "lx" for m in _MAV_MODELS):
        _m_dirs["lx"] = _m_find("mavit_336px_seed48_fold0.pt")
    _m_data = _m_find("test_series.csv")
    _m_df = _m_pd.read_csv(_m_pub, dtype={"StudyInstanceUID": str})
    _m_ids = _m_df.StudyInstanceUID.tolist()
    _m_labels = [c for c in _m_df.columns if c != "StudyInstanceUID"]
    open("/kaggle/working/_mavit_ids.txt", "w").write("\n".join(_m_ids))
    for _m in _MAV_MODELS:
        _m["path"] = _m_os.path.join(_m_dirs[_m["src"]], _m["file"])
        assert _m_os.path.isfile(_m["path"]), _m["path"]
    print("[MaVIT] models:", [(m["name"], m["size"], m["k"]) for m in _MAV_MODELS], "| studies", len(_m_ids), flush=True)

    _m_script = _m_tw.dedent("""
        import os, sys, json, time, traceback
        import numpy as np, pandas as pd, torch
        hk, data, cfg_json, shard, nsh, ids_file = sys.argv[1:7]
        shard, nsh = int(shard), int(nsh)
        sys.path.insert(0, hk)
        from model import KneeModel
        from my_fastread_knee import make_knee_dataset_reader
        LABEL = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
                 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
        cfg = json.loads(cfg_json)
        ids = open(ids_file).read().split("\\n")
        ids = ids[shard::nsh]
        SERIES = pd.read_csv(os.path.join(data, "test_series.csv"))
        DDIR = os.path.join(data, "test_series")

        def build(path):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            sd = saved
            if isinstance(saved, dict):
                for k in ("model_state_dict", "model", "state_dict", "ema", "ema_state_dict", "net"):
                    if k in saved and isinstance(saved[k], dict):
                        sd = saved[k]
                        break
            best = None
            for strip in ("", "module.", "model.", "_orig_mod."):
                sd2 = {(k[len(strip):] if strip and k.startswith(strip) else k): v for k, v in sd.items()}
                m = KneeModel(pretrained=False)
                r = m.load_state_dict(sd2, strict=False)
                sc = len(r.missing_keys) + len(r.unexpected_keys)
                if best is None or sc < best[0]:
                    best = (sc, m, r)
            print("load", os.path.basename(path), "missing", len(best[2].missing_keys), "unexpected",
                  len(best[2].unexpected_keys), flush=True)
            return best[1].cuda().eval()

        for c in cfg:
            model = build(c["path"])
            reader = make_knee_dataset_reader(out_size=c["size"], num_slice=c["k"], percentile=(1.0, 99.0),
                                              crop_mm=c["crop_mm"], io_threads=12, crop_only_if_nonsquare=True)
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
                        print("FAIL", c["name"], sid[:24], repr(e)[:200], flush=True)
                    p = np.full(12, np.nan, np.float32)
                rows.append(p)
                if (i + 1) % 100 == 0:
                    print(c["name"], "shard", shard, i + 1, "/", len(ids), "%.2fs/study" % ((time.time() - t0) / (i + 1)), flush=True)
            out = pd.DataFrame(np.array(rows), index=ids, columns=LABEL).rename_axis("StudyInstanceUID")
            out.to_csv(f"/kaggle/working/_mavit_{c['name']}_s{shard}.csv")
            print("done", c["name"], "shard", shard, "fails", fails, "sec/study %.2f" % ((time.time() - t0) / max(1, len(ids))), flush=True)
            del model
            torch.cuda.empty_cache()
    """)
    open("/kaggle/working/_mavit_infer.py", "w").write(_m_script)

    _m_env = dict(_m_os.environ)
    _m_own = "/kaggle/working/_own_env"
    if _m_os.path.isdir(_m_own):
        _m_env["PYTHONPATH"] = _m_own + _m_os.pathsep + _m_env.get("PYTHONPATH", "")
    _m_ng = max(1, min(2, _m_torch.cuda.device_count()))
    _m_procs = []
    for _s in range(_m_ng):
        _e = dict(_m_env)
        _e["CUDA_VISIBLE_DEVICES"] = str(_s)
        _lg = open(f"/kaggle/working/_mavit_shard{_s}.log", "w")
        _m_procs.append((_m_sp.Popen([_m_sys.executable, "-u", "/kaggle/working/_mavit_infer.py", _m_dirs["hk"], _m_data,
                                      _m_json.dumps(_MAV_MODELS), str(_s), str(_m_ng), "/kaggle/working/_mavit_ids.txt"],
                                     env=_e, stdout=_lg, stderr=_m_sp.STDOUT), _lg))
    for _p, _lg in _m_procs:
        _p.wait(timeout=_MAV_TIMEOUT)
        _lg.close()
        assert _p.returncode == 0, f"MaVIT shard failed rc={_p.returncode}"
    for _s in range(_m_ng):
        print(f"--- shard{_s} log tail ---\n" + "".join(open(f"/kaggle/working/_mavit_shard{_s}.log").readlines()[-6:]), flush=True)

    # merge shards, average probabilities over MaVIT models
    _m_all = []
    for _m in _MAV_MODELS:
        _d = _m_pd.concat([_m_pd.read_csv(f"/kaggle/working/_mavit_{_m['name']}_s{_s}.csv", dtype={"StudyInstanceUID": str})
                           for _s in range(_m_ng)]).set_index("StudyInstanceUID").reindex(_m_ids)[_m_labels]
        _m_all.append(_d.values)
    _mv = _m_np.nanmean(_m_np.stack(_m_all, 0), axis=0)
    _mav = _m_pd.DataFrame(_mv, index=_m_ids, columns=_m_labels)
    _fail = float(_mav.isna().all(axis=1).mean())
    print(f"[MaVIT] failed-study fraction {_fail:.4f}", flush=True)
    assert _fail < 0.05, "too many MaVIT failures"

    _apex = _m_pd.read_csv(_m_bak, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[_m_ids, _m_labels]
    _ra = _apex.rank(method="average", pct=True)
    _rm = _mav.rank(method="average", pct=True).fillna(_ra)
    _fused = _ra.copy()
    for _c in _m_labels:
        _w = _MAV_W.get(_c, 0.2)
        _fused[_c] = (1.0 - _w) * _ra[_c] + _w * _rm[_c]
        if len(_m_ids) >= 20:
            print(f"[MaVIT] {_c:18s} w={_w:.2f} corr(apex,mavit)={_m_np.corrcoef(_ra[_c], _rm[_c])[0, 1]:.3f}", flush=True)
    _fused = _fused.rank(method="average", pct=True)
    assert _fused.notna().all().all(), "fused NaN"
    assert (_fused.nunique() > 1).all() or len(_m_ids) < 5, "fused constant"
    _fused.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(_m_pub, index=False)
    print(f"[MaVIT SUCCESS] blended in {_m_time.time() - _m_t0:.0f}s ->", _m_pub, flush=True)
except Exception as _e:
    import traceback as _tb
    _tb.print_exc()
    print(f"[MaVIT EXCEPTION] {type(_e).__name__}: {_e}", flush=True)
    print("[MaVIT FALLBACK] restoring the Apex output", flush=True)
    _m_shutil.copy(_m_bak, _m_pub)
print("Final submission:", _m_os.path.getsize(_m_pub), "bytes", flush=True)
'''

stage = STAGE.replace("__WEIGHTS__", json.dumps(weights)).replace("__MODELS__", json.dumps(cfg))
compile(stage, "stage", "exec")  # sintaks

nb = json.loads(BASE.read_text(encoding="utf-8"))
nb = copy.deepcopy(nb)
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
    "## MaVIT third reader (hengck23 / lixin73)\n",
    "Tahap akhir: pembaca MaVIT (arsitektur independen) di-rank-blend per temuan di atas keluaran Apex. ",
    "Kredit: hengck23 (kode MaVIT + dataset), nartaa (fastreader), lixin73 (bobot 336px), serta semua kredit Apex di atas."]})
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                    "source": stage.lstrip("\n").splitlines(keepends=True)})

slug = "rsna-knee-" + NAME.replace("_", "-")
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
ds = ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "nartaa/rsna-knee-publication-swa-weights-20261007",
      "hengck23/hengck23-rnsa-knee"]
if any(m["src"] == "lx" for m in cfg):
    ds.append("lixin73/rsna-knee-mavit-336px-v51-weights")
meta = {
    "id": f"dimaspashaakrilian/{slug}",
    "title": "RSNA Knee " + NAME.replace("_", " "),
    "code_file": f"{slug}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ds, "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| slug", slug, "| models", MODELS, "| w", W, "| tier", TIER, "| datasets", ds)
