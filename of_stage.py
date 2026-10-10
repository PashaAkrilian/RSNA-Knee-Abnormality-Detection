"""Templat tahap OrthoFoundation-L fold0 v2 (dipakai build_submit_legs.py dan build_scale_of.py).

Pelajaran uji skala 10 Okt (kernel ofscalea/ofscaleb) + controller heliosli:
- keluaran tiap studi ±330 MB (cache native + anatomi) -> potongan kecil (8 studi) + penjaga ruang disk (statvfs);
- CPU (4 vCPU) bukan GPU yang menjadi hambatan -> 4 proses (2 per GPU), thread BLAS/torch/cv2 = 1, MALLOC_ARENA_MAX=2;
- tiap potongan memakai akar-data kecil (symlink hanya studi potongan) agar DirectorySource tak memetakan seluruh test_series (84 dtk/panggilan);
- studi diurut menurut PRIORITAS (disagreement nartaa vs reader) sehingga bila waktu habis yang selesai = yang paling informatif;
- keluaran bertahap (append) + batas waktu lunak (pekerja berhenti) + pembunuhan proses keras; kaki yang tak selesai tidak merusak notebook.

Variabel yang harus sudah ada di lingkup notebook: _l_os,_l_sys,_l_glob,_l_json,_l_time,_l_shutil,_l_sp,_l_tw,_l_hl,_l_tb,_l_np,_l_pd,
_l_torch,_l_find,_l_env,_l_ng,LEG_OUT,_l_ids,_l_labels, _of_series_dir,_of_series_csv,_of_test_csv, _of_order (daftar prioritas).
Placeholder: __WO__ (bobot), __NPROC__, __CHUNK__, __STOP_H__, __KILL_H__, __MINCOV__ (liputan minimum agar kaki dipakai), __REUSE__ (1 = reuso runtime anatomi).
"""

OF_STAGE = r'''
# ---------------- leg: OrthoFoundation-L fold0 (heliosof) v2 ----------------
_l_t0 = _l_time.time()
try:
    _OF = _l_find("weight_parts.json")
    _UL = _l_find("assets/bus.jpg")          # yaml ultralytics 8.4.166 (dataset privat sendiri; paket vendored tanpa yaml)
    _MD = "/kaggle/working/of"
    _CC = "/kaggle/working/cloud_code"
    _TG = "/kaggle/working/_of_env"
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
    for _w in sorted(_l_glob.glob(f"{_OF}/dependencies/wheels/*.whl")):
        _l_sp.run([_l_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _TG, _w], check=False)
    # reuso do runtime de anatomia (3 jaringan) entre potongan: infer.py cria um AnatomyRuntime NOVO a cada chamada (~45 s)
    if __REUSE__:
        _ip = f"{_CC}/training/infer.py"
        _src = open(_ip).read()
        _old = "build(cache/'manifest.json',output/'geometry',weights,output/'anatomy')"
        _new = ("from .anatomy_runtime import AnatomyRuntime;g=globals();g['_RT']=g.get('_RT') or AnatomyRuntime(weights,'cuda',None);"
                "build(cache/'manifest.json',output/'geometry',weights,output/'anatomy',runtime=g['_RT'])")
        if _old in _src:
            open(_ip, "w").write(_src.replace(_old, _new, 1))
            print("[OF] infer.py patched: anatomy runtime reused across chunks", flush=True)
        else:
            print("[OF] infer.py patch NOT applied (pattern missing)", flush=True)
    print("[OF] setup done %.0fs" % (_l_time.time() - _l_t0), flush=True)

    _NPROC, _CHUNK = __NPROC__, __CHUNK__
    _stop_at = _l_time.time() + __STOP_H__ * 3600.0
    _kill_at = _l_time.time() + __KILL_H__ * 3600.0
    _parts = [_of_order[i::_NPROC] for i in range(_NPROC)]
    _script = _l_tw.dedent("""
        import os, sys, time, shutil, json
        os.environ["OMP_NUM_THREADS"] = "1"
        import pandas as pd
        import torch
        _o = torch.set_num_threads
        torch.set_num_threads = lambda n: _o(1)
        torch.set_num_threads(1)
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass
        shard, ids_json, stop_at, chunk, series_dir, series_csv, test_csv, md = (int(sys.argv[1]), sys.argv[2], float(sys.argv[3]),
                                                                                 int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
        ids = json.load(open(ids_json))
        from training.infer import run_inference
        ser = pd.read_csv(series_csv)
        test_cols = list(pd.read_csv(test_csv, nrows=1).columns)
        ser_cols = list(pd.read_csv(series_csv, nrows=1).columns)
        outcsv = f"/kaggle/working/_leg_of_w{shard}.csv"
        t0, done, fails = time.time(), 0, 0
        for k in range(0, len(ids), chunk):
            if time.time() > stop_at:
                print("STOP_AT reached; remaining", len(ids) - k, flush=True)
                break
            ch = ids[k:k + chunk]
            # penjaga disk: tunggu bila ruang < kebutuhan potongan (proses lain sedang menulis)
            need = chunk * 0.42 + 1.5
            waited = 0
            while True:
                st = os.statvfs("/kaggle/working")
                free = st.f_bavail * st.f_frsize / 1e9
                if free >= need or time.time() > stop_at or waited > 40:
                    break
                time.sleep(30); waited += 1
            root = f"/kaggle/working/fr/w{shard}_{k}"
            out = f"/kaggle/working/ofo/w{shard}_{k}"
            try:
                os.makedirs(root + "/test_series", exist_ok=True)
                pd.DataFrame({c: (ch if c == "StudyInstanceUID" else None) for c in test_cols}).to_csv(root + "/test.csv", index=False)
                sub = ser[ser["StudyInstanceUID"].isin(ch)]
                sub[[c for c in ser_cols if c in sub.columns]].to_csv(root + "/test_series.csv", index=False)
                for sid in ch:
                    os.symlink(series_dir + "/" + sid, root + "/test_series/" + sid)
                tc = time.time()
                fr = run_inference(root, md + "/best.pt", out, anatomy_weights=md + "/anatomy.pt", workers=1)
                first = not os.path.exists(outcsv)
                fr.to_csv(outcsv, mode="a", header=first, index=False)
                done += len(fr)
                print("CHUNK w%d k=%d n=%d ok=%d sec=%.0f (%.1f s/study) free_GB=%.1f cumul=%d in %.0fs" % (
                    shard, k, len(ch), len(fr), time.time() - tc, (time.time() - tc) / len(ch), free, done, time.time() - t0), flush=True)
            except Exception as e:
                fails += len(ch)
                print("CHUNK FAIL w%d k=%d %s" % (shard, k, repr(e)[:260]), flush=True)
            finally:
                shutil.rmtree(out, ignore_errors=True)
                shutil.rmtree(root, ignore_errors=True)
        print("WORKER_DONE w%d done=%d fails=%d elapsed=%.0f" % (shard, done, fails, time.time() - t0), flush=True)
    """)
    open("/kaggle/working/_leg_of_worker.py", "w").write(_script)
    _procs = []
    for _i in range(_NPROC):
        _gpu = _i % max(1, _l_torch.cuda.device_count())
        _ij = f"/kaggle/working/_leg_of_ids{_i}.json"
        _l_json.dump(_parts[_i], open(_ij, "w"))
        _e = dict(_l_env)
        _e.update(CUDA_VISIBLE_DEVICES=str(_gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  PYTHONPATH=_TG + _l_os.pathsep + _CC + _l_os.pathsep + _l_env.get("PYTHONPATH", ""))
        _lg = open(f"/kaggle/working/_leg_of_w{_i}.log", "w")
        _procs.append((_l_sp.Popen([_l_sys.executable, "-u", "/kaggle/working/_leg_of_worker.py", str(_i), _ij, str(_stop_at), str(_CHUNK),
                                    _of_series_dir, _of_series_csv, _of_test_csv, _MD], env=_e, stdout=_lg, stderr=_l_sp.STDOUT), _lg))
    print("[OF] launched", _NPROC, "workers | stop_at +%.1fh kill_at +%.1fh" % (__STOP_H__, __KILL_H__), flush=True)
    _last = 0.0
    while any(p.poll() is None for p, _ in _procs):
        _l_time.sleep(15)
        if _l_time.time() > _kill_at:
            for p, _ in _procs:
                if p.poll() is None:
                    p.kill()
            print("[OF] KILL at deadline", flush=True)
            break
        if _l_time.time() - _last > 900:
            _last = _l_time.time()
            _st = _l_os.statvfs("/kaggle/working")
            _n = 0
            for _i in range(_NPROC):
                try:
                    _n += sum(1 for _ in open(f"/kaggle/working/_leg_of_w{_i}.csv")) - 1
                except OSError:
                    pass
            print("[OF] progress %.2fh | studies done %d | disk free %.1f GB" % ((_l_time.time() - _l_t0) / 3600, _n, _st.f_bavail * _st.f_frsize / 1e9), flush=True)
    for _i, (p, lg) in enumerate(_procs):
        lg.close()
        print(f"--- of worker {_i} log tail ---\n" + "".join(open(f"/kaggle/working/_leg_of_w{_i}.log").readlines()[-5:]), flush=True)
    _dfs = []
    for _i in range(_NPROC):
        _f = f"/kaggle/working/_leg_of_w{_i}.csv"
        if _l_os.path.exists(_f):
            _dfs.append(_l_pd.read_csv(_f, dtype={"StudyInstanceUID": str}))
    assert _dfs, "no OF output"
    _d = _l_pd.concat(_dfs).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(_l_ids)[_l_labels]
    _cov = 1.0 - float(_d.isna().all(axis=1).mean())
    print(f"[OF] coverage {_cov:.3f} ({int(_cov * len(_l_ids))}/{len(_l_ids)}) | {_l_time.time() - _l_t0:.0f}s", flush=True)
    assert _cov >= __MINCOV__, "OF coverage too low"
    _d.to_csv("/kaggle/working/_leg_of.csv")
    LEG_OUT["of"] = ("/kaggle/working/_leg_of.csv", __WO__)
except Exception as _e:
    _l_tb.print_exc()
    print(f"[OF EXCEPTION] {type(_e).__name__}: {_e} -> leg skipped", flush=True)
'''
