"""Tahap OrthoFoundation-L fold0 v4 — dijalankan SEBELUM Apex (build_apexof4.py).

Latar (10 Okt malam): submisi 57044355 (v3: OF sesudah Apex) selesai dlm +-1 j 35 mnt dgn skor 0,950 = Apex murni, padahal OF butuh +-6,8 j utk
1.322 studi -> kaki OF jatuh ke Apex tanpa terlihat (log run penilaian tak terbaca). Dugaan utama: RAM (proses induk Apex masih besar + 4 pekerja OF).
Perbaikan v4:
- OF jalan PERTAMA (proses induk masih kecil, tanpa torch/CUDA); Apex berjalan sesudahnya seperti biasa -> tak ada ketergantungan pada sisa memori Apex.
- Antrean potongan (8 studi) dgn 'klaim' atomik (os.mkdir) -> pekerja yg mati tak menghilangkan potongan lain; supervisor induk: reaping,
  pelepasan klaim yatim (2 kali crash = potongan 'bad'), restart pekerja (anggaran 8), tidak menambah pekerja bila RAM sempit / setelah mirip-OOM (rc -9).
- Pekerja didaur ulang tiap MAXCHUNKS potongan atau RSS > RSSMAX (cegah kebocoran memori/GPU selama 6 jam); oom_score_adj=500 (pekerja dikorbankan dulu, bukan notebook).
- Keluaran per potongan ditulis atomik (tmp + os.replace) ke _of_res/<k>.csv; agregasi di akhir; liputan minimum MINCOV.
- Pemutus waktu: berhenti lunak STOP_H, bunuh grup proses KILL_H (killpg: termasuk anak DataLoader).
Placeholder: __WD__ (/kaggle/working), __NPROC__, __CHUNK__, __STOP_H__, __KILL_H__, __MINCOV__, __MAXCHUNKS__, __RSSMAX__, __INJECT__ (uji: bunuh pekerja 0 pd +__INJECT_AFTER__ dtk; 360 di rehearsal), __POLL__ (jeda supervisor dtk; 10), __PRINTEVERY__ (dtk entre linhas de progresso; 300), __MEMPERW__ (5.5 GB/pekerja), __MEMFREE__ (6.5 GB minimum utk spawn), __DISKPER__ (0.42 GB/studi) & __DISKBASE__ (1.5 GB) = penjaga disk, __FORCEW__ (0 = otomatis; >0 = paksa jumlah pekerja, uji), __MMAP__ (1 = torch.load mmap=True), __EXTRAENV__ (dict literal: env tambahan pekerja, mis. MALLOC_*),
__DATA_PRESET__ (kode penentu akar data; pseudo-root utk rehearsal), __SETUP__ (1 = pasang aset OF; 0 = uji lokal).
"""

OF4_CELL = r'''
# ================= leg OrthoFoundation-L fold0 (heliosli) v4 — JALAN SEBELUM APEX =================
import os as _os, sys as _sys, glob as _glob, json as _json, time as _time, shutil as _shutil, subprocess as _sp, signal as _sig, hashlib as _hl
import traceback as _tb
import pandas as _pd

_WD = "__WD__"
_OF_T0 = _time.time()
_OF_LEG = _WD + "/_leg_of.csv"          # keluaran sukses (dibaca sel BLEND)
_OF_STATE = {"cov": 0.0, "ok": False, "why": "not started"}

def _of_find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(_glob.glob(pat)):
            if _os.path.exists(_os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(f"{name} not found in /kaggle/input")

def _of_mem():
    """(MemAvailable GB, headroom EFETIVO do cgroup GB | None). O headroom desconta o page-cache (recuperavel): sem isso ele parece baixo demais."""
    a = None
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                a = int(l.split()[1]) / 1048576.0
                break
    except Exception:
        pass
    h = None
    for base, cur, mx, key in (("/sys/fs/cgroup", "memory.current", "memory.max", "file"),
                               ("/sys/fs/cgroup/memory", "memory.usage_in_bytes", "memory.limit_in_bytes", "cache")):
        try:
            m = open(f"{base}/{mx}").read().strip()
            if m != "max" and int(m) < (1 << 50):
                used = int(open(f"{base}/{cur}").read())
                fc = 0
                try:
                    for l in open(f"{base}/memory.stat"):
                        k, v = l.split()
                        if k == key:
                            fc = int(v)
                            break
                except Exception:
                    pass
                h = (int(m) - used + fc) / 1e9
                break
        except Exception:
            pass
    return a, h

def _of_rss(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 1048576.0
    except Exception:
        pass
    return 0.0

def _of_gpus():
    try:
        o = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return [tuple(int(x) for x in ln.split(",")) for ln in o.strip().splitlines() if ln.strip()]
    except Exception:
        return []

_OF_WORKER_SRC = r"""
import os, sys, time, json, shutil
os.environ["OMP_NUM_THREADS"] = "1"
try:
    open("/proc/self/oom_score_adj", "w").write("500")
except Exception:
    pass
import pandas as pd
try:
    import torch
    _o = torch.set_num_threads
    torch.set_num_threads = lambda n: _o(1)
    torch.set_num_threads(1)
except Exception:
    torch = None
try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass

if __MMAP__ and torch is not None:      # checkpoints (ViT-L 1,2 GB x2 + anatomia) mapeados em memoria: page-cache compartilhado entre pekerja
    _ol = torch.load
    def _ml(f, *a, **k):
        k.setdefault("mmap", True)
        return _ol(f, *a, **k)
    torch.load = _ml

def rss_gb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 1048576.0
    except Exception:
        pass
    return 0.0

def rss_anon_gb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("RssAnon:"):
                return int(l.split()[1]) / 1048576.0
    except Exception:
        pass
    return 0.0

(slot, chunks_json, stop_at, series_dir, series_csv, test_csv, md, claims, res_dir, wd, maxchunks, rss_max) = (
    int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],
    sys.argv[10], int(sys.argv[11]), float(sys.argv[12]))
chunks = json.load(open(chunks_json))
from training.infer import run_inference
ser = pd.read_csv(series_csv)
test_cols = list(pd.read_csv(test_csv, nrows=1).columns)
ser_cols = list(pd.read_csv(series_csv, nrows=1).columns)
curf = f"{wd}/_cur_w{slot}"
t0 = time.time()
nch = 0
done_n = 0
reason = "no more chunks"
stop = False
while not stop:
    claimed_any = False
    for k, ch in enumerate(chunks):
        if time.time() > stop_at:
            reason = "STOP_AT"; stop = True; break
        if nch >= maxchunks:
            reason = "RECYCLE chunks"; stop = True; break
        if rss_gb() > rss_max:
            reason = "RECYCLE rss=%.1fGB" % rss_gb(); stop = True; break
        cd = f"{claims}/{k}"
        try:
            os.mkdir(cd)
        except FileExistsError:
            continue
        claimed_any = True
        nch += 1
        open(curf, "w").write(str(k))
        need = len(ch) * __DISKPER__ + __DISKBASE__
        waited = 0
        while True:
            st = os.statvfs(wd)
            free = st.f_bavail * st.f_frsize / 1e9
            if free >= need or time.time() > stop_at or waited > 40:
                break
            time.sleep(30); waited += 1
        root = f"{wd}/fr/w{slot}_{k}"
        out = f"{wd}/ofo/w{slot}_{k}"
        ok = False
        try:
            shutil.rmtree(root, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
            os.makedirs(root + "/test_series", exist_ok=True)
            pd.DataFrame({c: (ch if c == "StudyInstanceUID" else None) for c in test_cols}).to_csv(root + "/test.csv", index=False)
            sub = ser[ser["StudyInstanceUID"].isin(ch)]
            sub[[c for c in ser_cols if c in sub.columns]].to_csv(root + "/test_series.csv", index=False)
            for sid in ch:
                os.symlink(series_dir + "/" + sid, root + "/test_series/" + sid)
            tc = time.time()
            fr = run_inference(root, md + "/best.pt", out, anatomy_weights=md + "/anatomy.pt", workers=1)
            tmp = f"{res_dir}/{k}.csv.tmp"
            fr.to_csv(tmp, index=False)
            os.replace(tmp, f"{res_dir}/{k}.csv")
            open(cd + "/done", "w").write(str(len(fr)))
            done_n += len(fr)
            ok = True
            print("CHUNK w%d k=%d n=%d ok=%d sec=%.0f (%.1f s/study) free_GB=%.1f rss=%.1fGB anon=%.1fGB cumul=%d in %.0fs" % (
                slot, k, len(ch), len(fr), time.time() - tc, (time.time() - tc) / len(ch), free, rss_gb(), rss_anon_gb(), done_n, time.time() - t0), flush=True)
        except Exception as e:
            fc = f"{claims}_fail/{k}"
            try:
                n = int(open(fc).read()) + 1 if os.path.exists(fc) else 1
            except Exception:
                n = 1
            open(fc, "w").write(str(n))
            print("CHUNK FAIL w%d k=%d (fail #%d) %s" % (slot, k, n, repr(e)[:300]), flush=True)
            if n >= 2:
                open(cd + "/bad", "w").write(repr(e)[:200])
            else:
                shutil.rmtree(cd, ignore_errors=True)        # libera p/ nova tentativa
        finally:
            shutil.rmtree(out, ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)
            open(curf, "w").write("")
    if not claimed_any:
        break
print("WORKER_DONE w%d chunks=%d studies=%d elapsed=%.0f reason=%s rss=%.1fGB" % (slot, nch, done_n, time.time() - t0, reason, rss_gb()), flush=True)
"""

try:
    _OF_MAXW, _OF_CHUNK = __NPROC__, __CHUNK__
    _OF_STOP, _OF_KILL = _OF_T0 + __STOP_H__ * 3600.0, _OF_T0 + __KILL_H__ * 3600.0
    __DATA_PRESET__
    _OF_SERDIR, _OF_SERCSV, _OF_TESTCSV = _of_data + "/test_series", _of_data + "/test_series.csv", _of_data + "/test.csv"
    _ids = _pd.read_csv(_OF_TESTCSV, dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
    _a0, _h0 = _of_mem()
    print("[OF4] studies %d | data %s | cpus %s | MemAvailable %s GB | cgroup headroom %s GB | parent RSS %.1f GB | GPUs(used,total MiB) %s" % (
        len(_ids), _of_data, _os.cpu_count(), None if _a0 is None else round(_a0, 1), None if _h0 is None else round(_h0, 1), _of_rss(_os.getpid()), _of_gpus()), flush=True)
    try:
        print("[OF4] top RSS procs (pid,rss_kB,comm):", _sp.run("ps -eo pid,rss,comm --sort=-rss | head -6", shell=True, capture_output=True, text=True, timeout=20).stdout.replace("\n", " ; "), flush=True)
    except Exception:
        pass
    _MD, _CC, _TG = _WD + "/of", _WD + "/cloud_code", _WD + "/_of_env"
    if __SETUP__:
        _OF = _of_find("weight_parts.json")
        _UL = _of_find("assets/bus.jpg")          # yaml ultralytics 8.4.166 (dataset privat sendiri; paket vendored tanpa yaml)
        _os.makedirs(_MD, exist_ok=True)
        if not _os.path.exists(_CC):
            _shutil.copytree(f"{_OF}/cloud_code", _CC)
        _shutil.copytree(_UL, f"{_CC}/training/vendor_runtime/kneexnet/ultralytics", dirs_exist_ok=True)
        assert _os.path.exists(f"{_CC}/training/vendor_runtime/kneexnet/ultralytics/cfg/default.yaml")
        _h = _hl.sha256()
        with open(f"{_MD}/best.pt", "wb") as _out:
            for _p in sorted(_glob.glob(f"{_OF}/weights.part*")):
                with open(_p, "rb") as _f:
                    while True:
                        _b = _f.read(1 << 24)
                        if not _b:
                            break
                        _out.write(_b)
                        _h.update(_b)
        assert _h.hexdigest().startswith("4f424cc7"), "OF best.pt sha mismatch"
        if not _os.path.exists(f"{_MD}/anatomy.pt"):
            _shutil.copy2(f"{_OF}/anatomy.pt", f"{_MD}/anatomy.pt")
        if not _os.path.exists(f"{_MD}/kneexnet"):
            _shutil.copytree(f"{_OF}/dependencies/kneexnet", f"{_MD}/kneexnet")
        for _w in sorted(_glob.glob(f"{_OF}/dependencies/wheels/*.whl")):
            _sp.run([_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _TG, _w], check=False)
        try:    # decoders DICOM (pylibjpeg) do dataset goodpjw, como o Apex faz depois; so decoders (sem timm) para nao alterar o timm validado nos testes de escala
            _GP = _of_find("cnxt_v0_fold0.pt")
            _sp.run([_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _WD + "/_own_env", "--find-links", _GP,
                     "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
        except Exception as _e0:
            print("[OF4] decoder wheels skipped:", repr(_e0)[:120], flush=True)
    print("[OF4] setup done %.0fs" % (_time.time() - _OF_T0), flush=True)

    _chunks = [_ids[i:i + _OF_CHUNK] for i in range(0, len(_ids), _OF_CHUNK)]
    _claims, _res = _WD + "/_of_claims", _WD + "/_of_res"
    for _d in (_claims, _claims + "_fail", _res, _WD + "/fr", _WD + "/ofo"):
        _shutil.rmtree(_d, ignore_errors=True)
        _os.makedirs(_d, exist_ok=True)
    _cj = _WD + "/_of_chunks.json"
    _json.dump(_chunks, open(_cj, "w"))
    _wsrc = _WD + "/_of_worker.py"
    open(_wsrc, "w").write(_OF_WORKER_SRC)
    _ng = max(1, len(_of_gpus()))
    # concorrencia inicial: limitada pela RAM disponivel (cada pekerja ±5-6 GB RSS); GPUs: 2 pekerja/GPU
    _avail = [x for x in (_a0, _h0) if x is not None]
    _w0 = __NPROC__ if not _avail else max(1, min(__NPROC__, int((min(_avail) - 2.5) // __MEMPERW__)))
    _target = min(_w0, 2 * _ng) if not __FORCEW__ else __FORCEW__
    print("[OF4] chunks %d x %d | workers target %d (max %d, gpus %d) | stop +%.1fh kill +%.1fh" % (
        len(_chunks), _OF_CHUNK, _target, __NPROC__, _ng, __STOP_H__, __KILL_H__), flush=True)

    _slots, _crash, _restarts, _spawned, _inj_done = {}, {}, 8, 0, False
    _last_print, _low_since = 0.0, None

    def _of_spawn(slot):
        global _spawned
        env = dict(_os.environ)
        own = _WD + "/_own_env"
        pp = [_TG, _CC] + ([own] if _os.path.isdir(own) else []) + [env.get("PYTHONPATH", "")]
        env.update(CUDA_VISIBLE_DEVICES=str(slot % _ng), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   PYTHONUNBUFFERED="1", PYTHONPATH=_os.pathsep.join(x for x in pp if x))
        env.update(__EXTRAENV__)
        lg = open(f"{_WD}/_leg_of_w{slot}.log", "a")
        p = _sp.Popen([_sys.executable, "-u", _wsrc, str(slot), _cj, str(_OF_STOP), _OF_SERDIR, _OF_SERCSV, _OF_TESTCSV,
                       _MD, _claims, _res, _WD, str(__MAXCHUNKS__), str(__RSSMAX__)], env=env, stdout=lg, stderr=_sp.STDOUT, start_new_session=True)
        _slots[slot] = (p, lg)
        _spawned += 1

    def _of_killgrp(p):
        try:
            _os.killpg(p.pid, _sig.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _of_counts():
        d = b = c = 0
        for k in range(len(_chunks)):
            cd = f"{_claims}/{k}"
            if _os.path.exists(cd + "/done"):
                d += 1
            elif _os.path.exists(cd + "/bad"):
                b += 1
            elif _os.path.isdir(cd):
                c += 1
        return d, b, c

    while True:
        _time.sleep(__POLL__)
        _now = _time.time()
        # 1) reaping
        for _s in list(_slots):
            _p, _lg = _slots[_s]
            _rc = _p.poll()
            if _rc is None:
                continue
            _lg.close()
            del _slots[_s]
            _of_killgrp(_p)                                   # anak yatim (DataLoader) ikut mati
            try:
                _tail = "".join(open(f"{_WD}/_leg_of_w{_s}.log").readlines()[-3:]).strip().replace("\n", " | ")[-400:]
            except Exception:
                _tail = ""
            try:
                _k = open(f"{_WD}/_cur_w{_s}").read().strip()
            except Exception:
                _k = ""
            if _rc != 0:
                print("[OF4] worker slot %d DIED rc=%s chunk=%s | %s" % (_s, _rc, _k or "-", _tail), flush=True)
                _restarts -= 1
                if _k != "":
                    _kk = int(_k)
                    _cd = f"{_claims}/{_kk}"
                    if _os.path.isdir(_cd) and not _os.path.exists(_cd + "/done") and not _os.path.exists(_cd + "/bad"):
                        _crash[_kk] = _crash.get(_kk, 0) + 1
                        if _crash[_kk] >= 2:
                            open(_cd + "/bad", "w").write("crash x2 rc=%s" % _rc)
                        else:
                            _shutil.rmtree(_cd, ignore_errors=True)
                    for _dd in (f"{_WD}/fr/w{_s}_{_kk}", f"{_WD}/ofo/w{_s}_{_kk}"):
                        _shutil.rmtree(_dd, ignore_errors=True)
                if _rc in (-9, 137):
                    _target = max(1, min(_target, len(_slots)))
                    print("[OF4] OOM-like death -> concurrency target now %d" % _target, flush=True)
            else:
                print("[OF4] worker slot %d exit rc=0 | %s" % (_s, _tail), flush=True)
        _dn, _bd, _cl = _of_counts()
        _pending = len(_chunks) - _dn - _bd
        _unclaimed = _pending - _cl
        if _pending == 0:
            print("[OF4] all chunks finished (done %d, bad %d)" % (_dn, _bd), flush=True)
            break
        if _now > _OF_KILL:
            for _p, _lg in _slots.values():
                _of_killgrp(_p)
            print("[OF4] KILL at deadline (done %d / %d)" % (_dn, len(_chunks)), flush=True)
            break
        if not _slots and (_now > _OF_STOP or _restarts <= 0 or _unclaimed == 0 and _cl == 0):
            print("[OF4] no workers left (restarts_left %d, stop_reached %s, done %d, bad %d, pending %d)" % (_restarts, _now > _OF_STOP, _dn, _bd, _pending), flush=True)
            break
        # 2) spawn
        if _now < _OF_STOP and _restarts > 0:
            _want = min(_target, max(1, _unclaimed))
            while len(_slots) < _want and _unclaimed > 0 and _restarts > 0:
                _a, _h = _of_mem()
                _lim = [x for x in (_a, _h) if x is not None]
                if _lim and min(_lim) < __MEMFREE__ and (_slots or _low_since is None or _now - _low_since < 180):
                    if _low_since is None:
                        _low_since = _now
                    break
                _low_since = None
                _sid = min(i for i in range(8) if i not in _slots)
                _of_spawn(_sid)
                print("[OF4] spawn slot %d (alive %d, spawned %d, restarts_left %d, MemAvail %s, headroom %s)" % (
                    _sid, len(_slots), _spawned, _restarts, None if _a is None else round(_a, 1), None if _h is None else round(_h, 1)), flush=True)
        # 3) uji tahan banting (rehearsal): bunuh pekerja 0 sekali pada +6 menit
        if __INJECT__ and not _inj_done and _now - _OF_T0 > __INJECT_AFTER__ and 0 in _slots:
            _inj_done = True
            print("[OF4] INJECT: SIGKILL slot 0 (uji supervisor)", flush=True)
            _of_killgrp(_slots[0][0])
        # 4) progres
        if _now - _last_print > __PRINTEVERY__:
            _last_print = _now
            _a, _h = _of_mem()
            _st = _os.statvfs(_WD)
            print("[OF4] progress %.2fh | chunks done %d/%d bad %d | workers %s | MemAvail %s GB headroom %s | disk free %.1f GB | GPUs %s" % (
                (_now - _OF_T0) / 3600, _dn, len(_chunks), _bd, {s: round(_of_rss(v[0].pid), 1) for s, v in _slots.items()},
                None if _a is None else round(_a, 1), None if _h is None else round(_h, 1), _st.f_bavail * _st.f_frsize / 1e9, _of_gpus()), flush=True)

    for _p, _lg in list(_slots.values()):
        _of_killgrp(_p)
    for _i in range(8):
        try:
            print(f"--- of worker {_i} log tail ---\n" + "".join(open(f"{_WD}/_leg_of_w{_i}.log").readlines()[-4:]), flush=True)
        except OSError:
            pass
    _parts = [_pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in sorted(_glob.glob(_res + "/*.csv"))]
    assert _parts, "no OF output"
    _d = _pd.concat(_parts).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(_ids)
    _labs = [c for c in _d.columns]
    _cov = 1.0 - float(_d.isna().all(axis=1).mean())
    print(f"[OF4] coverage {_cov:.3f} ({int(round(_cov * len(_ids)))}/{len(_ids)}) | {_time.time() - _OF_T0:.0f}s | spawned {_spawned} | restarts_left {_restarts}", flush=True)
    assert _cov >= __MINCOV__, "OF coverage too low"
    _d.to_csv(_OF_LEG)
    _OF_STATE.update(cov=_cov, ok=True, why="ok")
except Exception as _e:
    _tb.print_exc()
    _OF_STATE.update(ok=False, why=f"{type(_e).__name__}: {_e}")
    print(f"[OF4 EXCEPTION] {type(_e).__name__}: {_e} -> leg skipped (Apex roda sem OF)", flush=True)
finally:
    try:
        for _p, _lg in list(_slots.values()):
            _of_killgrp(_p)
    except Exception:
        pass
    for _d2 in (_WD + "/of", _WD + "/cloud_code", _WD + "/_of_env", _WD + "/fr", _WD + "/ofo", _WD + "/_of_claims", _WD + "/_of_claims_fail"):
        _shutil.rmtree(_d2, ignore_errors=True)
    _st = _os.statvfs(_WD)
    print("[OF4] cleanup done | disk free %.1f GB | state %s" % (_st.f_bavail * _st.f_frsize / 1e9, _OF_STATE), flush=True)
'''

BLEND4 = r'''
# ---------------- blend v4: Apex (submission.csv) + kaki OF (rank-mean ponderado) ----------------
import os as _os, shutil as _shutil, traceback as _tb
import numpy as _np, pandas as _pd
_pub = "/kaggle/working/submission.csv"
_bak = "/kaggle/working/_apex_before_legs.csv"
_shutil.copy(_pub, _bak)
try:
    _leg = "/kaggle/working/_leg_of.csv"
    if not _os.path.exists(_leg):
        raise RuntimeError("no OF leg file | OF state: " + str(globals().get("_OF_STATE")))
    _apex = _pd.read_csv(_bak, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    _ids, _labels = _apex.index.tolist(), list(_apex.columns)
    _d = _pd.read_csv(_leg, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(_ids)[_labels]
    _cov = 1.0 - float(_d.isna().all(axis=1).mean())
    assert _cov >= __MINCOV__, f"OF coverage {_cov:.3f} too low"
    _ra = _apex.rank(method="average", pct=True)
    _rl = _d.rank(method="average", pct=True).fillna(_ra)
    _fused = ((1.0 - __WO__) * _ra + __WO__ * _rl).rank(method="average", pct=True)
    assert _fused.notna().all().all(), "fused NaN"
    assert (_fused.nunique() > 1).all() or len(_ids) < 5, "fused constant"
    if len(_ids) >= 20:
        print("[BLEND] OF w=%.2f coverage %.3f mean corr(apex,leg)=%.3f" % (__WO__, _cov, _np.mean([_np.corrcoef(_ra[c], _rl[c])[0, 1] for c in _labels])), flush=True)
    _fused.reset_index().to_csv(_pub, index=False)
    print(f"[BLEND SUCCESS] Apex + OF (coverage {_cov:.3f}) -> {_pub}", flush=True)
except Exception as _e:
    _tb.print_exc()
    print(f"[BLEND EXCEPTION] {type(_e).__name__}: {_e}", flush=True)
    if __LOUD__:
        raise            # modo percobaan: gagal KERAS supaya status submisi (ERROR) membuktikan kaki OF tidak jalan di run penilaian
    print("[BLEND] restoring Apex-only output", flush=True)
    _shutil.copy(_bak, _pub)
print("Final submission:", _os.path.getsize(_pub), "bytes", flush=True)
'''

DATA_PRESET_PROD = '_of_data = _of_find("test_series.csv")'
DATA_PRESET_PSEUDO = '_of_data = "/kaggle/working/pseudo" if _os.path.exists("/kaggle/working/pseudo/test_series.csv") else _of_find("test_series.csv")'
