#!/usr/bin/env python3
"""Kernel profil memori pekerja OF (commit-run, tanpa submisi).

Latar (10 Okt 23:50): rehearsal apexof4r menunjukkan tiap pekerja OF mencapai RSS 8-10 GB (7,3-8,5 GB di akhir potongan pertama, 10,6 GB setelah potongan kedua);
3 pekerja bersamaan -> OOM (rc -9) di mesin 30 GB. Kernel ini mengukur kurva RSS (anon/file) tiap 5 dtk utk beberapa konfigurasi, 1 pekerja per konfigurasi
(2 konfigurasi paralel), utk memilih: ukuran potongan, malloc (MMAP/TRIM threshold, tcmalloc bila ada), DataLoader workers=0, daur-ulang per potongan.

Konfigurasi (nama:env;chunk;workers_dl;n_potongan):
  base      : env v4; chunk 8; workers 1; 2 potongan
  mallocthr : + MALLOC_MMAP_THRESHOLD_=1048576, MALLOC_TRIM_THRESHOLD_=1048576; chunk 8; workers 1; 2 potongan
  chunk4    : env v4; chunk 4; workers 1; 4 potongan
  w0        : env v4; chunk 8; workers 0; 2 potongan
  tcmalloc  : LD_PRELOAD tcmalloc/jemalloc bila ada; chunk 8; workers 1; 2 potongan
Pemakaian: python3 build_ofmem.py name=ofmem1 n=16
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "ofmem1")
N = int(args.get("n", "16"))
GROUPS = args.get("groups", "base,mallocthr;chunk4,w0;tcmalloc")   # grupos separados por ; (paralelos dentro do grupo, separados por ,)

CELL = r'''
import os, sys, glob, json, time, shutil, subprocess as sp, hashlib, random, textwrap, threading
import pandas as pd
WD = "/kaggle/working"
T0 = time.time()
COMP = next(p for p in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"] if os.path.exists(p + "/train.csv"))
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
tr = pd.read_csv(COMP + "/train.csv")
gold = set(tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"])
ids = [s for s in tr["StudyInstanceUID"].tolist() if s not in gold]
random.Random(7).shuffle(ids)
ids = ids[:__N__]
PS = WD + "/pseudo"
shutil.rmtree(PS, ignore_errors=True)
os.makedirs(PS + "/test_series")
for s in ids:
    os.symlink(f"{COMP}/train_series/{s}", f"{PS}/test_series/{s}")
trs = pd.read_csv(COMP + "/train_series.csv")
pcols = [c for c in pd.read_csv(COMP + "/test_series.csv", nrows=2).columns if c in trs.columns]
trs[trs["StudyInstanceUID"].isin(set(ids))][pcols].to_csv(PS + "/test_series.csv", index=False)
ocols = list(pd.read_csv(COMP + "/test.csv", nrows=1).columns)
pd.DataFrame({c: (ids if c == "StudyInstanceUID" else None) for c in ocols}).to_csv(PS + "/test.csv", index=False)

def find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)

OF = find("weight_parts.json"); UL = find("assets/bus.jpg")
MD, CC, TG = WD + "/of", WD + "/cloud_code", WD + "/_of_env"
os.makedirs(MD, exist_ok=True)
shutil.copytree(f"{OF}/cloud_code", CC)
shutil.copytree(UL, f"{CC}/training/vendor_runtime/kneexnet/ultralytics", dirs_exist_ok=True)
h = hashlib.sha256()
with open(f"{MD}/best.pt", "wb") as out:
    for p in sorted(glob.glob(f"{OF}/weights.part*")):
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                out.write(b); h.update(b)
assert h.hexdigest().startswith("4f424cc7")
shutil.copy2(f"{OF}/anatomy.pt", f"{MD}/anatomy.pt")
shutil.copytree(f"{OF}/dependencies/kneexnet", f"{MD}/kneexnet")
for w in sorted(glob.glob(f"{OF}/dependencies/wheels/*.whl")):
    sp.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", TG, w], check=False)
GP = find("cnxt_v0_fold0.pt")
sp.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", WD + "/_own_env", "--find-links", GP,
        "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
print("[MEM] setup done %.0fs | studies %d" % (time.time() - T0, len(ids)), flush=True)
print("[MEM] allocators:", sp.run("ldconfig -p | grep -i -E 'tcmalloc|jemalloc' | head -5; ls /usr/lib/x86_64-linux-gnu | grep -i -E 'tcmalloc|jemalloc' | head -5", shell=True, capture_output=True, text=True).stdout.replace("\n", " ; "), flush=True)
print("[MEM] mem:", open("/proc/meminfo").read().splitlines()[0:3], "| cpus", os.cpu_count(), flush=True)

WSRC = r"""
import os, sys, time, json, shutil, threading
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import torch
_o = torch.set_num_threads
torch.set_num_threads = lambda n: _o(1)
torch.set_num_threads(1)
try:
    import cv2; cv2.setNumThreads(1)
except Exception:
    pass
class TS:
    def __init__(self, s): self.s = s; self.t0 = time.time(); self.buf = ""
    def write(self, x):
        self.buf += x
        while "\n" in self.buf:
            l, self.buf = self.buf.split("\n", 1)
            if "get_mempolicy" in l or "pkg_resources" in l or "__import__" in l: continue
            self.s.write("%7.1f %s\n" % (time.time() - self.t0, l[:180])); self.s.flush()
    def flush(self): self.s.flush()
sys.stdout = TS(sys.__stdout__)
if os.environ.get("OFMEM_MMAP") == "1":
    _ol = torch.load
    def _ml(f, *a, **k):
        k.setdefault("mmap", True)
        return _ol(f, *a, **k)
    torch.load = _ml
    print("torch.load patched with mmap=True", flush=True)
def rss():
    d = {}
    for l in open("/proc/self/status"):
        if l.startswith(("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM")):
            k, v = l.split(":"); d[k] = int(v.split()[0]) / 1048576.0
    ch = 0.0
    try:
        for pid in os.listdir("/proc"):
            if pid.isdigit() and int(pid) != os.getpid():
                try:
                    st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                    if int(st[1]) == os.getpid():
                        for l in open(f"/proc/{pid}/status"):
                            if l.startswith("VmRSS"): ch += int(l.split()[1]) / 1048576.0
                except Exception:
                    pass
    except Exception:
        pass
    d["children"] = ch
    return d
stop = False
def sampler():
    while not stop:
        d = rss()
        print("RSS anon=%.2f file=%.2f shmem=%.2f vmrss=%.2f hwm=%.2f children=%.2f GB" % (d.get("RssAnon", 0), d.get("RssFile", 0), d.get("RssShmem", 0), d.get("VmRSS", 0), d.get("VmHWM", 0), d["children"]), flush=True)
        time.sleep(5)
threading.Thread(target=sampler, daemon=True).start()
name, chunk, wdl, nchunks, md, ps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
ids = pd.read_csv(ps + "/test.csv", dtype=str).StudyInstanceUID.tolist()
off = int(sys.argv[7])
ids = ids[off: off + chunk * nchunks]
from training.infer import run_inference
ser = pd.read_csv(ps + "/test_series.csv")
test_cols = list(pd.read_csv(ps + "/test.csv", nrows=1).columns)
t0 = time.time()
for k in range(0, len(ids), chunk):
    ch = ids[k:k + chunk]
    root = f"/kaggle/working/fr/{name}_{k}"; out = f"/kaggle/working/ofo/{name}_{k}"
    shutil.rmtree(root, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
    os.makedirs(root + "/test_series", exist_ok=True)
    pd.DataFrame({c: (ch if c == "StudyInstanceUID" else None) for c in test_cols}).to_csv(root + "/test.csv", index=False)
    ser[ser["StudyInstanceUID"].isin(ch)].to_csv(root + "/test_series.csv", index=False)
    for sid in ch: os.symlink(ps + "/test_series/" + sid, root + "/test_series/" + sid)
    tc = time.time()
    print("CHUNKSTART k=%d n=%d" % (k, len(ch)), flush=True)
    fr = run_inference(root, md + "/best.pt", out, anatomy_weights=md + "/anatomy.pt", workers=wdl)
    d = rss()
    print("CHUNKEND %s k=%d n=%d ok=%d sec=%.0f (%.1f s/study) anon=%.2f file=%.2f hwm=%.2f children=%.2f GB" % (name, k, len(ch), len(fr), time.time() - tc, (time.time() - tc) / len(ch), d.get("RssAnon", 0), d.get("RssFile", 0), d.get("VmHWM", 0), d["children"]), flush=True)
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(root, ignore_errors=True)
stop = True
print("DONE %s elapsed=%.0f peak_hwm=%.2f" % (name, time.time() - t0, rss().get("VmHWM", 0)), flush=True)
"""
open(WD + "/_mem_worker.py", "w").write(WSRC)

BASEENV = dict(os.environ)
BASEENV.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join([TG, CC, WD + "/_own_env", os.environ.get("PYTHONPATH", "")]))
MT = {"MALLOC_MMAP_THRESHOLD_": "1048576", "MALLOC_TRIM_THRESHOLD_": "1048576"}
CFG = {
    "base": ({}, 8, 1, 2),
    "mallocthr": (MT, 8, 1, 2),
    "chunk4": ({}, 4, 1, 4),
    "w0": ({}, 8, 0, 2),
    "mmap": ({"OFMEM_MMAP": "1"}, 8, 1, 2),
    "mmapw0": ({"OFMEM_MMAP": "1"}, 8, 0, 2),
    "mmapchunk4": ({"OFMEM_MMAP": "1"}, 4, 1, 4),
    "mmapmalloc": (dict(MT, OFMEM_MMAP="1"), 8, 1, 2),
}
tc = sp.run("ls /usr/lib/x86_64-linux-gnu/libtcmalloc*.so* /usr/lib/x86_64-linux-gnu/libjemalloc*.so* 2>/dev/null | head -1", shell=True, capture_output=True, text=True).stdout.strip()
if tc:
    CFG["tcmalloc"] = ({"LD_PRELOAD": tc}, 8, 1, 2)
    CFG["mmaptcmalloc"] = ({"LD_PRELOAD": tc, "OFMEM_MMAP": "1"}, 8, 1, 2)
print("[MEM] configs:", list(CFG), "| tcmalloc/jemalloc:", tc or "none", flush=True)

def launch(name, gpu, off):
    envx, chunk, wdl, nch = CFG[name]
    e = dict(BASEENV); e.update(envx); e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lg = open(f"{WD}/_mem_{name}.log", "w")
    return sp.Popen([sys.executable, "-u", WD + "/_mem_worker.py", name, str(chunk), str(wdl), str(nch), MD, PS, str(off)], env=e, stdout=lg, stderr=sp.STDOUT, start_new_session=True), lg

order = [tuple(n for n in g.split(",") if n in CFG) for g in GROUPS.split(";")]
order = [g for g in order if g]
for grp in order:
    ps_ = [launch(n, i % 2, 0) for i, n in enumerate(grp)]
    t1 = time.time()
    while any(p.poll() is None for p, _ in ps_):
        time.sleep(20)
        if time.time() - t1 > 1500:
            for p, _ in ps_:
                p.kill()
            print("[MEM] timeout group", grp, flush=True)
            break
    for (p, lg), n in zip(ps_, grp):
        lg.close()
        txt = open(f"{WD}/_mem_{n}.log").read().splitlines()
        ce = [l for l in txt if "CHUNKEND" in l or l.split(" ", 1)[-1].startswith("DONE")]
        mx = 0.0
        for l in txt:
            if "RSS anon" in l:
                try:
                    mx = max(mx, float(l.split("vmrss=")[1].split()[0]))
                except Exception:
                    pass
        print(f"[MEM] {n}: rc={p.returncode} max sampled vmrss={mx:.2f} GB", flush=True)
        for l in ce:
            print("[MEM]   " + l.strip(), flush=True)
for n in [x for g in order for x in g]:
    try:
        txt = open(f"{WD}/_mem_{n}.log").read().splitlines()
        rs = [l for l in txt if "RSS anon" in l]
        print(f"[MEM] curve {n}: " + " | ".join(l.split(" ", 1)[0] + "s:" + l.split("vmrss=")[1].split()[0] for l in rs[::6][:40]), flush=True)
    except Exception as e:
        print("[MEM] curve fail", n, repr(e)[:100], flush=True)
for d in (WD + "/of", WD + "/cloud_code", WD + "/_of_env", WD + "/fr", WD + "/ofo", WD + "/pseudo"):
    shutil.rmtree(d, ignore_errors=True)
print("[MEM] ALL DONE %.0fs" % (time.time() - T0), flush=True)
'''

cell = CELL.replace("__N__", str(N))
compile(cell, "ofmem", "exec")
inner = cell.split('WSRC = r"""', 1)[1].split('"""', 1)[0]
compile(inner, "ofmem-worker", "exec")
nb = {
    "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# RSNA Knee - OF memory profile (commit-run)\n"]},
              {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell.lstrip("\n").splitlines(keepends=True)}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": ["heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "dimaspashaakrilian/rsna-knee-ultralytics-cfg",
                        "goodpjw2008/rsna-knee-2-5d-convnext-reader"],
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| slug", slug, "| n", N)
