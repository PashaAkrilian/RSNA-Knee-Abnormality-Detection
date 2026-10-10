#!/usr/bin/env python3
"""Uji SKALA OrthoFoundation fold0 (commit-run, tanpa submisi): meniru kondisi scoring.

Penyebab dugaan Notebook Timeout 57028008: (1) tiap potongan memanggil run_inference pada akar data penuh sehingga
DirectorySource memetakan SELURUH test_series (~1.322 studi) berulang kali; (2) oversubscription CPU (4 vCPU; torch
set_num_threads(4) di predict + BLAS tanpa batas); (3) tanpa pembunuhan proses pada batas waktu.
Uji ini: (a) mengukur waktu pemetaan akar besar (1.322 studi simbolik), (b) menjalankan P proses (P/2 per GPU) dengan akar
per-potongan + OMP/MKL=1 + batas thread torch, selama TLIMIT menit, dan (c) mencetak throughput (studi/menit/proses).
Pemakaian: python3 build_scale_of.py name=ofscale2 procs=2 workers=2 chunk=30 tlimit=28
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = args.get("name", "ofscale2")
PROCS = int(args.get("procs", "2"))
WORKERS = int(args.get("workers", "2"))
CHUNK = int(args.get("chunk", "30"))
TLIMIT = int(args.get("tlimit", "28"))
THREADS = int(args.get("threads", "2"))

CODE = r'''
import os, sys, time, json, glob, shutil, hashlib, subprocess, textwrap, threading, csv
import pandas as pd, torch

PROCS, WORKERS, CHUNK, TLIMIT_MIN, THREADS = __PROCS__, __WORKERS__, __CHUNK__, __TLIMIT__, __THREADS__
COMP = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
def find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)

T0 = time.time()
OF = find("weight_parts.json"); UL = find("assets/bus.jpg")
MD, CC, TGT = "/kaggle/working/of", "/kaggle/working/cloud_code", "/kaggle/working/_of_env"
os.makedirs(MD, exist_ok=True)
shutil.copytree(f"{OF}/cloud_code", CC, dirs_exist_ok=True)
shutil.copytree(UL, f"{CC}/training/vendor_runtime/kneexnet/ultralytics", dirs_exist_ok=True)
h = hashlib.sha256()
with open(f"{MD}/best.pt", "wb") as out:
    for p in sorted(glob.glob(f"{OF}/weights.part*")):
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b: break
                out.write(b); h.update(b)
assert h.hexdigest().startswith("4f424cc7")
shutil.copy2(f"{OF}/anatomy.pt", f"{MD}/anatomy.pt")
shutil.copytree(f"{OF}/dependencies/kneexnet", f"{MD}/kneexnet", dirs_exist_ok=True)
for w in sorted(glob.glob(f"{OF}/dependencies/wheels/*.whl")):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", TGT, w], check=False)
print("SETUP done %.0fs" % (time.time() - T0), flush=True)

# --- (a) custo de enumerar uma raiz grande: 1322 estudos simbolicos
tr_series = pd.read_csv(f"{COMP}/train_series.csv")
tr = pd.read_csv(f"{COMP}/train.csv")
LABEL = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
         'Synovitis', "Baker's", 'Contusion', 'Fracture']
gold = set(tr[tr[LABEL].notna().all(axis=1)]["StudyInstanceUID"])
all_ids = [s for s in tr["StudyInstanceUID"].tolist() if s not in gold]
BIG = all_ids[:1322]
big_root = "/kaggle/working/bigroot"
os.makedirs(f"{big_root}/test_series", exist_ok=True)
for sid in BIG:
    d = f"{big_root}/test_series/{sid}"
    if not os.path.exists(d):
        os.symlink(f"{COMP}/train_series/{sid}", d)
t = time.time()
n_files = sum(1 for _ in __import__("pathlib").Path(f"{big_root}/test_series").glob("*/*/*.dcm"))
print("ENUM_BIGROOT studies", len(BIG), "dcm files", n_files, "seconds %.1f" % (time.time() - t), flush=True)

# --- ordenar por custo (n. de arquivos DICOM) e intercalar entre os processos (licao do controller heliosli)
def cost(uid):
    n = 0
    try:
        for s in os.scandir(f"{COMP}/train_series/{uid}"):
            if s.is_dir():
                n += sum(1 for _ in os.scandir(s.path))
    except OSError:
        pass
    return n
costs = {u: cost(u) for u in BIG}
order = sorted(BIG, key=lambda u: (-costs[u], u))
parts = [order[i::PROCS] for i in range(PROCS)]
parts = [p[::2] + p[1::2] for p in parts]
print("COSTS mean dcm/study %.0f" % (sum(costs.values()) / len(costs)), "| parts", [len(p) for p in parts], flush=True)
tr_series_idx = tr_series.set_index("StudyInstanceUID", drop=False)
real_test_cols = list(pd.read_csv(f"{COMP}/test.csv", nrows=1).columns)
real_ts_cols = list(pd.read_csv(f"{COMP}/test_series.csv", nrows=1).columns)

worker = textwrap.dedent("""
    import os, sys, time, shutil, json, subprocess
    import pandas as pd
    import torch
    _o = torch.set_num_threads
    torch.set_num_threads = lambda n: _o(min(int(n), __THREADS__))
    torch.set_num_threads(__THREADS__)
    shard, ids_json, deadline, chunk, workers = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    COMP = sys.argv[6]; MD = sys.argv[7]
    ids = json.load(open(ids_json))
    from training.infer import run_inference
    tr_series = pd.read_csv(COMP + "/train_series.csv")
    real_test_cols = list(pd.read_csv(COMP + "/test.csv", nrows=1).columns)
    real_ts_cols = list(pd.read_csv(COMP + "/test_series.csv", nrows=1).columns)
    t0 = time.time(); done = 0; log = []
    for k in range(0, len(ids), chunk):
        if time.time() > deadline:
            print("DEADLINE", flush=True); break
        ch = ids[k:k + chunk]
        root = f"/kaggle/working/fr/s{shard}_{k}"
        os.makedirs(root + "/test_series", exist_ok=True)
        pd.DataFrame({c: (ch if c == "StudyInstanceUID" else None) for c in real_test_cols}).to_csv(root + "/test.csv", index=False)
        sub = tr_series[tr_series["StudyInstanceUID"].isin(ch)]
        sub[[c for c in real_ts_cols if c in sub.columns]].to_csv(root + "/test_series.csv", index=False)
        for sid in ch:
            os.symlink(COMP + "/train_series/" + sid, root + "/test_series/" + sid)
        out = f"/kaggle/working/ofo/s{shard}_{k}"
        tc = time.time()
        try:
            fr = run_inference(root, MD + "/best.pt", out, anatomy_weights=MD + "/anatomy.pt", workers=workers)
            ok = len(fr)
        except Exception as e:
            ok = 0; print("CHUNK FAIL", repr(e)[:300], flush=True)
        dt = time.time() - tc
        try:
            du = subprocess.run(["du", "-sm", out], capture_output=True, text=True).stdout.split()[0]
        except Exception:
            du = "?"
        shutil.rmtree(out, ignore_errors=True); shutil.rmtree(root, ignore_errors=True)
        done += ok
        print("CHUNK shard", shard, "k", k, "n", len(ch), "ok", ok, "sec %.0f" % dt, "sec/study %.1f" % (dt / len(ch)), "out_MB", du,
              "| cumul %d studies in %.0fs = %.1f s/study" % (done, time.time() - t0, (time.time() - t0) / max(1, done)), flush=True)
    print("WORKER_DONE shard", shard, "studies", done, "elapsed %.0f" % (time.time() - t0), flush=True)
""").replace("__THREADS__", str(THREADS))
open("/kaggle/working/_scale_worker.py", "w").write(worker)

deadline = time.time() + TLIMIT_MIN * 60
procs = []
for i in range(PROCS):
    gpu = i % max(1, torch.cuda.device_count())
    ids_json = f"/kaggle/working/_ids_{i}.json"
    json.dump(parts[i], open(ids_json, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=TGT + os.pathsep + CC)
    lg = open(f"/kaggle/working/_scale_w{i}.log", "w")
    procs.append((subprocess.Popen([sys.executable, "-u", "/kaggle/working/_scale_worker.py", str(i), ids_json, str(deadline),
                                    str(CHUNK), str(WORKERS), COMP, MD], env=env, stdout=lg, stderr=subprocess.STDOUT), lg))
print("LAUNCHED procs", PROCS, "workers", WORKERS, "chunk", CHUNK, "threads", THREADS, "tlimit_min", TLIMIT_MIN, flush=True)

def gpu_mon():
    while any(p.poll() is None for p, _ in procs):
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
            la = open("/proc/loadavg").read().split()[0]
            print("MON %.0fs gpu(util,mem): %s | load1 %s" % (time.time() - T0, o, la), flush=True)
        except Exception:
            pass
        time.sleep(90)
th = threading.Thread(target=gpu_mon, daemon=True); th.start()

hard = deadline + 25 * 60
while any(p.poll() is None for p, _ in procs):
    time.sleep(10)
    if time.time() > hard:
        for p, _ in procs:
            if p.poll() is None:
                p.kill()
        print("KILLED stragglers", flush=True); break
for i, (p, lg) in enumerate(procs):
    lg.close()
    print(f"--- worker {i} log (chunk lines) ---", flush=True)
    for line in open(f"/kaggle/working/_scale_w{i}.log").read().splitlines():
        if line.startswith(("CHUNK", "WORKER_DONE", "DEADLINE")):
            print(line[:260], flush=True)
tot = 0
for i in range(PROCS):
    for line in open(f"/kaggle/working/_scale_w{i}.log").read().splitlines():
        if line.startswith("WORKER_DONE"):
            tot += int(line.split("studies")[1].split()[0])
print("SCALE_SUMMARY procs", PROCS, "workers", WORKERS, "chunk", CHUNK, "studies_done", tot, "total_elapsed_min %.1f" % ((time.time() - T0) / 60), flush=True)
print("PROJECTION: full 1322-study run would take %.1f h of inference (+setup)" % ((time.time() - T0 - 120) / 3600 * 1322 / max(1, tot)), flush=True)
'''

code = (CODE.replace("__PROCS__", str(PROCS)).replace("__WORKERS__", str(WORKERS)).replace("__CHUNK__", str(CHUNK))
        .replace("__TLIMIT__", str(TLIMIT)).replace("__THREADS__", str(THREADS)))
compile(code, "scale", "exec")
nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# RSNA Knee - OrthoFoundation scale test (commit-run, tanpa submisi)\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.lstrip("\n").splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
slug = "rsna-knee-" + NAME
out = ROOT / f"kaggle_kernel_{NAME}"
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"dimaspashaakrilian/{slug}", "title": "RSNA Knee " + NAME, "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": ["heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "dimaspashaakrilian/rsna-knee-ultralytics-cfg"],
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ditulis:", out, "| procs", PROCS, "workers", WORKERS, "chunk", CHUNK, "tlimit", TLIMIT)
