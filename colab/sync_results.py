#!/usr/bin/env python3
"""Tarik hasil latih dari VM Colab (checkpoint + OOF) lalu unggah sbg dataset Kaggle (buat / versi baru).
Pemakaian: python3 colab/sync_results.py <outtag> <dataset_slug> [sesi=rsna] [vm_dir=/content/kw]
Contoh   : python3 colab/sync_results.py cheap_radresnet50_ep12_s1 rsna-knee-radresnet50-ep12 rsna
Hanya berkas fold yang SELESAI (ada .pt dan oof_*.csv) yang diambil. Dataset mendarat di
/kaggle/input/datasets/dimaspashaakrilian/<slug>/ -> glob leg `build_submit_plus.py` menemukannya apa adanya."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
tag, slug = sys.argv[1], sys.argv[2]
sess = sys.argv[3] if len(sys.argv) > 3 else "rsna"
vm = sys.argv[4] if len(sys.argv) > 4 else "/content/kw"
env = {**os.environ, "PATH": os.environ["PATH"] + ":" + os.path.expanduser("~/.local/bin")}
def run(c, **k): return subprocess.run(c, shell=True, capture_output=True, text=True, env=env, **k)
import ast
raw = subprocess.run(["colab", "exec", "-s", sess], input=f"import os; print(sorted(os.listdir('{vm}')))\n", capture_output=True, text=True, env=env).stdout
lst = [l for l in raw.splitlines() if l.startswith("[") and "'" in l]
files = [x for x in (ast.literal_eval(lst[-1]) if lst else []) if x.startswith((tag + "_f", "oof_" + tag + "_f"))]
folds = sorted({f.rsplit("_f", 1)[1].split(".")[0] for f in files if f.endswith(".pt")} & {f.rsplit("_f", 1)[1].split(".")[0] for f in files if f.endswith(".csv")})
print("fold selesai:", folds)
if not folds: sys.exit("belum ada fold selesai")
out = Path(tempfile.gettempdir()) / f"colab_sync_{slug}"; out.mkdir(exist_ok=True)
for k in folds:
    for f in (f"{tag}_f{k}.pt", f"oof_{tag}_f{k}.csv"):
        if not (out / f).exists():
            r = run(f"colab download -s {sess} {vm}/{f} {out}/{f}"); print(f, "ok" if (out / f).exists() else r.stderr[-200:])
(out / "dataset-metadata.json").write_text(json.dumps({"title": slug.replace("-", " "), "id": f"dimaspashaakrilian/{slug}", "licenses": [{"name": "CC0-1.0"}]}))
import time as _t
for attempt in range(6):
    r = run(f"kaggle datasets version -p {out} -m 'folds {','.join(folds)}'")
    txt = r.stdout + r.stderr
    if "not found" in txt.lower() or "404" in txt:
        r = run(f"kaggle datasets create -p {out}"); txt = r.stdout + r.stderr
    if "429" in txt or "Too Many Requests" in txt:
        print("429 -> tunggu 90 dtk"); _t.sleep(90); continue
    break
print(txt[-400:])
