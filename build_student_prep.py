#!/usr/bin/env python3
"""Bangun kernel student-prep (cache256 + labels konsensus+teacher) untuk akun alver."""
import json, os

MD = {
    "id": "alverlfarfaraway/rsna-knee-student-prep",
    "title": "RSNA Knee student prep",
    "code_file": "rsna-knee-student-prep.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "category": "analysis",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "stevenleehans/rsna-knee-llm-report-labels",
        "pilkwang/rsna-knee-llm-labels",
        "lixin73/rsna-knee-llm-report-labels-sol56",
        "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels",
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
        "alverlfarfaraway/rsna-knee-teacher-r384-train",
    ],
    "keywords": ["rsna", "knee", "student", "prep"],
    "kernel_sources": [],
}

CELL0 = """# RSNA Knee — student-prep (Fase B langkah 2a)
# Cache uint8 256 (FOV 128 mm, max 28 slice) + label konsensus 4 tabel ⊕ teacher r384 (50/50).
# Output: cache256_train/*.npy, cache256_train_meta.csv, labels_student_v0.csv → dataset prep alver.
"""

CELL1 = r'''
import os, sys, glob, json, time, shutil, subprocess
from multiprocessing import Pool
import numpy as np, pandas as pd

t0 = time.time()
def say(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

# ---- cari mount ----
def find(pats, base="/kaggle/input"):
    for p in pats:
        for f in glob.glob(os.path.join(base, "**", p), recursive=True):
            return f
    for p in pats:
        for f in glob.glob(os.path.join(base, p), recursive=True):
            return f
    return None

COMP = None
for c in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
          "/kaggle/input/rsna-knee-abnormality-detection"] + sorted(glob.glob("/kaggle/input/*")):
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(os.path.join(c, "train_series.csv")):
        COMP = c; break
    for sub in sorted(glob.glob(os.path.join(c, "*"))):
        if os.path.exists(os.path.join(sub, "train.csv")) and os.path.exists(os.path.join(sub, "train_series.csv")):
            COMP = sub; break
    if COMP: break
assert COMP, "competition mount tidak ketemu; listing: " + str(sorted(os.listdir("/kaggle/input"))[:30])
say("COMP =", COMP)
TRAIN = pd.read_csv(f"{COMP}/train.csv")
SERIES = pd.read_csv(f"{COMP}/train_series.csv")
say("train", len(TRAIN), "series rows", len(SERIES))

EXT = {}
# labels sudah di dataset cache256 (diunggah dari lokal) — cell labels dilewati di v4 cache-only
TEACHER = None
say("cache-only v4: labels dari dataset cache256")

# pylibjpeg utk decode cepat (wheel dari dataset goodpjw)
whl_dir = os.path.dirname(find(["pylibjpeg-2.1.0-py3-none-any.whl"]) or "")
if whl_dir:
    for w in sorted(glob.glob(os.path.join(whl_dir, "*.whl"))):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", w], check=False)
say("pylibjpeg:", subprocess.run([sys.executable, "-m", "pip", "show", "pylibjpeg"], capture_output=True).returncode == 0)
'''

# ---------- preprocess (adaptasi goodpjw; SIZE 256, MM 0.5, MAX 28) ----------
CELL2 = r'''
import cv2, pydicom, numpy as np, pandas as pd, os
from multiprocessing import Pool

DATA = COMP
WORK = "/tmp"          # ephemeral disk lebih besar dari /kaggle/working
MM_PER_PX = 0.5
SIZE = 256             # FOV 128 mm
MAX_SLICES = 28

CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
PLANES = ["Sagittal", "Coronal", "Axial"]

def _read(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]

def _canon_slice(img, iop, ps, plane):
    ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2
    if iop is None:
        return img, ps
    r, c = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]):
        img, r, c, ps = img.T, c, r, ps[::-1]
    if r[cax] * csgn < 0:
        img = img[:, ::-1]
    if c[rax] * rsgn < 0:
        img = img[::-1]
    return img, ps

def _resample(img, ps):
    out = np.zeros((SIZE, SIZE), np.float32)
    h, w = img.shape
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
    cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
    nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    res = cv2.resize(crop, (nw, nh), interpolation=interp)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out

def load_series(sdir, plane_name=None):
    slices, bad = [], 0
    for fn in os.listdir(sdir):
        try:
            slices.extend(_read(os.path.join(sdir, fn)))
        except Exception:
            bad += 1
    info = {"n_raw": len(slices), "n_bad": bad}
    if not slices:
        return np.zeros((1, SIZE, SIZE), np.uint8), info
    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:]); plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
    info["plane"] = PLANES[plane]
    if normal is not None and all(s[2] is not None for s in slices):
        sgn = CANON[plane][0] * np.sign(normal[plane])   # urut by IPP (bukan filename)
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]
    vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    info["n"] = len(vol)
    return (vol * 255).round().astype(np.uint8), info

def _job(args):
    study, series, plane, out_dir = args
    out = f"{out_dir}/{series}.npy"
    info = {"study": study, "series": series}
    if os.path.exists(out):
        return None
    try:
        vol, extra = load_series(f"{DATA}/train_series/{study}/{series}")
        np.save(out + ".tmp.npy", vol); os.replace(out + ".tmp.npy", out)
        info.update(extra)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info

OUT = f"{WORK}/cache256_train"; os.makedirs(OUT, exist_ok=True)
tasks = [(a, b, p, OUT) for a, b, p in zip(SERIES.StudyInstanceUID, SERIES.SeriesInstanceUID, SERIES.Anatomical_Plane)]
rows = []
with Pool(4) as p:
    for i, r in enumerate(p.imap_unordered(_job, tasks, chunksize=8)):
        if r: rows.append(r)
        if i % 1000 == 0:
            say(f"cache {i}/{len(tasks)} files={len(os.listdir(OUT))} disk={shutil.disk_usage('/tmp').free/2**30:.1f}GB")
            if shutil.disk_usage("/tmp").free < 3 * 2**30:
                say("DISK HAMPIR HENTI — pakai cache parsial"); break
df = pd.DataFrame(rows)
if "error" in df: say("errors:", int(df["error"].notna().sum()))
meta = df.dropna(subset=["n"]).copy()
meta.to_csv("/kaggle/working/cache256_train_meta.csv", index=False)
say("meta rows", len(meta), "files", len(os.listdir(OUT)))
'''

# ---------- labels: konsensus 4 ⊕ teacher 0.5 ----------
CELL3 = r'''
say("labels: dilewati (cache-only v4) — labels_student_v0.csv sudah di dataset cache256")
'''

# ---------- push cache ke dataset cache256 ----------
CELL4 = r'''
import os, json, subprocess, glob, shutil
KC = None
for p in ("/root/.kaggle/kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json")):
    if os.path.exists(p): KC = p; break
assert KC, "kaggle.json credentials tidak ada"
cred = json.load(open(KC))
os.environ["KAGGLE_USERNAME"] = cred["username"]; os.environ["KAGGLE_KEY"] = cred["key"]
DS_ID = "alverlfarfaraway/rsna-knee-student-cache256"
meta = {
    "id": DS_ID, "title": "RSNA Knee student cache256",
    "licenses": [{"name": "Other"}], "keywords": ["rsna", "knee", "cache"], "resources": []
}
os.makedirs("/tmp/ds_push", exist_ok=True)
with open("/tmp/ds_push/dataset-metadata.json", "w") as f:
    json.dump(meta, f)
if os.path.exists("/kaggle/working/cache256_train_meta.csv"):
    shutil.copy("/kaggle/working/cache256_train_meta.csv", "/tmp/ds_push/")
assert os.path.exists("/tmp/cache256_train"), "cache tidak ada"
n = len(os.listdir("/tmp/cache256_train")); say("cache files", n)
os.system("ln -sfn /tmp/cache256_train /tmp/ds_push/cache256_train")
say("pushing dataset cache (bisa lama)...")
r = subprocess.run(["kaggle", "datasets", "version", "-p", "/tmp/ds_push", "-m", "cache256 train (24371 series)"],
                   capture_output=True, text=True)
say("push rc", r.returncode); say((r.stdout + r.stderr)[-600:])
say("DONE cache-push")
'''

def cell(src, ctype="code"):
    return {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if ctype == "code" else {})}

nb = {"cells": [cell(CELL0, "markdown"), cell(CELL1), cell(CELL2), cell(CELL3), cell(CELL4)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "kaggle_kernel_student_prep"
os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/rsna-knee-student-prep.ipynb", "w"), indent=1)
json.dump(MD, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("ditulis:", out)
