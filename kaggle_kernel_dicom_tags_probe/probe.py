import os, glob, json, collections, time
import numpy as np, pandas as pd, pydicom
t0 = time.time()
ROOT = next(b for b in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"] if os.path.exists(b + "/train.csv"))
ser = pd.read_csv(ROOT + "/train_series.csv")
tr = pd.read_csv(ROOT + "/train.csv")
uids = tr["StudyInstanceUID"].astype(str).tolist()
rng = np.random.RandomState(0)
gold = tr[tr[["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]].notna().all(axis=1)]["StudyInstanceUID"].astype(str).tolist()
pick = list(dict.fromkeys(gold + list(rng.choice(uids, 400, replace=False))))
rows, tagcount, tagvals = [], collections.Counter(), collections.defaultdict(collections.Counter)
for i, sid in enumerate(pick):
    for sr in ser[ser.StudyInstanceUID == sid].itertuples():
        d = f"{ROOT}/train_series/{sid}/{sr.SeriesInstanceUID}"
        fs = sorted(glob.glob(d + "/*.dcm"))
        if not fs: continue
        try:
            h = pydicom.dcmread(fs[len(fs)//2], stop_before_pixels=True)
        except Exception as e:
            continue
        rec = {"sid": sid, "series": sr.SeriesInstanceUID, "plane": sr.Anatomical_Plane, "n": len(fs)}
        for el in h:
            if el.keyword and el.VR not in ("SQ", "OB", "OW", "UN") and el.keyword not in ("PixelData",):
                v = str(el.value)[:60]
                rec[el.keyword] = v
                if v not in ("", "None"):
                    tagcount[el.keyword] += 1
                    if len(tagvals[el.keyword]) < 200: tagvals[el.keyword][v] += 1
        rows.append(rec)
    if i % 50 == 0: print(i, len(rows), round(time.time() - t0), flush=True)
df = pd.DataFrame(rows); df.to_csv("/kaggle/working/tags_per_series.csv", index=False)
print("series read:", len(df))
for k, c in tagcount.most_common(80):
    vals = tagvals[k]
    print(f"{k:34s} n={c:5d} uniq~{len(vals):4d} ex={list(vals.items())[:4]}")
