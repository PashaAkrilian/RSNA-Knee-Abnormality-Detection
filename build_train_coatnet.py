#!/usr/bin/env python3
"""Build rsna-knee-train-coatnet-v1: fine-tune PENUH CoAtNet resep dread/c5
di atas label flight0234 (confidence-weighted), gold-58 TIDAK ikut latih,
evaluasi akhir di gold-58 -> pembanding jujur utk c5 (0.9206).

Komponen model/preprocess dipakai ulang VERBATIM dari _KE_SRC (notebook publik),
jadi checkpoint hasilnya kompatibel dgn semua pipeline inferensi Raptor/DINOsaur.
"""
import ast, json
from pathlib import Path

REPO = Path("/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection")
OUT = REPO / "kaggle_kernel_train_coatnet_v1"
SRC_TXT = Path("/tmp/opencode/nb/raptor45w65/rsna-knee-raptor-v45-w065.txt")

line = [l for l in SRC_TXT.read_text().splitlines() if l.startswith("_KE_SRC")][0]
ke_src = ast.literal_eval(line.split("=", 1)[1].strip())

SETUP = SETUP = '''\
import os, sys, subprocess, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)

def timm_ok():
    try:
        import timm, torch
        m = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
                              pretrained=False, num_classes=0, in_chans=3)
        del m
        return True
    except Exception as e:
        tlog(f'timm bawaan gagal: {type(e).__name__}: {e}')
        return False

if not timm_ok():
    import glob
    hits = glob.glob('/kaggle/input/**/timm-*.whl', recursive=True)
    assert hits, 'wheel timm tidak ketemu'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--no-deps', hits[0]], check=True)
assert timm_ok(), 'timm tidak siap'
tlog('setup OK')
'''

CACHE = '''\
# ===== FASE 1: cache volume uint8 ke disk (dipakai lintas epoch) =====
import numpy as np, pandas as pd, os
KE = _KE_NS
LAB = KE['LAB']
ROOT = KE['find_test_root']()
tsdir = ROOT + '/train_series'
train = pd.read_csv(ROOT + '/train.csv')
train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv')
for c in ('StudyInstanceUID', 'SeriesInstanceUID'):
    tser[c] = tser[c].astype(str)
SER = {k: v[['SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive']].to_dict('records')
       for k, v in tser.groupby('StudyInstanceUID')}
GOLD = set(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in SER]
tlog(f'{len(ALL_IDS)} studi utk cache ({len(GOLD)} gold dikecualikan dr latihan nanti)')

CDIR = '/kaggle/tmp/volcache'
os.makedirs(CDIR, exist_ok=True)
reader = KE['_make_reader']()

def cache_one(sid):
    fn = f'{CDIR}/{sid}.npz'
    if os.path.exists(fn):
        return 'skip'
    try:
        vol, mask = KE['build_study'](sid, SER, tsdir, reader)
        np.savez_compressed(fn, vol=vol, mask=mask)
        return 'ok'
    except Exception as e:
        return f'fail:{type(e).__name__}'

from concurrent.futures import ThreadPoolExecutor
stat = {}
with ThreadPoolExecutor(8) as ex:
    for i, r in enumerate(ex.map(cache_one, ALL_IDS)):
        stat[r] = stat.get(r, 0) + 1
        if (i + 1) % 250 == 0:
            tlog(f'cache {i+1}/{len(ALL_IDS)} | {stat}')
tlog(f'cache SELESAI {stat}')
'''

TRAIN = '''\
# ===== FASE 2: fine-tune penuh dgn label flight0234 (confidence-weighted) =====
import numpy as np, pandas as pd, torch, torch.nn.functional as F, gc, math, random

DEV = 'cuda'
LAB = KE['LAB']; N_T = len(LAB)
import glob as _glob
_hits = _glob.glob('/kaggle/input/**/report_labels_v4hybrid.csv', recursive=True)
tlog('label csv: ' + repr(_hits))
assert _hits, 'flight label csv tidak ketemu di /kaggle/input'
fl = pd.read_csv(_hits[0])
fl['StudyInstanceUID'] = fl['StudyInstanceUID'].astype(str)
FL = fl.set_index('StudyInstanceUID')
CONF_COLS = {t: f'{t}__conf' for t in LAB}

train_ids = [s for s in ALL_IDS if s not in GOLD and s in FL.index]
tlog(f'train ids dgn label: {len(train_ids)}')

def load_study(sid):
    z = np.load(f'/kaggle/tmp/volcache/{sid}.npz')
    return z['vol'], z['mask']

def rand_windows(vol, mask, k, res):
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(max(1, lo), min(D - 1, hi)) ]
    if len(cs) == 0: cs = [max(1, min(D - 2, (lo + hi) // 2))]
    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))
    wins = np.empty((len(cs), 3, res, res), np.float32)
    MEAN = KE['_MEAN']; STD = KE['_STD']
    for j, c in enumerate(cs):
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        # jitter intensitas ringan
        if random.random() < 0.5:
            tri *= random.uniform(0.9, 1.1)
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            import torch.nn.functional as _F
            t = _F.interpolate(t[None], size=(res, res), mode='bilinear',
                               align_corners=False)[0]
        wins[j] = ((t - MEAN) / STD).numpy()
    return torch.from_numpy(wins)

EPOCHS = int(os.environ.get('EPOCHS', '4'))
K_WIN = 12
MICRO = int(os.environ.get('MICRO', '12'))
ACC_STUDIES = 8          # optimizer step tiap 8 studi (grad-accum)
LR_HEAD, LR_BB, WD = 1e-3, 2e-5, 0.05
WARMUP = 300

wp = KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')
model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, DEV)
model.backbone.requires_grad_(True)   # FINE-TUNE PENUH (beda dgn eksperimen lama)
head_params = list(model.norm.parameters()) + list(model.att.parameters()) + \\
              [model.clsW, model.clsb]
bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{'params': bb_params, 'lr': LR_BB},
                         {'params': head_params, 'lr': LR_HEAD}], weight_decay=WD)
total_steps = max(1, (len(train_ids) * EPOCHS) // ACC_STUDIES)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
scaler = torch.amp.GradScaler('cuda')
tlog(f'model siap res={RES} | steps={total_steps} | FULL FT bb lr={LR_BB}')

y_lab = FL.loc[train_ids, LAB].fillna(0.28).values.astype(np.float32)
c_lab = np.stack([FL.loc[train_ids, CONF_COLS[t]].fillna(0.15).values.astype(np.float32)
                  for t in LAB], 1)
W_CONF = 0.5 + np.clip(c_lab, 0, 1)   # bobot 0.5..1.5

step = 0
for ep in range(EPOCHS):
    order = np.random.permutation(len(train_ids))
    model.train()
    opt.zero_grad(set_to_none=True)
    ep_loss, nacc = 0.0, 0
    for ii, si in enumerate(order):
        sid = train_ids[si]
        try:
            vol, mask = load_study(sid)
            xw = rand_windows(vol, mask, K_WIN, RES)
            yt = torch.from_numpy(y_lab[si])[None].repeat(xw.shape[0], 1)
            wt = torch.from_numpy(W_CONF[si])[None].repeat(xw.shape[0], 1)
            for mb0 in range(0, xw.shape[0], MICRO):
                xb = xw[mb0:mb0 + MICRO].to(DEV)
                yb = yt[mb0:mb0 + MICRO].to(DEV)
                wb = wt[mb0:mb0 + MICRO].to(DEV)
                with torch.autocast('cuda', dtype=torch.float16):
                    p = torch.sigmoid(model(xb).float())
                    loss = (F.binary_cross_entropy(p, yb, reduction='none') * wb).mean()
                scaler.scale(loss / max(1, xw.shape[0] // MICRO)).backward()
                ep_loss += float(loss.detach())
            del vol, mask, xw
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tlog(f'OOM di {sid[:16]} -- skip studi ini')
            continue
        nacc += 1
        if nacc % ACC_STUDIES == 0 or ii == len(order) - 1:
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if (ii + 1) % 400 == 0:
            tlog(f'ep{ep+1} {ii+1}/{len(order)} loss~{ep_loss/(ii+1):.4f} '
                 f'lr_bb={sched.get_last_lr()[0]:.2e}')
    tlog(f'=== EPOCH {ep+1}/{EPOCHS} selesai | loss={ep_loss/max(1,len(order)):.4f}')

# ===== evaluasi gold-58 (model FINAL, tanpa seleksi dr gold) =====
model.eval()
gpreds = []
gids = sorted(GOLD)
reader = KE['_make_reader']()
with torch.no_grad():
    for i, sid in enumerate(gids):
        try:
            vol, mask = load_study(sid)
            xw = KE['eval_windows'](vol, mask, k=KE['K_EVAL'], res=RES, norm=KE['NORM'])
            gpreds.append(KE['infer_probs'](model, xw, DEV))
        except Exception as e:
            gpreds.append(np.full(N_T, 0.5, np.float32))
            tlog(f'eval fail {sid[:20]}: {e}')
P = np.stack(gpreds)
out = pd.DataFrame(P, columns=LAB); out.insert(0, 'StudyInstanceUID', gids)
out.to_csv('/kaggle/working/train_coatnet_v1_gold_preds.csv', index=False)

ytrue = train.set_index('StudyInstanceUID').loc[gids, LAB].values.astype(float)
def auc1(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
aucs = [auc1(ytrue[:, j], P[:, j]) for j in range(N_T)
        if 0 < ytrue[:, j].sum() < len(ytrue)]
tlog(f'MACRO AUC gold-58 (final, full-ft, label flight): {np.mean(aucs):.4f}')

torch.save({'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': RES,
            'model': model.state_dict()},
           '/kaggle/working/coatnet_flight_ft_v1.pt')
tlog('checkpoint disimpan: coatnet_flight_ft_v1.pt')
tlog('TRAINING DONE')
'''

nb_cells = [
    {"cell_type": "code", "source": SETUP.splitlines(keepends=True)},
    {"cell_type": "code", "source": [
        "_KE_SRC = " + repr(ke_src) + "\n",
        "_KE_NS = {'__name__': '_ke'}\n",
        "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n",
        "print('_KE loaded')"]},
    {"cell_type": "code", "source": CACHE.splitlines(keepends=True)},
    {"cell_type": "code", "source": TRAIN.splitlines(keepends=True)},
]
nb = {"cells": [{**c, "metadata": {}, "outputs": [], "execution_count": None}
                for c in nb_cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

OUT.mkdir(exist_ok=True)
(OUT / "train_coatnet_v1.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "dimaspashaakrilian/rsna-knee-train-coatnet-flight-v1",
    "title": "RSNA Knee - Train CoatNet flight ft v1",
    "code_file": "train_coatnet_v1.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "flight0234/rsna-knee-hybrid-report-labels",
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-widedense",
        "mattiaangeli/knee-mri-fold-weights",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
