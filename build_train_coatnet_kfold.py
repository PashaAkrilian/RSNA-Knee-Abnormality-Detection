#!/usr/bin/env python3
"""Build rsna-knee-kfold-coatnet-f{K}: SATU fold dari K-fold CoAtNet penghasil prediksi OUT-OF-FOLD
(OOF) untuk SEMUA studi lemah + 58 gold -- bahan baku NOISY-STUDENT label refinement.

MENGAPA (RESEARCH_NOTES §32): label laporan (guru, 0,9033 vs gold) dan prediksi model-gambar
(murid, v3 = 0,9234) punya kesalahan yang saling independen. Memfusikannya (rank-mean w=0,5-0,6)
menghasilkan label berkualitas 0,9317-0,9342 vs gold -- LEBIH BAIK dari keduanya (+0,03 di atas
guru). Melatih ulang model di atas label fusi = 1 putaran noisy-student. Syaratnya: prediksi murid
untuk tiap studi lemah HARUS dari model yang TIDAK melihat studi itu (OOF), kalau tidak cuma
menggaungkan label guru. Makanya K-fold.

DESAIN
- Fold sadar-GRUP: studi yang laporannya byte-identik (194 studi/47 grup, §27c) selalu satu fold,
  supaya kembaran laporan tidak bocor antar latih/uji. Penugasan deterministik (greedy, grup
  terbesar dulu) -> IDENTIK di semua kernel fold tanpa perlu berbagi berkas.
- gold-58 TIDAK PERNAH ikut latih di fold manapun -> prediksi gold dari tiap fold bersih; tiap
  kernel menuliskannya, nanti dirata-rata lintas fold.
- INIT: 'c5' (checkpoint MRI-pretrained publik, murid kuat tapi TELAH melihat studi lemah dgn
  label mirip guru -> berpotensi "mengaungkan" guru) atau 'imagenet' (bersih, lebih lemah).
  PROBE kontaminasi: sebelum latih, model inisial memprediksi 150 studi lemah held-out + gold;
  dibandingkan korelasinya dgn guru pada studi yg "dilihat" vs gold yg tidak pernah dilihat.
- Resep latih SAMA dgn v3 (bukti +0,005 di LB nyata): FT penuh, LR_HEAD 1e-3 / LR_BB 2e-5,
  confidence-weighted BCE, label raw-probability-mean flight0234+Claude. Hanya beda: 3 epoch
  (data 75%), WARMUP disesuaikan.

Pemakaian:  python3 build_train_coatnet_kfold.py <fold> [nfold=4] [init=c5|imagenet] [epochs=3]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NFOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 4
INIT = sys.argv[3] if len(sys.argv) > 3 else "c5"
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
assert 0 <= FOLD < NFOLD and INIT in ("c5", "imagenet")
TAG = f"{INIT}-f{FOLD}of{NFOLD}"
SLUG = f"rsna-knee-kfold-{INIT}-f{FOLD}"
OUT = REPO / f"kaggle_kernel_kfold_{INIT}_f{FOLD}"

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

SETUP = '''\
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
os.environ['FOLD'] = '__FOLD__'
os.environ['NFOLD'] = '__NFOLD__'
os.environ['INIT'] = '__INIT__'
os.environ['EPOCHS'] = '__EPOCHS__'
tlog('setup OK | fold __FOLD__/__NFOLD__ init=__INIT__ epochs=__EPOCHS__')
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
import glob as _g
MOUNTED = {}
for _d in (_g.glob('/kaggle/input/*/volcache') + _g.glob('/kaggle/input/*/*/volcache')
           + _g.glob('/kaggle/input/*/*/*/volcache')):   # kedalaman terbatas: JANGAN '**' (820rb berkas kompetisi)
    if os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.endswith('.npz'):
                MOUNTED[_f[:-4]] = os.path.join(_d, _f)
tlog(f'cache termount dari kernel CPU: {len(MOUNTED)} studi (sisanya di-decode di sini)')
reader = KE['_make_reader']()

def cache_one(sid):
    fn = f'{CDIR}/{sid}.npz'
    if sid in MOUNTED or os.path.exists(fn):
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
# ===== FASE 2: K-fold CoAtNet -> OOF utk noisy-student (RESEARCH_NOTES §32) =====
import numpy as np, pandas as pd, torch, torch.nn.functional as F, gc, math, random, hashlib, os
import glob as _glob

DEV = 'cuda'
LAB = KE['LAB']; N_T = len(LAB)
FOLD = int(os.environ['FOLD']); NFOLD = int(os.environ['NFOLD'])
EPOCHS = int(os.environ['EPOCHS']); INIT = os.environ['INIT']
K_EVAL_OOF = 32

def _bglob(name):
    return [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/')
            for p in _glob.glob(d + name)]
_hits = _bglob('report_labels_blended_rawmean_v1.csv')
tlog('label csv: ' + repr(_hits))
assert _hits, 'blended label csv tidak ketemu di /kaggle/input'
fl = pd.read_csv(_hits[0])
fl['StudyInstanceUID'] = fl['StudyInstanceUID'].astype(str)
FL = fl.set_index('StudyInstanceUID')
CONF_COLS = {t: f'{t}__conf' for t in LAB}

# ---- fold sadar-GRUP (laporan byte-identik = satu grup), deterministik ----
weak_ids = [s for s in ALL_IDS if s not in GOLD and s in FL.index]
rep = train.set_index('StudyInstanceUID')['Report'] if 'Report' in train.columns else None
def gkey(s):
    r = None if rep is None else rep.get(s)
    if r is None or isinstance(r, float) or not str(r).strip():
        return 'u:' + s
    return 'r:' + hashlib.md5(' '.join(str(r).split()).encode()).hexdigest()
groups = {}
for s in weak_ids:
    groups.setdefault(gkey(s), []).append(s)
glist = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
sizes = [0] * NFOLD; fold_of = {}
for gk, members in glist:
    f = int(np.argmin(sizes))
    for m in members:
        fold_of[m] = f
    sizes[f] += len(members)
tlog(f'grup={len(groups)} studi_lemah={len(weak_ids)} ukuran_fold={sizes}')
train_ids = [s for s in weak_ids if fold_of[s] != FOLD]
held_ids = [s for s in weak_ids if fold_of[s] == FOLD]
gold_ids = sorted(GOLD)
tlog(f'fold {FOLD}: latih={len(train_ids)} held-out={len(held_ids)} gold={len(gold_ids)}')
# keamanan: tak boleh ada laporan kembar lintas latih/held-out
_tg = {gkey(s) for s in train_ids}; _hg = {gkey(s) for s in held_ids}
assert not (_tg & _hg), 'BOCOR: ada grup laporan yang sama di latih & held-out'

def load_study(sid):
    fn = f'/kaggle/tmp/volcache/{sid}.npz'
    if not os.path.exists(fn):
        fn = MOUNTED[sid]
    z = np.load(fn)
    return z['vol'], z['mask']

def rand_windows(vol, mask, k, res):
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(max(1, lo), min(D - 1, hi))]
    if len(cs) == 0: cs = [max(1, min(D - 2, (lo + hi) // 2))]
    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))
    wins = np.empty((len(cs), 3, res, res), np.float32)
    MEAN = KE['_MEAN']; STD = KE['_STD']
    for j, c in enumerate(cs):
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        if random.random() < 0.5:
            tri *= random.uniform(0.9, 1.1)
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            import torch.nn.functional as _F
            t = _F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        wins[j] = ((t - MEAN) / STD).numpy()
    return torch.from_numpy(wins)

K_WIN = 12
ACC_STUDIES = 8
LR_HEAD, LR_BB, WD = 1e-3, 2e-5, 0.05
WARMUP = 150

if INIT == 'c5':
    wp = KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')
    model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, DEV)
else:
    raise SystemExit('INIT=imagenet belum diimplementasikan di builder ini')

def predict_ids(ids, k=K_EVAL_OOF):
    model.eval(); out = []
    with torch.no_grad():
        for sid in ids:
            try:
                v, m = load_study(sid)
                xw = KE['eval_windows'](v, m, k=k, res=RES, norm=KE['NORM'])
                out.append(KE['infer_probs'](model, xw, DEV))
            except Exception as e:
                out.append(np.full(N_T, 0.5, np.float32))
    return np.stack(out)

def corr_rows(P, ids, tag):
    """korelasi Pearson rata-rata (lintas 12 label) antara prediksi dan label guru."""
    T = FL.loc[ids, LAB].fillna(0.28).values.astype(float)
    cs = [np.corrcoef(P[:, j], T[:, j])[0, 1] for j in range(N_T)]
    tlog(f'   korelasi(pred, guru) [{tag}] rata2={np.nanmean(cs):.4f}')
    return float(np.nanmean(cs))

# ---- PROBE kontaminasi: model inisial pada studi lemah held-out vs gold (tak pernah dilihat) ----
rs = np.random.RandomState(7)
probe_ids = [held_ids[i] for i in rs.permutation(len(held_ids))[:150]]
Pp = predict_ids(probe_ids)
Pg0 = predict_ids(gold_ids)
pd.DataFrame(Pp, columns=LAB).assign(StudyInstanceUID=probe_ids).to_csv(
    f'/kaggle/working/probe_init_weak_f{FOLD}.csv', index=False)
pd.DataFrame(Pg0, columns=LAB).assign(StudyInstanceUID=gold_ids).to_csv(
    f'/kaggle/working/probe_init_gold_f{FOLD}.csv', index=False)
tlog('PROBE (model inisial, sebelum latih):')
c_seen = corr_rows(Pp, probe_ids, 'studi lemah = DILIHAT c5 saat pra-latih')
# guru utk gold: pakai label blend Claude-pilot+flight tidak tersedia di kernel -> pakai label
# gold asli sbg acuan alternatif utk perbandingan korelasi terhadap KEBENARAN
Yg = train.set_index('StudyInstanceUID').loc[gold_ids, LAB].values.astype(float)
cg = [np.corrcoef(Pg0[:, j], Yg[:, j])[0, 1] for j in range(N_T)]
tlog(f'   korelasi(pred, GOLD asli) [gold = TAK pernah dilihat] rata2={np.nanmean(cg):.4f}')

model.backbone.requires_grad_(True)
head_params = list(model.norm.parameters()) + list(model.att.parameters()) + [model.clsW, model.clsb]
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
W_CONF = 0.5 + np.clip(c_lab, 0, 1)

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
            xb = xw.unsqueeze(0).to(DEV)
            yb = torch.from_numpy(y_lab[si])[None].to(DEV)
            wb = torch.from_numpy(W_CONF[si])[None].to(DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(xb).float()
                loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb).mean()
            scaler.scale(loss).backward()
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
        if (ii + 1) % 400 == 0:
            tlog(f'ep{ep+1} {ii+1}/{len(order)} loss~{ep_loss/(ii+1):.4f} '
                 f'lr_bb={sched.get_last_lr()[0]:.2e}')
    tlog(f'=== EPOCH {ep+1}/{EPOCHS} selesai | loss={ep_loss/max(1,len(order)):.4f}')

torch.save({'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': RES,
            'model': model.state_dict()}, f'/kaggle/working/kfold_{INIT}_f{FOLD}.pt')
tlog('checkpoint fold disimpan')

# ---- OOF: prediksi studi held-out (TAK dilihat model ini) + gold-58 ----
Ph = predict_ids(held_ids)
Pg = predict_ids(gold_ids)
oof = pd.DataFrame(np.vstack([Ph, Pg]), columns=LAB)
oof.insert(0, 'StudyInstanceUID', held_ids + gold_ids)
oof['is_gold'] = [0] * len(held_ids) + [1] * len(gold_ids)
oof['fold'] = [FOLD] * len(held_ids) + [-1] * len(gold_ids)
oof.to_csv(f'/kaggle/working/oof_{INIT}_f{FOLD}.csv', index=False)
tlog(f'OOF ditulis: {len(held_ids)} held-out + {len(gold_ids)} gold')

def auc1(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
ytrue = train.set_index('StudyInstanceUID').loc[gold_ids, LAB].values.astype(float)
aucs = [auc1(ytrue[:, j], Pg[:, j]) for j in range(N_T)]
tlog(f'MACRO AUC gold-58 (fold {FOLD}, {INIT}-init): {np.nanmean(aucs):.4f}  '
     f'[acuan v3=0.9234 (LB 0.932), c5=0.9213]')
tlog('PASCA-LATIH:')
corr_rows(Ph, held_ids, 'studi lemah HELD-OUT (tak dilihat fine-tune; c5 pra-latih melihatnya)')
cg2 = [np.corrcoef(Pg[:, j], ytrue[:, j])[0, 1] for j in range(N_T)]
tlog(f'   korelasi(pred, GOLD asli) [gold] rata2={np.nanmean(cg2):.4f}')
tlog('KFOLD DONE')
'''

for a, b in (("__FOLD__", str(FOLD)), ("__NFOLD__", str(NFOLD)),
             ("__INIT__", INIT), ("__EPOCHS__", str(EPOCHS))):
    SETUP = SETUP.replace(a, b)

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
nb = {"cells": [{**c, "metadata": {}, "outputs": [], "execution_count": None} for c in nb_cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

OUT.mkdir(exist_ok=True)
NAME = f"kfold_{INIT}_f{FOLD}.ipynb"
(OUT / NAME).write_text(json.dumps(nb, indent=1))
meta = {
    "id": f"dimaspashaakrilian/{SLUG}",
    "title": f"RSNA Knee - KFold {INIT} f{FOLD}",
    "code_file": NAME,
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dimaspashaakrilian/rsna-knee-blended-report-labels-rawmean-v1",
        "dreaddevelopment/raptor-knee-maxspan",
    ],
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-s{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT, "| tag:", TAG)
