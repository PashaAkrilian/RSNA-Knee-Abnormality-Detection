#!/usr/bin/env python3
"""Build rsna-knee-train-coatnet-v4-weakfocus: lanjutan v3, menargetkan 3 label yang
DIAGNOSISNYA menunjukkan model masih tertinggal dari label gurunya sendiri.

DASAR (RESEARCH_NOTES §29f, diagnosis guru-vs-murid di 58 gold):
  PF OA            guru 0,9003 vs murid 0,8571  (-0,0431)
  Lateral OA       guru 0,8491 vs murid 0,8124  (-0,0368)
  Lateral Meniscus guru 0,8826 vs murid 0,8472  (-0,0354)
Di 8 label lain model SUDAH MELAMPAUI gurunya (Effusion +0,103, Contusion +0,065, Synovitis
+0,058) -- jadi label bukan lagi pembatas di sana, dan menambah sumber label terbukti gagal 2x
(§27d, §29c). Sisa headroom terkonsentrasi di 3 label di atas: kalau ketiganya ditarik setara
gurunya saja, itu ~ +0,010 macro AUC.

PERUBAHAN vs v3 (semua terisolasi, arsitektur TIDAK diubah supaya checkpoint tetap drop-in
kompatibel dgn daftar ARMS pipeline publik -- lihat §50):
  1. Bobot loss per-label: 3 label lemah di atas dikali LABEL_BOOST (2,0), sisanya 1,0.
  2. EPOCHS 4 -> 6 (loss v3 masih menurun saat berhenti di epoch 4: 0,5535->0,5144).
  3. Seleksi checkpoint pakai HOLD-OUT weak-label (200 studi non-gold yang TIDAK ikut latih),
     bukan gold-58 -- persis peringatan §27c bahwa seleksi di gold-58 menyesatkan. gold-58
     tetap dipakai HANYA utk pelaporan akhir, sekali, di akhir.
  4. Guard checkpoint: hanya simpan kalau val hold-out membaik (protokol wajib proyek).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "kaggle_kernel_train_coatnet_v4_weakfocus"
# v2 (26 Agustus): sumber asli /tmp/opencode/nb/... sudah hilang (ephemeral, beda sesi) --
# ke_src sekarang disimpan sbg file terpisah yg DI-COMMIT ke repo (bukan /tmp), supaya builder
# ini reproducible lintas sesi. Isinya sama persis (di-extract dari notebook v1 yg sudah pernah
# di-push), bukan sumber baru.
SRC_TXT = REPO / "build_train_coatnet_ke_src.txt"
ke_src = SRC_TXT.read_text()

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
# ===== FASE 2: fine-tune penuh dgn label BLENDED raw-probability-mean flight0234+Claude (confidence-weighted) =====
import numpy as np, pandas as pd, torch, torch.nn.functional as F, gc, math, random

DEV = 'cuda'
LAB = KE['LAB']; N_T = len(LAB)
import glob as _glob
_hits = _glob.glob('/kaggle/input/**/report_labels_blended_rawmean_v1.csv', recursive=True)
tlog('label csv: ' + repr(_hits))
assert _hits, 'blended label csv tidak ketemu di /kaggle/input'
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

EPOCHS = int(os.environ.get('EPOCHS', '6'))
K_WIN = 12  # v2: micro-batching MICRO dihapus (dead-code no-op krn K_WIN==MICRO==12 default,
            # lihat komentar fix di loop bawah) -- 1 forward per studi memakai semua K_WIN window.
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

# --- v4: bobot loss per-label, fokus ke 3 label yg modelnya masih tertinggal dr gurunya (§29f)
WEAK_LABELS = ['PF OA', 'Lateral OA', 'Lateral Meniscus']
LABEL_BOOST = float(os.environ.get('LABEL_BOOST', '2.0'))
LW = np.ones(N_T, np.float32)
for _t in WEAK_LABELS:
    LW[LAB.index(_t)] = LABEL_BOOST
LW = LW / LW.mean()          # normalisasi supaya skala loss total sebanding dgn v3
LW_T = torch.from_numpy(LW).to(DEV)[None]
tlog(f'bobot label: ' + ', '.join(f'{t}={LW[i]:.2f}' for i, t in enumerate(LAB)))

# --- v4: hold-out weak-label utk SELEKSI checkpoint (gold-58 TIDAK dipakai utk seleksi, §27c)
_rs = np.random.RandomState(2026)
_perm = _rs.permutation(len(train_ids))
N_VAL = int(os.environ.get('N_VAL', '200'))
val_ids = [train_ids[i] for i in _perm[:N_VAL]]
train_ids = [train_ids[i] for i in _perm[N_VAL:]]
y_val = FL.loc[val_ids, LAB].fillna(0.28).values.astype(np.float32)
tlog(f'split: {len(train_ids)} latih | {len(val_ids)} hold-out validasi (weak-label)')
# y_lab/W_CONF dihitung ulang utk train_ids yg sudah dikurangi hold-out
y_lab = FL.loc[train_ids, LAB].fillna(0.28).values.astype(np.float32)
c_lab = np.stack([FL.loc[train_ids, CONF_COLS[t]].fillna(0.15).values.astype(np.float32)
                  for t in LAB], 1)
W_CONF = 0.5 + np.clip(c_lab, 0, 1)

def _auc_np(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def eval_holdout():
    model.eval(); preds = []
    with torch.no_grad():
        for sid in val_ids:
            try:
                v, m = load_study(sid)
                xw = KE['eval_windows'](v, m, k=KE['K_EVAL'], res=RES, norm=KE['NORM'])
                preds.append(KE['infer_probs'](model, xw, DEV))
            except Exception:
                preds.append(np.full(N_T, 0.5, np.float32))
    Pv = np.stack(preds)
    yb_ = (y_val > 0.5).astype(int)
    a = [_auc_np(yb_[:, j], Pv[:, j]) for j in range(N_T)]
    model.train()
    return float(np.nanmean(a)), a

best_val = -1.0

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
            xw = rand_windows(vol, mask, K_WIN, RES)              # (K, 3, H, W)
            # BUG v1 (ERROR run 26 Agustus): RaptorClassifier.encode() didesain utk input 5D
            # (B_studi, K_window, 3, H, W) -- lihat cell _KE_SRC: `B, K = x.shape[:2]` lalu
            # `x.flatten(0,1)` menggabung B & K sblm masuk backbone, HARUS 5D. Kode lama
            # langsung slice `xw[mb0:mb0+MICRO]` (4D, tanpa dim batch-studi) lalu panggil
            # `model(xb)` -- encode() salah tafsir MICRO sbg B & channel(3) sbg K, flatten
            # jadi (MICRO*3, H, W) 3D, PyTorch auto-unsqueeze ke (1, MICRO*3, H, W) --
            # persis match traceback: "expected input[1, 36, 384, 384] to have 3 channels,
            # but got 36" (MICRO=12 default -> 12*3=36). Selain itu label direpeat per-window
            # (`yt=...repeat(xw.shape[0],1)`) padahal head SUDAH pool atas semua K window jadi
            # SATU prediksi per studi -- salah konsep, bukan cuma salah shape.
            # FIX: 1 forward per studi, xw di-unsqueeze(0) jadi (1,K,3,H,W) sesuai desain
            # encode(), label/weight TIDAK direpeat (1 nilai per studi, bukan per window).
            # K_WIN==MICRO==12 secara default jadi loop micro-batch lama itu efektif no-op
            # (cuma 1 iterasi) -- dihapus, bukan cuma dipertahankan sbg dead code.
            xb = xw.unsqueeze(0).to(DEV)                             # (1, K, 3, H, W)
            yb = torch.from_numpy(y_lab[si])[None].to(DEV)           # (1, 12)
            wb = torch.from_numpy(W_CONF[si])[None].to(DEV)          # (1, 12)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(xb).float()
                # v3 ERROR (26 Agustus): sigmoid+binary_cross_entropy eksplisit di dalam
                # autocast() DILARANG pytorch ("unsafe to autocast") krn cek-nya berdasar
                # NAMA OP, bukan dtype runtime aktual -- walau logits sudah di-.float() di
                # atas. Fix resmi yg direkomendasikan pesan errornya sendiri: gabung jadi
                # binary_cross_entropy_with_logits (aman di-autocast, scr matematis setara
                # sigmoid+BCE, malah lebih stabil numerik).
                loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb * LW_T).mean()
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
            step += 1
        if (ii + 1) % 400 == 0:
            tlog(f'ep{ep+1} {ii+1}/{len(order)} loss~{ep_loss/(ii+1):.4f} '
                 f'lr_bb={sched.get_last_lr()[0]:.2e}')
    tlog(f'=== EPOCH {ep+1}/{EPOCHS} selesai | loss={ep_loss/max(1,len(order)):.4f}')
    _vm, _va = eval_holdout()
    tlog(f'    hold-out macro={_vm:.4f} | ' +
         ', '.join(f'{t}={_va[i]:.3f}' for i, t in enumerate(LAB) if t in WEAK_LABELS))
    if _vm > best_val:
        best_val = _vm
        torch.save({'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': RES,
                    'model': model.state_dict()},
                   '/kaggle/working/coatnet_v4_weakfocus.pt')
        tlog(f'    checkpoint BARU disimpan (hold-out {best_val:.4f})')
    else:
        tlog(f'    tidak membaik (terbaik {best_val:.4f}) -- checkpoint lama dipertahankan')

# muat kembali checkpoint TERBAIK sebelum evaluasi gold-58 (bukan model epoch terakhir)
_ck = torch.load('/kaggle/working/coatnet_v4_weakfocus.pt', map_location='cpu', weights_only=False)
model.load_state_dict(_ck['model']); model.to(DEV); del _ck
tlog(f'checkpoint terbaik dimuat ulang (hold-out {best_val:.4f}) utk evaluasi gold-58')

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
out.to_csv('/kaggle/working/train_coatnet_v4_weakfocus_gold_preds.csv', index=False)

ytrue = train.set_index('StudyInstanceUID').loc[gids, LAB].values.astype(float)
def auc1(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
aucs = [auc1(ytrue[:, j], P[:, j]) for j in range(N_T)
        if 0 < ytrue[:, j].sum() < len(ytrue)]
tlog(f'MACRO AUC gold-58 (v4 weak-focus, checkpoint terbaik by hold-out): {np.mean(aucs):.4f}')
tlog('pembanding: v3=0.9234 (LB solo 0.932), c5=0.9206 (LB solo 0.927), v1=0.9193, v2=0.9170')
for _j, _t in enumerate(LAB):
    tlog(f'  {_t}: {auc1(ytrue[:, _j], P[:, _j]):.4f}')

tlog('checkpoint final: coatnet_v4_weakfocus.pt (sudah disimpan via guard hold-out)')
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
(OUT / "train_coatnet_v4_weakfocus.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "dimaspashaakrilian/rsna-knee-train-coatnet-v4-weakfocus",
    "title": "RSNA Knee - Train CoatNet v4 weakfocus",
    "code_file": "train_coatnet_v4_weakfocus.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dimaspashaakrilian/rsna-knee-blended-report-labels-rawmean-v1",
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-widedense",
        "mattiaangeli/knee-mri-fold-weights",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
