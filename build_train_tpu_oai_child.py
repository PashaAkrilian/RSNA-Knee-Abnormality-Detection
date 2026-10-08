#!/usr/bin/env python3
"""Fine-tune "anak" dari checkpoint OAI publik (nartaa r384, LB 0,949) di TPU, pada label fusi r1 kita.

Latar (RESEARCH_NOTES §36f/§37): tambahan model berbasis data kompetisi saja TIDAK menambah di atas 0,949;
satu-satunya keunggulan kita yg terbukti adalah KUALITAS LABEL (fusi teacher+OOF murid: gold-58 0,9348 vs label
LLM ~0,90; fine-tune c5 pada label itu: LB solo 0,927 -> 0,933). Ide: pakai bobot publik (pre-trained model, diizinkan
Rules) sbg inisialisasi, fine-tune LEMBUT pada label fusi + regularisasi KD ke induk (LwF) supaya pengetahuan OAI
tidak lupa, lalu ensembel induk+anak di notebook inferensi.

Geometri: cache `z` (build_volcache_cpu2.py z 384 SLOTS96 0.02 0.98) = persis korpus ckpt (96 irisan/384 px/2-98%).
Varian dijalankan berurutan dlm satu sesi TPU: VARIANTS = [(tag, lambda_KD, seed)].
Keluaran: oai_child_<tag>_final.pt / _swa.pt (format {'arch','res','model'} spt ckpt nartaa) + parent_soft_k16.csv
(prediksi induk in-sample pada semua studi latih; diagnosis memorisasi & bahan fusi label r2).
Pemakaian: python3 build_train_tpu_oai_child.py
"""
import json
import re
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "kaggle_kernel_train_tpu_oai_child"
SLUG = "rsna-knee-train-tpu-oai-child"
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
tlog('setup OK')
'''

TRAIN = '''\
import numpy as np, pandas as pd, os, json, glob as _glob, copy, math, random, contextlib, gc
import torch, torch.nn as nn, torch.nn.functional as F
import torch_xla.core.xla_model as xm
from concurrent.futures import ThreadPoolExecutor
KE = _KE_NS
SCALER = None
DEV = xm.xla_device()
LAB = KE['LAB']; N_T = len(LAB)
ROOT = KE['find_test_root']()
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv'); tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
SERIDS = set(tser['StudyInstanceUID'])
GOLD = set(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in SERIDS]

def _bglob(name):
    return [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/')
            for p in _glob.glob(d + name)]
MOUNTED = {}
for _d in (_glob.glob('/kaggle/input/*/volcache') + _glob.glob('/kaggle/input/*/*/volcache')
           + _glob.glob('/kaggle/input/*/*/*/volcache')):          # kedalaman terbatas: JANGAN '**' (820rb berkas kompetisi)
    if os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.endswith('.npz'):
                MOUNTED[_f[:-4]] = os.path.join(_d, _f)
tlog(f'cache z termount: {len(MOUNTED)} studi | gold={len(GOLD)} | studi={len(ALL_IDS)}')
assert len(MOUNTED) > 4000, 'cache z tidak lengkap'

_hits = _bglob('fused_labels_r1.csv'); assert _hits, 'fused_labels_r1.csv tidak ketemu'
FL = pd.read_csv(_hits[0]); FL['StudyInstanceUID'] = FL['StudyInstanceUID'].astype(str); FL = FL.set_index('StudyInstanceUID')
CONF_COLS = {t: f'{t}__conf' for t in LAB}
train_ids = [s for s in ALL_IDS if s not in GOLD and s in FL.index and s in MOUNTED]
tlog(f'studi latih (non-gold, berlabel fusi, ter-cache): {len(train_ids)}')
y_lab = FL.loc[train_ids, LAB].fillna(0.28).values.astype(np.float32)
c_lab = np.stack([FL.loc[train_ids, CONF_COLS[t]].fillna(0.15).values.astype(np.float32) for t in LAB], 1)
W_CONF = (0.5 + np.clip(c_lab, 0, 1)).astype(np.float32)

PARENT = [p for p in _bglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')]; assert PARENT, 'ckpt induk tidak ketemu'
PARENT = PARENT[0]; tlog('induk: ' + PARENT)

def load_study(sid):
    z = np.load(MOUNTED[sid]); return z['vol'], z['mask']

MEAN = KE['_MEAN']; STD = KE['_STD']
RES = 384
def rand_windows(vol, mask, k):
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3: valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(max(1, lo), min(D - 1, hi))]
    if not cs: cs = [max(1, min(D - 2, (lo + hi) // 2))]
    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))
    while len(cs) < k: cs.append(cs[len(cs) % len(cs)])               # bentuk TETAP utk XLA
    tri = np.stack([vol[np.array([c - 1, c, c + 1])] for c in cs], 0).astype(np.float32) / 255.0   # (k,3,H,W)
    for j in range(k):
        if random.random() < 0.5: tri[j] *= random.uniform(0.9, 1.1)
    t = torch.from_numpy(tri)
    return ((t - MEAN) / STD).contiguous()

def build_model():
    ck = torch.load(PARENT, map_location='cpu', weights_only=False)
    arch = ck.get('arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')
    bb = KE['build_backbone'](arch, pretrained=False)
    m = KE['RaptorClassifier'](bb, F_dim=bb.num_features)
    m.load_state_dict(ck['model'], strict=True)
    del ck; gc.collect()
    return m, arch

try:
    with torch.autocast('xla', dtype=torch.bfloat16):
        _t = torch.ones(8, 8, device=DEV) @ torch.ones(8, 8, device=DEV)
    xm.mark_step(); float(_t.sum())
    AC = lambda: torch.autocast('xla', dtype=torch.bfloat16)
    tlog('autocast xla bf16: OK')
except Exception as _e:
    AC = lambda: contextlib.nullcontext()
    tlog(f'autocast xla GAGAL ({type(_e).__name__}: {_e}) -> fp32')

def bn_eval(m):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)): mod.eval()

_CHUNK_OK = {}
@torch.no_grad()
def predict(m, xw, chunk):
    """xw (K,3,H,W) -> probabilitas (12,). Backbone per potongan (bentuk tetap utk XLA), kepala atas SEMUA window.
    Bila potongan besar gagal (mis. OOM HBM) -> turun ke potongan 16 (sisa 14 dst) dan ingat pilihan itu."""
    chunk = _CHUNK_OK.get(chunk, chunk)
    for attempt in (chunk, 16):
        try:
            fs = []
            for c in xw.split(attempt):
                with AC(): fs.append(m.backbone(c.to(DEV)).float())
                xm.mark_step()
            feats = torch.cat(fs, 0).unsqueeze(0)
            with AC(): lg = m.head(feats).float()
            xm.mark_step()
            out = torch.sigmoid(lg)[0].cpu().numpy()
            _CHUNK_OK[chunk] = attempt
            return out
        except Exception as e:
            tlog(f'predict gagal (chunk={attempt}): {type(e).__name__}: {str(e)[:160]}')
            if attempt == 16: raise
    raise RuntimeError('predict gagal')

def auc1(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
gids = sorted(GOLD); gids = [g for g in gids if g in MOUNTED]
YG = train.set_index('StudyInstanceUID').loc[gids, LAB].values.astype(float)
def eval_gold(m, tag):
    m.eval(); P = []
    for sid in gids:
        vol, mask = load_study(sid)
        xw = KE['eval_windows'](vol, mask, k=94, res=RES, norm=KE['NORM'])     # (94,3,384,384) fp32
        P.append(predict(m, xw, 47))
    P = np.stack(P)
    a = [auc1(YG[:, j], P[:, j]) for j in range(N_T) if 0 < YG[:, j].sum() < len(YG)]
    tlog(f'[gold-58 | {tag}] macro AUC={np.mean(a):.4f} (TERKONTAMINASI: induk dilatih pada gold; hanya utk memantau drift)')
    return P, float(np.mean(a))

K_WIN = int(os.environ.get('K_WIN', '12'))
EPOCHS = int(os.environ.get('EPOCHS', '3'))
ACC_STUDIES = 8
LR_BB, LR_HEAD, WD, WARMUP, LRQ = 1e-5, 1e-4, 0.01, 100, 30
VARIANTS = json.loads(os.environ.get('VARIANTS', '[["a", 0.5, 1], ["b", 0.0, 2]]'))   # (tag, lambda_KD, seed)

# --- basis: induk di gold-58 ---
parent, ARCH = build_model(); parent.to(DEV).eval()
for p_ in parent.parameters(): p_.requires_grad_(False)
P0, A0 = eval_gold(parent, 'induk')

from collections import deque
RESUME_DIR = os.environ.get('RESUME_DIR')                 # mis. /content/kw (Colab); kosong = tanpa resume (perilaku TPU/Kaggle tak berubah)
CKPT_EVERY = int(os.environ.get('CKPT_EVERY', '400'))     # simpan status lanjut tiap >=400 studi (di batas langkah optimizer)

def _atomic_save(obj, path):
    tmp = path + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)

def _rng_state():
    st = {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available(): st['cuda'] = torch.cuda.get_rng_state_all()
    return st

def _rng_restore(st):
    random.setstate(st['py']); np.random.set_state(st['np']); torch.set_rng_state(st['torch'])
    if 'cuda' in st and torch.cuda.is_available():
        try: torch.cuda.set_rng_state_all(st['cuda'])
        except Exception: pass

def run_child(tag, lam, seed):
    tlog(f'===== ANAK {tag}: lambda_KD={lam} seed={seed} epochs={EPOCHS} =====')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model, _ = build_model(); model.to(DEV)
    model.backbone.requires_grad_(True)
    head_params = list(model.norm.parameters()) + list(model.att.parameters()) + [model.clsW, model.clsb]
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': bb_params, 'lr': LR_BB}, {'params': head_params, 'lr': LR_HEAD}], weight_decay=WD)
    total_steps = max(1, (len(train_ids) * EPOCHS) // ACC_STUDIES)
    def lr_scale(s):
        sc = min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
        return max(1, round(sc * LRQ)) / LRQ
    def set_lr(s):
        sc = lr_scale(s); opt.param_groups[0]['lr'] = LR_BB * sc; opt.param_groups[1]['lr'] = LR_HEAD * sc
    done_flag = f'{RESUME_DIR}/done_{tag}.flag' if RESUME_DIR else None
    if done_flag and os.path.exists(done_flag):
        tlog(f'anak {tag} sudah selesai (flag) -> dilewati'); return
    rp = f'{RESUME_DIR}/resume_{tag}.pt' if RESUME_DIR else None
    swa, nswa = None, 0
    start_ep, ii0, order0, step = 0, 0, None, 0
    ep_l0, ep_kd0, nacc0 = 0.0, 0.0, 0
    if rp and os.path.exists(rp):
        R = torch.load(rp, map_location='cpu', weights_only=False)
        model.load_state_dict(R['model']); opt.load_state_dict(R['opt'])
        try:
            if R.get('scaler') is not None: SCALER.load_state_dict(R['scaler'])
        except Exception as e_:
            tlog(f'scaler tak dimuat: {type(e_).__name__}')
        start_ep, ii0, order0, step = int(R['ep']), int(R['ii']), R['order'], int(R['step'])
        swa, nswa = R['swa'], int(R['nswa']); ep_l0, ep_kd0, nacc0 = float(R['ep_l']), float(R['ep_kd']), int(R['nacc'])
        _rng_restore(R['rng']); del R; gc.collect()
        tlog(f'RESUME {tag}: epoch {start_ep+1}, studi {ii0}, langkah optimizer {step}')
    ex = ThreadPoolExecutor(3)
    def prep(si):
        sid = train_ids[si]
        vol, mask = load_study(sid)
        return si, rand_windows(vol, mask, K_WIN)
    for ep in range(start_ep, EPOCHS):
        model.train(); bn_eval(model)
        opt.zero_grad(set_to_none=True)
        if ep == start_ep and order0 is not None:
            order = np.asarray(order0); i0 = ii0; ep_l, ep_kd, nacc = ep_l0, ep_kd0, nacc0
        else:
            order = np.random.permutation(len(train_ids)); i0 = 0; ep_l, ep_kd, nacc = 0.0, 0.0, 0
        pq = deque(ex.submit(prep, int(order[j])) for j in range(i0, min(i0 + 6, len(order)))); nxt = i0 + 6
        last_ck = i0
        for ii in range(i0, len(order)):
            try:
                f_ = pq.popleft()
                if nxt < len(order): pq.append(ex.submit(prep, int(order[nxt]))); nxt += 1   # isi ulang SEBELUM result() agar antrean tak rusak bila gagal
                si, xw = f_.result()
                xb = xw.unsqueeze(0).to(DEV)
                yb = torch.from_numpy(y_lab[si])[None].to(DEV); wb = torch.from_numpy(W_CONF[si])[None].to(DEV)
                with AC(): lg = model(xb).float()
                loss_y = (F.binary_cross_entropy_with_logits(lg, yb, reduction='none') * wb).mean()
                loss = loss_y
                if lam > 0:
                    with torch.no_grad(), AC(): tp = torch.sigmoid(parent(xb).float())
                    loss_kd = F.binary_cross_entropy_with_logits(lg, tp, reduction='mean')
                    loss = loss_y + lam * loss_kd
                    ep_kd += float(loss_kd.detach())
                (loss / 1.0).backward()
                xm.mark_step()
                ep_l += float(loss_y.detach())
            except Exception as e:
                tlog(f'GAGAL studi idx {ii}: {type(e).__name__}: {str(e)[:140]} -- dilewati'); xm.mark_step(); continue
            nacc += 1
            if nacc % ACC_STUDIES == 0 or ii == len(order) - 1:
                for p_ in (bb_params + head_params):
                    if p_.grad is not None: p_.grad.div_(ACC_STUDIES)
                set_lr(step); opt.step(); xm.mark_step(); opt.zero_grad(set_to_none=True); step += 1
            if RESUME_DIR and (nacc % ACC_STUDIES == 0) and (ii + 1 - last_ck) >= CKPT_EVERY and ii + 1 < len(order):
                try:
                    _atomic_save({'tag': tag, 'ep': ep, 'ii': ii + 1, 'order': order, 'step': step, 'nacc': nacc, 'ep_l': ep_l, 'ep_kd': ep_kd,
                                  'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}, 'opt': opt.state_dict(),
                                  'scaler': (SCALER.state_dict() if SCALER is not None else None), 'swa': swa, 'nswa': nswa, 'rng': _rng_state()}, rp)
                    last_ck = ii + 1
                except Exception as e_:
                    tlog(f'simpan status lanjut gagal: {type(e_).__name__}: {str(e_)[:100]}')
            if (ii + 1) % 400 == 0:
                tlog(f'{tag} ep{ep+1} {ii+1}/{len(order)} loss_y~{ep_l/(ii+1):.4f} kd~{ep_kd/(ii+1):.4f} lr_bb={LR_BB*lr_scale(step):.2e}')
        tlog(f'=== {tag} EPOCH {ep+1}/{EPOCHS} | loss_y={ep_l/max(1,len(order)):.4f} kd={ep_kd/max(1,len(order)):.4f}')
        sd = {k_: v_.detach().cpu().float() for k_, v_ in model.state_dict().items()}
        try:
            torch.save({'arch': ARCH, 'res': 384, 'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}},
                       f'/kaggle/working/oai_child_{tag}_ep{ep+1}.pt')
        except Exception as e:
            tlog(f'simpan ep{ep+1} gagal: {type(e).__name__}: {e}')
        if ep >= 1:     # SWA atas akhir epoch 2..E
            swa = sd if swa is None else {k_: swa[k_] + sd[k_] for k_ in swa}; nswa += 1
        if RESUME_DIR:
            try:
                _atomic_save({'tag': tag, 'ep': ep + 1, 'ii': 0, 'order': None, 'step': step, 'nacc': 0, 'ep_l': 0.0, 'ep_kd': 0.0,
                              'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}, 'opt': opt.state_dict(),
                              'scaler': (SCALER.state_dict() if SCALER is not None else None), 'swa': swa, 'nswa': nswa, 'rng': _rng_state()}, rp)
            except Exception as e_:
                tlog(f'simpan status batas-epoch gagal: {type(e_).__name__}: {str(e_)[:100]}')
        if ep < EPOCHS - 1:
            _, a_ = eval_gold(model, f'{tag} ep{ep+1}')
    P1, A1 = eval_gold(model, f'{tag} final')
    tlog(f'korelasi prediksi gold anak vs induk (rata2 per label): '
         f'{np.mean([np.corrcoef(P0[:, j], P1[:, j])[0, 1] for j in range(N_T)]):.4f}')
    torch.save({'arch': ARCH, 'res': 384, 'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}},
               f'/kaggle/working/oai_child_{tag}_final.pt')
    try:
      if swa is not None and nswa > 0:
        sdm = {k_: (v_ / nswa) for k_, v_ in swa.items()}
        # buffer integer (num_batches_tracked) tidak boleh float
        ref = model.state_dict()
        for k_ in sdm:
            if not ref[k_].is_floating_point(): sdm[k_] = ref[k_].detach().cpu()
            else: sdm[k_] = sdm[k_].to(ref[k_].dtype)
        model.load_state_dict(sdm, strict=True)
        P2, A2 = eval_gold(model, f'{tag} SWA(ep2..{EPOCHS})')
        torch.save({'arch': ARCH, 'res': 384, 'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}},
                   f'/kaggle/working/oai_child_{tag}_swa.pt')
    except Exception as e:
        tlog(f'SWA {tag} gagal: {type(e).__name__}: {e}')
    if done_flag:
        open(done_flag, 'w').write('ok')
        try: os.remove(rp)
        except Exception: pass
    tlog(f'anak {tag} disimpan | gold(induk)={A0:.4f} final={A1:.4f}')
    ex.shutdown(wait=False)
    del model, opt; gc.collect()

for tag, lam, seed in VARIANTS:
    try:
        run_child(tag, float(lam), int(seed))
    except Exception as e:
        import traceback; tlog(f'ANAK {tag} GAGAL: {type(e).__name__}: {e}'); tlog(traceback.format_exc()[-1500:])
        gc.collect()

# --- prediksi induk in-sample (K=16 window merata, 1 tampilan) utk diagnosis memorisasi & fusi label r2 ---
tlog('prediksi induk pada semua studi latih ...')
import time as _tm
t1 = _tm.time(); rows = []
allids = [s for s in ALL_IDS if s in MOUNTED]
for i, sid in enumerate(allids):
    try:
        vol, mask = load_study(sid)
        xw = KE['eval_windows'](vol, mask, k=16, res=RES, norm=KE['NORM'])
        rows.append([sid] + [float(x) for x in predict(parent, xw, 16)])
    except Exception as e:
        tlog(f'gagal {sid[:16]}: {type(e).__name__}')
    if (i + 1) % 500 == 0: tlog(f'induk-soft {i+1}/{len(allids)} ({_tm.time()-t1:.0f}s)')
    if _tm.time() - t1 > 2700: tlog('batas waktu 45 mnt, berhenti'); break
pd.DataFrame(rows, columns=['StudyInstanceUID'] + LAB).to_csv('/kaggle/working/parent_soft_k16.csv', index=False)
tlog('TRAINING DONE')
'''

nb_cells = [
    {"cell_type": "code", "source": SETUP.splitlines(keepends=True)},
    {"cell_type": "code", "source": [
        "_KE_SRC = " + repr(ke_src) + "\n",
        "_KE_NS = {'__name__': '_ke'}\n",
        "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n",
        "print('_KE loaded')"]},
    {"cell_type": "code", "source": TRAIN.splitlines(keepends=True)},
]
nb = {"cells": [{**c, "metadata": {}, "outputs": [], "execution_count": None} for c in nb_cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": f"dimaspashaakrilian/{SLUG}",
    "title": "RSNA Knee - Train TPU OAI child",
    "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": True, "enable_internet": False,
    "dataset_sources": ["dimaspashaakrilian/rsna-knee-fused-labels-r1",
                        "nartaa/rsna-knee-publication-swa-weights-20261007"],
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-z{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
# sanity: sintaks sel TRAIN
compile(TRAIN, "<train>", "exec")
print("built:", OUT.name)
