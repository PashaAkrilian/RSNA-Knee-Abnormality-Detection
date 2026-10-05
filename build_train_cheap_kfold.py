#!/usr/bin/env python3
"""MODEL MURAH (CNN kecil, 224px) K-fold di label FUSI -- jalur "pseudo-label + model kecil" ala tim papan atas.

ALASAN (RESEARCH_NOTES §36): diskusi 735304 (4 Okt) -- tim papan atas melaporkan model TUNGGAL kecil beresolusi
rendah mencapai 0,94-0,954 di LB (ResNet-50 5-fold 224px = 0,954; CoAtNet 224px single-fold = 0,950), dgn resep
yg sama: pseudo-label multi-guru lewat OOF + model kecil + fold. Itu persis jalur noisy-student kita, tapi kita
memakai CoAtNet 384px (0,85 dtk/studi) -- 10x lebih mahal. Model CNN kecil di 224px ~10x lebih murah, jadi satu
siklus (6 fold) ~1 jam GPU, dan keluarga CNN-murni tak berkorelasi dgn CoAtNet.

DESAIN:
  - 6 fold sadar-GRUP (laporan byte-identik satu grup), gold-58 TAK PERNAH ikut latih (prediksi gold tiap fold bersih)
  - SATU sesi Kaggle = 2 GPU T4: dua proses pekerja (CUDA_VISIBLE_DEVICES=0/1), 3 fold per GPU -> 2x hasil per jam kuota
  - 4 studi per forward (48 window) supaya statistik BatchNorm layak; 2 langkah akumulasi = 8 studi per update
  - cache volume (336px/64 irisan/rentang 2-98%) dimount dari kernel CPU `rsna-knee-volcache-w*`; window diperkecil
    336->224 dgn antialias; augmentasi: crop acak 88-100% + kecerahan/kontras (TANPA flip: kiri-kanan = medial-lateral)
  - EMA bobot (disimpan EMA); gold-58 dinilai tiap epoch HANYA diagnostik
Pemakaian: python3 build_train_cheap_kfold.py <arch=resnet50> [res=224] [epochs=6] [seed=1] [nfold=6] [tag=] [labels=r1]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 224
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 1
NFOLD = int(sys.argv[5]) if len(sys.argv) > 5 else 6
TAGX = sys.argv[6] if len(sys.argv) > 6 else ""
LABELS = sys.argv[7] if len(sys.argv) > 7 else "r1"
OPTS = dict(kv.split("=") for kv in sys.argv[9].split(",")) if len(sys.argv) > 9 and sys.argv[9] else {}
# opsi: bs (studi/forward), acc (akumulasi), warmup, kwin (window/studi saat latih), bias_init (1 = bias kepala di logit prior), ema, lr (lr_bb)
OPTS2 = dict(kv.split("=") for kv in sys.argv[10].split(",")) if len(sys.argv) > 10 and sys.argv[10] else None
# OPTS2 (argumen ke-10) = resep KEDUA utk GPU1 pada FOLD YANG SAMA (mode probe): kunci opsi sama + arch= res= epochs=
SMOKE = len(sys.argv) > 8 and sys.argv[8] == "smoke"     # uji jalur kode: 48 studi latih, 1 epoch, fold 0 (GPU0) & fold 1 (GPU1)
if SMOKE:
    TAGX = TAGX + "-smoke"
    EPOCHS = 1

DLW = "inyalala/dl-vision-weights"     # 10 GB; hf_weights/models--timm--<nama>/snapshots/<hash>/model.safetensors
def _dlw(name):
    return f"hf_weights/models--timm--{name}/snapshots/*/model.safetensors"
ARCHS = {
    # timm resnet50.a1_in1k (kunci = timm = torchvision)
    "resnet50":      dict(arch="resnet50", ds="ildarlukmanov/resnet50-a1-in1k-timm", rel="model.safetensors",
                          match="resnet50", lr_bb=2e-4),
    # RadImageNet ResNet-50 (1,35 jt citra MEDIS, termasuk MRI lutut); sumber HF Lab-Rasool, upstream_license=MIT.
    # kunci backbone.{0,1,4,5,6,7}.* -> timm conv1/bn1/layer1..4 ; normalisasi mean=std=0,5 (sama dgn encoder Rad pipeline publik)
    "radresnet50":   dict(arch="resnet50", ds="shigechan/radimagenet-resnet50-official", rel="ResNet50.pt",
                          match="", lr_bb=2e-4, fmt="rad", norm="rad"),
    # keluarga lain (timm, safetensors) dari dataset dl-vision-weights
    "convnext_tiny": dict(arch="convnext_tiny", ds=DLW, rel=_dlw("convnext_tiny.fb_in22k_ft_in1k"), match="", lr_bb=1e-4),
    "convnext_small": dict(arch="convnext_small", ds=DLW, rel=_dlw("convnext_small.fb_in22k_ft_in1k"), match="", lr_bb=8e-5),
    "effb3":         dict(arch="efficientnet_b3", ds=DLW, rel=_dlw("efficientnet_b3.ra2_in1k"), match="", lr_bb=3e-4),
    "resnext50":     dict(arch="resnext50_32x4d", ds=DLW, rel=_dlw("resnext50_32x4d.a1h_in1k"), match="", lr_bb=2e-4),
    "swin_tiny":     dict(arch="swin_tiny_patch4_window7_224", ds=DLW, rel=_dlw("swin_tiny_patch4_window7_224.ms_in1k"), match="", lr_bb=1e-4),
}
assert ARCH in ARCHS, f"arch tak dikenal: {ARCH} (pilihan {list(ARCHS)})"
A = ARCHS[ARCH]
TAG = f"cheap-{ARCH}{TAGX}-s{SEED}"
SLUG = f"rsna-knee-{TAG}".replace("_", "-")     # slug Kaggle: tanpa garis bawah
OUTDIR = REPO / f"kaggle_kernel_{TAG.replace('-', '_')}"
LABELDS = {"r1": ("dimaspashaakrilian/rsna-knee-fused-labels-r1", "fused_labels_r1.csv")}
assert LABELS in LABELDS, f"label tak dikenal: {LABELS}"
LDS, LFILE = LABELDS[LABELS]

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

WORKER = r'''
import os, sys, time, glob, math, random, hashlib, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
t0 = time.time()
GTAG = os.environ.get('GTAG', '?')
def tlog(m): print(f'[g{GTAG} t={time.time()-t0:.0f}s] {m}', flush=True)

KE = {'__name__': '_ke'}
exec(compile(open('/kaggle/working/_ke_src.py').read(), '<ke>', 'exec'), KE)
LAB = KE['LAB']; N_T = len(LAB)
CFG = json.loads(os.environ['CFG'])
ARCH, RES, EPOCHS, SEED, NFOLD = CFG['arch'], int(CFG['res']), int(CFG['epochs']), int(CFG['seed']), int(CFG['nfold'])
FOLDS = [int(x) for x in os.environ['FOLDS'].split(',') if x != '']
LR_BB, LR_HEAD, WD = float(CFG['lr_bb']), 2e-3, 0.02
K_WIN, BS, ACC, K_EVAL, WARMUP = int(CFG.get('kwin', 12)), int(CFG.get('bs', 4)), int(CFG.get('acc', 2)), 42, int(CFG.get('warmup', 150))
DEV = 'cuda'
NW = int(os.environ.get('NW', '3'))
OUTTAG = CFG['outtag']
tlog(f'worker mulai | arch={ARCH} res={RES} epochs={EPOCHS} seed={SEED} nfold={NFOLD} folds={FOLDS}')

def _bglob(name):
    return [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/')
            for p in glob.glob(d + name)]

# ---------------- data ----------------
ROOT = KE['find_test_root']()
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv'); tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
HAS_SER = set(tser['StudyInstanceUID'])
GOLD = set(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in HAS_SER]
MOUNTED = {}
for d in (glob.glob('/kaggle/input/*/volcache') + glob.glob('/kaggle/input/*/*/volcache') + glob.glob('/kaggle/input/*/*/*/volcache')):
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.npz'): MOUNTED[f[:-4]] = os.path.join(d, f)
tlog(f'studi={len(ALL_IDS)} gold={len(GOLD)} cache termount={len(MOUNTED)}')
hits = _bglob(CFG['label_file']); assert hits, 'csv label tak ketemu'
FL = pd.read_csv(hits[0]); FL['StudyInstanceUID'] = FL['StudyInstanceUID'].astype(str); FL = FL.set_index('StudyInstanceUID')
tlog('label csv: ' + hits[0])
weak_ids = [s for s in ALL_IDS if s not in GOLD and s in FL.index]
miss = [s for s in weak_ids + sorted(GOLD) if s not in MOUNTED]
assert not miss, f'cache hilang utk {len(miss)} studi'
gold_ids = sorted(GOLD)
SMOKE = bool(CFG.get('smoke'))
if SMOKE: gold_ids = gold_ids[:16]

rep = train.set_index('StudyInstanceUID')['Report'] if 'Report' in train.columns else None
def gkey(s):
    r = None if rep is None else rep.get(s)
    if r is None or isinstance(r, float) or not str(r).strip(): return 'u:' + s
    return 'r:' + hashlib.md5(' '.join(str(r).split()).encode()).hexdigest()
groups = {}
for s in weak_ids: groups.setdefault(gkey(s), []).append(s)
glist = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
sizes = [0] * NFOLD; fold_of = {}
for gk, members in glist:
    f = int(np.argmin(sizes))
    for m in members: fold_of[m] = f
    sizes[f] += len(members)
tlog(f'grup={len(groups)} studi_lemah={len(weak_ids)} ukuran_fold={sizes}')

Ycols = [t for t in LAB]
def label_arrays(ids):
    y = FL.loc[ids, Ycols].fillna(0.28).values.astype(np.float32)
    c = np.stack([FL.loc[ids, f'{t}__conf'].fillna(0.15).values.astype(np.float32) for t in LAB], 1)
    return y, (0.5 + np.clip(c, 0, 1)).astype(np.float32)

MEAN = KE['_MEAN']; STD = KE['_STD']
if CFG.get('norm') == 'rad':
    MEAN = torch.full((3, 1, 1), 0.5); STD = torch.full((3, 1, 1), 0.5)

def _resize(t, res):
    if t.shape[-1] != res:
        t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False, antialias=True)[0]
    return t

def rand_windows(vol, mask, k, res):
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3: valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(max(1, lo), min(D - 1, hi)))
    if not cs: cs = [max(1, min(D - 2, (lo + hi) // 2))]
    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))
    while len(cs) < k: cs.append(cs[np.random.randint(len(cs))])
    H = vol.shape[-1]
    wins = torch.empty((k, 3, res, res), dtype=torch.float32)
    for j, c in enumerate(cs):
        t = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        fr = random.uniform(0.88, 1.0); s = int(round(H * fr)); y0 = random.randint(0, H - s); x0 = random.randint(0, H - s)
        t = t[:, y0:y0 + s, x0:x0 + s]
        t = _resize(t, res)
        m_ = t.mean(); t = ((t - m_) * random.uniform(0.85, 1.15) + m_) * random.uniform(0.85, 1.15)
        wins[j] = (t.clamp(0, 1) - MEAN) / STD
    return wins

def eval_windows_aa(vol, mask, k, res):
    D = vol.shape[0]
    cs = KE['_eval_centers'](mask, D, k)
    wins = torch.empty((len(cs), 3, res, res), dtype=torch.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        t = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        wins[j] = (_resize(t, res) - MEAN) / STD
    return wins

class TrainDS(torch.utils.data.Dataset):
    def __init__(self, ids, Y, W): self.ids, self.Y, self.W = ids, Y, W
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        z = np.load(MOUNTED[self.ids[i]]); vol, mask = z['vol'], z['mask']
        return rand_windows(vol, mask, K_WIN, RES), torch.from_numpy(self.Y[i]), torch.from_numpy(self.W[i])

class EvalDS(torch.utils.data.Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        z = np.load(MOUNTED[self.ids[i]]); vol, mask = z['vol'], z['mask']
        return eval_windows_aa(vol, mask, K_EVAL, RES)

def worker_init(_):
    s = (torch.initial_seed() + 12345) % (2 ** 31)
    np.random.seed(s); random.seed(s)

# ---------------- model ----------------
def find_weights(rel, match=''):
    hits = set()
    for base in ('/kaggle/input/', '/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
        for p in glob.glob(base + rel):
            if match in p.lower(): hits.add(os.path.realpath(p))
    return sorted(hits)

def make_model():
    bb = KE['build_backbone'](ARCH, pretrained=False)
    wf = find_weights(CFG['rel'], CFG['match'])
    assert len(wf) == 1, f"bobot pra-latih harus tepat 1: {wf}"
    if CFG.get('fmt') == 'rad':
        raw = torch.load(wf[0], map_location='cpu', weights_only=True); sd = {}
        for k, v in raw.items():
            k2 = k[len('backbone.'):] if k.startswith('backbone.') else k
            head, _, rest = k2.partition('.')
            if head == '0': sd['conv1.' + rest] = v
            elif head == '1': sd['bn1.' + rest] = v
            elif head in ('4', '5', '6', '7'): sd[f'layer{int(head) - 3}.' + rest] = v
            else: raise RuntimeError(f'kunci RadImageNet tak dikenal: {k}')
    else:
        from safetensors.torch import load_file
        sd = load_file(wf[0])
    r = bb.load_state_dict(sd, strict=False)
    tlog(f'bobot {wf[0][-70:]}: missing={list(r.missing_keys)[:5]} unexpected={list(r.unexpected_keys)[:5]}')
    assert len(r.missing_keys) == 0, 'bobot tak lengkap'
    return KE['RaptorClassifier'](bb, F_dim=bb.num_features)

class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay; self.n = 0
        self.sd = {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        self.n += 1; d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.is_floating_point(): self.sd[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else: self.sd[k].copy_(v)

def auc1(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
Yg = train.set_index('StudyInstanceUID').loc[gold_ids, LAB].values.astype(float)
def macro(P, Y): return float(np.nanmean([auc1(Y[:, j], P[:, j]) for j in range(N_T)]))

@torch.no_grad()
def predict(model_, ids):
    model_.eval(); out = []
    dl = torch.utils.data.DataLoader(EvalDS(ids), batch_size=1, shuffle=False, num_workers=NW, worker_init_fn=worker_init)
    for xw in dl:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                p = torch.sigmoid(model_(xw.to(DEV)).float())
            out.append(p[0].cpu().numpy())
        except Exception as e:
            tlog(f'eval gagal: {type(e).__name__}: {str(e)[:100]}'); out.append(np.full(N_T, 0.5, np.float32))
    return np.stack(out)

# ---------------- latih per fold ----------------
for FOLD in FOLDS:
    seed = SEED * 100 + FOLD
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    train_ids = [s for s in weak_ids if fold_of[s] != FOLD]
    held_ids = [s for s in weak_ids if fold_of[s] == FOLD]
    assert not ({gkey(s) for s in train_ids} & {gkey(s) for s in held_ids}), 'BOCOR grup laporan'
    assert not (set(train_ids) & GOLD), 'gold bocor ke latih'
    if SMOKE: train_ids, held_ids = train_ids[:48], held_ids[:16]
    Y, W = label_arrays(train_ids)
    tlog(f'=== FOLD {FOLD}/{NFOLD}: latih={len(train_ids)} held-out={len(held_ids)} gold={len(gold_ids)}')
    model = make_model().to(DEV)
    if CFG.get('bias_init'):
        pr = np.clip((W * Y).sum(0) / W.sum(0), 0.02, 0.98)
        with torch.no_grad(): model.clsb.copy_(torch.logit(torch.tensor(pr, dtype=torch.float32)).to(DEV))
        tlog('bias kepala diinisialisasi ke logit prior: ' + ' '.join(f'{x:.2f}' for x in pr))
    head_params = list(model.norm.parameters()) + list(model.att.parameters()) + [model.clsW, model.clsb]
    hp = {id(p) for p in head_params}
    bb_decay = [p for p in model.backbone.parameters() if p.ndim > 1]
    bb_nodecay = [p for p in model.backbone.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': bb_decay, 'lr': LR_BB, 'weight_decay': WD},
                             {'params': bb_nodecay, 'lr': LR_BB, 'weight_decay': 0.0},
                             {'params': head_params, 'lr': LR_HEAD, 'weight_decay': WD}])
    base_lrs = [g['lr'] for g in opt.param_groups]
    steps_per_epoch = len(train_ids) // (BS * ACC)
    total_steps = steps_per_epoch * EPOCHS
    def set_lr(s):
        sc = min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
        for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * sc
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(model, float(CFG.get('ema', 0.998))); ema_model = copy.deepcopy(model)
    dl = torch.utils.data.DataLoader(TrainDS(train_ids, Y, W), batch_size=BS, shuffle=True, drop_last=True,
                                     num_workers=NW, worker_init_fn=worker_init, prefetch_factor=3, persistent_workers=True)
    tlog(f'steps={total_steps} ({steps_per_epoch}/epoch) lr_bb={LR_BB} lr_head={LR_HEAD}')
    step = 0
    for ep in range(EPOCHS):
        model.train(); torch.backends.cudnn.benchmark = True
        ep_loss, nb = 0.0, 0; t_ep = time.time()
        opt.zero_grad(set_to_none=True)
        for bi, (xb, yb, wb) in enumerate(dl):
            xb, yb, wb = xb.to(DEV, non_blocking=True), yb.to(DEV), wb.to(DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(xb)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), yb, reduction='none') * wb).mean()
            scaler.scale(loss / ACC).backward()
            ep_loss += float(loss.detach()); nb += 1
            if (bi + 1) % ACC == 0:
                set_lr(step); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); ema.update(model); step += 1
            if (bi + 1) % 200 == 0:
                tlog(f'f{FOLD} ep{ep+1} {(bi+1)*BS}/{len(train_ids)} loss~{ep_loss/nb:.4f} lr_bb={opt.param_groups[0]["lr"]:.2e} {(time.time()-t_ep):.0f}s')
        torch.backends.cudnn.benchmark = False
        ema_model.load_state_dict({k: v.to(ema_model.state_dict()[k].dtype) for k, v in ema.sd.items()})
        Pr = predict(model, gold_ids); Pe = predict(ema_model, gold_ids)
        tlog(f'=== f{FOLD} EPOCH {ep+1}/{EPOCHS} loss={ep_loss/max(1,nb):.4f} ({time.time()-t_ep:.0f}s) | gold-58 diag: raw={macro(Pr,Yg):.4f} ema={macro(Pe,Yg):.4f}')
    del dl
    # simpan EMA + OOF
    torch.save({'arch': ARCH, 'res': RES, 'norm': CFG.get('norm', 'imagenet'), 'model': {k: v.detach().cpu() for k, v in ema_model.state_dict().items()}},
               f'/kaggle/working/{OUTTAG}_f{FOLD}.pt')
    Ph = predict(ema_model, held_ids); Pg = predict(ema_model, gold_ids)
    oof = pd.DataFrame(np.vstack([Ph, Pg]), columns=LAB)
    oof.insert(0, 'StudyInstanceUID', held_ids + gold_ids)
    oof['is_gold'] = [0] * len(held_ids) + [1] * len(gold_ids); oof['fold'] = [FOLD] * len(held_ids) + [-1] * len(gold_ids)
    oof.to_csv(f'/kaggle/working/oof_{OUTTAG}_f{FOLD}.csv', index=False)
    Th = FL.loc[held_ids, LAB].fillna(0.28).values.astype(float)
    cs = [np.corrcoef(Ph[:, j], Th[:, j])[0, 1] for j in range(N_T)]
    tlog(f'*** FOLD {FOLD} SELESAI | gold-58 (EMA) = {macro(Pg, Yg):.4f} | korelasi(OOF, label fusi) rata2 = {np.nanmean(cs):.4f}')
    del model, ema_model, opt, ema; torch.cuda.empty_cache()
tlog('WORKER DONE')
'''

PARENT = r'''
import os, sys, time, glob, json, subprocess
import numpy as np, pandas as pd
t0 = time.time()
def tlog(m): print(f'[main t={time.time()-t0:.0f}s] {m}', flush=True)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
open('/kaggle/working/_ke_src.py', 'w').write(KE_SRC)
open('/kaggle/working/worker.py', 'w').write(WORKER_SRC)
import torch
tlog(f'GPU terlihat: {torch.cuda.device_count()} | {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')
NG = max(1, torch.cuda.device_count())
folds = list(range(CFG['nfold'])) if not CFG.get('smoke') else list(range(min(2, CFG['nfold'])))
assign = {g: [f for f in folds if f % NG == g] for g in range(NG)}
CFGS = {g: CFG for g in range(NG)}
if CFG_B is not None and NG >= 2:      # mode probe: dua resep pada fold yg SAMA
    assign = {0: [CFG['probe_fold']], 1: [CFG['probe_fold']]}; CFGS = {0: CFG, 1: CFG_B}
tlog(f'pembagian fold per GPU: {assign} | probe={CFG_B is not None}')
procs = {}
for g, fl in assign.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), FOLDS=','.join(map(str, fl)), GTAG=str(g), CFG=json.dumps(CFGS[g]))
    fh = open(f'/kaggle/working/worker{g}.log', 'w')
    procs[g] = (subprocess.Popen([sys.executable, '-u', '/kaggle/working/worker.py'], env=env, stdout=fh, stderr=subprocess.STDOUT), fh)
pos = {g: 0 for g in procs}
def pump():
    for g in procs:
        with open(f'/kaggle/working/worker{g}.log') as fh:
            fh.seek(pos[g]); new = fh.read(); pos[g] = fh.tell()
        for line in new.splitlines(): print(line, flush=True)
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(10); pump()
pump()
for g, (p, fh) in procs.items(): tlog(f'pekerja g{g} selesai rc={p.returncode}')

# ---- gabung: ensemble gold-58 lintas fold ----
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
files = sorted(glob.glob(f"/kaggle/working/oof_{CFG['outtag']}_f*.csv"))
if CFG_B is not None:
    tlog('MODE PROBE: ringkasan per resep dilewati (lihat baris "FOLD ... SELESAI" tiap GPU)'); files = []
tlog(f'berkas OOF: {len(files)}')
if files:
    ROOT = [p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(p + '/train.csv')][0]
    train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
    def auc1(yy, ss):
        pos_, neg = ss[yy == 1], ss[yy == 0]
        r = pd.Series(np.concatenate([pos_, neg])).rank().values
        return float((r[:len(pos_)].sum() - len(pos_) * (len(pos_) + 1) / 2) / (len(pos_) * len(neg)))
    G = []; weak = []
    for f in files:
        o = pd.read_csv(f, dtype={'StudyInstanceUID': str})
        G.append(o[o.is_gold == 1].set_index('StudyInstanceUID')); weak.append(o[o.is_gold == 0])
    ids = list(G[0].index); Y = train.set_index('StudyInstanceUID').loc[ids, LAB].values.astype(float)
    P = np.mean([g.loc[ids, LAB].values for g in G], 0)
    R = np.mean([np.column_stack([pd.Series(g.loc[ids, LAB].values[:, j]).rank(pct=True).values for j in range(12)]) for g in G], 0)
    for i, g in enumerate(G): tlog(f'  fold-model {i}: gold-58 = {np.mean([auc1(Y[:, j], g.loc[ids, LAB].values[:, j]) for j in range(12)]):.4f}')
    a = [auc1(Y[:, j], P[:, j]) for j in range(12)]
    tlog(f'ENSEMBLE {len(G)} fold | gold-58 prob-mean = {np.mean(a):.4f} | rank-mean = {np.mean([auc1(Y[:, j], R[:, j]) for j in range(12)]):.4f}   [acuan: c5 stok 0.9213, 4-murid-CoAtNet 0.9286, v3 0.9234]')
    for l, v in zip(LAB, a): tlog(f'    {l:18s} {v:.4f}')
    pd.concat(weak).to_csv(f"/kaggle/working/oof_weak_{CFG['outtag']}_all.csv", index=False)
tlog('SELESAI')
'''

CFG = dict(arch=A["arch"], res=RES, epochs=EPOCHS, seed=SEED, nfold=NFOLD, lr_bb=A["lr_bb"],
           rel=A["rel"], match=A["match"], fmt=A.get("fmt", "safetensors"), norm=A.get("norm", "imagenet"), label_file=LFILE, outtag=f"cheap_{ARCH}{TAGX}_s{SEED}".replace("-", "_"),
           smoke=SMOKE, bs=int(OPTS.get("bs", 4)), acc=int(OPTS.get("acc", 2)), warmup=int(OPTS.get("warmup", 150)),
           kwin=int(OPTS.get("kwin", 12)), bias_init=int(OPTS.get("bias_init", 0)), ema=float(OPTS.get("ema", 0.998)))
if "lr" in OPTS:
    CFG["lr_bb"] = float(OPTS["lr"])
CFG["probe_fold"] = int(OPTS.get("probe_fold", 0))
CFG_B = None
if OPTS2 is not None:
    B = dict(CFG)
    arch_b = OPTS2.get("arch", ARCH)
    assert arch_b in ARCHS, f"arch ke-2 tak dikenal: {arch_b}"
    AB = ARCHS[arch_b]
    B.update(arch=AB["arch"], rel=AB["rel"], match=AB["match"], fmt=AB.get("fmt", "safetensors"), norm=AB.get("norm", "imagenet"), lr_bb=AB["lr_bb"])
    for k_, v_ in OPTS2.items():
        if k_ in ("bs", "acc", "warmup", "kwin", "bias_init"): B[k_] = int(v_)
        elif k_ == "ema": B["ema"] = float(v_)
        elif k_ == "lr": B["lr_bb"] = float(v_)
        elif k_ in ("res", "epochs"): B[k_] = int(v_)
    B["outtag"] = CFG["outtag"] + "_B"
    CFG["outtag"] = CFG["outtag"] + "_A"
    CFG_B = B
    CFG["probe"] = True

cells = [
    "KE_SRC = " + repr(ke_src) + "\nWORKER_SRC = " + repr(WORKER) + "\nCFG = " + repr(CFG) + "\nCFG_B = " + repr(CFG_B) + "\n",
    PARENT,
]
nb = {"cells": [{"cell_type": "code", "source": c.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None}
                for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
OUTDIR.mkdir(exist_ok=True)
NAME = f"{SLUG}.ipynb"
(OUTDIR / NAME).write_text(json.dumps(nb, indent=1))
meta = {
    "id": f"dimaspashaakrilian/{SLUG}", "title": f"RSNA Knee - Cheap {ARCH}{TAGX} s{SEED}", "code_file": NAME,
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": list(dict.fromkeys([LDS, A["ds"]] + ([ARCHS[OPTS2.get("arch", ARCH)]["ds"]] if OPTS2 else []))),
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-w{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUTDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
# paket mandiri utk Colab (colab/<tag>/): harness, worker, cfg
COLAB = REPO / "colab" / TAG.replace("-", "_")
COLAB.mkdir(parents=True, exist_ok=True)
_cv = lambda t: t.replace("/kaggle/working/_ke_src.py", "/content/colab_pkg/_ke_src.py").replace("/kaggle/input", "/content/kin").replace("/kaggle/working", "/content/kw")   # /kaggle read-only di Colab
(COLAB / "_ke_src.py").write_text(_cv(ke_src))
(COLAB / "worker.py").write_text(_cv(WORKER))
(COLAB / "cfg.json").write_text(json.dumps(CFG))
(COLAB / "datasets.json").write_text(json.dumps({"label_ds": LDS, "weights_ds": A["ds"]}))
# kerangka sintaks: kompilasi tiap sel + worker
import ast
for c in cells:
    compile(c, "<cell>", "exec")
ast.parse(WORKER)
print("built:", OUTDIR.name, "| slug:", SLUG, "| cfg:", CFG)
