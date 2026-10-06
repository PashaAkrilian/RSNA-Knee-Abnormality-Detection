#!/usr/bin/env python3
"""Reader gaya KneeNet (resep publik goodpjw2008, LB solo 0,929 / +0,001 di stack) dilatih di LABEL FUSI kita dgn cache volcache kita.

Arsitektur (disalin dari rsna-knee-2-5d-convnext-reader/knee.py, N_SLOTS=5 utk slot cache kita): ConvNeXt-Tiny bersama (IN22k) pd window
3-irisan tiap slot -> token + embedding slot + posisi irisan -> transformer 2 lapis di tingkat STUDI -> attention-pooling per temuan.
Augmentasi di perangkat (rotasi +-12, skala, geser, gamma/gain) via affine_grid. Slot cache `w` (336px, SLOTS64, rentang 2-98%):
  sag-fluid [0:18) | sag-non [18:32) | cor-fluid [32:44) | cor-non [44:52) | axial [52:64)
Bentuk tensor TETAP (semua slot dijalankan, slot kosong di-mask) -> aman utk XLA/TPU.
Pemakaian: python3 build_train_kneenet.py <gpu|tpu> <folds=0|0+1+2|all> [epochs=14] [res=256] [k=8] [seed=1] [nfold=6] [tag=]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEVKIND = sys.argv[1]
FOLDS = sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 14
RES = int(sys.argv[4]) if len(sys.argv) > 4 else 256
KWIN = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 1
NFOLD = int(sys.argv[7]) if len(sys.argv) > 7 else 6
TAGX = sys.argv[8] if len(sys.argv) > 8 else ""
assert DEVKIND in ("gpu", "tpu")
fold_list = list(range(NFOLD)) if FOLDS == "all" else [int(x) for x in FOLDS.split("+")]
TAG = f"kneenet{TAGX}-{DEVKIND}-s{SEED}"
SLUG = f"rsna-knee-{TAG}"
OUTTAG = f"kneenet{TAGX}_s{SEED}".replace("-", "_")
OUT = REPO / f"kaggle_kernel_{TAG.replace('-', '_')}"
ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

WORKER = r'''
import os, sys, time, glob, math, random, hashlib, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
CFG = json.loads(os.environ['CFG']); DEVKIND = CFG['dev']
if DEVKIND == 'tpu':
    import torch_xla.core.xla_model as xm
    DEV = xm.xla_device(); AC = lambda: torch.autocast('xla', dtype=torch.bfloat16)
    def mark(): xm.mark_step()
else:
    DEV = 'cuda'; AC = lambda: torch.autocast('cuda', dtype=torch.float16)
    def mark(): pass
NW = int(os.environ.get('NW', '4'))
RES, KW, EPOCHS, SEED, NFOLD = int(CFG['res']), int(CFG['k']), int(CFG['epochs']), int(CFG['seed']), int(CFG['nfold'])
FOLDS = [int(x) for x in os.environ['FOLDS'].split(',') if x != '']
BS, ACC, K_EVAL = int(CFG['bs']), int(CFG['acc']), int(CFG['k_eval'])
LR_ENC, LR_HEAD, WD = 1e-4, 5e-4, 0.02
BOUNDS = [(0, 18), (18, 32), (32, 44), (44, 52), (52, 64)]; NS = len(BOUNDS)
exec(compile(open('/kaggle/working/_ke_src.py').read(), '<ke>', 'exec'), KE := {'__name__': '_ke'}) if False else None
KE = {'__name__': '_ke'}; exec(compile(open('/kaggle/working/_ke_src.py').read(), '<ke>', 'exec'), KE)
LAB = KE['LAB']; N_T = len(LAB)
def _bglob(name):
    return [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/') for p in glob.glob(d + name)]

ROOT = KE['find_test_root']()
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv'); tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
HAS = set(tser['StudyInstanceUID']); GOLD = set(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in HAS]
MOUNTED = {}
for d in (glob.glob('/kaggle/input/*/volcache') + glob.glob('/kaggle/input/*/*/volcache') + glob.glob('/kaggle/input/*/*/*/volcache')):
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.npz'): MOUNTED[f[:-4]] = os.path.join(d, f)
hits = _bglob('fused_labels_r1.csv'); assert hits
FL = pd.read_csv(hits[0]); FL['StudyInstanceUID'] = FL['StudyInstanceUID'].astype(str); FL = FL.set_index('StudyInstanceUID')
weak_ids = [s for s in ALL_IDS if s not in GOLD and s in FL.index]; gold_ids = sorted(GOLD)
assert not [s for s in weak_ids + gold_ids if s not in MOUNTED]
tlog(f'studi={len(ALL_IDS)} gold={len(gold_ids)} cache={len(MOUNTED)} dev={DEVKIND}')
rep = train.set_index('StudyInstanceUID')['Report'] if 'Report' in train.columns else None
def gkey(s):
    r = None if rep is None else rep.get(s)
    if r is None or isinstance(r, float) or not str(r).strip(): return 'u:' + s
    return 'r:' + hashlib.md5(' '.join(str(r).split()).encode()).hexdigest()
groups = {}
for s in weak_ids: groups.setdefault(gkey(s), []).append(s)
sizes = [0] * NFOLD; fold_of = {}
for gk, mem in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    f = int(np.argmin(sizes))
    for m in mem: fold_of[m] = f
    sizes[f] += len(mem)
tlog(f'grup={len(groups)} lemah={len(weak_ids)} fold={sizes}')
def label_arrays(ids):
    y = FL.loc[ids, LAB].fillna(0.28).values.astype(np.float32)
    c = np.stack([FL.loc[ids, f'{t}__conf'].fillna(0.15).values.astype(np.float32) for t in LAB], 1)
    return y, (0.5 + np.clip(c, 0, 1)).astype(np.float32)

def centres(valid, k, train):
    a, b = valid[0], valid[-1]
    if train:
        edges = np.linspace(a, b + 1, k + 1); c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    c = np.round(c).astype(int)
    return np.clip(c, a, b)

class SlotDS(torch.utils.data.Dataset):
    """Satu item = satu studi: uint8 [NS, K, 3, H, W], mask [NS], pos [NS, K], target, bobot."""
    def __init__(self, ids, Y=None, W=None, k=8, train=False): self.ids, self.Y, self.W, self.k, self.train = ids, Y, W, k, train
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        z = np.load(MOUNTED[self.ids[i]]); vol, m = z['vol'], z['mask']
        H = vol.shape[-1]; x = np.zeros((NS, self.k, 3, H, H), np.uint8); mk = np.zeros(NS, bool); pos = np.zeros((NS, self.k), np.float32)
        for s, (lo, hi) in enumerate(BOUNDS):
            valid = [j for j in range(lo + 1, hi - 1) if m[j] > 0 and m[j - 1] > 0 and m[j + 1] > 0]
            if len(valid) < 1: continue
            c = centres(valid, self.k, self.train)
            x[s] = np.stack([vol[[j - 1, j, j + 1]] for j in c]); mk[s] = True; pos[s] = (c - lo) / max(hi - lo - 1, 1)
        y = self.Y[i] if self.Y is not None else np.zeros(N_T, np.float32)
        w = self.W[i] if self.W is not None else np.ones(N_T, np.float32)
        return torch.from_numpy(x), torch.from_numpy(mk), torch.from_numpy(pos), torch.from_numpy(np.asarray(y, np.float32)), torch.from_numpy(np.asarray(w, np.float32))

MEAN_ = (0.485, 0.456, 0.406); STD_ = (0.229, 0.224, 0.225)
def make_views(x, res, train, group, fov=0.92):
    """x uint8 [N,3,H,W] (N = n_series*group). Recorte FOV tengah + resize; train: rotasi/skala/geser/gamma/gain bersama per slot."""
    n = x.shape[0]; ns = n // group; dev = x.device; x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12); sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12; ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5); gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev); sc = torch.full((ns,), fov, device=dev); tx = ty = torch.zeros(ns, device=dev); gamma = gain = None
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    if train:
        x = x.clamp_min(1e-4).pow(gamma.repeat_interleave(group).view(n, 1, 1, 1)) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean = torch.tensor(MEAN_, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD_, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std

class KneeNet(nn.Module):
    def __init__(self, backbone, dim=384, depth=2, drop=0.2, drop_path=0.1):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=False, num_classes=0, drop_path_rate=drop_path)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(NS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, N_T))
        self.cls_w = nn.Parameter(torch.randn(N_T, dim) * 0.02); self.cls_b = nn.Parameter(torch.zeros(N_T))
    def forward(self, x, mask, pos, res, train=False):
        b, s, k = x.shape[:3]
        f = self.enc(make_views(x.reshape(b * s * k, *x.shape[3:]), res, train, k))      # semua slot dijalankan (bentuk tetap)
        feat = f.view(b, s, k, -1)
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum('btc,btd->bcd', a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

def make_model():
    m = KneeNet('convnext_tiny')
    wf = sorted({os.path.realpath(p) for p in _bglob('hf_weights/models--timm--convnext_tiny.fb_in22k_ft_in1k/snapshots/*/model.safetensors')})
    assert len(wf) == 1, wf
    from safetensors.torch import load_file
    r = m.enc.load_state_dict(load_file(wf[0]), strict=False)
    tlog(f'bobot convnext_tiny IN22k: missing={list(r.missing_keys)[:4]} unexpected={list(r.unexpected_keys)[:4]}')
    assert len(r.missing_keys) == 0
    return m

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
    dl = torch.utils.data.DataLoader(SlotDS(ids, k=K_EVAL), batch_size=BS, shuffle=False, num_workers=NW)
    for x, mk, pos, _, _ in dl:
        n = x.shape[0]
        if n < BS:      # bentuk tetap utk XLA: isi ulang batch terakhir
            pad = BS - n; x = torch.cat([x, x[:1].expand(pad, *x.shape[1:])]); mk = torch.cat([mk, mk[:1].expand(pad, *mk.shape[1:])]); pos = torch.cat([pos, pos[:1].expand(pad, *pos.shape[1:])])
        with AC():
            p = torch.sigmoid(model_(x.to(DEV), mk.to(DEV), pos.to(DEV), RES).float())
        mark(); out.append(p[:n].cpu().numpy())
    return np.concatenate(out)

class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay; self.n = 0; self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        self.n += 1; d = self.decay if DEVKIND == 'tpu' else min(self.decay, (1 + self.n) / (10 + self.n))
        ep, mp = list(self.m.parameters()), list(model.parameters())
        torch._foreach_mul_(ep, d); torch._foreach_add_(ep, mp, alpha=1 - d)
        for be, bm in zip(self.m.buffers(), model.buffers()): be.copy_(bm)

for FOLD in FOLDS:
    np.random.seed(SEED * 100 + FOLD); random.seed(SEED * 100 + FOLD); torch.manual_seed(SEED * 100 + FOLD)
    train_ids = [s for s in weak_ids if fold_of[s] != FOLD]; held_ids = [s for s in weak_ids if fold_of[s] == FOLD]
    assert not ({gkey(s) for s in train_ids} & {gkey(s) for s in held_ids}) and not (set(train_ids) & GOLD)
    Y, W = label_arrays(train_ids)
    tlog(f'=== FOLD {FOLD}/{NFOLD}: latih={len(train_ids)} held={len(held_ids)} gold={len(gold_ids)}')
    model = make_model().to(DEV)
    pr = np.clip((W * Y).sum(0) / W.sum(0), 0.02, 0.98)
    with torch.no_grad(): model.cls_b.copy_(torch.logit(torch.tensor(pr, dtype=torch.float32)).to(DEV))
    enc_p = list(model.enc.parameters()); eid = {id(p) for p in enc_p}
    head_p = [p for p in model.parameters() if id(p) not in eid]
    opt = torch.optim.AdamW([{'params': enc_p, 'lr': LR_ENC}, {'params': head_p, 'lr': LR_HEAD}], weight_decay=WD)
    base = [g['lr'] for g in opt.param_groups]
    spe = len(train_ids) // (BS * ACC); total = spe * EPOCHS; warm = spe
    def set_lr(s):
        sc = min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
        if DEVKIND == 'tpu': sc = max(1, round(sc * 20)) / 20
        for g, b in zip(opt.param_groups, base): g['lr'] = b * sc
    scaler = torch.amp.GradScaler('cuda') if DEVKIND == 'gpu' else None
    ema = EMA(model, 0.998)
    dl = torch.utils.data.DataLoader(SlotDS(train_ids, Y, W, k=KW, train=True), batch_size=BS, shuffle=True, drop_last=True,
                                     num_workers=NW, prefetch_factor=2, persistent_workers=True)
    tlog(f'steps={total} ({spe}/epoch) bs={BS}x{ACC} res={RES} k={KW}x{NS} lr={LR_ENC}/{LR_HEAD}')
    step = 0
    for ep in range(EPOCHS):
        model.train(); t_ep = time.time(); eloss, nb = torch.zeros((), device=DEV), 0; opt.zero_grad(set_to_none=True)
        for bi, (x, mk, pos, yb, wb) in enumerate(dl):
            x, mk, pos, yb, wb = x.to(DEV), mk.to(DEV), pos.to(DEV), yb.to(DEV), wb.to(DEV)
            with AC():
                logits = model(x, mk, pos, RES, train=True)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), yb, reduction='none') * wb).mean()
            if scaler: scaler.scale(loss / ACC).backward()
            else: (loss / ACC).backward()
            eloss = eloss + loss.detach(); nb += 1
            if (bi + 1) % ACC == 0:
                set_lr(step)
                if scaler: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True); ema.update(model); step += 1
            mark()
            if (bi + 1) % 100 == 0: tlog(f'f{FOLD} ep{ep+1} {(bi+1)*BS}/{len(train_ids)} loss~{float(eloss)/nb:.4f} {(time.time()-t_ep):.0f}s')
        Pe = predict(ema.m, gold_ids)
        tlog(f'=== f{FOLD} EPOCH {ep+1}/{EPOCHS} loss={float(eloss)/max(1,nb):.4f} ({time.time()-t_ep:.0f}s) | gold-58 diag ema={macro(Pe, Yg):.4f}')
    del dl
    torch.save({'arch': 'kneenet_convnext_tiny', 'res': RES, 'k': K_EVAL, 'model': {k: v.detach().cpu() for k, v in ema.m.state_dict().items()}},
               f'/kaggle/working/{CFG["outtag"]}_f{FOLD}.pt')
    Ph = predict(ema.m, held_ids); Pg = predict(ema.m, gold_ids)
    oof = pd.DataFrame(np.vstack([Ph, Pg]), columns=LAB); oof.insert(0, 'StudyInstanceUID', held_ids + gold_ids)
    oof['is_gold'] = [0] * len(held_ids) + [1] * len(gold_ids); oof['fold'] = [FOLD] * len(held_ids) + [-1] * len(gold_ids)
    oof.to_csv(f'/kaggle/working/oof_{CFG["outtag"]}_f{FOLD}.csv', index=False)
    Th = FL.loc[held_ids, LAB].fillna(0.28).values.astype(float)
    cs = [np.corrcoef(Ph[:, j], Th[:, j])[0, 1] for j in range(N_T)]
    tlog(f'*** FOLD {FOLD} SELESAI | gold-58 (EMA) = {macro(Pg, Yg):.4f} | korelasi(OOF, label fusi) = {np.nanmean(cs):.4f}')
    del model, ema, opt
tlog('WORKER DONE')
'''
WORKER = WORKER.replace("exec(compile(open('/kaggle/working/_ke_src.py').read(), '<ke>', 'exec'), KE := {'__name__': '_ke'}) if False else None\n", "")
CFG = dict(dev=DEVKIND, res=RES, k=KWIN, epochs=EPOCHS, seed=SEED, nfold=NFOLD, bs=2, acc=4, k_eval=12, outtag=OUTTAG)

PARENT = r'''
import os, sys, time, json, subprocess, glob
t0 = time.time()
def tlog(m): print(f'[main t={time.time()-t0:.0f}s] {m}', flush=True)
os.makedirs('/kaggle/working', exist_ok=True)
open('/kaggle/working/_ke_src.py', 'w').write(KE_SRC); open('/kaggle/working/worker.py', 'w').write(WORKER_SRC)
if CFG['dev'] == 'gpu':
    import torch; NG = max(1, torch.cuda.device_count())
else:
    NG = 1
assign = {g: [f for i, f in enumerate(FOLDS) if i % NG == g] for g in range(NG)}
tlog(f'dev={CFG["dev"]} NG={NG} assign={assign}')
procs = {}
for g, fl in assign.items():
    if not fl: continue
    env = dict(os.environ, FOLDS=','.join(map(str, fl)), CFG=json.dumps(CFG), NW=('8' if CFG['dev'] == 'tpu' else '3'), PYTHONUNBUFFERED='1')
    if CFG['dev'] == 'gpu': env['CUDA_VISIBLE_DEVICES'] = str(g)
    else: env['PJRT_DEVICE'] = 'TPU'
    fh = open(f'/kaggle/working/worker{g}.log', 'w')
    procs[g] = (subprocess.Popen([sys.executable, '-u', '/kaggle/working/worker.py'], env=env, stdout=fh, stderr=subprocess.STDOUT), fh)
pos = {g: 0 for g in procs}
def pump():
    for g in procs:
        with open(f'/kaggle/working/worker{g}.log') as fh:
            fh.seek(pos[g]); new = fh.read(); pos[g] = fh.tell()
        for line in new.splitlines(): print(f'[g{g}] {line}', flush=True)
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(10); pump()
pump()
for g, (p, fh) in procs.items(): tlog(f'pekerja g{g} rc={p.returncode}')
tlog('SELESAI')
'''
cells = ["KE_SRC = " + repr(ke_src) + "\nWORKER_SRC = " + repr(WORKER) + f"\nCFG = {CFG!r}\nFOLDS = {fold_list!r}\n", PARENT]
for c in cells: compile(c, "<cell>", "exec")
compile(WORKER, "<worker>", "exec")
nb = {"cells": [{"cell_type": "code", "source": c.splitlines(keepends=True), "metadata": {}, "outputs": [], "execution_count": None} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(exist_ok=True)
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
accel = ({"enable_gpu": False, "enable_tpu": True} if DEVKIND == "tpu" else {"enable_gpu": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaT4"})
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": f"dimaspashaakrilian/{SLUG}", "title": "RSNA Knee " + TAG.replace("-", " "), "code_file": f"{SLUG}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_internet": False, **accel,
    "dataset_sources": ["dimaspashaakrilian/rsna-knee-fused-labels-r1", "inyalala/dl-vision-weights"],
    "kernel_sources": [f"dimaspashaakrilian/rsna-knee-volcache-w{i}" for i in range(3)],
    "competition_sources": ["rsna-knee-abnormality-detection"]}, indent=2))
print("built:", OUT.name, "| slug:", SLUG, "| dev:", DEVKIND, "| folds:", fold_list, "| epochs", EPOCHS, "| res", RES, "| k", KWIN)
