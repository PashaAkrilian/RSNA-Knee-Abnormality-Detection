
import os, sys, time, glob, math, random, hashlib, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
t0 = time.time()
GTAG = os.environ.get('GTAG', '?')
def tlog(m): print(f'[g{GTAG} t={time.time()-t0:.0f}s] {m}', flush=True)

KE = {'__name__': '_ke'}
exec(compile(open('/content/colab_pkg/_ke_src.py').read(), '<ke>', 'exec'), KE)
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
    return [p for d in ('/content/kin/*/', '/content/kin/*/*/', '/content/kin/*/*/*/', '/content/kin/*/*/*/*/')
            for p in glob.glob(d + name)]

# ---------------- data ----------------
ROOT = KE['find_test_root']()
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv'); tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
HAS_SER = set(tser['StudyInstanceUID'])
GOLD = set(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in HAS_SER]
MOUNTED = {}
for d in (glob.glob('/content/kin/*/volcache') + glob.glob('/content/kin/*/*/volcache') + glob.glob('/content/kin/*/*/*/volcache')):
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
    for base in ('/content/kin/', '/content/kin/*/', '/content/kin/*/*/', '/content/kin/*/*/*/', '/content/kin/*/*/*/*/'):
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
               f'/content/kw/{OUTTAG}_f{FOLD}.pt')
    Ph = predict(ema_model, held_ids); Pg = predict(ema_model, gold_ids)
    oof = pd.DataFrame(np.vstack([Ph, Pg]), columns=LAB)
    oof.insert(0, 'StudyInstanceUID', held_ids + gold_ids)
    oof['is_gold'] = [0] * len(held_ids) + [1] * len(gold_ids); oof['fold'] = [FOLD] * len(held_ids) + [-1] * len(gold_ids)
    oof.to_csv(f'/content/kw/oof_{OUTTAG}_f{FOLD}.csv', index=False)
    Th = FL.loc[held_ids, LAB].fillna(0.28).values.astype(float)
    cs = [np.corrcoef(Ph[:, j], Th[:, j])[0, 1] for j in range(N_T)]
    tlog(f'*** FOLD {FOLD} SELESAI | gold-58 (EMA) = {macro(Pg, Yg):.4f} | korelasi(OOF, label fusi) rata2 = {np.nanmean(cs):.4f}')
    del model, ema_model, opt, ema; torch.cuda.empty_cache()
tlog('WORKER DONE')
