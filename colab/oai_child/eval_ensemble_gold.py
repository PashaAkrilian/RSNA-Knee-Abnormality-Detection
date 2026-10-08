# Dijalankan DI VM Colab (inferensi saja, GPU dibagi dgn latihan -> potongan 16): AUC gold-58 utk induk r384 (plain / +cermin / crop 320), sibling r224, anak c/e (ep1)
# dan campuran rank-mean-nya. Gold-58 = pembanding kasar (SE ~0,01); tujuan: sanity pipeline + arah efek ensembel.
import os, sys, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm, gc, itertools
t0 = time.time()
def tlog(m): print(f'[t={time.time()-t0:.0f}s] {m}', flush=True)
KE = {'__name__': '_ke'}; exec(compile(open('/content/ke_src.txt').read().replace('/kaggle/input', '/content/kin'), '<ke>', 'exec'), KE)
LAB = KE['LAB']; MEAN = KE['_MEAN']; STD = KE['_STD']
ROOT = '/content/kin/competitions/rsna-knee-abnormality-detection'
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
gold = train[train[LAB].notna().all(axis=1)].set_index('StudyInstanceUID'); Y = gold[LAB].values.astype(float); gids = list(gold.index)
MOUNT = {}
for d in glob.glob('/content/kin/*/*/volcache') + glob.glob('/content/kin/*/volcache'):
    for f in os.listdir(d):
        if f.endswith('.npz'): MOUNT[f[:-4]] = os.path.join(d, f)
gids = [g for g in gids if g in MOUNT]; Y = gold.loc[gids, LAB].values.astype(float)
tlog(f'gold-58 ter-cache: {len(gids)}')
NW = '/content/kin/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/'
CK = {'p': NW + 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt', 'r224': NW + 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt',
      'c1': '/content/kw/oai_child_c_ep1.pt', 'e1': '/content/kw/oai_child_e_ep1.pt'}
SLOT = [26, 22, 18, 12, 18]; bounds = np.cumsum([0] + SLOT)
def mk(path, img):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    bb = timm.create_model(ck.get('arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'), pretrained=False, num_classes=0, in_chans=3, global_pool='avg', img_size=img)
    m = KE['RaptorClassifier'](bb, F_dim=bb.num_features); m.load_state_dict(ck['model'], strict=True); del ck
    return m.eval().cuda()
def wins(vol, mask, crop, res, mirror):
    D = vol.shape[0]; cs = np.array([max(1, min(c, D - 2)) for c in KE['_eval_centers'](mask, D, 94)])
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)]))
    if mirror:
        slot = np.searchsorted(bounds, cs, side='right') - 1; sag = torch.from_numpy(np.isin(slot, (0, 1))).view(-1, 1, 1, 1)
        tri = torch.where(sag, tri.flip(1), tri.flip(-1))
    lo = (384 - crop) // 2; tri = tri[:, :, lo:lo + crop, lo:lo + crop].float() / 255.
    if crop != res: tri = F.interpolate(tri, size=(res, res), mode='bilinear', align_corners=False)
    return ((tri - MEAN) / STD)
@torch.no_grad()
def pred(m, x, ch=16):
    fs = []
    for c in x.split(ch):
        with torch.autocast('cuda', dtype=torch.float16): fs.append(m.backbone(c.cuda()).float())
    with torch.autocast('cuda', dtype=torch.float16): lg = m.head(torch.cat(fs, 0).unsqueeze(0)).float()
    return torch.sigmoid(lg)[0].cpu().numpy()
def auc1(y, s):
    r = pd.Series(s).rank().values; p = int(y.sum()); n = len(y) - p
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))
def macro(P):
    return float(np.mean([auc1(Y[:, j], P[:, j]) for j in range(12) if 0 < Y[:, j].sum() < len(Y)]))
def rk(P): return pd.DataFrame(P).rank(pct=True).values
CFG = [('p384', 'p', 384, 384, 384, False), ('p384m', 'p', 384, 384, 384, True), ('p320', 'p', 320, 320, 320, False), ('p320m', 'p', 320, 320, 320, True),
       ('r224', 'r224', 224, 307, 224, False), ('c1', 'c1', 384, 384, 384, False), ('e1', 'e1', 384, 384, 384, False)]
OUT = {}
for name, ck, img, crop, res, mir in CFG:
    if not os.path.exists(CK[ck]): tlog(f'lewati {name}: {CK[ck]} tidak ada'); continue
    m = mk(CK[ck], img); P = []
    for sid in gids:
        z = np.load(MOUNT[sid]); P.append(pred(m, wins(z['vol'], z['mask'], crop, res, mir)))
    OUT[name] = np.stack(P); del m; gc.collect(); torch.cuda.empty_cache()
    tlog(f'{name}: macro AUC gold-58 = {macro(OUT[name]):.4f}')
def blend(names, w=None):
    w = w or [1.0] * len(names); return sum(wi * rk(OUT[n]) for wi, n in zip(w, names)) / sum(w)
if 'p384' in OUT and 'p384m' in OUT: OUT['p384+mirror'] = 0.5 * (OUT['p384'] + OUT['p384m']); tlog(f'p384 plain+cermin (rata2 prob): {macro(OUT["p384+mirror"]):.4f}')
if 'p320' in OUT and 'p320m' in OUT: OUT['p320+mirror'] = 0.5 * (OUT['p320'] + OUT['p320m']); tlog(f'p320 plain+cermin (rata2 prob): {macro(OUT["p320+mirror"]):.4f}')
base = 'p320+mirror' if 'p320+mirror' in OUT else 'p384'
for other in ('r224', 'c1', 'e1'):
    if other in OUT:
        for w in (0.2, 0.35, 0.5):
            tlog(f'campur rank {base} + {other} (bobot {other} {w}): {macro(blend([base, other], [1 - w, w])):.4f}')
if all(k in OUT for k in ('r224', 'c1', 'e1')): tlog(f'campur {base}+r224+c1+e1 (.4/.2/.2/.2): {macro(blend([base, "r224", "c1", "e1"], [.4, .2, .2, .2])):.4f}')
np.savez('/content/kw/gold_preds_eval.npz', **{k: v for k, v in OUT.items()}, ids=np.array(gids))
tlog('SELESAI EVAL')
