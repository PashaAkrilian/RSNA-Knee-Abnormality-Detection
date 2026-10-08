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
    hits = glob.glob('/content/kin/**/timm-*.whl', recursive=True)
    assert hits, 'wheel timm tidak ketemu'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--no-deps', hits[0]], check=True)
assert timm_ok(), 'timm tidak siap'
tlog('setup OK')

_KE_SRC = 'import os, glob, time, gc\nos.environ.setdefault(\'HF_HUB_OFFLINE\', \'1\')\nos.environ.setdefault(\'TRANSFORMERS_OFFLINE\', \'1\')\nos.environ.setdefault(\'HF_HUB_DISABLE_TELEMETRY\', \'1\')\nimport numpy as np\nimport torch, torch.nn as nn, torch.nn.functional as F\nimport timm\ntorch.backends.cudnn.benchmark = True\ntorch.backends.cuda.matmul.allow_tf32 = True\nIMG = 336\nCROP_MM = 140.0\nSLOTS = [(\'Sagittal\', 1, 18), (\'Sagittal\', 0, 14), (\'Coronal\', 1, 12), (\'Coronal\', 0, 8), (\'Axial\', -1, 12)]\nMAXS = sum((s[2] for s in SLOTS))\nK_EVAL = 42\nNORM = \'imagenet\'\nLAB = [\'ACL\', \'MCL\', \'Medial Meniscus\', \'Lateral Meniscus\', \'Medial OA\', \'Lateral OA\', \'PF OA\', \'Effusion\', \'Synovitis\', "Baker\'s", \'Contusion\', \'Fracture\']\n_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)\n_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)\nARMS = [{\'file\': \'raptor_ft_coatnet_v5_full_swa.pt\', \'arch\': \'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k\', \'res\': 384, \'w\': 1.0}]\n\ndef build_backbone(arch, pretrained=False):\n    hybrid = arch.startswith((\'maxvit\', \'maxxvit\', \'coatnet\', \'coat_\', \'convnext\'))\n    is_vit = not hybrid and any((k in arch for k in (\'vit\', \'deit\', \'dinov2\', \'eva\', \'beit\')))\n    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)\n    if is_vit:\n        kw.update(global_pool=\'token\', dynamic_img_size=True)\n    else:\n        kw.update(global_pool=\'avg\')\n    return timm.create_model(arch, **kw)\n\nclass RaptorClassifier(nn.Module):\n\n    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):\n        super().__init__()\n        self.backbone = backbone\n        self.norm = nn.LayerNorm(F_dim)\n        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))\n        self.clsW = nn.Parameter(torch.zeros(n, F_dim))\n        self.clsb = nn.Parameter(torch.zeros(n))\n        nn.init.trunc_normal_(self.clsW, std=0.02)\n        self.n = n\n\n    def encode(self, x):\n        B, K = x.shape[:2]\n        f = self.backbone(x.flatten(0, 1))\n        return f.view(B, K, -1)\n\n    def head(self, feats):\n        h = self.norm(feats)\n        a = self.att(h)\n        a = torch.softmax(a, dim=1)\n        pooled = torch.einsum(\'bkn,bkf->bnf\', a, h)\n        logits = (pooled * self.clsW).sum(-1) + self.clsb\n        return logits\n\n    def forward(self, x):\n        return self.head(self.encode(x))\n\ndef load_model(pt_path, arch_default, res_default, device, ngpu=1):\n    ck = torch.load(pt_path, map_location=\'cpu\', weights_only=False)\n    arch = ck.get(\'arch\', arch_default)\n    ck_res = int(ck.get(\'res\', res_default))\n    bb = build_backbone(arch, pretrained=False)\n    model = RaptorClassifier(bb, F_dim=bb.num_features)\n    model.load_state_dict(ck[\'model\'], strict=True)\n    model.eval().to(device)\n    del ck\n    gc.collect()\n    return (model, ck_res)\n\ndef _eval_centers(mask, D, k):\n    valid = np.where(mask > 0)[0]\n    if len(valid) < 3:\n        valid = np.arange(min(3, D))\n    lo, hi = (int(valid.min()), int(valid.max()))\n    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]\n    if not cs:\n        cs = [max(1, min((lo + hi) // 2, D - 2))]\n    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)\n    return [cs[i] for i in idx]\n\ndef eval_windows(vol, mask, k, res, norm=NORM):\n    D = vol.shape[0]\n    cs = _eval_centers(mask, D, k)\n    wins = np.empty((len(cs), 3, res, res), np.float32)\n    for j, c in enumerate(cs):\n        c = max(1, min(c, D - 2))\n        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0\n        t = torch.from_numpy(tri)\n        if t.shape[-1] != res:\n            t = F.interpolate(t[None], size=(res, res), mode=\'bilinear\', align_corners=False)[0]\n        wins[j] = t.numpy()\n    x = torch.from_numpy(wins)\n    if norm == \'imagenet\':\n        x = (x - _MEAN) / _STD\n    return x\n\n@torch.no_grad()\ndef infer_probs(model, xwins, device):\n    x = xwins.unsqueeze(0).to(device)\n    use_cuda = device != \'cpu\' and str(device).startswith(\'cuda\')\n    if use_cuda:\n        try:\n            with torch.autocast(\'cuda\', dtype=torch.float16):\n                o = torch.sigmoid(model(x).float())\n            return o[0].cpu().numpy()\n        except RuntimeError:\n            torch.cuda.empty_cache()\n            o = torch.sigmoid(model(x).float())\n            return o[0].cpu().numpy()\n    o = torch.sigmoid(model(x).float())\n    return o[0].cpu().numpy()\n\ndef rankpct(x):\n    order = x.argsort(0).argsort(0).astype(np.float64)\n    return order / max(1, x.shape[0] - 1)\n\ndef _make_reader():\n    import pydicom, cv2\n    from pydicom.pixel_data_handlers.util import apply_modality_lut\n\n    def order_and_meta(sdir):\n        fs = glob.glob(sdir + \'/*.dcm\')\n        recs = []\n        ps_list = []\n        for f in fs:\n            try:\n                h = pydicom.dcmread(f, stop_before_pixels=True)\n                iop = getattr(h, \'ImageOrientationPatient\', None)\n                ipp = getattr(h, \'ImagePositionPatient\', None)\n                if iop is not None and ipp is not None and (len(iop) == 6):\n                    r = np.array(iop[:3], float)\n                    c = np.array(iop[3:], float)\n                    n = np.cross(r, c)\n                    pos = float(np.dot(np.array(ipp, float), n))\n                else:\n                    pos = float(getattr(h, \'InstanceNumber\', 0) or 0)\n                ps = getattr(h, \'PixelSpacing\', None)\n                ps = float(ps[0]) if ps is not None else 0.5\n                ps_list.append(ps)\n                recs.append((pos, f, ps))\n            except Exception:\n                recs.append((0.0, f, 0.5))\n        recs.sort(key=lambda x: x[0])\n        med_ps = float(np.median(ps_list)) if ps_list else 0.5\n        return ([(f, ps) for _, f, ps in recs], med_ps)\n\n    def read_px(f):\n        d = pydicom.dcmread(f)\n        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)\n        if str(getattr(d, \'PhotometricInterpretation\', \'\')) == \'MONOCHROME1\':\n            a = a.max() - a\n        return a\n\n    def mm_crop_resize(a, ps):\n        h, w = a.shape\n        cpx = int(round(CROP_MM / max(ps, 0.001)))\n        cpx = min(cpx, min(h, w))\n        y0 = (h - cpx) // 2\n        x0 = (w - cpx) // 2\n        a = a[y0:y0 + cpx, x0:x0 + cpx]\n        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)\n    return (order_and_meta, read_px, mm_crop_resize)\n\ndef _pick_series_for_slot(rows, plane, fluid, used):\n    cands = [r for r in rows if r[\'Anatomical_Plane\'] == plane and r[\'SeriesInstanceUID\'] not in used]\n    if fluid in (0, 1):\n        pref = [r for r in cands if int(r.get(\'Fluid_Sensitive\', 0) or 0) == fluid]\n        if pref:\n            return pref[0]\n    return cands[0] if cands else None\n\ndef build_study(sid, ser_records, tsdir, reader):\n    order_and_meta, read_px, mm_crop_resize = reader\n    rows = ser_records.get(sid, [])\n    vol = np.zeros((MAXS, IMG, IMG), np.uint8)\n    idx = 0\n    used = set()\n    for plane, fluid, k in SLOTS:\n        r = _pick_series_for_slot(rows, plane, fluid, used)\n        if r is None:\n            idx += k\n            continue\n        used.add(r[\'SeriesInstanceUID\'])\n        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r[\'SeriesInstanceUID\']}")\n        if not files:\n            idx += k\n            continue\n        n = len(files)\n        lo, hi = (int(n * 0.06), int(n * 0.94) - 1)\n        hi = max(hi, lo)\n        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k\n        arrs = []\n        pss = []\n        for p in picks:\n            fp, ps = files[min(p, n - 1)]\n            try:\n                arrs.append(read_px(fp))\n                pss.append(ps)\n            except Exception:\n                arrs.append(None)\n                pss.append(med_ps)\n        valid = [a for a in arrs if a is not None]\n        if valid:\n            allpx = np.concatenate([a.ravel() for a in valid])\n            loq, hiq = np.percentile(allpx, [2.0, 98.0])\n        else:\n            loq, hiq = (0.0, 1.0)\n        for a, ps in zip(arrs, pss):\n            if idx >= MAXS:\n                break\n            if a is None:\n                idx += 1\n                continue\n            aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)\n            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)\n            vol[idx] = (aw * 255).astype(np.uint8)\n            idx += 1\n        if idx >= MAXS:\n            break\n    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)\n    return (vol, mask)\n\ndef find_test_root():\n    cands = [\'/content/kin/competitions/rsna-knee-abnormality-detection\', \'/content/kin/rsna-knee-abnormality-detection\']\n    for b in cands:\n        if os.path.exists(b + \'/test.csv\'):\n            return b\n    for d, _, f in os.walk(\'/content/kin\'):\n        if \'test.csv\' in f and (os.path.isdir(d + \'/test_series\') or os.path.isdir(d + \'/test_images\')):\n            return d\n    for d, _, f in os.walk(\'/content/kin\'):\n        if \'test.csv\' in f:\n            return d\n    raise RuntimeError(\'no test root under /content/kin\')\n\ndef find_weight_file(fname):\n    direct = [f\'/content/kin/raptor-knee-arms/{fname}\', f\'/content/kin/raptor-knee-arms/1/{fname}\', f\'/content/kin/raptor-cnn336/{fname}\']\n    for p in direct:\n        if os.path.exists(p):\n            return p\n    for d in sorted(glob.glob(\'/content/kin/*/\')):\n        if \'competition\' in d.lower():\n            continue\n        hits = glob.glob(os.path.join(d, \'**\', fname), recursive=True)\n        if hits:\n            return hits[0]\n    raise RuntimeError(f\'{fname} not found under /content/kin\')\n\ndef main():\n    import pandas as pd\n    t0 = time.time()\n    dev = \'cuda\' if torch.cuda.is_available() else \'cpu\'\n    ngpu = torch.cuda.device_count()\n    print(f\'device {dev} | gpus {ngpu} | torch {torch.__version__}\', flush=True)\n    ROOT = find_test_root()\n    tsdir = ROOT + \'/test_series\'\n    if not os.path.isdir(tsdir):\n        tsdir = ROOT + \'/test_images\'\n    print(\'test root:\', ROOT, \'| series dir:\', tsdir, flush=True)\n    test = pd.read_csv(ROOT + \'/test.csv\')\n    test[\'StudyInstanceUID\'] = test[\'StudyInstanceUID\'].astype(str)\n    test_ids = test[\'StudyInstanceUID\'].tolist()\n    tser = pd.read_csv(ROOT + \'/test_series.csv\')\n    tser[\'StudyInstanceUID\'] = tser[\'StudyInstanceUID\'].astype(str)\n    tser[\'SeriesInstanceUID\'] = tser[\'SeriesInstanceUID\'].astype(str)\n    SER = {k: v.to_dict(\'records\') for k, v in tser.groupby(\'StudyInstanceUID\')}\n    print(f\'test studies {len(test_ids)} | test series {len(tser)}\', flush=True)\n    sub_cols = [\'StudyInstanceUID\'] + LAB\n    ssub = os.path.join(ROOT, \'sample_submission.csv\')\n    if os.path.exists(ssub):\n        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)\n    reader = _make_reader()\n    N = len(test_ids)\n    A = len(ARMS)\n    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]\n    for a, arm in enumerate(ARMS):\n        wp = find_weight_file(arm[\'file\'])\n        model, res = load_model(wp, arm[\'arch\'], arm[\'res\'], dev)\n        print(f"[arm {a}] loaded {arm[\'file\']} | res {res} | {time.time() - t0:.0f}s", flush=True)\n        for i, sid in enumerate(test_ids):\n            try:\n                vol, mask = build_study(sid, SER, tsdir, reader)\n                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)\n                arm_probs[a][i] = infer_probs(model, xw, dev)\n                del vol, mask, xw\n            except Exception as e:\n                print(f\'  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})\', flush=True)\n            if (i + 1) % 100 == 0 or i + 1 == N:\n                print(f\'  [arm {a}] {i + 1}/{N} | {time.time() - t0:.0f}s\', flush=True)\n        del model\n        gc.collect()\n        if str(dev).startswith(\'cuda\'):\n            torch.cuda.empty_cache()\n        print(f\'[arm {a}] done + freed | {time.time() - t0:.0f}s\', flush=True)\n    _w = np.array([float(a.get(\'w\', 1.0)) for a in ARMS], dtype=np.float64)\n    _w = _w / _w.sum()\n    print(f"[blend] weighted rank-mean w={dict(zip([a[\'file\'] for a in ARMS], _w.round(4)))}", flush=True)\n    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]), axes=(0, 0))\n    if not np.isfinite(ranks).all():\n        ranks[~np.isfinite(ranks)] = 0.5\n    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)\n    sub.insert(0, \'StudyInstanceUID\', test_ids)\n    sub = sub[sub_cols]\n    assert list(sub.columns) == sub_cols, \'column order drift\'\n    assert sub[\'StudyInstanceUID\'].tolist() == test_ids, \'row identity drift\'\n    assert np.isfinite(sub[LAB].values).all()\n    out = \'/content/kw/_raptor.csv\'\n    sub.to_csv(out, index=False)\n    print(\'wrote\', out, \'|\', len(sub), \'rows x\', len(sub.columns), \'cols\', flush=True)\n    print(sub.head().to_string(index=False), flush=True)\n    print(f\'DONE {time.time() - t0:.0f}s\', flush=True)'
_KE_NS = {'__name__': '_ke'}
exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)
print('_KE loaded', flush=True)
import numpy as np, pandas as pd, os, json, glob as _glob, copy, math, random, contextlib, gc
import torch, torch.nn as nn, torch.nn.functional as F
xm = type('XM', (), {'mark_step': staticmethod(lambda: None)})()
from concurrent.futures import ThreadPoolExecutor
KE = _KE_NS
SCALER = None
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LAB = KE['LAB']; N_T = len(LAB)
ROOT = KE['find_test_root']()
train = pd.read_csv(ROOT + '/train.csv'); train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
tser = pd.read_csv(ROOT + '/train_series.csv'); tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
SERIDS = set(tser['StudyInstanceUID'])
GOLD = set(train.loc[train[LAB].notna().all(axis=1), 'StudyInstanceUID'])
ALL_IDS = [s for s in train['StudyInstanceUID'] if s in SERIDS]

def _bglob(name):
    return [p for d in ('/content/kin/*/', '/content/kin/*/*/', '/content/kin/*/*/*/', '/content/kin/*/*/*/*/')
            for p in _glob.glob(d + name)]
MOUNTED = {}
for _d in (_glob.glob('/content/kin/*/volcache') + _glob.glob('/content/kin/*/*/volcache')
           + _glob.glob('/content/kin/*/*/*/volcache')):          # kedalaman terbatas: JANGAN '**' (820rb berkas kompetisi)
    if os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.endswith('.npz'):
                MOUNTED[_f[:-4]] = os.path.join(_d, _f)
tlog(f'cache z termount: {len(MOUNTED)} studi | gold={len(GOLD)} | studi={len(ALL_IDS)}')
assert len(MOUNTED) > (100 if os.environ.get('SMOKE') else 4000), 'cache z tidak lengkap'

_hits = _bglob('fused_labels_r1.csv'); assert _hits, 'fused_labels_r1.csv tidak ketemu'
FL = pd.read_csv(_hits[0]); FL['StudyInstanceUID'] = FL['StudyInstanceUID'].astype(str); FL = FL.set_index('StudyInstanceUID')
CONF_COLS = {t: f'{t}__conf' for t in LAB}
train_ids = [s for s in ALL_IDS if s not in GOLD and s in FL.index and s in MOUNTED]
tlog(f'studi latih (non-gold, berlabel fusi, ter-cache): {len(train_ids)}')
if os.environ.get('SMOKE'): train_ids = train_ids[:int(os.environ['SMOKE'])]; tlog(f'SMOKE: {len(train_ids)} studi latih')
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

if torch.cuda.is_available():
    AC = lambda: torch.autocast('cuda', dtype=torch.float16)
    try: SCALER = torch.amp.GradScaler('cuda')
    except Exception: SCALER = torch.cuda.amp.GradScaler()
    tlog('autocast cuda fp16 + GradScaler')
else:
    AC = lambda: contextlib.nullcontext()
    try: SCALER = torch.amp.GradScaler('cuda', enabled=False)
    except Exception: SCALER = torch.cuda.amp.GradScaler(enabled=False)
    tlog('MODE CPU fp32 (smoke)')

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
if os.environ.get('SMOKE'): gids = gids[:int(os.environ.get('SMOKE_GOLD', '4'))]
EVAL_K = int(os.environ.get('EVAL_K', '94')); EVAL_CH = max(1, EVAL_K // 2)
YG = train.set_index('StudyInstanceUID').loc[gids, LAB].values.astype(float)
def eval_gold(m, tag):
    m.eval(); P = []
    for sid in gids:
        vol, mask = load_study(sid)
        xw = KE['eval_windows'](vol, mask, k=EVAL_K, res=RES, norm=KE['NORM'])     # (94,3,384,384) fp32
        P.append(predict(m, xw, EVAL_CH))
    P = np.stack(P)
    a = [auc1(YG[:, j], P[:, j]) for j in range(N_T) if 0 < YG[:, j].sum() < len(YG)]
    tlog(f'[gold-58 | {tag}] macro AUC={np.mean(a):.4f} (TERKONTAMINASI: induk dilatih pada gold; hanya utk memantau drift)')
    return P, float(np.mean(a))

K_WIN = int(os.environ.get('K_WIN', '12'))
EPOCHS = int(os.environ.get('EPOCHS', '3'))
ACC_STUDIES = 8
LR_BB, LR_HEAD, WD, WARMUP, LRQ = float(os.environ.get('LR_BB', '1e-5')), float(os.environ.get('LR_HEAD', '1e-4')), float(os.environ.get('WD', '0.01')), int(os.environ.get('WARMUP', '100')), 30
VARIANTS = json.loads(os.environ['VARIANTS'])   # (tag, lambda_KD, seed)

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
                SCALER.scale(loss / ACC_STUDIES).backward()
                xm.mark_step()
                ep_l += float(loss_y.detach())
            except Exception as e:
                tlog(f'GAGAL studi idx {ii}: {type(e).__name__}: {str(e)[:140]} -- dilewati'); xm.mark_step(); continue
            nacc += 1
            if nacc % ACC_STUDIES == 0 or ii == len(order) - 1:
                set_lr(step); SCALER.step(opt); SCALER.update(); opt.zero_grad(set_to_none=True); step += 1
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
                       f'/content/kw/oai_child_{tag}_ep{ep+1}.pt')
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
               f'/content/kw/oai_child_{tag}_final.pt')
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
                   f'/content/kw/oai_child_{tag}_swa.pt')
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

if os.environ.get('DO_PARENT_SOFT', '0') != '1':
    tlog('SELESAI (tanpa parent_soft)'); raise SystemExit(0)
tlog('prediksi induk pada semua studi latih ...')
import time as _tm
t1 = _tm.time(); rows = []
allids = [s for s in ALL_IDS if s in MOUNTED]
if os.environ.get('SMOKE'): allids = allids[:8]
for i, sid in enumerate(allids):
    try:
        vol, mask = load_study(sid)
        xw = KE['eval_windows'](vol, mask, k=16, res=RES, norm=KE['NORM'])
        rows.append([sid] + [float(x) for x in predict(parent, xw, 16)])
    except Exception as e:
        tlog(f'gagal {sid[:16]}: {type(e).__name__}')
    if (i + 1) % 500 == 0: tlog(f'induk-soft {i+1}/{len(allids)} ({_tm.time()-t1:.0f}s)')
    if _tm.time() - t1 > 2700: tlog('batas waktu 45 mnt, berhenti'); break
pd.DataFrame(rows, columns=['StudyInstanceUID'] + LAB).to_csv('/content/kw/parent_soft_k16.csv', index=False)
tlog('TRAINING DONE')
