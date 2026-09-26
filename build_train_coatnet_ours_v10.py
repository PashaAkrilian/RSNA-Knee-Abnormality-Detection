#!/usr/bin/env python3
"""Build rsna-knee-train-coatnet-v3-rawmean: fine-tune PENUH CoAtNet resep dread/c5
di atas label BLENDED RAW-PROBABILITY-MEAN flight0234+Claude (§46 PROGRESS.md, alpha=0.65-0.70,
gold-58 label-matching macro=0,9033-0,9038), confidence-weighted via 2*|p-0.5|. gold-58 TIDAK
ikut latih, evaluasi akhir di gold-58 -> pembanding jujur utk c5 (0.9206), v1 flight0234-only
(0.9193, §38), DAN v2 rank-mean blend (0.9170, REGRESI -- §46: rank-mean merusak kalibrasi
target BCE dgn memaksa tiap kolom label jadi distribusi uniform, buang skew/base-rate asli).
v3 ganti rank-mean -> raw-probability-mean (weighted average nilai mentah, tanpa transformasi
rank) supaya kalibrasi/skew asli tiap label tetap terjaga sbg target training.

Satu-satunya beda dari build_train_coatnet_v2_blended.py (v2): sumber label CSV (rank-mean ->
raw-mean) & dataset_sources. Semua komponen model/preprocess/training loop identik (masih
VERBATIM dari _KE_SRC), jadi checkpoint hasilnya tetap kompatibel dgn pipeline inferensi
Raptor/DINOsaur.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "kaggle_kernel_rsna_knee_train_coatnet_ours_v10"
# v2 (26 Agustus): sumber asli /tmp/opencode/nb/... sudah hilang (ephemeral, beda sesi) --
# ke_src sekarang disimpan sbg file terpisah yg DI-COMMIT ke repo (bukan /tmp), supaya builder
# ini reproducible lintas sesi. Isinya sama persis (di-extract dari notebook v1 yg sudah pernah
# di-push), bukan sumber baru.
SRC_TXT = REPO / "build_train_coatnet_ke_src.txt"

# --- geometri harness disesuaikan varian v10 (img=384, k_eval=62) ---
_KE_SRC_PATCHED = 'import os, glob, time, gc\nos.environ.setdefault(\'HF_HUB_OFFLINE\', \'1\')\nos.environ.setdefault(\'TRANSFORMERS_OFFLINE\', \'1\')\nos.environ.setdefault(\'HF_HUB_DISABLE_TELEMETRY\', \'1\')\nimport numpy as np\nimport torch, torch.nn as nn, torch.nn.functional as F\nimport timm\ntorch.backends.cudnn.benchmark = True\ntorch.backends.cuda.matmul.allow_tf32 = True\nIMG = 384\nCROP_MM = 140.0\nSLOTS = [(\'Sagittal\', 1, 18), (\'Sagittal\', 0, 14), (\'Coronal\', 1, 12), (\'Coronal\', 0, 8), (\'Axial\', -1, 12)]\nMAXS = sum((s[2] for s in SLOTS))\nK_EVAL = 62\nNORM = \'imagenet\'\nLAB = [\'ACL\', \'MCL\', \'Medial Meniscus\', \'Lateral Meniscus\', \'Medial OA\', \'Lateral OA\', \'PF OA\', \'Effusion\', \'Synovitis\', "Baker\'s", \'Contusion\', \'Fracture\']\n_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)\n_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)\nARMS = [{\'file\': \'raptor_ft_coatnet_v5_full_swa.pt\', \'arch\': \'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k\', \'res\': 384, \'w\': 1.0}]\n\ndef build_backbone(arch, pretrained=False):\n    hybrid = arch.startswith((\'maxvit\', \'maxxvit\', \'coatnet\', \'coat_\', \'convnext\'))\n    is_vit = not hybrid and any((k in arch for k in (\'vit\', \'deit\', \'dinov2\', \'eva\', \'beit\')))\n    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)\n    if is_vit:\n        kw.update(global_pool=\'token\', dynamic_img_size=True)\n    else:\n        kw.update(global_pool=\'avg\')\n    return timm.create_model(arch, **kw)\n\nclass RaptorClassifier(nn.Module):\n\n    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):\n        super().__init__()\n        self.backbone = backbone\n        self.norm = nn.LayerNorm(F_dim)\n        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))\n        self.clsW = nn.Parameter(torch.zeros(n, F_dim))\n        self.clsb = nn.Parameter(torch.zeros(n))\n        nn.init.trunc_normal_(self.clsW, std=0.02)\n        self.n = n\n\n    def encode(self, x):\n        B, K = x.shape[:2]\n        f = self.backbone(x.flatten(0, 1))\n        return f.view(B, K, -1)\n\n    def head(self, feats):\n        h = self.norm(feats)\n        a = self.att(h)\n        a = torch.softmax(a, dim=1)\n        pooled = torch.einsum(\'bkn,bkf->bnf\', a, h)\n        logits = (pooled * self.clsW).sum(-1) + self.clsb\n        return logits\n\n    def forward(self, x):\n        return self.head(self.encode(x))\n\ndef load_model(pt_path, arch_default, res_default, device, ngpu=1):\n    ck = torch.load(pt_path, map_location=\'cpu\', weights_only=False)\n    arch = ck.get(\'arch\', arch_default)\n    ck_res = int(ck.get(\'res\', res_default))\n    bb = build_backbone(arch, pretrained=False)\n    model = RaptorClassifier(bb, F_dim=bb.num_features)\n    model.load_state_dict(ck[\'model\'], strict=True)\n    model.eval().to(device)\n    del ck\n    gc.collect()\n    return (model, ck_res)\n\ndef _eval_centers(mask, D, k):\n    valid = np.where(mask > 0)[0]\n    if len(valid) < 3:\n        valid = np.arange(min(3, D))\n    lo, hi = (int(valid.min()), int(valid.max()))\n    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]\n    if not cs:\n        cs = [max(1, min((lo + hi) // 2, D - 2))]\n    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)\n    return [cs[i] for i in idx]\n\ndef eval_windows(vol, mask, k, res, norm=NORM):\n    D = vol.shape[0]\n    cs = _eval_centers(mask, D, k)\n    wins = np.empty((len(cs), 3, res, res), np.float32)\n    for j, c in enumerate(cs):\n        c = max(1, min(c, D - 2))\n        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0\n        t = torch.from_numpy(tri)\n        if t.shape[-1] != res:\n            t = F.interpolate(t[None], size=(res, res), mode=\'bilinear\', align_corners=False)[0]\n        wins[j] = t.numpy()\n    x = torch.from_numpy(wins)\n    if norm == \'imagenet\':\n        x = (x - _MEAN) / _STD\n    return x\n\n@torch.no_grad()\ndef infer_probs(model, xwins, device):\n    x = xwins.unsqueeze(0).to(device)\n    use_cuda = device != \'cpu\' and str(device).startswith(\'cuda\')\n    if use_cuda:\n        try:\n            with torch.autocast(\'cuda\', dtype=torch.float16):\n                o = torch.sigmoid(model(x).float())\n            return o[0].cpu().numpy()\n        except RuntimeError:\n            torch.cuda.empty_cache()\n            o = torch.sigmoid(model(x).float())\n            return o[0].cpu().numpy()\n    o = torch.sigmoid(model(x).float())\n    return o[0].cpu().numpy()\n\ndef rankpct(x):\n    order = x.argsort(0).argsort(0).astype(np.float64)\n    return order / max(1, x.shape[0] - 1)\n\ndef _make_reader():\n    import pydicom, cv2\n    from pydicom.pixel_data_handlers.util import apply_modality_lut\n\n    def order_and_meta(sdir):\n        fs = glob.glob(sdir + \'/*.dcm\')\n        recs = []\n        ps_list = []\n        for f in fs:\n            try:\n                h = pydicom.dcmread(f, stop_before_pixels=True)\n                iop = getattr(h, \'ImageOrientationPatient\', None)\n                ipp = getattr(h, \'ImagePositionPatient\', None)\n                if iop is not None and ipp is not None and (len(iop) == 6):\n                    r = np.array(iop[:3], float)\n                    c = np.array(iop[3:], float)\n                    n = np.cross(r, c)\n                    pos = float(np.dot(np.array(ipp, float), n))\n                else:\n                    pos = float(getattr(h, \'InstanceNumber\', 0) or 0)\n                ps = getattr(h, \'PixelSpacing\', None)\n                ps = float(ps[0]) if ps is not None else 0.5\n                ps_list.append(ps)\n                recs.append((pos, f, ps))\n            except Exception:\n                recs.append((0.0, f, 0.5))\n        recs.sort(key=lambda x: x[0])\n        med_ps = float(np.median(ps_list)) if ps_list else 0.5\n        return ([(f, ps) for _, f, ps in recs], med_ps)\n\n    def read_px(f):\n        d = pydicom.dcmread(f)\n        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)\n        if str(getattr(d, \'PhotometricInterpretation\', \'\')) == \'MONOCHROME1\':\n            a = a.max() - a\n        return a\n\n    def mm_crop_resize(a, ps):\n        h, w = a.shape\n        cpx = int(round(CROP_MM / max(ps, 0.001)))\n        cpx = min(cpx, min(h, w))\n        y0 = (h - cpx) // 2\n        x0 = (w - cpx) // 2\n        a = a[y0:y0 + cpx, x0:x0 + cpx]\n        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)\n    return (order_and_meta, read_px, mm_crop_resize)\n\ndef _pick_series_for_slot(rows, plane, fluid, used):\n    cands = [r for r in rows if r[\'Anatomical_Plane\'] == plane and r[\'SeriesInstanceUID\'] not in used]\n    if fluid in (0, 1):\n        pref = [r for r in cands if int(r.get(\'Fluid_Sensitive\', 0) or 0) == fluid]\n        if pref:\n            return pref[0]\n    return cands[0] if cands else None\n\ndef build_study(sid, ser_records, tsdir, reader):\n    order_and_meta, read_px, mm_crop_resize = reader\n    rows = ser_records.get(sid, [])\n    vol = np.zeros((MAXS, IMG, IMG), np.uint8)\n    idx = 0\n    used = set()\n    for plane, fluid, k in SLOTS:\n        r = _pick_series_for_slot(rows, plane, fluid, used)\n        if r is None:\n            idx += k\n            continue\n        used.add(r[\'SeriesInstanceUID\'])\n        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r[\'SeriesInstanceUID\']}")\n        if not files:\n            idx += k\n            continue\n        n = len(files)\n        lo, hi = (int(n * 0.06), int(n * 0.94) - 1)\n        hi = max(hi, lo)\n        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k\n        arrs = []\n        pss = []\n        for p in picks:\n            fp, ps = files[min(p, n - 1)]\n            try:\n                arrs.append(read_px(fp))\n                pss.append(ps)\n            except Exception:\n                arrs.append(None)\n                pss.append(med_ps)\n        valid = [a for a in arrs if a is not None]\n        if valid:\n            allpx = np.concatenate([a.ravel() for a in valid])\n            loq, hiq = np.percentile(allpx, [2.0, 98.0])\n        else:\n            loq, hiq = (0.0, 1.0)\n        for a, ps in zip(arrs, pss):\n            if idx >= MAXS:\n                break\n            if a is None:\n                idx += 1\n                continue\n            aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)\n            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)\n            vol[idx] = (aw * 255).astype(np.uint8)\n            idx += 1\n        if idx >= MAXS:\n            break\n    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)\n    return (vol, mask)\n\ndef find_test_root():\n    cands = [\'/kaggle/input/competitions/rsna-knee-abnormality-detection\', \'/kaggle/input/rsna-knee-abnormality-detection\']\n    for b in cands:\n        if os.path.exists(b + \'/test.csv\'):\n            return b\n    for d, _, f in os.walk(\'/kaggle/input\'):\n        if \'test.csv\' in f and (os.path.isdir(d + \'/test_series\') or os.path.isdir(d + \'/test_images\')):\n            return d\n    for d, _, f in os.walk(\'/kaggle/input\'):\n        if \'test.csv\' in f:\n            return d\n    raise RuntimeError(\'no test root under /kaggle/input\')\n\ndef find_weight_file(fname):\n    direct = [f\'/kaggle/input/raptor-knee-arms/{fname}\', f\'/kaggle/input/raptor-knee-arms/1/{fname}\', f\'/kaggle/input/raptor-cnn336/{fname}\']\n    for p in direct:\n        if os.path.exists(p):\n            return p\n    for d in sorted(glob.glob(\'/kaggle/input/*/\')):\n        if \'competition\' in d.lower():\n            continue\n        hits = glob.glob(os.path.join(d, \'**\', fname), recursive=True)\n        if hits:\n            return hits[0]\n    raise RuntimeError(f\'{fname} not found under /kaggle/input\')\n\ndef main():\n    import pandas as pd\n    t0 = time.time()\n    dev = \'cuda\' if torch.cuda.is_available() else \'cpu\'\n    ngpu = torch.cuda.device_count()\n    print(f\'device {dev} | gpus {ngpu} | torch {torch.__version__}\', flush=True)\n    ROOT = find_test_root()\n    tsdir = ROOT + \'/test_series\'\n    if not os.path.isdir(tsdir):\n        tsdir = ROOT + \'/test_images\'\n    print(\'test root:\', ROOT, \'| series dir:\', tsdir, flush=True)\n    test = pd.read_csv(ROOT + \'/test.csv\')\n    test[\'StudyInstanceUID\'] = test[\'StudyInstanceUID\'].astype(str)\n    test_ids = test[\'StudyInstanceUID\'].tolist()\n    tser = pd.read_csv(ROOT + \'/test_series.csv\')\n    tser[\'StudyInstanceUID\'] = tser[\'StudyInstanceUID\'].astype(str)\n    tser[\'SeriesInstanceUID\'] = tser[\'SeriesInstanceUID\'].astype(str)\n    SER = {k: v.to_dict(\'records\') for k, v in tser.groupby(\'StudyInstanceUID\')}\n    print(f\'test studies {len(test_ids)} | test series {len(tser)}\', flush=True)\n    sub_cols = [\'StudyInstanceUID\'] + LAB\n    ssub = os.path.join(ROOT, \'sample_submission.csv\')\n    if os.path.exists(ssub):\n        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)\n    reader = _make_reader()\n    N = len(test_ids)\n    A = len(ARMS)\n    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]\n    for a, arm in enumerate(ARMS):\n        wp = find_weight_file(arm[\'file\'])\n        model, res = load_model(wp, arm[\'arch\'], arm[\'res\'], dev)\n        print(f"[arm {a}] loaded {arm[\'file\']} | res {res} | {time.time() - t0:.0f}s", flush=True)\n        for i, sid in enumerate(test_ids):\n            try:\n                vol, mask = build_study(sid, SER, tsdir, reader)\n                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)\n                arm_probs[a][i] = infer_probs(model, xw, dev)\n                del vol, mask, xw\n            except Exception as e:\n                print(f\'  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})\', flush=True)\n            if (i + 1) % 100 == 0 or i + 1 == N:\n                print(f\'  [arm {a}] {i + 1}/{N} | {time.time() - t0:.0f}s\', flush=True)\n        del model\n        gc.collect()\n        if str(dev).startswith(\'cuda\'):\n            torch.cuda.empty_cache()\n        print(f\'[arm {a}] done + freed | {time.time() - t0:.0f}s\', flush=True)\n    _w = np.array([float(a.get(\'w\', 1.0)) for a in ARMS], dtype=np.float64)\n    _w = _w / _w.sum()\n    print(f"[blend] weighted rank-mean w={dict(zip([a[\'file\'] for a in ARMS], _w.round(4)))}", flush=True)\n    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]), axes=(0, 0))\n    if not np.isfinite(ranks).all():\n        ranks[~np.isfinite(ranks)] = 0.5\n    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)\n    sub.insert(0, \'StudyInstanceUID\', test_ids)\n    sub = sub[sub_cols]\n    assert list(sub.columns) == sub_cols, \'column order drift\'\n    assert sub[\'StudyInstanceUID\'].tolist() == test_ids, \'row identity drift\'\n    assert np.isfinite(sub[LAB].values).all()\n    out = \'/kaggle/working/_raptor.csv\'\n    sub.to_csv(out, index=False)\n    print(\'wrote\', out, \'|\', len(sub), \'rows x\', len(sub.columns), \'cols\', flush=True)\n    print(sub.head().to_string(index=False), flush=True)\n    print(f\'DONE {time.time() - t0:.0f}s\', flush=True)'

ke_src = _KE_SRC_PATCHED

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

EPOCHS = int(os.environ.get('EPOCHS', '4'))
K_WIN = 12  # v2: micro-batching MICRO dihapus (dead-code no-op krn K_WIN==MICRO==12 default,
            # lihat komentar fix di loop bawah) -- 1 forward per studi memakai semua K_WIN window.
ACC_STUDIES = 8          # optimizer step tiap 8 studi (grad-accum)
LR_HEAD, LR_BB, WD = 1e-3, 2e-5, 0.05
WARMUP = 300

wp = KE['find_weight_file']('raptor_ft_coatnet_v10_full.pt')
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
out.to_csv('/kaggle/working/train_rsna-knee-train-coatnet-ours-v10_gold_preds.csv', index=False)

ytrue = train.set_index('StudyInstanceUID').loc[gids, LAB].values.astype(float)
def auc1(yy, ss):
    pos, neg = ss[yy == 1], ss[yy == 0]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
aucs = [auc1(ytrue[:, j], P[:, j]) for j in range(N_T)
        if 0 < ytrue[:, j].sum() < len(ytrue)]
tlog(f'MACRO AUC gold-58 (final, full-ft, label blended RAW-MEAN flight+claude): {np.mean(aucs):.4f}')
tlog('pembanding: c5=0.9206, v1 flight0234-only=0.9193, v2 rank-mean blend=0.9170 (REGRESI, akar masalah: rank-mean rusak kalibrasi BCE target)')

torch.save({'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': RES,
            'model': model.state_dict()},
           '/kaggle/working/coatnet_ourlabels_v10.pt')
tlog('checkpoint disimpan: coatnet_ourlabels_v10.pt')
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
(OUT / "rsna-knee-train-coatnet-ours-v10.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "dimaspashaakrilian/rsna-knee-train-coatnet-ours-v10",
    "title": "RSNA Knee - Train CoatNet ours v10",
    "code_file": "rsna-knee-train-coatnet-ours-v10.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "dimaspashaakrilian/rsna-knee-blended-report-labels-rawmean-v1",
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-widedense",
        "dreaddevelopment/raptor-knee-native384dense",
        "mattiaangeli/knee-mri-fold-weights",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)
