#!/usr/bin/env python3
"""Bangun kernel student-dino (K1): 1 model DINOv2-S 2.5D one-shot, akun alver.

Pakai: python3 build_student_dino.py

Catatan loading backbone (diverifikasi lokal thd aset tonylica dinov2-small/pytorch_model.bin):
  - 223 kunci: embeddings.{cls_token,mask_token,position_embeddings,patch_embeddings.projection.*},
    encoder.layer.N.{norm1,attention.attention.{query,key,value},attention.output.dense,
    layer_scale1.lambda1,norm2,mlp.{fc1,fc2},layer_scale2.lambda1}, layernorm.*
  - pos-embed (1,1370,384) => model dilatih @518px (grid 37) => input 224 perlu interpolasi bicubic.
  - KARENA ITU ViTModel HF dari_pretrained TIDAK dipakai (kunci q_proj/layer_norm1 tak cocok +
    layer-scale tak ada di ViT) — dibangun ViT-S/14 setara dgn PyTorch murni + load strict.
"""
import json, os

SLUG = "rsna-knee-student-dino"
MD = {
    "id": f"alverlfarfaraway/{SLUG}",
    "title": "RSNA Knee student DINOv2-S one-shot",
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "category": "video",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "alverlfarfaraway/rsna-knee-student-cache256",
        "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    ],
    "keywords": [],
    "kernel_sources": [],
    "docker_image": "registry Kaggle Python",
    "machine_shape": " NvidiaTeslaT4",
}

CELL0 = """# RSNA Knee — student-dino (Fase B, langkah 2c)
# DINOv2-S 2.5D one-shot: train SEMUA studi weak (tanpa fold) thd labels_student_v0; emas-58 val only.
# Backbone: ViT-S/14 dinov2-small (aset tonylica) 384-dim; head = gaya KneeNet goodpjw (slot emb 6,
# pos emb continuous, TransformerEncoder d=2/8head, attention-pooling per finding).
# 1 jendela 3-slice -> encode 224 (patch14) -> MEAN-pool patch token -> 1 token; 6 slot x 8 = 48 token.
# Output: /kaggle/working/dino_full.pt (bobot EMA) + resume atomik per epoch.
"""

CELL1 = r'''
import os, sys, glob, json, time, copy, math, shutil, subprocess
import numpy as np, pandas as pd, torch
t0 = time.time()
def say(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.backends.cudnn.benchmark = True

def find(pats, base="/kaggle/input"):
    for p in pats:
        for f in glob.glob(os.path.join(base, "**", p), recursive=True):
            return f
    return None

COMP = None
for c in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(c, "train.csv")): COMP = c; break
assert COMP
PREP = None
for c in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(c, "labels_student_v0.csv")): PREP = c; break
assert PREP, "prep dataset belum ada"
GOODPJW = None
for c in glob.glob("/kaggle/input/*"):
    if os.path.exists(os.path.join(c, "knee.py")): GOODPJW = c; break
assert GOODPJW
DINO = None
for c in sorted(glob.glob("/kaggle/input/*")):
    if os.path.exists(os.path.join(c, "dinov2-small", "pytorch_model.bin")):
        DINO = os.path.join(c, "dinov2-small"); break
assert DINO, "dinov2-small/ (aset tonylica) tidak ketemu"
sys.path.insert(0, GOODPJW)
say("COMP", COMP); say("PREP", PREP); say("GOODPJW", GOODPJW); say("DINO", DINO)

whl_dir = os.path.dirname(find(["pylibjpeg-2.1.0-py3-none-any.whl"]) or "")
if whl_dir:
    for w in sorted(glob.glob(os.path.join(whl_dir, "*.whl"))):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", w], check=False)

from knee import LABELS, KneeNet, StudyDataset, build_study_table, make_views, N_SLOTS  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

TRAIN = pd.read_csv(f"{COMP}/train.csv")
SERIES = pd.read_csv(f"{COMP}/train_series.csv")
lab = pd.read_csv(f"{PREP}/labels_student_v0.csv", index_col=0)
CACHE = f"{PREP}/cache256_train"
if os.path.isdir(CACHE) and len(os.listdir(CACHE)) > 1000:
    meta = pd.read_csv(f"{PREP}/cache256_train_meta.csv")
    say("cache dari dataset", CACHE)
else:
    # ---- bangun cache on-demand (resep prep v3: SIZE 256, MM 0.5, MAX 28, IPP sort) ----
    import cv2, pydicom
    from multiprocessing import Pool
    say("cache dataset absen — build on-demand dari DICOM")
    MM_PER_PX, SIZE, MAX_SLICES = 0.5, 256, 28
    CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
    PLANES = ["Sagittal", "Coronal", "Axial"]
    def _read(path):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
        if slope is not None and inter is not None:
            arr = arr * float(slope) + float(inter)
        iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
        ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
        ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
        inst = float(ds.get("InstanceNumber") or 0)
        frames = list(arr) if arr.ndim == 3 else [arr]
        return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]
    def _canon_slice(img, iop, ps, plane):
        ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2
        if iop is None:
            return img, ps
        r, c = np.array(iop[:3]), np.array(iop[3:])
        _, (cax, csgn), (rax, rsgn) = CANON[plane]
        if abs(r[cax]) < abs(r[rax]):
            img, r, c, ps = img.T, c, r, ps[::-1]
        if r[cax] * csgn < 0:
            img = img[:, ::-1]
        if c[rax] * rsgn < 0:
            img = img[::-1]
        return img, ps
    def _resample(img, ps):
        out = np.zeros((SIZE, SIZE), np.float32)
        h, w = img.shape
        ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
        cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
        nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
        nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
        interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
        res = cv2.resize(crop, (nw, nh), interpolation=interp)
        oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
        out[oy:oy + nh, ox:ox + nw] = res
        return out
    def load_series(sdir, plane_name=None):
        slices, bad = [], 0
        for fn in os.listdir(sdir):
            try:
                slices.extend(_read(os.path.join(sdir, fn)))
            except Exception:
                bad += 1
        info = {"n_raw": len(slices), "n_bad": bad}
        if not slices:
            return np.zeros((1, SIZE, SIZE), np.uint8), info
        iop = next((s[1] for s in slices if s[1] is not None), None)
        if iop is not None:
            normal = np.cross(iop[:3], iop[3:]); plane = int(np.abs(normal).argmax())
        else:
            normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
        info["plane"] = PLANES[plane]
        if normal is not None and all(s[2] is not None for s in slices):
            sgn = CANON[plane][0] * np.sign(normal[plane])
            slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
        else:
            slices.sort(key=lambda s: s[4])
        if len(slices) > MAX_SLICES:
            keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
            slices = [slices[i] for i in keep]
        vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
        lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        info["n"] = len(vol)
        return (vol * 255).round().astype(np.uint8), info
    def _job(a):
        study, series, plane = a
        out = f"{CACHE}/{series}.npy"
        info = {"study": study, "series": series}
        if os.path.exists(out):
            return None
        try:
            vol, extra = load_series(f"{COMP}/train_series/{study}/{series}")
            np.save(out + ".tmp.npy", vol); os.replace(out + ".tmp.npy", out)
            info.update(extra)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"[:200]
        return info
    os.makedirs(CACHE, exist_ok=True)
    tasks = [(a, b, p) for a, b, p in zip(SERIES.StudyInstanceUID, SERIES.SeriesInstanceUID, SERIES.Anatomical_Plane)]
    rows = []
    with Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(_job, tasks, chunksize=8)):
            if r: rows.append(r)
            if i % 1000 == 0:
                say(f"cache {i}/{len(tasks)} files={len(os.listdir(CACHE))}")
                if shutil.disk_usage("/tmp").free < 3 * 2**30:
                    say("DISK HAMPIR HENTI"); break
    df = pd.DataFrame(rows)
    if "error" in df: say("errors:", int(df["error"].notna().sum()))
    meta = df.dropna(subset=["n"]).copy()
    say("meta rows", len(meta), "files", len(os.listdir(CACHE)))
say("lab", lab.shape, "meta", meta.shape, "cache files", len(os.listdir(CACHE)))
say("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available()
'''

# Sel model — dipakai identik di K2 (rsna-knee-student-infer).
CELL_MODEL = r'''
import torch, torch.nn as nn, torch.nn.functional as F
from knee import LABELS, N_SLOTS, make_views

# --- DINOv2-S/14 setara (kunci state-dict IDENTIK dgn tonylica/dinov2-small/pytorch_model.bin) ---
class _QKV(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.query = nn.Linear(dim, dim); self.key = nn.Linear(dim, dim); self.value = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, _ = x.shape
        q = self.query(x).view(B, N, self.heads, self.hd).transpose(1, 2)
        k = self.key(x).view(B, N, self.heads, self.hd).transpose(1, 2)
        v = self.value(x).view(B, N, self.heads, self.hd).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, -1)

class _AttnOut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = _QKV(dim, heads)
        self.output = _AttnOut(dim)
    def forward(self, x):
        return self.output(self.attention(x))

class _MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio)); self.fc2 = nn.Linear(int(dim * ratio), dim)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * self.lambda1

class _Block(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = _Attn(dim, heads)
        self.layer_scale1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MLP(dim)
        self.layer_scale2 = _LayerScale(dim)
    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))

class _Embeddings(nn.Module):
    def __init__(self, dim=384, patch=14, grid=37):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.position_embeddings = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.grid, self.patch = grid, patch
    def forward(self, x):
        B, _, H, W = x.shape
        p = self.patch_embeddings.projection(x).flatten(2).transpose(1, 2)
        e = torch.cat([self.cls_token.expand(B, -1, -1), p], 1)
        gh, gw = H // self.patch, W // self.patch
        if e.shape[1] == self.position_embeddings.shape[1] and gh == gw == self.grid:
            pos = self.position_embeddings.to(e.dtype)
        else:  # interpolasi bicubic pos-embed 37x37 (latih @518) -> grid input (mis. 16x16 @224)
            cls_pos = self.position_embeddings[:, :1]
            pp = self.position_embeddings[:, 1:].reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
            pp = F.interpolate(pp.float(), size=(gh, gw), mode="bicubic", align_corners=False).to(e.dtype)
            pos = torch.cat([cls_pos.to(e.dtype), pp.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], 1)
        return e + pos

class DinoSmall(nn.Module):
    """ViT-S/14 DINOv2: embed 384, 12 blok, 6 head, layer-scale; output [B, 1+P, 384]."""
    def __init__(self, dim=384, depth=12, heads=6):
        super().__init__()
        self.embeddings = _Embeddings(dim)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x):
        e = self.embeddings(x)
        for blk in self.encoder.layer:
            e = blk(e)
        return self.layernorm(e)

class StudentDino(nn.Module):
    """Head = gaya KneeNet goodpjw; bedanya hanya backbone & 1 token/jendela (mean-pool patch)."""
    def __init__(self, dino_dir=None, dim=384, depth=2, drop=0.2, chunk=32):
        super().__init__()
        self.enc = DinoSmall()
        if dino_dir:
            sd = torch.load(f"{dino_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)
            missing, unexpected = self.enc.load_state_dict(sd, strict=False)
            assert not unexpected, f"bobot tak terpakai: {unexpected[:8]}"
            assert not missing, f"kunci hilang: {missing[:8]}"
            del sd
        self.chunk = chunk
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(384, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    def _encode(self, imgs, k, train):
        """uint8 [n,3,H,W] -> [n,384]; make_views(fov .92, res 224) lalu ViT per chunk, mean-pool patch."""
        x = make_views(imgs, 224, train, k)
        out = []
        for ch in x.split(self.chunk):
            out.append(self.enc(ch)[:, 1:].mean(1))
        return torch.cat(out, 0)

    def forward(self, x, mask, pos, res=224, train=False):
        """x uint8 [B,S,K,3,H,W] -> logits [B,12]."""
        b, s, k = x.shape[:3]
        keep = mask.view(-1)
        xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        f = self._encode(xv, k, train)
        feat = f.new_zeros(b * s, k, f.shape[-1])
        feat[keep] = f.view(-1, k, f.shape[-1])
        feat = feat.view(b, s, k, -1)
        b2, s2, k2, _ = feat.shape
        slot = torch.arange(s2, device=feat.device).view(1, s2, 1).expand(b2, s2, k2)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(b2, s2 * k2, -1)
        pad = ~mask.unsqueeze(-1).expand(b2, s2, k2).reshape(b2, s2 * k2)
        pad = pad & ~pad.all(1, keepdim=True)
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b
'''

CELL3 = r'''
import os, time, copy, math
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

WORK = "/kaggle/working"
RUNS = f"{WORK}/runs_dino"; os.makedirs(RUNS, exist_ok=True)
EPOCHS, BS, KWIN, KEVAL = 12, 8, 8, 12          # bs efektif 8 studi; jendela train k=8 / eval k=12
LR_ENC, LR_HEAD, WD, WARM, EMA_D, CLIP = 1e-5, 5e-4, 0.05, 200, 0.999, 3.0
table = build_study_table(SERIES, dict(zip(meta.series, meta.n.fillna(0))))
gold = lab[lab.is_gold == 1]; weak = lab[lab.is_gold == 0]
say(f"one-shot: train weak {len(weak)} studi (tanpa fold split) | emas-58 val only {len(gold)}")

def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs

@torch.no_grad()
def predict(model, loader):
    model.eval(); out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x, mask, pos, res=224).float().cpu())
    return torch.cat(out).numpy()

dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
train_dl = DataLoader(
    StudyDataset(weak.index, table, CACHE, weak[LABELS].values.astype(np.float32), KWIN, train=True),
    batch_size=BS, shuffle=True, drop_last=True, **dl_kw)
gold_dl = DataLoader(StudyDataset(gold.index, table, CACHE, None, KEVAL), batch_size=BS, shuffle=False, **dl_kw)
gold_y = gold[[f"gold_{c}" for c in LABELS]].values

torch.manual_seed(0); np.random.seed(0)
model = StudentDino(DINO).cuda()
say("param total %.1fM | backbone %.1fM | head %.1fM" % (
    sum(p.numel() for p in model.parameters()) / 1e6,
    sum(p.numel() for p in model.enc.parameters()) / 1e6,
    sum(p.numel() for n, p in model.named_parameters() if not n.startswith("enc.")) / 1e6))
ema = copy.deepcopy(model).eval()
for p in ema.parameters(): p.requires_grad_(False)
enc = list(model.enc.parameters())
head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENC}, {"params": head, "lr": LR_HEAD}], weight_decay=WD)
steps, warm = EPOCHS * len(train_dl), WARM
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
scaler = torch.amp.GradScaler("cuda")   # fp16 + scaler (T4; jangan bf16)

state, step, start_ep, preds = f"{RUNS}/state_dino.pt", 0, 0, None
if os.path.exists(state):
    st = torch.load(state, map_location="cuda", weights_only=False)
    model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
    step, start_ep = st["step"], st["epoch"] + 1
    say(f"resume epoch {st['epoch']}")

DEADLINE = t0 + 6.5 * 3600
for ep in range(start_ep, EPOCHS):
    model.train(); t1, tot, n = time.time(), 0.0, 0
    for x, mask, pos, y in train_dl:
        x, mask, pos, y = x.cuda(non_blocking=True), mask.cuda(), pos.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(x, mask, pos, res=224, train=True)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        d = min(EMA_D, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
        tot += loss.item(); n += 1
        if n % 100 == 0: say(f"  ep{ep} {n}/{len(train_dl)} loss {tot/n:.4f} {(time.time()-t1)/n:.2f}s/it")
    preds = predict(ema, gold_dl)
    g_auc, aucs = macro_auc(gold_y, preds)
    say(f"ep{ep:2d} loss {tot/n:.4f} time {time.time()-t1:.0f}s | emas58 {g_auc:.4f}")
    say("    per-finding:", {c: (None if not np.isfinite(a) else round(float(a), 3)) for c, a in zip(LABELS, aucs)})
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": ep},
               state + ".tmp")
    os.replace(state + ".tmp", state)
    if time.time() > DEADLINE:
        say("deadline 6.5 jam tercapai — stop (state per-epoch tersimpan, boleh resume)"); break

if preds is None:
    preds = predict(ema, gold_dl)
torch.save({"model": ema.state_dict(), "args": {"res": 224, "k": KWIN, "backbone": "dinov2-small"}},
           f"{WORK}/dino_full.pt")
say(f"FINAL emas58 {macro_auc(gold_y, preds)[0]:.4f} | dino_full.pt "
    f"{os.path.getsize(f'{WORK}/dino_full.pt')/1e6:.0f} MB")
say("DONE files:", sorted(os.listdir(RUNS)))
'''


def cell(src, ctype="code"):
    c = {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True)}
    if ctype == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


def main():
    nb = {"cells": [cell(CELL0, "markdown"), cell(CELL1), cell(CELL_MODEL), cell(CELL3)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"<cell{i}>", "exec")   # validasi syntactic
    out = "kaggle_kernel_student_dino"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(f"{out}/{SLUG}.ipynb", "w"), indent=1)
    json.dump(MD, open(f"{out}/kernel-metadata.json", "w"), indent=2)
    print("ditulis:", out, "| cells:", len(nb["cells"]), "| id:", MD["id"])


if __name__ == "__main__":
    main()
