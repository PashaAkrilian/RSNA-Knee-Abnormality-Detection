#!/usr/bin/env python3
"""Versi TPU dari build_train_coatnet_kfold.py -- fold K-fold CoAtNet utk noisy-student (§32).

BUKTI TPU LAYAK (probe 2 Okt, rsna-knee-tpu-probe-coatnet, torch 2.8 + torch_xla): CoAtNet-RMLP-2
384px, K=12 window, forward+backward+AdamW jalan di XLA; compile ~50 dtk dua kali lalu 0,5 dtk/langkah
stabil, loss turun. Kuota TPU (20 jam/minggu) TERPISAH dari GPU dan GPU sedang habis/dibagi proyek
lain; GPU juga wajib disisakan utk submission (pipeline publik CUDA).

Beda dari versi GPU (semua lewat penggantian teks yang di-assert, bukan penulisan ulang):
  1. perangkat = xm.xla_device(); model dimuat di CPU lalu dipindah.
  2. bentuk tensor TETAP: jumlah window latih selalu K_WIN (diisi ulang bila volume pendek) --
     bentuk berubah = rekompilasi ~50 dtk tiap kali.
  3. autocast bf16 'xla' (bobot tetap fp32 -> aman utk LR kecil 2e-5; JANGAN XLA_USE_BF16 yg membuat
     bobot ikut bf16 dan memakan update kecil). Dites sekali di awal; kalau gagal -> fp32.
  4. tanpa GradScaler; xm.mark_step() tiap studi memotong graf; opt.step() tiap ACC_STUDIES.
  5. LR dikuantisasi 30 level -> menjaga jumlah rekompilasi graf optimizer tetap terbatas bila
     skalar tak jadi device-data. Waktu tiap langkah dilog supaya terlihat kalau ada rekompilasi.
  6. checkpoint disimpan dari state_dict yg dipindah ke CPU.
Pemakaian: python3 build_train_coatnet_kfold_tpu.py <fold> [nfold=4] [epochs=3]
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NFOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 4
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

src = (REPO / "build_train_coatnet_kfold.py").read_text()


def must(text, old, new, count=1):
    assert text.count(old) >= 1, f"pola tidak ketemu: {old[:80]!r}"
    return text.replace(old, new, count)


# 1. perangkat + muat model di CPU
src = must(src, "DEV = 'cuda'\nLAB = KE['LAB']; N_T = len(LAB)",
           "import torch_xla.core.xla_model as xm\nDEV = xm.xla_device()\nLAB = KE['LAB']; N_T = len(LAB)")
src = must(src, "model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, DEV)",
           "model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, 'cpu')\n"
           "    model = model.to(DEV)")

# 2. bentuk tetap: isi ulang window sampai tepat k
src = must(src, "    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))\n",
           "    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))\n"
           "    while len(cs) < k:                      # bentuk TETAP utk XLA\n"
           "        cs.append(cs[len(cs) % len(cs)])\n")

# 3-6. blok latih: dari 'model.backbone.requires_grad_(True)' s.d. 'checkpoint fold disimpan'
a = src.index("model.backbone.requires_grad_(True)")
b_marker = "tlog('checkpoint fold disimpan')"
b = src.index(b_marker) + len(b_marker)

NEW_TRAIN = r'''model.backbone.requires_grad_(True)
head_params = list(model.norm.parameters()) + list(model.att.parameters()) + [model.clsW, model.clsb]
bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{'params': bb_params, 'lr': LR_BB},
                         {'params': head_params, 'lr': LR_HEAD}], weight_decay=WD)
total_steps = max(1, (len(train_ids) * EPOCHS) // ACC_STUDIES)
LRQ = 30
def lr_scale(s):
    sc = min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    return max(1, round(sc * LRQ)) / LRQ          # terkuantisasi, tak pernah 0
def set_lr(s):
    sc = lr_scale(s)
    opt.param_groups[0]['lr'] = LR_BB * sc
    opt.param_groups[1]['lr'] = LR_HEAD * sc

# autocast bf16 XLA (bobot tetap fp32). Uji sekali; kalau gagal -> fp32.
import contextlib
try:
    with torch.autocast('xla', dtype=torch.bfloat16):
        _t = torch.ones(8, 8, device=DEV) @ torch.ones(8, 8, device=DEV)
    xm.mark_step(); float(_t.sum())
    AC = lambda: torch.autocast('xla', dtype=torch.bfloat16)
    tlog('autocast xla bf16: OK')
except Exception as _e:
    AC = lambda: contextlib.nullcontext()
    tlog(f'autocast xla GAGAL ({type(_e).__name__}: {_e}) -> fp32')
tlog(f'model siap res={RES} | steps={total_steps} | FULL FT bb lr={LR_BB} | TPU')

y_lab = FL.loc[train_ids, LAB].fillna(0.28).values.astype(np.float32)
c_lab = np.stack([FL.loc[train_ids, CONF_COLS[t]].fillna(0.15).values.astype(np.float32)
                  for t in LAB], 1)
W_CONF = 0.5 + np.clip(c_lab, 0, 1)

import time as _tm
step = 0
for ep in range(EPOCHS):
    order = np.random.permutation(len(train_ids))
    model.train()
    opt.zero_grad(set_to_none=True)
    ep_loss, nacc = 0.0, 0
    t_ep = _tm.time()
    for ii, si in enumerate(order):
        sid = train_ids[si]
        ts = _tm.time()
        try:
            vol, mask = load_study(sid)
            xw = rand_windows(vol, mask, K_WIN, RES)
            xb = xw.unsqueeze(0).to(DEV)
            yb = torch.from_numpy(y_lab[si])[None].to(DEV)
            wb = torch.from_numpy(W_CONF[si])[None].to(DEV)
            with AC():
                logits = model(xb)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), yb, reduction='none') * wb).mean()
            loss.backward()
            xm.mark_step()
            ep_loss += float(loss.detach())
            del vol, mask, xw
        except Exception as e:
            tlog(f'GAGAL studi {sid[:16]}: {type(e).__name__}: {str(e)[:120]} -- dilewati')
            xm.mark_step()
            continue
        nacc += 1
        if nacc % ACC_STUDIES == 0 or ii == len(order) - 1:
            set_lr(step)
            opt.step(); xm.mark_step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if ep == 0 and ii < 12:
            tlog(f'  [awal] studi {ii} dt={_tm.time()-ts:.1f}s (kompilasi wajar di langkah awal)')
        if (ii + 1) % 200 == 0:
            tlog(f'ep{ep+1} {ii+1}/{len(order)} loss~{ep_loss/(ii+1):.4f} '
                 f'lr_bb={LR_BB*lr_scale(step):.2e} | {(_tm.time()-t_ep)/(ii+1):.2f}s/studi')
    tlog(f'=== EPOCH {ep+1}/{EPOCHS} selesai | loss={ep_loss/max(1,len(order)):.4f} | {(_tm.time()-t_ep)/60:.1f} mnt')

torch.save({'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': RES,
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}},
           f'/kaggle/working/kfold_{INIT}_f{FOLD}.pt')
tlog('checkpoint fold disimpan')'''
src = src[:a] + NEW_TRAIN + src[b:]

# identitas kernel & metadata TPU
src = must(src, 'OUT = REPO / f"kaggle_kernel_kfold_{INIT}_f{FOLD}"',
           'OUT = REPO / f"kaggle_kernel_kfold_tpu_{INIT}_f{FOLD}"')
src = must(src, 'SLUG = f"rsna-knee-kfold-{INIT}-f{FOLD}"', 'SLUG = f"rsna-knee-kfold-tpu-{INIT}-f{FOLD}"')
src = must(src, 'NAME = f"kfold_{INIT}_f{FOLD}.ipynb"', 'NAME = f"kfold_tpu_{INIT}_f{FOLD}.ipynb"')
src = must(src, '"title": f"RSNA Knee - KFold {INIT} f{FOLD}",', '"title": f"RSNA Knee - KFold TPU {INIT} f{FOLD}",')
src = must(src, '"enable_gpu": True, "enable_tpu": False, "enable_internet": False,\n    "machine_shape": "NvidiaTeslaT4",',
           '"enable_gpu": False, "enable_tpu": True, "enable_internet": False,')

out_py = REPO / f"build_train_coatnet_kfold_tpu_f{FOLD}.py"
out_py.write_text(src)
r = subprocess.run([sys.executable, str(out_py), str(FOLD), str(NFOLD), "c5", str(EPOCHS)],
                   capture_output=True, text=True, cwd=REPO)
print(r.stdout.strip() or r.stderr.strip()[-1200:])
