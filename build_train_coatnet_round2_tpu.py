#!/usr/bin/env python3
"""Versi TPU dari putaran-2 noisy-student (build_train_coatnet_round2.py).

Mengambil skrip GPU yang dihasilkan builder putaran-2 lalu menukar bagian CUDA-spesifik dgn padanan XLA,
dgn pola yang SAMA seperti build_train_coatnet_kfold_tpu.py (probe TPU 2 Okt membuktikan CoAtNet 384px
fwd+bwd jalan di XLA, 0,5 dtk/langkah). Semua penukaran di-assert: bila pola tak ketemu build GAGAL.

Beda utama: bentuk tensor tetap (window diisi ulang sampai K_WIN), autocast bf16 'xla' dgn bobot fp32
(bukan XLA_USE_BF16), tanpa GradScaler, mark_step per studi, LR dikuantisasi 30 level, checkpoint dari
state_dict di CPU. Kernel TPU: enable_tpu, tanpa GPU.
Pemakaian: python3 build_train_coatnet_round2_tpu.py <round=1> <init=c5|v10|v8> [epochs=4]
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
INIT = sys.argv[2] if len(sys.argv) > 2 else "v10"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

# 1) hasilkan skrip GPU (dan kernel GPU-nya) dulu
subprocess.run([sys.executable, str(REPO / "build_train_coatnet_round2.py"), str(R), INIT, str(EPOCHS)],
               check=True, capture_output=True, cwd=REPO)
TAG = f"round2-{INIT}-r{R}"
gpu_py = REPO / f"build_train_{TAG.replace('-', '_')}.py"
src = gpu_py.read_text()


def must(text, old, new):
    assert old in text, f"pola tidak ketemu: {old[:90]!r}"
    return text.replace(old, new, 1)


src = must(src, "DEV = 'cuda'", "import torch_xla.core.xla_model as xm\nDEV = xm.xla_device()")
src = must(src, "384, DEV)\nmodel.backbone.requires_grad_(True)",
           "384, 'cpu')\nmodel = model.to(DEV)\nmodel.backbone.requires_grad_(True)")
src = must(src, "    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))\n",
           "    cs = list(np.random.choice(cs, size=min(k, len(cs)), replace=False))\n"
           "    while len(cs) < k:                      # bentuk TETAP utk XLA\n"
           "        cs.append(cs[len(cs) % len(cs)])\n")

# 2) optimizer/scheduler/scaler -> versi TPU
a = src.index("sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) *")
b_mark = "scaler = torch.amp.GradScaler('cuda')\n"
b = src.index(b_mark) + len(b_mark)
src = src[:a] + '''LRQ = 30
def lr_scale(s):
    sc = min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    return max(1, round(sc * LRQ)) / LRQ          # terkuantisasi, tak pernah 0
def set_lr(s):
    sc = lr_scale(s)
    opt.param_groups[0]['lr'] = LR_BB * sc
    opt.param_groups[1]['lr'] = LR_HEAD * sc
import contextlib, time as _tm
try:
    with torch.autocast('xla', dtype=torch.bfloat16):
        _t = torch.ones(8, 8, device=DEV) @ torch.ones(8, 8, device=DEV)
    xm.mark_step(); float(_t.sum())
    AC = lambda: torch.autocast('xla', dtype=torch.bfloat16)
    tlog('autocast xla bf16: OK')
except Exception as _e:
    AC = lambda: contextlib.nullcontext()
    tlog(f'autocast xla GAGAL ({type(_e).__name__}: {_e}) -> fp32')
''' + src[b:]

# 3) isi langkah latih: autocast/loss/backward + penanganan OOM + langkah optimizer
old_fwd_start = "            with torch.autocast('cuda', dtype=torch.float16):\n                logits = model(xb).float()\n"
a2 = src.index(old_fwd_start)
b2_mark = "            scaler.scale(loss).backward()\n"
b2 = src.index(b2_mark) + len(b2_mark)
src = src[:a2] + '''            with AC():
                logits = model(xb)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), yb, reduction='none') * wb).mean()
            loss.backward()
            xm.mark_step()
''' + src[b2:]
src = must(src, "        except torch.cuda.OutOfMemoryError:\n            torch.cuda.empty_cache()\n"
                "            tlog(f'OOM di {sid[:16]} -- skip studi ini')\n            continue\n",
           "        except Exception as e:\n"
           "            tlog(f'GAGAL studi {sid[:16]}: {type(e).__name__}: {str(e)[:120]} -- dilewati')\n"
           "            xm.mark_step()\n            continue\n")
src = must(src, "            scaler.step(opt); scaler.update(); sched.step()\n            opt.zero_grad(set_to_none=True)\n",
           "            set_lr(step)\n            opt.step(); xm.mark_step()\n            opt.zero_grad(set_to_none=True)\n")
src = must(src, "f'lr_bb={sched.get_last_lr()[0]:.2e}')", "f'lr_bb={LR_BB*lr_scale(step):.2e}')")

# 4) checkpoint dari CPU
src = must(src, "            'model': model.state_dict()},",
           "            'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}},")

# 5) identitas & metadata TPU
slug_gpu = f"rsna-knee-train-{TAG}"
slug_tpu = f"rsna-knee-train-tpu-{TAG}"
dir_gpu = f"kaggle_kernel_train_{TAG.replace('-', '_')}"
dir_tpu = f"kaggle_kernel_train_tpu_{TAG.replace('-', '_')}"
src = src.replace(dir_gpu, dir_tpu).replace(slug_gpu, slug_tpu)
src = src.replace(f'"title": "RSNA Knee - Train {TAG}"', f'"title": "RSNA Knee - Train TPU {TAG}"')
src = must(src, '"enable_gpu": True, "enable_tpu": False, "enable_internet": False,\n    "machine_shape": "NvidiaTeslaT4",',
           '"enable_gpu": False, "enable_tpu": True, "enable_internet": False,')

out_py = REPO / f"build_train_tpu_{TAG.replace('-', '_')}.py"
out_py.write_text(src)
r = subprocess.run([sys.executable, str(out_py)], capture_output=True, text=True, cwd=REPO)
print(r.stdout.strip() or r.stderr.strip()[-1500:])
# dedupe dataset_sources
mp = REPO / dir_tpu / "kernel-metadata.json"
if mp.is_file():
    m = json.loads(mp.read_text()); m["dataset_sources"] = list(dict.fromkeys(m["dataset_sources"]))
    mp.write_text(json.dumps(m, indent=2))
