#!/usr/bin/env python3
"""Build kernel integrasi: fork prvsiyan toes (DINOsaur V10 @0.936) + arm kita.

Pemakaian:
  python3 build_submit_integration.py --arm-file coatnet_flight_ft_v1.pt \
      --w 1.0 --push

- Menyisipkan arm baru ke daftar ARMS CoAtNet di cell 4 notebook toes.
- Checkpoint diambil dari output kernel training via kernel_sources
  (find_weight_file glob /kaggle/input/* menemukannya otomatis).
- Metadata: 2 dataset wajib toes + kompetisi + model hub dinov2 + kernel training.
"""
import argparse
import json
from pathlib import Path

TOES = Path("/tmp/opencode/nb/toes/head-and-shoulders-knees-and-toes.ipynb")
OUT = Path(__file__).resolve().parent / "kaggle_kernel_submit_integration_v1"
TRAIN_KERNEL_SLUG = "dimaspashaakrilian/rsna-knee-train-coatnet-flight-ft-v1"
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"

p = argparse.ArgumentParser()
p.add_argument("--arm-file", default="coatnet_flight_ft_v1.pt")
p.add_argument("--w", type=float, default=1.0)
p.add_argument("--keep-legacy", action="store_true",
               help="aktifkan juga v4_full sbg arm tambahan")
p.add_argument("--push", action="store_true")
args = p.parse_args()

nb = json.loads(TOES.read_text())

# --- sisipkan arm ke cell yang memuat ARMS ---
for c in nb["cells"]:
    src = "".join(c["source"])
    if "ARMS = [" not in src or "raptor_ft_coatnet" not in src:
        continue
    old = '''ARMS = [
    {
        "file": "raptor_ft_coatnet_v5_full_swa.pt",
        "arch": "%s",
        "res": 384,
        "w": 1.0,
    },
]''' % ARCH
    assert old in src, "blok ARMS standar tidak ketemu — format toes berubah?"
    arms = ['''ARMS = [
    {
        "file": "raptor_ft_coatnet_v5_full_swa.pt",
        "arch": "%s",
        "res": 384,
        "w": 1.0,
    },
    {
        "file": "%s",
        "arch": "%s",
        "res": 384,
        "w": %s,
    },%s
]''' % (ARCH, args.arm_file, ARCH, args.w,
        '''
    {
        "file": "raptor_ft_coatnet_v4_full.pt",
        "arch": "%s",
        "res": 384,
        "w": 1.0,
    },''' % ARCH if args.keep_legacy else "")]
    c["source"] = [src.replace(old, arms[0])]
    print("cell 4 dipatch: ARMS += %s (w=%s%s)" %
          (args.arm_file, args.w, ", +legacy" if args.keep_legacy else ""))
    break
else:
    raise SystemExit("cell ARMS tidak ditemukan")

OUT.mkdir(exist_ok=True)
(OUT / "submit_integration_v1.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "dimaspashaakrilian/rsna-knee-submit-integration-v1",
    "title": "RSNA Knee - Submit integration v1 own coatnet",
    "code_file": "submit_integration_v1.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-widedense",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [
        "metaresearch/dinov2/PyTorch/small/1",
        "metaresearch/dinov2/PyTorch/base/1",
    ],
    "kernel_sources": [TRAIN_KERNEL_SLUG],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built:", OUT)

if args.push:
    import subprocess
    r = subprocess.run(["kaggle", "kernels", "push", "-p", "."],
                       cwd=str(OUT), capture_output=True, text=True)
    print(r.stdout, r.stderr)
