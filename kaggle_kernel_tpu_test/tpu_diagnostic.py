import time, sys, traceback

t0 = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START tpu_diagnostic")

print("--- torch ---")
try:
    import torch
    print("torch version:", torch.__version__)
    print("torch cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch import FAILED:", e)
    traceback.print_exc()

print("--- torch_xla ---")
try:
    import torch_xla
    print("torch_xla version:", getattr(torch_xla, "__version__", "unknown"))
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print("xla device:", dev)
    x = torch.randn(4, 4, device=dev)
    y = torch.randn(4, 4, device=dev)
    z = torch.matmul(x, y)
    xm.mark_step()
    print("matmul result shape:", z.shape, "sum:", float(z.sum()))
    print("XLA_OK: torch_xla usable on this Kaggle TPU session")
except Exception as e:
    print("torch_xla FAILED:", e)
    traceback.print_exc()

print("--- transformers AutoModel sanity (DINOv2 style) ---")
try:
    from transformers import AutoModel
    print("transformers import OK (not loading full model, just checking import)")
except Exception as e:
    print("transformers import FAILED:", e)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END tpu_diagnostic, elapsed={time.time()-t0:.1f}s")
