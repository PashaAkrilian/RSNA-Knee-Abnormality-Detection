#!/bin/bash
# Tunggu kuota GPU Kaggle pulih (Sabtu 10 Okt 07:00 WIB), lalu luncurkan anak-OAI c,d di GPU (2xT4 paralel).
# SYARAT: berkas penanda ~/.cache/rsna_gpu_launch_ok ada (dibuat agen setelah log TPU anak a/b diperiksa & sehat).
# Pakai:  nohup bash launch_gpu_after_refresh.sh > /tmp/launch_gpu.log 2>&1 &     (atau '! bash launch_gpu_after_refresh.sh' dari prompt)
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")" || exit 1
for _ in $(seq 1 2400); do
  rem=$(kaggle quota 2>/dev/null | awk '$1=="GPU"{gsub("h","",$3); print $3}')
  if [ -n "$rem" ] && awk "BEGIN{exit !($rem >= 20.0)}"; then
    echo "[$(date +%H:%M:%S)] GPU pulih: sisa ${rem}h"
    if [ ! -f "$HOME/.cache/rsna_gpu_launch_ok" ]; then
      echo "Penanda ~/.cache/rsna_gpu_launch_ok TIDAK ada -> tidak meluncurkan otomatis. Periksa log TPU lalu: touch ~/.cache/rsna_gpu_launch_ok"; exit 2
    fi
    python3 build_train_gpu_oai_child.py || exit 3
    for t in 1 2 3 4; do
      r=$(kaggle kernels push -p kaggle_kernel_train_gpu_oai_child 2>&1 | tail -2); echo "push GPU anak (coba $t): $r"
      echo "$r" | grep -q "successfully pushed" && exit 0; sleep 30
    done
    exit 4
  fi
  sleep 300
done
echo "TIMEOUT menunggu kuota GPU"; exit 5
