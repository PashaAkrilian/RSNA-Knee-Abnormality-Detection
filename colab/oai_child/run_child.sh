#!/bin/bash
# VM Colab: jalankan/lanjutkan latih anak dgn auto-restart (status lanjut di /content/kw). Pakai: nohup bash /content/run_child.sh > /content/run_child.out 2>&1 &
cd /content
export VARIANTS="${VARIANTS:-[[\"c\",0.5,3]]}" DO_PARENT_SOFT="${DO_PARENT_SOFT:-0}" CUDA_VISIBLE_DEVICES=0 RESUME_DIR=/content/kw CKPT_EVERY="${CKPT_EVERY:-400}"
mkdir -p /content/kw
for i in 1 2 3 4 5 6 7 8; do
  python3 -u child_train_colab.py >> /content/child_c.log 2>&1
  if ls /content/kw/done_*.flag >/dev/null 2>&1 && [ "$(python3 -c "import json,os;v=json.loads(os.environ['VARIANTS']);print(all(os.path.exists('/content/kw/done_%s.flag'%x[0]) for x in v))")" = "True" ]; then
    echo "TRAINING DONE (run_child)" >> /content/child_c.log; break
  fi
  echo "[run_child] proses berhenti tanpa semua flag selesai; ulang ke-$i dlm 20 dtk (lanjut dari status tersimpan)" >> /content/child_c.log; sleep 20
done
