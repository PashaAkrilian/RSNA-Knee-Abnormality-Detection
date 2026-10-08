#!/bin/bash
# Lokal (latar): tunggu bootstrap VM selesai, lalu luncurkan latih anak (run_child.sh) + pencadangan (sync_ckpt.py) di VM Colab.
export PATH="$HOME/.local/bin:$PATH"; cd "$(dirname "$0")/../.." || exit 1
T=$(mktemp /tmp/lw_XXXX.py); printf "import subprocess\nprint(subprocess.run('tail -n 3 /content/bootstrap.log', shell=True, capture_output=True, text=True).stdout)\n" > "$T"
for _ in $(seq 1 240); do
  out=$(timeout 100 colab exec -s rsna -f "$T" --timeout 60 2>&1)
  if echo "$out" | grep -q "BOOTSTRAP DONE"; then break; fi
  if echo "$out" | grep -q "BOOTSTRAP INCOMPLETE"; then echo "BOOTSTRAP TIDAK LENGKAP: $out"; exit 1; fi
  sleep 60
done
echo "$out" | grep -q "BOOTSTRAP DONE" || { echo "TIMEOUT bootstrap"; exit 2; }
echo "$(date +%T) bootstrap selesai -> unggah skrip & luncurkan"
colab upload colab/oai_child/child_train_colab.py /content/child_train_colab.py -s rsna 2>&1 | tail -1
colab upload colab/oai_child/run_child.sh /content/run_child.sh -s rsna 2>&1 | tail -1
colab upload colab/oai_child/sync_ckpt.py /content/sync_ckpt.py -s rsna 2>&1 | tail -1
L=$(mktemp /tmp/ll_XXXX.py); cat > "$L" <<'PY'
import subprocess
subprocess.run("mkdir -p /content/kw; rm -f /content/kw/*; cd /content; (nohup bash /content/run_child.sh > /content/run_child.out 2>&1 &); (nohup python3 -u /content/sync_ckpt.py > /content/sync.log 2>&1 &); sleep 40; tail -n 8 /content/child_c.log", shell=True)
print(subprocess.run("tail -n 8 /content/child_c.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout)
PY
colab exec -s rsna -f "$L" --timeout 120 2>&1 | tail -n 14
echo "$(date +%T) DILUNCURKAN"
