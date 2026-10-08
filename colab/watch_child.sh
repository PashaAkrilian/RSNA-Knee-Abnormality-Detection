#!/bin/bash
# Lokal (latar/Monitor): pantau log latih anak di VM Colab; cetak HANYA kejadian penting. Otomatis readopt bila catatan sesi lokal terhapus.
# Kejadian: EPOCH, RESUME, [gold-58, GAGAL, Traceback, DONE, SYNC, VM hilang.  Pakai:  bash colab/watch_child.sh [log_vm=/content/child_c.log]
export PATH="$HOME/.local/bin:$PATH"; cd "$(dirname "$0")/.." || exit 1
LOGVM="${1:-/content/child_c.log}"; PY=~/.local/share/uv/tools/google-colab-cli/bin/python; SEEN=""
T=$(mktemp /tmp/wc_XXXX.py); printf "import subprocess\nprint(subprocess.run(\"tail -n 40 %s; echo; tail -n 3 /content/sync.log 2>/dev/null\", shell=True, capture_output=True, text=True).stdout)\n" "$LOGVM" > "$T"
while true; do
  out=$(timeout 100 colab exec -s rsna -f "$T" --timeout 60 2>&1)
  if echo "$out" | grep -qiE "not found|no such session|Session .* not"; then echo "$(date +%T) sesi lokal hilang -> readopt"; $PY colab/readopt.py rsna 2>&1 | tail -1; sleep 20; continue; fi
  echo "$out" | grep -E "EPOCH|RESUME|\[gold-58|GAGAL|Traceback|TRAINING DONE|SYNC|Error" | while read -r l; do
    k=$(echo "$l" | md5sum | cut -c1-12); case "$SEEN" in *$k*) ;; *) echo "$(date +%T) $l"; SEEN="$SEEN $k";; esac
  done
  echo "$out" | grep -q "TRAINING DONE" && { echo "$(date +%T) SELESAI"; exit 0; }
  sleep 300
done
