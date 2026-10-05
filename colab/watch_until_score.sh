#!/usr/bin/env bash
# Latar: sinkronkan fold Colab yang selesai ke Kaggle; KELUAR saat ada submission hari ini yang berubah dari PENDING
# (agar Claude diberi tahu). Log: /tmp/rsna_watch.log
export PATH="$HOME/.local/bin:$PATH"; cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
SYNCED="${SYNCED:-0 1 2}"
while true; do
  out=$(echo "print(open('/content/train.log').read())" | timeout 120 colab exec -s rsna 2>&1)
  if echo "$out" | grep -q "not found"; then echo "$(date +%T) readopt" >> /tmp/rsna_watch.log; $PY colab/readopt.py rsna >> /tmp/rsna_watch.log 2>&1; fi
  for f in $(echo "$out" | grep -oE "FOLD [0-9] SELESAI" | grep -oE "[0-9]"); do
    if ! echo " $SYNCED " | grep -q " $f "; then
      echo "$(date +%T) fold $f selesai -> sync" >> /tmp/rsna_watch.log
      timeout 900 python3 colab/sync_results.py cheap_radresnet50_ep12_s1 rsna-knee-radresnet50-ep12 rsna >> /tmp/rsna_watch.log 2>&1 && SYNCED="$SYNCED $f"
    fi
  done
  echo "$out" | grep -q "WORKER DONE" && echo "$(date +%T) WORKER DONE" >> /tmp/rsna_watch.log
  done_subs=$(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | python3 -c "
import sys,csv
for r in list(csv.DictReader(sys.stdin))[:5]:
    if r['date']>='2026-10-05' and r['status'].split('.')[-1]!='PENDING': print(r['ref'], r['status'].split('.')[-1], r['publicScore'] or '-', r['description'][:50])")
  if [ -n "$done_subs" ]; then echo "SUBMISSION SELESAI:"; echo "$done_subs"; echo "fold tersinkron: $SYNCED"; exit 0; fi
  sleep 420
done
