#!/bin/bash
# Vigia ativos publicos (heliosli/nartaa/goodpjw + notebooks) a cada 30 min; registra mudancas em watch.log
export PATH="$HOME/.local/bin:$PATH"
cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection || exit 1
LOG=/tmp/rsna_watch.log
for i in $(seq 1 48); do
  echo "=== $(date '+%F %T')" >> $LOG
  for u in heliosli nartaa goodpjw2008; do
    echo "-- datasets $u" >> $LOG; kaggle datasets list --user $u --sort-by updated 2>&1 | head -5 | cut -c1-170 >> $LOG
    echo "-- kernels $u" >> $LOG; kaggle kernels list --user $u --sort-by dateRun 2>&1 | head -5 | cut -c1-170 >> $LOG
  done
  timeout 240 python3 scan_public_updates.py 2>&1 | grep -E "^\[|baru:" | head -20 >> $LOG
  sleep 1800
done
