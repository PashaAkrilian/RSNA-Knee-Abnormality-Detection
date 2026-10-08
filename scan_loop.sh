#!/bin/bash
# Pindai publik tiap 20 menit; hanya item BARU yang ditulis ke ~/.cache/rsna_scan_log.txt (baca saat "cek").
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")" || exit 1
LOG="$HOME/.cache/rsna_scan_log.txt"; mkdir -p "$HOME/.cache"
for _ in $(seq 1 2000); do
  out=$(python3 scan_public_updates.py 2>&1)
  if ! echo "$out" | grep -q "baru: 0 dataset, 0 notebook"; then
    { echo "=== $(date '+%F %T %Z')"; echo "$out"; } >> "$LOG"
  fi
  sleep 1200
done
