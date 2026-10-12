#!/bin/bash
# Pantau kuota GPU 3 akun Kaggle (main / alver / nxtirfan) — append ke data/quota_watch.log.
# Jalankan: setsid nohup bash scripts/quota_watch.sh >/dev/null 2>&1 &
# Hentikan: kill $(pgrep -f quota_watch.sh)
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT=/home/pasha/Documents/RSNA-Knee-Abnormality-Detection
LOG="$ROOT/data/quota_watch.log"
INTERVAL=${INTERVAL:-300}          # detik antar cek (default 5 mnt)
DIR_MAIN="$HOME/.kaggle"
DIR_ALVER="$HOME/.kaggle-acct2"
DIR_IRFAN="$ROOT/akun_kaggle"

gpu() { # gpu <KAGGLE_CONFIG_DIR> -> "used|remaining|total|refresh"
  KAGGLE_CONFIG_DIR="$1" kaggle quota 2>/dev/null \
    | awk '/^GPU /{printf "%s|%s|%s|%s", $2, $3, $4, $5; exit}'
}

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "WATCH_START interval=${INTERVAL}s"
while true; do
  m=$(gpu "$DIR_MAIN"); a=$(gpu "$DIR_ALVER"); i=$(gpu "$DIR_IRFAN")
  say "main[$m] alver[$a] nxtirfan[$i]"
  sleep "$INTERVAL"
done
