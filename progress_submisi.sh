#!/bin/bash
# Laporan progres submisi Kaggle tiap 30 menit. Pakai: bash progress_submisi.sh [ID]
# Jalankan: setsid nohup bash progress_submisi.sh >> progress_submisi.log 2>&1 < /dev/null &
# Henti:    kill $(cat progress_submisi.pid)
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")" || exit 1
ID=${1:-57015077}
TODAY=$(date +%F)
notify() {
  command -v notify-send >/dev/null 2>&1 || return 0
  notify-send -u normal "Kaggle 30 menit" "$1" 2>/dev/null && return 0
  DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" notify-send -u normal "Kaggle 30 menit" "$1" 2>/dev/null || true
}
report() { echo "[$(date '+%d %b %H:%M:%S')] $1"; notify "$1"; }
for i in $(seq 1 13); do
  out=$(bash kg2.sh competitions submissions -c rsna-knee-abnormality-detection 2>/dev/null)
  row=$(printf '%s\n' "$out" | grep "^$ID" || true)
  used=$(printf '%s\n' "$out" | awk -v t="$TODAY" 'NR>2 && $3 ~ "^"t' | wc -l)
  if [ -z "$row" ]; then
    report "cek #$i: baris $ID tak terbaca (auth/rate-limit); jatah hari ini terpakai $used/5"
  else
    st=$(printf '%s' "$row" | grep -o 'SubmissionStatus\.[A-Z]*' | head -1 | cut -d. -f2)
    if [ "$st" = "COMPLETE" ]; then
      sc=$(printf '%s' "$row" | awk '{for(i=1;i<=NF;i++) if($i ~ /^0\.[0-9]+$/){print $i; exit}}')
      report "SKOR MASUK: submisi $ID = ${sc:-?} (COMPLETE). Jatah hari ini $used/5. Monitor selesai."
      exit 0
    fi
    report "cek #$i: submisi $ID status $st (belum ada skor); jatah hari ini $used/5"
  fi
  sleep 1800
done
report "monitor 6,5 jam selesai; skor $ID belum masuk — jalankan ulang bila perlu."
