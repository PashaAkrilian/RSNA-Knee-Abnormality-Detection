#!/bin/bash
# Orkestrator malam pipeline murid RSNA (12 Okt 2026).
# a (alver fold0,1) + b (main fold2,3) + dino (irfan) PARALEL:
#   a COMPLETE -> push c (alver fold4) + tarik fold0,1
#   b COMPLETE -> tarik fold2,3 ; dino COMPLETE -> tarik dino_full.pt
#   semua ckpt -> dataset alver rsna-knee-student-folds -> push infer (alver)
#   infer COMPLETE -> tarik submission.csv + student_test_preds.csv
#     -> version dataset student-preds -> submit murid solo (alver)
#     -> push s6student (alver) -> submit s6student (alver)
# Jangan sentuh kernel vlm-edge-*. Log: /tmp/opencode/night.log
set -u
export PATH="$HOME/.local/bin:$PATH"
cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection || exit 1
KG2="bash kg2.sh"; KG3="bash kg3.sh"
export KAGGLE_CONFIG_DIR_MAIN="$HOME/.kaggle"
LOG=/tmp/opencode/night.log
DL=/tmp/opencode/night_dl
mkdir -p "$DL"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

st2() { $KG2 kernels status "$1" 2>/dev/null | grep -oE 'COMPLETE|ERROR|RUNNING|QUEUED' | head -1; }
st3() { $KG3 kernels status "$1" 2>/dev/null | grep -oE 'COMPLETE|ERROR|RUNNING|QUEUED' | head -1; }
stM() { KAGGLE_CONFIG_DIR="$KAGGLE_CONFIG_DIR_MAIN" kaggle kernels status "$1" 2>/dev/null | grep -oE 'COMPLETE|ERROR|RUNNING|QUEUED' | head -1; }

wait_for() { # wait_for <fn> <ref> <jam-maks>
  local fn="$1" ref="$2" max="${3:-9}" t0=$(date +%s)
  while true; do
    local s; s=$("$fn" "$ref")
    if [ "$s" = "COMPLETE" ] || [ "$s" = "ERROR" ]; then echo "$s"; return; fi
    if [ $(( $(date +%s) - t0 )) -gt $(( max * 3600 )) ]; then echo "TIMEOUT"; return; fi
    sleep 600
  done
}

say "START night orchestrator"

# ---------- tunggu paralel a/b/dino ----------
SA=""; SB=""; SD=""
while [ -z "$SA" ] || [ -z "$SB" ] || [ -z "$SD" ]; do
  [ -z "$SA" ] && { x=$(st2 alverlfarfaraway/rsna-knee-student-train-a); case "$x" in COMPLETE|ERROR) SA=$x; say "train-a=$x";; esac; }
  [ -z "$SB" ] && { x=$(stM dimaspashaakrilian/rsna-knee-student-train-b); case "$x" in COMPLETE|ERROR) SB=$x; say "train-b=$x";; esac; }
  [ -z "$SD" ] && { x=$(st3 nxtirfan/rsna-knee-student-dinov2-s-one-shot); case "$x" in COMPLETE|ERROR) SD=$x; say "dino=$x";; esac; }
  # begitu train-a selesai: push train-c (retry bila batch limit)
  if [ "$SA" = "COMPLETE" ] && [ ! -f "$DL/.c_pushed" ]; then
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
      out=$($KG2 kernels push -p kaggle_kernel_student-train-c 2>&1 | tail -1)
      say "push train-c try$i: $out"
      case "$out" in *"successfully pushed"*) touch "$DL/.c_pushed"; break;; esac
      sleep 600
    done
  fi
  [ -z "$SA" ] || [ -z "$SB" ] || [ -z "$SD" ] && sleep 300
done

# ---------- tarik outputs ----------
if [ "$SA" = "COMPLETE" ]; then
  $KG2 kernels output alverlfarfaraway/rsna-knee-student-train-a -p "$DL/a" -q >> "$LOG" 2>&1
  say "dl train-a done: $(find "$DL/a" -name 'fold*.pt' | tr '\n' ' ')"
fi
if [ "$SB" = "COMPLETE" ]; then
  KAGGLE_CONFIG_DIR="$KAGGLE_CONFIG_DIR_MAIN" kaggle kernels output dimaspashaakrilian/rsna-knee-student-train-b -p "$DL/b" -q >> "$LOG" 2>&1
  say "dl train-b done: $(find "$DL/b" -name 'fold*.pt' | tr '\n' ' ')"
fi
if [ "$SD" = "COMPLETE" ]; then
  $KG3 kernels output nxtirfan/rsna-knee-student-dinov2-s-one-shot -p "$DL/d" -q >> "$LOG" 2>&1
  say "dl dino done: $(find "$DL/d" -name '*.pt' | tr '\n' ' ')"
fi

# ---------- tunggu train-c ----------
SC=""
while [ "$SA" = "COMPLETE" ] && [ -z "$SC" ]; do
  x=$(st2 alverlfarfaraway/rsna-knee-student-train-c)
  case "$x" in COMPLETE|ERROR) SC=$x; say "train-c=$x";; esac
  [ -z "$SC" ] && sleep 600
done
if [ "$SC" = "COMPLETE" ]; then
  $KG2 kernels output alverlfarfaraway/rsna-knee-student-train-c -p "$DL/c" -q >> "$LOG" 2>&1
  say "dl train-c done: $(find "$DL/c" -name 'fold*.pt' | tr '\n' ' ')"
fi

# ---------- dataset folds ----------
FOLDS_FOUND=$(find "$DL" -name 'fold[0-9].pt' | wc -l)
DINO_FOUND=$(find "$DL" -name 'dino_full.pt' | wc -l)
say "folds=$FOLDS_FOUND dino=$DINO_FOUND"
if [ "$FOLDS_FOUND" -lt 5 ]; then
  say "FOLD TIDAK LENGKAP — stop sebelum folds-dataset/infer"; exit 1
fi
FDS=/tmp/opencode/folds_ds
rm -rf "$FDS"; mkdir -p "$FDS"
for f in $(find "$DL" -name 'fold[0-9].pt'); do cp "$f" "$FDS/"; done
[ "$DINO_FOUND" -gt 0 ] && cp "$(find "$DL" -name 'dino_full.pt' | head -1)" "$FDS/"
cat > "$FDS/dataset-metadata.json" <<'JSON'
{"id": "alverlfarfaraway/rsna-knee-student-folds", "title": "RSNA Knee student folds", "licenses": [{"name": "Other"}], "keywords": ["rsna","knee","ckpt"], "resources": []}
JSON
$KG2 datasets version -p "$FDS" -m "fold0-4 + dino_full" >> "$LOG" 2>&1
say "folds dataset version pushed"

# ---------- infer ----------
for i in 1 2 3 4 5 6; do
  out=$($KG2 kernels push -p kaggle_kernel_student_infer 2>&1 | tail -1)
  say "push infer try$i: $out"
  case "$out" in *"successfully pushed"*) break;; esac
  sleep 600
done
SI=$(wait_for st2 alverlfarfaraway/rsna-knee-student-infer 4)
say "infer=$SI"
[ "$SI" != "COMPLETE" ] && { say "INFER GAGAL — stop"; exit 1; }
$KG2 kernels output alverlfarfaraway/rsna-knee-student-infer -p "$DL/inf" -q >> "$LOG" 2>&1
say "dl infer: $(ls "$DL/inf" 2>/dev/null | tr '\n' ' ')"

# ---------- preds dataset + submit murid solo ----------
if [ -f "$DL/inf/student_test_preds.csv" ]; then
  PDS=/tmp/opencode/preds_ds
  rm -rf "$PDS"; mkdir -p "$PDS"
  cp "$DL/inf/student_test_preds.csv" "$PDS/"
  cat > "$PDS/dataset-metadata.json" <<'JSON'
{"id": "alverlfarfaraway/rsna-knee-student-preds", "title": "RSNA Knee student test preds", "licenses": [{"name": "Other"}], "keywords": ["rsna","knee","preds"], "resources": []}
JSON
  $KG2 datasets version -p "$PDS" -m "student_test_preds (cnxt+dino)" >> "$LOG" 2>&1
  say "preds dataset version pushed"
else
  say "student_test_preds.csv TIDAK ada di output infer"
fi

submit_wait() { # submit_wait <cfgdir> <slug> <ver> <msg>
  local cfg="$1" slug="$2" ver="$3" msg="$4"
  while true; do
    local lim rem
    lim=$(KAGGLE_CONFIG_DIR="$cfg" kaggle competitions submission-limits rsna-knee-abnormality-detection 2>&1 | tr '\n' ' ')
    rem=$(echo "$lim" | grep -oE 'Remaining today: [0-9]+' | grep -oE '[0-9]+')
    say "cek limit $slug: rem=$rem"
    if [ -n "${rem:-}" ] && [ "$rem" -ge 1 ]; then
      out=$(KAGGLE_CONFIG_DIR="$cfg" kaggle competitions submit rsna-knee-abnormality-detection -k "$slug" -v "$ver" -f submission.csv -m "$msg" 2>&1 | tail -2)
      say "SUBMIT $slug v$ver: $out"
      return 0
    fi
    sleep 600
  done
}
submit_wait "$HOME/.kaggle-acct2" "alverlfarfaraway/rsna-knee-student-infer" "1" "murid solo convnext+dino"

# ---------- s6student ----------
for i in 1 2 3; do
  out=$($KG2 kernels push -p kaggle_kernel_s6student 2>&1 | tail -1)
  say "push s6student try$i: $out"
  case "$out" in *"successfully pushed"*) break;; esac
  sleep 300
done
SS=$(wait_for st2 alverlfarfaraway/rsna-knee-s6student 4)
say "s6student=$SS"
if [ "$SS" = "COMPLETE" ]; then
  submit_wait "$HOME/.kaggle-acct2" "alverlfarfaraway/rsna-knee-s6student" "1" "S6 0.65 + murid 0.35"
fi
say "NIGHT ORCHESTRATOR SELESAI"
