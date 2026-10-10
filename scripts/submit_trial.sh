#!/bin/bash
# Submete o kernel de percobaan v4 (OF primeiro; modo loud) APOS o reset diario de submissoes (11 Okt 07:03 WIB) e acompanha o placar a cada 5 min.
# Uso: submit_trial.sh <slug> <versao> "<mensagem>"   (so submete se kernel COMPLETE e restarem >= 3 submissoes hoje)
export PATH="$HOME/.local/bin:$PATH"
cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection || exit 1
SLUG="$1"; VER="$2"; MSG="$3"
TARGET=$(date -d '2026-10-11 07:03:00' +%s)
echo "$(date '+%F %T') aguardando ate $(date -d @$TARGET '+%F %T') para submeter $SLUG v$VER"
while [ "$(date +%s)" -lt "$TARGET" ]; do sleep 60; done
echo "$(date '+%F %T') hora de submeter; verificando condicoes"
st=$(kaggle kernels status dimaspashaakrilian/$SLUG 2>&1 | grep -oE 'COMPLETE|RUNNING|ERROR|QUEUED' | head -1)
lim=$(kaggle competitions submission-limits rsna-knee-abnormality-detection 2>&1 | tr '\n' ' ')
echo "kernel=$st | $lim"
rem=$(echo "$lim" | grep -oE 'Remaining today: [0-9]+' | grep -oE '[0-9]+')
if [ "$st" != "COMPLETE" ] || [ -z "$rem" ] || [ "$rem" -lt 3 ]; then
  echo "CONDICAO NAO ATENDIDA (kernel=$st remaining=$rem); NAO submetido"; exit 2
fi
kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/$SLUG -v "$VER" -f submission.csv -m "$MSG" 2>&1 | tail -3
sleep 25
ID=$(kaggle competitions submissions rsna-knee-abnormality-detection -v 2>&1 | sed -n '2p' | cut -d, -f1)
echo "ID da submissao: $ID"
[ -z "$ID" ] && exit 3
for i in $(seq 1 130); do
  line=$(kaggle competitions submissions rsna-knee-abnormality-detection -v 2>&1 | grep "^$ID" | cut -c1-260)
  echo "$(date +%H:%M) $line"
  case "$line" in *COMPLETE*|*ERROR*|*FAIL*) break;; esac
  sleep 300
done
