#!/bin/bash
# Submete DUAS variantes S6+ apos o reset diario (11 Okt 07:03 WIB) e acompanha os placares a cada 5 min.
# Uso: submit_two.sh <slug1> <versao1> "<msg1>" <slug2> <versao2> "<msg2>"
# Regras do projeto: agente <=2 submissoes/dia; so submete se kernels COMPLETE e restarem >=4 (1a) / >=3 (2a).
export PATH="$HOME/.local/bin:$PATH"
cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection || exit 1
S1="$1"; V1="$2"; M1="$3"; S2="$4"; V2="$5"; M2="$6"
TARGET=$(date -d "${AT:-2026-10-11 07:03:00}" +%s)   # AT='YYYY-MM-DD HH:MM:SS' (WIB) utk hari lain
echo "$(date '+%F %T') aguardando ate $(date -d @$TARGET '+%F %T') para submeter $S1 v$V1 e $S2 v$V2"
while [ "$(date +%s)" -lt "$TARGET" ]; do sleep 60; done
submit_one() {
  slug="$1"; ver="$2"; msg="$3"; need="$4"
  st=$(kaggle kernels status dimaspashaakrilian/$slug 2>&1 | grep -oE 'COMPLETE|RUNNING|ERROR|QUEUED' | head -1)
  lim=$(kaggle competitions submission-limits rsna-knee-abnormality-detection 2>&1 | tr '\n' ' ')
  rem=$(echo "$lim" | grep -oE 'Remaining today: [0-9]+' | grep -oE '[0-9]+')
  echo "$(date '+%F %T') $slug kernel=$st remaining=$rem (precisa >= $need)"
  if [ "$st" != "COMPLETE" ] || [ -z "$rem" ] || [ "$rem" -lt "$need" ]; then echo "CONDICAO NAO ATENDIDA para $slug; NAO submetido"; return 1; fi
  kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/$slug -v "$ver" -f submission.csv -m "$msg" 2>&1 | tail -2
  sleep 25
  ID=$(kaggle competitions submissions rsna-knee-abnormality-detection -v 2>&1 | sed -n '2p' | cut -d, -f1)
  echo "ID submissao $slug: $ID"
  echo "$ID" >> /tmp/rsna_submitted_ids.txt
  return 0
}
submit_one "$S1" "$V1" "$M1" 4
sleep 40
submit_one "$S2" "$V2" "$M2" 3
IDS=$(cat /tmp/rsna_submitted_ids.txt 2>/dev/null | sort -u)
for i in $(seq 1 140); do
  done_all=1
  for ID in $IDS; do
    line=$(kaggle competitions submissions rsna-knee-abnormality-detection -v 2>&1 | grep "^$ID" | cut -c1-230)
    echo "$(date +%H:%M) $line"
    case "$line" in *COMPLETE*|*ERROR*|*FAIL*) ;; *) done_all=0;; esac
  done
  [ "$done_all" = "1" ] && break
  sleep 300
done
