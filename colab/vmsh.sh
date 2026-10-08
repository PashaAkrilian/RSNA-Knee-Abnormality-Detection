#!/bin/bash
# Jalankan perintah shell di VM Colab sesi 'rsna':  colab/vmsh.sh "perintah" [timeout_detik]
# (colab exec hanya menjalankan berkas Python -> bungkus subprocess; keluaran dipangkas 6000 karakter terakhir)
CMD="$1"; TO="${2:-120}"
T=$(mktemp /tmp/vmsh_XXXX.py)
python3 - "$CMD" "$TO" > "$T" <<'PY'
import sys
cmd, to = sys.argv[1], sys.argv[2]
print(f"import subprocess\nr=subprocess.run({cmd!r}, shell=True, capture_output=True, text=True, timeout={int(to)})\nprint((r.stdout+r.stderr)[-6000:])\nprint('[rc]', r.returncode)")
PY
colab exec -s "${SESSION:-rsna}" -f "$T" --timeout "$((TO+30))" 2>&1 | tail -n 80
rm -f "$T"
