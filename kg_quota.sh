#!/bin/bash
# Cek sisa kuota GPU/TPU (kolom 'remaining') untuk KEDUA akun Kaggle. Jangan pernah cetak isi kaggle.json.
export PATH="$HOME/.local/bin:$PATH"
q() {
  local label="$1" dir="$2" out
  if [ -n "$dir" ]; then
    out=$(KAGGLE_CONFIG_DIR="$dir" kaggle quota 2>&1)
  else
    out=$(kaggle quota 2>&1)
  fi
  if printf '%s' "$out" | grep -q 'remaining'; then
    printf '== %s ==\n' "$label"
    printf '%s\n' "$out" | sed -n '1,6p'
  else
    printf '== %s ==\nGAGAL AUTH (401) -> regenerasi API key di settings/api atau `kaggle auth login`.\n' "$label"
  fi
  echo
}
q "akun utama: dimaspashaakrilian (~/.kaggle)" ""
q "akun kedua: alverlfarfaraway (~/.kaggle-acct2)" "$HOME/.kaggle-acct2"
