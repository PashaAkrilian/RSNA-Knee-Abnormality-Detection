#!/bin/bash
# Jalankan kaggle CLI di akun KEDUA (alverlfarfaraway). Pemakaian utk RSNA hanya bila izin
# tertulis panitia/Kaggle sudah diarsipkan (lihat MEMORY.md §5 butir 4 & docs/).
# Akun utama (dimaspashaakrilian) tetap default: pakai `kaggle ...` biasa.
# Pakai:  bash kg2.sh quota   |  bash kg2.sh kernels list --mine  |  bash kg2.sh kernels push -p <dir>
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle-acct2}"
exec kaggle "$@"
