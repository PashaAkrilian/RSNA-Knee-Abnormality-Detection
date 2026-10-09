#!/bin/bash
# Jalankan kaggle CLI di akun KETIGA (nxtirfan) — DIPAKAI BILA GPU AKUN KEDUA HABIS
# (perintah user 10 Okt 2026). Kredensial: akun_kaggle/kaggle.json (sudah di-.gitignore;
# JANGAN commit / cetak isinya).
# Pakai:  bash kg3.sh quota   |  bash kg3.sh kernels list --mine  |  bash kg3.sh kernels push -p <dir>
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/Documents/RSNA-Knee-Abnormality-Detection/akun_kaggle}"
exec kaggle "$@"
