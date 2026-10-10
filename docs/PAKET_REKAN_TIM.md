# Paket utk rekan satu tim (nxtirfan) — jalankan job dari AKUN-nya sendiri (kuota GPU-nya sendiri)
> **Diperbarui 10 Okt 2026 (sesi Claude Code).** Cara yang bersih menambah komputasi: rekan menjalankan notebook DARI AKUNNYA SENDIRI (ia = individu terpisah; Rules 5.b). Agen/user TIDAK memakai tokennya dari komputer ini
> (token di `akun_kaggle/` = akses penuh ke akunnya; pastikan ia setuju atau hapus). Berbagi kode di dalam tim boleh (Rules 5.d); notebook di bawah hanya memakai dataset publik + data kompetisi.
> Run scoring TIDAK memakai kuota GPU; kuota GPU hanya dipakai commit-run (stub 3 studi, ±10 menit) — jadi bantuan komputasi terbesar rekan = menjalankan job latih/probe, bukan submisi.

## Kandidat yang bisa dijalankan rekan (semua dari repo ini)
| Notebook | Isi | Dataset yang harus ditambahkan | Catatan |
|---|---|---|---|
| `kaggle_kernel_apexof2/rsna-knee-apexof2.ipynb` | Apex 0,70 + OrthoFoundation-L fold0 0,30 (v2: potongan 8 studi, 4 proses, penjaga disk, pemutus proses) | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, **`dimaspashaakrilian/rsna-knee-ultralytics-cfg`** (dataset privat user → user harus membagikannya ke rekan SATU TIM) | scoring ±6–7 jam; skor hanya bila ≥0,952 → kandidat |
| `kaggle_kernel_oai_fam_r224/...` (lama) | Apex + r224 | nartaa, goodpjw | sudah diuji 0,950 — TIDAK perlu diulang |

## Langkah (±10 menit kerja + menunggu)
1. Kaggle → **New Notebook** → *File → Import Notebook* → unggah `.ipynb` yang dipilih.
2. **Add Input**: Competition `rsna-knee-abnormality-detection` + dataset di tabel. Settings → **Accelerator: GPU T4 x2**, **Internet: Off**.
3. *Save Version → Save & Run All (Commit)*. Tunggu COMPLETE (stub 3 studi, ±5–10 menit). Bila ERROR, kirim log ke user/agen.
4. Setelah COMPLETE: *Submit to Competition* pada versi itu. Memakai 1 dari 5 submisi/hari TIM → koordinasikan dgn user (sisakan jatah).
5. Beri tahu user skornya. Aturan terima: **≥0,952** (Apex murni = 0,950).

## Catatan
- Estimasi waktu OrthoFoundation dari uji skala (≈15 dtk/studi agregat dgn 4 proses → ±5,5 j utk 1.322 studi) + Apex ±0,5 j; batas Kaggle 9 jam (Timeout = tanpa skor).
- Jangan bagikan kode/data ke tim LAIN sebelum merger resmi (Rules 5.d/6.a).
