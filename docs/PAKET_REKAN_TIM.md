# Paket utk rekan satu tim (nxtirfan) — jalankan job dari AKUN-nya sendiri (kuota GPU-nya sendiri)
> **Diperbarui 11 Okt 2026 04:00 WIB (sesi Claude Code).** Kandidat TERBARU: varian S6 (`kaggle_kernel_s6reader`, `kaggle_kernel_s6plus`) — lihat tabel; kandidat Apex+OF lama (apexof2/apexof4*) diturunkan prioritasnya (OF belum terbukti; resep heliosli = Raptor + compact). Cara yang bersih menambah komputasi: rekan menjalankan notebook DARI AKUNNYA SENDIRI (ia = individu terpisah; Rules 5.b). Agen/user TIDAK memakai tokennya dari komputer ini
> (token di `akun_kaggle/` = akses penuh ke akunnya; pastikan ia setuju atau hapus). Berbagi kode di dalam tim boleh (Rules 5.d); notebook di bawah hanya memakai dataset publik + data kompetisi.
> Run scoring TIDAK memakai kuota GPU; kuota GPU hanya dipakai commit-run (stub 3 studi, ±10 menit) — jadi bantuan komputasi terbesar rekan = menjalankan job latih/probe, bukan submisi.

## Kandidat yang bisa dijalankan rekan (semua dari repo ini)
| Notebook | Isi | Dataset yang harus ditambahkan | Catatan |
|---|---|---|---|
| `kaggle_kernel_s6reader/rsna-knee-s6reader.ipynb` | S6 heliosli (Raptor + compact) + reader goodpjw pada perna Raptor (bobot Apex) | `heliosli/rsna-knee-compact-v1`, `nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, `goodpjw2008/rsna-knee-2-5d-convnext-reader` | ±4,5–5 j (compact GPU-bound ±3,5 j); prioritas 1; mode `lenient` |
| `kaggle_kernel_s6plus/rsna-knee-s6plus.ipynb` | + perna OrthoFoundation fold0 (supervisor v4.1) sbg tahap TERAKHIR dgn deadline (STOP 7,9 j/KILL 8,25 j) | di atas + `dimaspashaakrilian/rsna-knee-ultralytics-cfg` (privat; minta user membagikan atau unggah ulang dari `of_...`) | OF hanya memakai waktu sisa; prior kecil |
| `kaggle_kernel_apexof2/rsna-knee-apexof2.ipynb` | Apex 0,70 + OrthoFoundation-L fold0 0,30 (v2: potongan 8 studi, 4 proses, penjaga disk, pemutus proses) | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, **`dimaspashaakrilian/rsna-knee-ultralytics-cfg`** (dataset privat user → user harus membagikannya ke rekan SATU TIM) | scoring ±6–7 jam; skor hanya bila ≥0,952 → kandidat |
| `kaggle_kernel_oai_fam_r224/...` (lama) | Apex + r224 | nartaa, goodpjw | sudah diuji 0,950 — TIDAK perlu diulang |

## Langkah (±10 menit kerja + menunggu)
1. Kaggle → **New Notebook** → *File → Import Notebook* → unggah `.ipynb` yang dipilih.
2. **Add Input**: Competition `rsna-knee-abnormality-detection` + dataset di tabel. Settings → **Accelerator: GPU T4 x2**, **Internet: Off**.
3. *Save Version → Save & Run All (Commit)*. Tunggu COMPLETE (stub 3 studi, ±5–10 menit). Bila ERROR, kirim log ke user/agen.
4. Setelah COMPLETE: *Submit to Competition* pada versi itu. Memakai 1 dari 5 submisi/hari TIM → koordinasikan dgn user (sisakan jatah).
5. Beri tahu user skornya. Pembanding: klon S6 (≈0,954); selisih <0,002 = noise (LB resolusi 0,001).

## Catatan
- Waktu: S6 ±4–4,5 j (Raptor ±0,5 j + compact ±3,5 j); reader +0,5 j; OF terakhir memakai sisa waktu (±200–290 studi/jam dgn 3–4 pekerja); batas Kaggle 9 jam (Timeout = tanpa skor).
- Jangan bagikan kode/data ke tim LAIN sebelum merger resmi (Rules 5.d/6.a).
