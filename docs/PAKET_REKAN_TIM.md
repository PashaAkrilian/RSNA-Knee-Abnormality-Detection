# Paket utk rekan satu tim (nxtirfan) — jalankan probe "Apex + sibling r224" dari AKUN-nya (kuota GPU-nya sendiri)
> Sah menurut Rules: berbagi kode di dalam tim diperbolehkan; notebook ini HANYA memakai dataset publik (nartaa, goodpjw) + data kompetisi. Memakai 1 dari 5 submisi/hari tim → koordinasikan dgn user.
> Berkas: `kaggle_kernel_oai_fam_r224/rsna-knee-oai-fam-r224.ipynb` (di repo). Kirim berkas .ipynb itu ke rekan (chat/Drive tim).

## Langkah (±10 menit kerja + ±1,5 jam menunggu skor)
1. Kaggle → **New Notebook** → *File → Import Notebook* → unggah `rsna-knee-oai-fam-r224.ipynb`.
2. **Add Input**: Competition `rsna-knee-abnormality-detection`; Dataset `nartaa/rsna-knee-publication-swa-weights-20261007`; Dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader`.
3. Settings → **Accelerator: GPU T4 x2**, **Internet: Off**.
4. *Save Version → Save & Run All (Commit)*. Tunggu COMPLETE (commit-run memakai 3 studi stub, ±10–20 menit). Bila ERROR, kirim log ke user/agen.
5. Setelah COMPLETE: *Submit to Competition* pada versi itu (deskripsi: "Apex: r384 .75 + sibling r224 .25 + reader goodpjw"). Skor muncul ±1–1,5 jam.
6. Beri tahu user skornya. Aturan terima: **≥0,951** (pembanding Apex murni = 0,950).

Catatan: notebook ini = pipeline Apex (nartaa 0,949 + reader goodpjw per-label) + arm tambahan sibling r224 (crop tengah 307→224, 1 tampilan; hash ckpt dicek). Isi diuji di kernel CPU Kaggle (kernel `rsna-knee-smoke-r224-cpu`) sebelum dikirim.
