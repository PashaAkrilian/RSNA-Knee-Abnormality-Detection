# Progress — RSNA Knee Abnormality Detection

Ringkasan kronologis pekerjaan yang sudah dilakukan untuk kompetisi [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Semua kode dijalankan di server Kaggle (bukan lokal — dataset 569.76GB tidak mungkin diproses di mesin lokal), hasil ditarik/dirujuk kembali ke repo ini.

## 1. EDA (`kaggle_kernel/`)

Kernel: [`rsna-knee-abnormality-detection-eda`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-abnormality-detection-eda)

- 4.407 studi train, 24.371 series (rata-rata 5.53 series/studi, rata-rata 33.7 slice/series)
- **Temuan kunci: 98.7% dari 12 kolom label kosong** — hanya 58 studi (1.3%) berlabel gold penuh. 100% studi punya `Report` teks radiologi
- Laporan **sangat multi-bahasa**: English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French
- `test.csv` publik cuma 3 studi — pola *code competition*, test asli tersembunyi

Hasil: `eda_results/` (16 grafik + notebook + log)

## 2. Rekomendasi model & strategi (artifact "Cetak Biru Model Lutut")

- Arsitektur: 2D CNN per-slice + attention pooling multi-slice & multi-series
- Strategi data: weak-labeling dari teks laporan (58 studi gold → potensi ~4.400 studi via labeler)
- Analisis DICOM lanjutan: heterogenitas scanner (≥10 vendor, 3 kekuatan medan magnet), 50 kombinasi resolusi unik, intensitas beda ~5× antar tipe sekuens
- Konteks leaderboard: peringkat 1 = **0.950**, top-20 rapat di 0.934–0.950, dari 1.611 tim
- Spesifikasi pipeline preprocessing konkret (jadi acuan tahap 4 di bawah)

## 3. Audit kualitas data — sampel (`kaggle_kernel_audit/`)

Kernel: [`rsna-knee-dicom-quality-audit`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-dicom-quality-audit)

Validasi awal metodologi audit pada 342 series (semua studi gold-label + stratifikasi). Hasil: 98.3% bersih, 0 file corrupt.

Hasil: `audit_results/`

## 4. Audit kualitas data — EXHAUSTIVE, semua 820rb file (`kaggle_kernel_audit_full/`)

Kernel: [`rsna-knee-dicom-full-exhaustive-audit`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-dicom-full-exhaustive-audit)

**Selesai 100% dalam 1 run (~2.77 jam), paralel 4 worker, checkpoint resumable.**

- 819.078 instance diaudit, **94.1% bersih**
- Per studi (4.407): 2.898 usable, 1.432 needs_review, 77 exclude
- **1 dari 58 studi gold-label kena flag "exclude"** (`...72853333...` — 4/5 series-nya 67–100% overexposed sistemik, kemungkinan anomali kalibrasi) — perlu review manual sebelum dipakai training/validasi
- Temuan langka lain (26 duplicate slice, 20 orientation anomaly, 5 empty/constant, 2 corrupt file) — semua di studi **non-gold**, dampak minimal

Hasil: `audit_results_exhaustive/`

## 5. Preprocessing pipeline (`kaggle_kernel_preprocessing/`)

Kernel: [`rsna-knee-preprocessing-pipeline`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-preprocessing-pipeline)

**Selesai 100%, 24.371/24.371 series, 0 error, 13.05 GB, dalam 54 menit.**

Pipeline: RescaleSlope/Intercept → invert kalau MONOCHROME1 → clip per-grup-sekuens → **resample fisik (PixelSpacing→0.5mm/px) sebelum resize** → resize 224×224 → dedup exact-duplicate → 16 slice representatif/series → simpan `.npz` terkompresi.

- **Tag `Laterality` akhirnya terjawab** (dari 24.371 series penuh): R=24.8%, L=24.4%, kosong=50.7% — **dataset benar-benar campur kiri-kanan**, flip augmentation horizontal wajib menukar label Medial↔Lateral kalau dipakai
- Output reusable via `kernel_sources` di kernel-metadata.json notebook lain (tidak perlu Kaggle Dataset terpisah)
- Teks laporan sudah dibersihkan (`study_manifest.csv`) + bahasa terdeteksi per studi

## Belum dikerjakan

- [ ] Review manual studi gold-label yang ter-flag exclude (bagian 4)
- [ ] Weak-labeler dari teks laporan (rule-based/NLP, validasi presisi/recall ke 58 studi gold sebelum skala penuh)
- [ ] Notebook modeling: arsitektur 2D CNN + attention pooling, attach preprocessing via `kernel_sources`
- [ ] Cross-validation stratified multi-label pada 58 studi gold
- [ ] Verifikasi metrik evaluasi resmi kompetisi (halaman Kaggle JS-rendered, belum terkonfirmasi otomatis)

## Struktur repo

```
kaggle_kernel/                 EDA
kaggle_kernel_audit/           Audit kualitas (sampel)
kaggle_kernel_audit_full/      Audit kualitas (exhaustive, 820rb file)
kaggle_kernel_preprocessing/   Pipeline preprocessing
eda_results/                   Output EDA (grafik, notebook, log)
audit_results/                 Output audit sampel
audit_results_exhaustive/      Output audit exhaustive
reference_metadata/            Salinan CSV metadata kompetisi
README.md                      Dokumentasi teknis detail per komponen
PROGRESS.md                    File ini
```
