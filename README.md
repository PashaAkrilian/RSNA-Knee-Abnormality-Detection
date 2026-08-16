# RSNA Knee Abnormality Detection — EDA Workspace

Workspace lokal untuk kompetisi Kaggle: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## Ringkasan kompetisi

- **Task:** memprediksi probabilitas 12 abnormalitas lutut per studi MRI:
  `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.
- **Reward:** $77,000 · **Deadline:** 2026-10-22 · **Kategori:** Research · ~1600+ tim.
- **Struktur data:**
  - `train.csv` — `StudyInstanceUID`, `Report` (teks laporan radiologi, multi-bahasa), + 12 kolom label.
  - `train_series.csv` — metadata tiap MRI series: `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` (Axial/Sagittal/Coronal).
  - `train_series/<StudyUID>/<SeriesUID>/*.dcm` — file DICOM per slice.
  - `test.csv`, `test_series.csv`, `test_series/` — struktur sama, tapi versi publik hanya berisi sampel kecil (khas *code competition*: test asli disembunyikan saat scoring).
  - `sample_submission.csv` — format submission (probabilitas 0–1 per label per studi).
- **Temuan kunci dari eksplorasi awal:** ~98–99% dari 12 kolom label di `train.csv` **kosong (NaN)** — hanya sebagian kecil studi (±1–2%) yang punya label pathology terstruktur lengkap. Mayoritas studi hanya punya teks `Report` radiologi (multi-bahasa: Spanish, Dutch/German, English, dll). Ini mengindikasikan skema **weak/semi-supervised learning** — strategi modeling perlu memanfaatkan teks laporan, bukan cuma label terstruktur yang jarang.

## Struktur folder

```
kaggle_kernel/            notebook EDA + kernel-metadata.json untuk push ke Kaggle
reference_metadata/       salinan lokal CSV metadata (hasil unduhan via Kaggle API, untuk referensi cepat offline)
eda_results/              (dibuat setelah pull) hasil eksekusi notebook dari Kaggle: notebook + gambar + summary
```

## Alur kerja: EDA jalan di Kaggle, ditarik ke lokal

Karena dataset berisi ribuan file DICOM (besar & lambat diunduh dari sandbox ini), notebook EDA didesain untuk **dieksekusi di server Kaggle** (akses data instan, gratis compute), lalu hasilnya (grafik, tabel ringkasan, notebook dengan output ter-render) ditarik kembali ke lokal.

### 1. Push notebook ke Kaggle sebagai kernel

```bash
cd kaggle_kernel
kaggle kernels push -p .
```

### 2. Cek status eksekusi

```bash
kaggle kernels status dimaspashaakrilian/rsna-knee-abnormality-detection-eda
```

### 3. Setelah selesai (`complete`), tarik hasilnya ke lokal

```bash
kaggle kernels output dimaspashaakrilian/rsna-knee-abnormality-detection-eda -p ../eda_results
```

Ini akan mengunduh:
- `__notebook__.ipynb` — notebook lengkap dengan semua output (grafik, tabel) ter-render.
- `__results__.html` — versi HTML siap-lihat dari hasil eksekusi.
- `eda_outputs/*.png` — semua grafik yang disimpan eksplisit oleh notebook.
- `eda_outputs/eda_summary.csv` — tabel ringkasan angka kunci.

### Iterasi ulang

Edit `kaggle_kernel/rsna_knee_eda.ipynb`, lalu ulangi langkah 1–3. Versi kernel akan bertambah otomatis (`kaggle kernels push` selalu membuat versi baru).

## Audit kualitas & anomali DICOM (kernel kedua)

Kernel terpisah: `kaggle_kernel_audit/rsna_knee_dicom_audit.ipynb` → [`rsna-knee-dicom-quality-audit`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-dicom-quality-audit) di Kaggle. Sama pola push/pantau/tarik seperti EDA, hasil di `audit_results/`.

Audit **sampel besar terstratifikasi** (bukan 820rb file penuh — keputusan sadar, exhaustive butuh ~9-12 jam/beberapa run): seluruh 342 series dari 58 studi berlabel-gold + tambahan stratifikasi plane/sekuens, 10.701 instance diperiksa (validitas file, tag wajib, blur/exposure/contrast dgn threshold terkalibrasi per grup sekuens, dimensi, duplikat, konsistensi antar-slice).

**Hasil:** 98.3% bersih, 0 file corrupt, 0 exception, ID CSV↔disk 100% konsisten, 51/64 studi sampel "usable" + 13 "needs_review" (0 "exclude"). Detail lengkap di `audit_results/audit_outputs/*.csv`.

**Catatan operasional penting:** `kaggle kernels status/logs/output` bisa lag **15-20+ menit** di belakang eksekusi sebenarnya — kernel yang selesai dalam 3 menit compute baru terlihat "COMPLETE" dari API jauh belakangan. Jangan buru-buru simpulkan hang dari status API saja.

## Audit EXHAUSTIVE — semua 820rb file (kernel ketiga)

Kernel: `kaggle_kernel_audit_full/rsna_knee_dicom_audit_full.ipynb` → [`rsna-knee-dicom-full-exhaustive-audit`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-dicom-full-exhaustive-audit). Paralel (4 CPU worker), resumable via checkpoint, **selesai 100% dalam satu run, ~2.77 jam**. Hasil di `audit_results_exhaustive/` (CSV ringkasan; file per-instance mentah ~440MB cuma ada di Kaggle, tidak ditarik ke lokal karena disk lokal nyaris penuh).

**Hasil: 819.078 instance, 94.1% bersih.** Per studi (4.407 total): 2.898 usable, 1.432 needs_review, 77 exclude. Khusus 58 studi gold-label: 38 usable, 19 needs_review, **1 exclude** (`...72853333220043794904856138561095171921` — 4/5 series-nya 67-100% OVEREXPOSED, kemungkinan anomali kalibrasi/rescale, perlu review manual sebelum dipakai training).

Temuan langka: 26 DUPLICATE_SLICE (terkonsentrasi di 2 series non-gold), 20 ORIENTATION_ANOMALY (1 series non-gold), 5 EMPTY_OR_CONSTANT, 2 CORRUPT_FILE — semua di studi **non-gold-label**, dampaknya kecil ke data prioritas.

## Preprocessing — siap pakai untuk notebook modeling (kernel keempat)

Kernel: `kaggle_kernel_preprocessing/rsna_knee_preprocessing.ipynb` → [`rsna-knee-preprocessing-pipeline`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-preprocessing-pipeline). **Selesai 100%, 24.371/24.371 series, 0 error, 13.05 GB, dalam 54 menit** (paralel 4 worker, ~478-490 series/menit).

**Isi output** (semua tetap di Kaggle, TIDAK di-download ke lokal — 13GB tidak muat di disk lokal yang nyaris penuh):
- `image_cache/<StudyUID>/<SeriesUID>.npz` — 16 slice/series, sudah resample fisik (PixelSpacing→0.5mm/px), clip per-grup-sekuens, resize 224×224, dedup exact-duplicate. Array `(16,224,224)` uint8, siap `np.load(path)['slices']`.
- `series_manifest.csv` — metadata per series + `Laterality` + status QC
- `study_manifest.csv` — label + teks laporan yang sudah dibersihkan + bahasa, per studi
- `group_norm_stats.csv` — statistik normalisasi per grup sekuens

**Temuan penting — Laterality (akhirnya terjawab, dari 24.371 series penuh):** R=6.047 (24.8%), L=5.953 (24.4%), kosong=12.367 (50.7%). **Dataset benar-benar campur kiri-kanan** — flip augmentation horizontal wajib memperhitungkan ini (tukar label Medial↔Lateral kalau di-flip tanpa penyesuaian).

### Cara pakai di notebook modeling lain

Tidak perlu bikin Kaggle Dataset terpisah — cukup tambahkan di `kernel-metadata.json` notebook modeling:
```json
"kernel_sources": ["dimaspashaakrilian/rsna-knee-preprocessing-pipeline"]
```
Output kernel ini otomatis ter-mount di `/kaggle/input/rsna-knee-preprocessing-pipeline/` saat notebook modeling dijalankan.

**Catatan operasional:** kernel dengan puluhan ribu file output (seperti ini) membuat `kaggle kernels output`/`files` gampang kena rate-limit (429) karena API perlu enumerasi semua file — `kaggle kernels logs` tetap jalan normal (cuma streaming teks, bukan daftar file) dan cukup untuk memantau progress/hasil akhir tanpa perlu file listing.

## Catatan teknis

- Kernel di-set `enable_internet: true` supaya bisa `pip install pydicom`/`langdetect` bila belum tersedia di image dasar Kaggle.
- `competition_sources` di `kernel-metadata.json` menautkan dataset kompetisi otomatis ke `/kaggle/input/...` saat kernel jalan — notebook mendeteksi path ini secara otomatis (auto-discover), dengan fallback ke folder `data/` lokal bila dijalankan di luar Kaggle.
- Notebook memakai **sampling** untuk bagian yang menyentuh file DICOM (bukan iterasi ke semua ribuan file) supaya eksekusi tetap cepat.
