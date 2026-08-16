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
kaggle_kernel/                notebook EDA + kernel-metadata.json untuk push ke Kaggle
kaggle_kernel_audit/          audit kualitas data (sampel terstratifikasi)
kaggle_kernel_audit_full/     audit kualitas data (exhaustive, 820rb file)
kaggle_kernel_preprocessing/  pipeline preprocessing (cache gambar siap-pakai)
kaggle_kernel_weak_labeler/   weak-labeler rule-based dari teks laporan
kaggle_kernel_modeling/       modeling (DINOv2-base + slot attention pooling)
reference_metadata/           salinan lokal CSV metadata (hasil unduhan via Kaggle API, untuk referensi cepat offline)
docs/RESEARCH_NOTES.md        riset arsitektur, foundation model, dataset publik
eda_results/, audit_results/, audit_results_exhaustive/, weak_labeler_results/, modeling_results/
                               (dibuat setelah pull) hasil eksekusi tiap kernel dari Kaggle
PROGRESS.md                   log kronologis progres per tahap
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
**Koreksi:** output kernel via `kernel_sources` ter-mount lebih dalam dari dugaan awal —
`/kaggle/input/notebooks/<owner>/<slug>/...`, bukan `/kaggle/input/<slug>/...` langsung. Jangan
`glob` dangkal atau hardcode path; pakai `os.walk` dengan pruning subtree besar (`image_cache`,
`train_series`, `test_series`) untuk cari folder yang mengandung file manifest yang dicari — pola ini
dipakai di semua notebook downstream (`kaggle_kernel_modeling`). **Juga penting:** kalau kernel sumber
menyimpan path file absolut/relatif di kolom CSV manifest-nya (mis. `cache_path`), path itu valid
untuk kernel ASLINYA saja — begitu dipakai ulang lewat `kernel_sources` di kernel lain, cwd & prefix
mount beda, jadi path yang tersimpan jadi tidak valid. Rekonstruksi path dari direktori hasil
auto-discovery + struktur relatif yang diketahui, jangan percaya string path yang tersimpan di CSV
(lihat bug §"Modeling" di bawah — ini yang bikin run v4 gagal belajar apa pun selama berjam-jam
tanpa error eksplisit, karena tertangkap `except Exception` yang terlalu luas).

**Catatan operasional:** kernel dengan puluhan ribu file output (seperti ini) membuat `kaggle kernels output`/`files` gampang kena rate-limit (429) karena API perlu enumerasi semua file — `kaggle kernels logs` tetap jalan normal (cuma streaming teks, bukan daftar file) dan cukup untuk memantau progress/hasil akhir tanpa perlu file listing.

## Weak-labeler dari teks laporan (kernel kelima)

Kernel: `kaggle_kernel_weak_labeler/rsna_knee_weak_labeler.ipynb` → [`rsna-knee-weak-labeler-from-reports`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-weak-labeler-from-reports). Rule-based (bukan LLM API — data pasien sensitif), negation/uncertainty-aware per-klausa, 9 bahasa. Label OA (Medial/Lateral/PF) pakai co-occurrence indikator+lokasi dalam klausa yang sama, bukan exact-phrase match (istilah di laporan asli sering terpisah, terutama non-Inggris).

**Hasil validasi ke 58 studi gold:** agreement 0.747, precision 0.686, recall 0.951 (rata-rata). Output `weak_labels_full.csv` (4.407 studi: 58 gold asli + 4.349 weak-label) — detail di `weak_labeler_results/` dan `PROGRESS.md` §7.

**Catatan operasional:** `langdetect` tidak ter-install default di image Kaggle — pip-install-if-missing eksplisit sebelum import, kalau tidak fallback diam-diam ke bahasa Inggris untuk semua laporan (bug nyata di run pertama).

## Modeling — DINOv2-base + slot attention pooling (kernel keenam)

Kernel: `kaggle_kernel_modeling/rsna_knee_modeling.ipynb` → [`rsna-knee-modeling-v1-dinov2`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-modeling-v1-dinov2). Arsitektur: DINOv2-base (`metaresearch/dinov2` via `model_sources`, 2 blok terakhir di-unfreeze) → CLS+mean-patch pooling per slice → rata-rata antar-slice → bag 6 "slot" (plane × sekuens fluid-sensitive) dengan attention pooling berprior anatomis (`SLOT_PRIOR_TABLE`) → sigmoid 12-label.

**Riwayat run & bug signifikan** (detail penuh di `PROGRESS.md` §8-9):
- v1-v2: error (path `kernel_sources` belum ditemukan; GPU P100 tidak kompatibel dengan build PyTorch — fix: `machine_shape: "NvidiaTeslaT4"` di `kernel-metadata.json`).
- v3: uji skala kecil sukses end-to-end (40 studi, 1 epoch).
- v4: run penuh (5 epoch) selesai tapi `val_macro_auc` mentok tepat ~0.50 di semua epoch.
- v5: push diagnostik murah (tanpa training) — ketemu akar masalah: kolom `cache_path` di manifest preprocessing berisi path relatif yang tidak valid lagi lewat `kernel_sources`, jadi 100% gambar & mask kosong sejak v3, model cuma belajar bias per-label. Silent-caught oleh `except Exception` yang terlalu luas.
- v6: fix path (rekonstruksi dari `PREP_DIR`, plus guard `DATA_OK` yang membatalkan training kalau ternyata masih kosong) **+ ganti sumber label training** dari rule-based labeler sendiri ke dataset publik `stevenleehans/rsna-knee-llm-report-labels` (CC0-1.0, macro AUC 0.89 vs gold, jauh di atas rule-based ~0.71) — lihat `docs/RESEARCH_NOTES.md` §7 untuk metodologi evaluasi & alasan skip dataset publik lain yang ternyata cuma re-package `train.csv` resmi. **SELESAI, fix terkonfirmasi:** `DATA_OK=True`, variasi embedding antar-studi 0.3870 (sebelumnya 0.0000), `val_macro_auc` naik konsisten 3 epoch: 0.6906 → 0.7507 → **0.7658**. Detail penuh & rencana lanjut (lebih banyak epoch, ensemble RadImageNet+BiomedCLIP) di `PROGRESS.md` §8.

**Lever baru: dataset publik hasil labeling LLM offline pihak lain.** Beberapa kompetitor mempublikasikan Kaggle Dataset berisi label report yang lebih akurat dari regex/rule-based (LLM dijalankan offline oleh mereka, hasil dipublikasikan statis, CC0). Sebelum memakai dataset publik seperti ini: **selalu verifikasi ulang klaim kualitasnya sendiri** (jangan percaya begitu saja) terhadap ground-truth yang kita punya, dan periksa apakah dataset itu benar representasi independen atau cuma re-package data resmi kompetisi (kasus `barun2104` di atas).

## Catatan teknis

- Kernel di-set `enable_internet: true` supaya bisa `pip install pydicom`/`langdetect` bila belum tersedia di image dasar Kaggle.
- `competition_sources` di `kernel-metadata.json` menautkan dataset kompetisi otomatis ke `/kaggle/input/...` saat kernel jalan — notebook mendeteksi path ini secara otomatis (auto-discover), dengan fallback ke folder `data/` lokal bila dijalankan di luar Kaggle.
- Notebook memakai **sampling** untuk bagian yang menyentuh file DICOM (bukan iterasi ke semua ribuan file) supaya eksekusi tetap cepat.
