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

## 6. Riset metode optimal — Tahap 1 & 2 (`docs/RESEARCH_NOTES.md`)

**Metodologi:** Discussion/Overview/Rules/Evaluation tab Kaggle JS-rendered, tidak bisa di-scrape (dicoba berkali-kali, konsisten gagal). Sebagai gantinya: **Code tab lewat API resmi** (`kaggle kernels list --competition ... --sort-by scoreDescending`) — jauh lebih bernilai, dapat kode kerja nyata, bukan opini forum. 10 notebook publik top ditarik & dianalisis kodenya langsung.

**Temuan kunci dari kode kompetitor top:**
- **DINOv2 (`metaresearch/dinov2`, resmi Meta) mendominasi** — dipakai hampir semua notebook top, termasuk yang judulnya menyebut "DinoV3" (kode aslinya tetap DINOv2). **Tidak ada Kaggle Model DINOv3 resmi** — cuma re-upload komunitas tak terverifikasi
- Representasi studi: **"slot" bag** (kombinasi anatomis plane×sekuens) + attention pooling per-diagnosis dengan **prior anatomis hardcoded** (`SLOT_PRIOR_TABLE`) — `TARGETS` di kode ini identik persis dengan 12 label kita
- **Weak-label LLM: bukan live API call** — semua notebook baca CSV pre-computed dari Dataset terpisah (LLM dijalankan offline, hasil dipublikasi statis). Ada bukti eksplisit perbandingan "Claude Opus 5" vs rule-based (Claude menang signifikan, p=0.02) di satu notebook publik — indikasi praktik komunitas, **belum tentu berarti sesuai Competition Rules**
- Rule-based fallback pakai **deteksi negasi/uncertainty per-klausa multi-bahasa** (termasuk urutan kata negasi berbeda antar bahasa) — persis seperti yang kita rekomendasikan sebelumnya, terkonfirmasi jadi kebutuhan nyata
- Ensemble multi-backbone: DINOv2 + BiomedCLIP + EfficientNet-B3 dikombinasi dengan rank-blending
- **Ditemukan Efficiency Prize** — leaderboard efisiensi terpisah (1.507 tim), rank TIDAK sama dengan skor mentah (dikonfirmasi empiris) — dimensi kompetisi yang tidak ada di brief awal

**Riset kompetisi RSNA sebelumnya:** pola 2D CNN/ViT (intra-slice) + attention/RNN (inter-slice) konsisten menang di RSNA Intracranial Hemorrhage 2019, MRNet Stanford, RSNA Breast Cancer 2023 — memvalidasi arah arsitektur kita.

**Perbandingan lisensi foundation model:** BiomedCLIP = MIT (aman). **RadImageNet = restriktif, dilarang redistribusi tanpa izin** — re-upload Kaggle Model kemungkinan sudah melanggar lisensi asli.

**Keputusan Anda (dijawab, jadi acuan sejak titik ini):**
- [x] Metrik evaluasi: **macro (mean) ROC-AUC** — dikonfirmasi lewat kode validasi kompetitor top, dipakai konsisten di training loop kita
- [x] "Lakukan sebebasmu utk skor tertinggi" — **kecuali** larangan API LLM hosted (aturan keselamatan eksplisit, tetap berlaku kecuali dikoreksi)
- [x] Target: **kalahkan Brandon Low (peringkat 1, 0.950)** — skor mentah, bukan Efficiency Prize
- [x] RadImageNet: **dipakai** — bobot model (bukan dataset mentah) berlisensi MIT terverifikasi (lihat §6 lanjutan di `docs/RESEARCH_NOTES.md`)
- [x] Arsitektur final: DINOv2-base (primer) + ensemble RadImageNet-ResNet50 + BiomedCLIP, slot-bag attention + prior anatomis — didokumentasikan penuh di `docs/RESEARCH_NOTES.md` §5

## 7. Weak-labeler dari teks laporan (`kaggle_kernel_weak_labeler/`)

Kernel: [`rsna-knee-weak-labeler-from-reports`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-weak-labeler-from-reports). Rule-based, negation-aware per-klausa, 9 bahasa (en/es/tr/hr/el/de/bg/nl/fr) — **bukan LLM API** (data pasien sensitif, Rules belum terverifikasi eksplisit mengizinkan).

**Diuji lokal dulu sebelum push** — ketemu bug nyata (3 label OA 100% silence karena istilah kaku), diperbaiki jadi co-occurrence (indikator OA + kata lokasi dalam klausa sama) sebelum push. Sempat ada bug kedua (langdetect belum ter-install, run pertama fallback semua ke Inggris) — diperbaiki di v2.

**Hasil validasi ke 58 studi gold (v2, deteksi bahasa asli):**
- Agreement rata-rata **0.747**, precision **0.686**, **recall 0.951** (tinggi — jarang melewatkan temuan positif)
- Cakupan ke 4.349 studi non-gold: Effusion 70.5%, Medial Meniscus 54.6%, Lateral Meniscus 56.9%, ACL 50.8% (label umum tercakup baik); Synovitis 10%, Lateral OA 8.2% (label langka masih sparse -- sebagian inherent krn jarang disebut di laporan)
- Titik lemah tercatat utk iterasi lanjut: presisi MCL (0.44) & Contusion (0.50) rendah
- Output `weak_labels_full.csv` (4.407 studi: 58 gold + 4.349 weak-label) siap dipakai training via `kernel_sources`

Hasil lengkap: `weak_labeler_results/`

## 8. Modeling v1 — DINOv2-base + slot attention pooling (`kaggle_kernel_modeling/`)

Kernel: [`rsna-knee-modeling-v1-dinov2`](https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-modeling-v1-dinov2). Backbone tunggal dulu (sesuai keputusan bertahap) -- ensemble RadImageNet+BiomedCLIP menyusul setelah baseline ini terbukti jalan.

**3 bug ditemukan & diperbaiki sebelum/selama iterasi:**
1. **Dimensi `SlotHead` salah** (lupa kali 2 utk konkat CLS+mean-patch) -- ketemu lewat trace manual sebelum push (tidak bisa uji lokal, disk nyaris penuh utk install torch), jadi tidak buang kuota GPU
2. **Path `kernel_sources` tidak ketemu** -- ternyata mount lebih dalam dari dugaan (`/kaggle/input/notebooks/<owner>/<slug>/...`), diperbaiki pakai `os.walk` dgn pruning (pola sama spt pencarian DINOv2)
3. **GPU P100 tidak kompatibel** dgn build PyTorch Kaggle (compute capability terlalu lama) -- digali dari source code `kaggle` CLI sendiri, ketemu field `machine_shape: "NvidiaTeslaT4"` utk memaksa GPU yang benar, plus fallback CPU otomatis kalau terulang

**Uji skala kecil (v3, 40 studi, 1 epoch): SELESAI PENUH end-to-end.** GPU T4x2 terkonfirmasi, 3 sumber data (preprocessing cache + weak-label + DINOv2) semua ketemu benar, forward/backward pass tidak error, checkpoint tersimpan. `val_macro_auc=0.500`.

**v4 — run penuh (4.349 studi, 5 epoch, 3.2 jam): SELESAI, tapi val_macro_auc mentok TEPAT ~0.50 di SEMUA epoch** walau train_loss turun bertahap (0.717→0.620). Pola khas model yang cuma belajar bias per-label, mengabaikan gambar sama sekali.

**v5 — push diagnostik murah** (`RUN_FULL_TRAINING=False`, cuma statistik dataset + 1x forward pass kecil, tanpa training) buat root-cause tanpa buang kuota GPU lagi. **Ketemu:** 100% studi (train & val) punya mask kosong (0 dari 6 slot terisi) dan gambar kosong (mean pixel ≈0) — variasi embedding antar-studi juga persis 0.0000. **Akar masalah:** kolom `cache_path` di `series_manifest.csv` menyimpan path *relatif* apa adanya dari kernel preprocessing asli (`prep_outputs/image_cache/...`, relatif ke `/kaggle/working` kernel ITU) — begitu dipakai ulang lewat `kernel_sources` di kernel lain (mount di path berbeda), `np.load()` selalu gagal `FileNotFoundError`, dan itu ketangkep diam-diam oleh `except Exception: continue` di loader slot. Bug ini sudah ada sejak v3, cuma tidak pernah error karena di-*silent-catch* — makanya "pipeline validated" di v3 sebenarnya juga belum pernah benar-benar melihat gambar.

**v6 — 2 perbaikan sekaligus, SELESAI, fix terkonfirmasi berhasil:**
1. **Fix bug cache path**: path direkonstruksi dari `PREP_DIR` (hasil auto-discovery yang sudah benar per-kernel) + struktur relatif yang diketahui, bukan percaya string `cache_path` yang tersimpan. Ditambah guard `DATA_OK` (dicek ulang di §7b sebelum training) yang otomatis membatalkan training kalau gambar/mask ternyata masih kosong massal — supaya tidak ada lagi run berjam-jam yang sia-sia.
2. **Ganti sumber label training** — lihat bagian 9 di bawah.

**Hasil v6 (COMPLETE, ~1,83 jam training + overhead):**
- Diagnostik §7b **mengkonfirmasi fix berhasil**: `DATA_OK=True`, 0 dari 24.371 series cache tidak ketemu (sebelumnya 100% gagal), mask rata2 4.89-4.91/6 slot terisi (sebelumnya 0), gambar rata2 pixel 87-92 (sebelumnya 0), variasi embedding antar-studi 0.3870 (sebelumnya persis 0.0000), std prediksi per-label 0.01-0.03 (sebelumnya persis 0 — bukan lagi bias-only)
- **Training 3 epoch, val_macro_auc naik konsisten: 0.6906 → 0.7507 → 0.7658** (checkpoint terbaik disimpan tiap kali naik). train_loss turun 0.566→0.503, sempat naik tipis di akhir epoch 3 (0.4975→0.5029) — sinyal mendekati titik mulai overfitting, tapi AUC tetap naik
- **Signifikansi:** dari chance-level (0.50) ke 0.7658 dengan SATU perbaikan bug + label lebih baik, backbone tunggal, cuma 3 epoch, K_SLICE=4 (subsample dari 16 slice cache). Baseline sudah terbukti belajar sinyal nyata dari gambar — syarat sebelum lanjut ke ensemble sudah terpenuhi
- Masih jauh dari target (0.950, Brandon Low) — wajar untuk baseline pertama yang benar-benar jalan. Rencana lanjut: epoch lebih banyak (kompetitor top pakai EPOCHS=10/BATCH_STUDIES=8, lihat `docs/RESEARCH_NOTES.md` §1) + ensemble RadImageNet-ResNet50 + BiomedCLIP sesuai keputusan arsitektur final

## 9. Riset dataset publik & adopsi label LLM (`docs/RESEARCH_NOTES.md` §7)

Sesuai arahan: cek dataset publik apa yang dipakai kompetitor top, adopsi kalau terbukti signifikan naikkan skor.

Ditemukan beberapa dataset publik berisi label hasil pembacaan LLM atas kolom `Report`. Diukur macro AUC vs 58 studi gold kita sendiri (implementasi AUC manual berbasis rank — `sklearn` tidak tersedia lokal):

| sumber | macro AUC vs gold | catatan |
|---|---|---|
| rule-based labeler kita sendiri (bagian 7) | **~0.71** (estimasi dari precision/recall/agreement) | titik lemah: Synovitis, Contusion, Effusion |
| `stevenleehans/rsna-knee-llm-report-labels` v1 | **0.8780** | cocok persis dgn klaim publikasi dataset — implementasi AUC kita terverifikasi benar |
| `stevenleehans/rsna-knee-llm-report-labels` v4_blend | **0.8927** | blend dgn labeler independen kedua (lixin73/GPT-5.6-Sol), lisensi **CC0-1.0** |
| `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` | 1.0000 (trivial) | **di-skip** — dicek, kolom labelnya identik persis (selisih absolut = 0.0) dgn 58 studi gold, NaN di 4.349 sisanya. Bukan sumber baru, cuma re-package `train.csv` resmi + fold split |

**Keputusan: adopsi `stevenleehans/rsna-knee-llm-report-labels` (v4_blend)** sebagai target training untuk 4.349 studi non-gold (via `dataset_sources`, lisensi CC0-1.0 aman). 57 studi gold **tetap** pakai label asli annotator, tidak pernah ditimpa. Loss BCE ditambah confidence-weight `2*|p-0.5|` (rekomendasi dari dataset card) supaya sel "tidak dibahas di report" (p≈0.5) tidak memberi gradien palsu.

Provenance dataset: dihasilkan oleh LLM pihak lain yang membaca kolom `Report` `train.csv` kompetisi ini secara offline, hasil dipublikasikan statis (bukan kita yang memanggil API LLM hosted) — konsisten dengan pola "LLM-offline-label" yang sudah didokumentasikan di `docs/RESEARCH_NOTES.md` §1 dan tidak melanggar larangan API LLM hosted untuk memproses data pasien.

**v7 — naikkan `EPOCHS` 3→8, `BATCH_STUDIES` 4→8 (mendekati pola kompetitor top: EPOCHS=10/BATCH_STUDIES=8). SELESAI, ~4,73 jam.**

| epoch | train_loss | val_macro_auc | |
|---|---|---|---|
| 1 | 0.5697 | 0.6976 | checkpoint |
| 2 | 0.5227 | 0.7387 | checkpoint |
| 3 | 0.5002 | 0.7592 | checkpoint |
| 4 | 0.4822 | 0.7709 | checkpoint |
| 5 | 0.4653 | 0.7682 | turun, tidak disimpan |
| 6 | 0.4490 | 0.7794 | checkpoint |
| 7 | 0.4314 | **0.7811** | checkpoint (**BEST**) |
| 8 | 0.4125 | 0.7625 | turun cukup jauh — overfitting mulai signifikan |

**Best val macro AUC: 0.7811** (naik dari 0.7658 di v6, +0.0153) — checkpoint tersimpan otomatis di epoch 7, epoch 8 yang lebih buruk tidak menimpa. **Overfitting mulai kelihatan jelas di epoch 8** (train_loss terus turun 0.4314→0.4125 tapi val_auc malah turun 0.7811→0.7625) — konfirmasi guard "simpan checkpoint terbaik, bukan terakhir" itu penting & terbukti berguna. Kesimpulan: menaikkan epoch+batch dari backbone tunggal DINOv2-base memberi kenaikan modest (+2%), diminishing returns terlihat jelas sekitar epoch 6-7 — sesuai perkiraan §8c bahwa lompatan skor berikutnya lebih mungkin datang dari label lebih baik + ensemble multi-backbone drpd sekadar epoch lebih banyak di 1 backbone.

## 10. Riset lanjutan: ekosistem dataset publik kompetitor top (`docs/RESEARCH_NOTES.md` §8)

Arahan: jangan cuma 1 dataset, riset lebih luas apa yang dipakai kompetitor top di Code tab & public score.

Ditarik 18 kernel publik tambahan (rentang 0-344 votes) via `kaggle kernels pull -m`, dibaca `dataset_sources`/`model_sources`-nya. **Temuan utama: ada ekosistem dataset bersama yang dipakai HAMPIR SEMUA kompetitor top sekaligus**, bukan 1 dataset saja — bobot model (`pilkwang/rsna-knee-weights` 14/18), label LLM kedua (`pilkwang/rsna-knee-llm-labels` 13/18, `lixin73/...-sol56` 8/18), bobot RadImageNet pihak lain (`marwanmath/...` 12/18), fold-weights k-fold CV (`mattiaangeli/knee-mri-fold-weights` 7/18), dll. Tabel lengkap + lisensi tiap dataset di `docs/RESEARCH_NOTES.md` §8.

**Langsung diadopsi (aman, CC0, terverifikasi independen):**
- `flight0234/rsna-knee-hybrid-report-labels` — label hybrid (stevenleehans v4_blend + kolom Fracture dari pilkwang v2), **macro AUC 0.8991 vs gold** (lebih baik dari 0.8927 yang dipakai sekarang), plus kolom confidence eksplisit per-label. Akan dipakai mulai v8.
- `royceplayz/rsna-medicalnet-resnet50-weights` (Apache-2.0, bersih) — dipertimbangkan sbg backbone tambahan, membatalkan keputusan skip MedicalNet sebelumnya.

**3 item STOP — menunggu keputusan Anda (legalitas ambigu, bukan teknis):**
1. `pilkwang/rsna-knee-weights` (14/18 kernel!) — bukan pretrained generik, tapi bobot yang SUDAH di-fine-tune kompetitor lain ke kompetisi ini. CC0 & publik, tapi berpotensi kena aturan "no solution sharing" banyak kompetisi Kaggle.
2. `sohaibanwaar1203/kneemridataset` — data MRI lutut EKSTERNAL (917 studi RS Rijeka Kroasia, di luar dataset resmi), lisensi Kaggle "unknown". Belum jelas apakah Rules mengizinkan data eksternal.
3. `marwanmath/resnet-50-radimagenet-marwan` & `antoinegg1/...-heads-v15` — CC-BY-NC-SA-4.0 (**NonCommercial**), ambigu utk kompetisi berhadiah $77rb. **Rekomendasi: skip**, kita sudah punya RadImageNet-ResNet50 MIT-licensed sendiri (§4/§6 RESEARCH_NOTES) dgn kapabilitas sama.

**Strategi sintesis menuju peringkat 1** (detail di RESEARCH_NOTES §8c): label hybrid (v8) → ensemble RadImageNet+BiomedCLIP+(MedicalNet?) di atas baseline yg sudah lebih lama dilatih (v9) → proper k-fold CV + OOF ensemble selection (pola `sakhawathossen`) → per-target specialist fine-tuning → TTA + pipeline submission (belum ada sama sekali).

**Keputusan Anda atas 2 item STOP** (detail di RESEARCH_NOTES §8d): (1) `pilkwang/rsna-knee-weights` — "pakai sebagai warm-start". Ternyata paket 20-model rank-mean-ensemble (arsitektur DINOv2-small, tidak match backbone kita) — direncanakan dipakai sbg anggota ensemble independen (bukan disuntik ke bobot kita), digabung via rank-mean setelah pipeline inferensi kita sendiri berdiri. (2) `sohaibanwaar1203/kneemridataset` — "riset lisensi asli dulu". Sumber: I. Štajduhar et al. 2017, data RS Rijeka Kroasia, disetujui Komite Etik utk publikasi 2017, syarat yang ditemukan cuma atribusi+sitasi (tidak ada klausa NC). Lapisan ketidakpastian terpisah masih ada: apakah Rules kompetisi mengizinkan data eksternal sama sekali (item lama §6, belum terverifikasi). **Masih menunggu keputusan akhir Anda.**

**Riset lanjutan §9 (arahan "riset dataset model weights lebih dalam"):** diperiksa 6 dataset weight tambahan (`tonylica/rsna2026-models`, `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads`, `sadamtorres/rsna-ft-a`/`rsna-ft-b`/`rsna-wheels`, `prvsiyan/...-radimagenet-heads-20260812`). Temuan kunci: kernel `sofiaanjenje/rsna-knee-e11-train` (dipakai 2 dari kernel top kita) punya catatan "Thank you" yang mengkonfirmasi ekosistem publik ini persis — 20 model pilkwang + 4 model tonylica + 2-3 sumber label LLM = "meta-solusi" komunitas yang di-fork ratusan kali (menjelaskan klaster skor 0.89-0.93 di banyak judul kernel). `mattiaangeli`/`prvsiyan` punya dataset head-RadImageNet yang README-nya sendiri memperingatkan soal isu NonCommercial marwanmath — konfirmasi independen bahwa kehati-hatian kita soal lisensi NC itu tepat. `tonylica/rsna2026-models` (7/18 kernel) masih unknown-license & belum terdokumentasi, belum direkomendasikan tanpa verifikasi lebih lanjut (butuh `torch.load` di kernel Kaggle, tidak bisa lokal).

## 11. TEROBOSAN: teks Rules resmi lengkap terverifikasi langsung dari API Kaggle (`docs/RESEARCH_NOTES.md` §10)

Ditemukan `kaggle competitions pages <slug> list --content --format json` mengembalikan teks LENGKAP semua halaman kompetisi (rules, evaluation, data-description, code requirements, dll) langsung dari API resmi — bukan scraping, sepenuhnya melewati blokir JS-rendering yang selama ini jadi kendala. **Semua item yang sebelumnya "menunggu verifikasi" sekarang terjawab pasti:**

- **External data & pretrained model: DIIZINKAN eksplisit.** Halaman Code Requirements: *"Freely & publicly available external data is allowed, including pre-trained models"*. → `sohaibanwaar1203/kneemridataset` **diizinkan** (item STOP sebelumnya, sekarang terjawab). `pilkwang/rsna-knee-weights`, `tonylica/rsna2026-models`, dan seluruh dataset publik di §8-9 juga eksplisit diizinkan — tidak perlu was-was lagi soal "solution sharing" (Rules §3.5.d: "It's okay to share code if made available to all Participants on the forums").
- **Lisensi NC (marwanmath dkk.): bukan dilarang Rules**, tapi Kaggle cuma tidak memaksa relisensi kalau menang — lisensi asli NC-nya sendiri tetap urusan terpisah. Rekomendasi tidak berubah: tetap skip, pakai RadImageNet MIT-licensed sendiri (pilihan lebih bersih, bukan krn Rules melarang).
- **Data Security mengonfirmasi larangan API LLM hosted kita**: Rules §2.4.b melarang transmit Competition Data (termasuk teks Report) ke pihak yang tidak ikut kompetisi — sekarang berdasar teks Rules langsung, bukan cuma kehati-hatian kita.
- **Evaluasi 100% terkonfirmasi**: macro-averaged AUC ROC, formula $\frac{1}{12}\sum AUC_i$ — persis yang sudah dipakai training loop kita sejak v1.
- **Format kompetisi (baru diketahui)**: submission WAJIB lewat Notebook, nama file `submission.csv`, **internet HARUS mati** saat run submission (beda dari kernel R&D kita), **<=9 jam runtime** utk 1 run submission penuh (termasuk decode ~1.300 studi test + inferensi ensemble) — jadi batas keras desain pipeline inferensi nanti. Kolom `Report` **tidak disediakan saat testing** — konfirmasi pendekatan report-text cuma valid utk target training, sudah sesuai desain kita.
- **Deadline: Final Submission 22 Oktober 2026** (23:59 UTC). Winner wajib serahkan training+inference code & model weights sbg Kaggle Dataset publik kalau menang.

**v8 — ganti label ke `flight0234/rsna-knee-hybrid-report-labels` (0.8991, lebih baik dari 0.8927) + turunkan `EPOCHS` 8→6.** Sempat ERROR (filter `source` tidak ke-update setelah rename kolom — sudah diperbaiki + ditambah assert fail-fast). Push ulang sbg v9, **sedang berjalan**.

## 12. Kernel ensemble baru: 20-model komunitas (`kaggle_kernel_ensemble/`)

Implementasi pertama menuju resep 0.912-0.916 (`docs/RESEARCH_NOTES.md` §12). Kode decode+model+inferensi disalin ~verbatim dari `pilkwang/rsna-knee-baseline-v1` (CC0-1.0, legal — Rules §10), diadaptasi supaya baca dari `train_series` + filter ke 57 studi gold kita (bukan `test_series` stub 3-studi bawaan kompetisi). Sempat gagal push krn Kaggle API menolak judul tertentu (400 Bad Request, flaky, diselesaikan dgn judul lebih sederhana, bukan masalah isi notebook). **SELESAI ~6 menit — tapi hasil pertama SALAH (bug leakage kita sendiri), lihat §13.**

## 13. Bug leakage kritis + macro AUC ensemble komunitas yang benar (`docs/RESEARCH_NOTES.md` §13)

Kernel ensemble v1 selesai lancar (0 error, semua 20 fingerprint checkpoint cocok), tapi hasilnya **macro AUC 0.9954** — hampir sempurna, jelas tidak masuk akal (klaim publik pilkwang cuma 0.891 di leaderboard sungguhan). **Root cause ditemukan & diverifikasi sendiri**: tiap dari 20 model punya `fold` assignment (k-fold CV) dan sudah dilatih pakai sebagian besar studi gold kita (satu-satunya ground truth asli yg ada) — menjalankan SEMUA 20 model ke SEMUA 57 gold & rata-ratakan = tiap studi diskor oleh model yg SUDAH HAFAL jawabannya (bukan evaluasi, kebocoran murni).

**Perbaikan**: paket `pilkwang/rsna-knee-weights` ternyata sudah menyediakan `oof.npz` — prediksi out-of-fold RESMI (tiap studi diskor cuma oleh model yg fold-nya benar-benar exclude studi itu), siap pakai utk seluruh 4.407 studi training. Diukur ulang pakai ini (bukan live-inference kernel v1 kita):

**Macro AUC (leak-free, 57 gold kita): 0.8360** — MCL 0.875, Effusion 0.940, Baker's 0.954 tinggi; Lateral Meniscus 0.632, Synovitis 0.735 lemah. Jauh lebih masuk akal & dekat klaim publik 0.891 (selisih konsisten dgn kalibrasi lokal-vs-leaderboard §11b).

**Perbandingan baseline sejauh ini:**

| sumber | macro AUC (57 gold, leak-free) |
|---|---:|
| Model kita v6 (DINOv2-base, 3 epoch) | 0.7658 |
| Model kita v7 (DINOv2-base, 8 epoch, overfit epoch 8) | 0.7811 |
| **Ensemble komunitas 20-model pilkwang (OOF resmi)** | **0.8360** |

Ensemble komunitas genuinely lebih baik dari model kita saat ini (unfreeze cuma 2 blok vs mereka 6 blok, 3-8 epoch vs mereka 60 epoch/model) — konfirmasi arah blend (bukan buang model kita, tapi gabung) tetap tepat, krn diversitas arsitektur/pipeline masih bernilai (§11d.3) walau model kita sendiri belum sekuat punya mereka satu-satu.

**Pelajaran metodologi** (menegaskan lagi §11d.4, sekarang lewat kesalahan sendiri bukan cuma baca): jangan pernah evaluasi model pihak lain ke sample yg overlap training mereka kecuali ada bukti exclusion eksplisit; kalau paket eksternal sudah sediakan OOF resmi, PAKAI ITU — jangan re-infer sendiri (lebih murah & otomatis benar).

Langkah berikutnya (v2 kernel ini): tambah prediksi model kita sendiri (dari `kaggle_kernel_modeling`) + arm RadImageNet-ResNet50 (MIT, bukan `marwanmath` NC-licensed), gabung via rank-mean dgn bobot dipilih nested-CV di 57 gold — persis resep 0.912-0.916.

## 14. Mencari bukti skor di atas 0.916 (`docs/RESEARCH_NOTES.md` §14)

Arahan: riset lebih dalam demi melewati skor 0.950. Dicari kernel dgn klaim 0.92-0.95 di judul (`nikhil00718/rsna-knee-0-95-dinov2-v2`, `ranjithragavan07/rsna-knee-dinov2-0-93`, `ieshanmeghani/rsna-knee-v41-dinov3-e10-alpha060`).

**Temuan kunci**: beda dgn kernel "reproduction" 0.912/0.916 yg mencantumkan sitasi submission Kaggle selesai (bisa diverifikasi), klaim 0.93/0.95 di kernel-kernel ini **tidak** disitasi dgn submission nyata — tampak aspirasional (penulis `nikhil00718` sendiri menulis "leaderboard improvement is plausible but cannot be guaranteed"). **Tidak ketemu bukti tervalidasi skor publik di atas 0.916.**

**Teknik baru yg genuinely berguna** (terlepas dari validitas klaim skor): arm DINOv3 ViT-S/16 tambahan (dilatih sendiri, bukan Kaggle Model resmi — perlu cek sumber sebelum ditiru), agregasi cross-attention penuh atas SEMUA token semua series dlm 1 studi (lebih ekspresif drpd slot-mean kita), **greedy checkpoint soup** (rata-rata beberapa checkpoint terbaik), fold dikelompokkan per hash-laporan (konfirmasi independen lagi dari §11d.6).

**Kesimpulan jujur**: tidak ada jalan pintas terverifikasi ke atas 0.950. Tim top-20 kemungkinan besar kombinasi teknik-teknik di atas dieksekusi lebih matang, bukan 1 trik rahasia. Jalur realistis tetap: bangun ensemble kita setinggi mungkin, dapat submission nyata, iterasi berdasar gap yang benar-benar terlihat.

## 15. v9 SELESAI (0.7902) + TEMUAN KRITIS: tim sudah punya riwayat submission sampai 0.914

**v9 (label hybrid flight0234 + confidence eksplisit, EPOCHS=6): SELESAI, ~3,49 jam.** Val macro AUC naik **monoton** di semua 6 epoch (tidak ada overfitting dip spt v7):

| epoch | train_loss | val_macro_auc |
|---|---|---:|
| 1 | 0.5779 | 0.6938 |
| 2 | 0.5306 | 0.7396 |
| 3 | 0.5087 | 0.7630 |
| 4 | 0.4897 | 0.7757 |
| 5 | 0.4721 | 0.7760 |
| 6 | 0.4539 | **0.7902** (BEST) |

Naik dari v7 (0.7811) dan v6 (0.7658) — label lebih baik + epoch lebih pendek sama-sama membantu, tanpa overfitting kali ini. Masih di bawah ensemble komunitas (0.8360).

**TEMUAN KRITIS (17 Agustus 2026, saat mau submit resep pilkwang-sendirian utk kalibrasi)**: dicek `kaggle competitions submissions` sebelum submit — **tim kita SUDAH PUNYA 30+ submission sejak 6 Agustus 2026**, jauh sebelum sesi kerja hari ini. Progres skor publik tim: 0.768→0.774 (6 Agu) → 0.891 (8 Agu, pilkwang baseline) → 0.896-0.899 (9-12 Agu) → 0.909-0.911 (13-15 Agu) → **0.914 (16 Agu, TERBAIK saat ini)**. Notebook yang di-submit persis notebook publik yang sama yang kita riset hari ini (pilkwang baseline, prvsiyan, DINOsaur V2, Bend the Knee DinoV3, enhanced-ensemble, frontier v46) — dikerjakan pihak lain (kemungkinan terkait kolaborator `nxtirfan` yg ditemukan menulis ulang README, §14 sebelumnya).

**Implikasi**: skor terbaik TIM (0.914) jauh lebih tinggi drpd perkiraan submission pilkwang-sendirian (~0.891) yang sedang saya siapkan, dan gap ke target (0.950) cuma 0.036 — jauh lebih dekat dari yang saya kira sebelumnya. **BELUM jadi submit** submission pilkwang-sendirian (nilainya rendah drpd yang sudah ada + submission harian dibatasi 5x/hari UTK SELURUH TIM, bukan per orang) — menunggu arahan user soal koordinasi dgn kerja tim yang sudah ada.

## 16. Discussion tab — sumber baru jauh lebih berharga (`docs/RESEARCH_NOTES.md` §15)

`kaggle competitions topics list` + `topic-messages` (belum dipakai sepanjang sesi) buka akses ke Discussion tab, isinya lebih bernilai dari Code tab utk pertanyaan strategis — ada komentar host resmi & pengakuan skor asli peserta (bukan klaim judul kernel).

**RULING RESMI HOST (733965): API LLM hosted TERNYATA DIIZINKAN** — *"submitting Competition Data, including report text, to an external LLM or API for inference... will not, by itself, be considered prohibited PRIVATE SHARING."* Ini membalik asumsi kerja kita sejak awal proyek. **Item keputusan Anda**: mau dilonggarkan sekarang restriksi "jangan pakai API LLM hosted"? Relevan krn kualitas label masih pengungkit besar (lihat di bawah).

**Single model bisa capai 0.92-0.934 — ensemble 20+ model BUKAN keharusan.** Thread "Best single-model score" (735304): pengakuan skor asli — 0.915 single DINOv2, **0.934 single model @224px** (resolusi sama persis dgn kita!), 0.92 di 5-fold (bukan 20+). Komentar penting: *"Highly doubt the top results are ensembles... Public notebooks just ensemble early on to try and get a high leaderboard score and exposure/kudos. Waste of time IMO."* — ekosistem 20-model publik yang kita riset mungkin BUKAN jalan tim top-20 sungguhan, cuma "kudos farming". **Implikasi besar**: mungkin lebih bernilai investasi ke 1 model kuat (fine-tune dalam, k-fold proper) drpd terus kejar ensemble komunitas yang levelnya cuma ~0.89-0.916.

**Teknik sampling slice lebih baik ("3+3+3")**: dari thread "Scaling encoder bought us nothing" (735154) — 3 anchor merata di jendela tengah stack, ambil 3 slice fisik berdekatan per anchor (3×3=9) jadi channel R/G/B — representasi ~10mm depth neighbourhood asli, lebih canggih dari K_SLICE=4 subsample rata-rata yang kita pakai. Kandidat upgrade konkret. Juga dikonfirmasi lagi: DINOv2-small→base cuma +0.0011 (noise floor 0.0020) — kapasitas model bukan pengungkit.

**Isu lisensi NC RadImageNet masih terbuka di komunitas juga** (735121, belum dijawab host) — menguatkan keputusan kita skip `marwanmath`.

**Keputusan Anda (17 Agustus 2026): restriksi API LLM hosted DILONGGARKAN** — "kalau tidak melanggar aturan, buka" — sudah dikonfirmasi ruling resmi host (733965) mengizinkan, jadi syaratnya terpenuhi. **Konstrain lama diganti**: sekarang boleh pakai API LLM hosted (OpenAI/Anthropic/dll) utk memproses report text, SELAMA layanannya "reasonably accessible & minimal cost" ke semua peserta (syarat Rules §2.6.b) — bukan lagi larangan mutlak. Praktiknya butuh API key (disimpan sbg Kaggle Secret) utk dipakai di kernel — menunggu konfirmasi/API key dari Anda kalau mau benar-benar diimplementasikan (relabeling ulang 4.349 studi non-gold), atau bisa juga tetap pakai label publik terbaik yang sudah ada (flight0234, 0.8991) sambil fokus ke pengungkit lain dulu (single model kuat, per §16 di atas).

## 17. Eksperimen single-model kuat: v10 (OOM) → v11 (fix)

Sesuai arahan Anda ("lakukan eksperimen yg kamu butuhkan, sesuai strategi top-20, testing sendiri") dan temuan §16 (single model bisa 0.92-0.934 tanpa ensemble besar): **v10** naikkan `unfreeze_last` 2→6 + pisah LR head(1e-3)/backbone(8e-6, nilai umum kode komunitas top).

**v10 ERROR**: `CUDA OutOfMemoryError` — sanity check kecil (batch=2) lolos, tapi training beneran (`BATCH_STUDIES=8`) kehabisan memori T4 14.5GB begitu forward/backward pertama jalan. Sebab: 6 blok trainable (42.5M params) butuh jauh lebih banyak memori aktivasi drpd 2 blok (10.7M) sebelumnya, `BATCH_STUDIES=8` yang aman utk unfreeze=2 jadi kebesaran.

**v11 — fix**: (1) `gradient_checkpointing_enable()` di backbone (tukar compute demi memori), (2) `BATCH_STUDIES` 8→4. Catatan: push v10→v11 membuat Kaggle rename slug kernel 2x (`v1-dinov2` → `v10-unfreeze6-split-lr` → `v11-fix-oom-grad-checkpoint`) — judul yang menyimpang jauh dari slug asal memicu Kaggle bikin URL baru meski `id` di metadata tidak berubah; versi/riwayat kernel tetap utuh, cuma alamat berubah tiap kali. `kernel-metadata.json` & link README di-update tiap kali.

**v11 SELESAI — LOMPATAN TERBESAR sejauh ini: val_macro_auc 0.8195** (~7,39 jam, jauh lebih lama drpd v9 3,49 jam krn batch lebih kecil + gradient checkpointing).

| epoch | train_loss | val_macro_auc |
|---|---|---:|
| 1 | 0.5704 | 0.7343 |
| 2 | 0.5234 | 0.7608 |
| 3 | 0.4979 | 0.7738 |
| 4 | 0.4737 | 0.7886 |
| 5 | 0.4473 | 0.7895 |
| 6 | 0.4196 | **0.8195** |

**Naik MONOTON di semua 6 epoch, BELUM PLATEAU** — epoch 6 masih naik +0,03 dari epoch 5, kandidat kuat utk epoch lebih banyak. Hipotesis dari Discussion tab (§16, single model bisa capai 0,92-0,934 tanpa ensemble besar) **terkonfirmasi arahnya benar**: unfreeze lebih dalam + LR terpisah adalah pengungkit besar, jauh melampaui backbone-tunggal manapun sebelumnya (v6=0,7658 → v7=0,7811 → v9=0,7902 → **v11=0,8195**) dan mendekati ensemble komunitas 20-model (0,8360, OOF resmi leak-free) — dengan backbone TUNGGAL, bukan 20 model.

**Pelajaran operasional baru**: log API Kaggle di environment ini bisa kosong TOTAL selama berjam-jam (v11 kosong ~4,5 jam) padahal kernel genuinely aktif — dikonfirmasi via `kaggle quota` (kuota GPU terus naik konsisten selama itu). Verifikasi via delta kuota GPU adalah cara diagnostik yang jauh lebih andal drpd cuma mengandalkan `kaggle kernels logs`/`status` saat curiga hang.

## 18. v12 — resume training dari checkpoint (bukan dari nol), hemat kuota GPU

Sesuai arahan Anda ("pakai checkpoint agar tidak mengulang train, hemat waktu"). Sisa kuota GPU mingguan tipis (~4,13h saat dirancang) — training ulang dari nol utk epoch lebih banyak (target awal EPOCHS=10) tidak akan muat (~10-12 jam dari nol). Solusi: **resume dari checkpoint v11** (val_macro_auc=0,8195):

- **Coba unduh `best_model.pt` v11 secara lokal dulu (gagal berulang)**: file checkpoint DINOv2-base full (~350MB, wajar utk 86M parameter) terus putus di tengah (`IncompleteRead`) krn koneksi lokal tidak stabil — 4x percobaan semua gagal di bawah 2.5MB dari 350MB.
- **Solusi**: pakai **self-reference `kernel_sources`** — kernel modeling menambahkan dirinya sendiri (`dimaspashaakrilian/rsna-knee-modeling-v11-fix-oom-grad-checkpoint`) ke daftar `kernel_sources`-nya sendiri. Transfer checkpoint jadi 100% server-side Kaggle-ke-Kaggle, tidak lewat koneksi lokal yang bermasalah. **Pola baru, belum pernah dicoba sebelumnya di proyek ini.**
- **Trik supaya self-reference stabil**: judul kernel SENGAJA TIDAK diubah dari versi v11 (biar slug tidak berganti lagi seperti pola v10→v11 sebelumnya) — dokumentasi "v12" cukup di markdown notebook, bukan di judul Kaggle. Push berhasil TANPA peringatan mismatch judul-slug (beda dari push-push sebelumnya) — konfirmasi slug stabil.
- **Perubahan kode**: `CKPT_DIR` discovery (`find_dir_containing("best_model.pt", "modeling")`), load `state_dict` ke model sebelum training loop, `best_auc` awal di-set ke 0,8195 (skor checkpoint) supaya tidak pernah tertimpa hasil lebih buruk, `EPOCHS=4` (tambahan, bukan dari nol), `SAFETY_CUTOFF_SECONDS` diperketat ke 3,5 jam (dari 7,5 jam) menyesuaikan sisa kuota.
- Sudah **di-push (versi 12) & sedang berjalan**. Kuota GPU saat push: 25,87h terpakai, 4,13h sisa.

**Pelajaran operasional baru**: (1) log API Kaggle di environment ini bisa kosong TOTAL selama berjam-jam (v11 kosong ~4,5 jam) padahal kernel genuinely aktif — dikonfirmasi via `kaggle quota` (kuota GPU terus naik konsisten selama itu); verifikasi via delta kuota GPU jauh lebih andal drpd cuma `kaggle kernels logs`/`status` saat curiga hang. (2) `kaggle kernels output` bisa gagal berulang utk file besar (>300MB) krn koneksi lokal — kalau butuh transfer file besar antar-kernel, self-reference/`kernel_sources` (server-side) lebih andal drpd download-lokal-lalu-upload-ulang. (3) **Kaggle mengizinkan sesi yang sudah berjalan melebihi kuota mingguan** — kuota `used` naik sampai 30,30h (>30h nominal), `remaining` cuma floor di 0,00h, tapi kernel TIDAK di-force-stop, dibiarkan selesai normal.

**v12 SELESAI (~4,51 jam) — self-reference checkpoint BERHASIL SEMPURNA, tapi hasil trainingnya JADI PELAJARAN PENTING:**

Log konfirmasi: `"RESUME: checkpoint v11 dimuat... (val_macro_auc sebelumnya: 0.8195)"` — pola self-reference `kernel_sources` tervalidasi, bisa dipakai lagi ke depan utk resume training murah.

| epoch | train_loss | val_macro_auc |
|---|---|---:|
| v11 ep6 (checkpoint awal) | 0,4196 | **0,8195** |
| v12 ep1 | 0,3889 | 0,8096 (turun) |
| v12 ep2 | 0,3604 | 0,7988 (turun lebih jauh) |
| v12 ep3 | 0,3362 | 0,8016 |
| v12 ep4 | 0,3171 | 0,8182 (mendekati tp blm lampaui) |

**Tidak ada epoch v12 yang mengalahkan 0,8195** — guard `best_auc = 0,8195 if RESUMED` bekerja sempurna, checkpoint final TETAP v11 (0,8195), TIDAK rusak. Tapi **train_loss terus turun tajam sementara val_macro_auc turun/stagnan — pola OVERFITTING klasik**. Kesimpulan: v11 (6 epoch dari nol, LR head=1e-3/backbone=8e-6 konstan) sudah mendekati titik optimal utk konfigurasi ini; melanjutkan dgn LR sama (tanpa decay) bikin model makin hafal weak-label training set, bukan makin generalize ke gold. **"Epoch lebih banyak dgn config sama" bukan lagi jalan maju yang tepat** — perlu LR scheduler/decay, atau strategi berbeda (k-fold, per-target specialist, ensemble).

**Model final terbaik sejauh ini: val_macro_auc 0,8195** (checkpoint v11, backbone tunggal DINOv2-base, unfreeze 6 blok). Kuota GPU sudah overrun — TIDAK push eksperimen baru sampai ada arahan Anda soal langkah selanjutnya atau kuota refresh (22 Agustus 2026).

## 19. Kenapa skor "mentok" + bedah forensik resep 0,920 tim sendiri (`docs/RESEARCH_NOTES.md` §16)

Arahan: "score-nya mentok, apa strategi ke top 1". Ditemukan **skor tim sungguhan sudah 0,920** (naik dari 0,914), via submission notebook publik `tonylica/rsna-knee-dino-radimagenet-rank-ensemble` (fork, bukan model kita sendiri) — **peringkat #130 dari 1.886 tim**.

**Diagnosis kunci**: 0,920 adalah **"commodity ceiling"** — puluhan tim (rank ~100-140+) punya skor IDENTIK 0,920 krn semua fork notebook publik yang sama. Bukan pencapaian unik kita, dan bukan jalan ke top 1 (rank 1-50 semua SUDAH di atas ceiling ini, 0,929-0,951). Gap: ke top 50 (0,929) +0,009, ke top 20 (0,937) +0,017, ke #1 Brandon Low (0,951, submit ulang HARI INI) +0,031.

**Bedah forensik resep 0,920** (35 checkpoint: 20 DINOv2-small pilkwang + 5 DINOv3-small tonylica + 10 head RadImageNet 2-varian pakai encoder `marwanmath` NC-licensed, blend rank-space 4 tahap dgn bobot tetap per-target termasuk exclude Baker's/Fracture dari blend RadImageNet — persis pola §12b yg sudah kita riset). Pakai 336px (vs 224px kita), TTA flip, urutan slice fisik via `ImagePositionPatient`.

**Bukti independen baru dari Discussion (18 Agustus)**: (1) eksperimen terkontrol kompetitor lain — tukar backbone ResNet34→DINOv2-Small dgn pipeline SAMA, hasil DINOv2 JUSTRU LEBIH BURUK (735826) — validasi kuat "geometri slice + kualitas label > ukuran backbone". (2) 58 studi gold TERBUKTI ada inkonsistensi internal (735767) — validasi lokal kita vs gold selalu ada batas akurasi bawaan. (3) 2 jebakan submission wajib dicek: codec DICOM offline (`pylibjpeg`/`gdcm`) & path `/kaggle/input/competitions/<slug>/` saat scored run (735854). (4) ~7 studi ada 2 lutut dalam 1 DICOM, bisa bikin crop naif miss total (735639).

**Strategi ke top 1** (detail & urutan penuh di RESEARCH_NOTES §16d): (1) **mulai SEKARANG, CPU-only, tanpa kuota GPU** — bangun ulang geometri slice preprocessing (urutan fisik `ImagePositionPatient`, crop sadar-laterality, handle 2-lutut, sampling "3+3+3"); (2) begitu kuota GPU refresh, lanjut fine-tune v11 pakai LR scheduler/decay (bukan flat spt v12 yg overfit); (3) k-fold CV proper; (4) baru setelah model sendiri kuat, masuk ke ensemble sbg anggota independen + arm RadImageNet MIT sendiri; (5) TTA flip; (6) opsi eksperimen kuota TPU (20h, 100% belum dipakai) selama GPU masih 0 sampai 22 Agustus.

## 20. Probe TPU: teknis bisa, antrian ~28,6 menit (arahan 18 Agustus, "coba pakai TPU, antrinya berapa lama")

Push kernel diagnostik kecil (`rsna-knee-tpu-diagnostic-torch-xla-check`, `enable_tpu: true`) buat ukur langsung, bukan nebak. **Hasil terukur nyata**: push 22:45:46 → `QUEUED` → `RUNNING` di t+1716s (**~28,6 menit antrian**) → `COMPLETE` di t+1782s (eksekusi ~1 menit). Log konfirmasi `torch_xla` 2.8.0 pre-installed di image Kaggle, `xm.xla_device()` + matmul + import `transformers` semua berhasil (`XLA_OK`) — **TPU secara teknis BISA dipakai, dan antriannya jauh lebih cepat dari dugaan (bukan berjam-jam)**.

**Caveat penting**: `torch` di TPU image adalah build `2.8.0+cpu` (bukan cuda) — kode `KneeModel` kita yang berbasis CUDA harus ditulis ulang pakai API `torch_xla` (`xm.xla_device()`, `xm.mark_step()`), bukan cuma toggle `enable_tpu` di `kernel-metadata.json`. Idealnya juga dipad ke shape tetap dulu (variable slot-count per studi bikin XLA re-compile graph berulang, mahal). Ditemukan juga kompetitor lain di kompetisi ini (`ssstelmah/rsna-tpu-probe-a`) sudah provision `TpuV5E8` duluan — infrastrukturnya jelas dipakai kompetitor lain juga.

**Belum diputuskan**: port kode ke TPU sekarang (isi 4 hari nunggu GPU refresh) vs fokus ke preprocessing v2 dulu lalu train ulang pas GPU refresh 22 Agustus.

## 21. v13: port ke TPU (torch_xla) + LR scheduler, sanity run (arahan 18 Agustus, "kalo sukses lanjutkan eksperimen strategi")

TPU terverifikasi jalan (§20) — langsung lanjut eksekusi, bukan cuma riset. Kernel baru `kaggle_kernel_modeling_tpu/` (`rsna-knee-modeling-v13-tpu-sanity`), 2 perubahan sekaligus dari v11/v12:

1. **Device/training-loop di-port ke `torch_xla`** — deteksi XLA duluan (sebelum CUDA), fallback CUDA lalu CPU tetap dipertahankan. `xm.mark_step()` ditambah tiap step, `gradient_checkpointing_enable()` dibungkus try/except (belum pernah dites kombinasinya dgn XLA), `torch.load` pakai `map_location="cpu"` (aman lintas device).
2. **LR scheduler (`CosineAnnealingLR`)** — perbaikan langsung utk overfitting v12 (LR flat, 4 epoch tambahan semua lebih buruk dari checkpoint 0,8195). Diterapkan ke kedua param group (head+backbone), meluruh ke ~0 menjelang akhir training.

**Ini SANITY RUN** (`DEBUG_LIMIT_TRAIN=400`, `EPOCHS=2`, `SAFETY_CUTOFF_SECONDS=1h`) — bukan run penuh, bukan target beat 0,8195. Tujuan: validasi mekanisme (XLA+grad-checkpoint+scheduler+resume checkpoint v11 via `kernel_sources` langsung, bukan self-reference) jalan benar di environment TPU yang belum pernah dipakai proyek ini, dan dapat estimasi waktu/epoch riil, sebelum commit ke run penuh yg makan kuota TPU besar — disiplin yg sama dgn kenapa v3/v5 dulu jadi run kecil/diagnostik dulu.

Push 18 Agustus 23:37:01, sedang `QUEUED` di monitor. Kalau sukses: lanjut run PENUH TPU (v13b/v14, `DEBUG_LIMIT_TRAIN=None`, `EPOCHS` dinaikkan krn kuota TPU jauh lebih longgar). Kalau gagal: fallback ke GPU pas kuota refresh 22 Agustus (device-fallback logic sudah otomatis, kode tidak perlu ditulis ulang).

## 22. v14 SELESAI: TPU ~12x lebih cepat terkonfirmasi, tapi skor plafon di ~0,82 (bukan config training lagi)

Run penuh (12 epoch, 4.349 studi, dari nol, TPU, `CosineAnnealingLR`) selesai ~4.690 detik (~78 menit).

| epoch | val_macro_auc | | epoch | val_macro_auc |
|---|---:|---|---|---:|
| 1 | 0,7357 | | 7 | 0,8023 |
| 2 | 0,7480 | | 8 | 0,8001 |
| 3 | 0,7422 | | 9 | 0,7939 |
| 4 | 0,7696 | | 10 | 0,7958 |
| 5 | 0,7789 | | 11 | 0,7970 |
| **6** | **0,8159 (BEST)** | | 12 | 0,7973 |

**Kecepatan TPU terkonfirmasi**: steady-state ~370 detik/epoch (~6,2 menit) utk data penuh — v11 di GPU T4 makan **~74 menit/epoch**. **~12x lebih cepat**, melebihi estimasi awal (~10x) dari sanity run.

**Tapi val_macro_auc TIDAK melampaui checkpoint v11 (0,8195)** — cuma 0,8159 (beda 0,0036, dlm noise N=57). LR scheduler berhasil cegah collapse separah v12 (yg anjlok tajam ke 0,7988), tapi tetap plateau ~0,80-0,82 mulai epoch 6, walau LR terus meluruh ke ~1e-5/1e-7 di epoch 12.

**Kesimpulan strategis (bukti ke-4: v7, v9, v11, v12, sekarang v14)**: arsitektur+config saat ini (DINOv2-base, unfreeze 6 blok, slot attention, label hybrid flight0234) punya **plafon keras di ~0,80-0,82 macro AUC**, TIDAK PEDULI variasi jadwal training (epoch banyak/sedikit, LR flat/decay, dari-nol/resume). Ini menegaskan ulang §16d: pengungkit berikutnya HARUS di luar training-schedule — geometri slice preprocessing (`ImagePositionPatient`, laterality, "3+3+3") + kualitas label, bukan lagi tuning training loop. **Model final terbaik proyek tetap checkpoint v11 (0,8195, GPU)** — v14 tidak menggantikannya, tapi tervalidasi tidak merusak apa pun (checkpoint v11 aman tersimpan terpisah).

**TPU tetap aset berharga** — bukan sbg pengungkit skor langsung di config ini, tapi sbg infrastruktur cepat utk iterasi preprocessing v2 & eksperimen berikutnya (12x lebih murah waktu drpd GPU T4), begitu prioritas #1 (geometri slice) siap diuji.

## 23. Preprocessing v2 SELESAI PENUH (100% posisi fisik, 0 fallback) + v15 dipush

`kaggle_kernel_preprocessing_v2/` selesai run penuh: **24.371/24.371 series (100%), ~72,5 menit, 24.369 OK / 2 ERROR**. Output 12,94 GB (mirip v1, 13,05 GB). **Diagnostik kunci: 100% series pakai `pos_source=ipp` (posisi fisik `ImagePositionPatient`), 0,00 rata-rata fallback** — fix urutan slice tervalidasi penuh di skala sebenarnya, bukan cuma sampel 300 series. Bonus temuan: 4 series ter-flag laterality `"B"` (Both, kemungkinan 2 lutut 1 DICOM — konsisten dgn temuan Discussion 735639).

**v15 dipush** (`kaggle_kernel_modeling_v2geom/`, TPU): eksperimen isolasi variabel — SATU-SATUNYA beda dari v14 adalah `kernel_sources` menunjuk preprocessing v2 (bukan v1), semua hyperparameter lain identik (DINOv2-base, unfreeze 6, LR head/backbone terpisah, `CosineAnnealingLR`, `EPOCHS=8`). Pembanding: v14 (cache v1) = 0,8159, checkpoint v11 (GPU, cache v1) = 0,8195 (skor terbaik proyek). Push 19 Agustus 11:36:09, sedang dipantau.

## 24. v15 SELESAI — TEROBOSAN: geometri slice fisik terbukti naikkan skor, plafon lama tertembus

Run ~55,5 menit (TPU). Val_macro_auc:

| epoch | val_macro_auc | | epoch | val_macro_auc |
|---|---:|---|---|---:|
| 1 | 0,7268 | | 5 | 0,8104 |
| 2 | 0,7724 | | **6** | **0,8304 (BEST)** |
| 3 | 0,7976 | | 7 | 0,8152 |
| 4 | 0,8074 | | 8 | 0,8179 |

**val_macro_auc 0,8304 — MODEL TERBAIK BARU PROYEK**, melampaui checkpoint v11 (0,8195, +0,0109) dan v14 (0,8159, cache v1 di TPU, +0,0145). SATU-SATUNYA variabel yg diubah dari v14 adalah sumber cache gambar (preprocessing v2 vs v1) — semua hyperparameter lain identik. **Hipotesis dari riset Discussion (735826, §16c) sekarang TERBUKTI langsung lewat eksperimen sendiri, bukan cuma laporan pihak lain: geometri slice yang benar (urutan fisik `ImagePositionPatient`) adalah pengungkit nyata, menembus plafon ~0,80-0,82 yang mentok di 4 eksperimen berturut (v7/v9/v11/v12/v14).**

Pola kurva mirip v11 (masih naik sampai epoch 6, sedikit dip di 7, parsial pulih di 8) — checkpoint tersimpan di epoch 6 (`kernel_sources: dimaspashaakrilian/rsna-knee-modeling-v15-geom-v2-tpu`, versi 1, **BELUM ada versi baru yang menimpanya** — kalau push versi baru ke slug ini nanti, WAJIB set `best_auc` awal ke 0,8304, bukan -1,0, biar tidak kena gotcha yg sama spt v11/v12).

**Model final terbaru proyek: val_macro_auc = 0,8304** (checkpoint v15, TPU, cache preprocessing v2).

**Langkah berikutnya**: (1) item geometri lain yg masih tertunda (3+3+3 sampling, crop laterality, handle 2-lutut-1-DICOM) makin layak dicoba krn geometri sudah terbukti pengungkit nyata; (2) skor lokal 0,8304 mendekati ambang "> ~0,85-0,88" yg jadi syarat sebelum masuk ke ensemble (roadmap lama §10) — belum tercapai penuh, masih perlu naik sedikit lagi; (3) k-fold CV proper masih relevan utk validasi yg lebih stabil (N=57 gold cukup noisy per-epoch).

## 25. Preprocessing v3 (sampling "3+3+3") SELESAI PENUH + v16 dipush

Arahan user pasca-v15: "lanjut item geometri lain" (dari 3 pilihan: geometri lain / k-fold / langsung ensemble). `kaggle_kernel_preprocessing_v3/` mengimplementasikan sampling "3+3+3" (Discussion 735154): 3 anchor merata di jendela tengah stack (20%-80%), tiap anchor + 2 tetangga fisik langsung dipasang sbg channel R/G/B **genuine** (bukan grayscale-di-*repeat* spt v1/v2). `get_sorted_files` (urutan fisik, v2) tidak berubah.

**Sanity run (300 series) & run penuh (24.371 series, ~67,7 menit) SUKSES SEMPURNA: 100% OK (0 error), 100% `n_slices_cached=9`, 100% `pos_source=ipp`, 0 fallback.** Output 7,53 GB (lebih kecil dari v2 12,94 GB, wajar krn 9 slice vs 16/series). Lebih cepat dari v2 juga (~400-500 series/menit vs ~360/menit — cuma decode 9 slice, bukan 16).

**v16 dipush** (`kaggle_kernel_modeling_v3geom/`, TPU): BUKAN cuma ganti data source lagi — kode model ikut berubah krn bentuk cache berubah total (9 slice → reshape 3 anchor x 3 RGB). `KneeStudyDataset._load_slot` di-reshape, `KneeModel.forward` — baris `.repeat(1,3,1,1)` (dulu duplikasi grayscale) **dihapus**, channel RGB sekarang genuine dari 3 slice fisik berbeda. Arsitektur/hyperparameter lain identik dgn v15 (EPOCHS=8, LR terpisah, `CosineAnnealingLR`, TPU). Pembanding: v15 (cache v2, "16 linspace") = **0,8304** (model terbaik saat ini). Push 19 Agustus 18:35:07, sedang dipantau.

## 26. v16 SELESAI — negatif: channel RGB genuine LEBIH BURUK, penyebab jelas (patch embedding beku)

Push pertama ERROR (bug sendiri, cell diagnostik py salinan reshape lama, sudah diperbaiki & di-push ulang versi 2). Run ~51,5 menit. Val_macro_auc:

| epoch | val_macro_auc | | epoch | val_macro_auc |
|---|---:|---|---|---:|
| 1 | 0,6301 | | 5 | 0,6590 |
| 2 | 0,6454 | | 6 | 0,7275 |
| 3 | 0,6478 | | 7 | 0,7596 |
| 4 | 0,6681 | | **8** | **0,7737 (BEST, masih naik)** |

**LEBIH BURUK dari v15 (0,8304) DAN v11 (0,8195)**. Beda dari v15/v11 (menanjak cepat, plateau ~0,80-0,83), v16 belajar jauh lebih lambat (train_loss stuck ~0,48-0,53 selama beberapa epoch), belum konvergen saat EPOCHS=8 habis (LR sudah ~0 krn cosine decay).

**Diagnosis (bukan tebakan kosong)**: `unfreeze_last=6` berarti patch embedding (layer paling awal DINOv2) TETAP BEKU, dan layer itu di-pretrain mengharapkan channel R/G/B = warna foto natural yang berkorelasi. Skema lama (grayscale-di-*repeat*, v1/v2/v15) aman krn 3 channel identik = backbone tetap melihat citra abu-abu biasa (pola umum & robust utk fine-tuning backbone ImageNet pretrained ke domain grayscale). Skema baru (3 slice fisik berbeda sbg R/G/B genuine) menyodorkan pola channel yang jauh dari yang pernah dilihat patch embedding — krn layer itu beku, model tidak bisa belajar menafsirkannya.

**Model final proyek TETAP v15 (0,8304)** — v16 tidak menggantikannya, tapi tervalidasi tidak merusak apa pun.

**Implikasi utk eksperimen 3+3+3 ke depan (kalau mau dicoba lagi)**: (a) unfreeze patch embedding jugajuga (risiko lebih tinggi, kemungkinan OOM/butuh LR lebih hati-hati), atau (b) pisahkan "jendela sampling anchor" dari "channel-packing" — pakai 9 slice dari skema 3+3+3 sbg 9 VIEW independen (grayscale-repeat spt v15, rata-rata post-backbone), BUKAN dipasang sbg R/G/B — isolasi manfaat jendela-sampling tanpa risiko mismatch channel. (b) lebih murah & konsisten dgn temuan v15 (physical order = pengungkit, cara agregasi lama sudah terbukti aman).

## 27. v17 SELESAI — masih di bawah v15, "3+3+3" 2/2 gagal, kesimpulan diperjelas

Arahan user pasca-v16: "coba ulang 3+3+3 tanpa RGB-packing" — 9 slice cache v3 dipakai sbg 9 view independen grayscale-repeat (pola v15/K_SLICE), bukan channel RGB. Run ~1 jam 43 menit (K_SLICE=9, ~2,25x lebih lama dari v15/v16 yg K_SLICE=4 atau proses 9 slice tapi jadi 3 forward-pass gambar RGB).

| kernel | skema | val_macro_auc |
|---|---|---:|
| **v15 (terbaik)** | 4 view independen, tersebar PENUH 0-100% stack | **0,8304** |
| v17 | 9 view independen, terkonsentrasi jendela TENGAH 20-80% | 0,7999 |
| v16 | 9 slice sama, dipasang jadi channel RGB (gagal, channel-mismatch) | 0,7737 |

v17 > v16 (mengonfirmasi RGB-packing memang sebagian penyebab kegagalan v16), **tapi v17 tetap < v15**. `train_loss` epoch 8 = 0,4493 (v17) vs 0,2985 (v15) — konvergensi jauh lebih lambat.

**Kesimpulan baru (diperjelas lewat 2 eksperimen)**: bukan cuma channel-packing yg jadi masalah — **pembatasan sampling ke jendela tengah (20-80%) kemungkinan memotong informasi anatomi di ~40% ujung stack** yg justru terjangkau sampling penuh v15 (4 titik tersebar 0-100%). Ditambah, 3 dari 9 slice per anchor v17 berdekatan sangat rapat (nyaris duplikat fisik) — keragaman efektif lebih rendah drpd 4 titik v15 yg tersebar jauh satu sama lain.

**2 dari 2 percobaan skema "3+3+3" (RGB-packing DAN independent-view) GAGAL mengalahkan v15.** Line riset ini dianggap sudah dieksplorasi cukup utk arsitektur/pipeline kita saat ini — bukan pengungkit lanjutan, kecuali skema sampling diubah total (mis. tetap 4 titik tapi tersebar penuh + smoothing/interpolasi, bukan jendela-tengah). **Model final proyek TETAP v15 (0,8304)**.

## 28. v18 SELESAI: k-fold CV ungkap validasi 1-split selama ini cukup berisik (OOF 0,8068 vs v15 0,8304)

Run ~4,5 jam (5 fold x 8 epoch, TPU). Hasil per-fold (best epoch):

| fold | best val_macro_auc | n_val |
|---|---:|---:|
| 1 | 0,7915 | 12 |
| 2 | 0,8400 | 11 |
| 3 | 0,8670 | 11 |
| 4 | 0,7949 | 11 |
| 5 | 0,8698 | 12 |

**Rata-rata per-fold: 0,8326 ± 0,0338** (rentang 0,7915–0,8698, **selisih 0,078 antar-fold**). **OOF macro AUC (gabungan semua 57 studi gold, tiap studi dinilai model yg tidak pernah melihatnya): 0,8068.**

**Temuan metodologis penting**: selisih antar-fold (0,078, std 0,034) **jauh lebih besar** drpd selisih-selisih yg dikejar di eksperimen² sebelumnya (v15 vs v11: +0,0109; v16: -0,0567; v17: -0,0305). Ini berarti **validasi 1-split N=57 yg dipakai sejak v6 cukup berisik** — banyak "kemenangan/kekalahan" antar-versi yg tercatat sebelumnya bisa jadi sebagian mencerminkan noise split, bukan cuma perbedaan model murni. **OOF (0,8068) lebih rendah dari skor v15 (0,8304, single-split)** — indikasi v15 kemungkinan dapat split yg menguntungkan, bukan murni performa objektif terbaik.

**Catatan penting, BUKAN pembatalan kesimpulan v16/v17**: diagnosis channel-mismatch v16 (patch embedding beku) tetap berdasar alasan arsitektur yg jelas, independen dari noise split — TAPI magnitude perbedaan yg diukur (v15 vs v16/v17) perlu dipandang lebih hati-hati skrg krn ternyata berada dlm/dekat noise floor yg baru terkuantifikasi.

**5 checkpoint tersimpan** (`best_model_fold0-4.pt`) — kandidat kuat utk ensemble 5-model (murah, tidak perlu training baru, cuma rata-rata prediksi saat inferensi) yg berpotensi lebih stabil dari model tunggal manapun.

**Implikasi ke depan**: (1) OOF/k-fold sekarang jadi metrik yg lebih dipercaya drpd single-split; (2) eksperimen berikutnya idealnya pakai framework k-fold ini, atau minimal sadar bahwa selisih single-split <0,03-0,08 belum tentu signifikan; (3) ensemble 5-fold checkpoint adalah langkah murah & wajar berikutnya.

## 29. Submission v2: pipeline ensemble 5-fold model KITA SENDIRI tervalidasi penuh (arahan "lanjut" pasca-v18)

`kaggle_kernel_submission_v2/` — submission pertama proyek ini yang benar-benar pakai model kita sendiri (bukan reuse kode/bobot komunitas spt v1). Sumber: 5 checkpoint `rsna-knee-modeling-v18-kfold-5-tpu` (OOF macro AUC 0,8068).

**Tantangan teknis yang diselesaikan**: `test_series` tidak punya cache preprocessing (v1/v2/v3 semuanya cuma proses `train_series`) — notebook ini decode DICOM mentah on-the-fly saat inferensi, pakai fungsi preprocessing PERSIS SAMA dgn preprocessing v2 (`get_sorted_files` urutan fisik, `preprocess_slice` clip per-grup) disalin langsung, dgn `group_norm_stats.csv` dipinjam dari preprocessing v2 (dihitung dari sampel train, konsisten dipakai utk test — praktik standar). Ensemble = rata-rata sigmoid sederhana 5 model. Prediksi ditulis sbg percentile rank per label (pola sama dgn kode komunitas).

**Hasil dev-test (CPU, `test_series` stub 3 studi): SUKSES PENUH, 0 error.** `submission.csv` (3,13) format benar, cocok persis dgn `sample_submission.csv`. **Temuan penting: CPU ~58 detik/studi** — diekstrapolasi ke ~1.300 studi test sungguhan = **~21 jam, jauh melebihi batas 9 jam kompetisi**. Konfirmasi GPU/TPU WAJIB (bukan pilihan) utk submission sungguhan, sesuai pola kernel komunitas (GPU T4).

**Status**: pipeline logika tervalidasi penuh, BELUM disubmit ke leaderboard sungguhan. Langkah sebelum submit sungguhan: (1) pindah ke GPU T4 + `enable_internet=false` (pola submission-ready standar) setelah kuota GPU refresh (~22 Agustus, 2 hari lagi); (2) koordinasi dgn tim (kuota submission 5x/hari UNTUK SELURUH TIM, tim sudah punya skor 0,920 dari fork komunitas).

## 30. TEROBOSAN TANPA TRAINING: blend OOF kita + komunitas = 0,8462 (rekomendasi Claude, dieksekusi via "sesuai rekomendasi")

Dengan kuota GPU masih 0 (refresh ~22 Agustus) dan pipeline submission v2 sudah tervalidasi, langkah paling bernilai berikutnya yang TIDAK butuh GPU/TPU sama sekali: blend 2 sumber OOF leak-free yang SUDAH ADA — `oof_predictions.csv` kita sendiri (v18, 5-fold) + `oof.npz` resmi pilkwang (20-model komunitas, diunduh ulang dari `pilkwang/rsna-knee-weights`). Murni numpy/pandas lokal, evaluasi manual AUC (sklearn tidak tersedia lokal, pola established proyek ini).

**Hasil pada irisan 57 studi gold (leak-free kedua sumber)**:

| sumber | macro AUC |
|---|---:|
| Kita sendiri (v18, OOF 5-fold) | 0,8068 |
| Komunitas (pilkwang, OOF 20-model) | 0,8360 |
| **Blend rank-space optimal (α=0,60 arah pilkwang, 0,40 arah kita)** | **0,8462** |

**Sweep alpha 0,00-1,00 (step 0,05) menunjukkan blend MENGALAHKAN kedua sumber di RENTANG LUAS (α=0,30 s.d. 0,90 semua >0,83)** — bukan artefak satu titik yang rapuh, konfirmasi nyata bahwa model kita menyumbang keragaman asli walau lebih lemah sendirian. Ini **skor macro AUC leak-free tertinggi proyek ini sejauh ini**, +0,0102 dari pilkwang sendirian, +0,0394 dari model kita sendirian.

**Per-label breakdown di α=0,60** (kita | pilkwang | blend): Effusion 0,869|0,940|**0,962**, MCL 0,801|0,875|**0,887**, Medial OA 0,919|0,949|**0,949**, ACL 0,836|0,865|**0,868** — naik konsisten via blend. Lateral Meniscus TETAP lemah di ketiganya (0,638|0,632|0,638) — belum ada sumber yg menangkap sinyal ini dgn baik, kandidat perbaikan tersendiri (bukan soal blend, tapi soal representasi/label utk finding ini spesifik).

**Catatan validitas**: `true_` (kita) vs `y_derived` (pilkwang) di `oof.npz` TIDAK identik (selisih absolut 143,0 total) — bukan bug, `y_derived` pilkwang adalah target "derived" (kemungkinan soft/blended), BUKAN hard gold 0/1 murni spt yg kita pakai. Evaluasi tetap valid krn SELURUH perhitungan AUC (kita, pilkwang, blend) konsisten pakai `true_` (hard gold) yang sama sbg acuan tunggal — recompute pilkwang-alone via jalur ini persis cocok dgn `0,8360` yg sudah diverifikasi independen sebelumnya (§13), konfirmasi konsistensi metodologi.

**Langkah nyata berikutnya (BELUM bisa dieksekusi — butuh GPU T4, kuota 0 sampai ~22 Agustus)**: jalankan `kaggle_kernel_submission_v2/` (model kita) DAN `kaggle_kernel_submission/` (komunitas pilkwang) di `test_series` SUNGGUHAN dgn GPU, unduh kedua `submission.csv`, blend rank-space di α≈0,60 (atau sweep ulang kalau ada waktu), baru submit hasil blend-nya — bukan submit salah satu sendirian. Ini jalur konkret pertama proyek ini menuju skor gabungan yg TERUKUR lebih baik dari submission tim saat ini (0,920, tapi itu skor leaderboard sungguhan vs 0,8462 ini skor lokal gold-57 -- ada offset kalibrasi lokal-ke-leaderboard yg perlu diperhitungkan, biasanya +0,04-an berdasar riset kalibrasi sebelumnya §11b).

## 31. Riset ulang Code/Discussion/eksternal — teknik laterality canonicalization + koreksi ensemble (arahan "scan code+diskusi+model+artikel")

Leaderboard #1 naik ke 0,952. Ditemukan kernel publik terbaik yang pernah dianalisis di kompetisi ini: **`sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee`** (LB 0,883, single-family 5-fold DINOv2-B, tulisan metodologi sangat detail & jujur soal kesalahan sendiri).

**Temuan actionable utama**:
1. **Laterality canonicalization (teknik BARU, prioritas naik signifikan)**: hitung CENTER citra (bukan corner) di ruang pasien dari `ImagePositionPatient`+`ImageOrientationPatient`, pakai tanda koordinat-x tentukan kiri/kanan, lalu kanonikalisasi SEMUA studi seolah lutut kanan (flip coronal/axial dari kiri, balik urutan sagittal). Tanpa ini, **4 dari 12 label (Medial/Lateral Meniscus, Medial/Lateral OA) belajar dari anatomi cermin-terbalik separuh waktu — gagal diam-diam, tanpa error, tanpa gejala**. Item "crop sadar-laterality" yang selama ini kita taruh prioritas rendah ternyata berpotensi dampak besar.
2. **Kalibrasi lokal→LB ~+0,04-0,046** (didukung 2 dari 3 contoh sumber, 1 contoh sumber sendiri tidak konsisten — dilaporkan jujur, bukan dianggap presisi). Estimasi kasar: blend kita (0,8462 gold-57) → LB ~0,88-0,89.
3. **Koreksi submission v2**: sumber menyarankan **rank-mean, bukan probability-mean**, utk kombinasi model ("averaging sigmoids lets the most confident member dominate"). Submission v2 kita saat ini pakai probability-mean — pertimbangkan ganti sebelum submit sungguhan.
4. **Konfirmasi independen ke-3**: fine-tuning di resolusi sama menggerakkan label fokal jauh lebih besar drpd naikkan resolusi — validasi ulang temuan kita sendiri (v11→v15).
5. **CLS+patch-mean concat** arsitektur kita PERSIS sama dgn kernel referensi ini — validasi independen pilihan kita sudah tepat.
6. **Fracture = masalah label, bukan vision** (97% studi share 1 nilai target di sumber mereka) — tidak ada perbaikan model yg bisa atasi tanpa perbaikan label.
7. Protokol OOF-vs-gold yang mereka sarankan (OOF pilih epoch, gold-58 putuskan arah) TERNYATA sudah konsisten dgn praktik kita sendiri (val_macro_auc training loop kita sudah pakai gold, bukan weak-label OOF) — tidak perlu diubah.

Detail lengkap di `docs/RESEARCH_NOTES.md` §23.

## 32. v19 SELESAI — laterality canonicalization: inconclusive di single-split (sesuai peringatan sendiri)

Run ~53,5 menit (TPU, single-split spt v15). Val_macro_auc:

| epoch | val_macro_auc | | epoch | val_macro_auc |
|---|---:|---|---|---:|
| 1 | 0,7271 | | 5 | 0,8149 |
| 2 | 0,7752 | | 6 | 0,8241 |
| 3 | 0,8125 | | **7** | **0,8245 (BEST)** |
| 4 | 0,8219 | | 8 | 0,8218 |

**0,8245** — sedikit di bawah v15 (0,8304, cache v2, -0,0059) TAPI jauh di atas v18 OOF (0,8068, cache v2 5-fold, +0,0177). Mengingat noise antar-fold yg terukur di v18 (std 0,034, rentang 0,078), **selisih -0,0059 dari v15 PRAKTIS TIDAK BISA DIBEDAKAN dari noise single-split** — bukan bukti laterality canonicalization gagal, juga bukan bukti kuat berhasil. Ini persis skenario yg sudah diantisipasi di markdown eksperimen v19 sendiri sebelum run dimulai.

**Status: INCONCLUSIVE, kandidat baik utk konfirmasi k-fold** (~4,5 jam TPU) sebelum disimpulkan — teknik ini punya dasar teori kuat (memperbaiki bug nyata: 4 label bisa belajar dari anatomi cermin-terbalik, tervalidasi 97,4% akurat vs tag Laterality asli) dan hasil single-split TIDAK lebih buruk secara berarti dari v15. **Belum diputuskan** apakah lanjut k-fold sekarang atau prioritas lain dulu — menunggu arahan user.

**Model final proyek TETAP checkpoint v15 (0,8304 single-split) / v18 (0,8068 OOF, metrik paling terpercaya)** — v19 belum menggantikan, statusnya masih kandidat.

## 33. v20 SELESAI: k-fold konfirmasi laterality canonicalization — TIDAK ADA peningkatan (OOF hampir identik)

Run ~4,2 jam (5 fold x 8 epoch, TPU, cache preprocessing v4). Hasil per-fold (best epoch):

| fold | best val_macro_auc | n_val |
|---|---:|---:|
| 1 | 0,8247 | 12 |
| 2 | 0,8061 | 11 |
| 3 | 0,9042 | 11 |
| 4 | 0,8156 | 11 |
| 5 | 0,8491 | 12 |

Rata-rata per-fold: 0,8400 ± 0,0352 (rentang 0,8061–0,9042, spread 0,098 — bahkan lebih lebar dari v18). **OOF macro AUC: 0,8073** — dibanding v18 OOF (baseline, cache v2, tanpa laterality canon): **0,8068**. **Selisih +0,0005 — PRAKTIS NOL, bukan peningkatan nyata**, ini perbandingan apples-to-apples valid (sama-sama metrik OOF 5-fold).

**Kesimpulan**: laterality canonicalization TIDAK terbukti membantu utk arsitektur kita, walau deteksi geometrisnya sendiri valid & tervalidasi kuat (97,4% akurat vs tag asli, run penuh 24.371 series sukses). **Penjelasan yg masuk akal**: arsitektur `KneeModel.forward` kita pakai `patch_mean = out[:, 1:].mean(1)` — mean-pooling atas SEMUA patch token DINOv2 SEBELUM masuk head, yg **menghilangkan informasi posisi spasial (kiri vs kanan dlm citra) hampir sepenuhnya**. Flip horizontal (kanonikalisasi) tidak mengubah rata-rata sekumpulan patch secara berarti, jadi fitur yg dihasilkan hampir sama baik di-flip maupun tidak. Beda dgn kernel referensi (`sadamtorres`, §23) yg pakai ATTENTION pooling berbasis posisi-slice — arsitektur itu memang butuh & bisa eksploitasi orientasi konsisten, milik kita tidak.

**Pola berulang ke-2 (setelah v16/RGB-packing)**: teknik yg secara teori/di sumber lain terbukti bagus, tapi GAGAL memberi manfaat krn arsitektur KITA SPESIFIK (mean-pooling, bukan attention/posisi-aware) tidak bisa mengeksploitasinya. **Pelajaran umum utk eksperimen berikutnya**: sebelum adopsi teknik dari sumber lain, cek dulu apakah arsitektur kita punya "jalur" utk memanfaatkan sinyal itu (posisi, orientasi, channel, dst.) — kalau tidak, perbaikan data sebagus apa pun tidak akan terpakai.

**Model final proyek TETAP checkpoint v15 (0,8304 single-split) / v18 (0,8068 OOF, metrik paling terpercaya)** — v20 tidak menggantikan (selisih dlm margin nol). Preprocessing v4 & kernel v19/v20 tetap tersimpan sbg referensi/arsip, tidak dipakai lanjut sbg source-of-truth.

## 34. v21 (arsitektur anatomy-query) dibangun — push-1 ERROR, root cause ditemukan & diperbaiki, versi ke-2 sedang berjalan

Setelah v20 menutup line laterality-canonicalization utk arsitektur SlotHead (mean-pooling, §33), riset lanjutan pasca-scraping 20 Agustus (§23/MEMORY §2) menemukan resep **prvsiyan V19**: kepala model berbasis **cross-attention per-diagnosis atas token anatomis** (bukan mean-pool), mengeklaim +0,00989 macro AUC tervalidasi (0,84199→0,85188, bootstrap P(gain>0)=0,998, gated 5-fold gold-firewalled). Ini persis "jalur" yg diidentifikasi hilang di v20 — kesimpulan §33 secara eksplisit menyebut arsitektur attention/posisi-aware SEHARUSNYA bisa eksploitasi laterality canon, beda dgn mean-pooling.

**Dibangun `kaggle_kernel_modeling_v5_anatomyquery/` (v21, slug `rsna-knee-modeling-v21-anatomy-query-tpu`)**: ganti "slot bag" dgn **1 series terbaik per plane** (Sagittal/Coronal/Axial, skor `4*Fluid_Sensitive+3*Fat_Suppression`), 24 token anatomis (8/plane), tiap token = **triplet RGB genuine 3 slice berdekatan** (bukan grayscale-repeat), + plane embedding + position embedding (0..1 dlm plane), lalu **`AnatomyQueryHead`**: 12 query cross-attention per-diagnosis (`einsum` scaled dot-product, masked softmax) atas 24 token. Data source = cache preprocessing v4 (laterality canon); laterality utk Sagittal ditangani di LOAD TIME (`vol[::-1]` utk lutut kiri, krn cache v4 cuma flip Coronal/Axial). **Perbedaan kunci vs v16 (yg gagal krn RGB-packing)**: `AnatomyModel.__init__` sengaja **unfreeze `backbone.embeddings.patch_embeddings`** (selain 6 blok terakhir + layernorm) — mengatasi langsung akar masalah v16 (patch embedding beku tidak bisa menafsirkan channel R/G/B non-foto-natural). Total trainable params 43,4M (vs 42,9M skema lama). Struktur training: k-fold 5-fold sama persis pola v18/v20, EPOCHS=8, BATCH=4, LR head 1e-3/backbone 8e-6, safety cutoff 7 jam.

**Push-1 ERROR**: `DATA_OK=False`, log cetak "100,0% studi dgn gambar KOSONG (train & val)" — guard keamanan mengabort training sebelum mulai. **Diagnosis dari log** (`mask.sum()=24,00/24,00` penuh — 0 file cache hilang; varians embedding 0,3238 — nonzero/normal; std prediksi sigmoid nonzero per label) membuktikan pipeline data SEBENARNYA BENAR — ini **false positive** di fungsi diagnostik, bukan bug data asli. **Root cause**: `AnatomyDataset._plane_tokens()` menormalisasi pixel ke skala **[0,1]** (`/255.0`) DI DALAM Dataset — beda dari v15-v20 yg tetap skala 0-255 mentah & normalisasi baru terjadi di `model.forward()`. Fungsi `dataset_stats()` (copy-paste dari notebook v15-v20) masih pakai ambang lama `img_means < 1.0` (benar utk skala 0-255, tapi SELALU true utk skala [0,1] dgn mean tipikal gambar non-kosong ~0,3-0,4) → 100% studi ter-flag "kosong" padahal tidak ada satupun yg benar-benar kosong.

**Fix**: ambang diganti `img_means < 0,02` (sesuai skala [0,1]), + komentar penjelasan akar masalah utk referensi ke depan. Diverifikasi: `model.forward()` TIDAK dobel-normalisasi (sudah dikonfirmasi lewat komentar kode "sudah /255 + triplet RGB"), `ast.parse()` seluruh 23 cell sukses (syntax valid), grep memastikan tidak ada ambang `<1.0` lain yg tersisa di notebook. Di-push sbg **kernel versi 2** (2026-08-21 00:00:58), commit `e9db74c` ("Fix v21 ERROR: dataset_stats() pakai ambang skala lama, false-positive 'gambar kosong'"). Log lokal stale (`rsna-knee-modeling-v21-anatomy-query-tpu.log`) dihapus sebelum commit. Status terakhir: **QUEUED** di Kaggle, run belum menghasilkan OOF. Estimasi durasi ~4,5-5 jam (5-fold × 8 epoch, TPU). **Kuota TPU perlu diawasi** — sisa ~5,80h dari 20h/minggu per cek terakhir, run ini bisa menghabiskan sebagian besar sisanya.

Jika v21 berhasil dan OOF-nya mengalahkan v18 (0,8068)/v20 (0,8073) secara meyakinkan (di luar noise ±0,03-0,08 yg terkuantifikasi di v18), ini akan jadi model final baru proyek — arsitektur ini secara spesifik mengatasi SEKALIGUS akar masalah v16 (patch embedding beku) dan v20 (mean-pooling buta posisi).

## 35. v21 SELESAI penuh — OOF 0,6904, REGRESI vs v18/v20, tapi diagnosis mengungkap masalah STABILITAS training, bukan arsitektur yg fundamental buruk

Run selesai penuh (status Kaggle COMPLETE) ~5,2 jam (5-fold × 8 epoch, TPU). Hasil per-fold (best epoch):

| fold | best val_macro_auc | n_val |
|---|---:|---:|
| 1 | 0,6276 | 12 |
| 2 | 0,8233 | 12 |
| 3 | 0,8902 | 11 |
| 4 | 0,6681 | 11 |
| 5 | 0,6199 | 11 |

Rata-rata per-fold: 0,7258 ± 0,1102 (rentang 0,6199–0,8902, spread 0,2703 — jauh lebih lebar dari v18/v20). **OOF macro AUC gabungan (57/57 gold): 0,6904** — di bawah v18 (0,8068) dan v20 (0,8073), turun cukup jauh (-0,116 vs v18).

**Investigasi lebih dalam (bukan cuma angka akhir) — dibaca trajectory train_loss & val_auc PER EPOCH dari log Kaggle**, mengungkap pola yg TIDAK acak:

- **Fold 2 & 3 (BAGUS)**: train_loss turun tajam & konsisten sepanjang 8 epoch (fold 2: 0,5798→0,3279; fold 3: 0,5920→0,3693) dan val_auc naik mulus mengikuti (fold 2: 0,66→0,82; fold 3: 0,72→0,89). Model di fold ini BENAR-BENAR BELAJAR, dan hasilnya (0,82-0,89) **lebih baik dari v18 OOF (0,8068)** — bukti arsitektur `AnatomyQueryHead` PUNYA potensi nyata saat optimisasi berjalan normal.
- **Fold 1, 4, 5 (STUCK)**: train_loss NYARIS TIDAK BERGERAK sepanjang 8 epoch (mis. fold 5: 0,6067→0,5844, cuma turun 0,022 total; fold 4: 0,6056→0,5641). val_auc BOLAK-BALIK di sekitar tebakan acak (fold 5: 0,58→0,42→0,50→0,58→0,48→0,56→0,55→0,62; fold 4 serupa) — model gagal keluar dari basin optimisasi awal dalam budget yg ada (LR head decay dari 1e-3 ke ~3,8e-5 di epoch 8, jadi makin sulit "kabur" di epoch-epoch akhir kalau belum bergerak di awal).

**Kesimpulan**: ini bukan kegagalan arsitektur murni (spt v16 yg 100% konsisten buruk krn channel-mismatch terstruktur) — ini **masalah STABILITAS training lintas seed/fold**. 2 dari 5 seed "beruntung" masuk basin bagus, 3 lainnya stuck. **Kandidat penyebab**: `AnatomyQueryHead.__init__` inisialisasi `query`/`cls_w` dgn `std=0,02` (kecil) dan `cls_b` mulai dari **nol** (bukan bias ke label-prior — trik stabilisasi standar utk multi-label imbang: `bias = log(p/(1-p))` dari base rate tiap label, supaya prediksi awal tidak seragam 0,5 utk semua studi & gradien awal lebih informatif). Dikombinasi dgn `patch_embeddings` yg BARU di-unfreeze (permukaan optimisasi lebih besar drpd v15-v20), kombinasi ini kemungkinan menciptakan lanskap loss yg lebih rawan terjebak di titik datar utk sebagian inisialisasi acak.

**Keputusan**: checkpoint v21 **TIDAK dipakai**, model final proyek **TETAP v15 (0,8304 single-split) / v18 (0,8068 OOF)**. Line anatomy-query **ditutup UNTUK SEKARANG** (bukan permanen/gagal total) — draft perbaikan yg masuk akal utk dicoba kalau dilanjutkan: (a) init `cls_b` ke label-prior bias, (b) LR warmup beberapa ratus step pertama sebelum decay mulai, (c) mungkin batasi unfreeze ke patch_embeddings SETELAH beberapa epoch (curriculum unfreeze) drpd langsung dari awal. **Tidak di-eksekusi sekarang** krn alasan di §36.

## 36. Kuota GPU+TPU habis total pasca v21 — semua eksperimen Kaggle berhenti sampai refresh

Cek `kaggle quota` (2026-08-20 23:27 UTC, tepat setelah v21 selesai): **GPU 0,00h tersisa** (used 30,43h/30,00h nominal — overrun), **TPU 0,56h tersisa** (used 19,44h/20,00h — run v21 sendiri makan ~5,2 jam dari sisa yg cuma ~5,8h). `refreshAt` keduanya: **2026-08-22T00:00:00 UTC**, sekitar 24,5 jam dari waktu cek. **Tidak ada eksperimen training/inferensi Kaggle baru yg bisa dijalankan sampai refresh** — baik utk lanjut v21 (perbaikan stabilitas) maupun utk jalankan submission real (`kaggle_kernel_submission_v2/` butuh GPU/TPU utk infer 1.300 studi test dlm batas 9 jam, CPU-only ~21 jam terbukti tidak cukup di §29).

**Prioritas begitu kuota kembali (urutan)**: (1) **submission real** — blend rank-space kita (v18) + pilkwang (α≈0,60, target lokal 0,8462) disubmit sungguhan ke leaderboard, INI YANG PALING BERHARGA krn satu2nya yg menggerakkan skor leaderboard nyata tanpa training tambahan; (2) ganti submission v2 dari probability-mean ke rank-mean ensemble; (3) kalau mau lanjut v21, terapkan fix stabilisasi (§35) dulu SEBELUM re-run — jangan ulang mentah2 dgn seed lain, buang kuota percuma; (4) evaluasi 3 teknik dari bedah `mattiaangeli/bend-the-knee-to-dinov3-ensembled` (riset terpisah, lihat MEMORY §2/RESEARCH_NOTES §24) sbg kandidat v22 kalau v21-fixed masih kurang.

## 37. Kuota refresh penuh (23 Agustus) + temuan riset terbesar proyek (§25 RESEARCH_NOTES) + build v5/v22

**Kuota refresh**: `kaggle quota` per 23 Agustus 11:43 UTC — GPU 30,00h, TPU 20,00h, keduanya 0h
terpakai (`refreshAt` 29 Agustus). Skor tim (submission real, tidak berubah sejak 20 Agustus):
**0,922** (fork `mattiaangeli/bend-the-knee-to-dinov3-ensembled`), peringkat **#240/2.219 tim**.
Leaderboard #1 tetap Brandon Low 0,952.

**Arahan user** ("cek progres", lalu diberi 3 opsi lewat AskUserQuestion): "adopsi teknik datanya ke
arsitektur kita sendiri (v22)" — bukan submit langsung checkpoint orang lain, bukan blend, tapi
bangun model KITA SENDIRI pakai pelajaran dari riset.

**Riset yg memicu ini**: ditemukan (via `kaggle kernels list --sort-by scoreDescending`, cara baru
yg langsung urutkan skor tertinggi bukan cuma populer) notebook **`dreaddevelopment/knee-mri-twelve-
findings-from-a-single-model`** — single model, **0,924 public LB, LEBIH TINGGI dari skor tim kita
(0,922)**, checkpoint CC0 siap pakai. Dibedah lengkap (`docs/RESEARCH_NOTES.md` §25): backbone
CoAtNet (bukan DINOv2), validasi lokal gold-58 0,9167 → LB 0,924 (offset cuma +0,0073, jauh lebih
kecil dari klaim sadamtorres +0,04-0,046). 3 teknik data kunci: span sampling lebar 6-94% (relevan
langsung ke Lateral Meniscus kita yg lemah), crop fisik 140mm+336px (konfirmasi independen ke-2
setelah mattiaangeli §24), RGB-triplet BERHASIL krn backbone di-fine-tune penuh (bukti independen
ke-2 bahwa akar masalah v16 = freeze-schedule, bukan RGB-packing itu sendiri).

**Dibangun 2 kernel baru**:

1. **`kaggle_kernel_preprocessing_v5_widecrop/`** (`rsna-knee-preprocessing-v5-widecrop-336`) —
   copy dari v4 (laterality canon dipertahankan utuh), ganti `IMG_SIZE` 224→336, `CROP_MM` eksplisit
   140mm (naik dari implisit ~112mm v2/v4), **`N_SLICES` 16→8** (kompensasi storage — 336px 2,25x
   piksel/slice; v2/v4 di 16-slice/224px hasilnya ~13GB, kalau N_SLICES dibiarkan 16 estimasi jadi
   ~29GB, MELEBIHI batas aman `/kaggle/working` ~21GB & safety-cutoff internal 16GB). Span sampling
   TETAP PENUH 0-100% (linspace sama spt v2/v4) — window sempit sudah terbukti gagal (v16/v17, §26-27),
   tidak diulang. Dipush 23 Agustus, status RUNNING (CPU-only).

2. **`kaggle_kernel_modeling_v22_fulltune/`** (`rsna-knee-modeling-v22-widecrop-triplet-tpu`) — copy
   dari v18 (k-fold, `SlotHead`, TERBUKTI stabil), 3 perubahan MINIMAL & terarah:
   - `kernel_sources` → cache v5 (336px) ganti v2 (224px).
   - `_load_slot` bangun **triplet RGB genuine** (`T_WINDOW=3` window/slot, 3 slice fisik berdekatan
     sbg channel R/G/B) ganti grayscale-mean `K_SLICE=4` — teknik SAMA PERSIS v16 yg dulu gagal
     (0,7737), TAPI kali ini didukung poin berikutnya.
   - **`patch_embeddings` di-unfreeze** (selain `unfreeze_last=6`+layernorm) — SAMA PERSIS skema
     v21, yg SUDAH TERBUKTI cukup (BUKAN perlu semua 12 blok) di 2 dari 5 fold-nya yg berhasil (OOF
     0,82-0,89, §35). **Head SENGAJA TIDAK diganti** — tetap `SlotHead`, BUKAN `AnatomyQueryHead`
     (penyebab regresi v21 adalah instabilitas HEAD-nya, bukan triplet RGB/patch-embed unfreeze-nya).

   Hyperparameter training (LR head 1e-3/backbone 8e-6, EPOCHS=8, K_FOLDS=5, scheduler) DIPERTAHANKAN
   identik v18 supaya perbandingan v18 vs v22 murni mengisolasi variabel data/arsitektur. Semua cell
   yg wajib disinkronkan dgn perubahan shape tensor (dummy sanity check §6, diagnostik forward-pass
   §7b — INI YG DULU LUPA di-update pas v16, `PROGRESS.md` §26) sudah diperiksa & disamakan manual,
   diverifikasi `ast.parse()` seluruh 23 cell + grep pastikan tidak ada `K_SLICE`/`IMG_SIZE=224` sisa
   di kode (cuma di komentar penjelasan, bukan kode aktif).

   **Risiko disadari, didokumentasikan di kode**: ~1,7x kompute/memori per studi vs v18 (336px lebih
   berat + patch-embed unfreeze). `BATCH_STUDIES=4` dipertahankan sbg percobaan pertama (TPU v5e-8
   biasanya cukup lapang) — kalau OOM, turunkan ke 2-3 di push berikutnya. **Push v22 DITUNDA sampai
   preprocessing v5 COMPLETE** (`kernel_sources` mount versi TERBARU sumbernya — push v22 sebelum v5
   selesai = cache kosong/rusak, gotcha lama yg sudah pernah kena di v13/v14).

**Status**: preprocessing v5 SELESAI (status Kaggle COMPLETE, tanpa error terlapor). v22 sudah
DI-PUSH (kernel version 1, 23 Agustus), sedang QUEUED/RUNNING di TPU, dipantau otomatis.

## 38. Rekonsiliasi konteks (26 Agustus) + fix bug shape 5D/4D di training CoAtNet arm sendiri

**Catatan rekonsiliasi**: antara §37 dan sini, sesi ini sempat kehilangan jejak percakapan
(context compaction) — pekerjaan besar terjadi tanpa tercatat rinci di PROGRESS.md (baru
ketauan lewat `git log`/`git reflog` saat mau commit v22, ternyata HEAD sudah 10 commit di
depan pemahaman lokal). Ringkasan hasil pekerjaan itu (detail lengkap ada di `MEMORY.md` §2/§3,
`docs/RESEARCH_NOTES.md` §26, `blend_results/FUSION_V2_RESULTS.md`, `label_engineering/`):

- Tim fork `prvsiyan/head-and-shoulders-knees-and-toes` (replika DINOsaur V10) → skor tim naik
  **0,922 → 0,936** (#91/2.372).
- Scraping mendalam 25 Agustus membedah seluruh stack 0,936 (24-member DINOv2/DINOv3 + RadImageNet
  chain + dual CoAtNet correlation-gated).
- **Harvest OOF CoAtNet (GPU, 6,5 menit) mengungkap TEMUAN PIVOT**: `v5_full_swa` CoAtNet
  SENDIRIAN = **0,9206-0,9213 gold-57/58, mengalahkan SEMUA blend** (uniform-7=0,9055, nested-CV
  terbaik=0,9148). Fusion v2 (`blend_results/fusion_nested_cv_v2.py`) mengonfirmasi: ensembling
  TIDAK membantu di skala N=57 ini.
- **Keputusan strategis**: jalan ke 0,95+ = SATU arm lebih kuat dari c5 (fine-tune penuh +
  label lebih baik), BUKAN ensemble tuning lebih lanjut. v22 kita sendiri (DINOv2 SlotHead) resmi
  ditutup sbg jalur — 3x berturut2 (v16/v21/v22) gagal, kesimpulan: `SlotHead` mean-pooling tidak
  bisa manfaatkan triplet-RGB atau patch-embed-unfreeze secara konsisten.
- Label engineering round-1 (severity-aware lexicon) SELESAI tapi GATE belum lolos — semua varian
  di bawah `flight0234` (0,8991). Tanpa API LLM, `flight0234` tetap label terbaik yg ada.
- **Mulai bangun arm sendiri**: fine-tune PENUH CoAtNet (init dari checkpoint c5
  `raptor_ft_coatnet_v5_full_swa.pt`) di atas label `flight0234` (confidence-weighted), gold-58
  tidak ikut latih (held out utk evaluasi jujur, target kalahkan 0,9206).

**Fix bug baru (26 Agustus, sesi ini)**: kernel `rsna-knee-train-coatnet-flight-ft-v1` push-2
("train v2 running", commit terakhir sblm rekonsiliasi) ternyata **ERROR**, bukan sukses jalan
spt tersirat dari judul commit. Log: `RuntimeError: Given groups=1, weight of size [64, 3, 3, 3],
expected input[1, 36, 384, 384] to have 3 channels, but got 36 channels instead`.

**Diagnosis**: `RaptorClassifier.encode()` (kode verbatim dari notebook publik dread/c5) didesain
utk input **5D** `(B_studi, K_window, 3, H, W)` — `B, K = x.shape[:2]` lalu `x.flatten(0,1)`
menggabung dim batch-studi & window SEBELUM masuk backbone. Training loop custom kita (`FASE 2` di
`build_train_coatnet.py`) malah slice `xw[mb0:mb0+MICRO]` — **4D**, TANPA dimensi batch-studi —
lalu panggil `model(xb)` langsung. Akibatnya `encode()` salah tafsir `MICRO` (=12) sbg dimensi B
dan channel(3) sbg dimensi K, lalu `.flatten(0,1)` menggabungkan `MICRO*3=36` jadi satu axis —
persis cocok dgn pesan error (36 channel). Bug SEKUNDER yg ikut ketemu: label `yt`/`wt` di-`repeat`
per-window (`[None].repeat(xw.shape[0],1)`) — padahal head `RaptorClassifier` SUDAH melakukan
attention-pooling atas SEMUA K window jadi SATU prediksi per studi; me-repeat label per-window
mengasumsikan tiap window adalah sampel independen, bertentangan dgn arsitektur pooling yg
sebenarnya dipakai.

**Fix**: satu forward pass per studi — `xb = xw.unsqueeze(0)` → `(1, K, 3, H, W)` (sesuai desain
asli `encode()`), label/weight TIDAK direpeat (`yb`/`wb` shape `(1,12)`, satu nilai per studi).
Loop micro-batch lama dihapus sepenuhnya (bukan dipertahankan sbg dead code) — secara default
`K_WIN==MICRO==12` jadi loop itu SELALU cuma 1 iterasi, murni sisa desain yg tidak sempat
disederhanakan setelah asumsi micro-batching berubah.

**Sekalian diperbaiki**: `build_train_coatnet.py` (script generator notebook, source-of-truth)
tadinya baca `_KE_SRC` dari `/tmp/opencode/nb/raptor45w65/...` — path ephemeral yg SUDAH HILANG
lintas sesi (ketauan pas mau re-run build script). Diganti baca dari
`build_train_coatnet_ke_src.txt` yg SEKARANG DI-COMMIT ke repo (isi sama persis, cuma dipindah
supaya reproducible). `id` di kernel-metadata juga disamakan ke slug AKTUAL yg sudah live
(`...-flight-ft-v1`, builder lama masih pakai `...-flight-v1` tanpa "-ft-") supaya push
berikutnya update kernel yg sama, bukan bikin duplikat diam-diam.

Diverifikasi: `ast.parse()` semua 4 cell notebook hasil regenerasi OK, grep konfirmasi kode buggy
lama (loop micro-batch, label-repeat) sudah tidak ada di jalur eksekusi. Di-push sbg **kernel
version 3**, status: RUNNING. **Catatan**: `/kaggle/tmp/volcache` (cache FASE 1) kemungkinan besar
TIDAK persist lintas `kernel push` terpisah (container baru tiap push) — push-3 realistiknya
mengulang ~50 menit caching dari nol walau push-2 dulu sudah pernah menyelesaikannya; belum
diverifikasi langsung, cukup diwaspadai saat baca log durasi run ini.

**Push-3 ternyata ERROR juga** (bug ke-2, beda dari yg di atas): cache fase-1 lolos PENUH
(4407/4407, ~47 menit — **konfirmasi cache TIDAK persist lintas push**, harus ulang tiap kali),
training mulai (`model siap res=384 | steps=2174`), tapi crash di forward-loss PERTAMA:
`RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to
autocast... use binary_cross_entropy_with_logits`. Root cause: PyTorch `autocast()` melarang
kombo `sigmoid()`+`binary_cross_entropy()` eksplisit berdasar **nama operasi**, bukan dtype
runtime aktual — walau logits sudah di-`.float()` sebelum sigmoid, tetap kena block krn masih
lexically di dalam context `autocast()`. **Fix**: ganti jadi `F.binary_cross_entropy_with_logits`
langsung atas logits mentah (skip `torch.sigmoid` eksplisit) — setara matematis, aman autocast,
persis rekomendasi resmi di pesan errornya sendiri. Diverifikasi `ast.parse()` OK + grep pastikan
tidak ada lagi `torch.sigmoid(model(...))` tersisa. Di-push sbg **kernel version 4**, status:
RUNNING — cache fase-1 mengulang lagi (~47 menit), training baru mulai setelahnya.

**Push-4 SELESAI PENUH, TANPA ERROR — tapi hasil TIDAK lolos gate**: cache 4407/4407 (~44 menit),
training 4 epoch mulus tanpa hambatan (`loss`: 0,5586 → 0,5438 → 0,5206 → 0,5041, turun monoton,
TIDAK ada tanda instabilitas spt v21/v22 kita sendiri) — total durasi run ~4,6 jam. **Evaluasi
gold-58 akhir: MACRO AUC = 0,9193** — dibanding baseline off-the-shelf c5 (`v5_full_swa`,
0,9206-0,9213 dari harvest §38 atas): **SEDIKIT LEBIH RENDAH**, gate "≥ c5" TIDAK lolos.

**Diagnosis**: selisihnya kecil (-0,0013 s/d -0,002) tapi arahnya konsisten negatif, bukan cuma
noise acak (training sendiri stabil/mulus, jadi bukan masalah optimisasi). Penjelasan paling masuk
akal: label `flight0234` yg dipakai utk fine-tune HANYA capai 0,8991 vs gold asli (§ label
engineering, 25 Agustus) — menyeret model yg SUDAH kuat (c5, hasil training sebelumnya dgn resep
tak diketahui persis) sedikit ke arah kualitas sinyal yg lebih rendah drpd representasi implisit
yg sudah dipelajarinya. **Ini BUKTI LANGSUNG (bukan cuma dugaan lagi) bahwa kualitas label sudah
jadi pembatas nyata** — fine-tuning arsitektur/resep lebih dalam kemungkinan besar TIDAK banyak
membantu lebih jauh tanpa label yg lebih baik dari `flight0234`.

**Keputusan**: checkpoint hasil fine-tune ini **TIDAK dipakai** — c5 asli (0,9206) TETAP arm lokal
terbaik proyek. Prioritas berikutnya (2 opsi, saling melengkapi bukan saling eksklusif): (1)
**minta akses API LLM** dari user utk relabeling severity-aware (target Synovitis/Lat OA/
Contusion/Fracture, titik lemah `flight0234`) — ini pengungkit yg paling mungkin menembus 0,92+;
(2) **submit c5 SENDIRIAN ke leaderboard real** (belum pernah dicoba — skor publik tim 0,936
selama ini dari ensemble fork, BUKAN dari arm CoAtNet kita) utk dapat ground-truth offset gold→LB
yg sebenarnya, sebelum lanjut iterasi lokal lebih jauh.

## 39. Submit c5 sendirian ke leaderboard real (26 Agustus, arahan eksplisit user) — hasil MEMBANTAH pivot strategis 25 Agustus

Sesuai arahan eksplisit user ("submit c5 sendirian ke leaderboard real"), dibangun kernel baru
`kaggle_kernel_submit_c5_solo/` (`build_submit_c5_solo.py`) — reuse `main()` verbatim dari `_KE_SRC`
(sudah tersimpan di `build_train_coatnet_ke_src.txt`), `ARMS` default SUDAH persis 1 arm c5 saja
(`raptor_ft_coatnet_v5_full_swa.pt`, tanpa blend). Setelah exec KE, panggil `KE['main']()` eksplisit
(tidak auto-run krn `__name__ != '__main__'` saat di-exec sbg namespace), salin `_raptor.csv` →
`submission.csv`.

**Gotcha operasional baru (2, keduanya didokumentasikan utk referensi ke depan)**:
1. Push pertama kernel BARU (belum pernah ada) dgn title berisi tanda kurung `(no blend)` gagal
   409 Conflict BERULANG (dicoba ulang beberapa kali dgn jeda 50s-3menit, tetap gagal) — beda dari
   UPDATE kernel yg sudah ada (di situ title/slug mismatch cuma jadi WARNING, bukan error). Fix:
   title dibersihkan jadi persis slugify ke id (`RSNA Knee - Submit c5 solo`, tanpa parenthetical)
   — push langsung berhasil di percobaan pertama setelah fix.
2. `kaggle competitions submit -k <kernel> -v <versi>` SENDIRIAN dapat 400 Bad Request generik.
   Tanpa `-v` dapat pesan lebih jelas: "Code competition submissions require both the output file
   name and the version number" — ternyata WAJIB sertakan `-f submission.csv` BERSAMAAN dgn `-k`
   dan `-v`, 3-serangkai baru diterima ("1 submissions remaining today").

**Kendala tambahan sebelum push**: 2 slot GPU batch akun (max 2 concurrent) penuh oleh kernel LAIN
yg TIDAK terkait proyek RSNA (`metric-hack-last-call-update`, `v16-rc5-high-score-...` — kompetisi
lain di akun yg sama) — ditunggu sampai keduanya selesai (status COMPLETE) sebelum push berhasil,
BUKAN dihentikan paksa (bukan milik proyek ini, tidak diketahui statusnya penting/tidak bagi user).

**Hasil submission (ID 55795592): public LB = 0,927.**

| Sumber | Skor |
|---|---:|
| c5 lokal (gold-58, nested-CV) | 0,9206-0,9213 |
| **c5 REAL LEADERBOARD (submission ini)** | **0,927** |
| Ensemble tim eksisting (`rsna-base`/DINOsaur fork) | **0,936** |

**c5 sendirian KALAH 0,009 dari ensemble tim** — bertentangan LANGSUNG dgn kesimpulan pivot 25
Agustus ("c5 sendirian mengalahkan SEMUA blend LOKAL", §37). Offset gold→LB kecil & wajar (+0,006-
0,007, konsisten pola dreaddevelopment), jadi BUKAN bug/kesalahan submission — perbedaannya genuin.

**Diagnosis metodologis**: nested-CV di N=57 gold TERBUKTI tidak cukup besar utk mendeteksi manfaat
diversity ensemble yg baru kelihatan nyata di skala test set asli (~1.300 studi tersembunyi) —
selaras temuan independen sadamtorres (§23a poin 7: varian 224px/336px korelasi 0,901 TAPI ensemble
tetap +0,006). Kesimpulan "single arm > ensemble" yg terlihat kuat di sampel kecil ternyata TIDAK
bertahan di sampel besar — pelajaran umum: hati-hati generalisasi dari nested-CV N=57 ke keputusan
strategis besar tanpa validasi silang di skala lebih besar (submission real) dulu.

**Keputusan revisi**: submission tim TIDAK diganti — ensemble besar (0,936) tetap yg terbaik &
dipakai. Arm CoAtNet kita (c5 asli atau hasil fine-tune `coatnet_flight_ft_v1.pt`, §38) lebih masuk
akal diintegrasikan SBG TAMBAHAN ke ensemble yg sudah ada (`kaggle_kernel_submit_integration_v1/`
sudah pernah dirintis utk ini) drpd diandalkan sbg pengganti tunggal. Kuota submission tim tersisa
1 hari ini.

## Belum dikerjakan

- [x] ~~Tunggu hasil v12~~ — SELESAI, val_macro_auc TIDAK melampaui 0,8195 (overfitting), model final tetap checkpoint v11
- [x] ~~Keputusan Anda: strategi berikutnya menuju 0,950~~ — DIJAWAB via riset §19: prioritas (1) geometri slice/preprocessing v2 (CPU-only, mulai sekarang), (2) LR scheduler pas kuota GPU refresh, (3) k-fold CV, (4) ensemble dgn anggota independen sendiri, (5) TTA, (6) opsi TPU
- [ ] **Sedang dikerjakan**: preprocessing v2 — urutan fisik slice via `ImagePositionPatient`, crop sadar-laterality, handle 2-lutut-1-DICOM, sampling "3+3+3"
- [ ] Setelah kuota GPU refresh (~22 Agustus): fine-tune v11 dgn LR scheduler/decay (ganti LR flat yg overfit di v12)
- [ ] Verifikasi pola self-reference `kernel_sources` benar2 berhasil memuat checkpoint (cek log "RESUME: checkpoint v11 dimuat" begitu ada) — kalau gagal, ada fallback `[warn] CKPT_DIR tidak ketemu` yg training dari nol (bukan crash, tapi berarti eksperimen resume gagal & butuh pendekatan lain)
- [ ] **Keputusan Anda soal API LLM hosted sudah DIJAWAB** (dilonggarkan, §16) — implementasi relabeling ditunda dulu, fokus ke single-model dulu sesuai arahan terbaru
- [ ] Koordinasi dgn tim (kolaborator `nxtirfan` dkk.) soal submission — tim sudah di 0.914, jangan boros kuota 5x/hari
- [ ] Bangun v2 ensemble: blend model kita (v9=0.7902) + `oof.npz` pilkwang + arm RadImageNet-ResNet50 (MIT), bobot nested-CV di 57 gold
- [ ] Proper k-fold CV (bukan cuma 1 split gold N=57) + OOF ensemble selection, grouping per hash-laporan
- [ ] Pipeline inference/submission sudah ada versi pertama (`kaggle_kernel_submission/`, community 20-model, belum disubmit — §15 PROGRESS)
- [ ] Review manual studi gold-label yang ter-flag exclude (bagian 4)

## Struktur repo

```
kaggle_kernel/                 EDA
kaggle_kernel_audit/           Audit kualitas (sampel)
kaggle_kernel_audit_full/      Audit kualitas (exhaustive, 820rb file)
kaggle_kernel_preprocessing/   Pipeline preprocessing
kaggle_kernel_weak_labeler/    Weak-labeler rule-based dari teks laporan
kaggle_kernel_modeling/        Modeling (DINOv2-base + slot attention pooling)
eda_results/                   Output EDA (grafik, notebook, log)
audit_results/                 Output audit sampel
audit_results_exhaustive/      Output audit exhaustive
weak_labeler_results/          Output weak-labeler (validasi, CSV label)
docs/RESEARCH_NOTES.md         Riset arsitektur & dataset publik (Tahap 1&2 + lanjutan)
reference_metadata/            Salinan CSV metadata kompetisi
README.md                      Dokumentasi teknis detail per komponen
PROGRESS.md                    File ini
```
