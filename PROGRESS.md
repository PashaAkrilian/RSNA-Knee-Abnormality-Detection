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
