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

## 40. Riset Discussion segar (26 Agustus, arahan "scraping seluruh strategi, beri strateginya")

Karena Code-tab (skor tertinggi) sudah dibedah habis 2 hari terakhir dan makin obfuscated/susah
digali lebih jauh, riset kali ini fokus ke Discussion terbaru (post 24-26 Agustus, belum pernah
dibaca) + 1 pengujian lokal langsung dari klaim yg ditemukan. Detail penuh: `docs/RESEARCH_NOTES.md`
§27, ringkasan di `MEMORY.md` §2.

Ringkasan cepat 4 temuan: (1) Synovitis lemah krn cuma 16% laporan menyebutnya eksplisit + RadImageNet
pretraining terbukti membantu label ini scr independen; (2) hipotesis GAP/mean-pooling melunakkan
sinyal temuan kecil (relevan langsung ke `SlotHead` kita); (3) peringatan checkpoint-selection di
gold-58 bisa menyesatkan — **langsung terverifikasi via pengalaman kita sendiri** (c5-solo menang
lokal 0,9206 tapi kalah real LB 0,927 vs ensemble tim 0,936, persis pola yg diperingatkan); (4)
duplicate-report leakage (194 studi/47 grup) diverifikasi via `reference_metadata/train.csv` lokal.

Diuji lokal (`label_engineering/blend_labels.py`, tanpa GPU, <1 menit): blend tabel label
(flight0234+stevenleehans+pilkwang) — **hasil negatif**, tidak ada kombinasi yg mengalahkan
flight0234 sendirian (0,8991 vs blend terbaik 0,8978). Line ditutup.

**Strategi disintesis & disampaikan ke user** (6 poin, ranked): (1) bangun arm RadImageNet sendiri
(target Synovitis); (2) ganti GAP jadi spatial attention pooling di level patch (kandidat arsitektur
baru genuin, target Lateral Meniscus/PF OA/MCL); (3) perbaiki k-fold split soal duplicate-report
(utang teknis kecil); (4) disiplin proses — submission real jadi pemutus keputusan besar, bukan
gold-58 saja; (5) blend label ditutup; (6) LLM API relabeling tetap prioritas tertinggi jangka
panjang (SEMUA sumber independen sepakat label ~0,89-0,90 adalah plafon single-model).

## 41. v23 (spatial attention pooling) SELESAI — sendirian tidak menang, tapi mengungkap pola komplementer bernilai: blend v18+v23 = OOF terbaik baru keluarga SlotHead

Semua 5 fold selesai stabil (rentang 0,7891-0,8717, spread mirip noise floor v18 — TIDAK ada
instabilitas spt v21/v22). **OOF gabungan 57/57: 0,7967** — sedikit di bawah v18 (0,8068, -0,0101),
dalam noise floor yg sudah terkuantifikasi (std antar-fold ~0,03).

Sesuai kriteria keputusan yg sudah ditulis di notebook SEBELUM hasil keluar ("kalau v23 ~= v18 atau
lebih rendah, hipotesis GAP-sebagai-pembatas tidak terbukti"), macro AUC sendirian TIDAK
mengonfirmasi hipotesis. TAPI dibongkar per-label (`blend_results/blend_v18_v23.py`), ceritanya
jauh lebih menarik:

| Label | v18 (GAP) | v23 (SpatialAttnPool) | delta |
|---|---:|---:|---:|
| **Medial Meniscus** | 0,773 | 0,876 | **+0,103** |
| **Lateral Meniscus** | 0,638 | 0,702 | **+0,064** |
| Effusion | 0,869 | 0,920 | +0,051 |
| Fracture | 0,822 | 0,852 | +0,030 |
| PF OA | 0,790 | 0,795 | +0,005 |
| ACL | 0,836 | 0,837 | +0,001 |
| Lateral OA | 0,789 | 0,767 | -0,022 |
| Synovitis | 0,772 | 0,722 | -0,049 |
| Baker's | 0,804 | 0,754 | -0,050 |
| Medial OA | 0,919 | 0,868 | -0,051 |
| Contusion | 0,870 | 0,774 | -0,096 |
| MCL | 0,801 | 0,694 | -0,107 |

**Spatial attention pooling menang TELAK di kedua Meniscus** — PERSIS hipotesis yg diuji
(`docs/RESEARCH_NOTES.md` §27b): robekan meniscus adalah fitur kecil-diskret, GAP melunakkannya,
attention spasial menangkapnya. TAPI kalah di temuan yg sifatnya DIFUS/tersebar (MCL sprain,
edema kontusio, penebalan sinovial, OA) — attention yg fokus sempit kehilangan sinyal yg
tersebar luas di area yg lebih besar, GAP justru lebih cocok utk pola ini.

**Blend rank-mean v18+v23** (tanpa training tambahan, murni pandas lokal): **0,8176 @ α(v18)≈0,5-
0,65 — OOF TERBAIK BARU keluarga SlotHead**, melampaui v18 (0,8068) +0,0108. Robust di rentang
lebar (α 0,30-0,85 semua >0,81) — diverifikasi BUKAN artefak titik tunggal, sweep penuh tersimpan
di `blend_results/blend_v18_v23_output.log`.

**Kesimpulan**: v23 BUKAN eksperimen gagal spt v21/v22 — nilainya bukan sbg pengganti v18,
tapi sbg ANGGOTA ENSEMBLE yang genuinely diverse (spesialisasi berlawanan per-label). 10 checkpoint
tersedia (5 fold v18 + 5 fold v23). **Belum diuji di submission real** — perlu kernel inferensi baru
yg load 10 checkpoint & blend, atau setidaknya validasi lanjut sebelum pakai kuota submission.

**Ide v24 (belum dibangun)**: daripada 2 model terpisah di-blend, gabung GAP+SpatialAttnPool sbg
2 fitur PARALEL dlm SATU forward pass (mis. concat [CLS, GAP-mean, spatial-attn-pool] sebelum
`SlotHead`) — biarkan attention per-label `SlotHead` yg sudah ada belajar sendiri kombinasi optimal
per temuan, tanpa perlu 2x biaya training terpisah.

## 42. v24 (dual-pool dalam 1 model) SELESAI — negatif, line ditutup, kesimpulan diperjelas

Semua 5 fold selesai stabil (0,759-0,856, tidak ada instabilitas). **OOF: 0,7968** — nyaris identik
dgn v23 sendirian (0,7967, delta +0,0001, murni noise), TIDAK mendekati blend manual v18+v23
(0,8176).

Kriteria keputusan sudah ditulis di notebook v24 SEBELUM hasil keluar (spy tidak bias menilai
belakangan): "v24≤0,8068 → SlotHead gagal manfaatkan fitur tambahan, kembali ke blend manual
v18+v23 sbg hasil terbaik." **Hasil jatuh persis di kategori ini.**

**Pelajaran metodologis yg diperoleh**: memberi SATU model fitur input lebih kaya (GAP+attention
digabung jadi satu vektor) TIDAK bisa mereplikasi manfaat MELATIH DUA MODEL independen lalu blend
prediksinya. Sumber "diversity" yg bikin blend v18+v23 menang (0,8176) ternyata BUKAN dari
arsitektur gabungan itu sendiri, tapi dari 2× training run terpisah (random init beda, urutan data
beda, trajectory optimisasi beda) — dua hal yang genuinely berbeda, bukan bisa disubstitusi.

**Keputusan final**: line "dual-pool dalam 1 model" DITUTUP, tidak dilanjutkan lagi. **Hasil terbaik
keluarga SlotHead proyek TETAP blend manual v18+v23 (0,8176)** — checkpoint v24 TIDAK dipakai.

**Prioritas berikutnya** (2 opsi, urutan disarankan): (1) **validasi blend v18+v23 di leaderboard
real** — kita punya 10 checkpoint (5 fold v18 + 5 fold v23), tinggal bangun/update kernel inferensi
utk ensemble & rank-mean blend, lalu submit. Ini menguji langsung apakah keunggulan 0,8176 lokal
benar2 bertahan di skala nyata, mengingat pelajaran c5-solo (§39 PROGRESS, §27c RESEARCH_NOTES)
bahwa perbandingan lokal N=57-58 bisa menyesatkan. (2) arm RadImageNet sendiri (§27a, target
Synovitis) — investasi lebih besar (backbone baru), ditunda sampai (1) selesai divalidasi.

## 43. Submission v3: ensemble v18+v23 (10 checkpoint, rank-mean) — dev-test bersih, disubmit real

Sesuai prioritas yg dicatat setelah v24 gagal (§42): validasi blend v18+v23 (OOF lokal terbaik
0,8176) di leaderboard real, bukan cuma percaya angka lokal N=57 begitu saja (pelajaran c5-solo,
§39: lokal bisa menyesatkan).

**Perubahan di `kaggle_kernel_submission_v2/`** (slug lama dipakai ulang, isi baru sepenuhnya):
- Tambah `KneeModelV23` (`SpatialAttnPool`) berdampingan `KneeModelV18` (GAP) — dua kelas
  arsitektur beda, checkpoint HARUS dipisah per asal `kernel_sources` (nama file sama persis
  `best_model_fold{0-4}.pt` di kedua kernel, tapi struktur `state_dict` beda — glob nama file
  polos spt v2 lama TIDAK cukup, harus filter by path).
- **Fix pending lama sekalian dibereskan**: ensembling ganti dari `np.mean(preds)` (probability-
  mean) jadi rank-mean equal-weight lintas 10 model — sesuai temuan riset sadamtorres (§23a:
  "averaging sigmoids lets the most confident member dominate; AUC reads order"). Restrukturisasi
  perlu krn rank butuh distribusi lintas SEMUA studi test dulu (bukan bisa dihitung per-studi
  spt probability-mean) — `infer_study()` sekarang return prediksi MENTAH per-model (bukan sudah
  dirata-ratakan), rank+gabung baru terjadi setelah loop test PENUH selesai.
- `kernel-metadata.json`: `enable_gpu=true, machine_shape=NvidiaTeslaT4, enable_internet=false`
  (config submission-siap, bukan CPU dev-test lagi), tambah `kernel_sources` v23.

**Dev-test (kernel version 2, stub 3-studi) SUKSES TOTAL, tanpa perlu perbaikan apa pun**:
`device: cuda` (GPU jalan), semua paket (pydicom/transformers/sklearn) preinstalled walau
`enable_internet=false` (tidak perlu fallback wheel), 5 checkpoint v18 + 5 checkpoint v23 SEMUA
ketemu & termuat benar ke kelas yg tepat (tidak ada `state_dict` mismatch), 3/3 studi berhasil
diinferensi, rank-mean blend terhitung benar (`all_preds shape: (3, 10, 12)`). **Total ~74 detik**
(~57s setup+load 10 model, ~17s murni inferensi 3 studi = ~5,7 detik/studi) → ekstrapolasi test
set asli (~1.300 studi) **~2 jam, jauh di bawah batas 9 jam kompetisi** — JAUH lebih cepat dari
estimasi CPU lama (~21 jam utk 5 model saja, §29).

**Disubmit real** (submission ID `55820944`, 27 Agustus): "Ensemble v18(GAP)+v23(SpatialAttnPool)
10-checkpoint rank-mean, OOF lokal terbaik 0,8176 - validasi vs ensemble tim 0,936". Status PENDING,
menunggu skor publik. Kuota submission tim tersisa 4 hari ini.

## 44. HASIL submission v3: 0,837 real LB — line DINOv2+SlotHead (v11-v24) DITUTUP RESMI, kesimpulan strategis besar

Submission 55820944 selesai: **0,837 public LB**. Offset dari OOF lokal (0,8176) = **+0,019**,
arahnya WAJAR (positif, model kuat cenderung LB > lokal, sesuai pola umum) — TAPI angka absolutnya
jauh di bawah c5-solo (0,927, §39) dan ensemble tim (0,936).

**Rekap 2 tes real-LB minggu ini, keduanya kalah dari submission tim eksisting**:

| Sumber | OOF lokal | Real LB |
|---|---:|---:|
| c5 CoAtNet solo | 0,9206-0,9213 | 0,927 |
| Blend v18+v23 (DINOv2 SlotHead) | 0,8176 | **0,837** |
| Ensemble tim (fork community, `rsna-base`) | — | **0,936** (masih terbaik) |

**Kesimpulan strategis besar**: arsitektur rumahan kita (DINOv2-base + slot-bag + `SlotHead`) —
SEMUA 14 varian yg pernah dicoba dari v11 sampai v24 (breakthrough unfreeze dalam, geometri slice
v2, sampling "3+3+3" v16/v17, k-fold v18, laterality canon v19/v20, anatomy-query v21, widecrop+
triplet v22, spatial-attn-pool v23, dual-pool v24) — **plafon nyatanya di real leaderboard cuma
~0,84**, gap ~0,09-0,10 dari CoAtNet arm SOLO (0,927), apalagi dari ensemble tim (0,936). Ini BUKAN
lagi soal butuh 1-2 eksperimen lagi utk nutup gap — 14 percobaan sistematis (mencakup hampir semua
sumbu variasi yg masuk akal: freeze-schedule, geometri, sampling, laterality, pooling) sudah cukup
utk simpulkan DINOv2-base+SlotHead SECARA FUNDAMENTAL bukan arsitektur yg kompetitif utk kompetisi
ini, terlepas dari tuning lebih lanjut.

**Keputusan**: line DINOv2+SlotHead **DITUTUP RESMI**, tidak ada v25 dst di jalur ini. Checkpoint
v18/v20/v23 disimpan sbg arsip riset (bukti kerja & pembelajaran metodologis), TIDAK dipakai lanjut
sbg model produksi.

**Pelajaran metodologis diperbarui** (setelah 2 kasus real-LB minggu ini): OOF gold-58 CUKUP
RELIABLE utk ranking relatif ANTAR-VARIAN dalam SATU keluarga arsitektur (semua eksperimen v18-v24
konsisten arah lokal↔real) — TAPI offset absolut ke real-LB BISA JAUH BEDA antar-ARSITEKTUR BEDA
TOTAL (CoAtNet offset +0,007 vs SlotHead offset +0,019, base rate keduanya beda jauh). Jangan
generalisasi angka OOF mentah lintas-arsitektur tanpa validasi submission real.

**Prioritas final yg tersisa** (per `MEMORY.md` §5, urutan): (1) **LLM API key relabeling** —
satu-satunya lever besar yg BELUM PERNAH dicoba; dua jalur besar lain (arsitektur SlotHead DAN
fine-tune CoAtNet di label flight0234) SAMA-SAMA sudah dicoba & mentok, keduanya kembali ke
kesimpulan yg sama: kualitas label adalah pembatas nyata; (2) arm RadImageNet (§27a) sbg pelengkap
kalau (1) tidak memungkinkan; (3) submission tim TETAP 0,936 — TIDAK diganti apa pun yg kita bangun
sendiri sejauh ini.

## 45. Relabeling LLM tanpa API key terpisah — pilot + scale-up penuh 4.349 studi SELESAI, blend baru siap dipakai (28 Agustus – 2 September)

**Insight pemicu (user, 28 Agustus)**: "kenapa engga langsung relabeling menggunakan disini aja
engga usah menggukan API kan ini ngikutnya APINya Anthropic" — sesi Claude Code ini SUDAH jalan di
atas API Anthropic, jadi TIDAK perlu prosedur API key terpisah spt yg direncanakan sebelumnya
(§44 prioritas 1). Claude bisa baca teks `Report` (9 bahasa: Inggris ~37%, Spanyol, Yunani,
Bulgaria/Cyrillic, Jerman, Kroasia/Serbia, Turki, Belanda, Prancis) & hasilkan label 12-kolom
sendiri — SAMA PERSIS semangat ruling 733965 (LLM API utk proses Report text diizinkan Rules §2.6.b).

**Pilot 58 studi gold (blind read — jawaban TIDAK dilihat dulu, baru discore setelahnya)**: macro
AUC Claude sendirian = **0,8953**, kompetitif vs sumber LLM eksternal terbaik yg ada (flight0234
0,8991, stevenleehans v4 0,8927, pilkwang v2 0,8700). **Blend rank-mean flight0234+Claude =
0,9047 @ α(flight)=0,6 — MENGALAHKAN KEDUANYA sendirian**, robust di rentang α 0,2-0,9 (semua >0,90).
User memilih skala: "Skala penuh 4.349 studi (Rekomendasi)" atas 2 opsi yg ditawarkan.

**Eksekusi scale-up — 2 kali gagal total krn rate-limit + scratchpad wipe, direstart dgn sistem
checkpoint**: percobaan pertama (10 subagent paralel @~436 studi) SEMUA kena `429 rate_limit`
(monthly spend limit) di tengah jalan, lalu jeda ~7,5 jam menyebabkan scratchpad di-wipe TOTAL —
semua progres yg TIDAK sempat di-commit ke git HILANG (cuma pilot yg selamat, sudah di git).
Percobaan kedua (20 chunk lebih kecil @~218 studi, gelombang 4 agent) — gelombang 1 (chunk 00-03)
berhasil & langsung di-backup, tapi gelombang 2 (chunk 04-07) kena rate-limit lagi DAN scratchpad
di-wipe KEDUA KALINYA sebelum sempat diselamatkan. **User instruksikan eksplisit**: "coba tambahkan
sistem checkpoint agar tidak menjalankan ulang dari awal".

**Sistem checkpoint yang dibangun** (dan terbukti bekerja sampai selesai):
- Input 20 chunk `label_engineering/relabel_chunks_input/chunk_00..19.csv` di-commit ke repo git
  (bukan scratchpad) — jadi tidak pernah perlu diregenerasi ulang meski scratchpad wipe lagi.
- Tiap subagent diinstruksikan menulis output LANGSUNG ke path repo
  `label_engineering/relabel_out_backup/out_XX.csv` (bukan scratchpad), inkremental per sub-batch
  ~25-30 baris (append, bukan tulis-ulang) — jadi kalau agent mati di tengah (429, disconnect API),
  baris yg SUDAH ditulis tetap selamat di disk.
- `label_engineering/relabel_status.py` — skrip cek status: banding row-count `chunk_XX.csv` vs
  `out_XX.csv` per chunk, laporkan DONE/PARTIAL/MISSING — dijalankan tiap siklus utk tau persis
  chunk mana yg perlu di-resume.
- Begitu 1 agent selesai (bukan nunggu 1 gelombang penuh 3 agent), progresnya LANGSUNG di-commit+push
  ke git — pola ini yg menyelamatkan semua progres parsial lewat ~6 kejadian rate-limit berikutnya
  (reset time bervariasi tiap kejadian: 3:30, 8:30, 1:30, 2:20 WIB — bukan jendela kalender tetap).
- Kalau agent gagal di tengah, baris yg sudah tertulis di-diff terhadap UID input chunk utk hitung
  "remainder" persis (bukan reproses dari nol), lalu remainder itu dikirim ke agent kecil baru.
  Beberapa kali agent yg dilaporkan "failed" oleh notifikasi rate-limit ternyata SUDAH selesai
  menulis semua barisnya sebelum betul-betul mati — jadi tiap "failed" tetap dicek row-count aktual
  di disk dulu sebelum diasumsikan perlu di-resume.

**Temuan data quality**: 2 dari 4.351 baris chunk-input ternyata bukan 2 studi terpisah, melainkan
pecahan 1 laporan yg sama akibat bug quoting CSV di `reference_metadata/train.csv` ASLI (bukan bug
yg kita buat) — direkonstruksi manual via cross-reference ke `train.csv`, studi asli berhasil
dipulihkan & diberi label (bukan data yg hilang, cuma representasinya rusak di source file).

**Hasil akhir — 100% coverage, disimpan ke git**:
- `label_engineering/claude_labels_nongold_full.csv` — **4.349 studi unik berlabel**, semua nilai
  valid [0,1], tanpa null, coverage 100% terhadap 4.351 baris scope chunk asli (minus 2 baris korup
  yg sudah dipulihkan jadi 1 studi valid).
- `label_engineering/blend_fullscale.py` — re-validasi metodologi rank-mean di gold-58: **mereproduksi
  PERSIS hasil pilot (macro=0,9047 @ α(flight)=0,60)**, alpha-sweep penuh 0,00-1,00 dicoba, 0,60
  terbukti puncak global (bukan cuma lokal di sekitar 0,6 dari pilot).
  Korelasi raw-value Claude vs flight0234 (4.349 studi non-gold, dua sumber LLM independen):
  0,78-0,98 per label (Fracture terendah 0,785, Medial Meniscus tertinggi 0,975) — sinyal kuat
  keduanya menangkap pola nyata dari teks laporan, cukup independen utk saling melengkapi (bukan
  cuma noise yg kebetulan mirip).
- `label_engineering/blended_labels_fullscale_train.csv` — **4.349 baris, rank-mean α(flight)=0,60,
  siap dipakai sbg label training CoAtNet fine-tune berikutnya.**

**Belum dikerjakan (next step)**: jalankan `build_train_coatnet.py` (ganti sumber label dari
flight0234-only ke `blended_labels_fullscale_train.csv`) di Kaggle, validasi gold-58 (target: kalahkan
baseline c5 0,9206 DAN fine-tune flight0234-only sebelumnya yg gagal di 0,9193 — §38), lalu KALAU
menang, submit real ke leaderboard utk konfirmasi (jangan ulang kesalahan metodologis §39: keputusan
besar wajib dikonfirmasi submission real, gold-58 nested-CV terbukti bisa menyesatkan lintas-arsitektur).

## 46. v2 CoAtNet fine-tune (label blended rank-mean) SELESAI — HASIL NEGATIF, root cause ditemukan: rank-mean merusak kalibrasi target BCE

**Hasil**: `kaggle_kernel_train_coatnet_v2_blended` selesai training (4 epoch, loss turun monoton
0,5635→0,5582→0,5485→0,5431, tidak ada tanda bug). **MACRO AUC gold-58 = 0,9170** — LEBIH RENDAH
dari KEDUA pembanding: c5 baseline (0,9206) DAN v1 fine-tune flight0234-only (0,9193). Regresi
nyata, bukan noise (gap 0,0023-0,0036 dari kedua baseline).

**Ini kontra-intuitif**: label blend rank-mean SKOR LEBIH TINGGI sbg predictor vs gold-58 (0,9047
vs flight0234 sendirian 0,8991, §45) — tapi model yg DILATIH pakai label itu skor LEBIH RENDAH.
Dua metrik yg beda: (1) AUC label-vs-gold (threshold-free, cuma peduli urutan ranking) vs
(2) AUC model-terlatih-vs-gold (peduli MAGNITUDE target krn loss BCE literal minimize
`-y*log(p)-(1-y)*log(1-p)` dgn `y` = label kontinu qt).

**Root cause (dianalisis lokal, tanpa GPU)**: rank-mean (`pd.Series.rank(pct=True)`) MEMAKSA tiap
kolom label jadi distribusi **uniform** (by construction — itu definisi percentile rank), MENGHAPUS
skew asli tiap label (mis. Fracture yg secara alami base-rate rendah, sebagian besar studi memang
prob rendah) — jadi target BCE effectively kehilangan sinyal kalibrasi/prevalensi yg model perlu
pelajari, walau URUTAN relatif antar studi (yg dipakai AUC) tetap benar. **Rank-mean bagus utk
ensembling PREDIKSI MODEL (dipakai sukses di `kaggle_kernel_submission_v2` §32-33) krn di situ yg
dibutuhkan cuma urutan relatif utk threshold-free eval — TAPI SALAH utk label TARGET TRAINING**,
yg butuh magnitude terkalibrasi, bukan cuma urutan.

**Fix yg diuji (lokal, cepat)**: ganti rank-mean → **raw-probability-mean** (weighted average nilai
probabilitas mentah, TANPA transformasi rank) — preserve skew/kalibrasi asli. Alpha-sweep gold-58:
puncak **α(flight)=0,70, macro=0,9033** (sedikit di bawah rank-mean 0,9047 sbg label-matching
metric, TAPI ini WAJAR krn raw-mean tidak "curang" pakai urutan optimal spt rank — yg penting
apakah SKOR MODEL TERLATIH lebih baik, bukan skor label-matching). Std per label jauh lebih dekat
ke distribusi asli flight0234 (mis. Medial Meniscus: rank-blend std=0,285 vs raw-blend std=0,399
vs flight_raw std=0,427 — raw-blend jauh lebih mirip bentuk distribusi asli, bukan uniform paksa).

**Keputusan**: bangun v3 dgn label raw-probability-mean (α=0,70) sbg target training, retest di
Kaggle. Kalau v3 MENANG vs v1 (0,9193) DAN c5 (0,9206) → konfirmasi hipotesis rank-mean adalah
akar masalah, blend jadi berguna. Kalau v3 TETAP kalah → kemungkinan blending label report dgn
sumber independen TIDAK membantu fine-tuning CoAtNet sama sekali (label flight0234 murni tetap
superior sbg training signal walau kalah sbg label-matching metric thd blend) — line label-blending
utk TRAINING (bukan cuma evaluasi) perlu ditutup, balik ke flight0234 murni sbg training source.

## 47. v3 CoAtNet fine-tune (raw-probability-mean blend) DITEMUKAN SELESAI di Kaggle (13 September) — HASIL POSITIF, MENANG vs c5 baseline, konfirmasi hipotesis §46

**Konteks penemuan**: sesi ini dimulai dari permintaan user "pelajari project ini dan progres apa yang
sudah dicapai", lalu pertanyaan susulan soal ketersediaan `kaggle.json` utk push remote dari terminal
lokal. Kredensial sudah ada (`~/.kaggle/kaggle.json`, akun dimaspashaakrilian) tapi CLI `kaggle` belum
terpasang di sandbox container ini (fresh container, `pip`/`ensurepip` pun tidak ada) — dipasang via
`sudo apt-get install python3-pip pipx` + `pipx install kaggle` (hindari `pip install --user` krn
PEP 668 externally-managed-environment). Setelah CLI aktif & auth terverifikasi (akun cocok, kernel
riwayat muncul), dicek status semua kernel CoAtNet training — ternyata
**`dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean` (dibangun 2 September, commit `ea36e7d`,
saat itu cuma tercatat "built") sudah berstatus `KernelWorkerStatus.COMPLETE`** — belum pernah ditarik
hasilnya krn CLI lokal belum pernah aktif sejak kernel itu dipush. Ini murni gap operasional (kernel
selesai jalan sendiri di Kaggle, tidak ada yg mengecek), bukan kerja baru yg sengaja dijadwalkan.

**Hasil (ditarik via `kaggle kernels output ... -p .`, log `.log` + `train_coatnet_v3_rawmean_gold_preds.csv`,
diverifikasi ulang independen via `manual_auc` lokal — angka log & hitung ulang match persis)**:

```
MACRO AUC gold-58 (final, full-ft, label blended RAW-MEAN flight+claude): 0.9234
pembanding: c5=0.9206, v1 flight0234-only=0.9193, v2 rank-mean blend=0.9170 (REGRESI)
```

Training 4 epoch, loss monoton turun (0,5535→0,5415→0,5260→0,5144), tidak ada tanda instabilitas,
runtime ~4,6 jam (16.581 detik) GPU T4. Per-label AUC (58 gold, urut lemah→kuat):

| Label | AUC | Label | AUC |
|---|---|---|---|
| Lateral OA | 0,8124 | Contusion | 0,9406 |
| Synovitis | 0,8399 | Medial Meniscus | 0,9675 |
| Lateral Meniscus | 0,8472 | Effusion | 0,9689 |
| PF OA | 0,8571 | Baker's | 0,9710 |
| Fracture | 0,9319 | Medial OA | 0,9798 |
| | | MCL | 0,9819 |
| | | ACL | 0,9828 |

Titik lemah (Lateral OA, Synovitis, Lateral Meniscus, PF OA) konsisten dgn pola lama yg berulang kali
tercatat di proyek ini (§25/§27a: Synovitis sulit krn cuma ~16% laporan sebut eksplisit; Lateral
Meniscus titik lemah historis sejak v15-v24).

**Signifikansi**: ini adalah **MACRO AUC gold-58 tertinggi proyek utk model tunggal** (mengalahkan c5
off-the-shelf 0,9206, dan seluruh keluarga SlotHead v11-v24 yg plafonnya ~0,80-0,83 OOF). Sekaligus
**mengkonfirmasi diagnosis §46**: masalah v2 memang rank-mean merusak kalibrasi BCE target (bukan
blending label report itu sendiri yg salah) — raw-probability-mean menyelesaikannya. Blending 2 sumber
LLM independen (flight0234 + Claude) sbg training signal TERBUKTI bermanfaat, asal transformasinya
tepat.

**BELUM dilakukan (keputusan menunggu user, alasan metodologis eksplisit)**: submit checkpoint v3 ke
leaderboard real. Proyek ini SUDAH 2 KALI kena pelajaran pahit bahwa kemenangan di gold-58 lokal tidak
otomatis bertahan di real LB (c5-solo §39: menang nested-CV lokal 0,9206 tapi cuma 0,927 di LB, kalah
dari ensemble tim 0,936; v18+v23 blend §44: 0,8176 lokal jadi cuma 0,837 LB). Selisih v3 vs c5 (+0,0028)
JAUH lebih kecil dari margin yg dulu terbukti tidak reliable lintas lokal→LB. **Rekomendasi eksplisit**:
JANGAN treat 0,9234 sbg kemenangan final sebelum submission real — butuh kernel inferensi/submission baru
(checkpoint `coatnet_rawmean_ft_v3.pt`, 292MB, masih di Kaggle output belum ditarik ke dataset permanen)
+ koordinasi kuota submission tim (5x/hari, ada nxtirfan) sebelum dieksekusi.

## 48. Hybrid submission: v3 CoAtNet arm disisipkan ke pipeline publik terbaik (0,941) — DIBANGUN, DIUJI, DISUBMIT REAL (13 September, arahan eksplisit user "gabungkan dengan hasil yang sudah ada dan submit")

Eksekusi langsung dari strategi §28d (RESEARCH_NOTES): user secara eksplisit minta jangan cuma
riset, tapi benar-benar bangun & submit gabungan notebook publik terbaik + hasil kita sendiri
("kemarin itu mentok di 80 persen" — merujuk plafon arm SlotHead/CoAtNet solo kita sendiri).

**Pendekatan**: fork kernel kita sendiri yg sudah proven `bend-the-knee-to-the-dinosaurs-7d6471`
(real LB 0,941, arsitektur DINOv2 20-model + DINOv3 5-fold + RadImageNet + CoAtNet 4-arm+resgated
+ per-label outer-routing) — BUKAN membangun dari nol, krn arm kita sendiri sudah berkali-kali
terbukti tidak cukup kuat sendirian (SlotHead 0,837 LB; c5-solo 0,927 LB). Ditambah SATU cell baru
di ujung pipeline yg menyisipkan checkpoint v3 kita (fine-tune CoAtNet di atas label blend raw-mean
flight0234+Claude, gold-58=0,9234, §47) sbg arm tambahan, rank-blend α=0,15 (stack publik tetap
dominan 85%).

**Desain keselamatan (krn menempel di pipeline milik orang lain yg sensitif waktu, budget 9 jam,
margin riil tidak diketahui pasti)**:
1. Self-gating waktu — pakai global `T0` milik notebook itu sendiri, hitung sisa waktu ke batas
   9 jam, skip seluruhnya kalau sisa < 10 menit.
2. Deadline internal DI DALAM loop per-studi (patch `build_train_coatnet_ke_src.txt`) — kalau lambat,
   berhenti lebih awal, studi yg belum sempat diproses otomatis tetap di rank netral 0,5 (sudah jadi
   default array di kode asli) — degradasi lembut, bukan timeout keras.
3. SELURUH blok dibungkus try/except luas — kegagalan apa pun (checkpoint hilang, OOM, exception
   tak terduga) di-log lalu **submission.csv yg sudah proven (0,941) dibiarkan utuh**. Penyisipan
   ini SECARA DESAIN cuma bisa menolong atau diam, tidak bisa merusak punya mereka.
4. `K_EVAL` diturunkan 42→16 utk arm v3 ini spesifik (hemat waktu, wajar krn bobot blend kecil).

**Bug tertangkap & diperbaiki tepat sesuai desain**: push pertama (v1) ERROR internal
(`NameError: _v3_os tidak terdefinisi`, typo penamaan variabel) — TAPI try/except bekerja PERSIS
seperti dirancang: notebook keseluruhan tetap COMPLETE, submission.csv dasar (0,941) tidak
tersentuh, cuma pesan log "[v3-arm] FAILED, keeping existing submission.csv untouched". Diperbaiki,
push kedua (v2) bersih: log `[v3-arm] blended OUR CoAtNet ... into submission.csv at alpha=0.15`
muncul, dev-test (3-studi stub) SUKSES, format submission.csv benar (13 kolom, rank [0,1], no NaN).
Blend logic (align by StudyInstanceUID, rank-mean) diverifikasi terpisah dgn data dummy lokal
sebelum push kedua.

**SUDAH DISUBMIT REAL** (submission 56196772, 13 Sep 02:00 UTC) — kuota tim: 4/5 tersisa hari ini.
Deskripsi submission: "Bend the Knee to the Dinosaurs (0.941 base) + our v3 CoAtNet arm (raw-mean
flight0234+Claude relabel FT, gold-58=0.9234) blended at alpha=0.15, self-gated for safety".

**HASIL REAL (dicek ulang di sesi berikutnya, ~11 jam kemudian — monitor background sempat mati
krn sesi terputus, dicek manual): MACRO AUC public LB = 0,941 — SAMA PERSIS dgn base pipeline TANPA
v3 (byte-identik 3 desimal).** Tim tetap di posisi #223/3.653 (leaderboard terus tumbuh; skor #1
tidak berubah, 0,956).

**Diagnosis — TIDAK BISA dipastikan penyebabnya, 2 hipotesis bersaing**: dicoba 3 cara berbeda
utk menarik log run submission REAL (`kaggle kernels output` slug polos, `/2`, `/3` eksplisit) —
**SEMUANYA cuma mengembalikan log dev-test stub 3-studi yg SAMA (durasi 0,058 jam identik)**, BUKAN
log run sungguhan (1.322 studi test asli, harusnya berjam-jam). **Kesimpulan operasional penting**:
Kaggle code-competition SENGAJA tidak mengekspos log/output dari run submission sungguhan via API
(mencegah kebocoran info test set) — `kaggle kernels output` HANYA PERNAH mengembalikan hasil "Save
& Run All" biasa (stub publik), TIDAK PERNAH hasil submission run, walau kernel version-nya sama.
**Ini keterbatasan permanen, dicatat sbg pengetahuan operasional baru** utk semua eksperimen
submission-langsung ke depan — kita TIDAK PERNAH bisa lihat log run sungguhan, cuma skor akhirnya.

Dua hipotesis yg TIDAK bisa dibedakan tanpa info tambahan:
1. **Arm v3 di-skip total oleh self-gating waktu** — base pipeline (20+5+RadImageNet+CoAtNet
   4-arm+resgated) kemungkinan sudah pakai sebagian besar budget 9 jam (versi lama pipeline ini,
   RESEARCH_NOTES §26c, punya "adaptive degradation ladder" bertarget 8-8,72 jam/9 jam) — kalau sisa
   < 10 menit saat giliran v3, kodenya skip total by design, hasil = 100% identik dgn base (SESUAI
   temuan 0,941=0,941 persis).
2. **Arm v3 jalan tapi kontribusinya washed out** — bobot α=0,15 di atas ensemble yg SUDAH berisi
   4+ varian CoAtNet lain mungkin memang terlalu kecil utk menggeser rank-blend melewati ambang
   pembulatan 3 desimal — selaras temuan lama proyek sendiri (§16/§26d: "LB desimal ke-3 sering
   tidak resolvable").

**Keputusan**: JANGAN buru-buru simpulkan "v3 tidak berguna" — data yg ada TIDAK CUKUP utk
membedakan 2 hipotesis di atas. Opsi diagnostik: (a) submit v3 SENDIRIAN (langkah asli §28d yg
dilewati) utk dapat titik data independen ttg apakah v3 punya lift nyata di real LB, terlepas dari
masalah integrasi; (b) naikkan α jauh lebih tinggi utk tes berikutnya (kalau hipotesis 2 benar,
sinyal harusnya lebih kelihatan); (c) submit build APA ADANYA (tanpa v3, kernel dinosaurs polos)
sbg kontrol utk pastikan base pipeline itu sendiri REPRODUCIBLE persis 0,941 (bukan varian by
chance) — belum dilakukan, kuota tersisa hari ini perlu dicek ulang sebelum eksperimen lanjutan.

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

## 49. v3 SOLO = 0,932 di LB nyata vs c5 solo 0,927 — LABEL ENGINEERING TERBUKTI BEKERJA (13 September sore)

Eksperimen yang diputuskan mandiri (arahan user "riset sangat mendalam dan putuskan sendiri"),
dipilih justru KARENA murah tapi informasinya menentukan — detail alasan di RESEARCH_NOTES §29d.

**Hasil: `rsna-knee-submit-v3-solo` (submission 56207189) = 0,932 public LB.**
Pembanding langsung: `c5 solo` (submission 55795592, 26 Agustus) = **0,927**. Harness `_KE_SRC`
SAMA PERSIS, `K_EVAL=42` sama, 1 arm tanpa blend, sama-sama tanpa TTA tambahan — **satu-satunya
variabel yang berbeda adalah bobot checkpoint**. Jadi:

> **Fine-tune di atas label blend raw-mean (flight0234 + relabeling Claude 4.349 studi) memberi
> +0,005 macro AUC di leaderboard NYATA dibanding checkpoint stock yang dipakai seluruh komunitas.**

Ini menjawab pertanyaan yang menggantung sejak §47: keunggulan +0,0028 di 58 studi gold BUKAN noise
— transfernya ke data nyata bahkan lebih besar (+0,005). Line label engineering (§45-§47) TERBUKTI
memberi nilai nyata, bukan cuma lokal.

**Catatan kuota (koreksi estimasi §29a)**: run solo ternyata cuma memakan **1,24 GPU-jam**
(17,27h → 18,51h), jauh lebih murah dari perkiraan 4,6 jam. Submission 1-arm = eksperimen murah;
pipeline penuh = ~9,7 GPU-jam. Sisa kuota setelah ini: 11,49h (refresh 19 Sep).

**Tindak lanjut langsung (berjalan)**: `rsna-knee-submit-v3-inarm` — integrasi v3 SECARA BENAR ke
pipeline publik 0,941, yaitu sebagai ANGGOTA di dalam daftar `ARMS` arm CoAtNet (bukan ditempel di
luar hasil akhir seperti percobaan pertama yang hasilnya null). Bobot baru: **v3 0,35 (terbesar)**,
c5 0,30 (tetap dipertahankan demi diversity), v10 0,08, v5-reverse 0,07, v8 0,20. Dasar bobot:
(a) v3 terbukti > c5 di LB nyata (+0,005), (b) uji blend lokal §29b menunjukkan konfigurasi terbaik
adalah mempertahankan c5 sambil memberi v3 bobot >= c5. Tidak ada logika lain yang disentuh.

## 50. v3 IN-ARM: integrasi benar ke pipeline publik — dev-test bersih, disubmit real (13 September malam)

Perbaikan struktural atas percobaan hybrid yang hasilnya null (§48). Bedanya mendasar:

| | §48 (hybrid, NULL) | §50 (in-arm, ini) |
|---|---|---|
| Titik sisip | rank-blend di ATAS submission.csv final | **anggota di dalam daftar `ARMS` arm CoAtNet** |
| Bobot v3 | α=0,15 atas seluruh ensemble | **w=0,35 (terbesar)**, c5 turun 0,60→0,30 |
| Masalahnya | v3 "bersaing dgn dirinya sendiri" (c5 induknya sudah di dalam), efek encer | v3 menggantikan sebagian peran c5 di slot dominan, lalu ikut diperkuat per-label routing mereka |
| Dasar bukti | belum ada bukti LB | **v3 solo 0,932 > c5 solo 0,927 (§49)** + uji blend lokal §29b |

**Gotcha yang ditemukan (penting untuk dicatat)**: push pertama ERROR —
`RuntimeError: unpinned Raptor checkpoint: coatnet_rawmean_ft_v3.pt`. Pipeline publik itu
**mem-PIN checkpoint yang boleh dimuat** (`find_weight_file`): whitelist nama file + path eksplisit
+ **verifikasi ukuran file persis**. Checkpoint asing ditolak mentah-mentah. Diperbaiki dgn menambah
cabang khusus untuk checkpoint kita yang TETAP memverifikasi (harus match tepat 1 file + ukuran
292.874.395 byte), tanpa melemahkan pin milik mereka. **Dev-test murah (~2 menit) menangkap ini —
kalau langsung submit, 9,7 GPU-jam + 1 slot submission hangus.**

**Dev-test v2 BERSIH**: log mengonfirmasi `[arm 0] ours-v3-rawmean` termuat & jalan, kelima arm
berurutan, dan baris blend menunjukkan bobot persis rancangan
(`ours-v3 0.35, maxspan-v5 0.30, v10 0.08, v5-reverse 0.07, v8 0.20`), lalu `_coat_substitute`
(resgated blend α=0,40) dan per-label outer routing mereka tetap berjalan normal.

**DISUBMIT REAL** (13 Sep malam). Kuota GPU setelah submit: 10,90h tersisa; kuota submission tim:
2/5 tersisa hari ini. Hasil menyusul.

**Ekspektasi jujur**: kalau perbaikan arm CoAtNet (+0,005 terukur di level solo) diteruskan oleh
bobot per-label mereka (CoAtNet dapat 0,60-1,00 tergantung label), perkiraan kasar +0,003-0,005 →
**0,944-0,946**. Tapi ini tetap ekstrapolasi; efek ensemble hampir selalu teredam, dan §48 sudah
membuktikan intuisi bisa meleset. Angka realnya yang memutuskan.


---

## 51. Siklus probe r224 selesai (0,950 = buang), jalur heliosli blokir, fork rabari disubmit, akun ketiga (9–10 Okt malam)
- **57012143** (probe r224, push dari akun2) = **0,950** → BUANG; varian r224/tweak bobot ditutup (noise).
- Riset 4 subagent + sintesis → `reports/Strategi meraih perak RSNA.md` + `research_notes/Strategi meraih perak RSNA/` + `laporan_push.md` (wadah evaluasi silver §1–§6); ter-commit & ter-push.
- **Arm heliosli blend-gold BLOKIR**: `heliosli/rsna-knee-compact-v1` RETIRED 9 Okt 13:50 UTC + `ofl-ssl-weights` 403 → fork `evgendvorkin/rsna-versia-7` & `samanyu1808/blend3` ikut mati. fold0 tetap publik tapi manifest/bundle S5 hilang.
- Scan 9 Okt temukan `rabari9999/...dual-engine-stack-lb-0-951` (klaim 0,951+, input publik: raptor r384+MIRROR_TTA ⊕ reader ⊕ r224@0,3; commit-run penulis OK) → fork ke akun2 `alverlfarfaraway/rsna-knee-dual-engine-tta-fork` v1 (23:33) → commit-run COMPLETE (23:35, output 12 kolom) → **submisi 57015077 (23:37) PENDING**; sisa jatah 2.
- Monitor `progress_submisi.sh` (ID via argumen, jatah fix 3/5) berjalan; 6 commit ter-push (…`7fa2137`).
- **10 Okt 00:42 — akun KETIGA `nxtirfan`** (perintah user): kredensial di `akun_kaggle/` (3 file, chmod 600, `.gitignore` folder utuh), wrapper `kg3.sh`, `kg_quota.sh` 3 akun (GPU akun3 27,68h). Urutan komputasi: utama → akun2 → **akun3** → Colab. MD diperbarui: MEMORY, HANDOFF, push, laporan_push, PAKET_REKAN.

## Arsip MEMORY.md (versi penuh sebelum dipadatkan, 8 Okt 2026 21:30 WIB)
> Disalin apa adanya dari MEMORY.md agar tidak ada informasi hilang. MEMORY.md aktif kini padat (lihat HANDOFF.md untuk langkah operasional).

# MEMORY — RSNA Knee Abnormality Detection
> Living memory. Update bagian relevan setiap ada temuan/keputusan baru.
> Jangan tambah terus ke bawah — overwrite bagian yang sudah usang.
> Last updated: 2026-10-05T15:30:00+07:00

## 1. Fakta Kompetisi (jarang berubah)
- Kompetisi: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — code competition, 1.611 tim.
- **Metrik: macro (mean) ROC-AUC** atas 12 kolom label, $\frac{1}{12}\sum AUC_i$ — dikonfirmasi resmi via `kaggle competitions pages` (Evaluation).
- **Deadline: Final Submission 22 Oktober 2026, 23:59 UTC.** Entry/Team Merger deadline 15 Oktober 2026. Winner wajib serahkan training+inference code & model weights (Kaggle Dataset publik).
- Data: 4.407 studi train, 24.371 series, 569.76GB (mustahil diproses lokal — semua eksekusi WAJIB di Kaggle). Hanya **58 studi (1.3%) gold-label penuh**, sisanya 98.7% kosong di 12 kolom label tapi 100% punya teks `Report`. `test.csv` publik cuma 3 studi (test asli tersembunyi).
- **Submission**: WAJIB via Notebook, output `submission.csv`, **internet HARUS mati** saat run submission, **≤9 jam runtime** per run (termasuk decode ~1.300 studi test + inferensi ensemble). Kolom `Report` TIDAK tersedia saat testing (cuma valid sbg target training, bukan input inferensi).
- **External data & pretrained model DIIZINKAN eksplisit** (Code Requirements: "Freely & publicly available external data is allowed, including pre-trained models"). Sharing kode publik juga diizinkan (Rules §3.5.d).
- **Data Security (Rules §2.4.b)**: dilarang transmit Competition Data (termasuk teks `Report`) ke pihak yang TIDAK ikut kompetisi. **API LLM hosted (OpenAI/Anthropic/dll) DIIZINKAN** untuk memproses `Report` (ruling resmi host, topic 733965) SELAMA layanannya "reasonably accessible & minimal cost" ke semua peserta (Rules §2.6.b) — bukan larangan mutlak.
- Ada **Efficiency Prize** terpisah (leaderboard beda urutan) — di luar fokus kita, target tetap skor mentah.
- **Leaderboard (snapshot 3 Okt 2026, 5.024 tim)**: #1 = **0,961**; #10 = 0,958; #20 = 0,957; #50 = 0,954; #251 (≈ top 5%) = 0,945; #502 (≈ top 10%) = 0,943. **Tim KITA = 0,943, rank #561** (sebelumnya #808/0,941 pada 26 Sep; rekan tim aktif submit fork `d4-blend`/`d4-lite`, plafon notebook publik 0,943). Ambang medali memakai aturan Kaggle standar (perunggu top 10%, perak top 5%, emas top 10 + 0,2% tim) — BELUM terverifikasi berlaku di kompetisi ini. Papan terus bergerak (fork publik); deadline 22 Okt. **Ekspektasi realistis dari kerja kita ≈ 0,945–0,947 (perak tipis)**; #50 (0,954) tidak realistis, #1 (0,961) tidak kredibel. Detail: RESEARCH_NOTES §28–§35.
- Compute Kaggle: GPU T4x2 atau P100 tersedia; **P100 tidak kompatibel** dengan build PyTorch Kaggle saat ini (compute capability terlalu lama) — paksa `machine_shape: "NvidiaTeslaT4"` di kernel-metadata.json.

## 2. Keputusan Arsitektur Terkini (overwrite kalau berubah)
- **ARAH TERKINI (4 Okt 2026) — BACA DULU; menggantikan butir lama yang bertentangan** (detail RESEARCH_NOTES §30–§35a, PROGRESS §47–§50):
  1. **Latih-ulang arm CoAtNet dari init c5 SUDAH HABIS diperas.** Terukur (gold-58, bersih): label guru 0,9033 → 4 murid fold OOF → label fusi (w_murid 0,5, ruang probabilitas) **0,9348**, TAPI model yg dilatih di label fusi TIDAK lebih baik (putaran-2: c5 0,9200 vs stok 0,9213; v8 0,9138 vs 0,9116). Model soup (rata-rata bobot) ≈ stok (0,922–0,9245) vs rata-rata prediksi 4 murid 0,9286 → NEGATIF. Semua gabungan keluarga CoAtNet jenuh ~0,93 (stok + 4 murid α=0,6 = 0,9310). Penyebab: noise gold-58 (SE ~0,01/model) + model sudah melampaui kualitas guru.
  2. **Syarat anggota baru: KUAT DAN beragam** (simulasi §30). HASIL 4-5 Okt: CoAtNet init-ImageNet & ResNet-50 init-ImageNet lambat belajar (gold 0,83-0,84 @6 epoch) -> DITUTUP. **RadImageNet-ResNet50 (domain medis, MIT di hulu, `shigechan/radimagenet-resnet50-official`) 224px**: 6 epoch 0,868-0,879/fold (ens 6 fold 0,880); **12 epoch + bias_init + bs2xacc2 = 0,895/fold** (288px tak membantu). Panjang latih adalah pengungkit. Solo LB (kernel solo, K=42): c5 0,927; v3 0,932; **r2c5 0,933; 4 murid CoAtNet fold 0,934** -> keluarga CoAtNet jenuh ~0,934. Builder: `build_train_cheap_kfold.py` (2 GPU/sesi, mode probe), `build_train_coatnet_kfold2.py` (murid CoAtNet 2 GPU), `build_submit_multi.py` (solo), `build_submit_plus.py` (terintegrasi).
  3. **Integrasi SELESAI & tervalidasi (5 Okt)**: `build_submit_plus.py` memakai basis publik terbaru `kaggle_base/pj_rsna_knee_d4_blend_20260928.ipynb`; sel 31 diganti leg fleet kita (dekode sekali, model dibagi 2 GPU, penjaga waktu, resi cakupan), sel 32 = blend rank bobot global (pola pjmathematician, bawaan 0,45 miliknya; kita 0,30-0,35). Dev-run stub lolos utk students4 & rad6. Disubmit real 5 Okt: `rsna-knee-d4-plus-students4` (w .35) & `-rad6` (w .30) -> hasil menunggu. Bukti papan atas: penulis publik dgn leg privat 0,947-0,951 vs dasar publik 0,943 (§36b).
  4. **Pelajaran metodologis yg berlaku**: gold-58 TERKONTAMINASI utk stack publik (0,9515 gold vs 0,943 LB) dan SE ~0,01 per model; LB hanya 3 desimal — keputusan besar tetap diputus submission real. Percobaan tanpa hasil (jangan ulang): bobot global (v3 α/w → 0,941 = base), sumber label publik, prior situs, model soup, putaran-2 label fusi.
- **TEMUAN PIVOT STRATEGIS (25 Agu) — SEBAGIAN DIBANTAH oleh real submission 26 Agustus, lihat koreksi di §3**: harvest OOF CoAtNet (`blend_results/FUSION_V2_RESULTS.md`) menemukan **CoAtNet v5_full_swa SENDIRIAN = 0,9206 gold-57 nested-CV, mengalahkan SEMUA blend LOKAL** (uniform-7=0,9055; bootstrap vs e9 P(menang)=1,000) — kesimpulan awal saat itu: "jalan ke 0,95+ = satu arm lebih kuat, bukan ensemble tuning". **TAPI submit c5 sendirian ke leaderboard real (26 Agustus) = 0,927 — LEBIH RENDAH 0,009 dari skor ensemble tim eksisting (0,936)**. Nested-CV N=57 TERBUKTI tidak cukup besar utk mendeteksi manfaat diversity ensemble yg baru kelihatan di skala test set asli (~1.300 studi) — selaras temuan independen sadamtorres (§27 lama/§23a: diversity tetap bernilai walau korelasi model tinggi). **Revisi strategi**: ensemble besar (0,936) TETAP submission terbaik/dipakai; arm CoAtNet kita (c5 atau hasil fine-tune) lebih masuk akal sbg TAMBAHAN ke ensemble drpd pengganti tunggal. Kode TRAINING CoAtNet asli tidak dirilis penulis c5 — arm kita sendiri (`kaggle_kernel_train_coatnet_v1/`) sudah berhasil dibangun & jalan (hasil 0,9193 gold-58, sedikit di bawah c5 asli — §3).
- **Backbone utama: DINOv2-base** (`metaresearch/dinov2`, Apache-2.0) — dominan dipakai kompetitor top, tervalidasi kompatibel `transformers.AutoModel`. DINOv2-small→base kapasitas TERBUKTI TIDAK berpengaruh (+0.0011 AUC, dalam noise floor) — **domain adaptation (unfreeze lebih dalam) adalah pengungkit nyata, BUKAN kapasitas/resolusi model** — sudah dikonfirmasi independen 2x.
- **Representasi studi**: "slot" bag (kombinasi anatomis plane×sekuens) + attention pooling per-diagnosis dengan prior anatomis hardcoded (`SLOT_PRIOR_TABLE`). Sampling `K_SLICE=4` (linspace-then-mean) per series — **belum diupgrade** ke teknik "3+3+3" (3 anchor merata + 2 tetangga fisik/anchor = 9 slice, representasi ~10mm depth asli) yang ditemukan lebih canggih di Discussion (735154). **v16/v17 dari line "3+3+3" NEGATIF & DITUTUP** (lihat §2, bullet 38).
- **ANATOMY-QUERY HEAD (v21, terinspirasi prvsiyan V19) — SELESAI DIUJI, HASIL NEGATIF (OOF 0,6904), TAPI BUKAN kegagalan arsitektur murni — lihat diagnosis di §3**: ganti "slot bag" dgn **1 series terbaik per plane** (Sagittal/Coronal/Axial, skor pilih `4*Fluid_Sensitive+3*Fat_Suppression`), 24 token anatomis (8 per plane), tiap token = **triplet RGB 3 slice berdekatan** (i-1,i,i+1) dari anchor merata, + **plane embedding + depth/position embedding**, lalu **12 query cross-attention per-diagnosis** atas 24 token (plane yg tak ada di-mask), + **unfreeze patch_embeddings** (selain 6 blok + layernorm). Resep prvsiyan V19 asli mengeklaim +0,00989 macro AUC tervalidasi, TAPI implementasi kita TIDAK mereplikasi klaim itu — **line ditutup** sbg arsitektur produksi kecuali distabilkan ulang (kandidat fix: bias-init classifier ke label prior, LR warmup, lihat §3).
- **Strategi weak-label**: label 58 studi gold **tidak pernah ditimpa** (selalu pakai annotator asli). 4.349 studi non-gold pakai label eksternal terbaik yang sudah diverifikasi macro AUC vs gold: **`flight0234/rsna-knee-hybrid-report-labels` (0.8991)**, dipakai sejak v8/v9 — plus confidence-weight `2*|p-0.5|` di loss BCE. Weak-labeler rule-based kita sendiri (`kaggle_kernel_weak_labeler/`) sudah selesai (agreement 0.747) tapi kalah akurat dari label eksternal, tidak lagi jadi sumber utama training.
- **Optimizer/fine-tuning (v11, breakthrough terbesar)**: `unfreeze_last=6` blok terakhir backbone (naik dari 2), LR terpisah `LR_HEAD=1e-3` / `LR_BACKBONE=8e-6` (nilai umum kode komunitas top), `gradient_checkpointing_enable()` wajib (hemat memori krn 6 blok trainable jauh lebih berat), `BATCH_STUDIES=4` (turun dari 8, batas T4 14.5GB dgn unfreeze=6).
- **Strategi resume/checkpoint**: pola self-reference `kernel_sources` (kernel modeling menambahkan slug dirinya sendiri ke `kernel_sources`-nya sendiri) — tervalidasi bekerja, dipakai utk resume training tanpa download lokal file besar. **Judul kernel Kaggle JANGAN diubah** kalau butuh slug stabil utk self-reference (title yg menyimpang dari slug asal memicu Kaggle rename URL).
- **Guard checkpoint wajib**: `best_auc` awal di-set ke skor checkpoint yang di-resume (bukan `-1.0`) supaya epoch tambahan yang lebih buruk TIDAK PERNAH menimpa checkpoint terbaik — pola ini WAJIB dipakai di setiap eksperimen resume ke depan.
- **Hal yang SUDAH diputuskan, TIDAK perlu didiskusikan ulang**:
  - **Temuan scraping 20 Agustus (sumber: prvsiyan V19, Mattia Angeli V104, thread 733932/736268/735826)**: (a) resep anatomy-query prvsiyan (detail §2 arah baru); (b) **Mattia Angeli V104 = rank-mean 20×DINOv2 + 1×DINOv3 ViT-S/16, public 0.922, TAPI ia sendiri memperingatkan public LB SUDAH overfit** — jangan jadikan public 0.922 target; (c) **konsensus forum: bottleneck terbesar = kualitas label** (k256.dev: gold label kontradiktif; PC Jimmmy: "the entire competition is the label making") — mendukung arah label-strengthening jangka panjang; (d) dataset baru tersedia utk eksplorasi label: `stevenleehans/rsna-knee-llm-report-labels`, `pilkwang/rsna-knee-llm-labels`, `lixin73` label (nama dataset blm dicek, belum dipakai), `antoinegg1/rsna-knee-e11-diverse-heads-v20`/`...e9-radimagenet-heads-v15` (head eksternal), **`mattiaangeli/knee-mri-fold-weights` (slug BENAR, koreksi dari "rsna-knee-mri-fold-weights" yg salah/tidak ketemu)**, `barun2104/rsna-knee-mri-processed-3d-volumes` (volume 3D utuh, belum dipakai); (e) archi DINOv3 = reference `facebook/dinov3` (bukan dari `metaresearch`); (f) **rombongan notebook publik yg sama sudah jadi "commodity" — semua fork dgn skor ~0.920, BUKAN keunggulan** (selaras keputusan §2 commodity-ceiling).
  - **TEMUAN TERBAIK PROYEK sejauh ini — 23 Agustus (`dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`, detail §25 RESEARCH_NOTES)**: notebook single-model (bukan ensemble) skor **0,924 public LB — LEBIH TINGGI dari submission tim kita (0,922)**. Backbone **CoAtNet** (`coatnet_rmlp_2_rw_384`, timm) — BUKAN DINOv2, dibandingkan sistematis oleh author vs Swin/EfficientNetV2 dan menang. Checkpoint (280MB, CC0-1.0, bebas pakai) di dataset `dreaddevelopment/raptor-knee-widedense`. Validasi lokal gold-58 (held-out): **0,9167** → offset ke LB cuma **+0,0073** (jauh lebih kecil dari klaim sadamtorres +0,04-0,046 — makin kuat model, makin kecil offset). Teknik data kunci: (a) **span sampling LEBAR 6-94% dari stack** (bukan window tengah) — author eksplisit sebut ini krn ligamen kolateral & Lateral Meniscus ADA DI TEPI stack, LANGSUNG relevan ke titik lemah Lateral Meniscus kita yg belum terpecahkan; (b) **crop fisik 140mm + resolusi 336px** — konfirmasi independen KE-2 (setelah mattiaangeli CROP_MM=130/SIZE=336) bahwa physical-mm-crop+resolusi>224 adalah pola top-performer; (c) **RGB-triplet 3-slice SAMA PERSIS teknik v16 kita yg gagal, tapi BERHASIL di sini** krn backbone di-fine-tune PENUH (bukan freeze-parsial) — bukti independen ke-2 (setelah `DepthCompress` mattiaangeli) bahwa akar masalah v16 adalah freeze-schedule, bukan RGB-packing itu sendiri; (d) attention pooling per-label (mirip filosofi kita) tervalidasi independen ke-3. **Belum diputuskan**: submit langsung (opsi tercepat naikkan skor tim), blend dgn OOF kita, atau adopsi teknik data ke v22 — nunggu arahan user.
  - **Temuan scraping 21 Agustus (bedah kode penuh `mattiaangeli/bend-the-knee-to-dinov3-ensembled`, detail §24 RESEARCH_NOTES)**: author SENDIRI memperingatkan notebooknya overfit ke public LB akibat race fork-tweak komunitas — validasi kuat independen thesis "commodity ceiling" kita. 3 teknik arsitektur bernilai utk v22 kalau v21 kurang: **(1) `DepthCompress`** — modul gated-conv learnable (bukan RGB-packing literal spt v16 yg gagal) utk kompres N_SLICE=16 slice mentah → 3 channel pseudo-RGB SEBELUM masuk backbone, residual+layer-scale kecil (stabil dari awal training); **(2) `ViTSlotToken`** — sisip token kategori slot (plane×fat-suppression) ke PREFIX TOKEN ViT langsung (bukan cuma ditambah di head spt `plane` embedding kita), shg self-attention DI DALAM backbone bisa pakai info slot; **(3) `TokenXAttnPool`** — cross-attention `nn.MultiheadAttention` atas token SEMUA series dlm studi (var-length + key-padding-mask), BUKAN pilih 1 series terbaik per plane spt v21 kita — kandidat penyebab kalau v21 kurang menang: pemilihan 1-per-plane mungkin buang informasi. Dataset checkpoint SIAP PAKAI (CC0): **`mattiaangeli/knee-mri-fold-weights`** = 5-fold DINOv3 ViT-S/16 terlatih (`m_f0..f4.pt`, ~94MB/fold), butuh pipeline preprocessing beda (SIZE=336, CROP_MM=130 crop fisik, bukan resize 224 kita) — cheap-win ensemble tapi integrasi non-trivial, antri setelah submission real pertama.
  - **HASIL SCRAPING PENUH 25 Agustus (arahan "pelajari notebook 0,936 + model/dataset/diskusi sampai bisa kalahkan top 1") — detail penuh RESEARCH_NOTES §26**: (a) **puncak Code-tab publik = keluarga DINOsaur/Raptor (romantamrazov), proven 0.936**; replika terbersih = `prvsiyan/head-and-shoulders-knees-and-toes` (port Apache-2.0 setia dari DINOsaur V10 scriptVersionId 344688577, mount distandarkan); (b) anatomi stack 0.936: manifest DINOv2/DINOv3 24-member + A5 fold-family timm @0.45 + RadImageNet chain (E10/E13/E11 + V18 calibrator 88-fitur embedded) + dual CoAtNet MaxSpan-v5-SWA/WideDense-v4 dgn per-diagnosis correlation-gated weights — SEMUA rank-space fusion; klaim internal: CoAtNet single 0.924 LB / 0.9167 gold-58; "ensembling worth ~+0.001 LB"; offset gold→LB ≈ +0.007-0.012; author fork-race sendiri memperingatkan public-LB overfit; (c) dataset pendukung baru: `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, `dreaddevelopment/raptor-knee-maxspan`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`, `sofiaanjenje/rsna-knee-e11-train`/`e13-train`, `lixin73/rsna-knee-train-checkpoints` (DINOv2-Small SlotHead attention — mirip arsitektur kita, dipakai fork publik @w=0,10); deskripsi dataset `antoinegg1/...e9/e10` membuka metodologi nested-CV weight selection di gold-58 (nested 0.862-0.873 vs base 0.840); (d) sinyal diskusi: backbone besar/TTA/more-members ≈ zero gain; pengungkit nyata = label severity-aware ("small effusion"→~0,2 bukan 1,0; borderline=negative; monotonic+shrinkage+floor; kalibrasi in-fold) + geometri; LB desimal ke-3 sering tidak resolvable; (e) **strategi menuju top-1 sudah disusun (RESEARCH_NOTES §26f): Fase A pegang kode stack 0.936 → Fase B label engineering severity-aware + LLM (gate: kalahkan flight0234 0.8991) → Fase C fusion nested-CV di gold-58 (bukan public-fit) → Fase D 2 final submissions profil beda** — menunggu arahan user sebelum eksekusi.
  - **RISET DISCUSSION SEGAR 26 Agustus (arahan "scraping seluruh strategi, beri strateginya") — detail penuh RESEARCH_NOTES §27**: 4 temuan baru, 1 diuji langsung. **(1) Root cause Synovitis dikonfirmasi independen**: cuma ~16% laporan sebut synovitis eksplisit; RadImageNet-pretrained 0,78 vs ImageNet-pretrained 0,62-0,72 di label ini — kita PUNYA lisensi RadImageNet tapi belum pernah bangun arm sendiri. **(2) Hipotesis pooling spasial** (belum diuji siapa pun): GAP/mean-pooling melunakkan sinyal temuan kecil (meniscus ~3mm = 0,01% frame) — LANGSUNG relevan ke `patch_mean.mean(1)` di `SlotHead` kita, kandidat arsitektur BARU (attention di level patch-dalam-gambar, beda dari v21 yg attention di level window). **(3) Peringatan metodologis KERAS + terverifikasi via pengalaman kita sendiri hari ini**: checkpoint selection di gold-57/58 bisa MENYESATKAN ("model terakhir di 58 studi, tapi pertama di LB") — persis pola c5-solo kita (menang lokal 0,9206, kalah real LB 0,927 vs ensemble 0,936). **JANGAN lagi putuskan strategi besar murni dari gold-58, submission real WAJIB jadi pemutus.** Ditemukan & DIVERIFIKASI lokal: 194 studi/47 grup Report byte-identical (`reference_metadata/train.csv`), 1 dari 58 gold berbagi report dgn studi weak-label lain — risiko leakage kecil di skema k-fold kita (belum diperbaiki, utang teknis). **(4) Diuji langsung (lokal, `label_engineering/blend_labels.py`, tanpa GPU)**: blend flight0234+stevenleehans+pilkwang **TIDAK mengalahkan flight0234 sendirian** (0,8991 vs blend terbaik 0,8978) — line DITUTUP, flight0234 tetap satu-satunya label source.
  - Target akhir: kalahkan skor 0.952 (Brandon Low, terbaru 25 Agu), fokus skor mentah bukan Efficiency Prize.
  - Backbone tunggal DINOv2-base dulu sebelum ensemble multi-backbone (RadImageNet-ResNet50 MIT + BiomedCLIP MIT) — pendekatan bertahap.
  - Metode bebas ("terserah") selama mengejar 0.950, asal tidak melanggar Rules resmi.
  - Strategi ikut pola top-20 sungguhan: **single model kuat > ensemble komunitas 20+ model** (Discussion 735304: single model 0.92–0.934 tervalidasi; komentar komunitas menyebut ensemble publik besar "waste of time", cuma kudos-farming) — jadi prioritas riset sekarang adalah memperkuat 1 backbone (fine-tuning lebih dalam, k-fold, LR scheduler), BUKAN menambah anggota ensemble komunitas.
  - Dataset/model eksternal ber-lisensi **CC0/MIT/Apache-2.0 dipakai bebas**; **lisensi NonCommercial (CC-BY-NC, mis. `marwanmath/resnet-50-radimagenet-marwan`) di-SKIP** (bukan krn dilarang Rules kompetisi, tapi risiko lisensor asli terpisah) — kita sudah punya RadImageNet-ResNet50 MIT-licensed sendiri sbg pengganti.
  - `sohaibanwaar1203/kneemridataset` (data eksternal RS Rijeka Kroasia) **diizinkan dipakai** (lisensi cuma minta atribusi+sitasi, Rules kompetisi eksplisit izinkan data eksternal) — belum diimplementasikan.
  - `pilkwang/rsna-knee-weights` (paket 20-model DINOv2-small rank-mean ensemble, CC0) **dipakai sbg anggota ensemble independen** (bukan disuntik ke bobot kita) — evaluasi HARUS pakai `oof.npz` resmi (out-of-fold), JANGAN live-infer ke gold set (lihat §4 soal jebakan leakage).
  - Restriksi API LLM hosted **sudah DILONGGARKAN** (dikonfirmasi tidak melanggar Rules) — implementasi relabeling LLM DITUNDA dulu (bukan dibatalkan), fokus saat ini tetap penguatan single-model.
  - **Skor 0,920 tim (via fork `tonylica/rsna-knee-dino-radimagenet-rank-ensemble`) adalah "commodity ceiling"** — puluhan tim lain (rank ~100-140+) punya skor identik krn fork notebook publik yang sama. BUKAN target akhir, BUKAN pencapaian unik — jangan disalahartikan sbg progress asli menuju top 1. Prioritas: bangun keunggulan yang TIDAK dibagikan publik (geometri slice, model sendiri yang lebih kuat), bukan menambah anggota fork lagi.
  - **v14 (TPU, cache v1, LR decay) mentok di 0,8159 — bukti ke-4 bahwa tuning training-schedule saja SUDAH MENTOK di config lama.** Prioritas dipindah ke geometri slice, DAN INI TERBUKTI BERHASIL: **v15 (TPU, cache preprocessing v2 — urutan slice fisik `ImagePositionPatient`) mencapai val_macro_auc 0,8304, MODEL TERBAIK BARU PROYEK** (naik dari checkpoint v11 0,8195, +0,0109). SATU-SATUNYA variabel yg beda dari v14 adalah data source — bukti langsung (bukan cuma laporan pihak lain) bahwa geometri slice adalah pengungkit nyata.
  - **TPU tervalidasi ~12x lebih cepat dari GPU T4** (~6,2 menit/epoch vs ~74 menit/epoch) — dipakai utk semua eksperimen sejak v13.
  - **Checkpoint v15 (0,8304) BELUM di-protect dgn guard** — kalau push versi baru ke slug `rsna-knee-modeling-v15-geom-v2-tpu`, WAJIB set `best_auc` awal ke 0,8304 (bukan -1,0), supaya tidak kena gotcha yg sama spt v11/v12 (checkpoint tertimpa/hilang dari akses `kernel_sources`).
  - **Sampling "3+3+3" SUDAH DICOBA 2x (v16, v17) — KEDUANYA NEGATIF, line riset ini DITUTUP**: v16 (channel RGB genuine) = 0,7737 (channel-mismatch, patch embedding beku). v17 (9 view independen grayscale-repeat) = 0,7999 — lebih baik dari v16 tapi TETAP < v15 (0,8304, TAPI lihat catatan k-fold di bawah). **CATATAN PENTING (pasca-v18)**: magnitude perbedaan v15 vs v16/v17 (0,03-0,06) ternyata berada DALAM/DEKAT noise-floor k-fold yg baru terkuantifikasi (std antar-fold 0,034) — diagnosis channel-mismatch v16 tetap valid scr arsitektur, tapi jangan terlalu yakin soal MAGNITUDE bedanya.
  - **k-fold CV (v18) SELESAI — TEMUAN METODOLOGIS BESAR**: 5-fold CV atas 57 gold, tiap studi divalidasi model yg tak pernah melihatnya (OOF). **Hasil per-fold: 0,7915 / 0,8400 / 0,8670 / 0,7949 / 0,8698 (rata-rata 0,8326 ± 0,0338, rentang 0,078!). OOF macro AUC gabungan (metrik paling terpercaya): 0,8068 — LEBIH RENDAH dari v15 (0,8304, single-split).** Ini mengungkap validasi 1-split N=57 yg dipakai sejak v6 CUKUP BERISIK — selisih antar-fold (0,078) jauh lebih besar dari selisih² yg dikejar antar-eksperimen selama ini (biasanya 0,01-0,06). v15 kemungkinan dapat split yg menguntungkan, bukan murni performa objektif terbaik. **OOF/k-fold sekarang metrik yg lebih dipercaya drpd single-split** utk eksperimen berikutnya.
  - **5 checkpoint fold (v18) tersimpan** (`best_model_fold0-4.pt`) — kandidat kuat utk ensemble 5-model (murah, cuma rata-rata prediksi, tidak perlu training baru).
  - **Submission v2 (ensemble 5-fold, model KITA SENDIRI) tervalidasi penuh** (`kaggle_kernel_submission_v2/`): dev-test CPU di `test_series` stub 3 studi SUKSES 0 error, format `submission.csv` benar. Decode DICOM mentah on-the-fly (test tidak punya cache), pakai fungsi preprocessing v2 (urutan fisik + clip per-grup, `group_norm_stats.csv` dipinjam dari preprocessing v2). **Temuan: CPU ~58 detik/studi → ekstrapolasi ~21 jam utk 1.300 studi test, jauh lewat batas 9 jam** — GPU/TPU WAJIB utk submission sungguhan. **BELUM disubmit ke leaderboard.**
  - **Blend OOF kita + pilkwang = 0,8462 (α=0,60 pilkwang/0,40 kita)** — skor lokal leak-free tertinggi proyek, ditemukan TANPA training baru (§3). Robust di rentang α 0,30-0,90.
  - **Laterality canonicalization SELESAI diimplementasikan (preprocessing v4) & DIUJI — TIDAK TERBUKTI MEMBANTU, line DITUTUP utk arsitektur SlotHead/mean-pooling**: deteksi geometris tervalidasi 97,4% akurat vs tag `Laterality` asli (24.371 series, run penuh sukses). v19 (single-split) = 0,8245 (inconclusive vs v15 0,8304 krn noise floor). **v20 (k-fold, konfirmasi) SELESAI: OOF macro AUC 0,8073 vs v18 baseline (SlotHead, tanpa laterality) 0,8068 — selisih +0,0005, PRAKTIS NOL.** Per-fold: 0,8247/0,8061/0,9042/0,8156/0,8491 (rata2 0,8400±0,0352). **Kesimpulan: laterality canonicalization TIDAK memberi peningkatan nyata pada arsitektur SlotHead** — diagnosis: `patch_mean = out[:,1:].mean(1)` (mean-pool token DINOv2) menghilangkan informasi spasial/posisi SEBELUM head, jadi flip kiri-kanan tidak banyak mengubah fitur yg sudah di-pool. **Pola berulang ke-2** setelah v16 (RGB-packing gagal krn patch embedding beku) — dua kasus di mana perbaikan sisi-data tidak bisa terpakai krn ARSITEKTUR tidak punya "jalur" utk memanfaatkannya. **Model final tetap checkpoint v15 (0,8304 single-split) / v18 (0,8068 OOF)** utk SlotHead. Laterality canon TETAP dipakai di cache v4 sbg data source utk v21 (anatomy-query), krn arsitektur itu attention-pooling sadar-posisi, bukan mean-pool — punya "jalur" utk memanfaatkan flip yg SlotHead tidak punya.
  - **v21 SELESAI PENUH (5-fold, ~5,2 jam TPU) — OOF MACRO AUC 0,6904 — REGRESI vs v18 (0,8068) & v20 (0,8073), TIDAK dipakai sbg model final**: push-1 sempat ERROR (`DATA_OK=False` false-positif, sudah diperbaiki §3), push-2 (fixed) jalan penuh. Per-fold: **0,6276 / 0,8233 / 0,8902 / 0,6681 / 0,6199** (rata2 0,7258±0,1102). **Diagnosis dari training-loss trajectory per fold (BUKAN cuma angka akhir)**: pola BIMODAL, bukan random noise merata — fold 2&3 train_loss turun tajam (0,58→0,33 & 0,59→0,37) dan val_auc naik mulus ke 0,82-0,89 (LEBIH BAIK dari v18!); fold 1,4,5 train_loss NYARIS TIDAK BERGERAK (mis. fold 5: 0,6067→0,5844 sepanjang 8 epoch) & val_auc bolak-balik di sekitar acak (0,42-0,63) — model di fold ini gagal keluar dari basin optimisasi awal dalam budget 8 epoch/LR decay yg dipakai. **Kesimpulan: arsitektur `AnatomyQueryHead` PUNYA potensi nyata (2/5 seed capai >0,82) tapi TIDAK STABIL lintas fold/seed** — kandidat penyebab: init `query`/`cls_w` std=0,02 + `cls_b` mulai dari nol (bukan bias ke label-prior, trik stabilisasi umum utk multi-label imbang) dikombinasi unfreeze `patch_embeddings` baru → training landscape lebih rawan stuck. **Line ditutup UNTUK SEKARANG** (bukan permanen) — kalau mau dilanjutkan, perlu fix stabilisasi (bias-init label-prior, LR warmup) sebelum re-run, BUKAN sekadar ulang dgn seed beda. Model final proyek TETAP checkpoint v15 (0,8304 single-split) / v18 (0,8068 OOF).
  - **v23 (spatial attention pooling) SELESAI — sendirian TIDAK menang (0,7967 < v18 0,8068), TAPI per-label mengungkap pola KOMPLEMENTER bernilai, blend v18+v23 = OOF TERBAIK BARU (0,8176)**: `kaggle_kernel_modeling_v23_spatialpool/rsna-knee-modeling-v23-spatialpool-tpu`. 5/5 fold selesai stabil (rentang 0,7891-0,8717, spread mirip noise floor v18). **Per-label (vs v18/GAP)**: `SpatialAttnPool` JAUH lebih baik di temuan kecil/terlokalisir — **Medial Meniscus +0,103, Lateral Meniscus +0,064** (PERSIS hipotesis §27b yg diuji) — tapi LEBIH BURUK di temuan difus/tersebar luas: **MCL -0,107, Contusion -0,096, Medial OA -0,051, Baker's -0,050, Synovitis -0,049**. **Penjelasan fisiologis masuk akal**: robekan meniscus = fitur kecil-diskret (attention spasial menang), sedangkan sprain MCL/edema kontusio/penebalan sinovial = proses DIFUS tersebar di area luas (GAP/rata-rata menang krn attention yg fokus sempit malah kehilangan sinyal yg tersebar). **Blend rank-mean v18+v23 (`blend_results/blend_v18_v23.py`) = 0,8176 @ α(v18)≈0,5-0,65 — OOF TERBAIK BARU keluarga SlotHead**, ROBUST di rentang lebar (α 0,30-0,85 semua >0,81, bukan artefak titik tunggal, pola sama spt blend v18+pilkwang §22). **10 checkpoint tersedia** (5 fold v18 + 5 fold v23) sbg kandidat ensemble nyata. Keputusan: JANGAN buang v23 sbg "gagal" — nilainya ada di DIVERSITY utk blend, bukan sbg pengganti v18. Prioritas berikutnya: (1) uji blend v18+v23 di submission real (butuh kernel inferensi 10-checkpoint baru); (2) arm RadImageNet (§27a, target Synovitis) tetap di antrian.
  - **v24 (dual-pool dlm 1 model) SELESAI — GAGAL menggabungkan manfaat, OOF 0,7968 ≈ v23 sendirian (0,7967), TIDAK mendekati blend manual (0,8176)**: `kaggle_kernel_modeling_v24_dualpool/rsna-knee-modeling-v24-dualpool-tpu`. 5/5 fold stabil (0,759-0,856), tidak ada instabilitas. Sesuai kriteria keputusan yg ditulis SEBELUM hasil ("v24≤0,8068 → kembali ke blend manual"): **hasil GAGAL** — memberi `SlotHead` fitur lebih kaya (GAP+attn digabung, 3×dim) TIDAK memberi manfaat yg sama dgn benar2 melatih 2 model independen lalu blend. **Pelajaran penting**: manfaat diversity dari ensemble (2 model, random init/trajectory beda) TIDAK bisa direplikasi cuma dgn kasih 1 model fitur input lebih kaya — sumber diversity-nya memang dari 2× training run terpisah, bukan dari arsitektur gabungan. **Keputusan final**: line dual-pool-dalam-1-model DITUTUP. **Hasil terbaik keluarga SlotHead TETAP blend manual v18+v23 (0,8176)** — checkpoint v24 TIDAK dipakai. Prioritas berikutnya: (1) validasi blend v18+v23 via submission real (uji apakah 0,8176 lokal benar2 lebih baik di LB nyata, mengingat pelajaran c5-solo bahwa lokal bisa menyesatkan §27c); (2) arm RadImageNet (§27a, target Synovitis).
  - **Submission v3 (ensemble v18+v23, 10 checkpoint, rank-mean) DIBANGUN, DEV-TEST SUKSES BERSIH, SUDAH DISUBMIT REAL (27 Agustus)**: `kaggle_kernel_submission_v2/rsna-knee-submission-v2-own-5fold-ensemble` (slug lama, isi baru). Ditambah `KneeModelV23` berdampingan `KneeModelV18`, checkpoint dipisah per path kernel_sources asal. **Fix pending lama SEKALIAN DIBERESKAN**: probability-mean → rank-mean equal-weight lintas 10 model (restrukturisasi `infer_study()` simpan prediksi mentah dulu, rank+rata-rata setelah loop test penuh). Config diubah ke GPU T4 + `enable_internet=false` (submission-siap). **Dev-test (kernel version 2, stub 3-studi) SUKSES TOTAL**: device=cuda, semua 10 checkpoint ketemu & termuat benar ke kelas arsitektur yg tepat, 3/3 studi berhasil, ~74 detik total (~5,7 detik/studi murni inferensi) → ekstrapolasi test set asli (~1.300 studi) **~2 jam, jauh di bawah batas 9 jam**. Semua paket (pydicom/transformers/sklearn) preinstalled, TIDAK butuh internet. **HASIL REAL (submission 55820944): 0,837 public LB.** Offset lokal→LB POSITIF (+0,019 dari OOF 0,8176 — arah sesuai pola umum, model kuat cenderung LB > lokal), TAPI angka absolutnya JAUH di bawah c5-solo (0,927) & ensemble tim (0,936).

**KESIMPULAN STRATEGIS BESAR (28 Agustus) — line DINOv2+SlotHead (v11-v24) DITUTUP RESMI**: dua tes real-LB minggu ini (c5-solo 0,927; blend v18+v23 0,837) SAMA-SAMA KALAH dari submission tim eksisting (0,936, fork community CoAtNet-based). Arsitektur rumahan kita (DINOv2-base + slot-bag + SlotHead, SEMUA varian v11 s/d v24 — GAP, laterality, triplet-RGB, widecrop, spatial-attn, dual-pool) plafon nyata di real LB ternyata **~0,84**, gap ~0,09-0,10 dari CoAtNet arm SOLO (0,927) — bukan cuma gap kecil yg bisa ditutup tuning lebih lanjut. **13 eksperimen (v11-v24) sudah cukup utk simpulkan: keluarga arsitektur ini TIDAK kompetitif**, DINOv2-base+SlotHead bukan levernya. **Line ini DITUTUP, TIDAK dilanjutkan lagi** (tidak ada lagi v25 dst di jalur SlotHead) — checkpoint terbaik v18/v23 disimpan sbg arsip riset, bukan dipakai lanjut.
  - **Pelajaran metodologis besar TERKONFIRMASI 2x berturut2**: local nested-CV/OOF (N=57-58) TERBUKTI TIDAK RELIABLE utk memprediksi ranking real-LB antar STRATEGI BESAR (arsitektur beda) — c5 "menang" vs blend lokal, TAPI blend v18+v23 (0,8176 lokal) < c5 (0,9206 lokal) DI KEDUANYA lokal MAUPUN real (0,837 < 0,927) — jadi kali ini rankingnya KONSISTEN lokal-vs-real, cuma magnitude absolutnya yg jauh dari community CoAtNet. **Pelajaran diperbarui**: OOF gold-58 CUKUP RELIABLE utk ranking relatif ANTAR-VARIAN dalam SATU arsitektur (v18 vs v20 vs v23, dst — semua akhirnya konsisten arahnya), TAPI TIDAK reliable utk membandingkan ANTAR-ARSITEKTUR BEDA TOTAL (SlotHead vs CoAtNet) krn skala/offset ke real-LB bisa beda jauh — c5 offset cuma +0,007, SlotHead family offset +0,019, keduanya "wajar" sendiri2 tapi base rate-nya beda jauh.
  - **Prioritas final (28 Agustus) — SEBAGIAN TERJAWAB per 2 September, lihat v3 di §3**: LLM API relabeling (prioritas #1 saat itu) SUDAH dieksekusi penuh (pilot→scale-up 4.349 studi→blend→3x fine-tune CoAtNet v1/v2/v3) — **v3 (raw-probability-mean blend) MENANG vs c5 baseline (0,9234 vs 0,9206)**, gate ">c5" akhirnya LOLOS setelah v1/v2 gagal. (2) arm RadImageNet (§27a, target Synovitis) masih di antrian, belum dikerjakan. (3) submission tim TETAP 0,936 (fork community) sbg yg dipakai di leaderboard nyata — **v3 BELUM disubmit real**, jangan declare victory sebelum konfirmasi LB (pelajaran c5-solo & v18+v23 §39/§44: gold-58 lokal bisa menyesatkan).
  - **Gotcha operasional penting (masih berlaku)**: pola self-reference `kernel_sources` + guard checkpoint (v12) membuat checkpoint versi lama TIDAK BISA diakses kernel lain begitu versi terbaru kernel itu tidak menimpanya (`kernel_sources` selalu mount versi TERBARU, bukan versi spesifik). Kalau butuh akses checkpoint lintas-versi yg aman, simpan sbg Kaggle Dataset terpisah, jangan andalkan `kernel_sources` semata.

- **HASIL v3 IN-ARM (§50) = 0,941** — SAMA PERSIS dgn baseline, jadi **DUA kali** percobaan menyuntikkan model kita ke ensemble publik menghasilkan nol perubahan (α=0,15 di luar §48; w=0,35 di dalam ARMS §50). Ini BUKAN soal kode — melainkan bobot blending selalu ditebak, karena kita tak pernah punya prediksi ensemble publik di data berlabel.
- **SIMULASI ENSEMBLE (§30) menetapkan syarat kuantitatif**: (a) menumpuk model SEJENIS mentok ~0,937 pada korelasi terukur v3-vs-c5 (ρ=0,94), berapa pun jumlahnya; (b) model tambahan yang lebih lemah (0,900) dgn bobot SAMA malah MERUSAK (0,9334→0,9180), dan dgn bobot optimal pun cuma +0,004 pada ρ=0,75. **Anggota baru wajib KUAT (~0,93) DAN beragam (ρ≤0,8) sekaligus** — ini menggugurkan rencana melatih backbone baru dari ImageNet (terlalu lemah) dan menurunkan nilai v4 (CoAtNet lagi).
- **Arah yang tersisa & sedang dieksekusi**: (1) panen prediksi ensemble publik di 58 gold (`rsna-knee-harvest-public-gold58`) supaya blending jadi TERUKUR, bukan ditebak — pipeline publik ternyata menghormati env `RSNA_COMP_ROOT` jadi bisa diarahkan ke studi gold; (2) terapkan resep label kita ke SELURUH arm CoAtNet publik (v10 & v8, builder terparameter `build_train_coatnet_variant.py`), bukan satu arm saja, supaya cabang CoAtNet (bobot 0,60-1,00 di blend akhir) naik utuh.
- **Catatan koordinasi**: rekan tim aktif submit (d4-blend, d4-lite). 26 Sep terpakai 4/5 slot, salah satunya kernel panen diagnostik yang tersubmit TIDAK SENGAJA (bukan oleh saya) — kernel diagnostik sebaiknya diberi nama jelas atau `competition_sources` dilepas agar tidak tampil sbg kandidat submission.

## 3. Status Pipeline (overwrite tiap update)
- **PROGRAM 8 Okt malam (arahan user: "riset mendalam, eksekusi tanpa GPU (limit) -> TPU + CPU Kaggle; GPU dipakai lagi setelah pulih 10 Okt 07:00 WIB")** (detail RESEARCH_NOTES §37): jalur = ANAK fine-tune dari ckpt OAI publik nartaa r384 pada label fusi r1 + KD ke induk, lalu ensembel induk+anak di pipeline Apex. Langkah: (1) cache geometri nartaa `rsna-knee-volcache-z{0,1,2}` (CPU, 96 irisan/384 px/2-98%); (2) TPU `rsna-knee-train-tpu-oai-child` (varian a: KD 0,5; b: tanpa KD; ~6 jam; keluaran `oai_child_{a,b}_{ep*,final,swa}.pt` + `parent_soft_k16.csv`) didorong otomatis oleh skrip tunggu-cache; (3) GPU (setelah 10 Okt): `build_train_gpu_oai_child.py` (varian c,d paralel di 2xT4) + commit `build_submit_oai_family.py name=fam1` (induk+2 anak+reader goodpjw) -> submit; terima bila LB >=0,951. Probe CPU `rsna-knee-dicom-tags-probe` mengecek apakah header DICOM punya usia/jenis kelamin (sinyal baru utk label OA). Gold tetap TIDAK realistis (+0,009; butuh ensembel banyak model ~0,951 + data eksternal).
- **STATUS 8 Okt 19:40 WIB: tim rank 292/5.490 @ LB 0,950** (submisi 56943646, 8 Okt 08:06 UTC, notebook publik "RSNA Knee Apex Grandmaster Stack" v2 oleh ItsmeSujan; di-submit anggota tim, BUKAN agen). Isinya: checkpoint `nartaa/rsna-knee-publication-swa-weights-20261007` (Raptor CoAtNet 384, solo 0,949, **dilatih dgn data eksternal OAI/NDA akses terkontrol**) + reader goodpjw per-label. Batas medali (5.490 tim): emas <=20 (0,959), perak <=274 (0,950), perunggu <=549 (0,947). Dataran 0,950 = rank 146-397 (252 tim, mayoritas fork notebook publik); >=0,951: 145 tim, >=0,952: 122, >=0,953: 101; puncak 0,964. Perak = +1 tick (0,951) di atas dataran; emas = +0,009 (tidak realistis, ~1-2%).
- **RISIKO KEPATUHAN OAI**: thread host 743416 (YYama) belum dijawab host sampai 8 Okt; Rules: "Freely & publicly available external data is allowed, including pre-trained models"; lisensi bobot nartaa = "research & educational use, incl. RSNA Knee competition". Rencana: 2 submisi final = 1 BERSIH (tanpa OAI: replika 0,944 / d4 0,943) + 1 AGRESIF (Apex 0,950); putuskan 21-22 Okt setelah cek ulang thread 743416.
- **Probe publik (goodpjw, 8 Okt)** di atas 0,949: + stack 0,943 @40% = 0,949; + reader data-kompetisi-saja (0,933/0,929) @30% = 0,948 -> TIDAK ada tambahan dari model tanpa data eksternal. Konsisten dgn temuan kita. Sisa keuntungan = label + data eksternal. Reader KneeNet kita: fold 0 TPU gold-58 EMA puncak 0,896 @ep4 lalu DIVERGEN di ep5 (loss 7e10, tanpa grad-clip) -> 0,49; DITUNDA (tak lagi prioritas). Aset nartaa juga berisi ckpt r224 (0,945 solo) yg belum dicoba.
- **TARGET USER 6 Okt: silver, bila bisa gold.** Jujur: silver ~0,946 (rank <=263; perlu +0,002), rank 50 ~0,954, gold ~0,957 (rank <=20) sangat kecil peluangnya. Jalur silver: tambah KELUARGA INDEPENDEN (reader KneeNet buatan sendiri di label fusi: `build_train_kneenet.py`, kernel TPU `rsna-knee-kneenet-tpu-s1` fold 0 antre; `build_submit_plus.py` kini mendukung model `kneenet*` + reader goodpjw), uji bobot setelah GPU pulih 10 Okt. Pilih 2 submisi final di akhir.
- **HASIL TERBAIK 6 Okt 20:00 WIB: replika goodpjw (stack .7 + reader ConvNeXt-Tiny 2.5D publik .3) = LB 0,944 = ZONA PERUNGGU** (>=0,944: 484 dari 5.278 tim; batas perunggu rank 527; perak ~0,946 rank 263). Kernel `dimaspashaakrilian/rsna-knee-goodpjw-replica` v1 (submisi 56872456). Leg luar CoAtNet/RadImageNet kita MENURUNKAN (0,942), arm-in-stack netral (0,943) -> hanya keluarga independen kuat (~0,93) yg membantu. Berikutnya: bobot reader 0,40 (butuh GPU, pulih 10 Okt), reader buatan sendiri di TPU.
- (sebelumnya) **HASIL 6 Okt 02:00 WIB**: 3 submisi leg luar (students4 .35, rad6 .30, fleet2) SEMUA 0,942 (< 0,943 basis) -> leg luar lemah DITUTUP (§36d). Sedang dinilai: `rsna-knee-inarm2-c5v8` v2 = arm Raptor c5/v8 diganti versi label-fusi di DALAM stack.
- **STATUS 5 Okt 2026 ~22:00 WIB**: VM Colab DILEPAS server setelah ±5 jam (keep-alive 404), lalu `colab new --gpu T4` -> "Service Unavailable" (batas pemakaian GPU akun Colab gratis). Hasil Colab yg selamat: fold 0-2 RadImageNet 12 epoch (gold 0,891/0,892/0,884) di dataset `dimaspashaakrilian/rsna-knee-radresnet50-ep12`. Fold 3-5 dilanjutkan di **TPU Kaggle** (aturan: GPU Kaggle -> Colab -> TPU): kernel `rsna-knee-cheap-radresnet50-ep12-s1` (TPU v5e-8, port XLA via `build_train_cheap_kfold.py ... tpu=1,folds=3+4+5`). 3 submisi terintegrasi (students4 .35, rad6 .30, fleet2 = publik .60 + murid .20 + reader goodpjw .20) sedang dinilai.
- (status sebelumnya) **STATUS 5 Okt 2026 ~15:30 WIB**: GPU Kaggle HABIS (29,3/30 jam, pulih 10 Okt 07:00 WIB) -> latih di **GPU Colab** lewat `colab` CLI: sesi `rsna` (T4, 2 vCPU, 12GB), run `cheap_radresnet50_ep12_s1` 6 fold x 12 epoch (~84 mnt/fold) di `/content/kw`; paket `colab/cheap_radresnet50_ep12_s1/`, bootstrap `colab/bootstrap.py` (Kaggle CLI VM harus >=2.2 utk paginasi; /kaggle read-only -> root `/content/kin`,`/content/kw`; hanya 1 sesi GPU Colab sekaligus). Setelah selesai: unduh + unggah sbg dataset Kaggle -> `build_submit_plus.py`. WAJIB `colab stop -s rsna` bila selesai.
- (status lama 4 Okt) **STATUS 4 Okt 2026 ~11:30 WIB**: BERJALAN — `rsna-knee-train-inet-gpu-s1` & `-s2` (GPU T4, lihat §2 butir 2). SELESAI (hasil di §2): fold OOF `rsna-knee-kfold-c5-f0..f3`, putaran-2 `rsna-knee-train-round2-c5-r1`/`-v8-r1`, `rsna-knee-soup`, evaluator `rsna-knee-eval-*`; cache CPU `rsna-knee-volcache-{s,w,x,y}{0,1,2}` (COMPLETE). Kernel TPU `inet-tpu-s2` DIBATALKAN & dihapus atas arahan user (TPU hanya bila GPU habis). Kuota 11:01 WIB: GPU 14,19h terpakai / 15,81h sisa; TPU 0,17h / 19,83h; refresh berikut 10 Okt 07:00 WIB. Semua kerja ter-commit & ter-push kecuali builder/kernel seed-2 (`build_train_inet_gpu_s2.py`, `kaggle_kernel_train_inet_gpu_s2/`). Submission real terakhir tim: 0,943 (rank #561, 3 Okt).
- **Label engineering SELESAI round-1 (25 Agu malam) — GATE 0.8991 BELUM LOLOS**: `label_engineering/severity_labels.py` (lexicon publik daur ulang + cap severity sesuai heuristik resmi). Hasil vs gold-58: lexicon mentah 0.8617 / +severity cap 0.8620 / steven v4_blend 0.8927 / pilkwang v2 0.8639 / **flight0234 tetap terbaik 0.8991**; blend antar tabel tidak menembak gate (terbaik all3=0.8970); cap severity DI ATAS tabel LLM malah menurunkan tipis (mereka sudah encode severity). Titik lemah flight utk sasaran perbaikan: Synovitis 0.790, Lat OA 0.833, Contusion 0.860, Fracture 0.870. Output tabel tersimpan `label_engineering/*.csv`. Kesimpulan: tanpa LLM API, flight0234 = label terbaik yg ada.
- **TRAINING ARM SENDIRI — v1→v2→v3, GATE AKHIRNYA LOLOS di v3 (0,9234 > c5 0,9206)**: 3 iterasi fine-tune CoAtNet full-ft di atas checkpoint c5, isolated diff = sumber label training saja.
  - **v1** (`rsna-knee-train-coatnet-flight-ft-v1`, label flight0234-only): 0,9193 < c5 0,9206 — gate TIDAK lolos (detail push error & fix di `PROGRESS.md` §38).
  - **v2** (`rsna-knee-train-coatnet-v2-blended-labels`, label rank-mean blend flight+Claude, 0,9047 label-vs-gold): 0,9170 — REGRESI, LEBIH RENDAH dari v1 & c5, walau labelnya sendiri skor label-matching lebih tinggi dari flight0234 murni. **Root cause**: `rank(pct=True)` memaksa tiap kolom label jadi distribusi uniform, menghapus skew/base-rate asli → target BCE kehilangan sinyal kalibrasi (bagus utk rank-based AUC eval/ensembling prediksi, SALAH utk label target training). Detail `PROGRESS.md` §46.
  - **v3** (`rsna-knee-train-coatnet-v3-rawmean`, label RAW-PROBABILITY-MEAN blend α(flight)=0,70, TANPA transformasi rank): **MACRO AUC gold-58 = 0,9234 — MENANG vs c5 (0,9206), v1 (0,9193), DAN v2 (0,9170). Model tunggal terbaik proyek sejauh ini.** Status Kaggle COMPLETE (selesai training ~4,6 jam, ditemukan & diverifikasi 13 Sep via `kaggle kernels status`+`output` — belum pernah dicek sebelumnya, TIDAK ada di commit manapun). Per-label (58 gold, lemah→kuat): Lateral OA 0,8124, Synovitis 0,8399, Lateral Meniscus 0,8472, PF OA 0,8571, Fracture 0,9319, Contusion 0,9406, Medial Meniscus 0,9675, Effusion 0,9689, Baker's 0,9710, Medial OA 0,9798, MCL 0,9819, ACL 0,9828 — titik lemah (Lat OA/Synovitis/Lat Meniscus/PF OA) konsisten dgn temuan lama (§27a Synovitis, Lateral Meniscus). **Konfirmasi hipotesis §46**: rank-mean memang akar masalah v2; blending label report LLM (flight+Claude) TERBUKTI berguna sbg training signal SELAMA pakai raw-probability-mean, bukan rank-mean. **BELUM disubmit real ke LB** — per pelajaran metodologis berulang (c5-solo §39, v18+v23 §44), kemenangan gold-58 WAJIB dikonfirmasi submission real sebelum dianggap perbaikan nyata/dipakai ganti submission tim 0,936. Checkpoint tersimpan `coatnet_rawmean_ft_v3.pt` (292MB, output Kaggle, belum ditarik ke dataset permanen).
- **Fusion v2 SELESAI (25 Agu malam)**: `blend_results/fusion_nested_cv_v2.py` + `FUSION_V2_RESULTS.md`. Tabel 7-arm OOF gold-57 (pk/e9/e11/v18 + 3 CoAtNet hasil harvest). **c5 (CoAtNet v5_full_swa) = 0.9206 leak-free = baseline lokal terbaik; SEMUA blend kalah**; seleksi per-target honest-nested 0.9148 < c5. Sanity cocok: pk=0.8360, e9=0.8506, e11=0.8285, v18=0.8068.
- **GPU#1 harvest SELESAI PENUH (25 Agu ~21:30, cuma 6,5 menit T4x2!)**: `dimaspashaakrilian/rsna-knee-oof-harvest-coatnet-gold58` — v5_swa **0.9213**, v4_swa 0.9139, v4_full 0.9127 (58/58 gold, fail=0). Output `oof_coatnet_*.csv` tersimpan `/tmp/opencode/harvest/` lokal + output kernel Kaggle. Kuota GPU terpakai cuma ~0,1h.
- **EDA**: SELESAI. `kaggle_kernel/` → `eda_results/`.
- **Audit data (sampel)**: SELESAI, 98.3% bersih. `kaggle_kernel_audit/` → `audit_results/`.
- **Audit data (exhaustive, 820rb file)**: SELESAI 100%, 94.1% bersih. 1 dari 58 studi gold ter-flag "exclude" (`...72853333...`, overexposure sistemik) — **belum direview manual**. `kaggle_kernel_audit_full/` → `audit_results_exhaustive/`.
- **Preprocessing**: SELESAI 100%, 24.371/24.371 series, 0 error. `kaggle_kernel_preprocessing/` (kernel: `rsna-knee-preprocessing-pipeline`) — jadi source-of-truth cache gambar via `kernel_sources`.
- **Weak-labeler rule-based**: SELESAI (v2), agreement 0.747 vs gold — tidak lagi jadi sumber label utama (kalah akurat dari `flight0234` eksternal), tapi kode & hasil tetap tersimpan. `kaggle_kernel_weak_labeler/` → `weak_labeler_results/`.
- **Modeling (single backbone DINOv2-base)** — **SOURCE OF TRUTH SAAT INI**: `kaggle_kernel_modeling/` (kernel aktif: `dimaspashaakrilian/rsna-knee-modeling-v11-fix-oom-grad-checkpoint`).
  - v9 (label hybrid flight0234, EPOCHS=6): val_macro_auc **0.7902**, monoton naik.
  - v10: ERROR CUDA OOM (unfreeze=6 + batch=8 kebesaran).
  - **v11 (fix OOM: gradient checkpointing + batch=4): val_macro_auc 0.8195 — BEST/FINAL saat ini, lompatan terbesar proyek.**
  - v12 (resume checkpoint v11, +4 epoch via self-reference `kernel_sources`): SELESAI, semua 4 epoch tambahan (0.8096/0.7988/0.8016/0.8182) **underperform** vs checkpoint v11 — overfitting klasik (train_loss terus turun, val_auc turun/stagnan). Guard checkpoint bekerja sempurna, checkpoint final **TETAP v11 (0.8195)**, tidak rusak.
  - **Status: TIDAK ADA kernel training yang sedang berjalan.** Kuota GPU overrun (lihat §4). Menunggu arahan strategi berikutnya (lihat §5).
- **Ensemble komunitas 20-model (`pilkwang/rsna-knee-weights`)**: dievaluasi via `oof.npz` resmi (leak-free) → **macro AUC 0.8360** (57 gold kita). Kernel `kaggle_kernel_ensemble/` (`rsna-knee-ensemble-v1-community-20-model`) sempat run tapi hasil pertamanya (0.9954) **bug leakage**, sudah dikoreksi via oof.npz secara offline (bukan re-run kernel). v2 kernel (tambah arm model kita + RadImageNet, blend nested-CV) **belum dibangun**.
- **Submission pipeline**: v1 (`kaggle_kernel_submission/`, `rsna-knee-submission-v1-community-20-model`, reuse kode pilkwang, internet off) sudah SELESAI dibuat & lolos dev-test (3 studi stub, 0 error) — **BELUM PERNAH benar-benar disubmit** ke leaderboard (dipause krn TEMUAN tim sudah punya submission history sendiri, lihat §4).
- **Skor lokal leak-free TERBAIK proyek (termasuk sumber eksternal): 0,8462** — BLEND rank-space (α=0,60 pilkwang/0,40 kita) antara OOF kita sendiri (v18, 0,8068) + OOF resmi komunitas pilkwang (0,8360), dievaluasi 57 studi gold. **Ditemukan TANPA training tambahan**. Robust di rentang α luas (0,30-0,90 semua >0,83).
- **Skor lokal TERBAIK keluarga model KITA SENDIRI (SlotHead): 0,8176** — blend v18(GAP)+v23(SpatialAttnPool), lihat §2 v23. Checkpoint v15 (single-split 0,8304, historis, bukan OOF) & v18/v23 (5-fold OOF) semua tetap model dasar yg dipakai. Skor publik TIM (submission nyata, terpisah, dari fork komunitas bukan model kita) sudah **0.936** — peringkat **#91/2.372 tim**. Skor publik ARM KITA SENDIRI (c5 CoAtNet solo, real submission 26 Agustus) = **0,927**, di bawah ensemble tim (lihat §2/§3, RESEARCH_NOTES §26/§27).
- **Preprocessing v2 (geometri slice) SELESAI PENUH**: `kaggle_kernel_preprocessing_v2/rsna-knee-preprocessing-v2-physical-slice-order`. 24.371/24.371 series (100%), ~72,5 menit, CPU-only. **100% pakai posisi fisik (`ipp`), 0 fallback**. Output 12,94 GB.
- **v15 (modeling, cache v2, TPU) SELESAI — MODEL TERBAIK BARU**: `kaggle_kernel_modeling_v2geom/rsna-knee-modeling-v15-geom-v2-tpu`. val_macro_auc 0,8304, naik dari v14 (0,8159, cache v1) +0,0145 dan checkpoint v11 (0,8195) +0,0109 — SATU-SATUNYA variabel beda dari v14 adalah data source, geometri slice TERBUKTI pengungkit nyata.
- **Modeling v13/v14 (TPU, cache v1) SELESAI**: `kaggle_kernel_modeling_tpu/rsna-knee-modeling-v13-tpu-sanity`. v13 fix bug XLA (grad checkpointing) + temuan operasional (checkpoint lama tidak ketemu via kernel_sources). v14 (12 epoch dari nol, LR decay): 0,8159 — TIDAK melampaui v11, sekarang SUDAH TERLAMPAUI oleh v15.
- **Preprocessing v3 (sampling "3+3+3") SELESAI PENUH**: `kaggle_kernel_preprocessing_v3/rsna-knee-preprocessing-v3-3x3-anchor-sampling`. 3 anchor merata di jendela tengah stack + 2 tetangga fisik/anchor = 9 slice/series, dipasang sbg channel RGB genuine (bukan grayscale-repeat). 24.371/24.371 OK (0 error), 100% `n_slices_cached=9`, 100% `pos_source=ipp`. Output 7,53 GB, ~67,7 menit.
- **v16 (modeling, cache v3, TPU) SELESAI — NEGATIF**: `kaggle_kernel_modeling_v3geom/rsna-knee-modeling-v16-3x3-anchor-tpu`. val_macro_auc **0,7737 (epoch 8/8, masih naik saat berhenti)** — LEBIH BURUK dari v15 (0,8304) & v11 (0,8195). **Diagnosis**: `unfreeze_last=6` = patch embedding DINOv2 tetap beku, layer itu di-pretrain utk warna foto natural berkorelasi — channel R/G/B genuine dari 3 slice fisik berbeda (bukan grayscale-di-repeat identik) menyodorkan pola yg tidak bisa ditafsirkan layer beku itu. **Model final proyek TETAP v15 (0,8304)**, v16 tidak menggantikannya.
- **Preprocessing v4 (laterality canonicalization) SELESAI PENUH**: `kaggle_kernel_preprocessing_v4/rsna-knee-preprocessing-v4-laterality-canon`. Cache 16 slice/series, 224×224 uint8, urutan fisik IPP; coronal/axial di-flip horizontal utk lutut kiri; **sagittal TIDAK di-reverse di cache** (ditangani di load-time modeling v21 via `vol[::-1]` utk kiri). Deteksi laterality geometris tervalidasi 97,4% vs tag `Laterality` asli.
- **v20 (modeling k-fold laterality TPU) SELESAI — OOF 0,8073, TIDAK ADA peningkatan nyata vs v18**: `kaggle_kernel_modeling_kfold_lat/rsna-knee-modeling-v20-kfold-laterality-tpu` (kode mirror v18, cache v4 + laterality canon). Run ~4,2 jam. Per-fold: 0,8247/0,8061/0,9042/0,8156/0,8491 (rata2 0,8400±0,0352). **OOF macro AUC 0,8073 vs v18 0,8068 (+0,0005, praktis nol)** — detail diagnosis §2. Catatan: print log kernel salah label ("v18 OOF") krn baris itu belum di-parameterkan saat copy dari template v18 — nilai 0,8073 tetap benar milik v20.
- **v21 (modeling anatomy-query TPU) — SELESAI, OOF 0,6904, REGRESI, line ditutup sementara**: `kaggle_kernel_modeling_v5_anatomyquery/rsna-knee-modeling-v21-anatomy-query-tpu`. Arsitektur AnatomyQueryHead + AnatomyModel (detail §2 arah baru), dibangun terinspirasi resep prvsiyan V19. Dataset = `flight0234/rsna-knee-hybrid-report-labels` + kernel sources preprocessing-v4 + weak-labeler; TPU enable. 5-fold CV, EPOCHS=8, BATCH=4, LR head 1e-3/backbone 8e-6. Push-1 ERROR (`DATA_OK=False` false-positif, ambang diagnostik lama tidak sesuai skala [0,1] — fixed, commit `e9db74c`). **Push-2 (fixed) selesai penuh ~5,2 jam**: per-fold 0,6276/0,8233/0,8902/0,6681/0,6199 (rata2 0,7258±0,1102), **OOF gabungan 57/57 gold = 0,6904** — di bawah v18 (0,8068) & v20 (0,8073). **Diagnosis: BIMODAL bukan random** — fold 2&3 (loss turun tajam, auc naik ke 0,82-0,89) menunjukkan arsitektur BISA bekerja baik; fold 1,4,5 (loss nyaris statis, auc stuck di sekitar acak) menunjukkan training TIDAK STABIL lintas seed/fold — kandidat penyebab: cold-start init (`query`/`cls_w` std=0,02, `cls_b`=nol, bukan bias ke label-prior) dikombinasi permukaan optimisasi lebih besar (patch_embeddings baru di-unfreeze). Checkpoint v21 TIDAK dipakai. Kuota GPU+TPU habis pasca run ini (§4) — perbaikan/re-run ditunda sampai kuota refresh.
- **Preprocessing v5 (crop lebar 336px) SELESAI (status COMPLETE)**: `kaggle_kernel_preprocessing_v5_widecrop/rsna-knee-preprocessing-v5-widecrop-336`. Diadopsi dari temuan §25 (dreaddevelopment 0,924 & mattiaangeli): `IMG_SIZE` 224→336, `CROP_MM` eksplisit 140mm, `N_SLICES` 16→8 (kompensasi storage). Fitur v4 (laterality canon, physical ordering) dipertahankan utuh. Selesai tanpa error terlapor (status Kaggle COMPLETE, bukan ERROR) — detail isi manifest (coverage %, storage GB aktual) belum diverifikasi manual (download log sempat macet krn listing ribuan file output, tidak diulang krn status COMPLETE sudah sinyal kuat notebook jalan sampai akhir tanpa exception, sama seperti v2/v4 dulu). Kalau v22 (di bawah) nanti nunjuk DATA_OK=False atau mask kosong, itu sinyal utk balik cek manifest v5 lebih detail.
- **v22 (modeling widecrop+triplet TPU) SELESAI — REGRESI BESAR, LINE DITUTUP**: `kaggle_kernel_modeling_v22_fulltune/rsna-knee-modeling-v22-widecrop-triplet-tpu`. **OOF macro AUC 0,6713** (N=46 gold, hanya 4/5 fold selesai — kena safety time-limit ~7,1h TPU; 336px+triplet memperberat runtime drastis). Per-fold avg 0,6878±0,0477 vs v18 0,8068 / v20 0,8073. Pola konsisten ke-3 (setelah v16/v21): **SlotHead mean-pooling tidak punya jalur utk manfaatkan triplet-RGB + patch-embed-unfreeze** — kombinasi itu butuh head attention-pooling (model lixin73/mattiaangeli). STOP modifikasi data-side di SlotHead; arm sendiri yg kompetitif = tiru arsitektur lixin73 (DINOv2-Small SlotHead attention, 336px/crop130mm) yg terbukti jadi anggota w=0,10 di fork publik.

## 4. Constraint & Batasan Aktif
- **Eksekusi ML HANYA di Kaggle** — sandbox lokal tidak punya `sklearn`/`torch`, disk ~4.7GB free. Semua verifikasi lokal pakai AUC manual (numpy rank-based, Mann-Whitney U).
- **GPU wajib T4 (T4x2), HINDARI P100** (incompatible build PyTorch Kaggle) — set `machine_shape: "NvidiaTeslaT4"` di tiap `kernel-metadata.json`.
- **Kuota & slot (diperbarui 4 Okt 2026)**: GPU 30 jam/minggu, TPU 20 jam/minggu, refresh **Minggu 00:00 UTC** (= 07:00 WIB; berikut **2026-10-10**, lalu 17 Okt). Batas sesi akun: GPU 2 / TPU 1 / CPU 5. Akun DIBAGI dgn proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`) yg ikut menghabiskan kuota & slot — JANGAN sentuh kernel/berkas milik proyek lain. Cek `kaggle quota` tiap sesi (baca kolom `remaining`, BUKAN `total`). Kuota terpakai real-time (±1 jam per jam sesi jalan); submission kompetisi TIDAK memakai kuota. **ATURAN USER (4-5 Okt): urutan komputasi = GPU Kaggle selama ada slot & kuota -> bila habis, GPU COLAB lewat `colab` CLI (hasil dikirim balik ke Kaggle sbg dataset utk submission) -> TPU Kaggle terakhir** (builder TPU: `build_train_coatnet_*_tpu.py`, `build_train_coatnet_imagenet.py tpu <ep> <seed>`).
- **CATATAN OPERASIONAL**: kaggle CLI kadang tidak ada di `$PATH` shell baru (`~/.local/bin` tidak otomatis masuk) — kalau `kaggle: command not found`, jalankan `export PATH="$HOME/.local/bin:$PATH"` dulu di awal command Bash. **Kredensial `~/.kaggle/kaggle.json` (akun dimaspashaakrilian) SUDAH ADA & valid** (permission 600), tapi di sandbox container yg benar2 baru, binary `kaggle`/`pip` bisa TIDAK terpasang sama sekali (bukan cuma soal PATH) — Python image minimal tanpa `pip`/`ensurepip`. Fix tervalidasi (13 Sep): `sudo apt-get install -y python3-pip pipx` lalu `pipx install kaggle` (jangan `pip install --user`, kena PEP 668 externally-managed-environment). Utk analisis lokal (pandas/numpy dipakai script `label_engineering/*.py`) kalau belum ada: `sudo apt-get install -y python3-pandas python3-numpy`. Dgn kaggle CLI lokal aktif, status/output kernel Kaggle bisa langsung dicek dari terminal (`kaggle kernels status <slug>`, `kaggle kernels output <slug> -p <dir>`) tanpa buka browser — dipakai utk menemukan hasil v3 (§3) yg sudah COMPLETE tapi belum pernah ditarik.
- **TPU sudah diverifikasi bisa dipakai** (probe `rsna-knee-tpu-diagnostic-torch-xla-check`, 18 Agustus): `torch_xla` 2.8.0 pre-installed di image Kaggle, `xm.xla_device()` + matmul + `transformers` import semua berhasil. Antrian nyata terukur ~28,6 menit dari push sampai `RUNNING` (bervariasi, kadang jauh lebih cepat). **Caveat**: kode training kita (CUDA-based) HARUS ditulis ulang pakai API `torch_xla` (`xm.xla_device()`, `xm.mark_step()`) — bukan cuma toggle `enable_tpu`; idealnya juga di-pad ke shape tetap dulu (XLA re-compile mahal kalau shape variabel) — biaya rekayasa nyata, belum dikerjakan.
- **Leaderboard sungguhan (snapshot 25 Agustus 2026)**: #1 Brandon Low (zxkyouma) **0.952**. Cutoff: #10=0.945, #20=0.943, #50=0.937. **Tim kita (Krazy KIller, #91/2.372) di 0.936** — persis ceiling notebook publik (DINOsaur V10). Kompetisi ~2,5 bulan jalan (deadline final 22 Okt).
- **Submission harian dibatasi 5x/hari UNTUK SELURUH TIM** (bukan per-orang) — tim punya kolaborator aktif (`nxtirfan`/Muhammad Irfan) yang juga submit; skor terbaik tim **0.936** (#91, per 25 Agustus 2026). → **Jangan submit sembarangan, koordinasi dulu supaya tidak boros kuota harian tim.**
- **API LLM hosted**: BOLEH dipakai (ruling resmi host topic 733965), syarat layanan "reasonably accessible & minimal cost" ke semua peserta (Rules §2.6.b) — **status: SUDAH dikonfirmasi resmi, bukan lagi perlu verifikasi**. Implementasi (relabeling 4.349 studi non-gold via LLM) masih ditunda, prioritas saat ini di single-model strengthening.
- **Lisensi NonCommercial (CC-BY-NC)** — di-skip sbg kebijakan proyek (risiko lisensor asli, bukan larangan Rules kompetisi langsung).
- **Repo GitHub PRIVATE**, remote `https://github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection.git`, branch `master`. **Ada kolaborator lain dgn push access** (`nxtirfan`/Muhammad Irfan, sudah push commit README sendiri) — cek `git pull`/konflik sebelum push besar.
- **Commit HARUS TANPA trailer Co-Authored-By Claude** — aturan mutlak, berlaku semua commit di repo ini. Pesan commit multi-baris/berisi tanda kutip WAJIB pakai `git commit -F <tempfile>`, jangan inline `-m "..."` (pernah menyebabkan shell-quoting mishap).
- **GOTCHA BARU (13 Sep) — log run submission REAL competition code TIDAK PERNAH bisa ditarik via API**: `kaggle kernels output <slug>` (dgn/tanpa suffix `/<versi>` eksplisit) SELALU cuma mengembalikan hasil "Save & Run All" biasa (test.csv publik, 3 studi stub) walau kernel version yg sama SUDAH pernah dipakai submission sungguhan (1.300+ studi asli) — dicoba 3 cara, semua sama, kesimpulan: Kaggle sengaja tidak expose log/output run submission demi cegah kebocoran test set. **Implikasi**: kalau butuh tahu APAKAH suatu cell/arm benar2 jalan penuh saat submission real (bukan cuma dev-test), TIDAK ADA cara pasti dari API — cuma skor akhir yg terlihat. Rencanakan eksperimen submission dgn asumsi ini (mis. desain self-gating/fail-safe SEBELUM submit, bukan berharap bisa debug sesudahnya).

- **Baca log kernel yang MASIH BERJALAN (ditemukan 4 Okt)**: `kaggle kernels output` mengembalikan KOSONG selama RUNNING (log baru ada setelah selesai). Yang BERHASIL: buka `https://www.kaggle.com/code/dimaspashaakrilian/<slug>` di Chrome yg sudah login (alat claude-in-chrome) → tab **Logs** menampilkan tabel log langsung (kadang perlu klik/tunggu sebentar), lalu `get_page_text`. Dipakai utk melihat loss & diagnostik gold per epoch dan memutuskan stop dini. Log run submission real tetap TIDAK terbaca (lihat butir di atas).

## 5. Open Questions / Menunggu Keputusan User
- **RELABELING TANPA API KEY TERPISAH — SCALE-UP KE 4.349 STUDI SELESAI 100% (28 Agu–2 Sep)**: user menunjukkan sesi Claude Code ini SUDAH jalan di atas API Anthropic, jadi TIDAK perlu API key terpisah — Claude bisa baca teks `Report` (9 bahasa) & hasilkan label sendiri, SAMA PERSIS semangat ruling 733965. Pilot 58 studi gold (blind read): macro AUC Claude sendirian = 0,8953; blend rank-mean flight0234+Claude = 0,9047 @ α(flight)=0,6 (`label_engineering/claude_labels_pilot_58gold.csv`).
  **Scale-up SELESAI**: 20 chunk (~218 studi/chunk) diproses via subagent paralel (3 per gelombang, general-purpose), berulang kali kena rate-limit account-level (429, reset time bervariasi tiap kejadian) DAN 2x scratchpad wipe total — **sistem checkpoint yg dibangun (per instruksi eksplisit user "tambahkan sistem checkpoint agar tidak menjalankan ulang dari awal") TERBUKTI menyelamatkan semua progres**: input chunks di-commit ke `label_engineering/relabel_chunks_input/` (persisten, bukan scratchpad), tiap chunk output ditulis LANGSUNG ke `label_engineering/relabel_out_backup/` (path repo, bukan scratchpad) dan di-commit+push SEGERA begitu 1 agent selesai (bukan nunggu 1 gelombang penuh) — beberapa kali agent gagal di tengah (429 atau disconnect) tapi baris yg sudah tertulis SELALU selamat krn ditulis inkremental langsung ke path repo; sisa baris dihitung via diff UID lalu di-resume pakai agent kecil utk "remainder" saja. `label_engineering/relabel_status.py` dipakai tiap siklus utk cek chunk mana yg BELUM lengkap.
  **Hasil akhir**: `label_engineering/claude_labels_nongold_full.csv` = **4.349 studi unik, 100% coverage** dari scope asli 4.351 baris chunk input (2 baris di antaranya ternyata pecahan korup 1 laporan yg sama akibat bug quoting CSV di `reference_metadata/train.csv` asli — sudah direkonstruksi manual via cross-reference, bukan data hilang).
  **Blend full-scale (`label_engineering/blend_fullscale.py`)**: re-validasi rank-mean di gold-58 **mereproduksi persis hasil pilot** (macro=0,9047 @ α(flight)=0,60 — alpha sweep lengkap 0,00-1,00 dicoba, ini puncak globalnya). Korelasi raw-value Claude vs flight0234 di 4.349 studi non-gold: **0,78-0,98 per label** (Fracture terendah 0,785, Medial Meniscus tertinggi 0,975) — sinyal kuat kedua sumber LLM independen menangkap pola nyata dari teks laporan, cukup independen utk saling melengkapi. Tersimpan: `label_engineering/blended_labels_fullscale_train.csv` (4.349 baris, rank-mean α=0,60, siap dipakai sbg label training baru).
  **TERJAWAB 13 Sep sore — LABEL ENGINEERING TERBUKTI BEKERJA DI LB NYATA**: `v3 SOLO = 0,932` vs `c5 SOLO = 0,927` (harness/K_EVAL=42/struktur identik, satu-satunya variabel = bobot checkpoint) → **fine-tune di atas label blend flight0234+Claude memberi +0,005 macro AUC di leaderboard NYATA**, bahkan lebih besar dari selisih di gold-58 (+0,0028). Detail PROGRESS §49. Tindak lanjut berjalan: `rsna-knee-submit-v3-inarm` (v3 masuk ke DALAM daftar ARMS pipeline publik dgn bobot 0,35 — terbesar, di atas c5 0,30).
  **Percobaan pertama (hybrid tempel-luar α=0,15) — HASIL: NULL (0,941 = 0,941)**: kernel `dimaspashaakrilian/rsna-knee-submit-v3-hybrid` (fork `bend-the-knee-to-the-dinosaurs-7d6471` 0,941 + cell sisip arm v3 α=0,15, self-gated/fail-safe) dibangun, disubmit real (56196772, 13 Sep). **Skor real = 0,941, byte-identik dgn base TANPA v3.** Ditemukan keterbatasan operasional PENTING: **Kaggle TIDAK PERNAH mengekspos log run submission sungguhan via API** (`kaggle kernels output` cuma kembalikan dev-test stub 3-studi, dicoba 3 cara beda, semua sama) — jadi TIDAK BISA dipastikan apakah arm v3 (a) di-skip total oleh self-gating waktu (base pipeline diduga sudah dekat batas 9 jam), atau (b) jalan tapi α=0,15 washed-out di ensemble yg sudah berisi 4+ CoAtNet lain (selaras temuan lama "LB desimal ke-3 sering tidak resolvable"). Detail penuh PROGRESS.md §48. **Open question SEKARANG**: submit v3 SENDIRIAN (langkah original §28d yg dilewati krn user minta langsung gabung) utk dapat titik data independen ttg lift v3 yg terlepas dari masalah integrasi — kuota tim tersisa hari ini perlu dicek ulang dulu sebelum lanjut eksperimen (sudah pakai 1 dari 5 hari ini per submission di atas).
- **Submit c5 SENDIRIAN — SELESAI, HASIL PENTING: real LB (0,927) MENOLAK temuan pivot lokal**: kernel `kaggle_kernel_submit_c5_solo/`, submission 55795592, **public LB = 0,927**. Dibanding: local gold-58 harvest 0,9206-0,9213 (offset ke LB +0,006-0,007, kecil & wajar spt pola dreaddevelopment); TAPI **skor tim eksisting (ensemble besar `rsna-base`/DINOsaur fork) = 0,936, LEBIH TINGGI 0,009 dari c5 sendirian**. **INI BERTENTANGAN dgn kesimpulan pivot 25 Agustus** ("c5 sendirian mengalahkan SEMUA blend" — itu valid utk nested-CV LOKAL N=57 gold, TAPI TIDAK bertahan di real leaderboard N~1.300 studi tersembunyi). **Pelajaran metodologis penting**: nested-CV di N=57 kemungkinan TERLALU KECIL utk mendeteksi manfaat diversity ensemble yg baru kelihatan di skala real test set — selaras temuan independen sadamtorres (§23a poin 7: "diversity tetap bernilai walau model sangat mirip, ensemble +0,006" walau korelasi antar-model tinggi). **Keputusan revisi**: JANGAN ganti submission tim dari 0,936 ke c5 sendirian — ensemble besar TETAP submission terbaik utk saat ini. Pertimbangkan integrasi c5/arm kita SBG TAMBAHAN ke ensemble yg ada (bukan pengganti) drpd andalkan single-arm. Gotcha operasional (utk referensi push berikutnya): title kernel dgn tanda kurung bikin 409 saat CREATE pertama; `kaggle competitions submit -k/-v` WAJIB disertai `-f submission.csv` jg. Kuota submission tim: 1 tersisa hari ini.
- **Koordinasi dgn nxtirfan** soal kuota submission 5x/hari tim — siapa submit apa; jangan boros (tim sudah di 0.936 via jalur kolaborator).
- **Review manual** studi gold `...72853333...` yang ter-flag "exclude" di audit exhaustive (kemungkinan anomali kalibrasi/overexposure) — apakah tetap dipakai di training/validasi atau dikeluarkan?

## 6. Referensi Cepat
- Kompetisi: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- **Kernel aktif (source of truth modeling, checkpoint operasional 0,8304 single-split / ~0,8068 OOF)**: `dimaspashaakrilian/rsna-knee-modeling-v15-geom-v2-tpu` — https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-modeling-v15-geom-v2-tpu
- **Kernel v22 (widecrop+triplet TPU, SELESAI — OOF 0,6713 REGRESI, line ditutup)**: `dimaspashaakrilian/rsna-knee-modeling-v22-widecrop-triplet-tpu`
- **Notebook publik terbaik (proven 0.936, basis replika)**: `prvsiyan/head-and-shoulders-knees-and-toes` (port DINOsaur V10 romantamrazov) — https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes
- **Bundel asset stack 0.936**: `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`; checkpoint CoAtNet: `dreaddevelopment/raptor-knee-maxspan`; arm DINOv2-Small SlotHead: `lixin73/rsna-knee-train-checkpoints`
- **Kernel preprocessing v5 (crop 336px/140mm, SELESAI)**: `dimaspashaakrilian/rsna-knee-preprocessing-v5-widecrop-336`
- **Kernel v21 (anatomy-query TPU, SELESAI — OOF 0,6904, regresi, tidak dipakai)**: `dimaspashaakrilian/rsna-knee-modeling-v21-anatomy-query-tpu`
- **Kernel v20 (k-fold laterality TPU, SELESAI — OOF 0,8073, tidak ada peningkatan nyata vs v18)**: `dimaspashaakrilian/rsna-knee-modeling-v20-kfold-laterality-tpu`
- Kernel k-fold CV (5 checkpoint, OOF 0,8068, kandidat ensemble): `dimaspashaakrilian/rsna-knee-modeling-v18-kfold-5-tpu`
- Kernel submission v2 (ensemble 5-fold, tervalidasi dev-test, BELUM disubmit sungguhan): `dimaspashaakrilian/rsna-knee-submission-v2-own-5fold-ensemble`
- Kernel modeling v16 (eksperimen sampling 3+3+3, sedang berjalan): `dimaspashaakrilian/rsna-knee-modeling-v16-3x3-anchor-tpu`
- Kernel preprocessing v2 (urutan fisik, dipakai v15): `dimaspashaakrilian/rsna-knee-preprocessing-v2-physical-slice-order`
- Kernel preprocessing v3 (sampling 3+3+3, dipakai v16): `dimaspashaakrilian/rsna-knee-preprocessing-v3-3x3-anchor-sampling`
- Kernel preprocessing v4 (laterality canon, cache 16 slice, dipakai v20/v21): `dimaspashaakrilian/rsna-knee-preprocessing-v4-laterality-canon`
- Kernel modeling lama (v11, 0,8195, GPU, superseded oleh v15): `dimaspashaakrilian/rsna-knee-modeling-v11-fix-oom-grad-checkpoint` — https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-modeling-v11-fix-oom-grad-checkpoint
- Kernel preprocessing: `dimaspashaakrilian/rsna-knee-preprocessing-pipeline`
- Kernel weak-labeler: `dimaspashaakrilian/rsna-knee-weak-labeler-from-reports`
- Kernel ensemble komunitas: `dimaspashaakrilian/rsna-knee-ensemble-v1-community-20-model`
- Kernel submission (belum disubmit): `dimaspashaakrilian/rsna-knee-submission-v1-community-20-model`
- Kernel EDA/audit: `rsna-knee-abnormality-detection-eda`, `rsna-knee-dicom-quality-audit`, `rsna-knee-dicom-full-exhaustive-audit`
- Dataset label eksternal dipakai: `flight0234/rsna-knee-hybrid-report-labels` (CC0, 0.8991 vs gold)
- Dataset ensemble komunitas: `pilkwang/rsna-knee-weights` (CC0, 20-model DINOv2-small, termasuk `manifest.json` + `oof.npz` resmi)
- Backbone: `metaresearch/dinov2` (Apache-2.0, Kaggle Model resmi)
- Riset arsitektur & dataset lengkap: `docs/RESEARCH_NOTES.md` (§0–§15)
- Progress kronologis lengkap: `PROGRESS.md`
- Discussion penting: topic 733965 (ruling LLM API), topic 735304 (single-model score), topic 735154 ("3+3+3" slice sampling), topic 735121 (isu lisensi NC RadImageNet, belum dijawab host), topic 735826 (bottleneck: geometri+label > backbone size), topic 735767 (final ceiling + inkonsistensi gold label), topic 735854 (PSA: codec DICOM offline + path submission), topic 735639 (2 lutut 1 DICOM), topic 733932 (stevenleehans: "Knee Report LLM Labels" + pembahasan label), topic 736268 (noise public notebooks; DINOv3 = `facebook/dinov3`; adaptive per-finding weights dari romantamrazov), topic 735826 (bottleneck label; contoh kontradiksi gold label k256.dev)
- Kernel resep 0,920 tim (dibedah forensik, bukan dipakai langsung): `tonylica/rsna-knee-dino-radimagenet-rank-ensemble` — https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble

## 10 Okt 2026 (malam ~01:00–03:00 WIB) — izin + riset mendalam perak
- **Izin multi-akun panitia TERARSIP**: `docs/IZIN_AKUN_KEDUA.md` (teks verbatim "Oh, of course, that's allowed." + konteks + kronologi 10 submisi tim dgn akun masing-masing). Checklist MEMORY §5.4, laporan §6, HANDOFF, push.md ditandai selesai. Komit: 9d21873, 9906eb4.
- **Riset mendalam 3 subriset paralel** (deep-research skill) → `research_notes/Cara menembus 0951 perak/` (forensik_0951.md, inference_gratis_0001.md, arm_baru_realistis.md) + sintesis `reports/Cara menembus 0951 perak.md`:
  - Papan 9 Okt 18:25 UTC (5.606 tim): ≥0,951=296, ≥0,952=183; blok 0,951-persis 113 tim > garis perak 279 → **0,951 lotre, 0,952 aman**.
  - **PEMISAH 0,951+ = PSEUDO-LABELING** (735304): colum2131 LB 0,958 (single ConvNeXt-T 0,953 pseudo, no external), azamat1ch LB 0,960 (5-fold 0,959), hengck "model is not the key. it is your labels", Raymond Yuen 0,949→0,958.
  - Heliosli S6 = MATI PERMANEN (kode `find_mount` manifest tanpa fallback; bundle `209a32c8` + side-model `ab623e2b` hilang; fold0 publik tanpa manifest) — rekonstruksi JANGAN. Rabari klaim "0,951+" = 0,950. xianhan "0.957" = gold-58 lokal terkontaminasi (tim 0,950).
  - nartaa rilis cuma 2 ckpt (r384 0,949 + c08_r224 0,945) → SWA manual/seed-avg tertutup. `raptor-knee-arms`/`raptor-cnn336` = dreaddevelopment (arm lemah, kemungkinan bocor gold). Pola placebo baru: d5e-goldw (bobot fit di emas), d5f (FT on-the-fly label-LLM).
  - Rencana: **Fase A** harvest mirror H1–H3 (gratis, gate Δ≥0,005+CI95) → **Fase B** pseudo-label self-training 4.407 train (10–25 GPU-jam, muat sblm 15 Okt).
- Submisi 57015077 (fork rabari) = 0,950 → BUANG (monitor selesai). MaVIT eval gratis = 0,9202 DITOLAK. Forum 743416 host masih diam.
- Kuota: reset 10 Okt 07:00 WIB (~90 jam GPU total 3 akun + ~54 jam TPU); submisi 9 Okt 3/5 terpakai.

## Protokol Update (WAJIB diikuti agent apa pun yang bekerja di repo ini)

Update MEMORY.md SEGERA setelah:
1. Keputusan arsitektur/model baru dibuat/diubah → update bagian 2
2. Status pipeline berubah (mulai/selesai/gagal) → update bagian 3
3. Menemukan constraint/limitasi baru (misal GPU, quota, aturan kompetisi)
   → update bagian 4
4. Ada pertanyaan yang butuh keputusan user → tambah ke bagian 5, 
   HAPUS dari bagian 5 setelah user menjawab (jawabannya pindah ke 
   bagian 2/3/4 yang relevan)
5. Selesai satu unit kerja besar → update "Last updated" timestamp

ATURAN PENULISAN:
- Overwrite, jangan akumulasi — MEMORY.md harus selalu bisa dibaca 
  penuh dalam <2 menit
- Detail panjang/historis tetap masuk PROGRESS.md, BUKAN MEMORY.md
- Setiap update MEMORY.md, commit terpisah dengan pesan: 
  "memory: <ringkasan perubahan>"


## 52b. ARSIP MEMORY.md — versi penuh sebelum dipadatkan (10 Okt 2026 ~21:00 WIB)
> Dipadatkan oleh sesi Claude Code. Isi di bawah = salinan apa adanya (kecuali judul).

# MEMORY — RSNA Knee Abnormality Detection
> Living memory — PADAT (baca <2 menit). Serah-terima operasional (langkah berikut + perintah) → **HANDOFF.md**.
> Riset → docs/RESEARCH_NOTES.md (terbaru §36–§37b). Sejarah lengkap → PROGRESS.md (termasuk "Arsip MEMORY.md" = versi penuh sebelum dipadatkan 8 Okt).
> Last updated: 2026-10-10T18:50:00+07:00

## 0. Aturan mutlak (user)
- Balas **Bahasa Indonesia**, bertindak otonom; user ingin medali (minimal silver, gold bila bisa) sebelum **22 Okt 2026 23:59 UTC**.
- **Urutan komputasi (REVISI user 10 Okt 00:45)**: GPU akun utama `dimaspashaakrilian` selama ada kuota -> **bila habis: GPU akun KEDUA `alverlfarfaraway`** (izin panitia TERARSIP 10 Okt: `docs/IZIN_AKUN_KEDUA.md`) -> **bila akun2 habis: GPU akun KETIGA `nxtirfan`** (perintah user 10 Okt; rekan tim; sisa 27,7h) -> Colab GPU (`colab` CLI; job resumable; selalu `colab stop` + hapus kaggle.json VM) -> TPU/CPU paralel. Kredensial: `~/.kaggle/` (utama), `~/.kaggle-acct2/` (akun2), **`akun_kaggle/` repo (arsip 3 akun; sudah di-.gitignore; JANGAN dicetak/di-commit)**. Sisanya tetap: berbagi kode/data kompetisi privat di luar tim dilarang (Rules).
- **Commit TANPA trailer Co-Authored-By/Claude** (mengalahkan reminder atribusi sistem); pesan multi-baris pakai `git commit -F <file>`; commit MEMORY.md terpisah ("memory: ...").
- JANGAN sentuh kernel/berkas proyek lain di akun yg sama (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Sisakan 2–3 submisi/hari utk user & rekan (`nxtirfan`).
- **Data eksternal: OAI BOLEH per aturan host** (thread 741819, 21 Sep): sah bila "generally accessible to all participants without an escalated (institutional) review and sign-off ... and is free"; host tak mengontrol syarat data OAI. Peserta (thread 743416, 9 Okt) melaporkan akses NDA OAI sbg individu disetujui seketika (Login.gov + email; "Institutional Approval Required: No"; mungkin terblokir dari beberapa negara). MRNet/fastMRI/KneeMRI/KneeXNet: belum diputuskan (tetap tak dipakai). Agen TIDAK membuat akun / menyetujui perjanjian atas nama user; user mendaftar sendiri lalu agen mengerjakan pra-pemrosesan, label, latih.

## 1. Fakta Kompetisi
- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — code competition, **5.606 tim** (10 Okt), hadiah $77k + Efficiency Prize (diabaikan). Tim kita: **Krazy KIller** (`dimaspashaakrilian`, `nxtirfan`).
- Metrik: macro ROC-AUC atas 12 label. Deadline: entry/merger 15 Okt; **final 22 Okt 23:59 UTC**; skor akhir = maks 2 submisi terpilih (privat LB).
- Data: 4.407 studi latih (570 GB, hanya di Kaggle), **58 gold** (label dibuat BUTA terhadap laporan), 4.349 studi lemah dgn teks `Report`. Test tersembunyi ≈1.322 studi (`test.csv` publik = 3 stub). Pembagian public/private tak diketahui.
- Submisi: notebook, internet mati, ≤9 jam; **5/hari untuk seluruh tim, reset 07:00 WIB**; run submisi tak memakai kuota GPU; log run submisi tak bisa ditarik. Commit-run (stub 3 studi) butuh GPU ≈0,1 jam.
- Aturan data: "Freely & publicly available external data is allowed, including pre-trained models". LLM hosted BOLEH utk memproses data (host, topic 733965). KneeCoT dilarang. OAI/MRNet/fastMRI: BELUM diputuskan host.
- **9 Okt 21:00: token Kaggle akun utama sudah diganti user (kaggle.json valid, CLI normal; salinan di folder repo dihapus). Papan 10 Okt 01:25 WIB (9 Okt 18:25 UTC, 5.606 tim): perak ≤279; >=0,951: **296 tim**, >=0,952: 183; blok 0,951-persis = 113 tim (rank 184–296) MELEBIHI kapasitas perak → **0,951 = LOTRE, 0,952 = TARGET AMAN**. KITA @0,950 = di dataran 0,950.**
- **Papan 8 Okt (5.490 tim)**: #1 0,964; emas (rank 20) 0,959; perak (rank 274) 0,950; perunggu (rank 549) 0,947. **Tim kita 0,950 = rank 292** (dataran 0,950 = rank 146–397). Peta skor→rank: 0,951≈134 | 0,952≈112 | 0,953≈91 | 0,954≈77 | 0,956≈54 | 0,958≈32 | 0,960≈16. Garis BERGERAK cepat (perak 0,945 pada 3 Okt → 0,950 pada 8 Okt); proyeksi 22 Okt: perak 0,951–0,953, emas 0,961–0,963.

## 2. Keputusan & temuan terkunci
- **ARAH TERKINI (9 Okt 23:40)**: basis = Apex **0,950** (56943646). **Probe r224 (57012143, 9 Okt 21:39) = 0,950 → BUANG** (varian r224/weight-tweak = noise, sesuai riset — TUTUP). **Arm `heliosli` blend-gold (0,951) = BLOKIR**: dataset `heliosli/rsna-knee-compact-v1` di-RETIRE 9 Okt 13:50 UTC ("files not accessible") + `rsna-knee-ofl-ssl-weights` 403 → pipeline OF-nya tak bisa dijalankan siapa pun (fork `evgendvorkin/rsna-versia-7`, `samanyu1808/blend3` ikut mati). **Fork `rabari9999` dual-engine (57015077) = 0,950 → BUANG** (10 Okt 00:45; klaim "0,951+" tak terbukti; MIRROR_TTA + r224@0,3 tak mengangkat) → **keluarga r384+reader+r224 = plafon 0,950, TUTUP**. **MaVIT `hengck23` terevaluasi (10 Okt 00:56; eval GRATIS via commit-run `MODE=local` → `alverlfarfaraway/mavit-288-gold58-eval`)**: gold-58 **0,9202** < induk .9214 → **DITOLAK** (tak buang submisi); `lixin73/mavit-336px` tanpa kernel publik. **Thread 743416 (cek 10 Okt via `kaggle forums topics show`): MASIH BELUM DIJAWAB host** (konsensus peserta: host menghindar selain KneeCoT; OAI dianggap "reasonable" oleh Tucker/YYama, Satwik masih hati-hati) → keputusan 2 final tetap 21–22 Okt. **Resep forum 746792 (Danial Zakaria = nartaa, writeup 0,949) SUDAH DIBACA & SUDAH DIREPLIKASI**: native-crop 320 no-resize + mirror plane-aware (sagittal balik 3 channel / cor+ax flip lebar) + SWA{14,11,12} = persis stack kita → **tak ada celah baru**; naive all-plane flip TERBUKTI merugikan (0,940→0,939). **KONFIRMASI OAI**: file ckpt induk & r224 = `raptor_ft_alldata_t16_blendjev_oai_d96_{r384,c08_r224}_swa.pt` + 746792 ("both models used 2,399 OAI knees") → leg kita OAI-terkonfirmasi; jalur bersih mentok 0,944, jalur OAI sudah dipakai di skor 0,950 (risiko §5.1, keputusan final 21–22 Okt). Forum 745861 (hengck): label Claude solo 0,942 — path LLM-label di bawah plafon. **RISET MENDALAM 10 Okt (3 subriset → `reports/Cara menembus 0951 perak.md` + `research_notes/Cara menembus 0951 perak/`)**: pemisah 0,951+ = **PSEUDO-LABELING** (735304, 9 Okt: colum2131 LB 0,958 — single ConvNeXt-T 0,953 dgn pseudo, "no external data"; azamat1ch LB 0,960 — single 5-fold 0,959; hengck +5 "model is not the key. it is your labels"; Raymond Yuen pseudo 0,949→0,958). Rencana: **Fase A** harvest mirror gratis (H1 r224+mirror, H2 reader+mirror, H3 mirror sbg arm; commit-run `MODE=local`; sah bila Δgold ≥+0,005 & bootstrap CI95>0) → **Fase B pseudo-label self-training di 4.407 train** (≈10–25 GPU-jam, muat sblm 15 Okt; bahan: cache volcache-z + ckpt induk + OOF machinery). Tertutup: SWA manual/seed-avg (nartaa rilis cuma 2 ckpt), d5e-goldw & d5f (placebo), blending tanpa OAI (0,944), rekonstruksi heliosli fold0.
- **Mengapa mentok**: tangga nartaa — resep tanpa OAI ≈0,942; +contoh OAI +0,005…+0,008 → 0,949. Probe goodpjw di atas 0,949: stack 0,943 @40% → 0,949; reader sendiri @30% → 0,948. Keunggulan = PENGETAHUAN (data), bukan varians → bagging model tanpa OAI tak menolong. Atas-14 (Tucker Arrants): model tunggal ~0,951 lalu ensembel banyak → 0,959+.
- **Penilaian jujur**: emas ≈0–2% (butuh ≥~0,962), perak ~35–50%, perunggu ~95%. Model OAI buatan sendiri TIDAK layak (akun NDA hanya bisa dibuat user, unduhan ratusan GB, pemetaan label OAI→12 temuan tak diketahui, infrastruktur kita kecil).
- **Label**: gold-58 tak pernah ditimpa. Label fusi r1 (guru+OOF murid, w=0,5) selaras gold 0,9348 (vs LLM ~0,90): `dimaspashaakrilian/rsna-knee-fused-labels-r1`. Efek: fine-tune c5 0,927 → r2c5 0,933 (solo). Di dalam stack publik efeknya netral.
- **Terukur (LB solo)**: c5 0,927; v3 0,932; r2c5 0,933; 4 murid fold 0,934 (CoAtNet jenuh ~0,934). Leg luar lemah (murid, RadImageNet, fleet) pada stack 0,943 → 0,942 (MENURUNKAN). Hanya keluarga independen kuat (~0,93, reader goodpjw 30%) yg menambah (+0,001) pada stack 0,943 → 0,944.
- **DITUTUP (jangan ulangi)**: KneeNet reader sendiri (fold 0 TPU diverge ep5; plafon 0,93 tak menambah di atas 0,949); CoAtNet/ResNet init-ImageNet lambat; model soup (init sama); bagging tanpa OAI; metadata DICOM (tanpa usia, jenis kelamin lemah, PatientID unik); resolusi 288; bobot global v3.
- Gold-58: utk stack publik terkontaminasi; utk ckpt induk nartaa TIDAK jelas dilatih pada gold — AUC induk pada cache z (384 penuh, 94 window, 1 tampilan, fp16) = **0,9214** (terlihat out-of-sample, hanya dipakai utk seleksi epoch) → boleh jadi pembanding RELATIF anak vs induk (SE ≈0,01; sinyal kasar). Keputusan besar HANYA dari submisi nyata; LB 3 desimal, noise antar-latih ≈0,002.

## 3. Status (10 Okt 2026 ~07:40 WIB)
- **10 Okt 07:00–07:40 (sesi Claude Code; HANYA akun utama)**: GPU utama pulih 30 h. Skor per-notebook publik (dibaca via browser): heliosli S6 0,954 / versia-7, blend-gold, blend3 0,951 = semua MATI; **`kiyoshiohno/rsna-knee-template` 0,951 = launcher yg menjalankan logika dari dataset PRIVAT (tak bisa direplikasi)**; DINOsaur V5, V2-Velciraptor (klon heliosli, bundle compact-v1 sudah ditarik), Apex dan sisanya = 0,950. OF fold0 = ViT-L fine-tune + pipeline berat (SKM-TEA, KneeXNet; kode fastMRI/OpenKnee/KneeMRI) → probe kelayakan `rsna-knee-of-probe` (`build_probe_of.py`). **MaVIT hengck23**: gold 0,9202 direproduksi, LB solo 0,935, 1,5 dtk/studi; lixin73 336px TIDAK kompatibel (slot_emb 3 view vs 6; tanpa kode). `rsna-knee-apexmavit` (Apex + MaVIT@0,20 rank-blend; `build_submit_mavit.py`) lulus commit-run → submisi **57025414 = 0,949 → BUANG** (Δ−0,001 = noise/tak menambah; MaVIT 0,935 di 20% tak membantu — konsisten dgn teori galat berkorelasi; jatah hari ini sisa 4). Teori: model beda-arsitektur dgn label sama berkorelasi galat tinggi (reader hanya +0,001) → gain nyata butuh PENGETAHUAN label baru; garis perak/merger = opsi terbesar (126 tim ≥0,953 dgn ≤2 anggota & ≤250 submisi).
- **OrthoFoundation fold0 (probe `rsna-knee-of-probe` v5, 10 Okt 08:40 WIB)**: jalan end-to-end di Kaggle T4 setelah 3 perbaikan (salin `cloud_code` ke /kaggle/working agar bisa menulis; `anatomy.pt` + `kneexnet/` disalin, bukan symlink; yaml ultralytics 8.4.166 hilang dari paket vendored → dataset privat `dimaspashaakrilian/rsna-knee-ultralytics-cfg` dari wheel PyPI). Model = ViT-L DINOv3 (OrthoFoundation-L) 384px, anatomy_pool, epoch 7/16 EMA, `label_source_policy own_reports_and_official_gold_only_v1`, **48 dari 58 emas ikut latih fold0 → emas-58 0,9535 TERKONTAMINASI (bukan bukti)**. Kecepatan **27 dtk/studi** (1 GPU; ≈5 jam utk 1.322 studi di 2 GPU) → inferensi per 30 studi + hapus cache (disk). Notebook `rsna-knee-apexof` (Apex 0,70 + OF 0,30; `build_submit_legs.py legs=of wo=0.30`) sedang diuji commit-run; LB-test = 1 submisi (≈6 jam).
- **10 Okt 18:20 WIB — 57028008 (Apex 0,70 + OF fold0) = NOTEBOOK TIMEOUT (9 jam)**, tanpa skor. Perkiraan 5–6 jam meleset (dugaan: pemetaan seluruh `test_series` tiap potongan, oversubscription CPU, tanpa pembunuhan proses). Uji skala `rsna-knee-ofscalea` (2 proses) & `ofscaleb` (4 proses) berjalan (`build_scale_of.py`). **Bedah kode ber-skor terbaik**: notebook V2 menanamkan 7 sumber heliosli (compact_train/infer, probe_dual, px_cache, anchor bank) → ekstrak lewat `ast` dari `SOURCES` (lihat `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`). Pelajaran: target = report-state + prior; bobot blend tetap 0,7/0,3; pengendali dual-GPU berurut biaya + env thread=1 + keluaran bertahap + fallback. Kompilator aturan heliosli (`criteria.py`, lokal, 36 dtk) hanya 0,784 vs emas-58 → bukan jalan pintas label.

## 3-lama. Status (9 Okt 2026 ~23:40 WIB)
- **9 Okt malam**: probe r224 selesai siklus (push v1 → commit-run OK → submisi **57012143 = 0,950 → BUANG**). Riset 4 subagent selesai → `reports/Strategi meraih perak RSNA.md` + `research_notes/Strategi meraih perak RSNA/` (ter-commit & ter-push); `laporan_push.md` = wadah evaluasi silver (§1–§6, terintegrasi riset). Monitor `progress_submisi.sh` selesai (PID mati; log `progress_submisi.log`).
- **Jalur heliosli OF (0,951) MATI PERMANEN** (forensik kode 10 Okt, `research_notes/Cara menembus 0951 perak/arm_baru_realistis.md` Q1–Q2): S6 Stage-2 = `find_mount('rsna-knee-compact-v1','manifest.json')` + `find_mount('rsna-knee-ofl-ssl-weights','weight_parts.json')` **tanpa fallback** → kini RuntimeError (compact-v1 = README 183B per 9 Okt 13:50; ofl-ssl 403); OF fold0 publik (837 file) **TIDAK punya manifest.json**; receipt S6 8 Okt: bundle `209a32c8b50a6b1c` (4-fold E_lrprior) + side-model `ab623e2b` = HILANG semua; fork nursrijan jatuh ke raptor049 (0,949). Rekonstruksi fold0 = JANGAN (1 fold training-ckpt, engineering tinggi, EV rendah). Keluarga fork (d5d, h01–h05, versia-7, blend3) ikut tutup.
- **57015077 (fork rabari) = 0,950 → BUANG** (10 Okt ~00:45 WIB) — MIRROR_TTA/r224@0,3 tak mengangkat; plafon keluarga r384+reader+r224 = 0,950.
- **Akun KETIGA `nxtirfan` (perintah user 10 Okt 00:42)**: kredensial ada di repo `akun_kaggle/` (`kaggle.json`=nxtirfan dipakai wrapper, `kaggle1_dimaspashaakrilian.json`, `kaggle2_alverlfarfaraway.json` arsip; folder masuk `.gitignore`). Alat: `bash kg3.sh <arg>` (CLI akun3), `bash kg_quota.sh` (kuota 3 akun). Kuota awal: GPU 27,68h / TPU 20h. **Pakai HANYA bila GPU akun2 habis.**
- **Fork rabari (SELESAI & DITUTUP)**: dir `kaggle_kernel_rabari_dual_a2/` → `alverlfarfaraway/rsna-knee-dual-engine-tta-fork` → **57015077 = 0,950 → BUANG** (10 Okt 00:45).
- **Cache `rsna-knee-volcache-z{0,1,2}`** (CPU; geometri persis korpus nartaa: 96 irisan 26/22/18/12/18, 384 px, rentang 2–98%) = COMPLETE, 4.407/4.407 ok, sampel tervalidasi (96×384×384, 5 slot terisi).
- **TPU `dimaspashaakrilian/rsna-knee-train-tpu-oai-child`** (v1) = **QUEUED** sejak ±21:00 WIB (batas sesi TPU = 1). Isi: induk = ckpt r384 nartaa; anak a (KD 0,5, seed 1) lalu b (tanpa KD, seed 2), 3 epoch, LR bb 1e-5/head 1e-4, BN dibekukan, SWA epoch 2–3; lalu prediksi induk in-sample (`parent_soft_k16.csv`). ≈6 jam; keluaran `oai_child_{a,b}_{ep1..3,final,swa}.pt`. Log awal HARUS dicek (belum pernah jalan; XLA).
- **Colab GPU (T4) SELESAI/DIHENTIKAN 9 Okt 01:35** (sesi di-`colab stop`, kaggle.json dihapus dari VM): anak `c` (KD .5, LR 1e-5) ep1 gold-58 0,9207; anak `e` (resep r2, KD .25) ep1 0,9138; induk 0,9214. Evaluasi inferensi gold-58 (58 studi, SE ≈0,01): induk 384 plain 0,9214 | +cermin 0,9221 | crop 320 0,9219 (+cermin 0,9225) | sibling r224 0,9150 | campuran rank induk(320+cermin)+r224 @.2/.35/.5 = 0,9239/0,9238/0,9223 | +c1 @.2 0,9233 | +e1 @.35 0,9231. => anak TIDAK menambah (≈induk atau lebih rendah); r224 +0,001 (selisih ≈noise). Garis anak-OAI DIHENTIKAN (jangan buat penanda `rsna_gpu_launch_ok`; peluncur GPU anak c,d dibatalkan). Status latih terakhir di dataset `rsna-knee-oai-child-colab-ckpt` (resume c,e).
- **Siap-pakai / selesai 9 Okt**: probe r224 **siklus penuh selesai** (push `alverlfarfaraway/rsna-knee-oai-fam-r224` v1 → commit-run OK → submisi 57012143 **0,950 = BUANG**). Paket rekan: `docs/PAKET_REKAN_TIM.md`; merger: `docs/KANDIDAT_MERGER.md`. `scan_public_updates.py` + `scan_loop.sh` (fast-follower; 9 Okt temukan 4 kernel baru: rabari dual-engine 0,951+, sameerk d5f, rainycloudss v6, samanyu blend3 — blend3/versia-7 ikut mati kena blokir heliosli; pola baru 9–10 Okt: **d5e-goldw = bobot per-finding di-FIT di emas-58 → overfit, jangan ditiru; d5f = fine-tune reader on-the-fly dgn label-LLM → placebo + output run kosong**).
- **Kuota 9 Okt 23:30**: akun utama GPU 0,00 h (pulih **10 Okt 07:00 WIB**) · TPU 14,41 h; akun2 GPU ~29,9 h (reset juga 10 Okt 07:00); **submisi tim 9 Okt terpakai 3/5** (template, r224, fork) **→ sisa 2**, reset 10 Okt 07:00 WIB. Cek: `bash kg_quota.sh`.
- **Submisi tim terbaik**: 0,950 (Apex v2) · 0,944 (replika goodpjw) · 0,943 (d4-blend, in-arm r2c5).
- 9 Okt: TPU anak MATI (XLA batch_norm_backward) setelah antre 6 jam (induk TPU gold 0,9212 = Colab 0,9214); push kernel GPU ditolak (kuota mingguan habis) -> probe r224 menunggu GPU pulih Sabtu 10 Okt 07:00 WIB (atau kuota `nxtirfan`, `docs/PAKET_REKAN_TIM.md`). Target user peringkat ~10 (≈0,961): dijelaskan tidak realistis (12/20 teratas solo GM/juara RSNA). Git: sinkron dgn origin (3280526).

- **10 Okt malam (~01:00–03:00 WIB)**: izin panitia **TERARSIP** (`docs/IZIN_AKUN_KEDUA.md`, teks "Oh, of course, that's allowed." + kronologi 10 submisi; semua checklist dokumen ditandai). Riset mendalam 3 subriset selesai → `reports/Cara menembus 0951 perak.md` (sintesis) + `research_notes/Cara menembus 0951 perak/` (forensik papan, inference gratis, arm baru). Alat baru: **forum bisa dibaca via CLI** `kaggle forums topics show <id>`.

## 4. Constraint & operasional
- ML HANYA di Kaggle (lokal: tanpa torch/sklearn, RAM 5 GB, disk kecil). T4 wajib (`machine_shape: NvidiaTeslaT4`; P100 tak kompatibel). 1 sesi GPU = 2×T4 → jalankan 2 proses (`CUDA_VISIBLE_DEVICES`) utk 2× throughput.
- Kuota: GPU 30 jam/minggu, TPU 20 jam/minggu, reset Sabtu 00:00 UTC (= 07:00 WIB): 10 Okt, 17 Okt. Sesi akun: GPU 2 / TPU 1 / CPU 5. Akun DIBAGI dgn proyek lain → jangan jalankan job GPU proyek lain saat kuota pulih sebelum job RSNA terkirim. Baca `kaggle quota` kolom `remaining`.
- **Akun kedua (mulai 9 Okt) & ketiga (mulai 10 Okt)**: kredensial `~/.kaggle-acct2/kaggle.json` (`alverlfarfaraway`) & `akun_kaggle/kaggle.json` (`nxtirfan`); perintah: `bash kg2.sh <arg kaggle>`, `bash kg3.sh <arg kaggle>`, `bash kg_quota.sh` (kuota 3 akun). Push kernel akun2 pakai dir ber-id `alverlfarfaraway/...` (`kaggle_kernel_oai_fam_r224_a2/`); jangan push dir ber-id `dimaspashaakrilian` dari akun2 (403). DILARANG `VERBOSE=true` (mencetak header Authorization = membocorkan kunci; kejadian 8 & 9 Okt).
- TPU/XLA: bentuk tensor tetap, autocast bf16 'xla', tanpa GradScaler, `xm.mark_step()`, LR dikuantisasi (30 level), cegah recompile. Kecepatan CoAtNet-384: ≈0,5 dtk/langkah (12 window).
- Kaggle CLI: tambah `~/.local/bin` ke PATH bila `kaggle` tak ada. 429 (rate limit) → ulang dgn jeda. `kaggle kernels output` KOSONG selama RUNNING → baca log langsung lewat Chrome (halaman `/code/<owner>/<slug>` → tab Logs, JS ambil baris `^\d+(\.\d+)?s\s+\d+\s+`); tutup tab yg dibuka. `kaggle kernels pull` menyertakan `id_no` → hapus sebelum push salinan (kalau tak, 403). Judul notebook harus menghasilkan slug yg sama. `kaggle competitions submit -k -v` WAJIB disertai `-f submission.csv`. **Forum bisa dibaca via CLI**: `kaggle forums topics show <topicId> --page-size 200` (tanpa browser; utk cek host 743416 & baca resep) — `kaggle kernels list --user <u>` (bukan `-u`).
- Tool lokal: JANGAN `pkill -f <pola yg ada di perintah sendiri>` (membunuh shell); tugas latar bisa dibunuh saat memori ketat → bila tak ada kabar, jalankan skrip tunggu manual. Scratchpad sesi terhapus saat restart (simpan hal penting di repo).
- Repo GitHub PRIVATE `PashaAkrilian/RSNA-Knee-Abnormality-Detection` (branch `master`); rekan `nxtirfan` punya akses push → `git fetch` dulu sebelum push.

## 5. Menunggu keputusan user
1. **Risiko kepatuhan OAI** (ckpt nartaa dilatih dgn data OAI/NDA; host belum menjawab thread 743416): pilih 2 submisi final = 1 bersih + 1 agresif; putuskan 21–22 Okt setelah cek ulang thread. Jika host melarang → kembali ke jalur bersih.
2. **Koordinasi dgn nxtirfan**: siapa submit apa; tahan submisi tak penting; pastikan tak ada job GPU proyek lain mencuri kuota pada 10 Okt 07:00 WIB.
3. (Opsional, tidak disarankan) model OAI buatan sendiri via NDA — hanya user yg bisa mendaftar; lihat RESEARCH_NOTES §37–§37b utk kelayakan.
4. **Akun kedua & ketiga — IZIN TERARSIP (10 Okt)**: teks panitia **"Oh, of course, that's allowed."** (konteks: permintaan tim memakai akun ganda, per user) + kronologi submisi kini di **`docs/IZIN_AKUN_KEDUA.md`**. Sisa mitigasi: (a) simpan email asli di inbox sbg bukti utama, (b) pastikan beda orang per akun (Rules 3.5.b), (c) submisi final 21–22 Okt diusahakan dari akun utama. Jangan simpan/cetak kredensial di repo/log. **Catatan sesi Claude Code 10 Okt**: tidak memakai akun 2/3 untuk RSNA (Rules 3.5.a/3.5.b; izin tertulis hanya satu kalimat tanpa pertanyaan/pengirim; GPU utama 30 h cukup utk rencana). Saran: tanya Kaggle support secara tertulis (draf: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`) dan jalankan komputasi rekan oleh rekan sendiri dari akunnya (`docs/PAKET_REKAN_TIM.md`).

## 6. Referensi cepat
- Kernel/dataset inti: `dimaspashaakrilian/rsna-knee-train-tpu-oai-child` (TPU), `rsna-knee-volcache-z{0,1,2}` (cache), `rsna-knee-fused-labels-r1` (label), `rsna-knee-goodpjw-replica` (0,944), publik `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` (0,950) + `nartaa/rsna-knee-publication-swa-weights-20261007` (ckpt r384 0,949 & r224 0,945) + `goodpjw2008/rsna-knee-2-5d-convnext-reader`.
- Basis notebook di repo: `kaggle_base/apex_grandmaster_stack_20261008.ipynb` (+ `_metadata.json`), `kaggle_base/pj_rsna_knee_d4_blend_20260928.ipynb`.
- Builder: `build_train_tpu_oai_child.py`, `build_train_gpu_oai_child.py`, `build_submit_oai_family.py`, `build_submit_plus.py`, `build_submit_multi.py`, `build_volcache_cpu2.py <tag> <img> <SLOTS64|SLOTS44|SLOTS96> <lo> <hi>`, `scan_public_updates.py`, `label_engineering/fuse_multi_student.py`.
- Forum penting: **735304 (pseudo-label → skor top: colum2131 0,958 / azamat1ch 0,960 — PEMISAH 0,951+)**, 743416 (data eksternal, host masih diam), 741819 (OAI sah bila accessible), 733965 (LLM boleh), 746792 (resep nartaa 0,949), 745861 (tips hengck23), 745214 (label), 744519 (modalitas test, tak dijawab), 747658 (laporan↔gold tak sinkron).
- Riset ensembling/bobot (9 Okt, 4 Q: teknik top publik, post-mortem imaging, pilih bobot N kecil, sinyal vs noise) → `research_notes/Strategi meraih perak RSNA/teknik_ensembel_bobot.md`. Inti: rank-mean = standar; lift terbukti = arm baru/TTA (+0,001–0,003), per-label exclude (tanpa angka), optimisasi bobot halus +0,0002; ambang Δ<0,002 = noise.
- **Laporan 9 Okt: `reports/Strategi meraih perak RSNA.md`** (jalur lama heliosli = kini MATI). **Laporan 10 Okt (TERBARU): `reports/Cara menembus 0951 perak.md`** — pemisah = pseudo-labeling (735304); papan 0,951 lotre / 0,952 aman; rencana Fase A (mirror H1–H3, gratis) → Fase B (pseudo-label self-training 10–25 GPU-jam); daftar tutup 12 baris; gating gold-58 (Δ≥0,005 + CI95).

## Protokol Update (WAJIB diikuti agent apa pun yang bekerja di repo ini)

Update MEMORY.md SEGERA setelah:
1. Keputusan arsitektur/model baru dibuat/diubah → update bagian 2
2. Status pipeline berubah (mulai/selesai/gagal) → update bagian 3
3. Menemukan constraint/limitasi baru (misal GPU, quota, aturan kompetisi) → update bagian 4
4. Ada pertanyaan yang butuh keputusan user → tambah ke bagian 5, HAPUS dari bagian 5 setelah user menjawab (jawabannya pindah ke bagian 2/3/4 yang relevan)
5. Selesai satu unit kerja besar → update "Last updated" timestamp

ATURAN PENULISAN:
- Overwrite, jangan akumulasi — MEMORY.md harus selalu bisa dibaca penuh dalam <2 menit
- Detail panjang/historis tetap masuk PROGRESS.md, BUKAN MEMORY.md
- Setiap update MEMORY.md, commit terpisah dengan pesan: "memory: <ringkasan perubahan>"


## 52. 10 Okt 2026 (sesi Claude Code; HANYA akun utama) — kronologi
- 06:57 papan 5.625 tim, kita 0,950 rank 414; ≥0,951 = 309; GPU utama pulih 07:00 (30 j). Baca semua .md hasil OpenCode; temuan kepatuhan: izin akun ganda = satu kalimat; Rules 5.a/5.b → sesi tidak memakai akun 2/3.
- 07:00–08:30 browser: skor per notebook; `kiyoshiohno/template` 0,951 (logika privat); MaVIT hengck23 (gold 0,9202, LB solo 0,935), lixin73 336px tak kompatibel; `build_probe_mavit.py`, `build_submit_mavit.py`; **57025414 Apex+MaVIT = 0,949 (buang)**.
- 07:30–09:10 OrthoFoundation fold0: probe (v1–v5; 3 perbaikan), `build_submit_legs.py`/`rsna-knee-apexof`; **57028008 Apex+OF** submit 09:10 → **Notebook Timeout** (dilihat 18:20).
- 18:20–19:50 bedah kode heliosli (7 sumber dari V2), kompilator aturan 0,784, uji skala v1 (`ofscalea/b`: disk 330 MB/studi, CPU hambatan), `of_stage.py` v2, `build_scale_of2.py`, `ofscale3/4`, `apexof2` (stub lulus).
- 19:37 papan 5.666 tim: kita 0,950 rank 530; ≥0,951 = 348, ≥0,952 = 217. Scan publik: tak ada notebook ≥0,951 runnable; klaim "0,954" = kode Apex. Rules resmi diambil (5.a–5.d, 6.a/b, merger 5.c). Simulasi peluang (`silver_prob_sim.py`).
- 19:50–21:30 dokumen: `STRATEGI_PERAK.md` (baru), MEMORY.md dipadatkan (arsip penuh di §52b), HANDOFF.md, push.md, laporan_push.md, README.md, CLAUDE.md/AGENTS.md, docs (KANDIDAT_MERGER, DRAF_PESAN_KAGGLE_SUPPORT, IZIN_AKUN_KEDUA catatan, PAKET_REKAN_TIM), RESEARCH_NOTES §39–§41, banner arsip pada laporan/riset lama.
- Commit lokal (belum di-push): lihat `git log`. Tugas latar: pemantau `ofscale3/4`.
- 20:15–20:30 hasil uji skala: `ofscale3` 194 studi/jam (6,8 j, 0 gagal), `ofscale4` (reuso anatomi) 150/jam → reuso dimatikan; `apexof2` v3 lulus stub (GPU kosong 15 GB). Pewaktu submit `apexof2 -v 3` diset 11 Okt 07:03 WIB (jatah agen ≤2/hari; skrip `submit_at.sh`).
- 21:10 atas "lanjut" user: **57044355** (apexof2 v3) disubmit, pewaktu 07:03 dibatalkan (jatah hari ini sisa 2); paket merger lengkap (`docs/KANDIDAT_MERGER.md`: aturan 5.c/5.d, cara kirim, kandidat dgn tautan profil, templat pesan, pelacak); thread 745861 dibaca (resep label hengck23 = fusi emas ±0,938, tak baru).

## 53. 10 Okt 2026 malam (sesi Claude Code; HANYA akun utama) — dataset privat papan atas + Plan B
- Menjawab "isi dataset privat tim perak": salinan privat bundel heliosli `rsna-knee-compact-v1` (4 ckpt fold ConvNeXt-S 2.5D + state_spec + model-sisi + manifest) dan `rsna-knee-ofl-ssl-weights` (OF-L SSL), bundel logika/Takakura; isi dataset tim 0,955–0,964 tak diketahui (rincian & sumber: `docs/RESEARCH_NOTES.md` §43).
- Dibuat `docs/PLAN_B.md` (pohon keputusan skor 57044355, merger Kelompok A/B, dua final terdekorelasi, tuas teknis, jadwal/pemicu); `docs/KANDIDAT_MERGER.md` §3b; STRATEGI_PERAK §2 (kolom perunggu + koreksi), HANDOFF, laporan_push, MEMORY diselaraskan.
- KOREKSI: perunggu bukan 85–95 % (Apex 0,950 → 25–41 %); jalur bersih 0,944 bukan lagi zona perunggu (rank ≈1.387 vs garis 0,950 rank 567).
- Ledger publik prvsiyan (90 submisi; terbaik 0,949) menguatkan teori pengetahuan>varians. Scan 22:05: gold-apex-v7 (kerabat Apex) ditutup.
- 57044355 masih PENDING (22:00 WIB); hasil ±04:00 WIB 11 Okt.

## 54. 11 Okt 2026 dini hari (sesi Claude Code) — penyebab kegagalan OF di run penilaian + perbaikan v4.1
- **Gejala**: 57044355 (apexof2 v3 = Apex lalu OF) = 0,950 (COMPLETE 22:45 WIB, ±1 j 35 mnt) padahal OF butuh ±6,8 j utk 1.322 studi → OF tidak jalan (liputan <15 % → blend Apex murni).
- **Rehearsal `apexof4r`** (OF dulu, 96 studi pseudo-uji dari TRAIN tanpa emas, injeksi kill): mekanisme v4 jalan end-to-end, tetapi **RSS tiap pekerja OF 8–10 GB** (4,0 GB @5 mnt → 7,3–8,5 GB akhir potongan-1 → 10,6 GB potongan-2); 3 pekerja bersamaan → OOM nyata (rc −9, dua pekerja mati serentak ±17 mnt; induk hanya 0,2 GB). Throughput: 1 pekerja 44–52 dtk/studi; 3–4 pekerja 62–80 dtk/studi/pekerja.
- **Diagnostik `apexof4d`** (urutan LAMA OF sesudah Apex, 1.322 studi pseudo): proses induk notebook memegang **18,0 GB RSS** setelah Apex (MemAvail 12,0 GB) → pekerja OF mati OOM → koverasi 0,012 → PENYEBAB 57044355 terkonfirmasi. Waktu Apex 1.322 studi: Raptor 1.567 dtk (pre 1.477 dtk) + reader 1.755 dtk = 3.323 dtk (0,92 j).
- **Profil memori `ofmem1/ofmem3`** (1 pekerja, 8 studi/potongan; puncak RSS potongan-1/potongan-2 GB · dtk/studi): base 8,8/11,7 · 48,1; malloc-threshold 8,7/10,5 · 52,4; chunk4 6,5/10,1 · 41; workers=0 9,2/12,1 · 47,7; tcmalloc 9,7/12,9 · 42,3; nopin 6,3/9,0 · 44,9; **nopin+mmap 5,1/6,5 · 38,7**. Sumber lonjakan: pin_memory (±2,5 GB shmem host-allocator CUDA), `torch.load` checkpoint ViT-L dimuat 2× (1,2 GB), anon +2–3 GB per potongan (fragmentasi). CPU per pekerja ≈1 core. Fase potongan 8 studi (±385 dtk): prepare 23 %, geometri 11 %, anatomi 46 %, predict 17 %.
- **v4.1**: `of_stage_v4.py` (antrean klaim `os.mkdir`, supervisor reaping/restart/daur-ulang, guard RAM MemAvailable, oom_score_adj 500, keluaran atomik, pemutus lunak/keras), `build_apexof4.py` (of_first|apex_first, pseudo=N, inject, loud, nopin, mmap, extraenv), `test_of_v4_local.py` (4 skenario stub lulus).
- **Rehearsal `apexof4r2`** (v4.1; 4 pekerja, kill-injection, 240 studi): LULUS — 25/30 potongan dlm 3.713 dtk (194 studi/jam dgn 3 pekerja), MemAvail min 19,2 GB, RSS pekerja 3,0–4,8 GB, Apex 240 studi 320 dtk + reader 293 dtk, blend OK (corr 0,775). Stub produksi `apexof4t` LULUS (OF 3 studi 100 %).
- **Rencana**: `apexof4r3` (4 pekerja tanpa injeksi) → bila stabil, submit `apexof4t` (loud) 11 Okt 07:03 WIB.

## 55. 11 Okt 2026 dini hari (sesi Claude Code) — S6 heliosli hidup lagi, riset forum, build_s6plus
- **Kejadian**: heliosli memulihkan `rsna-knee-compact-v1` (02:50 WIB; thread 748141) + notebook S6 publik (0,954 satu klik). User menyalin & men-submit S6 (57051680, 02:56 WIB). Papan 02:59: 5.699 tim, kita rank 599 (0,950; di bawah garis perunggu publik rank 570).
- **Riset**: kebun-binatang bundel V1–V13 (kagglehub per versi); KOREKSI premis (resep heliosli = Raptor + compact, bukan OF); forum "best single-model" (ConvNeXt-T 5-fold single 0,953–0,959 dgn pseudo-label); validasi OOF label r=0,87 vs LB; trik 1−p; thread kepatuhan; notebook publik baru (Tri-Reader 0,954 klaim = Apex+r224). Rinci `docs/RESEARCH_NOTES.md` §44.
- **Eksekusi**: sonda `s6probe` (160 studi pseudo-uji: 80/80 per pekerja, 0 fallback, ±19 dtk/studi/pekerja; ERROR akhir = artefak `sample_submission.csv` pseudo, diperbaiki); `build_s6plus.py` (reader/OF opsional + blend 3 perna + monitor RAM + mode lenient); rehearsal `s6plusr`; dokumen STRATEGI_PERAK/HANDOFF/laporan_push/PLAN_B/KANDIDAT_MERGER/CLAUDE/AGENTS/README/push.md/MEMORY diperbarui.
- **Keputusan**: strategi berpusat S6; OF turun peringkat (prior ≤20 %); submit varian S6+ setelah rehearsal dan setelah jatah reset 07:00 WIB; merger tetap tuas terbesar (aksi user).
- **Rehearsal `apexof4r3` (11 Okt 03:45 WIB)**: 4 pekerja tanpa injeksi, 200 studi pseudo-uji: 228 studi/jam, 0 pekerja mati, MemAvail min 13,8 GB; v4.1 siap dipakai (S6+OF memakai 4 pekerja). Sonda `s6probe`: compact 160 studi = 1.530 dtk (±19 dtk/studi/pekerja; ERROR akhir = artefak pseudo).

## 56. 12 Okt 2026 pagi (sesi OpenCode) — pipeline murid: 3 gelombang ERROR + fix bertahap
- **Gelombang 1 (00:15 push, ERROR 00:30)**: train-a (alver), train-b (main), dino (nxtirfan) serentak. Finder mount 2-level + rekursif SUKSES (COMP/PREP/GOODPJW ketemu; pylibjpeg wheel cp311/cp312 ditolak di py3.13 — tidak fatal, jalur fallback). Namun `CACHE = f"{PREP}/cache256_train"` menunjuk ke `/kaggle/input/datasets/...` yang **read-only** → `OSError: [Errno 30]` saat `os.makedirs`. Orkestrator `night_student.sh` (PID 34500) exit 00:35 (fold=0). **Fix**: cache on-demand dialihkan ke `/tmp/cache256_train` (`build_student_train.py` + `build_student_dino.py`); regen notebook (arg `name=`/`folds=`/`acct=`, bukan posisional).
- **Gelombang 2 (02:23 push, ERROR ~06:15)**: cache on-demand SUKSES (24.371 file, ±72 mnt — 5,6 dtk/file dengan Pool); warm-start ConvNeXt dari `goodpjw` SUKSES (missing=0/unexpected=0); fold0 mulai → **`OutOfMemoryError: CUDA` di batch pertama** (`bs=8`, percobaan alokasi 468 MiB, 14,14/14,56 GiB terpakai). Catatan: goodpjw asli juga default `bs=8` di `train.py`, tapi mereka melatih di GPU lokal (kemungkinan >24 GB), bukan T4. **Fix**: bs 8→4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Gelombang 3 (~07:45 push, ERROR ~10:15)**: OOM LAGI dengan bs=4 (alokasi 126 MiB gagal; 12,75 GiB allocated + 1,62 GiB fragmented). Akar masalah: `KneeNet.forward` menjalankan backbone pada `x.view(b*s,k,...)[keep].flatten(0,1)` = **bs × series × kwin gambar 256 px sekaligus** (dengan bs=4, kwin=8, ±3 series/studi → ±96 gambar ConvNeXt-T). Dino beda penyakit: **`NotImplementedError: Module [_AttnOut] is missing forward`** — kelas wrapper DINOv2 buatan sendiri (`build_student_dino.py` CELL_MODEL) lupa mendefinisikan `forward` pada sub-modul `dense` (hanya `__init__`). **Fix**: bs 4→2 (train & dino; catatan di kode); `_AttnOut.forward` ditambahkan.
- **Gelombang 4 (~10:45 push)**: v6/v4/v6 RUNNING pada saat penulisan. Titik kritis OOM = ±30 mnt setelah log "warm-start" (±75 mnt setelah start, karena cache build ±72 mnt). Keputusan bila OOM lagi: (a) gradient checkpointing monkey-patch `model.enc`, ATAU (b) kwin 8→6; jangan turunkan res (resep goodpjw = 256).
- **Kuota (08:23 WIB)**: alver 17,27 j sisa; main 15,84 j; **nxtirfan 3,51 j** (hati-hati: cukup 1 run dino; kalau gagal, pindahkan dino ke main/alver). Empat gelombang membakar total ±4–5 j/akun.
- **Operasional**: `kaggle kernels push` = push + run otomatis (CLI 2.2.4 tidak punya `kernels run`); log kernel hanya bisa diunduh setelah run selesai (`kernels output`); `kaggle datasets download -f <file> -p <dir> -o` untuk tarik 1 file dari dataset publik. Dataset goodpjw TIDAK berisi cache (hanya ckpt + kode + wheels).

## 57. 12 Okt 2026 siang (sesi Claude Code, mesin Windows) — scrape mendalam, strategi 0,960, S6-eq, guru OOF compact
- **Lingkungan baru**: repo di-clone ke `C:\Users\LE\01- project\`; Kaggle CLI 2.2.4 dipasang; kredensial akun alver (`kaggle (2).json` dari user) disalin ke `~/.kaggle-acct2` (dipakai `kg2.sh`/`KAGGLE_CONFIG_DIR`); berkas aslinya dikecualikan lokal lewat `.git/info/exclude`. Akun utama & nxtirfan tidak ada di mesin ini.
- **Scrape mendalam (13:25–13:55 WIB)** ke `..\kaggle-scrape-rsna-knee-20261012\`: 10 halaman resmi, papan penuh (5.807 tim), 182 topik (komentar + postingan pembuka via `competitions topic-messages`), indeks 1.100 notebook + kode 169, 233 dataset + 157 model. Sintesis: `reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md`.
- **Temuan**: perak & perunggu publik sama-sama 0,954 (dataran 604 tim; kita rank 455→466); notebook publik #1 = S6 0,5/0,5 + rank sadar-gagal (heliosli); 25/25 notebook teratas = keluarga Raptor+compact; velci-kcd25-20 (S6 + kaki privat 0,2) = tim 0,955; pemisah papan atas = murid pseudo-label sendiri (klaim tunggal 0,953–0,959).
- **Verifikasi kode compact** (`compact_train.py` tertanam di notebook S6 + `manifest.json` bundel): 4 fold hash `int(sha1(uid)[:8],16) % 4` atas 4.344 non-gold, `gold_in_training 0`, `compact_infer.py` memuat model via `glob(arm_s*_f*.pt)` → OOF compact dapat direkonstruksi per fold. Builder `build_compact_oof.py` (fold 0–3 + gold) dibuat & diuji lokal (ukuran fold lokal 1.033/1.139/1.125/1.071).
- **Strategi**: `STRATEGI_0960.md` (komponen +0,006, notebook final S6-eq + murid A/B, fase, gerbang G0–G4, anggaran ±35 j GPU, Plan B). Perkiraan jujur: ≥0,960 ±20–30 %, ≥0,957 ±50–60 %.
- **Submisi (perintah user: 2)**: `build_s6eq.py` → `s6eq` (notebook #1 heliosli tanpa ubahan) dan `s6eqr` (+ Part B reader goodpjw); push pertama ditolak "Maximum batch GPU session count of 2" (alver menjalankan student-train-a & -b) → `scripts/submit_watch.py` mencoba push tiap 2 mnt, menunggu COMPLETE, cek `submission.csv`, submit, lalu memantau skor. Kuota alver 10,22 j @14:33 dgn laju ±2 j/jam.
- **Scrape putaran-2 (berjalan)**: papan 07:29 UTC (≥0,954 = 803; emas rank 21 = 0,962; kita rank 466), delta topik (1 berubah) & 4 notebook baru, pemindaian aset publik 297 anggota tim top-200.
- **Git**: atas perintah user dokumen diperbarui dan di-push ke branch + PR untuk dievaluasi rekan, tanpa atribusi Claude.

## 58. 12 Okt 2026 sore (sesi OpenCode, mesin Linux) — merge PR, eksekusi 2 submisi test, aturan dataset pin versi
- **`git pull` + merge PR #1** (`scrape-strategi-0960-12okt` → master, `99313b6`): masuk `STRATEGI_0960.md`, laporan scrape peta perak, `build_s6eq.py`, `build_compact_oof.py`, `scripts/submit_watch.py`, kernel `s6eq`/`s6eqr`/`compact-oof-f0..f3`/`compact-gold`. Baca penuh scrape + strategi sebagai dasar keputusan (perintah user: riset dulu, baru eksekusi).
- **Papan segar 16:00 WIB** (unduh CSV, 5.812 tim): kita **rank 499** (0,954; 114 submisi); ≥0,954 = **835** (649 tim tepat 0,954), ≥0,955 = 186, ≥0,956 = 122, ≥0,960 = 39; perak 290 = 0,954, perunggu 581 = 0,954, emas 21 = 0,962. Dataran tumbuh ±34 tim/hari — posisi di dalam 0,954 makin tidak menentukan; yang menentukan = naik ke ≥0,955 ATAU informasi di luar blok klon.
- **Keputusan akun**: perintah user 12 Okt = ketiga akun boleh, akun utama JANGAN untuk training. Alver 2/2 sesi (train-a/b) + kuota terbakar ±1,9 j/jam → habis ±20:00; nxtirfan kuota 0 (dino sudah COMPLETE); → `s6eq`/`s6eqr` dijalankan di **akun utama** (inferensi/submisi, bukan training; main sisa ±13 j).
- **Temuan operasional penting (400 pertama)**: submit `CreateCodeSubmission` ditolak `"Submission not allowed: Your Notebook uses non-versioned datasets [heliosli/rsna-knee-compact-r6cnxfsss] (see Dataset Settings)"`. Submisi 11 Okt masih diterima tanpa versi → aturan baru mulai 12 Okt. Fix: `build_s6eq.py` mem-pin `owner/slug/N` (ver via SDK `datasets.dataset_api_client.get_dataset`: r6cnxfsss **v2** sesuai bundel `209a32c8`, nartaa **v1**, OF fold0 **v1**, reader **v1**) → push v2 sukses. `submit_watch.py` juga diberi mode `--resume <ver>` (lanjut tanpa push ulang) dan `MAX_COMMIT_H` 3,0→5,5 j (S6 penuh ±3 j 45 mnt).
- **Eksekusi 16:12**: `s6eq` v2 + `s6eqr` v2 push (commit-run 3 studi ≈5 mnt) → dua `submit_watch --resume 2` (PID 25063/25065, log `data/watch_s6eq*_12okt.log`) menunggu COMPLETE lalu **submit otomatis** (jatah 5/5) → skor run penuh ±3 j 45 mnt ≈ **20:15 WIB**. G0 `STRATEGI_0960.md`: ≥0,955 = lulus.
- **Pipeline murid**: train-a/b tetap RUNNING dan sudah melewati titik OOM (bs=2 teruji ±5 jam masuk); dino COMPLETE (nxtirfan) — `dino_full.pt` siap ditarik. train-c + inferensi murid **tidak bisa malam ini** (alver kuota habis, nxtirfan 0, main dilarang training) → antre reset **17 Okt 07:00 WIB**; malam ini hanya tarik output train-a/b (CPU) + pantau skor.
- **MD**: MEMORY §3/§4, HANDOFF §0/§0c/§G, laporan_push §2 (baris 400 + eksekusi), PROGRESS §58 diperbarui.

## 59. 12 Okt 2026 17:05 — akar 400 terpecahkan: mirror dataset, s6eq/s6eqr v3 TERKIRIMA
- **v2 juga 400** (submit otomatis watcher `--resume 2` → sama "non-versioned datasets"). Pin versi `owner/slug/N` di kernel-metadata TIDAK menolong: `ApiSaveKernelRequest.dataset_data_sources` didokumentasikan hanya `{username}/{dataset-slug}`; GET metadata server memang menampilkan tanpa versi untuk kernel siapa pun (termasuk milik helios/goodpjw). Jadi BUKAN masalah metadata kernel.
- **Akar sebenarnya** (riset web, thread Ariel Data Challenge 2024): pesan error merujuk setting **"All Versions Accessible" di halaman DATASET itu sendiri** — hanya pemilik dataset yang bisa mengubahnya. `heliosli/rsna-knee-compact-r6cnxfsss` punya akses versi mati; `nartaa/…`, `heliosli/rsna-orthofoundation-…`, `goodpjw2008/…reader` lolos (terbukti hanya r6cnxfsss yang disebut di pesan 400, padahal ketiganya ikut tertempel).
- **Fix = mirror**: unduh r6cnxfsss v2 (823 MB zip → 6 file: 4×`E_lrprior_s0_f*.pt` 200 MB, `side_cls.pt`, `manifest.json`, `state_spec.json`), unggah ulang sebagai `dimaspashaakrilian/rsna-knee-compact-r6cnxfsss` v1 (887 MB, public, slug SAMA → path mount di notebook tidak berubah sama sekali, deskripsi kredit heliosli). Metadata kini memakai mirror `/1`.
- **Hasil**: `build_s6eq.py` + `build_s6eq.py name=s6eqr reader=1` → push v3 16:47 → commit-run COMPLETE 16:49 → submit manual 16:50 **DITERIMA** (ref 57103358 s6eq, 57103359 s6eqr; jatah 3/5 tersisa) → skor penuh ±20:35 WIB.
- **Catatan**: dataset yang kita buat sendiri lolos pengecekan (bukti v3 diterima) — penting untuk `rsna-knee-student-*` nanti. Upload gagal 2× saat init (judul >50 kar, subtitle >80 kar) → perbaiki jarak judul/subtitle. Lokal: `data/mirror_upload/` (bisa dihapus setelah skor keluar); watcher lama mati saat SELESAI (400) — cek skor via `kaggle competitions submissions -v`.

## 60. 12 Okt 2026 23:05 — skor s6eq/s6eqr = 0,954/0,954 (G0 gagal), train-a/b COMPLETE, folds v1
- **Skor masuk ±18:10**: `s6eq` v3 = **0,954** (57103358), `s6eqr` v3 = **0,954** (57103359) — identik klon S6 → **gerbang G0 ≥0,955 TIDAK lulus**; reader ke-3 konsisten net-zero (s6reader 11 Okt juga 0,954→0,954). Kesimpulan: plafon S6/compact ±0,954; kenaikan harus dari murid (azamat1ch 0,959, datnt114 0,961 = train-on-all pseudo). Jatah submisi 3/5. Watcher skor (`data/watch_v3_score_12okt.log`) berhenti dicatat 18:10 lalu dimatikan manual.
- **train-a/b COMPLETE ±18:10** (alver, ~9 j total): fold0–3 + oof_fold0–3 tarik lokal `data/student_out_a|b` (471 MB + log). Val AUC per fold (baris terakhir tiap fold): **fold0 0,9255 / fold1 0,9257 / fold2 0,9271 / fold3 0,9279** (gold 0,9189–0,9223, ep13). OOF CSV = 928 baris val/fold × 4 = 3.712 = seluruh train split. Log: `TRAIN_DONE files: [...]`.
- **dino_full.pt ditarik** dari `nxtirfan/rsna-knee-student-dinov2-s-one-shot` (95 MB + log) — slug kernel dino ≠ `student-dino` (periksa metadata!). Lokasi: `data/student_dino/`.
- **Dataset `dimaspashaakrilian/rsna-knee-student-folds` v1 (privat) LIVE**: fold0–3.pt (123 MB ×4) + oof_fold0–3.csv; deskripsi mencatat fold4 = train-c 17 Okt (naik ke v2). Upload privat lolos (mirip mirror publik tadi → setting versi bawaan aman). `build_student_infer.py` dicatat belum mount folds → tambah saat push.
- **Bersih-bersih**: `data/mirror_upload` dihapus (846 MB; dataset mirror sudah di Kaggle) → disk 19 GB free. Malam ini TIDAK ada push GPU lagi (alver sisa 4,6 j dibiarkan — train selesai; main 12,67 j; nxtirfan 0).
- **MD**: MEMORY §3 (23:05), HANDOFF §0c/§G (4–6), laporan_push baris 20/21, PROGRESS §60.
