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

## Belum dikerjakan

- [ ] Tunggu hasil v7 — cek apakah val_macro_auc terus naik melewati 0.7658 atau mulai overfit
- [ ] v8: swap label ke `flight0234/rsna-knee-hybrid-report-labels` (0.8991) — siap push begitu v7 selesai (tidak interupsi run yang sedang jalan)
- [x] ~~Keputusan Anda soal 3 item STOP~~ — 2/3 terjawab via Rules resmi (§11): external data & pretrained model kompetitor lain **diizinkan eksplisit**. Sisa: `marwanmath` dkk. tetap direkomendasikan skip (bukan krn Rules, krn ada alternatif MIT lebih bersih)
- [ ] Lanjut ke ensemble RadImageNet-ResNet50 + BiomedCLIP (+MedicalNet?) sesuai `docs/RESEARCH_NOTES.md` §5/§8c — syarat baseline belajar sinyal nyata sudah terpenuhi di v6
- [ ] Proper k-fold CV (bukan cuma 1 split gold N=57) + OOF ensemble selection
- [ ] Belum ada pipeline inference/submission — **sekarang ada spesifikasi keras dari Rules resmi (§11)**: wajib via Notebook, nama file `submission.csv`, internet MATI, <=9 jam runtime utk ~1.300 studi test + inferensi ensemble, kolom `Report` tidak tersedia saat testing. Perlu dibangun dgn `dataset_sources`/`model_sources` (bukan `pip install` on-the-fly) sebelum submit ke leaderboard
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
