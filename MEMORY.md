# MEMORY — RSNA Knee Abnormality Detection
> Living memory. Update bagian relevan setiap ada temuan/keputusan baru.
> Jangan tambah terus ke bawah — overwrite bagian yang sudah usang.
> Last updated: 2026-08-19T18:40:00+07:00

## 1. Fakta Kompetisi (jarang berubah)
- Kompetisi: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — code competition, 1.611 tim.
- **Metrik: macro (mean) ROC-AUC** atas 12 kolom label, $\frac{1}{12}\sum AUC_i$ — dikonfirmasi resmi via `kaggle competitions pages` (Evaluation).
- **Deadline: Final Submission 22 Oktober 2026, 23:59 UTC.** Entry/Team Merger deadline 15 Oktober 2026. Winner wajib serahkan training+inference code & model weights (Kaggle Dataset publik).
- Data: 4.407 studi train, 24.371 series, 569.76GB (mustahil diproses lokal — semua eksekusi WAJIB di Kaggle). Hanya **58 studi (1.3%) gold-label penuh**, sisanya 98.7% kosong di 12 kolom label tapi 100% punya teks `Report`. `test.csv` publik cuma 3 studi (test asli tersembunyi).
- **Submission**: WAJIB via Notebook, output `submission.csv`, **internet HARUS mati** saat run submission, **≤9 jam runtime** per run (termasuk decode ~1.300 studi test + inferensi ensemble). Kolom `Report` TIDAK tersedia saat testing (cuma valid sbg target training, bukan input inferensi).
- **External data & pretrained model DIIZINKAN eksplisit** (Code Requirements: "Freely & publicly available external data is allowed, including pre-trained models"). Sharing kode publik juga diizinkan (Rules §3.5.d).
- **Data Security (Rules §2.4.b)**: dilarang transmit Competition Data (termasuk teks `Report`) ke pihak yang TIDAK ikut kompetisi. **API LLM hosted (OpenAI/Anthropic/dll) DIIZINKAN** untuk memproses `Report` (ruling resmi host, topic 733965) SELAMA layanannya "reasonably accessible & minimal cost" ke semua peserta (Rules §2.6.b) — bukan larangan mutlak.
- Ada **Efficiency Prize** terpisah (leaderboard beda urutan) — di luar fokus kita, target tetap skor mentah.
- **Leaderboard**: peringkat 1 = **Brandon Low, 0.950** macro ROC-AUC. Top-20 rapat di 0.934–0.950.
- Compute Kaggle: GPU T4x2 atau P100 tersedia; **P100 tidak kompatibel** dengan build PyTorch Kaggle saat ini (compute capability terlalu lama) — paksa `machine_shape: "NvidiaTeslaT4"` di kernel-metadata.json.

## 2. Keputusan Arsitektur Terkini (overwrite kalau berubah)
- **Backbone utama: DINOv2-base** (`metaresearch/dinov2`, Apache-2.0) — dominan dipakai kompetitor top, tervalidasi kompatibel `transformers.AutoModel`. DINOv2-small→base kapasitas TERBUKTI TIDAK berpengaruh (+0.0011 AUC, dalam noise floor) — **domain adaptation (unfreeze lebih dalam) adalah pengungkit nyata, BUKAN kapasitas/resolusi model** — sudah dikonfirmasi independen 2x.
- **Representasi studi**: "slot" bag (kombinasi anatomis plane×sekuens) + attention pooling per-diagnosis dengan prior anatomis hardcoded (`SLOT_PRIOR_TABLE`). Sampling `K_SLICE=4` (linspace-then-mean) per series — **belum diupgrade** ke teknik "3+3+3" (3 anchor merata + 2 tetangga fisik/anchor = 9 slice, representasi ~10mm depth asli) yang ditemukan lebih canggih di Discussion (735154).
- **Strategi weak-label**: label 58 studi gold **tidak pernah ditimpa** (selalu pakai annotator asli). 4.349 studi non-gold pakai label eksternal terbaik yang sudah diverifikasi macro AUC vs gold: **`flight0234/rsna-knee-hybrid-report-labels` (0.8991)**, dipakai sejak v8/v9 — plus confidence-weight `2*|p-0.5|` di loss BCE. Weak-labeler rule-based kita sendiri (`kaggle_kernel_weak_labeler/`) sudah selesai (agreement 0.747) tapi kalah akurat dari label eksternal, tidak lagi jadi sumber utama training.
- **Optimizer/fine-tuning (v11, breakthrough terbesar)**: `unfreeze_last=6` blok terakhir backbone (naik dari 2), LR terpisah `LR_HEAD=1e-3` / `LR_BACKBONE=8e-6` (nilai umum kode komunitas top), `gradient_checkpointing_enable()` wajib (hemat memori krn 6 blok trainable jauh lebih berat), `BATCH_STUDIES=4` (turun dari 8, batas T4 14.5GB dgn unfreeze=6).
- **Strategi resume/checkpoint**: pola self-reference `kernel_sources` (kernel modeling menambahkan slug dirinya sendiri ke `kernel_sources`-nya sendiri) — tervalidasi bekerja, dipakai utk resume training tanpa download lokal file besar. **Judul kernel Kaggle JANGAN diubah** kalau butuh slug stabil utk self-reference (title yg menyimpang dari slug asal memicu Kaggle rename URL).
- **Guard checkpoint wajib**: `best_auc` awal di-set ke skor checkpoint yang di-resume (bukan `-1.0`) supaya epoch tambahan yang lebih buruk TIDAK PERNAH menimpa checkpoint terbaik — pola ini WAJIB dipakai di setiap eksperimen resume ke depan.
- **Hal yang SUDAH diputuskan, TIDAK perlu didiskusikan ulang**:
  - Target akhir: kalahkan skor 0.950 (Brandon Low), fokus skor mentah bukan Efficiency Prize.
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
  - **Prioritas eksperimen berikutnya (updated 19 Agustus, pasca-v15)**: (1) item geometri lain yg masih tertunda — sampling "3+3+3", crop sadar-laterality, handle 2-lutut-1-DICOM — makin layak dicoba krn geometri sudah terbukti pengungkit nyata; (2) k-fold CV proper (bukan 1 split gold N=57, cukup noisy per-epoch); (3) skor lokal 0,8304 MENDEKATI tapi BELUM capai ambang "~0,85-0,88" utk masuk ensemble (roadmap lama §10) — masih perlu naik lagi sebelum digabung; (4) TTA flip (item murah, belum ada di pipeline kita).
  - **Gotcha operasional penting (masih berlaku)**: pola self-reference `kernel_sources` + guard checkpoint (v12) membuat checkpoint versi lama TIDAK BISA diakses kernel lain begitu versi terbaru kernel itu tidak menimpanya (`kernel_sources` selalu mount versi TERBARU, bukan versi spesifik). Kalau butuh akses checkpoint lintas-versi yg aman, simpan sbg Kaggle Dataset terpisah, jangan andalkan `kernel_sources` semata.

## 3. Status Pipeline (overwrite tiap update)
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
- **Model final terbaik proyek: val_macro_auc = 0,8304** (kernel `rsna-knee-modeling-v15-geom-v2-tpu`, epoch 6/8, TPU, cache preprocessing v2) — naik dari checkpoint v11 (0,8195). Skor publik TIM (submission nyata, terpisah dari model kita) sudah **0.920** (via fork `tonylica/rsna-knee-dino-radimagenet-rank-ensemble`, lihat §2) — peringkat **#130/1.886 tim**, sama dgn puluhan tim lain (commodity ceiling).
- **Preprocessing v2 (geometri slice) SELESAI PENUH**: `kaggle_kernel_preprocessing_v2/rsna-knee-preprocessing-v2-physical-slice-order`. 24.371/24.371 series (100%), ~72,5 menit, CPU-only. **100% pakai posisi fisik (`ipp`), 0 fallback**. Output 12,94 GB.
- **v15 (modeling, cache v2, TPU) SELESAI — MODEL TERBAIK BARU**: `kaggle_kernel_modeling_v2geom/rsna-knee-modeling-v15-geom-v2-tpu`. val_macro_auc 0,8304, naik dari v14 (0,8159, cache v1) +0,0145 dan checkpoint v11 (0,8195) +0,0109 — SATU-SATUNYA variabel beda dari v14 adalah data source, geometri slice TERBUKTI pengungkit nyata.
- **Modeling v13/v14 (TPU, cache v1) SELESAI**: `kaggle_kernel_modeling_tpu/rsna-knee-modeling-v13-tpu-sanity`. v13 fix bug XLA (grad checkpointing) + temuan operasional (checkpoint lama tidak ketemu via kernel_sources). v14 (12 epoch dari nol, LR decay): 0,8159 — TIDAK melampaui v11, sekarang SUDAH TERLAMPAUI oleh v15.
- **Preprocessing v3 (sampling "3+3+3") SELESAI PENUH**: `kaggle_kernel_preprocessing_v3/rsna-knee-preprocessing-v3-3x3-anchor-sampling`. 3 anchor merata di jendela tengah stack + 2 tetangga fisik/anchor = 9 slice/series, dipasang sbg channel RGB genuine (bukan grayscale-repeat). 24.371/24.371 OK (0 error), 100% `n_slices_cached=9`, 100% `pos_source=ipp`. Output 7,53 GB, ~67,7 menit.
- **v16 (modeling, cache v3, TPU) SEDANG BERJALAN**: `kaggle_kernel_modeling_v3geom/rsna-knee-modeling-v16-3x3-anchor-tpu`. `KneeModel.forward` diubah — `.repeat(1,3,1,1)` dihapus, channel RGB sekarang genuine. Pembanding: v15 (cache v2) = 0,8304 (terbaik saat ini). Push 19 Agustus 18:35.

## 4. Constraint & Batasan Aktif
- **Eksekusi ML HANYA di Kaggle** — sandbox lokal tidak punya `sklearn`/`torch`, disk ~4.7GB free. Semua verifikasi lokal pakai AUC manual (numpy rank-based, Mann-Whitney U).
- **GPU wajib T4 (T4x2), HINDARI P100** (incompatible build PyTorch Kaggle) — set `machine_shape: "NvidiaTeslaT4"` di tiap `kernel-metadata.json`.
- **Kuota GPU mingguan OVERRUN, masih 0.00h per 18 Agustus 2026** (`used` 30.43h dari 30.00h nominal). `refreshAt`: **22 Agustus 2026**. → **JANGAN push kernel GPU baru sebelum cek `kaggle quota` dulu.**
- **TPU sudah diverifikasi bisa dipakai** (probe `rsna-knee-tpu-diagnostic-torch-xla-check`, 18 Agustus): `torch_xla` 2.8.0 pre-installed di image Kaggle, `xm.xla_device()` + matmul + `transformers` import semua berhasil. **Antrian nyata terukur: ~28,6 menit dari push sampai `RUNNING`**, eksekusi ~1 menit tambahan. Kuota TPU 20h/minggu 100% belum kepakai. **Caveat**: kode training kita (CUDA-based) HARUS ditulis ulang pakai API `torch_xla` (`xm.xla_device()`, `xm.mark_step()`) — bukan cuma toggle `enable_tpu`; idealnya juga di-pad ke shape tetap dulu (XLA re-compile mahal kalau shape variabel) — biaya rekayasa nyata, belum dikerjakan.
- **Leaderboard sungguhan (snapshot 18 Agustus 2026)**: #1 Brandon Low **0.951** (naik dari 0.950 hari ini). Top 20 cutoff 0.937. Top 50 cutoff 0.929. **Tim kita (Krazy KIller, #130/1886) di 0.920** — sama persis dgn puluhan tim lain yg fork notebook publik sama (`tonylica/rsna-knee-dino-radimagenet-rank-ensemble`, dibedah forensik di §2 & RESEARCH_NOTES §16). 1.886 tim total, kompetisi baru 12 hari jalan, gerak leaderboard sangat cepat.
- **Submission harian dibatasi 5x/hari UNTUK SELURUH TIM** (bukan per-orang) — tim sudah punya kolaborator aktif (`nxtirfan`, kemungkinan "Muhammad Irfan") yang submit sendiri, riwayat 30+ submission sejak 6 Agustus 2026, skor terbaik tim **0.914** (16 Agustus 2026). → **Jangan submit sembarangan, koordinasi dulu supaya tidak boros kuota harian tim.**
- **API LLM hosted**: BOLEH dipakai (ruling resmi host topic 733965), syarat layanan "reasonably accessible & minimal cost" ke semua peserta (Rules §2.6.b) — **status: SUDAH dikonfirmasi resmi, bukan lagi perlu verifikasi**. Implementasi (relabeling 4.349 studi non-gold via LLM) masih ditunda, prioritas saat ini di single-model strengthening.
- **Lisensi NonCommercial (CC-BY-NC)** — di-skip sbg kebijakan proyek (risiko lisensor asli, bukan larangan Rules kompetisi langsung).
- **Repo GitHub PRIVATE**, remote `https://github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection.git`, branch `master`. **Ada kolaborator lain dgn push access** (`nxtirfan`/Muhammad Irfan, sudah push commit README sendiri) — cek `git pull`/konflik sebelum push besar.
- **Commit HARUS TANPA trailer Co-Authored-By Claude** — aturan mutlak, berlaku semua commit di repo ini. Pesan commit multi-baris/berisi tanda kutip WAJIB pakai `git commit -F <tempfile>`, jangan inline `-m "..."` (pernah menyebabkan shell-quoting mishap).

## 5. Open Questions / Menunggu Keputusan User
- **Koordinasi dgn tim** soal submission — apakah submission pipeline v1 (community-20-model, ~0.891) masih relevan disubmit sendiri, atau ditahan total krn tim sudah punya jalur skor lebih tinggi (0.920, fork tonylica)? Belum ada arahan eksplisit.
- **Hasil v16 (sampling 3+3+3) belum diketahui saat catatan ini ditulis** (§3) — kalau val_macro_auc > 0,8304: representasi channel RGB genuine terbukti pengungkit tambahan, lanjut item geometri sisa (crop laterality, 2-lutut-1-DICOM) atau k-fold. Kalau plateau/turun: urutan fisik saja (v15) sudah menangkap manfaat utama, geometri channel bukan pengungkit besar tambahan — pertimbangkan k-fold CV atau mulai arah ensemble.
- **Review manual** studi gold `...72853333...` yang ter-flag "exclude" di audit exhaustive (kemungkinan anomali kalibrasi/overexposure) — apakah tetap dipakai di training/validasi atau dikeluarkan?

## 6. Referensi Cepat
- Kompetisi: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- **Kernel aktif (source of truth modeling, model terbaik 0,8304)**: `dimaspashaakrilian/rsna-knee-modeling-v15-geom-v2-tpu` — https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-modeling-v15-geom-v2-tpu
- Kernel modeling v16 (eksperimen sampling 3+3+3, sedang berjalan): `dimaspashaakrilian/rsna-knee-modeling-v16-3x3-anchor-tpu`
- Kernel preprocessing v2 (urutan fisik, dipakai v15): `dimaspashaakrilian/rsna-knee-preprocessing-v2-physical-slice-order`
- Kernel preprocessing v3 (sampling 3+3+3, dipakai v16): `dimaspashaakrilian/rsna-knee-preprocessing-v3-3x3-anchor-sampling`
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
- Discussion penting: topic 733965 (ruling LLM API), topic 735304 (single-model score), topic 735154 ("3+3+3" slice sampling), topic 735121 (isu lisensi NC RadImageNet, belum dijawab host), topic 735826 (bottleneck: geometri+label > backbone size), topic 735767 (final ceiling + inkonsistensi gold label), topic 735854 (PSA: codec DICOM offline + path submission), topic 735639 (2 lutut 1 DICOM)
- Kernel resep 0,920 tim (dibedah forensik, bukan dipakai langsung): `tonylica/rsna-knee-dino-radimagenet-rank-ensemble` — https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble

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
