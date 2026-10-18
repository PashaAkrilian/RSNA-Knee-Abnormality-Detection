# Teknik Ensembling/Fusion untuk Naik dari 0,950 ke ≥0,951 (RSNA Knee 2026)

> **ARSIP/STATUS 10 Okt 21:30 WIB (sesi Claude Code)**: dokumen riset 9–10 Okt (OpenCode). Rencana Fase A/B (mirror/pseudo-label) DIGANTI oleh `STRATEGI_PERAK.md` (merger + Apex+OrthoFoundation + finals + kepatuhan). Fakta terbaru: `MEMORY.md`; riset terbaru: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`.


> Riset publik per 9 Okt 2026. Posisi: stack "Apex" (ckpt nartaa r384 + reader ConvNeXt goodpjw, rank-mean) = LB publik 0,950; perak butuh ≥0,951 (257 tim sudah ≥0,951 per papan 9 Okt — [MEMORY.md](../../MEMORY.md) §1, internal; sumber data = leaderboard publik Kaggle).
> Klaim berlabel "(internal)" berasal dari pengukuran tim di repo ini (MEMORY.md/PROGRESS.md), bukan sumber luar.

## Q1 — Teknik fusion apa yang dipakai peserta top RSNA Knee 2026, dan berapa liftnya?

### Takeaway
Rank-mean fusion adalah standar de-facto di kompetisi ini; satu-satunya teknik fusion dengan bukti numerik terpublikasi adalah (a) aturan per-label include/exclude arm, (b) per-label pooling (mean vs max di ruang probabilitas), dan (c) gating/spesialis per-label (ada pipeline-nya, TANPA angka lift). Tidak ada notebook/discussion publik ≥0,951 yang bisa dibuka (halaman Kaggle dirender JS), sehingga lift per teknik di zona 0,951+ tidak tersedia secara publik — yang membedakan stack 0,950 vs ≥0,951 kemungkinan besar adalah arm/kualitas model, bukan rumus bobot.

### Cited Findings
- **Rank-rata (percentile rank average), bukan rata-rata probabilitas, adalah konsensus semua notebook top**: "Because the metric is macro AUC, only the order of predictions matters... Every top notebook independently derives the same conclusion... combines models by averaging percentile ranks, never raw probabilities" — [research-report.md (ahmed-hesham07, kompilasi 353 notebook, 18 Agu 2026)](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md)
- **Bobot per-label (include/exclude arm)**: "hand-tuned per-label mixing weights (e.g., RadImageNet explicitly excluded from Baker's cyst and Fracture predictions in two independent notebooks — a convergent finding)" — [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md)
- **Notebook Tony Li "DINO-RadImageNet Rank Ensemble"**: public 0,940 (V11); resep sumber "verified 0.941 public LB"; nested rank-blend 35 checkpoint / 5 arsitektur (20 DINOv2-small, 5 DINOv3-small, 2 varian RadImageNet ResNet-50, dst.) — [Kaggle notebook](https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble); [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md). Catatan: commit yang tampil tidak dieksekusi (Accelerator: None) — periksa tab Logs sebelum percaya skor.
- **Aturan per-label pooling (max vs mean)**: reproduksi publik "Renta V5" = 20 member rank-average seragam + rata-rata probabilitas untuk 10 target + **probabilitas MAKSIMUM untuk Fracture & Contusion** → public 0,897 (bandingkan anchor 0,825) — [experiment_log.md (Beiciccc)](https://raw.githubusercontent.com/Beiciccc/rsna-knee-abnormality-detection/main/experiments/experiment_log.md), [public_scores.md](https://raw.githubusercontent.com/Beiciccc/rsna-knee-abnormality-detection/main/results/public_scores.md). Caveat: sebagian besar lift datang dari jumlah model (20 vs anchor), bukan aturan max-nya.
- **Five-arm rank ensemble (tiga arm dipilih → rank mean)**: EXP003 = public 0,839 vs anchor 0,825 — [experiment_log.md](https://raw.githubusercontent.com/Beiciccc/rsna-knee-abnormality-detection/main/experiments/experiment_log.md)
- **Gating/spesialis per-label**: pipeline "Full Train → OOF → Specialist → Gating → Submit" ada di notebook A³-Knee v2 (Best 0,911) — [Kaggle notebook sakhawathossen](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble); tapi commit tampilan tidak dieksekusi ([research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md)). Tidak ada angka lift gating yang terpublikasi.
- **Consensus + "weight governance"**: notebook amanatar (public 0,943) memuat "4-Way CoAtNet Consensus" dan "weight governance / ELITE BOOST governance" — [Kaggle notebook](https://www.kaggle.com/code/amanatar/rsna-knee-abnormality-detection). Notebook nishantkharga "Full 4-arm ensemble" = 0,936 ("Anhad-style 4-arm") — [Kaggle notebook](https://www.kaggle.com/code/nishantkharga/rsna-knee-full-4-arm-ensemble-v55).
- **Arm kuat bernilai lebih daripada teknik fusion**: testimony top-8 Tucker Arrants — single model 0,934→0,942; "giant 20+model public ensembles are chasing notebook votes" — [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md) + thread [Best single-model score (735304)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304). Catatan internal: bacaan tim atas thread yang sama — model tunggal ~0,951 lalu ensembel → 0,959+ (internal, MEMORY.md §2).
- **Di tim kita (internal)**: rank-mean + reader goodpjw menambah +0,001 (0,943→0,944); stack publik Apex 0,950 = nartaa r384 + goodpjw — [MEMORY.md](../../MEMORY.md) §2–§3.

### Inferences
- Karena metrik macro-AUC, "bobot per-label" hanya efektif lewat perubahan URUTAN → teknik paling rasional & berbukti: (1) buang arm yang lemah pada label tertentu, (2) max/top-k pooling untuk label fokal (Fracture, Contusion, Baker's — konsisten dengan temuan per-finding max vs mean di ["Let me Cook" 0,899](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook) sebagaimana dikutip research-report.md).
- Gating = generalisasi dari include/exclude; diperlakukan sebagai hipotesis, bukan fakta.
- Satu-satunya jalur realistis ke ≥0,951 = menambah arm independen kuat (mis. r224 sibling / reader) dengan rank-mean; merombak bobot pada stack yang sama kecil kemungkinannya menembus noise ±0,002.

### Gaps
- Discussion [742926 "Suggestions for improving a 0.940 RSNA Knee ensemble"](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926) dan [746792 (resep nartaa 0,949)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) tidak bisa di-fetch (render JS, tanpa konten) → klaim spesifik dari thread itu tak bisa dikutip di sini.
- Code tab tak bisa diurutkan/di-scrape → daftar notebook publik ≥0,951 beserta tekniknya tidak tersedia. Sejauh yang terdokumentasi, notebook publik tertinggi = 0,950 (Apex, internal) / 0,941–0,943 (Tony Li verified, amanatar).

## Q2 — Dari post-mortem kompetisi imaging lain: teknik yang konsisten menambah vs placebo

### Takeaway
Konsisten menambah: (1) rata-rata rank/bobot antar model DIVERSEN (3–4 arm; hampir semua writeup juara), (2) aturan per-target/max-pooling, (3) TTA/EMA/ensemble langkah kecil +0,001–0,003. Placebo/terbukti tak menambah: backbone lebih besar, menambah model satu keluarga, label smoothing. Optimisasi bobot halus memberi lift jauh di bawah noise (orde +0,0002).

### Cited Findings
- **RSNA 2019 ICH (juara 1, SeuTao)**: 15 ekstraktor intra-slice (SE-ResNeXt101/DenseNet169/DenseNet121 × 5 fold) + 2 sequence model, final submission weighted ensemble — [github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection). Paper pembanding: **model 3-arm weighted average (log-loss 0,04345) BEATING ensembel 20 model juara (0,04383)** — [arXiv:2205.07556](https://ar5iv.labs.arxiv.org/html/2205.07556) → kualitas & dekorelasi arm > jumlah model.
- **RSNA Intracranial Aneurysm 2025 (6th place)**: max-aggregation frame 0,87 CV → +sequence model 0,88 → **3-model ensemble 0,895 CV** (publik 0,84) — [github.com/TheoViel/RSNA-Intracranial-Aneurysm](https://github.com/TheoViel/RSNA-Intracranial-Aneurysm). Ensembel = +0,015 CV di akhir; tiap trik augment/crop ≈ +0,01 CV.
- **SIIM-FISABIO-RSNA COVID 2021 (juara 1)**: weighted ensembling 4 model segmentation "0.3[eb5] + 0.2[eb6] + 0.2[eb7] + 0.3[sr152]"; **EMA "increase LB by +0.001~0.002"**; single-fold lung detector "+0.003"; detection ensemble 4 model (weighted boxes fusion IoU 0.6) — [Kaggle writeup 1st place](https://www.kaggle.com/competitions/siim-covid19-detection/writeups/dungnb-1st-place-solution).
- **RSNA 2024 Lumbar (5th place)**: "ensemble: various backbone ensemble + tta-like ensemble"; **varian TTA-like = skor private tertinggi**; daftar "Not worked: bigger CNN backbone for stage2, label smoothing" — [github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification).
- **RSNA 2024 Lumbar (7th place, hengck23)**: model one-stage SCS "did not improve the public LB score when added to the team solution" (placebo nyata) — [github.com/hengck23/solution-rsna-2024-lumbar-spine](https://github.com/hengck23/solution-rsna-2024-lumbar-spine).
- **RSNA 2022 Cervical (5th place, Speedrun)**: "ensemble of two quite similar approaches" 3-stage — [github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place](https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place); evaluasi resmi 8 model teratas: mean AUC 0,96 (95% CI 0,95–0,96) — [Radiology:AI (PubMed 38169426)](https://pubmed.ncbi.nlm.nih.gov/38169426/).
- **RSNA Knee 2026 (null results terdokumentasi)**: DINOv2-base vs small = +0,001 (di dalam noise), DINOv3 lebih buruk (0,763 vs 0,775); "scaling the encoder bought us nothing" — [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md) + thread [735154](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154).
- **Dekorelasi > kualitas (analogi label)**: blend dua keyset label independen 0,8780 → 0,8927 vs gold-58; "the weaker standalone label set adds more lift... what matters is decorrelation" — [dataset stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels).
- **Optimisasi bobot = lift kecil**: Bagged Hill Climbing (50 bag × 2000 iter) vs Ridge non-negatif: 0,70771 vs 0,70751 (delta +0,0002) — [writeup 2nd place PS-S5E12](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an). (tabular, bukan imaging — indikasi orde besar, bukan bukti langsung)
- **Rank-average seragam = baseline kuat yang dipakai juara-juara**: "plain equal-weight rank average — no weight search, no meta-model, no tuning against the public leaderboard" — [Kaggle S6E9 rank-average notebook](https://www.kaggle.com/code/talhatursun/s6e9-simple-rank-average-ensemble); "Weighted Rank Average" klasikel meta-ensemble — [Kaggle general/20728](https://www.kaggle.com/general/20728); weighted rank average multi-CV — [kaz-Anova/ensemble_amazon](https://github.com/kaz-Anova/ensemble_amazon/blob/master/AUC_Rank_Weighted_Average.py).
- **Literatur akademik**: pembobotan rank (Borda) berbobot dengan bobot ter-estimasi (SUMMA) mengalahkan rata-rata rank seragam pada data uji (AUC 0,95 vs 0,89; model terbaik tunggal 0,74, eksperimen sintetis) — [JMLR v20: Unsupervised Evaluation and Weighted Aggregation of Ranked Classification Predictions](http://jmlr.org/papers/volume20/18-094/18-094.pdf).

### Inferences
- Urutan nilai harapan: **arm baru yang kuat & berkorelasi rendah ≫ per-label exclude/pooling > bobot tak-seragam halus > TTA/EMA kecil**; placebos: model sekeluarga tambahan, backbone lebih besar (konsisten dengan temuan internal tim: bagging tanpa OAI tak menolong — [MEMORY.md](../../MEMORY.md) §2).
- Angka +0,001–0,003 dari TTA/EMA/3-model ensemble di writeup juara = satu-satunya "lift kecil terbukti" yang cocok dengan skala kebutuhan kita (+0,001–0,002).

### Gaps
- VinDr-CXR: writeup juara tidak ditemukan (hanya paper dataset [Nature Sci Data](https://www.nature.com/articles/s41597-022-01498-w) dan repo juara-2 [ZFTurbo](https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection) yang tidak dibuka).
- Tidak ada A/B terkontrol "rank vs probabilitas" berskor LB di kompetisi imaging — klaim rank>prob adalah konsensus komunitas + teori AUC (lihat Q1), bukan hasil eksperimen tunggal.

## Q3 — Memilih bobot arm secara objektif dengan N kecil (58 gold) tanpa overfit LB publik

### Takeaway
Dengan 58 gold label (SE per AUC ≈0,01) dan noise LB ±0,002, optimisasi bobot halus mustahil dibedakan dari noise; pakai model bobot sederhana (2–3 arm, langkah kasar) dengan **shrinkage ke bobot seragam**, seleksi per-label hanya bila konsisten antar-fold, dan validasi dari OOF/CV — bukan dari LB publik.

### Cited Findings
- **Ukuran efek optimisasi bobot sangat kecil**: hill-climb vs ridge +0,0002 — [writeup 2nd place PS-S5E12](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an).
- **Overfitting LB kompetisi klasifikasi umumnya kecil, TAPI test-set kecil = outlier berisiko**: "classification competitions on Kaggle show little to no signs of overfitting... effect sizes typically small (<1%)... Very small (effective) test sets make it easier to reconstruct the public/private split (and then to overfit), and also make the public and private scores more noisy" — [A Meta-Analysis of Overfitting in Machine Learning, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf).
- **Overfitting adaptif tumbuh seperti √(k/n)**: "the error between the public leaderboard and actual performance can be large as O(√(k/n)), k is number of submission" — [The Ladder: A Reliable Leaderboard (slides, Columbia)](https://www.cs.columbia.edu/~djhsu/coms6998-f17/student-work/slides/leaderboards.pdf).
- **Tanpa OOF tak bisa membedakan sinyal vs noise**: "Without OOF validation, it is impossible to determine whether a subtle score bump is a generalized improvement or merely an accidental alignment with the public split noise"; LB-only ensembling = mengoptimalkan 20–30% data evaluasi — [Kaggle notebook Ensemble of Ensembles (najiama)](https://www.kaggle.com/code/najiama/ensemble-of-ensembles-0-97100).
- **Bobot seragam (WOC) adalah baseline kuat yang harus dikalahkan**: rata-rata rank tanpa bobot; metode berbobot butuh estimasi bobot dari data berlabel/unsupervised — [JMLR SUMMA](http://jmlr.org/papers/volume20/18-094/18-094.pdf).
- **Data internal**: gold-58 SE ≈0,01 per AUC; noise antar-latih pada LB 3 desimal ≈0,002; ukuran publik split tak diketahui (test ≈1.322) — [MEMORY.md](../../MEMORY.md) §2.

### Inferences (resep praktis, tidak ada satu sumber yang merangkum semuanya)
1. **Model bobot**: hanya 2–3 arm; rank-space; grid langkah 0,05–0,10; konstrain w≥0, ∑w=1. Jangan 12 bobot per-label bebas — 12×parameter dengan 58 observasi = pasti overfit (12 label × SE 0,01).
2. **Shrinkage**: w_final = λ·w_optimal + (1−λ)·w_uniform dengan λ≈0,5; naikkan λ hanya bila optimal weight stabil (selisih <0,1 antar-fold).
3. **Seleksi per-label = include/exclude saja, dengan syarat bukti**: pakai arm X untuk label L hanya bila ΔAUC_OOF > 2×SE dan tanda lift konsisten di ≥4/5 fold (mengikuti pola "RadImageNet dibuang di Baker's/Fracture" — dua notebook independen; [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md)). Selain itu: bobot seragam.
4. **Nested CV versi murah**: pilih bobot pada fold 0–3, evaluasi pada fold sisa (leave-one-fold-out); ulangi 5× — kalau ΔOOF < 0,003 (≈noise internal), jangan pakai bobot itu.
5. **Jangan pakai LB publik untuk memilih bobot**: 5 submisi/hari, 2 dipakai final; tiap probe = satu langkah √(k/n) overfitting ([Ladder](https://www.cs.columbia.edu/~djhsu/coms6998-f17/student-work/slides/leaderboards.pdf)). LB hanya untuk keputusan struktural (arm baru masuk/tidak) dengan ambang ≥ +0,001 dan konfirmasi arah yang sama di OOF.
6. **Gold-58 untuk screening arah**, bukan ranking presisi: dengan SE 0,01, beda 0,01–0,02 pun belum tentu signifikan per-label (internal).

### Gaps
- Tidak ditemukan (dalam ~23 panggilan riset) paper/panduan spesifik "shrinkage untuk bobot ensemble dengan N sangat kecil" yang bisa dikutip langsung — resep di atas adalah inferensi dari temuan Q2–Q4; Super Learner (nested-CV stacking) tidak berhasil di-fetch sebagai sumber.

## Q4 — Batasan LB noise ±0,002 pada 3 desimal: membedakan sinyal vs noise dalam 1–2 submisi

### Takeaway
Dengan noise ±0,002 dan maks 2 submisi final: delta <0,002 = noise; 0,002–0,003 = lemah (butuh konfirmasi OOF); ≥0,004 = sinyal. Bedakan sinyal dengan menguji DUA HIPOTESIS BERBEDA SECARA STRUKTUR (bukan dua titik bobot), plus konfirmasi arah di OOF/gold-58, dan pastikan keputusan tidak bergantung pada satu submisi.

### Cited Findings
- **Noise empiris internal**: noise antar-latih pada LB 3 desimal ≈0,002; garis perak bergerak 0,945 (3 Okt) → 0,950 (8 Okt) → proyeksi 0,951–0,953 (22 Okt) — [MEMORY.md](../../MEMORY.md) §1–§2 (internal; data = leaderboard publik Kaggle).
- **Biaya adaptif per submisi**: error LB tumbuh O(√(k/n)) — [Ladder slides](https://www.cs.columbia.edu/~djhsu/coms6998-f17/student-work/slides/leaderboards.pdf).
- **Test kecil → publik & private lebih noisy**: [NeurIPS 2019 meta-analysis](https://papers.neurips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf). Konteks: split publik RSNA Knee kemungkinan kecil dari ~1.322 studi test (ukuran pasti tak diketahui — internal).
- **Era kode-kompetisi: shakeup publik→privat besar** (sumber komunitas, kualitas sedang — pakai sebagai peringatan, bukan fakta): "in the modern code-competition era, a tiny public LB means the public and private boards are close to uncorrelated (2023+ median: 0.96)" — [shakeup-radar (GitHub, kalibrasi Meta Kaggle)](https://github.com/tusharhsharma/shakeup-radar).
- **Disiplin pencatatan**: skor hanya dicatat setelah muncul di tabel submisi resmi — [experiment_log.md (Beiciccc)](https://raw.githubusercontent.com/Beiciccc/rsna-knee-abnormality-detection/main/experiments/experiment_log.md).
- **Seringkali naik 0,001 = artefak**: kenaikan reader di stack internal (0,949→0,950) dan r224 (0,9223→0,9239 di gold-58) berada tepat di batas noise — internal (MEMORY §2–§3).

### Inferences (aturan operasional untuk 1–2 submisi)
1. **Ambang**: ΔLB < 0,002 → tolak / anggap noise. Δ = +0,001 → hanya dipakai jika OOF juga naik dengan arah sama. Δ ≥ +0,002 dengan arah OOF sama → terima.
2. **Dua submisi final harus beda mekanisme**, mis.: (A) stack 0,950 + 1 arm baru independen (r224/reader) dengan bobot tetap; (B) arm baru + aturan per-label exclude (bukti internal/kerangka Q1). Bukan dua varian bobot 0,6/0,4 vs 0,7/0,3 — keduanya akan berbeda <noise.
3. **Triangulasi sebelum submit**: setiap kandidat dinilai di (i) OOF fold, (ii) gold-58 (arah saja), (iii) prinsip literatur (Q2). Hanya kandidat yang lulus 2 dari 3 yang layak memakai kuota submisi.
4. **Rencana jika garis perak naik ke 0,953**: butuh +0,003 → rekayasa bobot pada stack lama tidak akan cukup (lihat Q1: lift teknik ≪ arm baru); prioritaskan arm tambahan / variance reduction (TTA/EMA/seed-avg, terbukti +0,001–0,003 di writeup juara — [DungNB](https://www.kaggle.com/competitions/siim-covid19-detection/writeups/dungnb-1st-place-solution)).
5. **Jangan re-submit varian identik** untuk "mengukur noise" — kuota 5/hari terlalu berharga dan setiap probe menambah risiko adaptif (Ladder).

### Gaps
- Angka ±0,002 berasal dari pengamatan internal tim, bukan literatur; SE eksak LB publik tak bisa dihitung tanpa tahu ukuran split publik (host tidak mempublikasikan).
- Tidak ditemukan post-mortem yang melaporkan A/B "dengan vs tanpa optimisasi bobot" dengan angka privat di kompetisi imaging — klaim tentang besaran lift optimisasi bobot (+0,0002) berasal dari kompetisi tabular.

## Ringkasan aksi (dari bukti di atas)

| Prioritas | Teknik | Bukti (angka) | Sumber |
|---|---|---|---|
| 1 | Tambah arm independen kuat + rank-mean | +0,001/stack internal; aneurysm +0,015 dari ensemble; juara ICH menang lewat ensemble arm beragam | internal; [TheoViel](https://github.com/TheoViel/RSNA-Intracranial-Aneurysm); [SeuTao](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection) |
| 2 | Per-label exclude arm lemah (Baker's/Fracture dst.) | konsvergensi 2 notebook independen (tanpa angka lift) | [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md) |
| 3 | Per-label max/top-k pooling utk label fokal | Renta V5 0,897 (campuran dgn efek jumlah model); "Let me Cook" max vs mean | [experiment_log.md](https://raw.githubusercontent.com/Beiciccc/rsna-knee-abnormality-detection/main/experiments/experiment_log.md) |
| 4 | TTA/cermin/seed-avg | +0,001–0,003 (writeup juara SIIM); internal +0,0007 (cermin) | [DungNB](https://www.kaggle.com/competitions/siim-covid19-detection/writeups/dungnb-1st-place-solution); internal |
| 5 | Gating/OOF-learned weights | pipeline ada, angka tak dipublikasikan; optimisasi bobot halus +0,0002 (tabular) | [A³-Knee](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble); [PS-S5E12](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an) |
| ✗ | Backbone lebih besar, model sekeluarga tambahan, label smoothing | +0,001 = noise / tak menambah / lebih buruk | [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md); [hengck23](https://github.com/hengck23/solution-rsna-2024-lumbar-spine); [siwooyong](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification) |

## Catatan status terkini — 19 Okt 2026 02:56 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; cutoff ini sementara karena medali memakai split privat 70%). Pada cek API 02:55, `g6c`, `g12`, `g13` masih PENDING. Dua kernel murid r4 di alver sudah dua kali di-requeue dan kini RUNNING pada cek 02:56. Kuota main/alver/nxtirfan: 19,87/17,75/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
