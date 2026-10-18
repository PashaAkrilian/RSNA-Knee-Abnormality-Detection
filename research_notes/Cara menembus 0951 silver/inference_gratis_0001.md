# Inference/Variance-Reduction GRATIS: Hipotesis +0,001 menuju LB 0,951 (RSNA Knee 2026)

> **ARSIP/STATUS 10 Okt 21:30 WIB (sesi Claude Code)**: dokumen riset 9–10 Okt (OpenCode). Rencana Fase A/B (mirror/pseudo-label) DIGANTI oleh `STRATEGI_SILVER.md` (merger + Apex+OrthoFoundation + finals + kepatuhan). Fakta terbaru: `MEMORY.md`; riset terbaru: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`.


> Riset 10 Okt 2026 (READ-ONLY di Kaggle; tidak ada push/submit/ubah kernel). Posisi: stack Apex 0,950 (r384 SWA nartaa ⊕ reader ConvNeXt goodpjw [fold0–2] ⊕ r224 sibling, rank-blend, mirror plane-aware, native crop 320, K94). Silver = ≥0,951 (rank ≤257 dari 5.584 tim per papan 9 Okt) — [MEMORY.md](../../MEMORY.md) §1.
> Sumber utama: MEMORY.md/laporan_push.md/reports (internal), forum 746792 (Danial Zakaria/nartaa, via `kaggle forums topics show`), daftar file dataset via `kaggle datasets files` (akun2, read-only), notebook `/tmp/opencode/nartaa0949/rsna-knee-0949-anatomical-mirror.ipynb`, `kaggle_base/apex_grandmaster_stack_20261008.ipynb`, `kaggle_kernel_hengck_gold_a2/mavit-288-gold58-eval.ipynb`.

## Q1 — Inventaris dataset `nartaa/rsna-knee-publication-swa-weights-20261007`: ada arm tambahan?

### Takeaway
Dataset publik nartaa **hanya berisi 2 checkpoint SWA** (r384 akurasi + c08_r224 efisiensi) + README + manifest. **TIDAK ADA** epoch terpisah (SWA manual mustahil), fold lain, seed lain, atau resolusi lain. Model crop-trained 0,946 yang disebut README **dirilis TIDAK** — tak jadi opsi arm.

### Cited Findings
- Daftar file lengkap (CLI `kaggle datasets files`, 10 Okt): `README.md` (2.983 B) · `checkpoint-manifest.json` (645 B) · `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (292,8 MB) · `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (292,8 MB). Total ~543 MB — [dataset nartaa](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007)
- Manifest: accuracy = r384 SWA (avg epoch **14,11,12**, sub 56906959 = **0.949**); efficiency = c08_r224 SWA (avg epoch **15,13,12**, sub 56707773 = **0.945**). Keduanya seed 42, 16 epoch — [checkpoint-manifest.json](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) (diunduh 10 Okt)
- README: "The later crop-trained 0.946 model **is not substituted**" — model 0,946 (native-320-crop training) ADA secara internal tapi TIDAK dirilis; juga "the unsuccessful 12-month arm is excluded from these release models" — [README.md](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007)
- Kedua model dilatih dgn 2.399 lutut OAI ("masked external supervision"); gold-58 dipakai ulang utk seleksi ("not independent cross-validation") — [forum 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- Notebook inferensi resmi menyebut dependensi privat `nartaa/rsna-knee-oai-original384-s42-pair` (privat, tak bisa diakses) — [forum 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

### Inferences
- **SWA epoch-averaging manual & seed-averaging: TERTUTUP** (tidak ada per-epoch ckpt / seed lain di publik). Satu-satunya "multi-ckpt nartaa" = r384 ⊕ r224 (sudah teruji LB: 0,950 = noise).
- Tidak ada arm tambahan yang bisa diekstrak dari paket nartaa; semua varian inference harus dimodifikasi DARI 2 ckpt ini.

### Gaps
- Model 0,946 crop-trained & 12-month arm sengaja tidak dirilis — tak bisa dinilai.

## Q2 — Dataset `raptor-knee-arms` & `raptor-cnn336` (milik siapa, isi apa, layak arm?)

### Takeaway
Kedua dataset itu milik **dreaddevelopment** (bukan rabari) = **arm baseline Raptor era korpus LAMA** (skor ~0,88–0,924, kalah jauh dari nartaa 0,949) dan — kritis — **kemungkinan besar terkontaminasi gold-58** (korpus lama memuat 45 dari 58 studi gold saat training) sehingga evaluasi gold-58 atas arm-arm ini TIDAK sah. Fork rabari sendiri HANYA memakai nartaa + goodpjw (bukti: `kernel-metadata.json` fork kita).

### Cited Findings
- `dreaddevelopment/raptor-knee-arms` (1,04 GB): `raptor_ft_cnv2b336.pt` (352 MB), `raptor_ft_coatnet384.pt` (293 MB), `raptor_ft_effv2l480.pt` (473 MB) — [dataset](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-arms) (CLI 10 Okt)
- `dreaddevelopment/raptor-cnn336` (185 MB): `raptor_ft_cnn336.pt` — [dataset](https://www.kaggle.com/datasets/dreaddevelopment/raptor-cnn336) (CLI 10 Okt)
- Fork rabari kita (`kaggle_kernel_rabari_dual_a2/kernel-metadata.json`): `dataset_sources` = **hanya** `goodpjw2008/rsna-knee-2-5d-convnext-reader` + `nartaa/rsna-knee-publication-swa-weights-20261007` (internal, 10 Okt); hasil LB 0,950 = BUANG — [laporan_push.md](../../laporan_push.md) §2 baris 8
- Kontaminasi: komentar sumber di notebook nartaa/Apex — korpus sebelumnya "held 3,200 of the 4,349 labelled studies and **only 45 of the 58 gold studies**", diperluas ke 4.407 dgn "58-study gate" (gold dikeluarkan) — [forum 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) (komentar `ARMS` di source nartaa/Apex)
- Kekuatan arm lama (panel emas 45-studi nartaa, terkontaminasi): coatnet384 0,9025 · cnn336 0,8833 · cnbase384 0,8754 · cnlarge384 0,8752 · maxvit384 0,8438; blend 3-arm 0,9068 vs coatnet solo 0,9025; "on the live leaderboard CoAtNet alone scored 0.914 while every blend scored 0.914–0.915" — source `ARMS` di notebook nartaa (idem 746792)
- goodpjw reader publik berisi **3 fold**: `cnxt_v0_fold0/1/2.pt` + kode (`infer.py` merata-rata prob multi-ckpt) — [dataset goodpjw](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) (CLI 10 Okt); **Apex kita sudah memakai ketiga fold** (`_o_ck = glob("cnxt_v0_fold*.pt")` → `infer.py` → `.mean(0)` per model) — [kaggle_base/apex_grandmaster_stack_20261008.ipynb](../../kaggle_base/apex_grandmaster_stack_20261008.ipynb) sel §Fusion
- Arm lain publik (single ckpt, provenance tak terdokumentasi): `tbongc/rsna-knee-raptor-v8geo-v7w-f-1-s42-imagenet` & `...-v10geo-...` (293 MB, fold-1, ImageNet-init), `royalacecat/raptor-knee-coatnet384-full-bs14` (293 MB) — [tbongc](https://www.kaggle.com/datasets/tbongc/rsna-knee-raptor-v10geo-v7w-f-1-s42-imagenet), [royalacecat](https://www.kaggle.com/datasets/royalacecat/raptor-knee-coatnet384-full-bs14) (CLI 10 Okt)

### Inferences
- dread arms: **jangan dipakai** (lemah + korpus lama mengandung gold-58 → gold-eval bocor ke atas + keluarga resep sama dgn nartaa → korelasi tinggi). Nilai historisnya justru berguna sbg bukti "arm sekeluarga = placebo" (Q4).
- Arm tambahan YANG LAYAK dievaluasi gratis (bila ada slot): `tbongc` v10geo/v8geo dan `royalacecat` — retrain CoAtNet-384 independen (seed/corpus tak diketahui; kemungkinan termasuk gold → gold-eval hanya arah kasar). Prioritas menengah.
- `pilkwang/rsna-knee-weights` (20× DINOv2-small + OOF resmi): live-infer ke gold-58 pernah menghasilkan 0,9954 = **kontaminasi nyata** (gold ada di train mereka) → evaluasi gratis mustahil; arm lemah (LB ~0,87–0,89) — [docs/RESEARCH_NOTES.md](../../docs/RESEARCH_NOTES.md) §8d, baris 447 (internal)

### Gaps
- Tak diketahui apakah tbongc/royalacecat melatih dgn mengecualikan gold-58 → klaim "arm baru sah" hanya bisa datang dari LB, bukan gold.

## Q3 — Blending: prob-mean vs rank-mean vs per-finding weights — mana yang pernah ditunjukkan menambah?

### Takeaway
Stack kita SUDAH memakai rank-blend + **bobot per-finding** (TARGET_WEIGHTS 0,03–0,35 dari perbandingan gold goodpjw vs nartaa). A/B prob-mean vs rank-mean pada stack ini **belum pernah diuji** (semua rekam jejak internal = rank). Grid bobot halus 0,05–0,1 = **placebo** (+0,0002 orde; tak pernah terbukti menambah di gold-58 di luar noise). Rekam jejak gold-58 internal (Colab 9 Okt) = kalibrasi penting: Δ gold ≈ +0,001–0,002 TIDAK pindah ke LB.

### Cited Findings
- TARGET_WEIGHTS Apex (sudah aktif di stack 0,950): Baker's 0,35 · Contusion 0,30 · Medial OA 0,25 · ACL 0,25 · Medial Meniscus 0,20 · MCL 0,20 · Lateral Meniscus/Fracture/Lateral OA/PF OA 0,08 · Effusion 0,04 · Synovitis 0,03; blend `fused = (1−w)·rank_sota + w·rank_own` lalu rankpct akhir — [kaggle_base/apex_grandmaster_stack_20261008.ipynb](../../kaggle_base/apex_grandmaster_stack_20261008.ipynb) sel §Fusion (internal)
- Optimisasi bobot = placebo: hill-climb 50×2000 iter vs ridge +0,0002 — [writeup PS-S5E12](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/2nd-place-solution-winning-based-on-id-shift-an); sintesis internal: "optimisasi bobot halus +0,0002; ambang Δ<0,002 = noise" — [teknik_ensembel_bobot.md](../Strategi%20meraih%20silver%20RSNA/teknik_ensembel_bobot.md) Q3–Q4
- Rank-mean = standar semua notebook top ("averaging percentile ranks, never raw probabilities") — [research-report.md (ahmed-hesham07)](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md); literatur: weighted Borda (SUMMA) ≥ rank seragam — [JMLR v20](http://jmlr.org/papers/volume20/18-094/18-094.pdf)
- Per-finding include/exclude: konvergen 2 notebook independen (RadImageNet dibuang di Baker's/Fracture), **tanpa angka lift** — [research-report.md](https://github.com/ahmed-hesham07/RSNA-Knee-Abnormality-Detection/blob/main/research-report.md)
- Kalibrasi emas internal (Colab 9 Okt, SE≈0,01): induk r384 plain 0,9214 → +mirror 0,9221 → crop320 0,9219 → +mirror 0,9225; r224 solo 0,9150; rank-blend r384(320+mirror)+r224 @0,2/0,35/0,5 = **0,9239/0,9238/0,9223** — [MEMORY.md](../../MEMORY.md) §3 (internal). Blend emas 0,9225→0,9239 (Δ +0,0014) TETAP berujung LB 0,950 (sub 57012143) → **bukti langsung: Δ gold <0,002 = noise di LB**.
- Fork rabari (MIRROR_TTA + r224@0,3 + reader per-finding) = 0,950 — [laporan_push.md](../../laporan_push.md) §2 (internal) → geometri blend alternatif pada keluarga sama = plafon 0,950.
- nartaa sendiri: blend multi-arm CoAtNet lama di LB = "0.914–0.915 vs 0.914 solo" (≈+0,001 = noise) — source `ARMS` notebook nartaa (idem 746792).

### Inferences
- Yang BELUM diuji & masih waras: (a) prob-mean antar ARM (bukan antar view) vs rank-mean di stack 3-leg; (b) bobot per-finding utk r224 (bukan cuma reader); (c) mirror diperlakukan sbg arm rank terpisah (bukan prob-mean dalam arm).
- Yang TIDAK layak: sweep grid halus 0,05 langkah utk 12 bobot (12 parameter × emas 58 = pasti overfit — [teknik_ensembel_bobot.md](../Strategi%20meraih%20silver%20RSNA/teknik_ensembel_bobot.md) Q3); "gating routing di-fit di emas" (rabari = contoh gagal).

### Gaps
- Tidak ada A/B publik berskor-LB "rank vs prob" di kompetisi imaging (klaimnya konsensus + teori AUC, bukan satu eksperimen).

## Q4 — TTA/variance-reduction tak teruji: daftar lengkap + klaim forum 746792

### Takeaway
Dari seluruh ruang TTA yang mungkin, hanya **mirror plane-aware (+0,001)** dan **K94 penuh (+0,001)** yang berbukti naik; yang berbukti turun: K94→24 (−0,006), flip semua bidang (−0,001), crop 304/288 (−0,001/−0,002), epoch-tunggal vs SWA (−0,001), training native-320 (−0,003). Sisa hipotesis gratis yang BELUM teruji: **mirror utk r224 & reader**, **rank-space vs prob-space mirror**, **crop 336/344**, **arm tambahan publik**. Seed-averaging & SWA manual tertutup oleh Q1.

### Cited Findings (tabel klaim forum 746792 — semua skor publik LB penuh 12 temuan)
| Perubahan (model nartaa) | LB | Δ | Status utk kita |
|---|---|---|---|
| 80 irisan/78 window → 96 iris/94 window | 0,939→0,940 | +0,001 | sudah dipakai (K94) |
| + label JEV/blend | 0,940→0,942 | +0,002 | label training (tak bisa diulang gratis) |
| + 2.399 lutut OAI | 0,942→0,945 | +0,005 | sudah di ckpt kita |
| 384 full-field | 0,945→0,948 | +0,003 | sudah |
| crop 320 native (bukan resize) | 0,948→0,948 | 0,000 | sudah (netral) |
| **mirror anatomis (prob-mean 0,5/0,5)** | 0,948→**0,949** | **+0,001** | sudah utk r384; **BELUM utk r224 & reader** |
| window 94→24 (model lama) | 0,940→0,934 | −0,006 | TERTUTUP (jangan kurangi window) |
| flip lebar SEMUA bidang (model lama) | 0,940→0,939 | −0,001 | TERTUTUP (mirror harus plane-aware) |
| crop 320→304 | 0,948→0,947 | −0,001 | crop lebih kecil = merugikan |
| crop 320→288 | 0,948→0,946 | −0,002 | idem |
| epoch 14 saja vs SWA {14,11,12} | 0,948→0,947 | −0,001 | SWA menang; epoch ckpt tak tersedia |
| training native-320 + mirror | 0,949→0,946 | −0,003 | training crop-320 = merugikan |

— semua baris: [forum 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) (via `kaggle forums topics show 746792`, 10 Okt). Catatan nartaa sendiri: "gold evaluation used **24 windows**" saat training-time vs K94 saat inference; "epoch 14 improved gold AUC while losing on the public board" → **emban-58 menyesatkan utk seleksi halus** (idem).

- Rumus inferensi resmi (akurat utk differensiasi, sel §"The final inference recipe" 746792 + source notebook): 96 iris/prastudi (slot 26/22/18/12/18), 94 window 3-iris, chunk 16, crop tengah [32:352]×[32:352] dari 384 **tanpa resize**, ImageNet norm, mirror: sagittal = balik 3-kanal iris, cor/axial = flip lebar, ditentukan dari iris tengah window, **sebelum normalisasi**, rata-rata 2 vektor prob sigmoid, keluaran = percentile rank per temuan; r224: crop 307→resize 224, K94, fp16, **satu view** (0,945).
- TTA/ensemble kecil +0,001–0,003 = satu-satunya "lift kecil terbukti" lintas kompetisi (EMA SIIM +0,001–0,002; TTA-like RSNA 2024 tertinggi di private) — [writeup SIIM juara 1](https://www.kaggle.com/competitions/siim-covid19-detection/writeups/dungnb-1st-place-solution); [siwooyong RSNA 2024](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification)
- Emas-58 internal utk mirror r384: 0,9214→0,9221 (+0,0007) — [MEMORY.md](../../MEMORY.md) §3 (internal) → konsisten dgn +0,001 LB nartaa; **r224 & reader = satu-satunya arm besar tanpa mirror** (r224 resmi "one view"; reader infer.py tanpa TTA) — manifest nartaa + `kaggle_base/apex...ipynb` sel §Fusion.

### Inferences (ruang gratis yang MASIH terbuka)
1. **r224 + mirror plane-aware** (hipotesis terkuat: efek mirror +0,001 tereplikasi di r384; r224 kini 1 view).
2. **reader (K16, 6 slot) + mirror plane-aware** (belum pernah; slot sagittal = balik urutan iris, cor/ax = flip lebar — logika CANON sama).
3. **mirror sbg arm rank terpisah** (rank-mean dari 4 rank: r384-plain, r384-mirror, …) vs prob-mean dalam arm — belum A/B.
4. **crop lebih besar (336/344 dari 384)** — simetris-netral dgn temuan 320-vs-384 (0,000); belum pernah dicoba publik.
5. **arm tambahan** tbongc/royalacecat (Q2) — dekorrelasi, tapi keluarga CoAtNet-384 (korelasi tinggi) & provenance emas tak jelas.
6. Tertutup/impossible: K<94, flip non-plane-aware, crop <320, seed-avg, SWA manual, epoch-tunggal, gating-fit-di-emas, rankpct keluaran (no-op utk AUC: monotone per kolom).

### Gaps
- Tak ada publikasi angka "mirror utk arm 224" atau "crop 336" — semua di Q4 butuh emas-58 kita sendiri.

## Q5 — Beda nyata: notebook Danial (0,949, satu arm) vs stack Apex kita (0,950)

### Takeaway
Arm r384 kita = **identik byte-level** dgn sumber nartaa (K94, crop 320, mirror prob-mean, fp16, dual-T4 threading). Seluruh beda skor 0,949→0,950 berasal dari **lapisan fusion kita**: arm reader goodpjw (K16, 6 slot, 3 fold di-average prob, tanpa mirror) dgn bobot per-finding + rankpct akhir. Konsekuensi: semua hipotesis "beda dgn Danial" tersisa = modifikasi ARM READER/R224, bukan arm r384.

### Cited Findings (tabel beda)
| Dimensi | Danial/nartaa (0,949) | Stack Apex (0,950) | Sumber |
|---|---|---|---|
| Arm | 1× CoAtNet-384 SWA {14,11,12} | r384 (sama) + reader ConvNeXt 2.5D (3 fold, prob-mean) ± r224@0,25 (varian probe) | notebook nartaa; `apex_grandmaster_stack_20261008.ipynb` |
| Window | K94 (chunk 16), 96 iris 5 slot | r384: K94 identik; **reader: K16** (default infer.py), 6 slot, MAX_SLICES 64, CROP_MM 140 | 746792; sel §Fusion/infer.py |
| Crop/res | native 384 → crop tengah 320, tanpa resize | identik utk r384; reader: 384 native (beda preprocessing sendiri) | 746792; own_src/preprocess.py |
| TTA | plain+mirror anatomis, prob-mean 0,5/0,5, per arm | r384: identik; **reader & r224: tanpa mirror** | 746792; sel §Fusion |
| Norm | ImageNet mean/std, setelah mirror, fp16 autocast | identik (sel yang sama dipakai mentah) | notebook nartaa |
| Fusion antar-arm | n/a (single arm); keluaran = rankpct per temuan | **rank-blend per-finding** (w 0,03–0,35) + rankpct akhir | sel §Fusion |
| Output | rankpct per kolom (monotone → no-op utk AUC) | idem | notebook nartaa |
| Emas-58 utk seleksi | dipakai (bukan CV independen; epoch14 contoh gold-vs-LB berlawanan) | historis kita: 0,9214 (induk) … blend 0,9239 — semua Δ<0,002 tak pindah LB | 746792; MEMORY §3 |

### Inferences
- Merombak arm r384 (crop, window, norm, urutan transform) = **mengulang eksperimen nartaa yang sudah berbayar penuh oleh mereka** (10 baris negatif di Q4) — nilai harapan rendah.
- Celah yang benar-benar belum dicakup siapa pun: mirror/TTA pada **reader** & **r224**, dan geometry blend antar-arm (prob vs rank).

### Gaps
- Tak ada catatan emas-58 stack Apex penuh (r384+reader, TARGET_WEIGHTS) di MEMORY — baseline emas stack harus diukur ulang sekali saat harvest (lihat Q6 Fase A) supaya Δ tiap hipotesis terukur vs stack incumbent, bukan vs induk solo.

## Q6 — TABEL HIPOTESIS (+0,001) berperingkat EV + protokol eval gold-58 GRATIS + batas kesimpulan

### Takeaway
Tujuh hipotesis gratis tersisa; 3 teratas (mirror r224, mirror reader, rank-space mirror) semuanya dilayani oleh **satu** harvest GPU commit-run + sweep CPU tanpa submisi. Ambang sah emas-58 (paired): **Δ ≥ +0,005 dan bootstrap-95% CI Δ > 0**; di bawah itu = noise. Empat hipotesis lain (crop 336, arm tbongc/royalacecat, prob-vs-rank antar-arm, bobot per-finding r224) = pelengkap murah dalam sweep yang sama.

### Protokol standar GRATIS (pola `alverlfarfaraway/mavit-288-gold58-eval`, MODE=local)
Template kernel eval emas-58 sudah ada & terbukti (commit-run, bukan submisi; stub 3-studi juga lolos) — [mavit-288-gold58-eval.ipynb](../../kaggle_kernel_hengck_gold_a2/mavit-288-gold58-eval.ipynb): `MODE="local"` → baca `train_golden58.csv` + `train_golden58_series.csv` (dari dataset `hengck23/hengck23-rnsa-knee` atau salinan kita) → loop 58 studi dari `train_series` kompetisi → sigmoid → cetak per-temuan + MEAN AUC; `MODE="submit"` memakai `test_series` (jangan aktif). Sumber fastread utk r384/r224 = sel inti notebook nartaa (sudah kita pinjam di Apex). **Aturan: JANGAN pernah `kaggle competitions submit` dari kernel eval; commit-run only.**

- **Fase A — Harvest prob per arm (1× GPU commit-run, ~20–30 mnt wall, T4×2, ≈0,3–0,5 jam GPU)**: fork kernel Apex/nartaa → MODE=local (58 studi gold) → hitung & SIMPAN CSV prob per arm: r384-plain, r384-mirror, r224-plain, r224-mirror (crop 307→224, K94, 1 view + mirror baru), reader-fold0/1/2-plain & -mirror (K16). Tarik CSV via `kaggle kernels output`. Estimasi kasar: r384 2 view ≈ 5–7 mnt; r224 2 view ≈ 2–3 mnt; reader 3 fold × 2 view ≈ 4–6 mnt; prep DICOM 58 studi ≈ 2–3 mnt.
- **Fase B — Sweep blend (CPU commit-run, ~5–10 mnt, TANPA kuota GPU, boleh banyak iterasi)**: muat CSV → semua kombinasi: rank-mean vs prob-mean; bobot {0,0/0,1/…/0,3} per leg; TARGET_WEIGHTS per-finding dgn/ tanpa bobot r224 per-temuan; include/exclude arm per temuan; mirror sbg arm vs mirror dalam-arm; crop-variant CSV (bila Fase A menambah varian crop 336). Tiap kandidat: macro-AUC emas-58 + **paired bootstrap 10.000×** (resample 58 studi, Δ vs baseline stack incumbent) → cetak Δ, CI95%.
- **Fase C — Gate (keputusan, bukan komputasi)**: submit LB HANYA bila (i) Δ ≥ +0,005 DAN CI95% Δ>0 di emas-58, DAN (ii) mekanisme sekelas TTA/arm-kuat (bukan bobot halus), DAN (iii) kuota submisi tersedia (≤2/hari agen). Gate LB ≥0,951 → kandidat final; =0,950 → BUANG ([laporan_push.md](../../laporan_push.md) §1).

### Tabel hipotesis peringkat EV = P(naik di emas) × dampak LB
| # | Hipotesis (gratis) | Bukti pro/kontra | P(gold sah) | Dampak bila sah | EV | Protokol khusus | Estimasi GPU |
|---|---|---|---|---|---|---|---|
| H1 | **r224 + mirror plane-aware** (kini 1 view) | mirror +0,001 tereplikasi (746792); r224 sudah di stack @0,2–0,35 dgn emas +0,0014 tapi LB noise | ~25% | +0,001 | tertinggi | Fase A baris r224-mirror; blend dgn bobot saat ini | +2–3 mnt GPU dlm harvest |
| H2 | **reader goodpjw + mirror plane-aware** | belum pernah diuji siapa pun; reader = leg satu-satunya non-nartaa; internal: leg independen kuat = satu-satunya penambah (+0,001) — [MEMORY.md](../../MEMORY.md) §2 | ~20% | +0,001 | tinggi | Fase A baris reader-mirror (terpakai utk fold0–2) | +4–6 mnt |
| H3 | **mirror sbg arm rank terpisah** (rank-mean 4 rank vs prob-mean 2 view) | rank-mean = standar publik; A/B internal tak pernah ada | ~10–15% | +0,000–0,001 | sedang | Fase B murni (pake CSV plain+mirror) | 0 GPU |
| H4 | **crop 336/344** (kurangi crop, tetap native) | 320-vs-384 = 0,000; 304/288 = minus (746792) → simetri netral-ragu | ~10% | +0,000–0,001 | rendah-sedang | Fase A varian tambahan (crop 336 utk r384) | +3–4 mnt |
| H5 | **arm tambahan tbongc v10geo / royalacecat** | retrain independen CoAtNet-384; keluarga sama (korelasi); provenance gold tak jelas; Tucker: ensemble model kuat → +0,008 (internal, MEMORY §2) | ~10–15% | +0,001 | rendah-sedang | Fase A terpisah (1 ckpt masing-masing, 1 view + mirror) | +8–12 mnt/arm |
| H6 | **prob-mean antar-arm** vs rank-mean; **bobot per-finding r224** | literatur: rank>prob konsensus (tanpa A/B LB); bobot halus = placebo +0,0002 | <10% | +0,000–0,001 | rendah | Fase B murni | 0 GPU |
| H7 | pilkwang 20-model / dread arms / MaVIT dll. | kontaminasi emas (0,9954 / korpus 45-gold) atau sudah ditolak (MaVIT 0,9202 < 0,9214) atau lemah (murid 0,943→0,942) | ~0–5% | +0,001 | placebo | jangan invest; tutup | — |
| ✗ | SWA manual, seed-avg, K<94, flip semua bidang, crop<320, epoch-tunggal, gating-fit-emas, rankpct keluaran | mustahil (Q1) atau sudah berbayar minus di 746792 / rabari | — | — | tertutup | — | — |

### Batas kesimpulan (emban-58, SE ≈ 0,01)
- **Sah utk layak submit**: Δ macro-AUC ≥ **+0,005** TERHADAP stack incumbent (bukan induk solo) **DAN** paired-bootstrap CI95% Δ melewati 0. Alasan: SE satu angka AUC ≈0,01 (internal); SE Δ paired utk 2 stack berkorelasi ≈0,003–0,006; +0,005 dengan CI>0 ≈ 2σ.
- **Ambigu (jangan submit krn ini sendirian)**: Δ +0,003–0,005 atau CI memuat 0. **Noise**: |Δ|<0,003.
- **Penalti adaptivitas**: bila Fase B mengetes >5 blend di emas yang sama, naikkan ambang utk blend-tuned jadi **+0,007** (√(k/n); [Ladder slides](https://www.cs.columbia.edu/~djhsu/coms6998-f17/student-work/slides/leaderboards.pdf)) — untuk TTA/hipotesis mekanistik (H1–H4) ambang +0,005 cukup.
- **Per-temuan (12 label)**: SE per-label ≈0,03–0,10 di 58 studi → keputusan per-label butuh |Δ|≈0,08 → **hanya pakai include/exclude kasar, jangan optim bobot per-label presisi**.
- Emas-58 TIDAK independen (nartaa & kemungkinan goodpjw memakainya utk seleksi; contoh terkenal: epoch14 gold naik LB turun — 746792) → emas = **screening arah**; keputusan submit tetap butuh sinyal kedua (mekanisme + preseden TTA +0,001–0,003 lintas kompetisi). Δ gold +0,0014 (blend r224) pernah kita alami → LB tetap 0,950: bukti langsung ambang +0,005 itu perlu.
- LB publik sendiri noisy (SE ≈0,006–0,01 pada ~400 studi publik; Δ+0,001 = noise per pelajaran `teknik_ensembel_bobot.md` Q4) — karenanya gerbang akhir = "≥0,951 = kandidat; konfirmasi arah di emas/OOF; jangan iterasi 2 titik bobot di LB" ([laporan_push.md](../../laporan_push.md) §1).

### Gaps
- Emas-58 stack incumbent penuh (r384+reader+TARGET_WEIGHTS) belum tercatat sebagai baseline numerik → wajib diukur di Fase A sebelum menafsirkan Δ.
- Belum dicek: apakah goodpjw mengecualikan gold-58 dari train fold0–2 (dataset punya `oof.py` tapi tak diverifikasi) — bila termasuk, Δ relatif stack tetap sah (baseline & kandidat sama-sama "melihat" emas), tapi angka absolut emas jangan dikutip.
- Provenance tbongc/royalacecat (korpus, fold, inklusi emas) tak terdokumentasi → H5 hanya screening; LB-lah yang memutus.
- `kaggle datasets files` untuk `nartaa/rsna-knee-oai-original384-s42-pair` tak bisa dicek (privat) — asumsi "tak ada ckpt tambahan di privat" tak bisa dipastikan.

## Catatan status terkini — 19 Okt 2026 02:56 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_SILVER.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; cutoff ini sementara karena medali memakai split privat 70%). Pada cek API 02:55, `g6c`, `g12`, `g13` masih PENDING. Dua kernel murid r4 di alver sudah dua kali di-requeue dan kini RUNNING pada cek 02:56. Kuota main/alver/nxtirfan: 19,87/17,75/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
