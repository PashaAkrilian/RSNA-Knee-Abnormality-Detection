# Arm BARU realistis menuju ≥0,951: opsi publik runnable vs training sendiri (per 10 Okt 2026, ~02:30 WIB)

> **ARSIP/STATUS 10 Okt 21:30 WIB (sesi Claude Code)**: dokumen riset 9–10 Okt (OpenCode). Rencana Fase A/B (mirror/pseudo-label) DIGANTI oleh `STRATEGI_PERAK.md` (merger + Apex+OrthoFoundation + finals + kepatuhan). Fakta terbaru: `MEMORY.md`; riset terbaru: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`.


> Riset 9–10 Okt 2026 (READ-ONLY di Kaggle; tidak push/submit/ubah kernel). Posisi: stack Apex **0,950** (r384 SWA nartaa ⊕ reader goodpjw ⊕ r224, rank-blend, mirror plane-aware, K94). Perak ≈rank 279; **0,951-persis = lotre (113 tim di 0,951 vs ~17 slot), 0,952 = aman** — [forensik_0951.md](forensik_0951.md) Q1/Q5.
> Catatan ini MENAMBAHI dua saudara: `forensik_0951.md` (papan/threads/kernel publik = siapa & mengapa) dan `inference_gratis_0001.md` (hipotesis TTA/blend gratis H1–H7 + protokol emas-58). Fokus di sini: **(A) status forensik final tiap kandidat arm publik 8–10 Okt + opsi rekonstruksi heliosli, (B) biaya GPU NYATA training sendiri (bukti timing internal), (C) peringkat EV/biaya + urutan eksekusi 10–15 Okt dgn ~90h GPU**.
> Sumber: `kaggle kernels pull/list/datasets files/metadata` (akun utama & akun2, 9–10 Okt), artefak di `/tmp/opencode/arm_scan/` (S6 notebook+output run 8-Okt, d5e/d5f/nursrijan/josef notebooks, listing OF fold0 5 halaman), `bash kg_quota.sh`, forum via `kaggle forums topics show`, PROGRESS.md/MEMORY.md (timing internal).

## Q1 — Heliosli S6: forensik MATINYA di level kode + inventaris aset yang tersisa

### Takeaway
Kernel `heliosli/rsna-knee-abnormality-detection` (S6, ⭐⭐, 33 vote; Stage-1+3 = blend raptor 0,7 ⊕ arm `E_lrprior` 0,3; terakhir run 8 Okt 18:51 UTC = sebelum compact-v1 di-retire 9 Okt 13:50) kini **mustahil dijalankan siapa pun**: Stage-2 memanggil `find_mount(C_SLUG, 'manifest.json')` dgn `C_SLUG='rsna-knee-compact-v1'` (kini hanya README 183 B) dan `find_mount(W_SLUG, 'weight_parts.json')` dgn `W_SLUG='rsna-knee-ofl-ssl-weights'` (403) **tanpa try/except** → RuntimeError di baris pertama Stage-2. Dataset `rsna-orthofoundation-fold0-epoch7-20261004` (MASIH publik, 837 file/5 halaman) **tidak punya `manifest.json` sama sekali** (grep 0 di semua halaman) → tak ada fallback alami. Bukti independen: receipt run kernel `nursrijan/rsna-knee-w-dinosaur` (fork publik dgn guard bundle) jatuh ke path `raptor049` (Raptor-only ≈0,949) karena bundle compact tak terdeteksi.

### Cited Findings
- S6 notebook (pull 10 Okt): `find_mount(CODE_SLUG='rsna-orthofoundation-fold0-epoch7-20261004', 'receipt.json')` · `find_mount(C_SLUG='rsna-knee-compact-v1', 'manifest.json')` · `find_mount(W_SLUG='rsna-knee-ofl-ssl-weights', 'weight_parts.json')` — pola `find_mount` = `INPUT.glob('datasets/*/' + slug)` lalu assert file ada; **tangkap-exception: nol di sekitar ketiganya**. — [kernel S6](https://www.kaggle.com/code/heliosli/rsna-knee-abnormality-detection), `/tmp/opencode/arm_scan/s6/rsna-knee-abnormality-detection.ipynb`
- `heliosli/rsna-knee-compact-v1`: isinya kini **hanya `README.md` 183 B**, creationDate 2026-10-09 13:50:47 UTC (RETIRED, "files not accessible"). `heliosli/rsna-knee-ofl-ssl-weights` = **403**. `blend-gold` (127 vote, klaim 0,951×2) = **Private**. — cek CLI 9–10 Okt
- OF fold0 (listing 5 halaman, 10 Okt): `receipt.json` (fold 0, epoch 7, "EMA best checkpoint", `validation_macro_auc_weak_reference` **0.93175**, best.pt 1.223.580.827 B sha `4f424cc7`) · `weight_parts.json` + `weights.part000–009` (total = best.pt; best.pt TIDAK ada sebagai file terpisah) · `anatomy.pt` 112.496.037 B sha `5904dd16…` · `dependencies/kneexnet/{localizer.pt,best_model_256…pth}` · `cloud_code/**` (~800 file kode training+inferensi). **Tidak ada**: `manifest.json`, `E_lrprior_s0_f*.pt`, ckpt 4-fold, side-model. — [dataset OF fold0](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004), `/tmp/opencode/arm_scan/of_p1-5.txt`
- Submission receipt run S6 8 Okt (stub 3 studi, `/tmp/opencode/arm_scan/s6out/`): `bundle_id` **`209a32c8b50a6b1c`**; models `E_lrprior_s0_f0..f3.pt` (convnext_small, 4 spatial+4 extent, fp16, 2×T4, v4 views, phys_mm 3.5); `anchor_weights_sha256` = `5904dd16…` = **sha `anatomy.pt` di OF fold0** (satu-satunya komponen bundle yang masih ada); `side_check.model_sha256` = `ab623e2b619d…` **tidak ada di listing OF fold0** (hilang bersama compact-v1). Blend receipt: `raptor 0.7 / ours 0.3`, 180 dtk.
- nursrijan w-dinosaur (pull 10 Okt): `EXPECTED_COMPACT_ID='c7ebab1eb953f1a6'` (bundle S5 era) · `REJECTED_NEW_BUNDLE='209a32c8b50a6b1c'` (bundle S6) · path aktif terakhir = **`raptor049`**, receipt: `path=raptor049`, `convnext_completed=null` → fallback Raptor-only ≈0,949; timnya sendiri 0,949. — [kernel](https://www.kaggle.com/code/nursrijan/rsna-knee-w-dinosaur)
- Keluarga fork ikut mati (sudah tercatat forensik Q3): `sameerk2004/sil-d5d-compact0951` (README eksplisit butuh compact), `ayodejiibrahimlateef/h01–h05` (salinan private heliosli), `evgendvorkin/rsna-versia-7` (344 vote; terakhir run 9 Okt 13:30 = 20 mnt sebelum retirement), `samanyu1808/blend3`, `hitarthjain0` blend-gold fork.

### Inferences
- **Dua bundle berbeda**: S5 = `c7ebab1eb953f1a6` (di-reject nursrijan sbg "bundle baru"? — nursrijan mengharapkan c7ebab; artinya compact-v1 pernah di-update heliosli ke `209a32c8` antara S5 dan S6, lalu di-retire total). Keduanya = artefak privat; tak bisa direkonstruksi dari publik.
- Rantai dependensi arm 0,951 heliosli = `compact-v1` (manifest + 4-fold `E_lrprior_s0_f*` + side-model) + `ofl-ssl-weights` (weight_parts) + `OF fold0` (anatomy+code, **satu-satunya yang masih ada**). Dua dari tiga hilang → **status: MATI permanen** (asumsi: tekanan thread 743416, konsisten dgn forensik Q2).
- Bagi scanner harian: **berhenti mengkategorikan heliosli/fork-nya sbg "kandidat arm 0,951"** — masuk daftar "tutup" permanen.

### Gaps
- Tak bisa memastikan isi `manifest.json` compact-v1 (file hilang; dari receipt S6 hanya ketahuan bundle_id + daftar model + sha kode).
- Side-model `ab623e2b…` (side_check) tak diketahui fungsinya selain "cek sisi lutut" — kemungkinan kecil menentukan skor, tapi tak terverifikasi.

## Q2 — Opsi rekonstruksi "setengah-runnable" dari OF fold0: kenapa JANGAN

### Takeaway
Satu-satunya potongan arm heliosli yang masih publik = **satu fold training** (fold0 best.pt, ckpt EMA dgn optimizer state — 1,22 GB utk convnext_small ~50M param fp32 ⇒ mengandung optimizer, model bisa diekstrak) + kode inferensi lengkap (`cloud_code/**`) + `anatomy.pt`. Ini BUKAN arm 0,951 (yang = ensemble 4-fold `E_lrprior_s0_f0..f3` + anchor bank + side-check); ini **satu fold, corak training-ckpt (bukan bundle inference)**, dan butuh penulisan ulang harness + fork-modifikasi kernel S6 utk mengganti `find_mount` dgn mount OF fold0. **EV rendah, biaya engineering tinggi (berjam–berhari), keluarga ConvNeXt compact korpus heliosli kemungkinan korelasi tinggi dgn r384+reader kita** → masuk daftar "tutup", bukan prioritas.

### Cited Findings
- best.pt fold0: val_macro_auc_weak_reference 0.93175 (label LEMAH, bukan emas; angka 0,93x sekelas murid CoAtNet kita 0,934) — receipt.json OF fold0 (10 Okt).
- Arm `E_lrprior` di receipt S6 = "compact fine-tuned image model on the pixel cache", `spatial_models: 4, extent_models: 4` — butuh 4 ckpt fold; publik hanya fold0 → rekonstruksi maksimal 25% dari ensemble.
- Provenance emas: tak ada keterangan fold0 mengecualikan gold-58 (kode `audit_patient_identity.py` dsb ada, tapi korpus training heliosli tak dipublikasikan) → emas-eval rekonstruksi hanya arah kasar.
- Biaya engineering (inferensi): pelajaran S6 = arsitektur `compact_infer.py`+`px_cache.py`+`knee_anchor_bank_v2.py` (sha tercatat di receipt) — menulis ulang = membaca ~ribuan baris kode asing + menyesuaikan; risiko tinggi gagal dgn hasil ~0.93x (setara murid sendiri).

### Inferences
- Nilai historis OF fold0: **bukti arsitektur compact ConvNeXt-S dgn anatomi-anchor = SUDAH terwakili** di stack kita (reader goodpjw = ConvNeXt 2.5D; r384 = CoAtNet dgn geometri serupa). Tak ada famili representasi baru di sini.
- Kalau pun ingin "arm dekorrelasi murah": jalur H5 (`tbongc` v10geo / `royalacecat`, [inference_gratis_0001.md](inference_gratis_0001.md) Q2/Q6) lebih murah (ckpt langsung pakai) daripada rekonstruksi fold0 ini.

### Gaps
- Belum diverifikasi apakah `best.pt` fold0 bisa diload sbg state_dict murni (vs bundel training) tanpa kode `compact_train.py` asli — salah satu alasan tambahan utk tidak invest.

## Q3 — Kandidat publik LAIN 8–10 Okt: status akhir (bukan heliosli)

### Takeaway
Tidak ada kandidat baru ≥0,951 yang runnable (forensik Q3 sudah final utk papan/klaim). Yang BELUM tercatat lengkap di dua saudara: **d5e-goldw = bobot per-finding DI-FIT langsung di emas-58 → overfit emas, eval tak sah, LB kemungkinan placebo**; **d5f = fine-tune reader on-the-fly dgn label-LLM saat submisi → resep placebo (karttik 11/12 finding tak beda; forum 745861 Claude solo 0.942) + output run KOSONG (tak terbukti jalan)**; tensorhydra = bobot DINOv2 publik TANPA kernel inferensi; zhuowamg = leg RadImageNet (sudah terukur menurunkan); prvsiyan = private; josef = notebook eksperimen encoder (bukan arm).

### Cited Findings
- `sameerk2004/rsna-knee-sil-d5e-goldw` (pull 10 Okt; inputs: reader goodpjw + nartaa): MD kernel — "Final fusion OWN_W from our **gold58 AUCs**: LatOA 0.40, Contusion 0.32, Baker 0.30, ACL 0.20" → **bobot per-finding di-fit pada 58 studi emas** (12 parameter × N=58 = pasti overfit; [teknik_ensembel_bobot.md](../Strategi%20meraih%20perak%20RSNA/teknik_ensembel_bobot.md) Q3–Q4: bobot halus = placebo +0,0002; emas = screening, jangan di-fit). Judul membandingkan "Routing 1.0 (0.950) vs 2.0 (0.952+)" — **tanpa bukti LB**; tim sameerk 0,950. → **TIDAK layak ditiru; gold-eval varian begini TIDAK sah**.
- `sameerk2004/rsna-knee-sil-d5f-ft7030` (pull 10 Okt; GPU T4; inputs: JEV + steven + lixin + pilkwang + reader + nartaa): Stage-2 = "fine-tune goodpjw ConvNeXt-Tiny folds on **improved soft labels**… Fallback: if FT fails → emit pure Raptor (never below 0.949)". Status COMPLETE tapi `kaggle kernels files` = **kosong** & log = baris `nbconvert` html conversion saja (tanpa jejak training/inferensi) → **tidak ada artefak; kemungkinan fine-tune gagal/terlewati dan fallback Raptor murni**. Tempatnya di riset LABEL (placebo per karttik + 745861), bukan arm.
- `tensorhydra/rsna-knee-meta-folds-v4` (7 Okt; dataset): 5× fold `.pt` 89.527.563 B + `cv_summary.json` (DINOv2-S 336px 5-fold, **CV 0.7989 pada label lemah konsensus** — angka lemah, bukan emas/LB); `tensorhydra` **tidak punya kernel RSNA** (kernels list = hanya LLM/MRI lama) → kalau mau dipakai, inferensi harus ditulis sendiri. Family DINOv2 = famili berbeda dari CoAtNet/ConvNeXt kita → satu-satunya arm-family "baru" yg tersisa dari publik; tapi CV 0.7989 (weak) menempatkannya di kelas murid lemah.
- `zhuowamg/rsna-knee-e13-train` (8 Okt): `rsna_rad_e11/v52_e11_heads.pt` 63.534.726 B + `v52_e11_oof.csv` + submission csv — leg RadImageNet E11/E13; rekam jejak internal: **leg RadImageNet MENURUNKAN stack (0,943→0,942)** — [MEMORY.md](../../MEMORY.md) §2. zhuowamg LB 0,952 tapi keluarga evgendvorkin/mattiaangeli (sudah kita ukur plafonnya via xianhan fast-parent).
- `prvsiyan/*` (di-scan 10 Okt sbg kandidat): **403 private**. `josefrancacer` (kernel RUNNING 10 Okt): `notebook89cfc47562.ipynb` = notebook eksperimen encoder 2.5D/ConvNeXt berbahasa Spanyol ("P3 · Experimentos de encoder") — bukan arm/submisi; pantau saja.
- Scan kernels `--competition --sort-by dateCreated` (9–10 Okt, forensik Q3 sudah katalog): bikrambarik CNN baseline · spark328 dinosaur-v5-top 0,949 · omn1v3r53 "Omniverse SOTA" 0,940 · guruprasaath dual-paragon 0,945 · rainycloudss v6 (fork Apex kita) · rian55 floor-apex 0,952 (fork keluarga sama) — **semua ≤0,950 utk timnya / keluarga sama**.

### Inferences
- Pola baru yang muncul 9–10 Okt: **"gold-tuned routing" (d5e)** dan **"fine-tune-dalam-submisi" (d5f)** — keduanya cara meniru bentuk arm heliosli (0,7/0,3) tanpa bobotnya; keduanya justru menyalahi protokol emas (fit di emas) atau resep placebo (label-LLM). Jangan terpancing meniru.
- Satu-satunya bahan publik 8–10 Okt yang punya nilai non-placebo: **bobot DINOv2 tensorhydra** (famili berbeda; murah utk dievaluasi gratis TAPI butuh kode inferensi sendiri → tunda; EV rendah).

### Gaps
- cv_summary.json tensorhydra tidak diunduh isinya (hanya metadata dataset); angka CV 0.7989 dari deskripsi dataset.
- Tak ada kandidat arm publik baru sama sekali yang menyebut pseudo-label sebagai sumber skor (konsisten: pemisah 0,95x = pseudo-label, [forensik_0951.md](forensik_0951.md) Q2 — orang tak mempublikasikan pipeline labelnya).

## Q4 — Grup B: training sendiri — biaya GPU NYATA (bukti internal) vs opsi

### Takeaway
Semua opsi "arm baru buatan sendiri" berdiri di atas satu fakta: **pseudo-label/self-training = satu-satunya jalur dgn bukti multi-sumber ke ≥0,951** (colum2131 0,953 single tanpa data eksternal; Raymond 0,949→0,958 via label; hengck "model is not the key. it is your labels"; Deotte: dgn pseudo-label "every single model can match our best ensemble"). Biaya total estimasi **≈12–35 jam GPU** (guru-infer + fine-tune beberapa ep) — **MUAT** dlm ~90h GPU (3 akun, reset 10 Okt 07:00 WIB) + backup TPU/Colab. Anak-OAI singkat & backbone-dari-nol-tanpa-pseudo = sudah berbayar/ditolak.

### Cited Findings — tabel bukti kecepatan (SEMUA angka internal repo, bukan tebakan)
| Pekerjaan | Hardware | Durasi terukur | Sumber |
|---|---|---|---|
| Training DINOv2-base, data penuh 4.349 | TPU v5e | **~6,2 mnt/epoch** (370 dtk) | PROGRESS §22 (v14) |
| idem | GPU T4 (1 proses) | **~74 mnt/epoch** | idem (v11) → TPU ≈12× |
| kfold murid 5-fold × 8 epoch (DINOv2-base) | TPU | **~4,2–5,2 jam/run** | PROGRESS (v18/v20/v21) |
| CoAtNet-384 fwd/bwd | TPU | **≈0,5 dtk/langkah (12 window)** | MEMORY §4 |
| Anak-OAI (2 run × 3 epoch CoAtNet, KD) | TPU | ~6 jam (estimasi; MATI XLA `batch_norm_backward`) | MEMORY §3, build_train_gpu_oai_child.py |
| Submisi penuh multi-arm (~1.322 studi) | 2×T4 | ~9,7 jam GPU-penuh | laporan_push §3 |
| Inferensi 1 arm (~1.322 studi) | 2×T4 | **~1,24 GPU-jam** (≈3,4 dtk/studi/arm) | idem |

- **Ekstrapolasi resep nartaa 16 ep penuh** (4.336 optimizer steps, batch1×accum16, 12 window/study, CoAtNet-384): 4.336 × 0,5 dtk ≈ 36 mnt murni compute TPU → realistis **×3–6 dgn eval 24-window + decode ≈ 2–4 jam TPU/fold**; di T4×2 (2 proses) ≈ **4–8 jam GPU per run**. → INFERENSI (nartaa tak pernah publikasi jam; angka turunan dari tabel di atas).
- **Inferensi guru utk pseudo-label di 4.407 train**: 4.407 × 3,4 dtk ≈ **4,2 jam GPU** (1 arm K94) — dgn K24 (~4× lebih cepat) ≈ **1 jam**. (Anchor: 1,24 GPU-jam per 1.322 studi.)
- **Fine-tune murid/anak 3–4 epoch** dari init r384/reader: setara separuh run penuh ≈ **2–4 jam GPU/run** (anak Colab 9 Okt = 1 ep/run, dlm sesi Colab harian).

### Opsi Grup B (diurut EV)
1. **B1 — Pseudo-label self-training** (EV TERTINGGI; bukti forum 735304, forensik Q5#1). Sketsa biaya: (a) prediksi guru = stack incumbent (r384+reader) di 4.407 train — 1–4,2 jam GPU; (b) fusi pseudo dgn label laporan dgn ambang confidence + mask kelas langka (resep spesifik TIDAK dipublikasikan siapa pun → rancang sendiri, tandai sbg inferensi); (c) fine-tune 2–4 murid (ConvNeXt-T 2.5D ala colum2131 ATAU CoAtNet alumni) 3–4 ep = 6–16 jam; (d) gate: OOF vs label-report (protokol 2-tingkat: OOF pilih epoch, emas arah) + emas-58 smoke. **Total ≈10–25 jam GPU.** Risiko: entry 15 Okt (5,5 hari lagi); noise LB; gold-58 terkontaminasi utk stack publik (Δ relatif stack incumbent tetap sah).
2. **B2 — Retrain resep nartaa 16ep dari nol (tanpa pseudo)**: plafon keluarga tanpa OAI ≈0,942 (tangga 746792) → **TIDAK LAYAK sbg arm baru**; dengan OAI = melipatgandakan risiko §5.1 dgn hasil maksimal setara 0,949 = TUTUP. (Kecuali sbg variasi inisiasi utk B1 — tidak perlu: init r384 sudah ada.)
3. **B3 — Backbone-family baru dari nol tanpa pseudo** (ConvNeXt-T/DINOv2-S/EffNetV2 2.5D): angka forum ≥0,951 SEMUA pakai pseudo-label juga; tanpa pseudo keluarga kita sudah jenuh 0,934 (4 murid fold) → **TUTUP** kecuali DALAM kerangka B1 (murid = famili berbeda dgn guru = dekorrelasi, sah).
4. **B4 — Anak-OAI fine-tune singkat**: **DITUTUP 9 Okt** (anak c ep1 emas 0,9207 < induk 0,9214; anak e 0,9138; TPU XLA mati) — jangan ulang.
5. **B5 — Rekonstruksi compact fold0 heliosli**: **TUTUP** (Q2) — biaya engineering tinggi, EV rendah, korelasi keluarga.

### Inferences
- B1 adalah SATU-satunya baris Grup B yang layak GPU; sisanya = penutup/blok yang sudah kita bayar.
- Urutan hardware utk B1 (per revisi user 10 Okt 00:45): guru-infer & fine-tune di **GPU akun2 `alverlfarfaraway` (29,9h tersedia SEKARANG)** atau **akun3 `nxtirfan` (27,7h)** — sisakan kuota akun utama (refresh 07:00 WIB) utk harvest H1/H2 (mirror reader/r224) + submisi gate. TPU (54h total) cadangan utk fine-tune andai XLA diperbaiki; Colab T4 utk 1–2 run kecil.
- Dekorrelasi dalam B1: murid DINOv2-S 2.5D (init tensorhydra?) vs murid ConvNeXt-T — **jangan keduanya**; pilih SATU (ConvNeXt-T = preseden colum2131 paling kuat; DINOv2 = satu-satunya famili beda tapi preseden publiknya lemah).

### Gaps
- Resep pseudo-label konkret (ambang, jumlah iterasi, KD) = inferensi kita; satu-satunya referensi forum = "weak-label multi-guru → OOF → pseudo → model akur" (735304) tanpa hyperparameter.
- Belum diukur: berapa jam wall-clock stack incumbent menghasilkan prediksi 4.407 train (estimasi di atas = turunan anchor submisi; bisa beda 2× karena decode DICOM).

## Q5 — Peringkat EV/biaya + protokol uji gratis + urutan eksekusi 10–15 Okt

### Takeaway
Peringkat akhir: **(1) B1 pseudo-label** (EV tertinggi, 10–25 GPU-jam) ⊕ **harvest mirror reader/r224 (H1/H2, ~0,5 GPU-jam, gratis)**; (2) eval gratis tbongc/royalacecat (H5, ~0,2 GPU-jam); (3) tensorhydra DINOv2 (tulis kode inferensi dulu, EV rendah); sisanya TERTUTUP (heliosli/fold0, d5e/d5f, MaVIT, dread/pilkwang, anak-OAI, B2/B3). Budget ~90h GPU setelah 07:00 WIB + TPU 54h + Colab: **rencana puncak ≈25–40 jam → muat dgn cadangan 2×**.

### Tabel peringkat (EV = P(naik) × dampak; protokol = [inference_gratis_0001.md](inference_gratis_0001.md) Q6 kecuali ditulis lain)
| # | Opsi | Status | EV | Biaya GPU | Protokol uji gratis | Keputusan |
|---|---|---|---|---|---|---|
| 1 | **H1/H2 mirror r224 & reader** (TTA pada arm yang belum pernah) | belum teruji siapa pun; mirror +0,001 tereplikasi di r384 | tertinggi (dampak kecil, P sedang) | ~0,3–0,5 jam (1 harvest commit-run emas) | Fase A/B/C emas-58, gate Δ≥+0,005 CI>0 | JALANKAN 10 Okt pagi |
| 2 | **B1 pseudo-label self-training** | bukti forum kuat; resep = rancangan sendiri | tertinggi (dampak besar) | 10–25 jam | OOF-vs-label-report utk seleksi + emas smoke; gate LB ≥0,952 | JALANKAN mulai 10 Okt (akun2/3) |
| 3 | H5 arm tbongc/royalacecat | keluarga CoAtNet (korelasi); provenance emas tak jelas | rendah-sedang | ~0,2 jam | Fase A terpisah | evaluasi gratis BILA slot kosong |
| 4 | tensorhydra DINOv2-S 5-fold | bobot publik, TANPA kernel; CV lemah 0,7989 | rendah | 0,3 jam infer emas + engineering | tulis inferensi sederhana → emas-58 | tunda; hanya dlm B1 sbg murid alternatif |
| 5 | Rekonstruksi compact fold0 (Q2) | setengah-runnable, 1 fold training-ckpt | rendah | 2+ jam engineering + infer | — | TUTUP |
| 6 | d5e gold-tuned OWN_W / d5f FT-on-the-fly | overfit emas / placebo label-LLM + output kosong | ~placebo | — | — | TUTUP (jangan ditiru) |
| 7 | heliosli S6/blend-gold & seluruh fork | MATI permanen (Q1) | 0 | — | — | TUTUP |
| 8 | MaVIT, anak-OAI, dread/pilkwang, B2, B3 | sudah ditolak/terukur | 0 | — | — | TUTUP |

- **Protokol emas-58 gratis (ringkas; detail di inference_gratis Q6)**: commit-run `MODE=local` (template `kaggle_kernel_hengck_gold_a2`, stub 3-studi lolos; JANGAN `kaggle competitions submit` dari kernel eval) → Δ ≥ **+0,005** vs stack incumbent **DAN** bootstrap CI95%>0 (adaptivitas: naik ke +0,007 bila >5 blend diem) → baru 1 submisi gate. LB: **≥0,951 = kandidat; 0,952 = aman; =0,950 = BUANG; Δ<0,002 = noise**.
- **Batas submisi**: 5/hari tim, reset 07:00 WIB; agen pakai maks 2/hari; sisakan 2–3 utk user & nxtirfan (aturan MEMORY §0).

### Urutan eksekusi 10–15 Okt (draft; budget ~90h GPU + 54h TPU + Colab)
1. **10 Okt 07:00 WIB**: `bash kg_quota.sh` (refresh ketiga akun) → pastikan tak ada job proyek lain mencuri kuota (keputusan user #2).
2. **10 Okt pagi (GPU utama, ~0,5 jam)**: Fase A harvest emas (r384±mirror, r224±mirror, reader fold0–2±mirror) → Fase B sweep CPU → keputusan H1/H2 (kemungkinan terbesar: +0,000–0,001; hanya submit kalau lolos gate).
3. **10–12 Okt (GPU akun2/akun3)**: B1 — (a) guru inferensi 4.407 (K94 atau K24), (b) bangun pseudo-label fusi, (c) fine-tune murid ConvNeXt-T 2.5D 2–3 fold × 3–4 ep; paralel: `scan_public_updates.py` harian (fast-follower; jangan lupa jeda 429).
4. **12–13 Okt**: gate OOF + emas → maks 1 submisi LB gate dgn stack+arm-baru.
5. **13–14 Okt**: iters lanjut (pseudo putaran-2 / murid DINOv2) HANYA bila gate-1 positif; backup: stack incumbent tetap 0,950 sbg arm bersih.
6. **14–15 Okt (sebelum entry 15 Okt 23:59 UTC = 16 Okt 06:59 WIB)**: freeze; koordinasi nxtirfan (keputusan user #2); sisakan submisi penutup utk user.
7. **21–22 Okt**: keputusan slot final (bersih vs agresif-OAI) + cek ulang 743416 — di luar scope catatan ini.

### Gaps
- Izin akun2 sudah diarsip (`docs/IZIN_AKUN_KEDUA.md`, 10 Okt) — verifikasi teks email asli tetap di inbox user sbg bukti utama.
- Angka jam B1 (guru-infer 4.407, fine-tune) = estimasi turunan; tolok ukur pertama = ukur wall-clock run pertama, lalu revisi jadwal.
- Belum ada baseline emas stack incumbent penuh (r384+reader+TARGET_WEIGHTS) tercatat — **wajib** diukur di Fase A (sudah ditandai di inference_gratis Q6 gaps) supaya Δ H1/H2 & B1 terukur benar.

---
*Metode: `kaggle kernels pull` (heliosli S6, nursrijan, d5e, d5f, josef → `/tmp/opencode/arm_scan/`), `kaggle kernels output` (S6 run 8 Okt → receipt), `kaggle datasets files` (OF fold0 5 halaman, compact-v1, ofl-ssl-weights, tensorhydra, zhuowamg — 9–10 Okt), `kaggle kernels list --competition --sort-by dateCreated`, `bash kg_quota.sh`, forum 735304/746792/745861/743416 via `kaggle forums topics show`, PROGRESS.md §22/kfold, MEMORY.md. Semua READ-ONLY; tidak ada push/submit.*
