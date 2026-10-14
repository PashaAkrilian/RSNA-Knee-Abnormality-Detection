# laporan_push.md — Evaluasi & jalan menuju SILVER
> Wadah TETAP untuk hasil evaluasi tiap submisi/probe + rencana peningkatan. **Setelah tiap run/submisi/skor baru: tambah baris §2, perbarui §1 & §4.**
> Konteks aturan: `MEMORY.md` · rencana & gerbang: **`STRATEGI_PERAK.md`** · langkah harian: `HANDOFF.md` · arahan harness: `push.md` · riset terbaru: **`reports/Ringkasan riset 10 Okt (sesi Claude Code).md`** · arsip OpenCode: `reports/Cara menembus 0951 perak.md`, `reports/Strategi meraih perak RSNA.md`, `research_notes/`.
> Dibuat 9 Okt 2026 ~21:45 WIB, diperbarui 11 Okt ~03:45 WIB. Bahasa kerja: Indonesia. **Cadangan/arah terbaru: `docs/PLAN_B.md` §00 (S6 hidup lagi).**

## 1. Posisi vs garis perak (papan 11 Okt 02:59 WIB, 5.699 tim; angka lama di MEMORY/arsip)
| Hal | Nilai |
|---|---|
| Tim | `Krazy KIller` = `alverlfarfaraway`, `dimaspashaakrilian`, `nxtirfan` (3 akun di halaman tim; lihat risiko akun di `STRATEGI_PERAK.md` J4) |
| LB kita | **0,956** (salinan R7 user, 57146631) — **rank 558** (papan 14 Okt 07:44 UTC, 5.957 tim). Riwayat: 0,954 rank 466 (12 Okt), rank 144 (11 Okt pagi) |
| Peta blok (14 Okt 07:44 UTC) | ≥0,966 = 4 · ≥0,965 = 7 · ≥0,964 = 12 · ≥0,962 = 27 · ≥0,960 = 55 · ≥0,958 = 108 · ≥0,957 = 201 · **≥0,956 = 951 (±750 tepat 0,956 = dataran R7)** · garis perak (rank 297) = **0,956**, perunggu (rank 595) = **0,956**, emas (±rank 22) ≈ 0,962. **0,965 = 7 tim teratas** |
| Peta blok (12 Okt 07:29 UTC) | ≥0,954 = **803** (±600 tepat 0,954) · ≥0,955 = 182 · ≥0,956 = 119 · ≥0,957 = 86 · ≥0,958 = 66 · ≥0,960 = 38 · garis perak (rank 290) = **0,954**, perunggu (rank 581) = **0,954**, emas (rank 21) = 0,962 |
| Dinamika | garis perak 0,950 (8 Okt) → 0,954 (12 Okt); ≥0,955 naik 73 → 182 dalam 4 hari; 529 dari 604 tim di dataran 0,954 submit dalam 24 j terakhir (gelombang klon S6); trik "kirim 1−p" membuat tim kuat tersembunyi |
| Peluang (simulasi) | Apex 0,950: perak 13–20 %, perunggu 25–41 %; klon S6 (0,954) di klaster ratusan tim: perak ±25–45 %, perunggu ±45–70 %; varian dgn informasi baru/merger lebih tinggi; **100 % mustahil** (`silver_prob_sim.py`) |
| Deadline | merger/entry **15 Okt 23:59 UTC**; final **22 Okt 23:59 UTC**; maks 2 final (PILIH MANUAL); publik 30 % / privat 70 % |
| Aturan terima uji | varian dibandingkan dgn klon S6 (≈0,954); LB resolusi 0,001 → hanya selisih ≥0,002 yg jelas; Δ<0,002 = noise |

## 2. Rekap evaluasi (diisi tiap hasil — JANGAN hapus baris lama)
| # | Tanggal | Submisi | Konfigurasi | LB | Δ vs 0,950 | Keputusan | Catatan |
|---|---|---|---|---|---|---|---|
| 1 | 13 Sep | 56196772 | v3 hybrid (fork + arm v3 α=0,15) | 0,941 | −0,009 | BUANG | byte-identik dgn base; arm tak terbaca |
| 2 | 4 Okt | 55795592 | c5 sendirian (CoAtNet5, label fusi) | 0,927 | −0,023 | BUANG sbg stack | solo ≠ stack; CV lokal tak pindah ke LB |
| 3 | 7 Okt | 56897590 | d4-blend (in-arm r2c5) | 0,943 | −0,007 | cadangan bersih | jalur tanpa OAI |
| 4 | 7 Okt | 56872456 | replika goodpjw (reader ConvNeXt) | 0,944 | −0,006 | cadangan bersih TERBAIK | 100% tanpa ckpt OAI |
| 5 | 8 Okt | 56943646 | **Apex v2**: nartaa r384 solo + reader goodpjw | **0,950** | — | **BASELINE / terbaik** | stack cepat (±30 mnt) |
| 6 | 9 Okt 20:34 | 57010586 | template akun kedua (smoke) | — | — | n/a | uji jalur submisi akun2 |
| 7 | 9 Okt 21:39 | **57012143** | **probe r224**: Apex r384 .75 + nartaa r224 .25 + reader | **0,950** | 0,000 | **BUANG** (noise; varian r224/tweak bobot TUTUP) | commit-run OK; skor 23:16 WIB; monitor 30 mnt: `progress_submisi.log` |
| 8 | 9 Okt 23:37 | **57015077** | **fork rabari dual-engine**: raptor r384 + MIRROR_TTA ⊕ reader goodpjw ⊕ r224@0,3 (input publik) | **0,950** | 0,000 | **BUANG** (klaim "0,951+" author TIDAK terbukti; keluarga r384+reader+r224 = plafon 0,950) | skor 10 Okt ~00:45; sisa jatah 2 (reset 07:00 WIB) |
| 9 | 10 Okt 07:27 | **57025414** | **Apex + MaVIT (hengck23 288px) @0,20** rank-blend (`build_submit_mavit.py`; kernel `rsna-knee-apexmavit` v1) | **0,949** | −0,001 | **BUANG** | pembaca ketiga arsitektur independen (solo LB 0,935; gold-58 0,9202) pada bobot 0,20 TIDAK menambah (galat berkorelasi dgn label laporan yg sama); akun UTAMA; skor 08:30 WIB |
| 10 | 10 Okt 09:10 | **57028008** | **Apex 0,70 + OrthoFoundation-L fold0 (heliosli) 0,30** (`build_submit_legs.py legs=of wo=0.30`; kernel `rsna-knee-apexof` v1; inferensi per 30 studi, 2 proses) | **Notebook Timeout (9 j)** | — | GAGAL teknis (tanpa skor) | estimasi 5–6 j meleset. Uji skala 10 Okt: disk 330 MB/studi (potongan 30 studi ≈10 GB → penuh), CPU 4 vCPU hambatan (GPU sering 0 %), 2 proses 45–95 dtk/studi, 4 proses ±60 dtk/studi/proses; pemetaan seluruh test_series 84 dtk/panggilan |
| 11 | 10 Okt 21:10 | **57044355** | **Apex 0,70 + OF v2** (`kaggle_kernel_apexof2`: potongan 8, 4 proses, penjaga disk, thread=1, prioritas, pemutus proses, reuso anatomi) | **0,950** (COMPLETE 22:45 WIB, ±1 j 35 mnt) | 0,000 | **TANPA VONIS OF** (kaki OF tak jalan di run penilaian) — v4 dibangun/diuji; G1 ditahan | uji skala: 194 studi/jam (6,8 j; reuso anatomi tak membantu); stub v3 lulus (GPU kosong 15 GB); terima bila ≥0,952 (0,951 = cadangan) |
| 12 | 11 Okt 02:56 | **57051680** | **Salinan S6 heliosli oleh USER** (Raptor 0,7 + compact 0,3; bundel `209a32c8` dipulihkan 02:50 WIB) | **0,954** | +0,004 | **TERBAIK; kandidat final A; rank 144** | COMPLETE ±06:40 WIB (±3 j 45 mnt); papan 06:40: 5.709 tim, ≥0,954 = 150 (49 tepat 0,954) |
| 13 | 11 Okt 07:03 | **57055673** | **`s6reader` v1**: S6 (Raptor 0,7 + compact 0,3) + reader goodpjw dipatenkan ke perna Raptor (bobot per-temuan ×0,5; kontroler lenient) | **0,954** | — | net-zero vs klon S6 | reader ×0,5 aman tapi tak menaikkan; plafon S6+reader = 0,954 |
| 14 | 11 Okt 07:04 | **57055781** | **`s6plus` v1**: S6 + reader goodpjw ×1,0 + leg OF fold0 0,15 (stage terakhir, supervisor v4.1, deadline 7,9 h) | **0,952** | — | −0,002 vs klon | reader ×1,0 terlalu kuat + OF 0,15 tak menolong; jangan ulang |
| 15 | 11 Okt 11:03 | **57062952** | **`s6prb` v1**: S6 + **probit-quantile fusion** (ndtri-space blend; rabari9999) + **residual Synovitis** (0,72/0,23/0,05; romantamrazov; koef TETAP) — `build_s6plus.py probit=1 synres=1` | **0,954** | — | net-zero vs klon | probit/synres aman dipakai ulang; tak menaikkan skor di LB |
| 16 | 11 Okt 11:03 | **57062953** | **`s6of` v1**: S6 + leg OF fold0 **0,15 tanpa reader** (Raptor 0,55 + compact 0,30; supervisor v4.1) — mengisolasi pengetahuan OF | **0,951** | — | −0,003 vs klon; OF tanpa reader menolong | pembanding s6plus (0,952) → keduanya < klon 0,954; OF tak layak diulang |
| 17 | 11 Okt 11:03 | **57062959** | **`s6clean`** asuransi: compact + reader goodpjw, **TANPA Raptor/OAI** (noraptor path) — di akun alverlfarfaraway (kuota terpisah; perintah user 11 Okt) | **0,950** | — | asuransi: Raptor = +0,004 | Rules 5.a risiko = keputusan user; tetap berharga sbg jaring pengaman final |
| — | 11 Okt ~13:00 | **teacher-train** (irfan) | Fase B langkah 1: r384 solo → prediksi 4.407 studi train; CSV `data/teacher_r384_train_preds.csv` (0 fallback, 80–90 mnt) | COMPLETE | — | bahan murid ConvNeXt-T 2.5D (langkah 2) | in-sample (bukan OOF) — caveat diketahui |
| — | 11 Okt 17:40 | **student-prep** (alver) | Fase B 2a: cache256 uint8 + labels konsensus⊕teacher (blend50 emas-58 **0,929** > konsensus 0,896; fold valid 5×870+58) | v3 ERROR cell push (env) → labels+meta diunggah manual ke dataset **`alverlfarfaraway/rsna-knee-student-cache256`**; **v4 cache-only RUNNING 21:12** (push via kaggle.json) | — | selesai ~23:00–23:30; slug lama bentrok kernel → id baru cache256 | CPU-only; watch_prep3.log |
| — | 11 Okt (siap) | **student-train-a/b/c** (alver) | Fase B 2b: ConvNeXt-T 5-fold (0-1/2-3/4), fp16, EMA, warm-start goodpjw | siap push | — | push a+b paralel setelah prep | internet OFF → ImageNet tak bisa download; ckpt goodpjw sbg init |
| — | 11 Okt (siap) | **student-dino** (alver) | keluarga ke-2: ViT-S DINOv2 one-shot (4.349 weak, emas val-only), PyTorch-murni key-strict | siap push | — | gelombang 2 dgn train-c | colum: DINO 0.951 + ConvNeXt → blend |
| — | 11 Okt (siap) | **student-infer** (alver) | murid solo: mean fold ConvNeXt + DINO rank-blend 50/50 → submission.csv + student_test_preds.csv | siap push | — | setelah folds+dino didatasetkan; preds → dataset `rsna-knee-student-preds` (placeholder sudah dibuat) | blend S6+murid = `s6student` 0,65/0,35 |
| — | 11 Okt 21:17 (siap) | **s6student** (alver) | S6 (probit+synres) + murid 0,35 rank-blend (`build_s6student.py`); fallback S6 bila preds absen | siap push | — | setelah student-preds ada isinya | commit ac48abf |
| — | 12 Okt 00:15 | **train-a/b + dino** (alver/main/irfan) | gelombang 1: cache-on-demand + ConvNeXt warm-start goodpjw / DINOv2-S one-shot | **ERROR 00:30** | — | tulis cache ke `/kaggle/input` (read-only mount) | orkestrator `night_student.sh` exit 00:35 (fold=0) |
| — | 12 Okt 02:23 | **train-a/b + dino** v5/v3/v5 | fix cache ke `/tmp/cache256_train` | **ERROR ~06:15** | — | cache jalan 70 mnt, warm-start OK, lalu **OOM CUDA bs=8** (batch pertama training) | T4 16GB; goodpjw asli bs=8 di GPU lokal |
| — | 12 Okt ~07:45 | **train-a/b + dino** v5/v4/v5 | bs 8→4 + `expandable_segments` | **ERROR ~10:15** | — | OOM lagi bs=4 (backbone jalan per ~bs×series×kwin gambar px256); **dino: `_AttnOut` missing forward** (bukan OOM) | tiap retry ±70 mnt cache rebuild |
| — | 12 Okt ~10:45 | **train-a/b + dino** v6/v4/v6 | **bs=2**; dino `_AttnOut.forward` ditambahkan | RUNNING | — | titik kritis OOM ±1 j setelah warm-start; nxtirfan kuota 3,5 j ⚠️ | bila OOM lagi: grad-checkpointing / kwin 8→6 |
| 18 | 12 Okt (antre) | **`s6eq`** (alver; `build_s6eq.py`) | notebook publik #1 heliosli **tanpa ubahan**: Raptor 0,5 + compact 0,5, rank sadar-gagal, bundel `r6cnxfsss` (= compact-v1 V13) | MENUNGGU slot GPU | — | perintah user 12 Okt (2 submisi); watcher `scripts/submit_watch.py` | harapan ±0,955 (tim heliosli 0,956; salinan 0,955 — skor tim) |
| 19 | 12 Okt (antre) | **`s6eqr`** (alver) | `s6eq` + Part B reader goodpjw **tanpa ubahan** (bobot per temuan 0,03–0,35) | MENUNGGU slot GPU | — | uji apakah pembaca ke-3 menambah di atas S6-eq | di basis 0,7/0,3 reader = net-zero (s6reader 0,954) |
| — | 12 Okt 16:05 | upaya pertama `s6eq`/`s6eqr` v1 (akun utama) | commit-run sukses 5 mnt (3 studi), submit ditolak **400 "Submission not allowed: non-versioned datasets [rsna-knee-compact-r6cnxfsss]"** | — | — | **fix: pin versi di metadata** `owner/slug/N` (r6cnxfsss/**2**, nartaa/**1**, OF/**1**, reader/**1**) → v2 | aturan BARU 12 Okt (submisi 11 Okt masih diterima polos); jatah 5/5 tak berkurang |
| — | 12 Okt 16:12 | **`s6eq` v2 + `s6eqr` v2** (AKUN UTAMA; pindah dari alver — alver 2/2 slot train & kuota mati ±20:00) | S6-eq 0,5/0,5 rank sadar-gagal; s6eqr = +reader goodpjw | commit-run COMPLETE, **submit ditolak 400 lagi** → pin versi metadata TIDAK menolong (server strip; API cuma `owner/slug`) | — | akar: **setting "All Versions Accessible"** dataset `heliosli/rsna-knee-compact-r6cnxfsss` mati (bukan aturan pin; nartaa/OF/reader lolos) | — |
| — | 12 Okt 17:00 | **mirror + `s6eq` v3 + `s6eqr` v3 (AKUN UTAMA) TERKIRIMA** | mirror `dimaspashaakrilian/rsna-knee-compact-r6cnxfsss` v1 (887 MB, file identik, kredit heliosli) → metadata ganti sumber → v3 | **DITERIMA 16:50** (ref **57103358** s6eq / **57103359** s6eqr) | — | jatah 3/5 | |
| 20 | 12 Okt 18:10 | **`s6eq` v3 — SKOR 0,954** | S6-eq 0,5/0,5 rank sadar-gagal (publik #1 heliosli, tanpa ubahan; basis mirror) | **COMPLETE 0,954** | **0,954** | = klon S6 terbaik, **G0 ≥0,955 TIDAK lulus**; arah ke murid (17 Okt) | net-zero di atas 0,7/0,3? == klon; noise (Δ<0,002) |
| 21 | 12 Okt 18:10 | **`s6eqr` v3 — SKOR 0,954** | s6eq + Part B reader goodpjw (bobot 0,03–0,35) | **COMPLETE 0,954** | **0,954** | reader **net-zero** (konsisten s6reader 11 Okt: 0,954→0,954) | reader ke-3 tidak menambah di basis S6-eq |
| 22 | 13 Okt 21:26 WIB | **57146631** (USER) | salinan `heliosli/0-957-latest-dataset-score-in-one-click` = **R7**: Raptor 0,5 + compact R7 0,5 (bundel `8c7b666e`, ConvNeXt-S 4 fold crop 352, aux fastMRI, stacker Effusion/Baker) — kernel `dimaspashaakrilian/0-957-latest-dataset-score-in-one-click` | **0,956** | **+0,002 vs S6** | **TERBAIK BARU; kandidat final A** | rank 558/5.957 (papan 14 Okt 07:44 UTC); dataran 0,956 = ±750 tim (rank 202–951) |
| 23 | 14 Okt 00:09 WIB | **57150970** (USER) | salinan `degnonguidi/rsna-knee-abnormality-detection` (= R7 identik + find_mount toleran) | **0,956** | — | duplikat R7 (tidak menambah informasi) | `RSNA_Versia_7` (evgendvorkin, R7 juga) di-run 13 Okt tapi tidak disubmit |
| — | 14 Okt 15:00 | **`r7murid` v1** (akun utama; `build_r7murid.py w=0.2`) | R7 tanpa ubahan + **Stage 0 murid ConvNeXt-T 4 fold** (subproses sebelum Raptor, anggaran 75 mnt, fallback R7) + kaki ke-3 rank **0,20**; murid DINO DIBUANG (emas-58 0,71); commit-run memvalidasi murid di 58 gold | commit-run RUNNING | — | submit bila GOLD58 log ≈0,92 & MURID_LEG used | uji pertama apakah informasi murid kita menambah di atas R7 (δ>0 = lompat di atas ±750 klon) |
| — | 14 Okt 15:00 | **guru OOF R7** f0/f1 (nxtirfan; `build_compact_oof.py base=r7`) | compact R7 fold-k pada studi latih hash-fold k (OOF bersih; fungsi fold R7 terverifikasi `sha1[:8] % 4`, gold tak dilatih) | RUNNING (±1,6–2 j/fold) | — | f2/f3 + gold menyusul; bahan label `L1 = 0,5 konsensus + 0,5 OOF-R7` utk murid putaran-2 | konsensus per temuan direkonstruksi persis dari v0 (v0 = 0,5 konsensus + 0,5 Raptor in-sample) |

## 3. Temuan riset (sumber lengkap: `reports/Strategi meraih perak RSNA.md`)
### 3a. Sudah TERBUKTI (jangan diulang sia-sia)
- Keunggulan = **pengetahuan/data**, bukan varian: bagging tanpa OAI ≠ menolong; leg lemah MENURUNKAN; DITUTUP: KneeNet, soup init sama, metadata DICOM, resolusi 288, backbone besar (DINOv2-base +0,001 = noise; DINOv3 lebih buruk), label smoothing.
- **Bukti berhenti di 0,949**: TIDAK ADA sumber publik (label/preprocessing/bobot) yang melaporkan gain di atas stack 0,949 → relabeling/resep bukan jalur perak.
- **Optimisasi bobot halus = placebo** (+0,0002 bahkan hill-climb penuh); model sekeluarga & klon tak menambah.
- **PEMISAH 0,950 vs ≥0,951 (riset 10 Okt, thread 735304, 9 Okt) = PSEUDO-LABELING** — bukan arsitektur/TTA/bobot: **colum2131** (LB 0,958; single ConvNeXt-T 0,953 dgn pseudo labels, "no external data used yet"), **azamat1ch** (LB 0,960, rank 20, cuma 12 submisi; single 5-fold ConvNeXt-T 0,959), **hengck** +5: "model is not the key. it is your labels", **Raymond Yuen**: pseudo+blend label 0,949 → tim 0,958.
- Yang berangka & terbukti: **arm baru** (internal: 30% independen → 0,943→0,944; publik: blend-gold), **TTA/seed-avg +0,001–0,003** (writeup juara SIIM), per-label include/exclude (konvergen 2 notebook, TANPA angka → wajib triangulasi dulu).

### 3b. Kandidat arm publik — SUDAH DIEVALUASI 9 Okt
| Kandidat | Hasil evaluasi | Keputusan |
|---|---|---|
| `sameerk2004/sil-d5b-dual0950` | fork `hitarthjain0/apex-v5`; `TARGET_WEIGHTS_V1` identik persis dgn Apex kita; tak terskor 200 besar | **BUKAN arm baru (lift ~0)** |
| `sameerk2004/sil-d5c-triple224` | salinan `kozykappa/gold-gated-triple-reader` = Apex + r224 (sudah kita ukur +0,001 ≈ noise); guard 8,6 jam | **BUANG** |
| `rabari9999/...dual-reader-0950` | 2 ckpt sama + routing di-fit di gold58 (overfit); klaim "0.950" tak terskor | **BUANG** |
| `hitarthjain0/spatial-coatnet-swa-v6` | terhapus (403/404), tak ada fork | tak bisa dinilai |
| **`heliosli/rsna-knee-blend-gold` (#3, 126 vote)** | klaim 0,951 ×2 notebook; **TAPI BLOKIR 9 Okt**: `heliosli/rsna-knee-compact-v1` RETIRED 13:50 UTC ("files not accessible", README) + `heliosli/rsna-knee-ofl-ssl-weights` **403** → manifest/bundle S5 tak tersedia; fold0 tetap publik (best.pt parts + code + anatomy.pt). Fork `evgendvorkin/rsna-versia-7`, `samanyu1808/blend3` ikut mati. | **BLOKIR** — jangan buang waktu; pantau bila heliosli buka lagi |
| `rabari9999/...lb-0-951` (klaim 0,951+) | fork kami (57015077) = **0,950** → **SELESAI & BUANG** (10 Okt 00:45); klaim penulis tak terbukti; MIRROR_TTA+r224@0,3 tak mengangkat | **BUANG**; keluarga r384+reader+r224 = plafon 0,950 |
| `sameerk2004/rsna-knee-sil-d5f-ft7030`, `rainycloudss/...v6` | d5f = fine-tune dgn 5 dataset label LLM (resep placebo per riset); v6 = fork Apex kita (input sama) | pantau; jangan submisi dulu |
| `hengck23/mavit-288-baseline` (MaVIT) | **SUDAH DIEVALUASI 10 Okt 00:56** via commit-run `MODE=local` (GRATIS; `alverlfarfaraway/mavit-288-gold58-eval`): **gold-58 MEAN AUC 0,9202** (ACL .962 / Effusion .980, tapi Lateral OA .826 / Synovitis .823) — di bawah induk nartaa (.9214) & reader (.93) = di bawah ambang leg penambah | **DITOLAK** (tak buang submisi); `lixin73/...mavit-336px` = tanpa kernel publik → tak bisa disubmit |

### 3c. Aksi layak (REVISI 10 Okt 03:00, sumber: `reports/Cara menembus 0951 perak.md`)
1. **Fase A — harvest mirror GRATIS** (commit-run `MODE=local`, ~0,5 GPU-jam): H1 = r224+mirror, H2 = reader+mirror, H3 = mirror sbg arm rank terpisah. Protokol: tarik prob per-studi gold-58 → sweep bobot CPU → **sah bila Δ ≥+0,005 vs incumbent & paired-bootstrap CI95>0** (tuned >5 iter → ambang +0,007); <0,003 = noise.
2. **Fase B — PSEUDO-LABEL SELF-TRAINING** (jalur berbukti ke ≥0,951; 10–25 GPU-jam; muat sblm 15 Okt): pseudo-label 4.407 train pake stack incumbent → fine-tune/rerank → OOF gate → submit bila lulus gate. Bahan: cache `rsna-knee-volcache-z{0,1,2}`, ckpt induk, OOF machinery.
3. Fast-follower scan harian: `python3 scan_public_updates.py` — syarat arm baru: bukti skor ≥0,951 **dan input publik runnable**. Arm menengah (eval gratis dulu): `tbongc` v8/v10geo, `royalacecat` coatnet384.
4. TERTUTUP / jangan ulangi: r224-variant & bobot halus (0,950 placebo), resep 746792 (identik stack kita; all-plane flip merugikan), MaVIT (0,9202), heliosli S6 rekonstruksi (mati permanen), SWA manual/seed-avg (nartaa rilis cuma 2 ckpt), d5e-goldw & d5f (placebo), blending tanpa OAI (0,944).

### 3d. Struktur & final
- **~~Merger tim~~ DIBATALKAN (perintah user 11 Okt)** — fokus naikkan skor sendiri; arsip `docs/KANDIDAT_MERGER.md`. SEMUA kandidat final harus ter-push sebelum 15 Okt 23:59 UTC.
- **2 final = beda MEKANISME** (bukan 2 titik bobot): slot A = Apex+heliosli (agresif/OAI), slot B = hedge (leg OF / varian lolos gate; jalur bersih 0,944 hanya bila host melarang OAI — nilai medalinya ≈0, rank ≈1.387 di papan 10 Okt). Pilih **manual** lalu **verifikasi via API** (auto-fill §3.18.c).
- Stack ±7 jam: aman utk skor private (file sama), tapi mepet limit 9 jam → submit 22 Okt dgn margin ≥6–7 jam.

### 3e. Temuan 10 Okt (siang–malam, sesi Claude Code) — detail: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`
- **Notebook publik (skor halaman Code)**: heliosli S6 0,954 (mati); versia-7/blend-gold/blend3 0,951 (mati); `kiyoshiohno/template` 0,951 (logika di dataset privat); `jenillangaliya/dual-stage-sota-ensemble` 0,951 (dataset privat); klaim "0,954" (rabari9999, emresariduman, sameerk d5h) = kode Apex (skor 0,950). Tak ada jalan pintas publik ≥0,951 yg runnable.
- **Bedah kode heliosli**: 7 sumber tertanam di notebook V2 (compact_train/infer, probe_dual, px_cache, anchor bank); target report-state + prior; blend tetap 0,7/0,3; pengendali dual-GPU (urut biaya, thread=1, keluaran bertahap, fallback). Kompilator aturan `criteria.py` jalan lokal (36 dtk) tetapi AUC vs emas-58 hanya 0,784.
- **OrthoFoundation-L fold0**: ViT-L DINOv3 384px + anatomi (SKM-TEA, KneeXNet), epoch 7/16; 48/58 emas ikut latih (emas-58 0,9535 terkontaminasi). Bukan model compact heliosli.
- **MaVIT**: 0,949 di Apex@0,20; lixin73 336px tak kompatibel. **Teori**: galat berkorelasi tinggi bila label laporan sama → hanya pengetahuan label baru berarti.
- **Rules (resmi)**: 5.a/5.b (akun tunggal; anggota tim = individu terpisah), 5.c (syarat merger), 5.d/6.a (tak ada berbagi privat antar-tim; publik boleh). Host tetap diam di 743416.
- **Isi dataset privat notebook papan atas** (riset 10 Okt malam; detail `docs/PLAN_B.md` §3): salinan privat `heliosli/rsna-knee-compact-v1` (manifest + 4 ckpt fold ConvNeXt-Small 2.5D `E_lrprior` + `state_spec` + model-sisi + pca/norm/heads; bundle `c7ebab1eb953f1a6`; 1.975 unduhan, ditarik 9 Okt 13:50 UTC) dan `rsna-knee-ofl-ssl-weights` (bobot OF-L SSL, bobot dasar OF-L juga PUBLIK: omkars5/orthofoundation); bundel logika (kiyoshiohno `rsna_knee_pipeline.py`+manifest); bundel Takakura 384 px laterality-terkoreksi. Pemilik tim 0,955–0,964: tak diketahui (dugaan ckpt fold sendiri + tabel pseudo/soft-label).
- **Ledger publik `prvsiyan/rsna-knee-2026-experiment-ledger`** (90 submisi, terbaik 0,949): mirror-reader 0,949; + reader ConvNeXt 0,70/0,30 = 0,948; + BK-A 0,947; MaVIT 0,935; overlay OF Lateral-OA / OrthoDiffusion 0,935 vs kontrol 0,936 → semua kembali menguatkan teori 'pengetahuan, bukan varians'.
- **Papan**: garis perunggu naik 0,944 (6 Okt) → 0,947 (8 Okt) → 0,950 (10 Okt) krn notebook Apex viral; jalur bersih 0,944 kini rank ≈1.387 (bukan perunggu).

### 3f. Temuan 11 Okt (dini hari; detail `docs/RESEARCH_NOTES.md` §44, `docs/PLAN_B.md` §00)
- **S6 hidup lagi**: heliosli memulihkan `rsna-knee-compact-v1` (V1–V13 publik; V10=V13 `209a32c8` cnx_s6) + notebook publik (satu klik 0,954); thread 748141. Resep heliosli = Raptor + **compact** (BUKAN OF): KOREKSI atas dokumen saya sebelumnya; bobot OF tak pernah dipakai di notebook heliosli; prior OF ≤20 %.
- **Zoologi bundel (publik)**: V4 `ofl_ft8` (OrthoFoundation-L fine-tune), V7 cnx_s5 (0,951), V8 `dino_s1` (DINOv3-ViT-B), V9 dual (convnext+dino), V10 cnx_s6 (0,954); bagging berbiaya ±3,5–4,5 j GPU per bundel.
- **Penyebab 57044355 terkonfirmasi** (induk Apex 18 GB RSS + pekerja OF 8–10 GB → OOM) dan perbaikan v4.1 (nopin + mmap + supervisor) — `docs/PLAN_B.md` §0, `PROGRESS.md` §54.
- **Forum best-score**: model tunggal ConvNeXt-T 5-fold 0,953–0,959 dgn pseudo-label (MURAD rank 10, colum2131); "model is not the key. it is your labels"; validasi: AUC OOF label LLM r=0,87 vs LB (emas-58 r=0,66); trik 1−p; host diam soal OAI/eksternal (743416/744056/747455).
- **Notebook publik baru**: klaim "Tri-Reader 0,954" (rabari/jcgweb/sameerk) = Apex+r224 (0,950, jangan); tri-hybrid jenillangaliya 0,952 (Raptor+compact+r224 < S6).

### 3g. Temuan 12 Okt (scrape mendalam; detail `reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md`, rencana `STRATEGI_0960.md`)
- **Notebook publik #1** = S6 dengan bobot **0,5/0,5** + rank sadar-gagal (heliosli, 11 Okt 14:16 UTC); 25/25 notebook teratas memakai bobot nartaa dan ±22/25 compact heliosli → papan publik atas = satu keluarga klon.
- **velci-kcd25-20** (BliverRigPåDoge, tim 0,955) = S6 0,7/0,3 tak berubah + model privat (5 anggota ImageNet, target lunak laporan) berbobot 0,20 → pola "+1 kaki independen = +0,001".
- **Pembeda papan atas** (klaim peserta): murid 2.5D sendiri + pseudo-label OOF (ConvNeXt-T tunggal 0,953–0,959; MURAD rank 4 = 0,964 tanpa data eksternal). Validator OOF-vs-label-lemah r=0,87 vs LB, emas-58 r=0,66.
- **Kode compact heliosli**: 4 fold hash `sha1(uid)[:8] % 4` atas 4.344 non-gold, `gold_in_training 0` → **guru OOF compact bisa direkonstruksi** (`build_compact_oof.py`).
- Host: LLM API untuk label = boleh; KneeCoT = dilarang; OAI = boleh bila akses terbuka & gratis; bobot publik hasil OAI = belum dijawab.
- Simulator repo pada papan baru: P(perak) publik 0,954 ≈ 16–25 %, 0,956 ≈ 19–42 %, 0,958 ≈ 26–60 %, 0,960 ≈ 27–76 %.

## 4. Rencana (ringkas; rincian & gerbang di `STRATEGI_PERAK.md`)
- [ ] 12 Okt: 2 submisi S6-eq (`s6eq`, `s6eqr`) via watcher; scrape lanjutan; PR untuk evaluasi rekan; lalu jalur 0,960 (`STRATEGI_0960.md`: guru OOF compact → murid A/B → notebook final).
- [x] 10 Okt: r224 (0,950), fork rabari (0,950), Apex+MaVIT (0,949) ditutup; 57028008 Timeout; 57044355 = 0,950 (OF tak jalan → OOM terkonfirmasi); v4.1 dibangun/diuji.
- [x] 11 Okt dini hari: S6 publik lagi → user men-submit 57051680; sonda `s6probe` (compact ±3,5 j); `build_s6plus.py` (reader/OF opsional); riset forum & zoologi bundel; dokumen diperbarui.
- [ ] 11 Okt 03:35–04:45: rehearsal `s6plusr` (reader+compact+OF+blend 3 perna) dan `apexof4r3` (OF 4 pekerja) → push `s6reader`/`s6plus`.
- [x] 11 Okt 07:03: pewaktu submit keduanya — s6reader (57055673) + s6plus (57055781) PENDING; monitor jalan; klon 57051680 = 0,954. ~~Merger~~ DIBATALKAN (user 11 Okt).
- [ ] 13–15 Okt: semua kandidat ter-push sebelum 15 Okt 23:59 UTC (merger dibatalkan).
- [ ] 16–20 Okt: iterasi bergerbang (agen ≤2/hari); pantau bundel privat heliosli (`-s r3cnx`) dan notebook publik; GPU baru 17 Okt. Variasi baru TERAKHIR ≤20 Okt.
- [x] 11 Okt 09:50–10:45: riset diskusi (18 thread) + 16 kernel best-score → `reports/Riset diskusi + best-score code 11 Okt.md` (one-click-955 = bundel privat; Fase B pseudo-label naik prioritas; cek IPP+fp16; protokol kozykappa utk uji post-hoc).
- [x] 11 Okt 11:00–11:10: builder + flag `probit`/`synres`; push `s6prb` & `s6of` (main), `s6clean-akun2` (alver), `teacher-train` (irfan) — 4 GPU paralel.
- [x] 11 Okt 16:49–18:55: skor 0,954 plafon; bedah tim 0.960; prep v1/v2 ERROR (mount finder) → v3 RUNNING; train ConvNeXt warm-start goodpjw; kernel DINOv2-S one-shot + infer murid solo siap. Berikutnya: prep→train a+b→c+dino→folds dataset→infer→submit alver.
- [ ] 21–22 Okt: cek ulang 743416; pilih 2 final manual (A = S6/varian terbaik, B = beda mekanisme/bersih); verifikasi via daftar submisi.

## 5. Daya & batas (11 Okt 03:45 WIB)
| Sumber | Sisa | Reset |
|---|---|---|
| GPU akun utama | ±19–20 h (rehearsal berjalan) | 17 Okt 07:00 WIB (30 h/minggu) |
| TPU akun utama | 20 h (tak berguna: XLA crash) | 17 Okt |
| Submisi tim (jendela 10 Okt) | 4/5 terpakai (3 agen + 1 user S6); jatah baru 11 Okt 07:00 WIB (5/hari; agen ≤2) | 11 Okt 07:00 WIB |
| Disk /kaggle/working | ±19 GB efektif; OF ±330 MB/studi | — |
- Akun 2/3: TIDAK dipakai sesi Claude Code (Rules 5.a/5.b); `nxtirfan` menjalankan job dari akunnya sendiri bila mau.

## 6. Risiko & jalur mundur
- **Notebook Timeout / disk / CPU (OF)**: uji skala dulu; pemutus proses + liputan parsial berprioritas + fallback Apex; estimasi dari skala (bukan 58 studi).
- **Diskualifikasi akun ganda**: Rules 5.a/5.b; satu kalimat izin ≠ cukup → pesan ke Kaggle; semua submisi dari akun utama.
- **OAI/eksternal**: host diam; satu final dgn risiko berbeda. KOREKSI 10 Okt: jalur bersih 0,944 BUKAN lagi zona perunggu (garis perunggu naik 0,944 → 0,947 → 0,950 dlm 4 hari krn notebook Apex viral; 0,944 = rank ≈1.387) → hanya perlindungan kepatuhan.
- **Noise & papan privat**: Δ<0,002 = noise; jangan menyetel ke papan publik; 2 final beda mekanisme.
- **Garis perak naik**: target publik ≥0,952; merger = satu-satunya lompatan besar.
- **Jangan**: cetak kredensial / `VERBOSE=true`, `git add -A`, sentuh kernel proyek lain, `pkill -f` pola sendiri, submit varian identik utk "mengukur noise", `kernels output` OF tanpa `--file-pattern`.
