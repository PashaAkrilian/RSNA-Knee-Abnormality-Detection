# laporan_push.md — Evaluasi & jalan menuju SILVER
> Wadah TETAP untuk hasil evaluasi tiap submisi/probe + rencana peningkatan. **Setelah tiap run/submisi/skor baru: tambah baris §2, perbarui §1 & §4.**
> Konteks aturan: `MEMORY.md` · rencana & gerbang: **`STRATEGI_PERAK.md`** · langkah harian: `HANDOFF.md` · arahan harness: `push.md` · riset terbaru: **`reports/Ringkasan riset 10 Okt (sesi Claude Code).md`** · arsip OpenCode: `reports/Cara menembus 0951 perak.md`, `reports/Strategi meraih perak RSNA.md`, `research_notes/`.
> Dibuat 9 Okt 2026 ~21:45 WIB, diperbarui 10 Okt ~21:30 WIB. Bahasa kerja: Indonesia.

## 1. Posisi vs garis perak (papan 10 Okt 19:37 WIB, 5.666 tim)
| Hal | Nilai |
|---|---|
| Tim | `Krazy KIller` = `alverlfarfaraway`, `dimaspashaakrilian`, `nxtirfan` (3 akun di halaman tim; lihat risiko akun di `STRATEGI_PERAK.md` J4) |
| LB kita | **0,950** (Apex v2, 56943646) — **rank 530**; 564 tim sama-sama 0,950 |
| Peta blok | ≥0,951 = **348** · ≥0,952 = **217** · ≥0,953 = 162 · ≥0,954 = 132 · ≥0,960 = 30 · garis perak (5 %) ≈ rank 283, perunggu ≈ 567 |
| Dinamika | tim ≥0,951 +39 dan ≥0,952 +20 dalam ±12 jam → 0,951 sudah di tengah blok yg hanya separuh dapat perak; **target kandidat final ≥0,952–0,954** |
| Peluang perak (simulasi) | Apex 0,950: 14–20 %; +0,002 sejati: 26–39 %; +0,004: 40–67 %; tim publik 0,956–0,958 (merger): 23–82 %; **100 % mustahil** (`silver_prob_sim.py`) |
| Deadline | merger/entry **15 Okt 23:59 UTC**; final **22 Okt 23:59 UTC**; maks 2 final (PILIH MANUAL); publik 30 % / privat 70 % |
| Aturan terima uji | **≥0,952 → kandidat** · 0,951 → cadangan · ≤0,950 → buang · Δ<0,002 = noise |

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
| 11 | 10 Okt 21:10 | **57044355** | **Apex 0,70 + OF v2** (`kaggle_kernel_apexof2`: potongan 8, 4 proses, penjaga disk, thread=1, prioritas, pemutus proses, reuso anatomi) | PENDING (disubmit 10 Okt 21:10 WIB) | — | G0 LULUS; **ID 57044355**; hasil ±04:00 WIB 11 Okt | uji skala: 194 studi/jam (6,8 j; reuso anatomi tak membantu); stub v3 lulus (GPU kosong 15 GB); terima bila ≥0,952 (0,951 = cadangan) |

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
- **Merger tim (15 Okt 23:59 UTC)** — hanya user; cek dulu `kaggle competitions submission-limits` (batas gabungan ~355–385); SEMUA kandidat final harus ter-push sebelum 15 Okt.
- **2 final = beda MEKANISME** (bukan 2 titik bobot): slot A = Apex+heliosli (agresif/OAI), slot B = hedge (jalur bersih 0,944 / leg OF / varian lolos gate). Pilih **manual** lalu **verifikasi via API** (auto-fill §3.18.c).
- Stack ±7 jam: aman utk skor private (file sama), tapi mepet limit 9 jam → submit 22 Okt dgn margin ≥6–7 jam.

### 3e. Temuan 10 Okt (siang–malam, sesi Claude Code) — detail: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`
- **Notebook publik (skor halaman Code)**: heliosli S6 0,954 (mati); versia-7/blend-gold/blend3 0,951 (mati); `kiyoshiohno/template` 0,951 (logika di dataset privat); `jenillangaliya/dual-stage-sota-ensemble` 0,951 (dataset privat); klaim "0,954" (rabari9999, emresariduman, sameerk d5h) = kode Apex (skor 0,950). Tak ada jalan pintas publik ≥0,951 yg runnable.
- **Bedah kode heliosli**: 7 sumber tertanam di notebook V2 (compact_train/infer, probe_dual, px_cache, anchor bank); target report-state + prior; blend tetap 0,7/0,3; pengendali dual-GPU (urut biaya, thread=1, keluaran bertahap, fallback). Kompilator aturan `criteria.py` jalan lokal (36 dtk) tetapi AUC vs emas-58 hanya 0,784.
- **OrthoFoundation-L fold0**: ViT-L DINOv3 384px + anatomi (SKM-TEA, KneeXNet), epoch 7/16; 48/58 emas ikut latih (emas-58 0,9535 terkontaminasi). Bukan model compact heliosli.
- **MaVIT**: 0,949 di Apex@0,20; lixin73 336px tak kompatibel. **Teori**: galat berkorelasi tinggi bila label laporan sama → hanya pengetahuan label baru berarti.
- **Rules (resmi)**: 5.a/5.b (akun tunggal; anggota tim = individu terpisah), 5.c (syarat merger), 5.d/6.a (tak ada berbagi privat antar-tim; publik boleh). Host tetap diam di 743416.

## 4. Rencana (ringkas; rincian & gerbang di `STRATEGI_PERAK.md`)
- [x] 10 Okt: r224 (0,950), fork rabari (0,950), Apex+MaVIT (0,949) → ditutup; riset notebook publik & bedah kode heliosli; uji kelayakan OF (probe) lulus; 57028008 timeout → uji skala + perbaikan v2.
- [ ] **G0** (malam 10 Okt): hasil `ofscale3/4` → submit `rsna-knee-apexof2` (hasil ±7 jam). **G1** (11–13 Okt): ≥0,952 kandidat utama · 0,951 cadangan · ≤0,950 tutup OF.
- [ ] **J1 Merger** (USER; ≤15 Okt 23:59 UTC): `docs/KANDIDAT_MERGER.md`; sebelum itu kepastian akun ke Kaggle (`docs/DRAF_PESAN_KAGGLE_SUPPORT.md`).
- [ ] 16–20 Okt: iterasi bergerbang (agen ≤2/hari); scan harian; GPU baru 17 Okt. Variasi baru TERAKHIR ≤20 Okt.
- [ ] 21–22 Okt: cek ulang 743416; pilih 2 final manual (beda mekanisme) dari submisi terskor; verifikasi via daftar submisi.

## 5. Daya & batas (10 Okt 21:30 WIB)
| Sumber | Sisa | Reset |
|---|---|---|
| GPU akun utama | 27,1 h (uji skala berjalan) | 17 Okt 07:00 WIB (30 h/minggu) |
| TPU akun utama | 20 h (tak berguna: XLA crash) | 17 Okt |
| Submisi tim hari ini | sisa 3 (agen ≤2/hari) | 11 Okt 07:00 WIB (5/hari) |
| Disk /kaggle/working | ±19 GB efektif; OF ±330 MB/studi | — |
- Akun 2/3: TIDAK dipakai sesi Claude Code (Rules 5.a/5.b); `nxtirfan` menjalankan job dari akunnya sendiri bila mau.

## 6. Risiko & jalur mundur
- **Notebook Timeout / disk / CPU (OF)**: uji skala dulu; pemutus proses + liputan parsial berprioritas + fallback Apex; estimasi dari skala (bukan 58 studi).
- **Diskualifikasi akun ganda**: Rules 5.a/5.b; satu kalimat izin ≠ cukup → pesan ke Kaggle; semua submisi dari akun utama.
- **OAI/eksternal**: host diam; satu final dgn risiko berbeda (jalur bersih 0,944 = hasil bronze).
- **Noise & papan privat**: Δ<0,002 = noise; jangan menyetel ke papan publik; 2 final beda mekanisme.
- **Garis perak naik**: target publik ≥0,952; merger = satu-satunya lompatan besar.
- **Jangan**: cetak kredensial / `VERBOSE=true`, `git add -A`, sentuh kernel proyek lain, `pkill -f` pola sendiri, submit varian identik utk "mengukur noise", `kernels output` OF tanpa `--file-pattern`.
