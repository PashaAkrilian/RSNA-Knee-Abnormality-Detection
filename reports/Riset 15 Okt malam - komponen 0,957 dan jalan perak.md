# Riset 15 Okt 2026 malam — komponen 0,957 dan jalan ke perak (sesi Claude Code)

> Permintaan user: "coba riset dengan mendalam lagi saya mau silver dan riset ini juga
> https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext".
> Sumber: `kaggle kernels pull` 13 notebook baru (pemindai `scan_public_updates.py`), papan publik 15 Okt 12:47 UTC,
> forum (749145 OAI, 749359 S6F nartaa, pindai ID topik 749360–749720).

## 1. Papan & posisi (15 Okt 19:49 WIB)
| | Nilai |
|---|---|
| Tim | 6.040 |
| ≥0,958 | 132 |
| ≥0,957 | **327** → garis perak (rank 301) = **0,957** |
| Tepat 0,956 | 941 (rank 328–1.268) |
| Kita | **rank 398** (0,956) ≈ ke-71 di blok 0,956 |

## 2. bliverrigpdoge "R7 + Raptor + knee-centred ConvNeXt" (0,957, tim rank 199)
- Basis = notebook R7 + Raptor heliosli (0,956) **tanpa ubahan** (tanpa r224, tanpa cermin).
- Model mereka ("KCD25"): ensembel ConvNeXt-S + EfficientNetV2-S pada **potongan knee-centred** (kompartemen medial & lateral dari seri sagital & koronal, plus aksial), anggota ImageNet, dilatih pada **label laporan lunak dreaddevelopment (d25)**.
- Blend akhir: rank per temuan **0,8 basis + 0,2 model mereka** → 0,956 → **0,957**. Versi 11 Okt (`velci-kcd25-20`): resep S6 0,954 + KCD25 0,2.
- **Bobot di dataset PRIVAT** `rsna-knee-kcd25leg` (fork = 0,956). Tak bisa dipakai; yang bisa diambil = pola.
- Pelajaran: model sendiri yang **beragam** (input & label berbeda), bobot ±0,2 → +0,001 publik. Sama dgn jalur murid kita (r2a: ConvNeXt-T 2.5D, label konsensus ⊕ OOF-R7).

## 3. Notebook publik lain (14–15 Okt)
| Notebook | Isi | Catatan |
|---|---|---|
| heliosli "=0.9573" (kini "old version") | R7 + r384 + r224 (A224 0,35), tanpa flip | = resep `r7r224` kita; salinan user 57223271 = 0,956 |
| **heliosli "=0.9576" (versi baru)** | **R6** (`fe055406`) + **flip-TTA v4c** + r384 + r224 | flip: tiap fold juga menilai jendela tercermin, medial/lateral ditukar balik, rata-rata logit, MCL dari lintasan biasa |
| leoprovorov "0.957 Explained" (posisi 206) | r384 (biasa+cermin) + r224 (biasa+cermin) + **R7 0,60 + S6F 0,40 di slot compact** | handbook 56 hal (dataset privat); estimasi ρ antar model publik 0,86–0,89 → 8 model ≈ 0,960 |
| hoshinoyumemi "0.939→0.956" | jurnal 14 submisi | **penanda auto-select di UI = 2 skor tak-dibulatkan tertinggi milik sendiri**; seri persis → submisi lebih awal menang; reader di atas R7 = turun |
| sujanmajhisuzan v14 dualraptor | R7 + flip v4c + r384 + r224 | ≈ `r7v4` kita |
| rjings router lokal | + 5 kantong R384 lokal (±8 mm) utk Lateral OA, PF OA, Synovitis (residual 35 %) | — |
| dennis6886 playbook | R7 0,7 + S6 (compact-v1) 0,3 + reader per-temuan kecil + r224 cermin | — |
| ltcsmokeboi gp958d, mizeroluckygall, isotriage h957, oscar | salinan persis notebook heliosli/varshitha | — |

## 4. S6F (nartaa, forum 749359)
- "S6 ConvNeXt, all data" = **0,950 publik sebagai model tunggal** (fallback CoAtNet hanya utk studi gagal); fold-0: OOF 0,907 vs CoAtNet 0,890, **gold-58 0,931** (≈ R7 0,9312 versi kita).
- Arsitektur heliosli S6 (convnext_small in22k, perhatian spasial per temuan + MIL jendela), **tampilan FOV penuh + potongan knee-centred 100 mm** (anchor SKM-TEA), cache 320 → crop 288; label **JEV-blend** (= keluarga label Raptor); 16 epoch, semua 4.263 studi berlabel laporan.
- Bobot `nartaa/rsna-knee-s6fall-weights` (1 checkpoint 200 MB + manifest kode; CC BY-NC 4.0). Peluncur nartaa: **anggaran tahap S6F sampai 14.400 dtk**; test tersembunyi = **1.284 studi**.
- Bukti: leoprovorov R7 0,6 + S6F 0,4 di slot compact → 0,957.

## 5. Implikasi & rencana
1. **Model tambahan yang beragam adalah satu-satunya pola naik ke 0,957** (bliverrigpdoge, leoprovorov). Menyetel bobot saja tidak menaikkan. → jalur murid kita tetap utama: `r7v4c40` (15 Okt malam), 4 fold + 2 full-data (17 Okt).
2. **S6F** = kandidat anggota ke-3 slot compact (publik, setara R7, label beda). Syarat: pengaman waktu (tahap terakhir, dilewati bila waktu < ambang); keputusan setelah durasi run `r7v4c40` diketahui.
3. **Penanda auto-select** (UI Submissions) → minta user membacanya: menjawab mana yg tertinggi di antara `r7r224m`, `r7v3c`, 57223271, `r7v4c40`.
4. R6+flip (heliosli 0,9576) vs R7: tak ada bukti kuat; tetap R7.

## 6. Eksekusi malam ini
- **57231466 `r7v4c40`** disubmit 20:08 (izin user: 1 jatah): r7r224m + flip-TTA compact stride 1 + murid r2a 0,40 di slot compact (commit: GOLD58 0,9282, GPU compact 18,8 dtk/studi). Rank saat submit 403 (≥0,957 = 331). Skor ±00:30–01:00.
- **Integrasi S6F** di `build_r7murid.py` (opsi `s6f=<bobot>`): skrip publik nartaa disimpan `kaggle_base/s6fall_solo_nartaa_20261015.py` (sha256 6e87350d…), ditanam UTUH dengan 8 patch berassert: komponen A dilewati (A = tahap-1 Raptor-384 kita, sumber byte-identik `bcd864cb…`), guard skrip = sisa waktu run (selesai ≤ NB_T0 + 8,0 j; dilewati bila sisa < 45 mnt), `submission.csv` induk tak disentuh, keluaran `k1/B/B_probs.csv` → `s6f_raw.csv`. Kaki S6F di slot compact SEBELUM murid; gerbang cakupan ≥97 % dan ρ(S6F, R7) rata-rata ≥0,5 (n ≥100); gagal → tanpa kaki S6F. Uji lokal `check_s6f()` lulus.
- Kernel: `r7v6` (flip stride 1 + S6F 0,35 + murid 0,30) dan **`r7v3s`** (TANPA flip + S6F 0,35 + murid 0,30; bobot efektif compact R7 0,455 / S6F 0,245 / murid 0,30) → commit-run.
- **Rencana 16 Okt 07:01**: `r7v3s` + `r7v3c40` (keduanya tanpa flip) → `r7v3s` vs `r7v3c40` = efek S6F; `r7v4c40` vs `r7v3c40` = efek flip. Varian tanpa flip dipilih agar S6F pasti sempat jalan (log run penilaian tak terbaca).

## 7. Riset kuantitatif (15 Okt 21:30–22:10)
### 7.1 Simulasi perak sadar-klaster (`silver_sim_cluster.py`, papan nyata 15 Okt)
Blok 0,956–0,957 = keluarga resep (salinan byte-identik → satu skor privat per keluarga, seri dipecah waktu submit). Proyeksi akhir 6.150 tim, perak = rank ≤308; keluarga: R7 dasar 380, R7+r224 520, R7+r224 cermin 230, R6+flip 180, r224 cermin + S6F 140, variasi kecil 150; tim kuat ≥0,958 200–300.

| δ privat vs R7 dasar | 0 | +0,0003 | +0,0006 | +0,0010 | +0,0015 | +0,0020 |
|---|---|---|---|---|---|---|
| P(perak), 12 skenario | 2–10 % | 13–19 % | 27–47 % | 49–86 % | 76–99 % | 91–100 % |

→ salinan publik ≈ gagal; target δ ≥ +0,001 (di atas keluarga S6F publik).

### 7.2 S6F pada gold-58 (kernel `dimaspashaakrilian/rsna-knee-s6f-gold`, 15 Okt 22:02)
- 58/58 terskor, 0 fallback, 320 dtk di 2 GPU (stabil ±7,3 dtk/studi/GPU) → tahap S6F di 1.284 studi test ≈ **1,3–1,5 j**.
- AUC gold-58: **S6F 0,9314** · R7 0,9312 · murid r2a 0,9282 · Raptor384* 0,9229 (*terkontaminasi). Spearman: S6F~R7 0,871, S6F~Raptor 0,879, S6F~murid 0,882, R7~murid 0,911, R7~Raptor 0,865.
- Blend vs dasar (Raptor .5 + R7 .5 = 0,9355; bootstrap 2.000×): +S6F .35 compact **+0,0010** (P 0,68) · +murid .40 saja +0,0003 (0,55) · +S6F .35 +murid .30 (= `r7v3s`) **+0,0014** (0,71) · +S6F .25 +murid .30 +0,0021 (0,83). Grid S6F 0,15–0,45 × murid 0,2–0,4: datar +0,001…+0,002.
- Struktur: **S6F di separuh Raptor .35 (+murid .30 compact) +0,0034 (P 0,92)**; S6F kaki-3 atas .20–.25 +0,0031–0,0035 (0,88); 4 kaki setara .25 +0,0044 (0,90). Catatan: proksi Raptor = r384 saja (tanpa r224/cermin) → struktur yg mengecilkan Raptor bisa terlalu dipuji → diuji di papan: **16 Okt `r7v3s` (compact) vs `r7v3sr` (separuh Raptor)**.
- Kesimpulan: S6F = model setara R7 yang beragam (+0,001 pasti arah); murid kita paling berguna BERSAMA S6F.

## 8. 16 Okt — papan, ketentuan host, evaluasi gold SETIA
- **Papan**: `r7v3s` (S6F compact) = 0,957 rank 478; **`r7v3sr` (S6F separuh Raptor) = 0,957, rank 495→317**. 13:40: 6.066 tim, ≥0,958 = 213 (naik cepat; salinan heliosli 0,958 = R6+flip+bobot per-temuan Syn .8/LatOA .2/PFOA .6/Baker .35), perak rank 303 = 0,957, kita 337.
- **Ketentuan host** (Chris Carr, Director of Informatics RSNA, 14 Okt, 743416): data eksternal yg hanya butuh registrasi + persetujuan lisensi = sah; butuh perjanjian institusi = tidak; minta transparansi pengungkapan dataset. OAI & fastMRI → sah.
- **Evaluasi gold setia** (`build_gold_full.py`: pipeline submisi persis dijalankan pd 58 gold sbg "test"; rekonstruksi `r7v3sr` = submission.csv-nya, selisih 6e-08; `gold_eval_full.py`): komponen r384m 0,9228 · r224m 0,9160 · R7 0,9312 · S6F 0,9314 · murid 0,9282.
  | Varian | AUC gold | vs r7v3sr |
  |---|---|---|
  | r7r224m (dasar) | 0,9354 | −0,0033 |
  | r7v3s (S6F compact) | 0,9355 | −0,0033 |
  | **r7v3sr** | **0,9387** | — |
  | r7v3sr + bobot heliosli | 0,9398 | +0,0010 (P 0,71) |
  | S6F Raptor 0,45 + murid 0,30 | 0,9401 | +0,0014 (P 0,88) |
  | S6F Raptor 0,55 + murid 0,30 | 0,9410 | +0,0023 (P 0,90) |
  | S6F Raptor 0,55 + murid 0,30 + bobot heliosli | 0,9416 | +0,0030 (P 0,91) |
  Murid 0,30 menambah +0,0011…+0,0014 (P 0,75–0,80) di konfigurasi ini. Catatan: puncak grid ±24 konfigurasi pd 58 studi = bias seleksi → dipilih nilai bulat konservatif **`r7v5a` = S6F 0,50 separuh Raptor + murid 0,30 + bobot heliosli**.
- **Flip-TTA R7 (run `gold-f`)**: R7flip AUC 0,9317 vs R7 0,9312, Spearman 0,9976 (hampir identik); di blend efek flip −0,0004…−0,0007 (P>0 0,28–0,34); tahap R7 +40 % waktu (1.036 vs 731 dtk / 58 studi). Papan: `r7v4c40` (flip) = 0,956. → **FLIP DIBUANG** dari semua varian final.

## 9. 16 Okt malam — papan 0,958, murid 4 fold, R6+flip
- Papan 19:48: 6.085 tim; ≥0,958 = 446 → **garis perak publik (rank 304) = 0,958**; =0,957 = 345; kita rank 461 = ke-15 di blok 0,957 (skor tak-dibulatkan dekat 0,9575). heliosli & varshitha 0,958.
- Murid r2a fold 2–3 (akun utama): gold 0,9311 / 0,9326 → ensembel 4 fold gold **0,9334**.
- Gold setia `gold-h958` (notebook heliosli 0,958): R6+flip **0,9366** (ρ~R7 0,979); resep heliosli direkonstruksi 0,9382 (selisih rekonstruksi 5,6e-3).
- Resep: `r7v6sr` (r7v3sr + murid4) 0,9399; **`r7v6a` (S6F .5 Raptor + murid4 .3 + wc) 0,9414** (+0,0027 vs r7v3sr, P 0,90); + R6flip di compact (0,3–1,0) = −0,0017…+0,0002 → tidak dipakai.

## Catatan status terkini — 19 Okt 2026 02:56 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; cutoff ini sementara karena medali memakai split privat 70%). Pada cek API 02:55, `g6c`, `g12`, `g13` masih PENDING. Dua kernel murid r4 di alver sudah dua kali di-requeue dan kini RUNNING pada cek 02:56. Kuota main/alver/nxtirfan: 19,87/17,75/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
