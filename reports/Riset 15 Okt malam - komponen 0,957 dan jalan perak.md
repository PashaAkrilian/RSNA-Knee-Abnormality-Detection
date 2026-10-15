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
