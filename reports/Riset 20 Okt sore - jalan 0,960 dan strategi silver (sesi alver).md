# Riset 20 Okt 2026 sore — jalan ke 0,960, validasi, dan strategi silver (sesi Claude Code di mesin alver)

> Permintaan user (alver): riset mendalam apa yang bisa mengejar 0,960 dan silver, validasi, lalu eksekusi dengan **1 submission** (jatah 20 Okt tersisa 2).
> Sumber: Kaggle CLI akun alver (papan unduhan 20 Okt 09:13 UTC = 16:13 WIB, `kernels list` urut skor/dateRun/vote, forum `competitions topics`, `datasets list/files`), kernel alver (`h4`, `h4m`, murid r4/r2a-full), dan data gold lokal repo. Skrip analisis ada di scratchpad sesi (tidak di-commit); angka di bawah bisa direproduksi dari data repo + OOF kernel alver.
> Tanda **[dugaan]** = kesimpulan tanpa bukti langsung.

## 0. Ringkasan
1. **0,960 publik tidak realistis dengan komponen publik**: plafon semua resep publik = 0,958 (dikonfirmasi lintas ≥4 salinan independen + papan). Tim 0,960+ (133 tim) memakai model sendiri berlabel pseudo dari GAMBAR (forum 735304: ConvNeXt-T 2.5D single 5-fold 0,953–0,959). Melatih model sekelas itu tidak muat di sisa waktu/kuota akun alver (GPU 1,65 j).
2. **Silver ditentukan papan PRIVAT, dan kuncinya keunikan + keunggulan nyata, bukan tampilan 0,959.** Simulasi klaster tim pada papan 20 Okt: salinan murni 0–12 %; resep unik δ=0 → 24–32 %; δ=+0,0003 → 39–62 %; δ=+0,0006 → 56–87 %; δ=+0,001 → 76–99 %.
3. **Bukti gold-58 (dihitung ulang)**: semua resep publik (goodpjw v27, heliosli, h4, h3) = 0,9382–0,9384; **+ S6F (separuh Raptor) + murid (compact) = 0,9404–0,9407 (+0,002, P≈0,75)**. Bobot per temuan V1/V2/seragam tidak terbedakan di gold. SE bootstrap gold = 0,012.
4. **Koreksi hipotesis 20 Okt siang**: reader Part B di h4 berbobot **0,003–0,035 per temuan** (bukan 0,15; `OWN_W` tidak dipakai karena `OWN_W_PER_TARGET` lengkap). Disimulasikan dengan proksi apa pun (ρ 0,69–0,94 ke stack), efeknya pada AUC gold = **+0,0000**. Jadi "reader = pembeda 0,0016" hampir pasti keliru → perkiraan h4 = **0,958** **[dugaan]**. h4m juga BUKAN "h4 + murid": basisnya goodpjw v27 (r224 bercermin) + V2 + murid 0,2 slot final + reader.
5. **Temuan operasional**: akun alver TIDAK bisa membaca dataset murid akun utama (403), tetapi **memiliki 8 model murid sendiri** (output kernel `student-r4-a/b` fold0–3 + `student-r2a-full/full2` fold6–9). Gold OOF gabungan = **0,9332** (r2a 4 fold akun utama 0,9334; ρ 0,990) → notebook kelas g6c bisa dibangun & disubmit dari alver tanpa dataset akun utama.
6. **Eksekusi**: `h5` = h4 PERSIS + S6F 0,5 di separuh Raptor + murid alver-8 0,4 di slot compact + R6 stride 2 (+ penjaga waktu reader 7,9 j). Lihat §6.

## 1. Papan publik 20 Okt 16:13 WIB
| | Nilai |
|---|---|
| Total tim | **6.261** (19 Okt 17:54: 6.219) |
| Garis medali | emas ≤ rank 22 (0,964) · **silver ≤ 313 (0,958)** · perunggu ≤ 626 (0,958) |
| ≥0,959 | **203 tim** (19 Okt: 188): 0,969×1 · 0,968×3 · 0,967×6 · 0,966×5 · 0,965×4 · 0,964×12 · 0,963×19 · 0,962×16 · 0,961×25 · 0,960×42 · 0,959×70 |
| Blok 0,958 | **1.366 tim** (rank 204–1569) · 0,957 = 175 · 0,956 = 284 |
| Krazy KIller | **rank 620** (0,958) = urutan ke-417 di blok 0,958 → publik pas di garis perunggu, bukan silver |
| Proyeksi tutup | ±6.350 tim → silver ≈ rank 317; ≥0,959 ±240 tim → tim 0,958 butuh ±top 6 % blok |

Catatan: tim ≥0,959 sebagian besar aktif (185 dari 203 submit sejak 19 Okt; median 100 submisi). Ada tim agen AI di top-25 (mis. "GM-Enhanced Kaggle Agent" 0,966, "tine.sh agent" 0,964).

## 2. Apa yang dilakukan tim 0,960+ (forum)
- 735304 "Best single-model score": colum2131 (kini 0,964): ConvNeXt-T 2.5D single 0,953, DINOv2-S 0,951, ensembel 0,955 tanpa data eksternal; **pseudo-label dibuat dari GAMBAR dengan input yang sama seperti saat submit, lalu diblend dengan label LLM pada rasio tetap** (komentar 16 Okt). azamat1ch (kini rank 2): ConvNeXt-T 5-fold **0,959**, EffNetV2-S 0,956, CoAtNet 0,952. Chris Deotte: dengan pseudo-label, model tunggal menyamai ensembel CV terbaik. hengck23: "model is not the key, it is your labels".
- 748111 "Single sub 0.960": satu submisi model tunggal (dikabarkan di Rednote).
- Pola ini **sama dengan jalur murid kita** (ConvNeXt-T 2.5D, label 0,5 konsensus + 0,5 OOF guru gambar). Murid kita gold 0,9332–0,9334 ≈ R6flip 0,9366; skor publik murid sendirian belum pernah diukur.

## 3. Notebook & dataset publik terbaru (sejak 18 Okt)
- **Tidak ada notebook publik ≥0,959.** Urutan skor teratas: versia-7, goodpjw 0,958, lavinwins (salinan), heliosli 0,9586, … — semua keluarga yang sudah diuji tim. Notebook baru 19–20 Okt (`basith007/model-ensembling`, `toghrulaliyev/...aze-team`, `seanmoch/...tri-reader`, `notfu4y/...svmri`) memakai dataset privat atau aset lama; tak ada komponen baru tervalidasi.
- `bliverrigpdoge/rsna-knee-trainkit` (19 Okt, 366 unduhan): **kit LATIH** dari tim 0,959 (label guru `D1a_teacher.csv`, `D1f0a_teacher.csv`, `consensus_labels.csv`, init ConvNeXt-S/T in12k dan **encoder S6F**), BUKAN model jadi. Berguna hanya untuk melatih murid baru (lihat §5 jalur opsional).
- `jorgerome/rsna-knee-weights-v2` (DINOv2 fold, tanpa kode/bukti), `cyruspinto/...engine9/10/11` (tanpa OOF/gold/skrip), `cyruspinto/...consensus-labels-public`, Mycelis (diperbarui 20 Okt; tim sudah membuktikan gold 0,596 = rusak) → **abaikan**.

## 4. Aturan & risiko (belum ada jawaban host)
- 750899 (20 Okt) & 749145: minta putusan apakah checkpoint publik hasil fine-tune OAI (Raptor nartaa) sah. Host belum menjawab; putusan negatif akan mengenai hampir seluruh papan ≥0,95 (bukan khusus kita). Jalur bersih tanpa OAI ±0,944–0,950 tidak kompetitif untuk medali → tidak ada lindung nilai murah; terima risiko kolektif.
- 750889 (fastMRI/SKM-TEA untuk bobot pra-latih: R7/kode OF), 750752 (label LLM via API) — juga belum dijawab.

## 5. Validasi kuantitatif (gold-58; skrip scratchpad `gold_blend_eval.py`)
| Komponen | Gold | | Kandidat blend | Gold | Δ vs goodpjw | P(Δ>0) |
|---|---|---|---|---|---|---|
| r384m | 0,9229 | | goodpjw v27 (V1, r224m) | 0,9384 | — | — |
| r224m / r224 polos | 0,9160 / 0,9150 | | heliosli/h958 (V1, r224 polos) | 0,9382 | −0,0001 | 0,42 |
| R6flip / R6 stride 2 | 0,9366 / 0,9377 | | h4 (V2, r224 polos) | 0,9384 | −0,0000 | 0,46 |
| R7 | 0,9312 | | g6a (V1, S6F .5 R, murid .3) | 0,9399 | +0,0016 | 0,70 |
| S6F | 0,9314 | | g6c (V1, S6F .5 R, murid .4) | 0,9407 | +0,0023 | 0,77 |
| murid r2a 4 fold (akun utama) | 0,9334 | | g6c dgn murid alver-8 | 0,9404 | +0,0021 | 0,74 |
| murid r4 (alver) | 0,9312 | | **h5 = h4 + S6F .5 R + alver-8 .4** | **0,9404** | **+0,0020** | 0,75 |
| murid r2a-full (alver) | 0,9328 | | h5 dgn V1 | 0,9407 | +0,0023 | 0,77 |
| **murid alver-8** | **0,9332** | | g12-ish (R6 .6/R7 .4 + S6F + murid) | 0,9406 | +0,0023 | 0,76 |

- Per temuan (gold, arah saja; SE per temuan ±0,03–0,05): Raptor unggul di Fracture (0,958 vs R6 0,887), Medial Meniscus, Effusion; compact/murid unggul di Lateral OA, PF OA, Synovitis, Contusion. Bobot V1/V2 (hasil probe papan oleh heliosli) memberi Raptor 0,6–0,85 di PF OA & Synovitis — berlawanan dengan arah gold, tetapi dalam derau → **jangan disetel di gold** (aturan tim tetap berlaku).
- ρ Spearman gold: alver-8 ~ r2a4 0,990 · murid ~ R6flip 0,93 · S6F ~ r384m 0,88 · R7 ~ R6flip 0,98.
- Asal +0,002 h5 vs goodpjw per temuan (gold): naik di MCL +0,011, Lateral Meniscus +0,015, Lateral OA +0,014, PF OA +0,017, ACL +0,005; turun di Medial Meniscus −0,010, Medial OA −0,008, Effusion −0,007, Fracture −0,010 (label tempat Raptor terkuat; S6F di separuh Raptor mengencerkannya). Pola konsisten dengan AUC komponen, tetapi tidak disetel per temuan (58 studi).
- Efek reader Part B (bobot 0,003–0,035) pada gold dengan 3 proksi (murid ρ 0,94, S6F ρ 0,91, murid berderau ρ 0,69): **Δ = +0,0000** di semua proksi.

## 6. Eksekusi 20 Okt: `h5` (builder `build_h5.py`, kernel `alverlfarfaraway/rsna-knee-h5`)
- Basis: `kaggle_base/h4_heliosli_0958_exact_20261020.ipynb` (salinan kernel h4 alver, tanpa ubahan).
- Tambahan (templat `build_r7murid.py` yang dipakai g6c/g10a/g12, tanpa ubahan logika): Stage 0 murid (8 fold dari output kernel alver, BUDGET 5.400 dtk; cek gold commit dimatikan untuk hemat GPU), S6F setelah R6 (penjaga NB_T0 + 8,0 j), blend: S6F 0,5 di separuh Raptor + murid 0,4 di slot compact, R6 eval-stride 2, penjaga Part B 7,9 j.
- Uji lokal sebelum push: (a) sel blend dieksekusi dengan data palsu di 5 skenario (lengkap; 3 studi; murid hilang; murid 5 % NaN; S6F hilang) → keluaran **sama persis** dengan implementasi ulang independen, fallback benar; (b) pencarian fold diuji pada tata letak `/kaggle/input/<slug>` dan `/kaggle/input/notebooks/<owner>/<slug>` → tepat 8 fold, kernel lain tidak ikut.
- Commit-run & submisi: lihat §8 (diisi setelah run).

## 7. Strategi silver 20–22 Okt
**Prinsip**: papan publik 30 % (±397 studi; SE ±0,002) tidak bisa membedakan resep 0,957–0,958; gold-58 juga tidak (SE 0,012). Yang bisa dikendalikan: (1) keunikan resep (menghindari skor privat bersama klaster salinan), (2) komponen dengan bukti positif independen (S6F + murid: gold +0,002 dan papan g8a>g8b, g6a>salinan goodpjw), (3) pilihan 2 final yang tidak berkorelasi sempurna.

1. **Final A (utama)** = resep unik terbaik dengan S6F + murid di keluarga R6: `g6a` (lompatan rank terbesar → skor publik tak-dibulatkan tertinggi tim) atau `g6c`/`h5` bila skornya ≥ g6a. Jangan pilih salinan publik murni (h4, versia-7, goodpjw copy) — peluang salinan 0–12 %. **Ini berlaku juga bila h4 tampil 0,959**: salinan byte-identik berbagi SATU skor privat dengan semua penyalin notebook heliosli (193 vote), dan seri dipecah waktu submit → kita (submit 20 Okt) di ekor keluarga itu. Varian unik di atas basis yang sama (`h5`, `h4m`) yang menangkap keunggulan basis tanpa terjebak di blok salinan.
2. **Final B (lindung nilai)** = resep unik dengan mekanisme berbeda yang tetap 0,958: `g12` (R6+R7 compact) atau `g10a` (V2, R6 stride 2). Hindari 0,957 (g13, g16/g17, r7v6a, h3): 0,957 vs 0,958 pada pasangan resep yang sangat berkorelasi adalah bukti lemah-sedang bahwa varian itu lebih buruk.
3. **Jatah 21–22 Okt (±10)**: hanya untuk (a) kandidat final yang belum terskor, (b) varian berbeda mekanisme dengan prior positif. Jangan menghabiskan jatah untuk variasi bobot (placebo). Submisi baru terakhir ≤22 Okt 15:00 UTC (skor ±6 j).
4. **Jalur opsional (akun utama, 18 j GPU, hanya bila Pasha sempat)**: murid ke-2 berarsitektur lain (EffNetV2-S atau DINOv2-S 2.5D) pada label r4 atau label guru bliverrigpdoge, 4 fold paralel 2 sesi (±6 j), gerbang gold ≥0,930 & ρ ke murid ConvNeXt < 0,95, lalu masuk slot compact bersama murid lama. Prior +0,0005–0,001 nyata; risiko jadwal tinggi → jangan mengorbankan pemilihan final.
5. **Jangan diulang**: murid di slot Raptor, B-leg, 24-kanal, bobot V2 tanpa S6F, tuning bobot di gold atau di papan publik.

## 8. Hasil eksekusi
- **Commit-run `alverlfarfaraway/rsna-knee-h5` v1** (push 16:32 WIB, COMPLETE 16:53; 1.056 dtk sampai blend, 3 studi): `MURID_STAGE rc 0` (8 fold: fold0–3, fold6–9; 3/3), Raptor 384 270 dtk, Raptor 224 93 dtk, `ASSETS_READY ... flip_tta True`, `OURS_STAGE_COMPLETE status PASS` (3/3, 0 fallback), `S6F_STAGE ok` (107 dtk), **`S6F_LEG used` (w 0,5, ρ 0,75)**, **`MURID_LEG used` (compact 0,4, 3/3)**, `NOTEBOOK_COMPLETE` (0 kegagalan), Part B: `own leg: blended 3 checkpoints` (galat pip `pylibjpeg-libjpeg` = sama persis dengan log h4, tidak fatal). `submission.csv` 3×13 dalam [0,1]. Kuota GPU alver setelahnya 1,32 j.
- **Submisi 57432552** (20 Okt 16:54 WIB, dari akun alver, izin user 1 jatah): PENDING; perkiraan skor ±22:00–23:00 WIB. Jatah 20 Okt **4/5** → sisa 1 untuk tim; reset 21 Okt 07:00 WIB.
- **Aturan baca hasil** (bersama h4 57427200 & h4m 57428613):
  - h5 ≥ 0,959 → kandidat final A terkuat (unik + S6F + murid). Final B = g6a/g6c.
  - h5 = 0,958 → simpan sebagai kandidat B (beda basis/bobot dari g6a/g6c: r224 polos, V2, R6 s2, murid alver-8). Naiknya rank tim setelah h5 selesai = skor tak-dibulatkan h5 > skor terbaik tim sebelumnya.
  - h5 = 0,957 → V2 + r224 polos lebih lemah di papan; jangan dijadikan final; final tetap keluarga g6a/g6c.
  - h4 (salinan murni) apa pun skornya **bukan** final (lihat §7.1).
