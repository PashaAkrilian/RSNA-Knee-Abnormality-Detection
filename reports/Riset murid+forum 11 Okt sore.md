# Riset mendalam 11 Okt sore — murid pseudo-label + forum 0,960
> Sumber: 2 subagen (16+ kernel kode murid; 180 thread forum + LB 11 Okt 17:05 WIB). Sintesis untuk Fase B.

## A. Papan (11 Okt 17:05, 5.734 tim)
- Kita **rank 289 @ 0,954**; dataran 0,954 = **244 tim** (pagi 49 → sore 244, klon S6 menyebar).
- Garis perak top 5 % = **rank 286** — jatuh DI DALAM dataran. **Perak aman = ≥0,955 (rank 95); ideal 0,957+ (rank 61).**
- ≥0,960 = 33 tim (rank 25). hengck23 (0,949) bilang tim 0,960 pakai "one-shot learning" — tanpa detail. **0,960 tidak dijanjikan**; realistis dari jalur murid = **0,956–0,959**.

## B. Metode >0,954 yang kredibel
1. **Pseudo-label student (ConvNeXt-T/DINOv2 2.5D)** — azamat1ch **0,959** (ConvNeXt-T 5-fold), colum2131 **0,959** (ConvNeXt-T 0,953 + DINOv2 0,951 → ens 0,955, lalu naik), Chris Deotte: "pseudo-label membuat single model = ensemble CV". Latih pada **probabilitas lembut**.
2. **Resep publik terlengkap** (nartaa/Danial 0,949): 94 window, 12 train/window, 16 ep, lr 4e-5/2e-3, wd 0.059, SWA top-3 epoch, **anatomical mirror TTA** (sagittal balik urutan channel), OAI +0,008.
3. **Validasi**: OOF-vs-label-lemah **r=0,87** dgn LB; emas-58 **r=0,66** (rusak). OOF-siswa vs label yang kamu perbaiki = boomerang (Tucker).
4. **Anti-korelasi**: anggota yang tidak setuju > yang setuju (Dread); bobot per-finding 0,03–0,35; flat 0,30 = rugi (prvsiyan 0,948 < solo 0,949).

## C. Template murid: goodpjw ConvNeXt-T 2.5D (kode publik lengkap)
- **Cache**: uint8 per series, canon LPS per plane, **urut by IPP**, resample 0,4 mm/px, crop FOV, percentile [1, 99.5] per series, max 64 sl (kita: 256 px / FOV 128 mm / max 28 sl utk disk).
- **Label**: mean 4 tabel publik (steven_v4 + pilkwang + lixin gpt56sol + laymond qwen) clip [0.01, 0.99]; **kita ⊕ teacher r384 50/50** (gaya nvap V7B); fold = GroupKFold(5) per teks laporan (anti bocor kembar), emas-58 val-only fold=-1.
- **Model**: ConvNeXt-T (timm) → token per jendela 3-slice + slot planar/fat-sup + posisi → TransformerEncoder 2 layer → attention-pool per finding.
- **Train**: k=8 (eval 12), res 256, crop fov 0.92, rot ±12°, AdamW 1e-4/5e-4, wd 0.02, warmup 1 ep cosine, clip 3.0, EMA 0.998, **fp16+GradScaler di T4 (bf16 = −0.14 diam)**, 14 ep, ckpt = EMA, resume atomic per ep.
- **Inference**: mean fold → **percentile-rank per finding** → rank-blend dgn S6, bobot per-finding 0,03–0,35 (JANGAN tune di publik split).
- **Masked-BCE** (keremelik): +0,0022 macro-AUC dgn konsensus bulat 3 sumber — opsional, uji paired-CV.

## D. Pitfall wajib
1. fp16 bukan bf16 di T4; cetak backbone yg benar2 dimuat.
2. Jangan regenerasi fold setelah ganti label (7 studi pindah fold = bocor).
3. Fracture ~97 % konstan → bedakan "tak disebut" vs "negatif eksplisit"; not-mentioned = 0.28 (FHZ982), bukan 0.
4. Synovitis senyap: proxy Effusion saja (0.28+0.45·(eff−0.28)); jangan imputasi kolom lain.
5. Jangan generic flip TTA — hanya anatomical mirror.
6. Res 320+mirror < res 384+crop (Danial); jangan numpuk 18 sl jadi channel.
7. Tuning bobot per-label di publik split = overfit (starkhushi 742050); flat weight utk final pick.
8. cUDNN crash ConvNeXt kadang perlu `cudnn.enabled=False` (probe).

## E. Jalur eksekusi (AKTIF, akun alver)
1. ~~teacher r384 train preds~~ ✅ `data/teacher_r384_train_preds.csv` (4.407, 0 fallback).
2. **student-prep** RUNNING: cache256 + labels konsensus4⊕teacher 0,5 + fold transplant → dataset `alverlfarfaraway/rsna-knee-student-prep`.
3. **student-train-a/b/c** siap: ConvNeXt-T 5-fold (0-1 / 2-3 / 4), fp16, EMA, 14 ep, deadline 7,2 h/fold-group.
4. Berikutnya: **student-infer** — murid solo + rank-blend S6 (bobot flat per-finding); validator OOF-vs-konsensus; submit dari alver (kuota 4 sisa).
5. Cadangan murah: anatomical mirror TTA utk Raptor/compact (+0,001 terukur, Danial); SWA murid.

## F. Sumber kode lokal
- `/tmp/opencode/arms_test/goodpjw_reader/` (train.py, knee.py, make_labels.py, preprocess.py, oof.py, infer.py).
- `/tmp/opencode/arms_study/` (ipynb murid lain: nvap, forwardidear, sadamtorres, keremelik).
- LB snapshot: forum subagen (11 Okt 10:05 UTC).

## Catatan status terkini — 19 Okt 2026 02:45 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; medali ditentukan split privat 70%). Pada cek API 02:44, `g6c`, `g12`, `g13` masih PENDING; dua kernel training murid r4 di alver, di-requeue v1 pukul 02:42, masih QUEUED. Timer memantau keduanya. Kuota main/alver/nxtirfan tetap 19,87/17,81/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
