# Ringkasan riset mendalam — 10 Okt 2026 (sesi Claude Code, hanya akun utama)

> Tujuan: perak (≥0,951; aman ≥0,952). Dokumen ini merangkum SEMUA yang ditemukan/dicoba pada 10 Okt, termasuk bedah kode notebook
> ber-skor terbaik. Fakta lain: `MEMORY.md`, `HANDOFF.md` §0c, `laporan_push.md`, `reports/Cara menembus 0951 perak.md` (OpenCode).

## 1. Posisi dan papan
- Papan 10 Okt 06:57 WIB (5.625 tim): kita **0,950, rank 414** (482 tim sama-sama 0,950). ≥0,951 = 309 (tadi malam 296, +13 dalam 5,5 jam),
  ≥0,952 = 197, ≥0,953 = 145, ≥0,960 = 28. Garis perak = 5 % ≈ rank 281 → **0,951 lotre, 0,952 target aman**. Emas ≥ ~0,960 (di luar jangkauan).
- Skor tim kita seluruhnya dari notebook publik Apex (nartaa r384 OAI-bobot + reader goodpjw); model bersih tanpa OAI mentok 0,944.

## 2. Peta percobaan (hasil bertanggal)
| Jalur | Hasil | Status |
|---|---|---|
| probe r224 (57012143), fork rabari dual-engine (57015077) | 0,950 / 0,950 | tutup (plafon keluarga r384+reader+r224) |
| **Apex + MaVIT hengck23 @0,20 (57025414)** | **0,949** | tutup. MaVIT solo LB 0,935, emas-58 0,9202; model beda arsitektur dgn label laporan yg sama = galat berkorelasi → tak menambah |
| lixin73 MaVIT-336 | tak bisa dipakai | `slot_emb` 3 view vs 6 di kode hengck23, tanpa kode publik |
| mattiaangeli CoAtNet resgated top-3 | emas-58 ±0,909 | keluarga CoAtNet sama, lebih lemah |
| MaVIT/anak-OAI/soup/dll (9 Okt) | ≤ induk | tutup |
| **Apex 0,70 + OrthoFoundation-L fold0 0,30 (57028008)** | **Notebook Timeout (9 jam)** | GAGAL teknis; perbaikan & uji skala berjalan (§5) |

## 3. Notebook publik ber-skor tertinggi (skor dibaca dari halaman Code, 10 Okt)
| Notebook | Skor | Bedah |
|---|---|---|
| heliosli S6 `rsna-knee-abnormality-detection` | 0,954 | Raptor(nartaa) 0,7 + 4 model compact ConvNeXt-S 0,3, bobot TETAP (tak di-fit). **MATI**: bundel `compact-v1` ditarik 9 Okt 13:50 UTC |
| versia-7 (347 vote), blend-gold, blend3 | 0,951 | klon rakitan heliosli → mati bersama bundelnya |
| `kiyoshiohno/rsna-knee-template` | 0,951 | HANYA peluncur: menjalankan `rsna_knee_pipeline.py` dari dataset PRIVAT (sumber publik: goodpjw, nartaa, OF fold0) → tak bisa direplikasi |
| DINOsaur V5, Velciraptor V2, Apex dan sisanya | 0,950 | V5 keluarga Apex; V2 = klon heliosli (bundel ditarik) |
Kesimpulan: tidak ada jalan pintas publik ≥0,951 yang masih bisa dijalankan.

## 4. BEDAH KODE ber-skor terbaik (heliosli) — temuan baru
Notebook V2 (klon S5) menanamkan 7 berkas sumber heliosli (`SOURCES` dict, dapat diekstrak): `compact_train.py` (45 KB), `compact_infer.py`,
`probe_infer.py`, `probe_dual.py`, `px_cache.py`, `knee_anchor_bank_v2.py`, `side_common.py`. Pelajaran:
1. **Model compact** = jendela 2.5D (3 irisan bertetangga sbg kanal) → encoder CNN (resnet34 default; `convnext_small` utk arm `E_lrprior`) →
   perhatian per kelas atas jendela studi → 12 logit. Latih K-fold di kumpulan non-emas; **tidak pernah memakai label emas/resmi** (emas = validasi bersih).
2. **Target = "report-state construct"**: keadaan laporan (mis. derajat robekan) dipetakan ke probabilitas via vektor prior (`state_spec`),
   loss gabungan biner + state (`--state-weight 0,5`); arm: S_strict, P_presence, E_prior, E_lrprior. Data target & spec ada di bundel yg ditarik (hilang).
3. **Kompilator laporan berbasis aturan** (`cloud_code/training/criteria.py`, multibahasa, deterministik, konservatif) dapat dijalankan lokal (36 dtk utk 4.407 laporan).
   **Ukur: AUC rata-rata vs emas-58 hanya 0,784** (liputan "unknown" tinggi; OA/Synovitis/Contusion lemah) → sbg label mentah jauh di bawah label fusi kita (0,9348)
   dan LLM publik (~0,89–0,90). Jadi kekuatan heliosli = kombinasi prior + model, bukan aturan saja; BUKAN jalan pintas label.
4. **Rekayasa runtime heliosli** (`probe_dual.py`): studi diurut menurut jumlah berkas DICOM dan diintersep antar-GPU (kerja seimbang);
   pekerja menulis keluaran bertahap + `--deadline`; pengendali TIDAK gagal bila satu studi/pekerja mati (studi lain dapat rerata/rank cadangan);
   env `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2` (cegah oversubscription CPU); resi per-studi (decode/render/gpu).
5. Blend akhir: Raptor 0,7 + compact 0,3, **bobot ditetapkan sebelum run dan tidak di-fit** (tak ada set berlabel bersih utk fit).

## 5. OrthoFoundation-L fold0 (dataset publik heliosli `rsna-orthofoundation-fold0-epoch7-20261004`)
- Isi ckpt: ViT-L DINOv3 (OrthoFoundation-L; prapelatihan ≈1,2 jt citra lutut termasuk OAI & fastMRI + data privat) 384 px, `anatomy_pool` (butuh 3 jaringan anatomi:
  SKM-TEA, YOLO localizer KneeXNet, UNet resnet34), epoch 7/16 EMA, kebijakan label `own_reports_and_official_gold_only_v1`. **48 dari 58 emas ikut latih** →
  emas-58 = 0,9535 TERKONTAMINASI (bukan bukti kualitas). `external_mcl=false`.
- Probe berhasil di Kaggle T4 setelah 3 perbaikan (cloud_code ditulis ke /kaggle/working; anatomy.pt+kneexnet disalin, bukan symlink; yaml ultralytics 8.4.166
  hilang dari paket vendored → dataset privat `rsna-knee-ultralytics-cfg` dari wheel PyPI). Kecepatan 27 dtk/studi (1 GPU; ±70–75 % tahap anatomi, ±3 dtk dekode, sisanya ViT).
- **Gagal skala (57028008)**: perkiraan 5–6 jam meleset → timeout 9 jam. Dugaan: (a) tiap potongan memetakan seluruh `test_series`; (b) oversubscription CPU;
  (c) tanpa pembunuhan proses pada batas waktu. Uji skala A (2 proses) dan B (4 proses) berjalan untuk mengukur throughput nyata.

## 6. Pelajaran dari forum (thread 735304 dibaca penuh) dan teori
- Pemisah ≥0,951 = **label**, bukan arsitektur. Resep terbuka: label laporan multi-sumber (publik, API, lokal) → model guru K-fold → OOF → *soft-bootstrapping 50/50*
  dgn label asli (mengganti total dgn prediksi lebih buruk daripada campuran) → latih ulang. Itu yang sudah kita lakukan (label fusi r1: selaras emas 0,9348; solo CoAtNet 0,927→0,933).
  Model tunggal 0,950+ di forum (CoAtNet 224 px, ResNet50 224, ConvNeXt-T) menandakan label mereka JAUH lebih baik, sumbernya tak dipublikasikan.
- Teori ensembel (binormal, galat berkorelasi ρ): dua model AUC 0,950 & 0,935 → campuran optimum 0,957 (ρ=0,7), 0,952 (ρ=0,8), ≤0,950 (ρ≥0,9). Model dgn label laporan sama
  punya ρ≈0,9 → hasil ±0 (terbukti: reader +0,001, MaVIT −0,001). Hanya model dgn PENGETAHUAN label berbeda (OAI nartaa, OrthoFoundation) layak.

## 7. Kepatuhan & risiko
- Akun kedua/ketiga TIDAK dipakai sesi ini (Rules 3.5.a/3.5.b; izin tertulis hanya satu kalimat). GPU utama 30 jam cukup. Draf kepastian ke Kaggle: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`.
- OAI/data eksternal: ckpt nartaa & OrthoFoundation membawa pengetahuan eksternal; host belum menjawab thread 743416 → 2 final harus beda mekanisme (satu cadangan bersih 0,944).

## 8. Peluang jujur & rencana
- Perak ±25–35 % tanpa perbaikan nyata; OF yang berhasil menambah +0,001–0,003 → ±45–55 %; merger dgn tim ≥0,955 → ≥80 % (126 kandidat muat batas; `docs/KANDIDAT_MERGER.md`). Emas ≈0–2 %.
- Rencana teknis: (1) selesaikan uji skala OF → konfigurasi terbukti (akar per-potongan, batas thread, pengurutan biaya, pembunuhan proses, keluaran bertahap, fallback rank Apex);
  (2) submit ulang Apex+OF dengan batas total ≤7,5 jam; (3) bila ≥0,951 → kandidat final; bila ≤0,950 → tutup OF dan fokus finals + merger.
- Tugas pengguna: merger sebelum 15 Okt 23:59 UTC; pesan ke Kaggle support; pastikan `nxtirfan` setuju soal token.

## 9. Pembaruan malam 10 Okt (19:30–21:30 WIB)
- **Papan 19:37 WIB (5.666 tim)**: ≥0,951 = 348, ≥0,952 = 217, ≥0,953 = 162, ≥0,954 = 132, ≥0,960 = 30; =0,950 = 564 (kita rank 530). Garis perak ≈ rank 283 → kandidat final publik ≥0,952–0,954.
- **Scan publik** (`scan_public_updates.py`): klaim "0,954" (rabari9999, emresariduman, sameerk d5h) = kode Apex (dataset goodpjw+nartaa; skor 0,950); `jenillangaliya/dual-stage-sota-ensemble` 0,951 memakai dataset privat (bundel heliosli yg ditarik). Dataset baru (`shishyan/finetuned-weights`, `sergiomc12/unifiis-modelo`, `kunalinfinite/qwen3-8b-...`) tanpa deskripsi/klaim skor; tak dievaluasi (EV rendah). **Tak ada jalan pintas publik ≥0,951.**
- **Rules resmi** (`kaggle competitions pages rsna-knee-abnormality-detection list --page-name rules --content`): 5.a akun tunggal (diskualifikasi bila >1 akun), 5.b anggota tim = individu terpisah, 5.c merger (gabungan ≤5; submisi gabungan ≤ jatah satu tim saat permintaan; sebelum batas), 5.d/6.a dilarang berbagi kode/data privat antar-tim (publik boleh).
- **Simulasi peluang perak** (`silver_prob_sim.py`; `docs/RESEARCH_NOTES.md` §40): Apex 14–20 %; +0,002 → 26–39 %; +0,004 → 40–67 %; tim publik 0,956–0,958 → 23–82 %. 100 % mustahil.
- **Uji skala OF** (v1): disk 330 MB/studi + CPU hambatan + 2 proses lambat → penyebab timeout; v2 (`of_stage.py`) diuji (`ofscale3/4`); `apexof2` lulus stub.
- **Rencana lengkap & gerbang keputusan: `STRATEGI_PERAK.md`.**

## 10. Pembaruan 11 Okt dini hari (02:30–04:00 WIB) — S6 hidup lagi; koreksi premis OF; riset "best score"
- **Peristiwa**: heliosli memulihkan `heliosli/rsna-knee-compact-v1` (02:50 WIB; V1–V13 publik) dan menerbitkan ulang notebook S6 (**Raptor 0,7 + compact 0,3 = 0,954 satu klik**; thread 748141 "blend my model with yours"). User men-submit salinan (57051680, 02:56 WIB, PENDING). Papan 02:59: 5.699 tim; kita rank 599 (0,950) — di bawah garis perunggu publik (rank 570); perak rank 285 (0,951); ≥0,954 = 141; ≥0,960 = 31.
- **KOREKSI (bagian 5 dan 8 di atas keliru)**: resep heliosli = Raptor + **compact ConvNeXt-Small** (bundel S5 `c7ebab1e` = 0,951 di blend-gold; S6 `209a32c8` = 0,954); bobot OrthoFoundation fold0 TIDAK dipakai di notebook heliosli mana pun (hanya kode `cloud_code`/`anatomy.pt` dari dataset OF). Nilai LB OF belum pernah diukur; prior ≥+0,001 di atas S6 ≤20 %.
- **Penyebab 57044355 = 0,950**: kaki OF mati OOM di run penilaian (induk Apex memegang 18 GB RSS + pekerja OF 8–10 GB, mesin 31 GB/4 vCPU) — terkonfirmasi lewat diagnostik `apexof4d` (urutan lama, 1.322 studi pseudo-uji) dan profil memori (`ofmem1/3`); perbaikan v4.1 (OF dulu/terakhir, supervisor klaim/restart/daur-ulang, `pin_memory=False`, `torch.load(mmap=True)`) lulus rehearsal (`apexof4r2`: 194 studi/jam dgn 3 pekerja, MemAvail min 19 GB). Rinci: `PROGRESS.md` §54, `docs/PLAN_B.md` §0.
- **S6 teknis**: tahap compact = `probe_dual.py` 2 pekerja (1/GPU) memakai decode `native_cache` yg sama dgn OF; GPU-bound ±18–20 dtk/studi/pekerja (decode ±8, render ±3, GPU ±18 per studi; ±600 jendela × 4 fold) → 1.322 studi ±3,5 j; Raptor ±0,5 j; total ±4–4,5 j; kontroler menolak bila >3 % fallback (ERROR tanpa skor).
- **Kebun-binatang bundel heliosli (publik)**: V4 `ofl_ft8` (OrthoFoundation-L fine-tune), V7 cnx_s5, V8 `dino_s1` (DINOv3-ViT-B), V9 dual (convnext+dino), V10=V13 cnx_s6. Bagging berbiaya ±3,5–4,5 j GPU/bundel.
- **Forum "best score"** (735304 dan terkait): model tunggal 5-fold ConvNeXt-T 0,953–0,959 dgn pseudo-label (azamat1ch/MURAD rank 10; colum2131); "model is not the key. it is your labels" (hengck23); Chris Deotte: pseudo-label meratakan model tunggal ke level ensembel; validasi: AUC OOF label-LLM berkorelasi r=0,87 dgn LB publik (emas-58 r=0,66); trik "kirim 1−p" (tim kuat tersembunyi); kepatuhan (744056/747455/743416) tanpa jawaban host; Rules 2.6.b mengizinkan model/data eksternal.
- **Notebook publik baru**: "Tri-Reader 0,954" (rabari/jcgweb/sameerk) = Apex+r224 → jangan; tri-hybrid jenillangaliya 0,952 (< S6).
- **Eksekusi**: `build_s6plus.py` (S6 + reader goodpjw pada perna Raptor + OF terakhir dgn deadline + blend 3 perna + monitor RAM + mode lenient), sonda `s6probe`, rehearsal `s6plusr`; submit varian setelah reset jatah 07:00 WIB.
- **Rencana**: `STRATEGI_PERAK.md` §0 (strategi berpusat S6; merger tetap tuas terbesar; final A = S6/varian terbaik, B = beda mekanisme/bersih).

## Catatan status terkini — 19 Okt 2026 02:56 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; cutoff ini sementara karena medali memakai split privat 70%). Pada cek API 02:55, `g6c`, `g12`, `g13` masih PENDING. Dua kernel murid r4 di alver sudah dua kali di-requeue dan kini RUNNING pada cek 02:56. Kuota main/alver/nxtirfan: 19,87/17,75/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
