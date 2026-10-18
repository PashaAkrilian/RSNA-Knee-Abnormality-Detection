# RENCANA EKSEKUSI 2×GPU + 1×TPU — REVISI 2 (status: MENUNGGU PERSETUJUAN)

> **ARSIP (status 10 Okt 21:30 WIB)**: dokumen lama; tidak mewakili rencana sekarang. Lihat `STRATEGI_PERAK.md` dan `MEMORY.md`.


> Revisi 25 Agustus 2026 malam, setelah konfirmasi submission history tim:
> 0.936 = fork DINOsaur V4/V10 (best notebook publik), naik bertahap dari
> 0.920→0.922→0.924→0.935→0.936 semuanya fork publik.
>
> DUA KOREKSI FUNDAMENTAL dari revisi ini:
> 1. Self-run notebook apapun di akun kita HANYA melihat public test stub (3 studi).
>    Artefak test skala penuh + prediksi test asli HANYA lahir saat SUBMIT
>    (rerun server-side Kaggle). Jadi "harvest artefak dari self-run" dan
>    "inferensi sendiri ke test penuh" sama-sama TIDAK BEGITU berguna/mungkin.
> 2. Validasi pipeline 0.936 sudah terbukti oleh LB sendiri. Mengulang vanilla run
>    = buang ~9h GPU tanpa informasi baru.
>
> Konsekuensi: GPU dialihkan ke hal yang BENAR-BENAR butuh compute fisik:
> OOF harvest utk fusion tuning + integrasi arm baru ke notebook submission.

---

## Fakta yang menjadi dasar revisi

- Skor tim per 25 Agu: 0.936 (#91/2.372) via `RSNA Knee | DINOsaur V4` V1+V2
  (keduanya 0.936). Sebelumnya: toes-port 0.935, dread-CoAtNet 0.924,
  bend-the-knee 0.922, tonylica 0.920 — semua fork publik.
- 137 tim ≥ 0.936 → 0.936 adalah lantai komoditas. Naik = butuh sesuatu yang
  TIDAK dibagikan publik: (a) label engineering severity-aware, (b) fusion weight
  hasil nested-CV leak-free (bukan public-fit), (c) arm sendiri yang lebih kuat.
- Kuota saat revisi: GPU 29,79h / TPU 12,89h (refresh 2026-08-29 00:00 UTC).

---

## LANGKAH 0 — GRATIS, TANPA KUOTA (kerjakan PERTAMA, sebelum sesi apapun)

### 0a. Inventarisasi artefak OOF yang SUDAH dipublikasikan orang
Unduh & inspeksi lokal (dataset Kaggle, CPU saja):
| Sumber | Artefak yang dicari | Status |
|---|---|---|
| `pilkwang/rsna-knee-weights` | `oof.npz` (20-model DINOv2) | SUDAH PUNYA (0.8360) |
| `antoinegg1/rsna-knee-e9-radimagenet-heads-v15` + e11-diverse-heads-v20 | `v52_oof.csv`, kontrak nested (nested macro 0.862-0.873) | cek isi |
| `mattiaangeli/knee-mri-fold-weights` | OOF A5 fold-family | cek |
| `dreaddevelopment/raptor-knee-maxspan` + widedense | OOF CoAtNet (kemungkinan TIDAK ada — CoAtNet tidak pernah latih di gold-58, jadi inferensi langsung ke gold-58 = leak-free) | cek |
| `lixin73/rsna-knee-train-checkpoints` | OOF arm lixin73 | cek |

### 0b. Fusion nested-CV OFFLINE dari tabel OOF gabungan
Join per-study-id di 57-58 gold → grouped 5-fold + bootstrap 3000 → weight map
per-target antar-arm (pola kontrak antoinegg1 E10). Output: daftar bobot kandidat +
estimasi gain. **Ini bisa memberi +0,002-0,004 tanpa menyentuh kuota GPU/TPU sedikit pun.**

### 0c. Label engineering severity-aware (kernel CPU, internet ON)
Regex multilingual + LLM ensemble dengan heuristik resmi Overview
(moderate/large effusion, borderline=negative, dst.) → soft target [0,1],
monotonic + shrinkage + confidence floor, kalibrasi IN-FOLD vs gold-58.
**Gate: agreement/OOF > flight0234 (0.8991)** — evaluasi numpy manual lokal.

---

## SESI 1 — GPU#1: "Gold-OOF Harvest" (REVISI: 9h → ~2-4h)

- **Kernel**: `dimaspashaakrilian/rsna-knee-oof-harvest-v1` (T4x2, internet off boleh).
- **Isi**: inferensi arm-arm yang TIDAK punya OOF publik ke atas **57-58 studi gold**
  (bukan 4.407 studi!):
  1. CoAtNet MaxSpan v5 SWA (`dreaddevelopment/raptor-knee-maxspan`) — leak-free
     karena corpus-nya tidak pernah melatih gold-58.
  2. CoAtNet WideDense v4 (complement) — idem.
  3. (Kalau 0a menemukan lubang lain, tambah di sini.)
- **Output**: `_oof_coatnet_v5.csv`, `_oof_coatnet_v4.csv` (study_id × 12 kolom rank)
  → disave sbg output kernel → dipakai Langkah 0b.
- **Estimasi**: 58 studi × (62 window × 2 model) ≈ <2h termasuk decode DICOM.
  Buffer sisanya: perluas ke ±300 studi train tambahan utk bootstrap weight yang
  lebih stabil (opsional, stop kapan saja — incremental).
- **Milestone hijau**: log per-studi sukses ≥98%, csv keluar lengkap 57 baris.
- **Merah**: OOM (turunkan batch/window), studi gagal >2%.

## SESI 2 — TPU#1: Training arm sendiri dgn label baru (TIDAK BERUBAH, ≤6,5h)

- Kernel: `dimaspashaakrilian/rsna-knee-modeling-v23-lixinstyle-labelsv2`.
- Arsitektur: DINOv2-small + SlotHead attention ala lixin73 (proj LN768→Lin768→256,
  slot_emb(6,256), query(12,256)) — bukan mean-pooling (mentok di v16/v21/v22).
- Data: cache v5 (336px/crop140mm) + label severity-aware baru; 58 gold tak tersentuh.
- Budget: hard-cap ~6h, EPOCHS=6-8, WAJIB 5/5 fold selesai (pelajaran v22).
- Decision rule PRE-REGISTERED: rata-rata fold 0+1 < 0,72 → STOP line;
  masuk fusion hanya jika OOF ≥ 0,82 DAN std antar-fold < 0,06.

## SESI 3 — GPU#2: Integrasi & validasi submission (REVISI: ringan, ~1-2h)

- **Kernel**: `dimaspashaakrilian/rsna-knee-submit-v11-fusion` — fork DINOsaur V4
  milik tim + modifikasi:
  1. Sisipkan arm baru (checkpoint Sesi 2 di-mount sbg dataset) → inferensi IN-KERNEL
     selama rerun scoring (biaya masuk budget 9h server-side — DINOv2-small
     ≈ +1-2h, aman).
  2. Ganti bobot fusion final dgn weight map hasil Langkah 0b (rank-space tetap).
- **Validasi di akun kita hanya 3-studi stub** (menit, bukan jam): 0 error, format
  submission.csv benar, semua stage selesai. Validasi skala penuh = saat submit
  (server-side, TIDAK memakan kuota GPU kita).
- **Submit = koordinasi nxtirfan** (kuota 5/hari TIM). Simpan 2 profil: fusion-tuned
  + conservative.

---

## Anggaran revisi

| Item | Lama | Revisi | Alasan |
|---|---|---|---|
| GPU#1 | ~9h vanilla run | **~2-4h OOF harvest** | self-run = 3 studi stub, percuma |
| TPU#1 | ≤6,5h | ≤6,5h (tetap) | inti diferensiasi |
| GPU#2 | ~3-6h infer test | **~1-2h integrasi+devtest** | test penuh = urusan server-side |
| TOTAL | ~18h | **~10-12h** | sisa kuota jadi buffer eksperimen |

## Playbook pemantauan (tetap berlaku, contoh per sesi revisi)

```bash
export PATH="$HOME/.local/bin:$PATH"
kaggle kernels status dimaspashaakrilian/<slug>
kaggle kernels output dimaspashaakrilian/<slug> -p dir --file-pattern "log" -q
grep -E "FOLD|OOF|val_macro_auc|ERROR|BATAS|sukses" *.log | tail -30
```

| Sesi | Hijau | Merah |
|---|---|---|
| GPU#1 harvest | ≥98% studi sukses, 57 baris csv lengkap | OOM, >2% gagal, csv kosong |
| TPU#2 train | loss turun konsisten, val_auc naik monoton | loss statis ≥2 fold (bimodal v21), DATA_OK=False (gotcha ambang [0,1]) |
| GPU#3 integrasi | 3-studi stub 0 error, format benar | error mount dataset checkpoint, OOM in-kernel |

Ritme: status tiap ±30-45 menit; log tiap ±1-2 jam; momen kritikal TPU = akhir
fold 0-1 (decision rule) dan akhir fold 4 (kelengkapan 5/5).

## Go/No-Go
1. Langkah 0b menunjukkan gain nested-CV < +0,001 → GPU#1 harvest tetap jalan
   (melengkapi tabel utk masa depan) tapi prioritas beralih ke Sesi 2.
2. Label baru gagal gate 0,8991 → batalkan Sesi 2, Sesi 3 tetap (fusion-weight saja).
3. Sesi 2 OOF < 0,82 / bimodal → arm tak masuk fusion; Sesi 3 pakai weight map 0b saja.
4. Dua kegagalan berturut di sesi manapun → audit manual sebelum retry (jangan boros kuota).

## Catatan status terkini — 19 Okt 2026 02:52 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; medali ditentukan split privat 70%). Pada cek API 02:52, `g6c`, `g12`, `g13` masih PENDING; dua kernel training murid r4 di alver sudah dua kali di-requeue dan tetap QUEUED setelah push kedua pukul 02:51. Timer memantau keduanya. Kuota main/alver/nxtirfan tetap 19,87/17,81/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
