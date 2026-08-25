# RENCANA EKSEKUSI: 2×SESI GPU + 1×SESI TPU (status: MENUNGGU PERSETUJUAN — belum dieksekusi)

> Disusun 25 Agustus 2026. Konteks: tim 0.936 (#91/2.372), top-1 Brandon Low 0.952,
> strategi induk di RESEARCH_NOTES §26f. Kuota saat penyusunan: GPU 29,79h / TPU 12,89h
> (refresh 2026-08-29 00:00 UTC). Pelajaran penting: v22 gugur di 4/5 fold karena
> safety time-limit → SEMUA run di bawah dirancang agar SELESAI penuh dalam budget.

---

## Peta 3 sesi (urutan + dependensi)

```
Sesi 1 (GPU#1, ~8,5–9h)  ──artefak rank──►  offline fusion nested-CV (TANPA kuota)
                                                    │
Label engineering (kernel CPU, TANPA kuota) ──label──► Sesi 2 (TPU#1, ≤6,5h) train arm baru
                                                    │ OOF ≥ 0,82?
                                                    ▼
                              Sesi 3 (GPU#2, ~3–6h) inferensi arm + fusion final
```

Sesi 1 dan jalur label-engineering saling INDEPENDEN (bisa jalan paralel).
Sesi 2 menunggu label lolos gate. Sesi 3 menunggu Sesi 2 lolos gate OOF.

---

## SESI 1 — GPU#1: Basis stack 0.936 milik kita sendiri

- **Kernel**: fork `prvsiyan/head-and-shoulders-knees-and-toes` → slug kita
  `dimaspashaakrilian/rsna-knee-basis-v10` (judul JANGAN diubah setelah push —
  gotcha slug self-reference). Machine shape WAJIB `"NvidiaTeslaT4"`.
- **Tujuan**:
  1. Validasi end-to-end mount standar jalan sampai submission.csv (basis yang KITA kontrol).
  2. **Panen semua artefak rank intermediate** (`submission_transformer_0920.csv`,
     `_raptor.csv`, csv E10/E13/pass-2, dsb.) → selanjutnya eksperimen fusion/bobot
     dilakukan OFFLINE (numpy lokal/kernel CPU) TANPA makan kuota lagi.
- **Estimasi**: stack dirancang budget 8–8,72h dari cap 9h → sisakan margin; JANGAN
  tambah stage apapun di run ini (run vanilla dulu, modif belakangan).
- **Milestone pemantauan** (urutan log):
  1. Queue → RUNNING (antrian T4 bisa 10–40 menit)
  2. Stage A: member manifest berjalan satu-satu (log per-member; boleh lambat di awal)
  3. File `submission_public_0899.csv` muncul = Stage A frontier selesai
  4. A5 blend @0.45 selesai → RadImageNet chain (log elapsed-minutes) → CoAtNet stage
  5. `submission.csv` final ada, ukuran wajar (~baris = studi test × 12)
- **Red flags**:
  - System-RAM OOM di CoAtNet/DataParallel (known issue; versi port sudah sequential,
    kalau tetap OOM → laporkan, jangan retry buta)
  - Log "fallback"/`fillna(0.5)` luas = banyak studi gagal decode → STOP, investigasi
  - Waktu lewat 8,9h tanpa submission.csv = kena cap 9h → run hangus (quota terpakai)

## ANTARA SESI — kerja TANPA kuota compute

1. **Label engineering severity-aware** (kernel CPU, internet ON — internet hanya
   dilarang saat submission):
   - Regex multilingual (lexicon sudah ada di stack publik) + LLM ensemble dengan
     heuristik RESMI annotator dari Overview (moderate/large effusion, high-grade MCL
     tear, borderline=negative, dst.) → soft target kontinu [0,1].
   - Monotonicity + shrinkage ke prior + confidence floor; kalibrasi vs gold-58 IN-FOLD.
   - **Gate GO/NO-GO: macro AUC label-baru vs gold-58 harus > flight0234 (0,8991).**
     Evaluasi pakai numpy rank-based manual (sandbox lokal sudah terbukti bisa).
2. **Fusion nested-CV offline** dari artefak Sesi 1: grouped 5-fold di gold-58 +
   bootstrap 3000 draw → weight map per-target (pola kontrak antoinegg1 E10).
   Hasil: daftar bobot kandidat untuk Sesi 3.
3. Siapkan kode arm baru (arsitektur lixin73-style) + `kernel-metadata.json` TPU.

## SESI 2 — TPU#1: Training arm sendiri dengan label baru

- **Kernel**: `dimaspashaakrilian/rsna-knee-modeling-v23-lixinstyle-labelsv2`.
- **Arsitektur**: DINOv2-SMALL + SlotHead attention ala `lixin73/rsna-knee-train-checkpoints`
  (proj LN768→Linear768→256, slot_emb(6,256), query(12,256), out Linear256→12) —
  BUKAN SlotHead mean-pooling kita (terbukti mentok di v16/v21/v22), bukan AnatomyQueryHead.
- **Data**: cache preprocessing v5 (336px/crop140mm) + label severity-aware baru;
  58 gold TIDAK PERNAH ditimpa (aturan tetap).
- **Budget safety (pelajaran v22)**:
  - Hard-cap total runtime ~6h: EPOCHS=6 (bukan 8) ATAU potong batch-window;
    scheduler harus habis dalam 5/5 fold — **fold 5 WAJIB tuntas** (OOF N=57 invalid
    kalau kurang).
  - Checkpoint guard: `best_auc` awal = -1.0 (training dari nol) + simpan per-fold.
  - Log per-epoch: train_loss + val_macro_auc per fold (format sama dgn v18-v22 supaya
    greppable).
- **Decision rule PRE-REGISTERED (disiplin anti-bias, pola Raihan)**:
  - Setelah fold 0 & 1 selesai: jika rata-rata keduanya < 0,72 → HENTIKAN line
    (jangan biarkan run lanjut — hemat ±4h TPU; fold awal rendah = pola v21 bimodal).
  - Syarat masuk fusion: OOF gabungan ≥ 0,82 DAN tidak bimodal (std antar-fold < 0,06).
  - Bandingkan head-to-head vs OOF v18 (0,8068) pada N gold yang sama.
- **Red flags**: train_loss nyaris statis di ≥2 fold (pola v21); val_auc bolak-balik
  acak; `DATA_OK=False` false-positive (cek dulu ambang skala [0,1] — gotcha v21 push-1).

## SESI 3 — GPU#2: Inferensi arm baru + fusion final

- **Kernel**: `dimaspashaakrilian/rsna-knee-infer-ownarm-v1` (T4x2).
- **Isi**:
  1. Load checkpoint fold terbaik/arm Sesi 2 → inferensi ke `test_series` penuh
     (~1.300 studi) → `_ownarm.csv` rank-space. Pakai `RSNA_ORDER_CACHE` dari Sesi 1
     kalau formatnya cocok (hemat 1–1,5h decoding-ordering).
  2. Re-blend offline-in-kernel: artefak Sesi 1 + `_ownarm.csv` dengan weight map
     hasil nested-CV → `submission.csv` kandidat.
  3. Sisa waktu: hasilkan profil kedua (conservative diverse) sbg cadangan shakeup.
- **Estimasi**: inferensi 1 arm DINOv2-small ~1,5–3h + blend menit → total 2–4h;
  aman di bawah 9h.
- **Red flags**: studi sukses <98% → jangan fusion, cek dulu studi yang gagal.

---

## PLAYBOOK PEMANTAUAN (dipakai kapan pun, TANPA mengeksekusi apa pun sekarang)

Setup awal tiap shell baru:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 1. Cek status kernel (ringan, boleh sering)
```bash
kaggle kernels status dimaspashaakrilian/<slug>
# COMPLETE | RUNNING | QUEUED | ERROR | CANCELACKNOWLEDGED
```

### 2. Tarik log incremental (berat sedang, cukup 1x per jam)
```bash
mkdir -p /tmp/opencode/mon && cd /tmp/opencode/mon
kaggle kernels output dimaspashaakrilian/<slug> -p . --file-pattern "log" -q
grep -E "FOLD|OOF|val_macro_auc|ERROR|Traceback|selesai|BATAS" *.log | tail -30
```

### 3. Pola yang dicari per jenis sesi

| Sesi | Hijau (lanjutkan) | Merah (intervensi) |
|---|---|---|
| GPU basis (S1) | stage maju bertahap, file csv frontier muncul | RAM OOM, fillna(0.5) luas, >8,9h tanpa submission.csv |
| TPU train (S2) | train_loss turun konsisten tiap fold, val_auc naik monoton | loss statis ≥2 fold (bimodal v21), val_auc acak, DATA_OK=False |
| GPU infer (S3) | studi sukses ≥98%, _ownarm.csv lengkap | studi gagal >2%, OOM, waktu >8h |

### 4. Ritme polling yang disarankan
- Antrian TPU/GPU historis: 10–40 menit dari push → RUNNING. Jangan panik selama itu.
- Saat RUNNING: cek `status` tiap ±30–45 menit; tarik `log` tiap ±1–2 jam cukup
  (log besar, jangan spam).
- Sesi TPU train: momen kritikal HANYA di akhir fold 0–1 (decision rule <0,72) dan
  akhir fold 4 (kelengkapan 5/5).

### 5. Setelah COMPLETE
```bash
kaggle kernels output dimaspashaakrilian/<slug> -p ./out_<nama>
```
- S1 → verifikasi `submission.csv` + daftar artefak rank (JANGAN download semua file
  kalau ribuan — pakai `--file-pattern`).
- S2 → ambil `model_outputs/*.pt`, oof npz/log; hitung ulang OOF independen.
- S3 → `submission.csv` siap diajukan (KOORDINASI dulu dgn nxtirfan — kuota 5/hari TIM).

---

## Anggaran kuota & jadwal

| Item | Estimasi | Kuota tersisa saat ini | Status |
|---|---|---|---|
| Sesi 1 GPU basis | ~8,5–9h | GPU 29,79h | muat, sisa ~21h |
| Sesi 2 TPU train | ≤6,5h | TPU 12,89h | muat, sisa ~6h |
| Sesi 3 GPU infer+fusion | ~2–4h (max 6h) | sisa GPU ~21h | muat |

- Total worst-case: GPU ~15h (dr 29,79h), TPU 6,5h (dr 12,89h) — masih ada buffer
  bahkan sebelum refresh 29 Agustus.
- Kalau Sesi 2 gagal & perlu re-run setelah refresh kuota → jadwal tetap aman
  (deadline entry merger 15 Okt, final 22 Okt).

## Go/No-Go ringkas
1. Label baru TIDAK mengalahkan 0,8991 → batalkan Sesi 2 (hemat TPU), langsung
   Sesi 3 hanya utk fusion weight nested-CV dari artefak Sesi 1.
2. Sesi 2 OOF < 0,82 atau bimodal → arm tidak masuk fusion; Sesi 3 tetap jalan
   (fusion weight saja) — skor naik lebih kecil tapi gratis risiko.
3. Sesi 1 ERROR → perbaiki mount lalu ulang SATU kali; dua kegagalan berturut =
   audit manual sebelum boros kuota lagi.
