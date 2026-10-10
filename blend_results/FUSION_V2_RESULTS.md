# Hasil Fusion v2 (25 Agustus 2026 malam)

> **ARSIP (status 10 Okt 21:30 WIB)**: dokumen lama; tidak mewakili rencana sekarang. Lihat `STRATEGI_PERAK.md` dan `MEMORY.md`.


Tabel OOF gabungan gold-57 (inner-join semua arm), rank-space, evaluasi grouped-CV
kolom `fold` milik e9. Skrip: `fusion_nested_cv_v2.py`.

## Sanity per-arm

| Arm | Macro AUC gold | Sumber |
|---|---|---|
| **CoAtNet v5_full_swa (c5)** | **0.9206** | harvest kita (`rsna-knee-oof-harvest-coatnet-gold58`, fail=0, ~6,5 menit) |
| CoAtNet v4_full_swa | 0.9117 | idem |
| CoAtNet v4_full | 0.9106 | idem |
| e9 RadImageNet heads | 0.8506 | antoinegg1 v52_oof.csv |
| pilkwang 20-model | 0.8360 | oof.npz resmi |
| e11 diverse heads | 0.8285 | antoinegg1 |
| v18 SlotHead (kita) | 0.8068 | oof_predictions_v18.csv |

## Temuan utama

1. **c5 sendirian (0.9206) mengalahkan SEMUA blend**: uniform-7 = 0.9055,
   uniform-6 tanpa V18 = 0.9096, uniform-CoAtNet-3 = 0.9183. Nested-CV memilih
   `c5_only` di 5/5 outer-fold. Bootstrap vs e9: P(menang)=1.000,
   dCI [+0.040,+0.100].
2. **Blend besar justru MENGEcilkan**: anggota lemah menyeret arm terkuat.
   Selaras klaim author dread ("ensembling worth ~+0.001 LB") dan konsisten
   dgn thesis commodity-ceiling: skor 0.936 DINOsaur kemungkinan besar berasal
   dr public-LB-fitted fusion+calibrator, BUKAN validasi objektif → risiko
   private shakeup tinggi bagi semua tim fork.
3. **Seleksi bobot per-target TIDAK reliable di N=57**: versi naive grid halus
   = 0.8095 (overfit); versi "surgical" yg didesign setelah melihat full-gold =
   0.9278 tapi BOCOR desainnya; versi nested HONEST (pilihan inner-fold saja) =
   0.9148 < 0.9206. Kesimpulan: di ukuran sampel ini, tuning halus tidak bisa
   dibedakan dari noise — satu-satunya keputusan valid = pilih model terkuat.

## Konsekuensi strategis

- Baseline lokal terbaik leak-free sekarang: **c5 = 0.9206** (kembar LB-nya 0.924,
  offset +0.003 — konsisten).
- Untuk mengalahkan 0.952, jalannya BUKAN menambah/menimbang anggota ensemble,
  melainkan **melatih SATU arm yang lebih kuat dari c5**: resep terbukti =
  fine-tune PENUH backbone kuat (CoAtNet/timm 384px/crop140mm/span lebar) +
  LABEL LEBIH BAIK (severity-aware, gate >0.8991 flight0234). Target: arm sendiri
  ≥0.93 gold-OOF → baru blend dengan c5 bermakna.
- Catatan operasional: kode training CoAtNet TIDAK dirilis penulis (yang publik
  hanya inferensi) — membangun pipeline training = investasi rekayasa nyata;
  alternatif: adaptasi kernel modeling kita (sudah ada skema unfreeze+TPU) ke
  arsitektur timm CoAtNet.
