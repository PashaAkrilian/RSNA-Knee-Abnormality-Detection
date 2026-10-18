# Riset 14 Okt 2026 malam — jalan ke perak (sesi Claude Code)

> Permintaan user: "cek dan riset dengan mendalam, saya mau bisa menjadi silver sebelum kompetisinya selesai".
> Sumber: papan publik 14 Okt 15:00 UTC, `kaggle kernels list` + tarik 8 notebook baru, forum (749145, 749250, 745861, 748295, 748181, 743416, 747658, 748111).

## 1. Papan & garis perak
| Waktu (UTC) | Tim | ≥0,957 | tepat 0,956 | Perak (5 %) | Kita |
|---|---|---|---|---|---|
| 12 Okt 07:29 | 5.809 | 86 | — (blok 0,954) | 0,954 | 466 (S6) |
| 14 Okt 07:44 | 5.957 | 201 | 750 | 0,956 (rank 297) | 558 (R7) |
| 14 Okt 15:00 | 5.983 | 213 | 840 | 0,956 (rank 299) | **615** |

- 188 dari 213 tim ≥0,957 submit dalam 24 jam terakhir → papan atas masih aktif.
- Proyeksi 22 Okt: ≥0,957 kemungkinan 300–400 tim → **garis perak publik bergeser ke 0,957**; papan privat (70 %) mengacak ulang blok.
- **Di dalam blok, rank mengikuti skor tak-dibulatkan**, tetapi salinan identik R7 tersebar ±400 rank (nondeterminisme GPU) → hanya lompatan besar yang bermakna.

## 2. Apa yang terbukti menaikkan rank (bukti publik)
| Komponen | Bukti | Status kita |
|---|---|---|
| r224 (checkpoint kedua nartaa) di slot Raptor | varshitha: rank 233→156 (13 Okt), turun ke 357 saat blok terisi | `r7r224` 57185636 PENDING |
| + TTA cermin anatomis utk r224 | varshitha "one free flip": 357→**289** (14 Okt) | dibangun `r7v3f`/`r7v3fs` (base `v224m`) |
| flip-TTA compact R7 (kode v4c heliosli, dimatikan di manifest) | jingshuifu menguji (stride 1; tim rank 268) | `r7v3f` (stride 2, biaya GPU ≈ sama) |
| **Model sendiri yang beragam, bobot 0,2** | **bliverrigpdoge: ConvNeXt-S + EffNetV2-S pada potongan kompartemen lutut, label laporan dreaddevelopment → 0,956→0,957** | murid r1 (distilasi Raptor) GAGAL; murid r2 17 Okt |
| Reader goodpjw sbg aliran ke-3 (mayankraaz v2) | belum ada hasil terklaim | — (kita: net-zero 11–12 Okt) |

## 3. Forum — sinyal penting
- **Tim teratas (azamat1ch 0,965, diet1236364): "short answer is labels… AI, not by hand. And don't rely on reports alone"**; tanpa data eksternal.
- Label gold dibuat radiolog dari GAMBAR (laporan bisa bertentangan, mis. Effusion 1 vs "no effusion") → label LLM dari laporan mentok ±0,89 AUC gold (hengck23). Trik label: biarkan LLM bilang "tidak tahu" & latih hanya sampel yang disepakati (>0,9/<0,1).
- **OAI (749145, 14 Okt)**: host 21 Sep — OAI tidak melanggar bila gratis & tanpa tinjauan institusi; masalah baru = blokir NIH utk Tiongkok dll. (Rule 2.6.b "geo restrictions"). Host diam; konsensus: dianggap sah sampai ada larangan; mayoritas papan memakai Raptor (OAI). Penegakan pada medali (bukan hadiah) tak jelas.
- 748181: komunitas mengeluh soal salinan notebook publik; Kaggle tidak bertindak.

## 4. Rencana menuju perak (prioritas)
1. **15 Okt 07:03**: submit `r7v3f` (R7 + r224 cermin + flip compact) dan `r7v3fs` (+ murid r1 di slot Raptor 0,25) → A/B efek murid; keduanya menyertakan komponen yang sudah terbukti.
2. **17 Okt 07:00** (kuota reset): dua murid paralel —
   - **r2a** (alver): label v1 = 0,5 konsensus + 0,5 OOF-R7, warm-start r1, 8 epoch, 2 fold/sesi → mirip R7 → uji di **slot compact**.
   - **r2c** (nxtirfan): label **konsensus saja** (tanpa guru → beragam, pola bliverrigpdoge), warm-start goodpjw, 12 epoch → uji sbg **kaki ke-3 0,2**.
3. 16 Okt (opsional): compact S6 di slot compact (+±1,7 j run).
4. 18–20 Okt: gabungkan komponen yang terbukti; variasi terakhir 20 Okt.
5. 21–22 Okt: user memilih 2 final (A = varian terbaik terverifikasi rank; B = varian mekanisme lain / jangkar R7).

## 5. Peluang (perkiraan, bukan janji)
- Salinan R7 saja: ±15–25 %.
- R7 + r224 cermin + flip (setara puncak blok): ±25–35 %.
- + model sendiri yang menambah +0,001 (mencapai 0,957): ±40–55 %.
- Merger dgn tim ≥0,958 (tenggat 15 Okt 23:59 UTC; keputusan user, ditolak 11 Okt): lebih tinggi.

## Catatan status terkini — 19 Okt 2026 02:52 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; medali ditentukan split privat 70%). Pada cek API 02:52, `g6c`, `g12`, `g13` masih PENDING; dua kernel training murid r4 di alver sudah dua kali di-requeue dan tetap QUEUED setelah push kedua pukul 02:51. Timer memantau keduanya. Kuota main/alver/nxtirfan tetap 19,87/17,81/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
