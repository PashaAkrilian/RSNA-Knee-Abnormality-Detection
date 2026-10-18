# Riset 19 Okt 2026 dini hari — posisi perak, aset publik, dan murid r4

> Audit riset dan antrean 19 Okt 2026 02:30 WIB; papan diperiksa ulang 02:33 WIB, API dan training 02:40–02:52 WIB. Data papan/submisi/kuota dibaca ulang lewat Kaggle CLI; dataset publik dan prediksi gold diperiksa lokal. Rencana aktif: `STRATEGI_PERAK.md` §0. Rincian operasi: `HANDOFF.md` §0c.

## Ringkasan keputusan

- Papan publik yang diunduh 02:33 WIB berisi **6.187 tim**. Krazy KIller **0,958 rank 467**; top 5% saat ini berakhir di rank **309**. Posisi publik membantu memilih resep, tetapi medalinya ditentukan setelah final pada leaderboard privat.
- `g6c` 57357991, `g12` 57359383, dan `g13` 57362095 masih `PENDING`; kernel inferensi mereka sudah `COMPLETE`. Belum ada submission baru sejak 18 Okt.
- Aset baru tidak memberi alasan mengganti R6/B-leg yang sudah diuji. **B-leg seed 1 tidak menambah gold**; Engine 8/9 belum punya bukti validasi; dua sumber label yang baru ditemukan berbeda dari target kita, tetapi tidak menyertakan OOF/gold yang bisa membuktikan kualitasnya.
- Jalur yang paling terdukung tetap melatih murid **r4** dari setengah konsensus laporan, seperempat OOF-R7, dan seperempat OOF-B-leg; baris tanpa OOF-B-leg memakai setengah konsensus + setengah OOF-R7. Empat fold sudah disiapkan dan dikirim sebagai dua kernel di akun alver. `g14` baru layak dibuat/submisi jika `g13` menunjukkan kenaikan pada papan.
- Estimasi kerja perak tetap **20–40%** untuk resep unik pada bukti sekarang. Angka ini simulasi berbasis posisi/klaster dan bukan probabilitas resmi atau jaminan.

## Papan, batas medali, dan risiko publik

Kaggle menyatakan kompetisi dengan lebih dari 1.000 tim memberi medali perak kepada 5% teratas; hitungan medali dibulatkan ke bawah. Dengan 6.187 tim sekarang, cutoff aritmetika publik = `floor(6187 × 0,05) = 309`. [Kaggle Progression](https://www.kaggle.com/progression/competitions)

Halaman resmi RSNA Knee menetapkan macro-average ROC AUC atas 12 label dan deadline final 22 Okt 2026 23:59 UTC. Papan kompetisi memakai 30% test untuk publik dan 70% untuk privat. Karena itu rank 467/6.187 dan cutoff publik 309 adalah penunjuk sementara; keputusan final harus mengutamakan gold/OOF yang tidak dipakai melatih serta ketahanan resep. [Overview kompetisi](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/)

Diskusi peserta kompetisi juga mencatat contoh pergeseran besar dari peringkat publik ke privat dan menyarankan validasi lokal yang meniru pembagian test/train. Ini anekdot peserta, bukan garansi tentang split kompetisi ini. [Diskusi tentang public/private drift](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744495)

## Pindai model dan dataset publik

`python3 scan_public_updates.py` menemukan tujuh dataset dan satu notebook yang belum tercatat sejak pemindaian sebelumnya. Notebook baru `govind123445/knee-12` berisi audit/pemeriksaan struktur dataset; bukan inferensi dengan skor publik. Daftar kernel berperingkat terbaru belum menunjukkan resep publik baru di atas keluarga 0,958 yang sudah diuji.

| Temuan | Bukti yang bisa diperiksa | Keputusan |
|---|---|---|
| B-leg seed 1 (`xhaitanya/rsna-knee-bleg-s1-f01`) | Dua ConvNeXt-S `E_prior`, warm-start dari R6, hanya fold 0–1; data menyatakan augmentasi/sampling berbeda, init sama dengan seed 0. | Evaluasi kuantitatif gold di bawah menunjukkan tidak ada peningkatan saat digabung dengan seed 0. Tidak dilatih ulang. |
| Engine 8/9 weights (`cyruspinto/rsna-knee-engine8-multiplane-weights`, `...engine9-384-weights`) | Hanya satu checkpoint per dataset; metadata deskripsi kosong, tanpa skrip inferensi, OOF, prediksi gold, atau hasil LB yang dapat dicocokkan. | Belum cukup informasi untuk mengintegrasikan atau menghabiskan kuota inferensi. |
| Dual-teacher labels (`srinath2495/rsna-knee-dual-teacher-consensus-4407`) | 4.407 baris probabilitas; reasoning ledger 4.332 baris; CC0. Spearman makro dengan target r3 kita 0,501. | Jauh berbeda, tetapi perbedaan tidak mengukur kebenaran. Dataset tidak menyertakan evaluasi OOF/gold dan metadata tak menjelaskan protokol kalibrasi; tidak dipakai untuk r4. |
| Argon4 labels (`abdullahmathar/rsna-knee-argon4-soft-labels`) | 4.407 baris, CC0; Spearman makro dengan target r3 kita 0,591. | Tidak ada sumber OOF/gold atau rincian kalibrasi yang cukup; simpan sebagai kandidat riset, jangan latih sebelum ada validasi yang independen. |

Seluruh file label publik hanya dipakai untuk membaca bentuk/distribusi dan kemiripan terhadap label internal. Korelasi yang rendah menunjukkan keragaman sumber, bukan mutu prediksi.

## Evaluasi ulang B-leg

Enam berkas gold logits `partial_preds_E_prior_s*.json` diunduh dari dataset publik B-leg seed 0/1 dan dipadankan dengan 58 label ahli lokal. AUC di bawah adalah rerata makro 12 label; emas-58 tetap kecil dan hanya dipakai sebagai arah.

| Kelompok | Gold macro AUC |
|---|---:|
| Seed 0, 4 fold | **0,92448** |
| Seed 1, 2 fold | 0,92065 |
| Rerata sama seed 0 + seed 1 | 0,92398 |

Spearman rerata per label antara seed 0 dan seed 1 = **0,9819**. Jadi seed 1 sangat berkorelasi dan menurunkan rerata gold sekitar 0,0005; tidak ada alasan untuk menjalankan sisa fold seed 1 atau menambahkannya ke final.

OOF seed 0 direkonstruksi dari `partial_oof_E_prior_s0.npy` dan `pool_uids.json`. Untuk tiap fold, tepat baris yang jatuh pada `int(sha1(uid)[:8], 16) % 4` berisi logits nonnol; empat file memberi **3.325 UID OOF unik**, tanpa baris gold. Setelah sigmoid, formatnya disamakan dengan keluaran `build_compact_oof.py` dan dipadankan ke OOF-R7.

Pada 3.325 baris B-leg, AUC terhadap label konsensus biner internal ialah **0,8860** untuk B-leg dan **0,8830** untuk R7. Ini ukuran kesesuaian terhadap konsensus, bukan AUC kebenaran. Sweep gabungan tersimpan sebelumnya pada irisan 2.651 studi menunjukkan B-leg .24 mengganti R7 .24 pada compact sebesar **+0,0033 (P=1,00)**; sweep gold terpisah menetapkan `g13` sebagai bobot terbaik. Uji integrasi gold B-leg lulus (`ρ=0,9984` terhadap logits penulis), sehingga bukti cukup untuk melatih murid r4, bukan untuk mengklaim perak.

## Murid r4: data, validasi, dan status eksekusi

Target r4 dibuat dengan `make_labels_v1.py` yang sudah ada:

```text
label = 0,5 × konsensus laporan
      + 0,25 × OOF-R7
      + 0,25 × OOF-B-leg
```

Pada 1.024 baris tanpa OOF-B-leg, guru menjadi OOF-R7 saja sehingga target = 0,5 konsensus + 0,5 OOF-R7. Dari 4.349 baris non-gold, 3.325 mempunyai OOF-B-leg dan 1.024 memakai fallback. Semua 58 label gold tetap identik dengan v0 dan `build_student_train.py` mengecualikan `is_gold == 1` dari loss latih.

Artefak lokal (tidak ditambahkan ke Git): `data/bleg_oof/compact_oof_f*.csv`, `data/labels/labels_student_v4.csv`, dan paket dataset. Dataset privat alver `alverlfarfaraway/rsna-knee-student-labels-r4` sudah diunggah agar label tidak diumumkan selama eksperimen.

| Kernel alver | Fold | Resep | Status pada 19 Okt 02:33 WIB (cek awal) |
|---|---|---|---|
| `rsna-knee-student-r4-a` | 0–1 | ConvNeXt-T, 12 epoch, init goodpjw, 2 fold paralel | QUEUED |
| `rsna-knee-student-r4-b` | 2–3 | Resep yang sama | QUEUED |

Kuota sebelum eksekusi GPU: main 19,87 jam, alver 17,81 jam, nxtirfan 4,62 jam; reset 24 Okt 07:00 WIB. Training hanya memakai alver, sehingga kuota nxtirfan tetap tersedia. Pantau status lewat `bash kg2.sh kernels status alverlfarfaraway/rsna-knee-student-r4-a` dan `...-b`; jika antrean berlarut, cek sesi aktif di Kaggle sebelum melakukan tindakan.

Gerbang setelah selesai:

1. Pastikan semua 4 fold selesai, tidak ada gold/label NaN, dan inferensi gold cocok dengan resep yang dievaluasi.
2. Bandingkan `g13` dan `g12` pada leaderboard publik saat keduanya selesai. Tetap gunakan nilai tak-dibulatkan melalui perubahan rank tim sebagai petunjuk; skor 0,958 yang sama tidak cukup untuk menyatakan kenaikan.
3. Hanya bila `g13` mengonfirmasi dan r4 lulus gerbang gold, bangun `g14 = g13 + r4 0,4` di akun utama untuk commit-run inferensi. Keputusan submisi menunggu hasil run/kuota hari itu; jatah 18 Okt yang tersisa tetap dihormati.
4. Final dipilih manual, maksimal dua, dengan resep berbeda mekanisme; submission terakhir disiapkan sebelum 22 Okt 15:00 UTC untuk margin proses.

## Referensi proyek

- Pipeline dan label: `build_student_train.py`, `make_labels_v1.py`, `build_r7murid.py`.
- Ringkasan operasional: `MEMORY.md` §2–§5, `HANDOFF.md` §0c, `laporan_push.md` §1–§2.
- Metadata publik: [B-leg seed 1](https://www.kaggle.com/datasets/xhaitanya/rsna-knee-bleg-s1-f01), [dual-teacher labels](https://www.kaggle.com/datasets/srinath2495/rsna-knee-dual-teacher-consensus-4407), [Argon4 labels](https://www.kaggle.com/datasets/abdullahmathar/rsna-knee-argon4-soft-labels), [Engine 9 weights](https://www.kaggle.com/datasets/cyruspinto/rsna-knee-engine9-384-weights).

## Tindak lanjut antrean — 19 Okt 2026 02:52 WIB
Papan terakhir yang diunduh tetap menunjukkan 0,958 rank 467/6.187; rankpoll 02:50 selesai sukses tanpa mencatat perubahan. API 02:52 masih mencatat g6c/g12/g13 sebagai PENDING. Dua kernel r4 di alver awalnya tetap QUEUED setelah push pertama 02:42. Atas instruksi user jika masih QUEUED, watchdog menjalankan push ulang kedua: r4-a dihapus 02:51:09 lalu berhasil dikirim v1 02:51:32; r4-b dihapus 02:51:36 lalu berhasil dikirim v1 02:51:59. Cek API 02:52 tetap menunjukkan keduanya QUEUED. Kuota GPU tidak berubah (main/alver/nxtirfan 19,87/17,81/4,62 jam), jadi training belum mulai. Timer tetap aktif memantau antrean dengan ambang lima menit. Tidak ada submission kompetisi baru.
