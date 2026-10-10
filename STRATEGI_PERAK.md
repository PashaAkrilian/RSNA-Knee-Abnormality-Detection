# STRATEGI PERAK — RSNA Knee 2026 (disusun 10 Okt 2026 ~21:00 WIB; sesi Claude Code; HANYA akun utama)
> Baca setelah `MEMORY.md`. Dasar: riset 10 Okt (`reports/Ringkasan riset 10 Okt (sesi Claude Code).md`), uji skala OrthoFoundation, Rules resmi,
> papan 10 Okt 19:37 WIB. Dokumen ini = RENCANA + ATURAN KEPUTUSAN; status harian ada di `HANDOFF.md` §0c dan `laporan_push.md`.

## 1. Sasaran dan ukuran
- Perak = 5 % teratas di papan **privat** (70 % data uji) ≈ rank 283 dari 5.666 tim. Papan publik (30 %) hanya penunjuk: ≥0,952 = 217 tim, ≥0,951 = 348, =0,950 = 564 (kita rank 530).
  Garis naik ≈ +20 tim ≥0,952 per 12 jam → proyeksi 22 Okt: kandidat final harus ≥**0,952–0,954** publik.
- **100 % tidak bisa dijamin** (papan privat acak di dataran 0,950). Tujuan realistis: menaikkan peluang dari ±15–20 % ke ±40–80 %.

## 2. Dasar kuantitatif (simulasi, model sederhana; rentang = s regresi 0,3–0,7 × σ privat 0,003–0,005)
| Kondisi (skill sejati) | P(perak) |
|---|---|
| Apex 0,950 (sekarang) | 14–20 % |
| + naik 0,002 | 26–39 % |
| + naik 0,004 | 40–67 % |
| + naik 0,006 | 55–85 % |
| tim publik 0,954 (mis. merger) | 21–51 % |
| tim publik 0,956–0,958 | 23–82 % |
| tim publik 0,960 | 30–92 % |
Pelajaran: (a) perbaikan kecil (+0,001) hampir tak mengubah apa pun; (b) hanya lompatan ≥ +0,004 atau masuk tim ≥0,956 yang berarti; (c) ketidakpastian
besar → jangan menyetel ke papan publik. Kode simulasi: lihat `docs/RESEARCH_NOTES.md` §40 (regresi ke rata-rata + noise; Tweedie tak valid krn lonjakan 564 salinan di 0,950).

## 3. Empat jalur (paralel; EV menurun)
### J1 — MERGER dengan tim ≥0,955 (EV terbesar; AKSI USER; batas 15 Okt 23:59 UTC)
- Rules 5.c: sah bila (i) gabungan ≤5 anggota, (ii) jumlah submisi gabungan ≤ jatah satu tim pada tanggal permintaan (kita 107; mitra ≤ ±250), (iii) sebelum batas merger, (iv) memenuhi Rules lain.
  Rules 5.d/6.a: dilarang berbagi kode/data secara privat antar-tim SEBELUM merger (berbagi PUBLIK di forum boleh).
- Kandidat: `docs/KANDIDAT_MERGER.md` (126 tim ≥0,953, ≤2 anggota, ≤250 submisi; fokus rank 40–75, skor 0,955–0,958). Hubungi banyak sekaligus; sebagian besar akan menolak.
- Syarat praktis: halaman tim kita menampilkan 3 akun → jelaskan jujur & selesaikan kepastian akun kedua (J4). Tawaran kita: ckpt OAI (pengetahuan label berbeda dari model berbasis laporan),
  pipeline inferensi cepat (<1 jam) + uji OrthoFoundation, kuota GPU akun utama.
- Setelah merger: ensembel model mereka + stack kita (rank-blend; gate bobot di OOF/LB), pilih 2 final bersama.
### J2 — TUAS TEKNIS: Apex 0,70 + OrthoFoundation-L fold0 0,30 (resep heliosli: Raptor 0,7 + OF 0,3 = 0,951–0,954; bobot tetap)
- Percobaan 1 (57028008) = **Notebook Timeout**. Uji skala: disk 330 MB/studi (potongan 30 studi menulis ±10 GB → disk penuh), CPU (4 vCPU) hambatan (GPU sering 0 %), 2 proses 45–95 dtk/studi,
  4 proses ±60 dtk/studi/proses (≈15 dtk/studi agregat). Perbaikan (`of_stage.py`): potongan 8 studi + akar-data kecil per potongan, 4 proses (2/GPU), penjaga ruang disk, thread=1 (+cv2),
  prioritas studi (disagreement nartaa vs reader), keluaran bertahap, berhenti lunak +6,6 j & bunuh +7,0 j, reuso runtime anatomi (−45 dtk/potongan), fallback rank Apex bila OF kurang.
- Gerbang G0 (**LULUS 10 Okt 20:15**): `ofscale3` (konfigurasi produksi) = 104 studi/31 mnt, 0 gagal, disk ≥14,8 GB → 194 studi/jam (6,8 j; akhir ≈224/jam = 5,9 j) dgn pemutus proses (+6,6/+7,0 j) → liputan ≈97–100 %. `ofscale4` (reuso anatomi) 150/jam → tak lebih cepat (dimatikan). Submit `apexof2 -v 3` dijadwalkan **11 Okt 07:03 WIB** (jatah agen).
- Gerbang G1 (hasil LB): **≥0,952 → kandidat utama; 0,951 → cadangan/hedge; ≤0,950 → TUTUP OF** (jangan membuang kuota untuk variasi kecil). Peluang kaki ini menambah ≥ +0,001: ±35–40 %.
- Ditutup (jangan ulangi): varian r224/bobot (0,950), MaVIT (0,949), CoAtNet resgated (emas 0,909), pseudo-label/anak-OAI (≈ induk), kompilator aturan heliosli (0,784 vs emas), Fase A mirror (EV≈0).
### J3 — FINAL TAHAN GUNCANGAN (21–22 Okt; pilih MANUAL & verifikasi via API; auto-fill §3.18.c)
- 2 final = beda MEKANISME: (A) kandidat LB terbaik (Apex atau Apex+OF), (B) hedge: Apex-keluarga lain / OF-blend / (bila host melarang OAI) jalur bersih 0,944.
- Variasi baru terakhir disubmit ≤20 Okt (run OF ±7 jam). Pilih dari submisi yang SUDAH terskor; tandai di UI Submissions → select 2. Jangan "mengukur noise" dgn submisi identik.
### J4 — KEPATUHAN (melindungi medali)
- Akun kedua/ketiga: Rules 5.a ("disqualified if ... more than one Kaggle account") & 5.b ("each Team member must be a single individual"). Izin tertulis yg ada = satu kalimat tanpa pertanyaan/pengirim/tanggal.
  Tindakan: kirim pesan ke Kaggle support (draf `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`); tidak ada submisi/komputasi RSNA dari akun 2/3; rekan `nxtirfan` menjalankan job dari akunnya sendiri bila mau.
- OAI/data eksternal: ckpt nartaa & OrthoFoundation membawa pengetahuan eksternal; host diam (743416, 35 komentar, 0 jawaban host) → pertahankan satu final dgn risiko berbeda.

## 4. Garis waktu dan gerbang (WIB)
| Waktu | Aksi | Gerbang / keputusan |
|---|---|---|
| 10 Okt malam | selesaikan uji skala; bila G0 lulus submit `rsna-knee-apexof2` (hasil ±7 jam) | agen maks 2 submisi/hari; sisa jatah utk user/rekan |
| 10–12 Okt | USER: hubungi tim kandidat merger + kirim pesan ke Kaggle support | merger butuh persetujuan mitra |
| 11–13 Okt | baca skor Apex+OF → G1; `scan_public_updates.py` harian | ≥0,952/0,951/≤0,950 |
| 13–15 Okt | bila ada mitra: protokol merger; semua kandidat ter-push sebelum 15 Okt (`kernelsPublishingDisabledDeadline`) | 15 Okt 23:59 UTC = 16 Okt 06:59 WIB |
| 16–20 Okt | iterasi bergerbang (maks 2/hari); GPU baru 30 j pada 17 Okt 07:00 WIB; bila ada merger: ensembel bersama | submisi variasi TERAKHIR 20 Okt |
| 21–22 Okt | cek ulang 743416; pilih 2 final manual; verifikasi via `kaggle competitions submissions` | 22 Okt 23:59 UTC |

## 5. Anggaran
- GPU akun utama: ±27 jam tersisa (sesi uji skala berjalan), +30 jam 17 Okt. Run scoring tidak memakai kuota GPU. Submisi tim: 5/hari (reset 07:00 WIB), agen ≤2, sisakan 2–3.
- Jangan jalankan job proyek lain di akun utama. Disk /kaggle/working ±19 GB efektif; keluaran OF ±330 MB/studi.

## 6. Risiko utama
| Risiko | Mitigasi |
|---|---|
| OF gagal lagi (timeout/disk) | pemutus proses, penjaga disk, liputan parsial berprioritas, fallback Apex; uji skala dulu |
| OF tidak menambah skor | aturan G1; pindah fokus ke J1/J3 |
| Papan privat mengacak dataran 0,950 | dua final beda mekanisme; jangan menyetel ke papan publik |
| Host melarang data eksternal/OAI | satu final jalur bersih 0,944 (hasil bronze) |
| Diskualifikasi akun ganda | J4; semua submisi dari akun utama |
| Garis perak naik cepat | target publik ≥0,952; merger = satu-satunya lompatan besar |
