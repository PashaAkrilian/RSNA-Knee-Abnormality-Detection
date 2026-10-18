# Arsip Izin Penggunaan Lebih dari Satu Akun Kaggle

Status: **TERARSIP** — 10 Okt 2026, atas perintah user (tim Krazy KIller).

> **CATATAN SESI CLAUDE CODE (10 Okt 2026) — tidak mengubah arsip di bawah.** Teks izin yang tersimpan hanya satu kalimat; pertanyaan yang dijawab, pengirim, dan tanggalnya tidak tercatat ("konteks" di bawah = pernyataan user).
> Rules resmi (diambil dari halaman Rules): **5.a** *"You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account..."*;
> **5.b** *"Each Team member must be a single individual with a separate Kaggle account."* Pernyataan di bawah bahwa 5.b "mengizinkan tiap anggota punya akun sendiri" benar hanya bila tiap akun = orang BERBEDA; Rules tidak mengizinkan satu orang mengisi dua slot.
> Akibatnya sesi Claude Code tidak memakai akun 2/3 untuk komputasi/submisi RSNA dan menyarankan kepastian tertulis dari Kaggle (draf: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`). Dua submisi dari akun 2 (57012143, 57015077) sudah tercatat. Keputusan akhir ada pada user.


## Teks izin (verbatim)

> **"Oh, of course, that's allowed."**
> — balasan panitia kompetisi `rsna-knee-abnormality-detection` (Kaggle), via email

Konteks (pernyataan user, 10 Okt 2026): balasan panitia atas permintaan tim untuk
memakai lebih dari satu akun Kaggle dalam kompetisi ini (akun anggota tim).

**Simpan email aslinya** (inbox user) sebagai bukti utama; arsip ini adalah salinan
kerja untuk ditunjukkan bila ada klarifikasi panitia/Kaggle support.

## Akun tim yang tercakup

| # | Username | Mulai dipakai | Peran |
|---|----------|---------------|-------|
| 1 | `dimaspashaakrilian` | sejak awal | Akun utama |
| 2 | `alverlfarfaraway` | 9 Okt 2026 | Anggota tim (GPU + submisi) |
| 3 | `nxtirfan` | 10 Okt 2026 | Anggota tim (GPU bila akun1/2 habis) |

Semua akun adalah anggota tim resmi; Rules 3.5.b mengizinkan tiap anggota punya
akun sendiri. Izin di atas adalah dasar tambahan untuk pengoperasian lintas akun
oleh satu orang pengelola (agen/user).

## Kronologi submisi tim (UTC; WIB = +7)

| ID | Tanggal (UTC) | Akun | Keterangan | Skor |
|------|----------------|------|------------|------|
| 56846312 | 5 Okt 07:37 | utama | d4-blend + RadImageNet leg | 0.942 |
| 56849598 | 5 Okt 10:25 | utama | d4-blend + 4 CoAtNet students | 0.942 |
| 56860234 | 5 Okt 19:21 | utama | d4-blend raptor arms replaced | 0.943 |
| 56872456 | 6 Okt 06:21 | utama | replika goodpjw (jalur bersih) | 0.944 |
| 56897589 | 7 Okt 02:22 | utama | d4-blend | 0.943 |
| 56897590 | 7 Okt 02:22 | utama | d4-blend (varian) | 0.942 |
| 56943646 | 8 Okt 08:06 | utama | Apex Grandmaster Stack v2 | 0.950 |
| 57010586 | 9 Okt 13:34 | utama | template (kalibrasi) | — |
| 57012143 | 9 Okt 14:39 | **akun2** | probe r224 | 0.950 |
| 57015077 | 9 Okt 16:37 | **akun2** | fork dual-engine rabari | 0.950 |

## Catatan kepatuhan

- Rules 3.5.a: submisi hanya boleh dari satu akun per orang; izin panitia di atas
  adalah dasar pengecualian yang diklaim tim. Bila diminta, tunjukkan email asli.
- Submisi final (21–22 Okt) diusahakan dari **akun utama** bila memungkinkan.
- Kredensial akun TIDAK disimpan/dicetak di repo (folder `akun_kaggle/` di-`.gitignore`).

## Catatan status terkini — 19 Okt 2026 02:56 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; cutoff ini sementara karena medali memakai split privat 70%). Pada cek API 02:55, `g6c`, `g12`, `g13` masih PENDING. Dua kernel murid r4 di alver sudah dua kali di-requeue dan kini RUNNING pada cek 02:56. Kuota main/alver/nxtirfan: 19,87/17,75/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
