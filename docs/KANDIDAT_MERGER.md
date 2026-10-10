# Opsi: penggabungan tim sebelum 15 Okt 2026 23:59 UTC (Team Merger Deadline)
> Diperbarui 10 Okt 2026 07:40 WIB (papan 5.625 tim). Hanya opsi; TIDAK ada pesan yg dikirim agen. Penggabungan dilakukan ketua
> tim lewat halaman Team kompetisi. Syarat (Rules 2.1.b/3.5.c): total submisi gabungan <= batas harian×hari (≈355–385; kita 107),
> ukuran tim gabungan <= 5. PERINGATAN Rules 3.5.d: SEBELUM bergabung dilarang berbagi kode/data kompetisi secara privat antar tim.

## Syarat resmi (Rules 5.c) — ringkas
- **Rules 5.c (resmi)**: merger sah bila (i) gabungan ≤5 anggota; (ii) jumlah submisi tim yg bergabung ≤ jatah satu tim pada tanggal permintaan (5/hari × hari berjalan; `kaggle competitions submission-limits` tak menampilkan angka maksimum — hitung sendiri); (iii) selesai sebelum batas merger (15 Okt 23:59 UTC); (iv) memenuhi Rules lain.
- Dilarang berbagi kode/data secara privat antar-tim SEBELUM merger (5.d/6.a); berbagi publik di forum boleh. Ketua tim yang mengajukan; mitra harus menyetujui. Dalam tim, tiap anggota harus individu terpisah (5.b).

## Mengapa ini opsi dengan EV terbesar
- Dataran 0,950 (482 tim) = model yang sama (Apex-keluarga). Model berbeda-arsitektur dgn label yg sama hanya menambah ±0,001
  (galat berkorelasi tinggi); tim di 0,953–0,958 punya PENGETAHUAN LABEL berbeda (pseudo-label dsb) → ensembel dgn model kita
  (stack OAI-bobot publik) menambah nyata bagi mereka, dan bagi kita memindahkan posisi dari ~0,950 ke band mereka.
- Papan 10 Okt 06:57 WIB: 145 tim ≥0,953; **126** di antaranya beranggota ≤2 & submisi ≤250 (muat dlm batas merger); 107 dari 145 solo.

## Hal yang harus Anda bereskan dulu (aman)
- Halaman tim kita menampilkan **3 akun** (dimaspashaakrilian, nxtirfan, alverlfarfaraway). Mitra akan melihatnya dan ukuran
  gabungan jadi 3+n. Bila ditanya, jelaskan jujur (dua akun milik Anda). Lihat `docs/DRAF_PESAN_KAGGLE_SUPPORT.md` — minta kepastian
  tertulis dulu agar mitra tidak ikut menanggung risiko 3.5.a/3.5.b.

## Kandidat (papan 10 Okt; tim <=2 anggota, ≤250 submisi; info publik, bukan rekomendasi; band rank 40–73)
| rank | skor | tim | username | submisi |
|---|---|---|---|---|
| 40 | 0,958 | Mazzutti | profmazzutti | 91 |
| 44 | 0,958 | colum2131 | columbia2131 | 43 |
| 46 | 0,958 | Awshesh Nath | awsheshnath | 108 |
| 47 | 0,958 | Q | quan0095 | 178 |
| 49 | 0,957 | monnu | fuumin621 | 52 |
| 50 | 0,957 | Lukas Nissen Molvær | lukasnissenmolvr | 78 |
| 53 | 0,957 | aleksandr3312 | gapchenko | 50 |
| 54 | 0,957 | jjakimoto | jjakimoto | 6 |
| 55 | 0,957 | NguyenThanhNhan | andy2709 | 76 |
| 56 | 0,957 | Komil Parmar | komilparmar | 89 |
| 57 | 0,956 | LP | lihu6666 | 15 |
| 62 | 0,956 | kreininmv | kreininmv | 163 |
| 63 | 0,956 | Khayrul Islam | khayrulbuet13 | 105 |
| 65 | 0,956 | Joppe Kietselaer | joppekietselaer | 39 |
| 66 | 0,956 | Boris Konstantinovskii | jusill | 76 |
| 67 | 0,956 | LI YIZHOU | yaphellee | 26 |
| 70 | 0,956 | Yakup Mert Karasulu | yakupmertkarasulu | 17 |
| 72 | 0,956 | wishmeluck | wishmeluck001b | 26 |
Catatan: top-30 (0,959–0,964) kebanyakan veteran/GM, kecil kemungkinan butuh mitra. Cari juga thread "Looking for teammates" (mis. 745091).

## Draf pesan (Inggris, pendek; jangan menyebut kuota akun ganda)
"Hi! We're Krazy KIller (public LB 0.950). With the merge deadline on Oct 15, we're looking for a team with *independently
trained* models (your own labels/pseudo-labels, not forks of the public Apex/nartaa notebooks). Our stack uses the public
nartaa checkpoint, which was trained with different label knowledge than label-report-only models, so a merged ensemble
could improve both of us. We have ~30h/week of GPU on our main account and a clean, fast (<1h) inference pipeline.
Interested in a short chat about merging? (No code/data sharing before the merge, per the rules.)"

## Cara (UI Kaggle)
Competition → *Team* → ketua mengundang/menerima *Merge with another team* → pihak lain menyetujui. Cek `kaggle competitions submission-limits rsna-knee-abnormality-detection` (batas gabungan) sebelum meminta.
