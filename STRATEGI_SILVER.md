# STRATEGI SILVER — RSNA Knee 2026 (rencana ditulis 15 Okt; status disegarkan 19 Okt 2026)
> **Target user: SILVER sebelum kompetisi berakhir (final 22 Okt 23:59 UTC = 23 Okt 06:59 WIB).** Dokumen ini = RENCANA + ATURAN KEPUTUSAN yang berlaku 16–22 Okt.
> Status harian dan kerja aktif: `HANDOFF.md` §0/§0c · hasil submisi: `laporan_push.md` §2 · bukti riset: `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md` · simulasi: `silver_sim_cluster.py` · `docs/PLAN_B.md` = arsip cadangan. Bagian "Arsip" di bawah = rencana 10–12 Okt (era S6), hanya sejarah.

## 0. RENCANA SILVER (AKTIF, diperbarui 19 Okt)

### 0.00 Pembaruan 18 Okt malam (arsip baseline, disusul §0.00b–§0.00c)
- **Posisi**: 0,958 rank 424/6.168; silver (rank ±308) ada di dalam blok 0,958 → yang menentukan = skor tak-dibulatkan (publik) dan, di akhir, skor PRIVAT 70 %. Salinan identik notebook publik berbagi satu skor privat → resep unik kita (S6F nartaa di separuh Raptor + murid r2a) adalah satu-satunya jalan melompati klaster.
- **Bukti komponen**: S6F wajib (g8b tanpa S6F 0,957 vs g8a 0,958); murid r2a menaikkan g6a di atas salinan goodpjw (rank 569→323); R7 di compact: OOF lokal +0,002 (P 1,00), gold +0,001, papan netral (g9a) → masuk kandidat `g12`.
- **Kandidat final**: A = resep unik dgn bukti publik terbaik (saat ini `g6a`; diganti bila `g6c`/`g10a`/`g12` menaikkan rank tim). B = resep dgn bukti OOF lokal terbesar (`g12`: R6+R7+murid .4 + S6F) bila bukan A; keduanya HARUS memuat S6F + murid. Salinan murni (57269378, 57322996) hanya cadangan.
- **Sisa tuas**: model publik baru yang beragam (riset `reports/Riset 18 Okt malam - model publik baru.md`), murid berlabel baru di akun 2/3 hanya bila label publik baru terbukti lebih baik di gold-58 (murid dgn guru berbeda ρ 0,97 → keragaman kecil).
- **Peluang jujur**: silver ±20–40 %; perunggu jauh lebih mungkin; emas ≈0 %.

#### 0.00b Pembaruan 19 Okt 00:35 (sesi OpenCode) — sweep OOF, pohon keputusan, rencana 19–22 Okt
- **Sweep OOF irisan 2.651 studi** (label konsensus v0c; `/tmp/opencode/oof_sweep19.py`): solo R6 0,8855 · R7 0,8810 · **B-leg 0,8841**; slot compact: g10a 0,9012 · g12 0,9027 · **g13 0,9060** → **B-leg .24 menggantikan R7 = +0,0033 (P 1,00)**; naikkan B-leg ke .3/.36 hanya +0,0003 (derau). **Gold-58**: **g13 0,9415 = terbaik** (B-leg .3: 0,9412; .36: 0,9407) → **desain `g13` sudah di optimum bobot; jangan dinaikkan**. (murid di eval lokal bias label konsensus — angka absolut dilebihkan; urutan B-leg>R7 tidak, karena B-leg tak dilatih label itu.)
- **Pohon keputusan (±01:00–03:40 WIB, dari rank poller)**: g13 rank < 452 → B-leg terbukti → lanjut tuas berikut; g12 naik g6c tidak → R7 bekerja; g6c naik → murid8 bekerja; semua datar → `g6a` tetap kandidat A.
- **Tuas berikut = murid r4** (keluarga model-milik-kita, pola bliverrigpdoge): label = 0,5 konsensus + 0,25 OOF-R7 + 0,25 **OOF-B-leg** (studi tanpa OOF B-leg → 0,5 konsensus + 0,5 OOF-R7); latih 4 fold di akun 2 (alver 17,8 j sisa) 19–20 Okt → `g14` = g13 dgn murid r4 (bobot .4, sama spt g13) → submit ±20–21 Okt bila g13 mengonfirmasi.
- **Jatah**: sisa 18 Okt (1) **TIDAK dipakai** (perintah user 19 Okt 00:10); 19–22 Okt: agen ≤1–2/hari, selalu sisakan ≥1 utk user; final 22 Okt = A (resep unik terbaik terskor) + B (mekanisme beda); submisi baru terakhir aman **≤22 Okt 15:00 UTC** (skor penuh ±4 j).

#### 0.00c Status papan/API 19 Okt 02:10 WIB
- Papan publik Kaggle (02:08 WIB): **0,958 rank 465/6.184**; terbaik COMPLETE `g9a`/`g6a` = 0,958; batas silver publik sekitar rank 309. Kaggle API (02:04 WIB): `g6c` 57357991, `g12` 57359383, `g13` 57362095 masih PENDING.
- Timer `rsna-rankpoll` (5 mnt) dan `rsna-requeue` (2 mnt) aktif. Kuota GPU main/alver/nxtirfan: 19,87/17,81/4,62 jam; reset 24 Okt 07:00 WIB.
- Tidak ada hasil baru yang mengubah pohon keputusan: tunggu skor g6c/g12/g13; jangan gunakan jatah tersisa 18 Okt; murid r4/`g14` hanya bila g13 terkonfirmasi.


#### 0.00d Status dan keputusan 19 Okt 02:56 WIB
- Papan publik 02:33 WIB: **0,958 rank 467/6.187**; cutoff aritmetika publik top 5% = rank 309. Medali ditentukan split privat 70%. `g6c` 57357991, `g12` 57359383, dan `g13` 57362095 masih PENDING.
- B-leg seed0 memberi 3.325 OOF UID tanpa gold overlap; korelasi logits gold dengan pembuat ρ=0,9984. Seed1 ditolak (gold 0,92065; rerata seed0+1 0,92398 < seed0 0,92448). Engine8/9 dan label dual-teacher/Argon4 ditahan karena tidak ada validasi independen.
- Dataset r4 privat dan dua kernel 12 epoch ada di alver; `student-r4-a/b` di-requeue pukul 02:42 dan 02:51 (kembali v1), lalu RUNNING pada cek 02:55. Kuota terbaru main/alver/nxtirfan 19,87/17,75/4,62 jam; timer aktif.
- Lanjutkan r4 setelah kernel berjalan; periksa semua fold, emas dikecualikan, inferensi gold, dan waktu run. Jangan buat `g14` kecuali g13 naik pada papan dan r4 lulus. Rincian metode, aset yang ditolak, dan limitasi metrik ada di laporan riset 19 Okt.

#### 0.00e Status dan keputusan 19 Okt 04:17 WIB — pemakaian jatah terakhir hari ini
- Papan publik Kaggle (unduhan 04:17 WIB): **0,958 rank 473/6.190**; batas aritmetika top 5% sekitar rank 310. API submission: `g6c` 57357991 COMPLETE 0,958; **`g10a` 57369237 PENDING**, `g12` 57359383 PENDING, `g13` 57362095 PENDING.
- Atas instruksi user, kirim **`g10a` v1** pada 04:16:44 WIB dari akun utama. Commit-run COMPLETE, pseudo-output 3 baris lolos validasi; resep = goodpjw v27 tanpa Part B, R6 stride 2 + S6F + murid r2a 8-model .4. Gold-58 g10a 0,9403 vs g6c 0,9408 dan g13 0,9415; ini sinyal terbatas, skor publik/privat tetap menunggu API.
- Jatah submission 19 Okt **5/5 terpakai, 0 tersisa** hingga reset harian 07:00 WIB. Lifetime submissions 143. Kuota GPU main/alver/nxtirfan **19,87/15,03/4,62 jam**; reset mingguan 24 Okt 07:00 WIB.
- Kernel `student-r4-a/b` terakhir RUNNING pada cek 04:17 (alver); empat fold, gold, dan waktu run belum diverifikasi. Pemeriksaan CLI lanjutan pada endpoint privat akun alver ditolak.
- Keputusan berikut: tunggu skor `g10a/g12/g13` dan selesaikan validasi r4. `g14` tetap **terkunci** kecuali g13 menaikkan papan dan r4 lolos gold. Tidak ada jatah harian tersisa saat status ini dicatat; dua final tetap dipilih manual sebelum tenggat.

### 0.1 Apa itu silver di kompetisi ini
- **PEMBARUAN 17 Okt 03:00**: papan 01:51 WIB (6.103 tim): ≥0,959 = 124, **blok 0,958 = 488 (kebanyakan salinan R6+flip goodpjw/heliosli), garis silver publik = rank 305 (0,958)**; **kita 0,958 rank 317** (salinan user goodpjw 57269378). Simulasi ulang (`silver_sim_1017.py`, δ relatif thd klaster goodpjw): salinan murni 0–12 %; unik δ 0 → 25–30 %; δ +0,0003 → 40–62 %; **δ +0,0006 → 57–86 %**; δ +0,001 → 76–99 %. Kandidat unik: `r7v6a` (gold +0,0031 vs goodpjw, P 0,84), `g6a` (gold +0,0015, P 0,69). Papan publik condong ke R6+flip (0,958 vs 0,957) → dua sinyal saling tarik, keduanya dlm derau.
- Silver = **5 % teratas papan PRIVAT** (70 % uji ≈ 900 dari 1.284 studi). Proyeksi ±6.150 tim saat tutup → **rank ≤ ±308**. Papan publik (30 %, ±385 studi) hanya penunjuk.
- Papan publik 15 Okt 19:49 WIB: ≥0,958 = 132 · **≥0,957 = 327 (garis silver publik rank 301 = 0,957)** · tepat 0,956 = 941 · kita **rank 398 (0,956)**. ≥0,957 tumbuh ±180/hari (salinan notebook publik yang beruntung), ≥0,958 ±20/hari.
- Medali ditentukan papan privat dari **2 final yang dipilih user secara manual**. 100 % tidak bisa dijamin.

### 0.2 Model kuantitatif: berapa keunggulan yang dibutuhkan (`silver_sim_cluster.py`)
Blok 0,956–0,957 = **keluarga resep** (ratusan salinan byte-identik → SATU skor privat per keluarga; seri dipecah waktu submit). Resep unik kita naik/turun sebagai satu unit terhadap keluarga-keluarga itu. P(silver) menurut keunggulan privat sejati δ terhadap keluarga "R7 dasar" (12 skenario: tim kuat ≥0,958 200–300 tim, σ keluarga 0,0003–0,0006, σ independen 0,0015–0,0025):

| δ privat vs R7 dasar | 0 (salinan) | +0,0003 | +0,0006 | **+0,0010** | +0,0015 | +0,0020 |
|---|---|---|---|---|---|---|
| P(silver) | 2–10 % | 13–19 % | 27–47 % | **49–86 %** | 76–99 % | 91–100 % |

- **Salinan publik murni = hampir pasti gagal** (dibagi rata dgn ratusan salinan, diurutkan waktu submit).
- Keluarga publik terkuat yang diketahui (r224 cermin + S6F, leoprovorov) ≈ δ +0,0006 → kita harus DI ATAS mereka: **target kerja δ ≥ +0,001** = komponen publik terbaik + model milik kita sendiri.

### 0.3 Pola yang terbukti (bukti publik, 13–15 Okt)
- Setiap kenaikan tampilan ke 0,957 datang dari **MENAMBAH MODEL beragam**, bukan menyetel bobot: bliverrigpdoge (model knee-centred pribadi bobot 0,2 di atas R7: 0,956→0,957), leoprovorov (S6F nartaa 0,4 di slot compact: 0,957, rank 212 dgn 4 submisi), varshitha (r224 + cermin: lompatan rank di blok).
- Menyetel bobot / reader goodpjw / S6 lama di slot compact: tidak menaikkan (MAYANK, Nut_RJ, kita 11–14 Okt).
- Klaim 4 desimal heliosli (0,9573/0,9576) tak terbukti: salinan byte-identik "0,9573" = 0,956 di dua tim (termasuk kita).

### 0.4 Resep final kita (kandidat A) & bukti tiap komponen
| Komponen | Peran | Bukti | Status |
|---|---|---|---|
| R7 heliosli (ConvNeXt-S 4 fold, aux fastMRI) | inti keluarga label R7 | gold-58 0,9312; R7 + Raptor = 0,956 publik | dipakai di semua varian |
| Raptor r384 + r224 nartaa, + cermin anatomis | inti keluarga label JEV | r224+cermin: kita rank 506→403; varshitha 357→289 | dipakai (`base=v224m`) |
| **S6F** nartaa (ConvNeXt-S all-data, label JEV, potongan knee-centred) | model ke-3 yg beragam | **gold-58 0,9314 (bersih)**, ρ~R7 0,871, ρ~Raptor 0,879; blend gold **+0,0010 di slot compact**, **+0,0034 di separuh Raptor** (P>0 0,92); leoprovorov 0,957 publik; waktu run ±1,3–1,5 j (gold: 7,3 dtk/studi/GPU) | terintegrasi (`s6f=`, `s6fslot=`); uji 16 Okt `r7v3s` vs `r7v3sr` |
| **Murid r2a kita** (ConvNeXt-T 2.5D; label 0,5 konsensus + 0,5 OOF-R7) | **pembeda milik kita** | gold-58 0,9282 (2 fold), ρ~R7 0,911; sendirian +0,0003, **bersama S6F +0,0014 s/d +0,0021** | 2 fold kini; 17 Okt → 4 fold + 2 full-data |
| Murid r2c (label konsensus saja) | keragaman tambahan | belum (latih 17 Okt) | opsional sbg `fold4–7.pt` |
| Flip-TTA R7 stride 1 | TTA | gold setia: ρ R7~R7flip 0,9976, efek blend −0,0004…−0,0007; papan `r7v4c40` 0,956; +40 % waktu R7 | **DIBUANG (16 Okt)** |

- **16 Okt 20:30 — murid 4 fold & R6**: murid r2a 4 fold gold **0,9334** (> R7); R6+flip (basis heliosli 0,958) gold 0,9366 tetapi **tak menambah** di resep kita bila murid 4 fold ada; resep heliosli 0,958 direkonstruksi = 0,9382 < `r7v6a` **0,9414**. Papan 19:48: garis silver publik **0,958**; kita ke-15 di blok 0,957.
- **16 Okt — gold SETIA (pipeline submisi pd 58 gold, `gold_eval_full.py`)**: r7v3sr 0,9387 (+0,0033 vs dasar; papan 495→317 ✔); S6F compact = ±0; S6F di separuh Raptor 0,55 + murid 0,30 = 0,9410; + bobot heliosli 0,9416; murid +0,0011…0,0014. **Resep kerja: S6F ±0,5 di separuh Raptor + murid 0,30 (6 model 17 Okt) ± bobot per-temuan heliosli** (`r7v5a`).
- Evaluasi gold-58 (`gold_eval_blend.py`): 58 studi, SE ±0,01 absolut; selisih berpasangan lebih presisi tetapi tetap ARAH. Proksi Raptor = r384 saja (tanpa r224/cermin, dan dipilih pd gold) → struktur yg mengurangi porsi Raptor bisa tampak terlalu bagus → **struktur diputuskan papan publik (16 Okt), bobot tetap a priori**.

### 0.5 Jadwal & gerbang keputusan (WIB; agen ≤2 submisi/hari, jatah ekstra hanya atas izin user)
| Waktu | Aksi (otomatis = systemd --user) | Gerbang / keputusan |
|---|---|---|
| 16 Okt ±01:00 | skor `r7v4c40` 57231466 (flip stride 1 + murid 0,40) dicatat poller | rank saat selesai < rank saat submit (403) → flip+murid ≥ terbaik lama |
| **17 Okt 07:01 (otomatis)** | submit **`r7v6a`** (S6F .5 separuh Raptor + murid 4 fold .3 + bobot heliosli) | rank saat selesai vs r7v3sr |
| **17 Okt ±14:30 (otomatis, bergerbang)** | `r7v7a` = resep `r7v6a` dgn murid 8 model | gold murid ≥0,930 & kaki terpakai |
| 16 Okt 07:01 & ±08:30 (otomatis) | submit **`r7v3s`** (S6F 0,35 di slot compact + murid 0,30) lalu **`r7v3sr`** (S6F 0,35 di separuh Raptor + murid 0,30); keduanya tanpa flip | struktur S6F mana yg menaikkan rank (log rank saat selesai, ±11:45 & ±13:15) + penanda auto-select dari user → struktur itu dipakai di `r7v5*` |
| 17 Okt 07:05 (otomatis) | latih murid r2a fold 2–3 + full-data fold 8/9 (alver), r2c fold 0–3 (nxtirfan) | gerbang gold per fold ≥0,90 (r2a) / ≥0,88 (r2c) di `run_after_train_17okt.sh` |
| 17 Okt 11:00–14:00 | dataset murid versi baru → `r7v5*` = komponen terbukti + murid 6 model (± r2c `fold4–7.pt`) → commit | GOLD58 murid naik (≥0,930) & `MURID_LEG`/`S6F_LEG used`; perkiraan run < ±7 j |
| 17–18 Okt | submit 2 varian/hari (agen) | pertahankan komponen yg tak menurunkan rank; buang yg menurunkan |
| 19–20 Okt | kandidat final; **variasi baru terakhir 20 Okt** | durasi run aman, tanpa fallback diam-diam |
| 21 Okt | agen merekomendasikan final A & B ke user | lihat 0.6 |
| 22 Okt (≤23:59 UTC) | **user memilih 2 final manual di UI**; agen verifikasi via `kaggle competitions submissions` | — |

### 0.6 Aturan memilih 2 final
- **A = resep terlengkap yang lulus semua uji**: R7 + r384/r224 bercermin + S6F + murid r2a (4 fold + 2 full-data, ± r2c) [+ flip stride 1 bila tidak menurunkan dan waktu run aman].
- **B = lindung nilai dengan risiko BERBEDA**: komponen publik terbukti + bobot murid lebih kecil, atau tanpa S6F bila S6F meragukan. Jangan dua salinan publik; jangan dua varian hampir identik (korelasi privat tinggi = sedikit tambahan peluang).
- Hanya submisi yang SUDAH terskor & tanpa tanda fallback; urutan skor tak-dibulatkan dibaca dari **penanda auto-select** (UI) + log rank `data/rank_watch.log`.

### 0.7 Risiko & mitigasi
| Risiko | Mitigasi |
|---|---|
| Run > 9 j (tanpa skor) | murid & S6F punya anggaran/guard sendiri (S6F selesai ≤ NB_T0+8,0 j atau dilewati); varian final diukur durasinya dari jam submit→selesai |
| Komponen diam-diam tidak jalan di run tersembunyi (log tak terbaca) | gerbang di commit-run (`GOLD58`, `MURID_LEG used`, `S6F_LEG used`); fallback = resep tanpa komponen itu (aman, bukan rusak) |
| Komponen merugikan di privat | uji per komponen 16–18 Okt; final B berbeda risiko; bobot ditetapkan a priori (tidak disetel ke papan publik) |
| Data eksternal (OAI di Raptor/S6F, fastMRI di R7) | **ketentuan host 14 Okt (Chris Carr, RSNA, thread 743416)**: dataset yg hanya butuh registrasi + persetujuan lisensi = memenuhi aturan; butuh perjanjian institusi = tidak. OAI (Login.gov + syarat NDA, tanpa institusi) & fastMRI → memenuhi. Host meminta transparansi: catat dataset eksternal semua komponen (siapkan teks pengungkapan bila diminta) |
| PC mati / reboot | unit systemd permanen + linger; Persistent=true menyusul saat boot; agen memeriksa `list-timers` tiap sesi |
| Kuota GPU latih habis | 17 Okt +30 j/akun (alver, nxtirfan); akun utama hanya commit-run |

### 0.8 Peluang jujur (15 Okt malam)
- Hanya salinan publik: **±5–15 %**. Komponen publik terbaik saja (≈ leoprovorov): **±30–45 %**. **+ murid kita menambah +0,0003–0,0006 nyata (δ total ≈ +0,0010): ±50–70 %**. Emas ≈0–2 %. 100 % tidak mungkin.

### 0.9 Yang hanya user bisa lakukan
1. Baca **penanda auto-select** di halaman *Submissions* (2 skor tak-dibulatkan tertinggi) dan kirim ke agen setelah tiap hari skor masuk.
2. Biarkan PC menyala 16–17 Okt (jadwal otomatis).
3. **22 Okt: pilih 2 final** sesuai rekomendasi 0.6 (agen menyiapkan daftar 21 Okt).

---
# ARSIP (rencana 10–12 Okt, era S6 — sejarah saja, jangan dieksekusi)

## A-1. Pembaruan 15 Okt malam (versi ringkas sebelum penulisan ulang)
- Papan 15 Okt 19:49 WIB (6.040 tim): ≥0,958 = 132; **≥0,957 = 327 → garis SILVER (rank 301) = 0,957**; blok tepat 0,956 = 941 tim; perunggu (rank 603) = 0,956. **Kita rank 398 (0,956)**.
- Pola terbukti publik: tim naik ke 0,957 dgn MENAMBAH MODEL beragam (bliverrigpdoge model pribadi 0,2; leoprovorov S6F 0,4 di slot compact). Menyetel bobot saja tidak menaikkan.
- Resep kita menuju final: R7 + r384/r224 bercermin + **S6F** (publik) + **murid r2a 4 fold + 2 full-data** (milik kita) ± flip stride 1 ± r2c. Uji 16 Okt: `r7v3s` (S6F + murid) & `r7v3c40` (murid 0,40); 15 Okt malam `r7v4c40` (flip + murid 0,40).
- Peluang silver (perkiraan jujur): hanya salinan publik ±15–25 %; + S6F & murid memberi +0,0005–0,001 nyata ±35–50 %; emas ≈0–2 %. 100 % tidak mungkin. Papan privat (70 %, ±900 studi) mengacak ulang blok salinan.

## A0. Pembaruan 12 Okt
- Papan 12 Okt 07:29 UTC (5.809 tim): silver rank 290 = **0,954**, perunggu rank 581 = **0,954** (keduanya di dalam dataran ±600 tim tepat 0,954), emas rank 21 = 0,962; ≥0,955 = 182; ≥0,956 = 119; kita rank 466. Garis silver bergerak 0,950 (8 Okt) → 0,954 (12 Okt).
- Notebook publik #1 = **S6-eq** heliosli (Raptor 0,5 + compact 0,5, rank sadar-gagal; bundel `r6cnxfsss` = compact-v1 V13) → tim heliosli 0,956; dibangun ulang `build_s6eq.py` (`s6eq`, `s6eqr` + reader goodpjw), 2 submisi menunggu slot GPU (`scripts/submit_watch.py`).
- Pembeda nyata ke ≥0,957: murid sendiri (ConvNeXt-T/EffNetV2-S 2.5D) dgn pseudo-label OOF; guru = **OOF compact** (4 fold hash `sha1(uid)%4`, gold tak dilatih; `build_compact_oof.py`). Rincian, gerbang, anggaran: `STRATEGI_0960.md`; bukti forum/kode: `reports/Scrape kompetisi 12 Okt 2026 - peta menuju silver.md`.

## A1. Perubahan besar 11 Okt
- **KEPUTUSAN USER (09:45 WIB): J1 merger & tindakan pesan support DIBATALKAN** — fokus naikkan skor sendiri dengan tim ini; semua submisi dari akun utama; arsip `docs/KANDIDAT_MERGER.md`/`docs/DRAF_PESAN_KAGGLE_SUPPORT.md` jangan dieksekusi.
- **S6 heliosli hidup lagi** (02:50 WIB): dataset `heliosli/rsna-knee-compact-v1` publik (V1–V13) + notebook publik "Dataset restored(...)" = **Raptor 0,7 + compact 0,3 = 0,954 satu klik** (thread 748141: "blend my model with yours"). **User men-submit salinannya (57051680, 02:56 WIB) → 0,954 COMPLETE ±06:40 WIB (rank 144 dari 5.709; run penuh ±3 j 45 mnt).**
- **Koreksi premis**: resep heliosli = Raptor + **compact** (bukan OF). OF fold0 belum pernah diukur di LB; prior ≥+0,001 di atas S6 ≤20 %.
- **Penyebab 57044355 = 0,950 (OF tidak jalan)** = OOM (induk Apex 18 GB + pekerja OF 8–10 GB); perbaikan v4.1 (OF dulu/terakhir, supervisor, `pin_memory=False`, `torch.load(mmap=True)`) lulus rehearsal; infrastrukturnya dipakai ulang di `build_s6plus.py`.
- **Papan**: kita rank 599 (0,950) — di bawah garis perunggu publik (rank 570, 0,950); silver rank 285 (0,951); dataran 0,954 akan membengkak (klon S6).

## A2. Sasaran dan ukuran (11 Okt)
- Silver = 5 % teratas di papan **privat** (70 % data uji) ≈ rank 285 dari 5.699 tim. Papan publik (30 %) hanya penunjuk (11 Okt 02:59 WIB): ≥0,960 = 31; ≥0,956 = 82; ≥0,954 = 141; ≥0,952 = 229; ≥0,951 = 367; =0,950 = 618 (kita rank 599).
- **100 % tidak bisa dijamin.** Klon S6 berimpit di ±0,954; urutan privat di antara klon ≈ undian + selisih kecil (ketidakdeterminan GPU), jadi perbaikan nyata ≥+0,001 atau anggota tim kuat yang mengubah peluang.

## A3. Dasar kuantitatif 11 Okt (simulasi `silver_prob_sim.py`; rentang = s regresi 0,3–0,7 × σ privat 0,003–0,005)
| Kondisi (skill sejati) | P(silver) | P(perunggu) |
|---|---|---|
| Apex 0,950 | 13–20 % | 25–41 % |
| + naik 0,002 | 25–39 % | 40–66 % |
| + naik 0,004 (≈ S6 0,954) | 40–67 % (tanpa pembengkakan klaster) | 58–85 % |
| tim publik 0,956–0,958 (merger) | 23–82 % | — |
- **Koreksi untuk S6**: bila ratusan klon berimpit di 0,954, P(silver | klon) ≈ (irisan klaster dgn 5 %) / ukuran klaster ≈ 25–45 % (klaster 400–800 tim), BUKAN 40–67 %. Perunggu ≈ 45–70 %.
- **Model "blok klon" (11 Okt)**: ratusan klon S6 memiliki prediksi (hampir) identik → skor privat bersama; blok bergerak sebagai satu unit relatif terhadap tim non-klon, dan di DALAM blok urutan ≈ acak (ketidakdeterminan fp16 GPU ~1e-5; bila persis sama, urutan = waktu submisi → klon TERLAMBAT seperti kita kalah). Varian yg BERBEDA dan sedikit LEBIH BAIK (δ>0) melompati seluruh blok; yg sedikit LEBIH BURUK (δ<0) jatuh di bawah seluruh blok (contoh: tri-hybrid jenillangaliya 0,952 < S6 0,954). Maka yg menentukan = TANDA δ, bukan besarnya: pilih perna dgn prior δ>0 tertinggi (reader goodpjw ≈60–75 %; OF ≈50 %; r224/dino-sejenis ≈ <40 %), hindari tambahan "demi keragaman". Aturan: final A = varian dgn LB publik ≥ klon (selisih berpasangan SE ±0,0005–0,001); final B = klon atau varian ke-2 (peluang minimal satu δ>0 naik). Peluang kasar: klon ≈ (irisan blok dgn 5 %)/ukuran blok ≈ 20–30 %; varian δ>0 ≈ P(δ>0) × P(blok di atas garis) ≈ 45–60 %.
- Dua final dgn galat privat tak berkorelasi menambah ±4–12 poin silver (`docs/PLAN_B.md` §1, §4); S6 + variannya berkorelasi tinggi (ρ≈0,95) → tambahan ≈ +3–6 poin.
- Catatan: trik "kirim 1−p" membuat tim kuat bisa bersembunyi di papan (forum 748011) → garis silver privat bisa lebih tinggi dari yg tampak.

## A4. Jalur 11 Okt (paralel; EV menurun)
### J0 — DASAR S6 (user sudah menjalankan; agen memantau)
- **HASIL: 57051680 = 0,954 (S6 terbukti jalan di run penilaian; kita rank 144; 49 tim tepat di 0,954 → dataran 0,954 mulai rank 102).** Rincian: salinan S6 (Raptor ±0,5 j + compact ±3,5 j [GPU-bound, 2 pekerja ±18–20 dtk/studi/pekerja, terukur pada pseudo-uji 160 studi: 80/80 per pekerja, 0 fallback] → total ±4–4,5 j). Risiko: kontroler compact menolak (ERROR) bila >3 % fallback (OOM/pekerja mati/DEADLINE 8,3 j). Aturan: bila ≈0,954 → final A; bila ≤0,952/ERROR → diagnosis (sonda `rsna-knee-s6probe` + rehearsal), gunakan mode `lenient`.
### J1 — MERGER — **DIBATALKAN (perintah user 11 Okt 09:45; fokus naikkan skor sendiri)** — isi di bawah = arsip saja, jangan dieksekusi
- Rules 5.c: sah bila (i) gabungan ≤5 anggota, (ii) submisi gabungan ≤ jatah satu tim saat permintaan (11 Okt = 335; kita 109 → mitra ≤ ±225), (iii) sebelum batas merger. Rules 5.d/6.a: dilarang berbagi privat sebelum merger.
- Target baru (`docs/KANDIDAT_MERGER.md` §3b): Kelompok B (15 tim ≥0,955 ≤30 submisi, mis. MURAD 0,962/19, NMarkS 0,961/30, "吃白饭的大肥鱼" 0,960/**1 submisi**, UEHA, DoyMed, jjakimoto, madmax0404, LP …), Kelompok C (mencari rekan: Tanuki_boosting, Prateek Grover, Nikita Ageev, heliosli solo 0,954). Pemegang bundel (Kelompok A lama) tak istimewa lagi karena S6 publik.
### J2 — TUAS TEKNIS di atas S6 (semuanya EV kecil: +0,000–0,001; LB resolusi 0,001)
1. **S6 + reader** (`python3 build_s6plus.py name=s6reader reader=1`): perna Raptor = fusi Apex (reader goodpjw, bobot per-temuan); +0,5 j; risiko rendah; prior +0,0005.
2. **S6 + OF terakhir** (`name=s6plus reader=1 of=1 wr=0.55 wc=0.30 wo=0.15`): OF dgn supervisor v4.1 hanya memakai waktu sisa (STOP 7,9 j/KILL 8,25 j sejak NB_T0; coverage ±60–70 %); prior ≥+0,001 ≤20 %; rehearsal `s6plusr` memverifikasi rantai penuh.
3. **Bagging bundel heliosli lain** (V4 ofl_ft8, V7 cnx_s5 0,951, V8 dino_s1, V9 dual): biaya ±3,5–4,5 j GPU/bundel (S6 sudah ±4 j) → hanya bila stride dikurangi; prior +0,0005–0,001.
4. **Final bersih** (tanpa Raptor/OAI): compact + reader; kualitas belum terukur (±0,950–0,952) — asuransi bila host melarang OAI; nilai medali nyata hanya dlm skenario itu.
5. DITUTUP (jangan ulangi): varian r224/bobot/mirror (0,950), MaVIT (0,949), CoAtNet resgated, pseudo-label/anak-OAI (≈ induk), kompilator aturan heliosli, "Tri-Reader 0,954" (rabari/jcgweb/sameerk; keluarga Apex+r224 0,950), klon DINOsaur.
### J3 — FINAL TAHAN GUNCANGAN (21–22 Okt; pilih MANUAL & verifikasi via API; auto-fill §3.18.c)
- Final A = S6 (klon user atau varian terbaik); Final B = varian paling berbeda yg masih kuat (S6+reader/OF, atau bersih). Pilih dari submisi yg SUDAH terskor; variasi baru terakhir ≤20 Okt.
### J4 — KEPATUHAN (melindungi medali)
- Akun kedua/ketiga: Rules 5.a/5.b — semua submisi dari akun utama; pesan ke support = DIBATALKAN (user 11 Okt); submisi tetap akun utama.
- OAI/eksternal: risiko KOLEKTIF seluruh klaster Raptor/S6 (host diam di 743416/747455/744056); Rules 2.6.b: model/data eksternal boleh kecuali dilarang host; 2.8/3.14.a hanya menyangkut kelayakan hadiah. Pertahankan satu final dgn risiko berbeda bila murah.

## A5. Garis waktu dan gerbang 11 Okt (WIB)
| Waktu | Aksi | Gerbang / keputusan |
|---|---|---|
| 11 Okt 03:00–07:00 | rehearsal `apexof4r3` (OF 4 pekerja) dan `s6plusr` (reader+compact+OF); update dokumen | agen tidak mensubmit sebelum 07:00 (jatah hari-ini habis) |
| 11 Okt 07:00+ | jatah baru 5/hari (agen ≤2): submit varian S6+ terpilih; baca skor klon S6 (57051680) ±07:00–08:30 | bandingkan: klon S6 vs varian |
| 11–13 Okt | ~~merger & support~~ DIBATALKAN (user 11 Okt) — fokus varian S6+ & monitor skor | — |
| 13–15 Okt | semua kandidat ter-push sebelum 15 Okt (J1 dibatalkan) | 15 Okt 23:59 UTC = 16 Okt 06:59 WIB |
| 16–20 Okt | iterasi bergerbang (agen ≤2/hari); GPU baru 30 j 17 Okt 07:00 WIB; pantau publikasi baru (heliosli S7?) | variasi TERAKHIR 20 Okt |
| 21–22 Okt | cek ulang 743416; pilih 2 final manual; verifikasi via `kaggle competitions submissions` | 22 Okt 23:59 UTC |

## A6. Anggaran (11 Okt)
- GPU akun utama: ±20 j tersisa (17 Okt reset +30 j). Run penilaian tak memakai kuota GPU. Submisi tim: 5/hari (reset 07:00 WIB), agen ≤2, sisakan 2–3 utk user & rekan.
- S6 ±4–4,5 j dari batas 9 j; sisa ±4,5 j dipakai perna tambahan (reader 0,5 j; OF terakhir ±3,4 j).
- Disk /kaggle/working ±19 GB efektif; /tmp & overlay besar.

## A7. Risiko utama (11 Okt)
| Risiko | Mitigasi |
|---|---|
| S6 gagal di run penilaian (RAM/DEADLINE/fallback >3 % → ERROR) | sonda/rehearsal dgn log terbaca + monitor RAM; mode `lenient`; klon user = deteksi dini |
| Klaster 0,954 membengkak → silver = undian | varian dgn informasi baru; final B beda mekanisme |
| Host melarang OAI/model eksternal | final bersih (compact + reader); keputusan 21–22 Okt setelah cek ulang |
| Papan privat mengacak dataran | dua final; jangan menyetel ke papan publik; Δ<0,002 = noise |
| Diskualifikasi akun ganda | J4; semua submisi dari akun utama |
| Tim kuat tersembunyi (1−p) | tidak bisa dikendalikan; targetkan skill sejati, bukan hanya publik |
