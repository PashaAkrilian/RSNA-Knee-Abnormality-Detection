# PLAN B — arsip cadangan historis (asal: Apex + OrthoFoundation 57044355)

> Status 19 Okt 2026: dokumen ini bukan rencana aktif. Ikuti `STRATEGI_PERAK.md` §0. Saat ini LB 0,958 rank 467/6.187, tiga submission terbaru masih PENDING, dan dua kernel murid r4 menunggu alokasi GPU di alver. Detail ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`.
> Disusun 10 Okt 2026 ~22:05 WIB (sesi Claude Code; hanya akun utama). Pelengkap `STRATEGI_PERAK.md`. Skor 57044355 keluar ±04:00 WIB 11 Okt; dokumen ini sudah memuat tindakan utk tiap kemungkinan hasil.
> Papan acuan: 10 Okt 21:43 WIB (5.673 tim; ≥0,951 = 352; ≥0,952 = 221; =0,950 = 582; kita rank 551; 5 % ≈ rank 284; 10 % ≈ rank 567).

## 0000. UPDATE 15 Okt malam — cadangan bila S6F/murid merugikan
- Rencana aktif: `STRATEGI_0960.md` §0.3 (R7 + r384/r224 cermin + S6F + murid r2a). Cadangan berurutan bila uji 16–18 Okt menunjukkan komponen merugikan (rank TURUN saat selesai, atau skor tampil < 0,956):
  1. Murid merugikan → final A = `r7r224m`-keluarga + S6F (tanpa murid); B = `r7r224m` (57198411).
  2. S6F merugikan / tak sempat jalan (run > ±7 j) → A = r224m + murid (`r7v3c40`/`r7v4c40`); B = `r7r224m`.
  3. Flip stride 1 merugikan → semua varian final tanpa flip (hemat ±1 j run).
- Final B selalu berbeda risiko dari A (dua final berkorelasi tinggi hanya menambah +3–6 poin peluang).

## 000. UPDATE 12 Okt — rencana aktif pindah ke `STRATEGI_0960.md`
- Varian S6 lama semuanya ≤0,954; basis baru = **S6-eq** (notebook publik #1 heliosli, 0,5/0,5, rank sadar-gagal; `build_s6eq.py`). Cadangan/hedge final B tetap: klon S6 0,954 (57051680) atau S6-eq tanpa murid.
- Tuas berikut = murid pseudo-label dengan guru **OOF compact** (`build_compact_oof.py`), bukan perna tambahan berlabel sama. Plan B bila murid gagal: S6-eq + (opsional) ckpt publik `jessky/rsna-knee-local-cv-cnx50` setelah dinilai di emas-58.

## 00. UPDATE 11 Okt 03:30 WIB — S6 HIDUP LAGI: strategi kini BERPUSAT S6 (mengganti prioritas §0–§3 di bawah)
- **Fakta**: 02:50 WIB heliosli memulihkan dataset `heliosli/rsna-knee-compact-v1` (publik; V10 = V13; semua versi V1–V13 publik) dan menerbitkan ulang notebook S6 publik (**Raptor 0,7 + compact 0,3 = 0,954 satu klik**; thread 748141 "feel free to blend my model with yours"). **User sudah men-submit salinan S6: 57051680 (02:56 WIB, PENDING)**; tahap compact GPU-bound ±25 dtk/studi/pekerja (2 pekerja) → ±4–5 j total → hasil ±07:30–08:30 WIB.
- **Papan 02:59 WIB**: 5.699 tim; kita rank 599 (0,950) = DI BAWAH garis perunggu publik (rank 570); perak rank 285 (0,951); ≥0,954 = 141. Dataran 0,950 = 618 tim; dataran 0,954 akan membengkak.
- **KOREKSI PREMIS OF**: "Raptor 0,7 + OF 0,3 = 0,951–0,954" SALAH — itu Raptor + COMPACT (S5 0,951; S6 0,954); notebook heliosli hanya memakai KODE dataset OF, bukan bobot `best.pt`. Nilai LB OF fold0 tidak pernah diukur; prior kontribusi ≥+0,001 di atas S6 ≤20 % (OF validasi lemah 0,9318, epoch 7/16). Tuas OF turun peringkat (infrastruktur v4.1 tetap siap bila perlu).
- **Model blok klon**: klon S6 berimpit (skor privat bersama, urutan dalam blok ≈ acak/waktu); varian berbeda dgn δ>0 melompati seluruh blok, δ<0 jatuh di bawahnya → pilih perna dgn prior δ>0 tertinggi (reader > OF > r224/dino-sejenis); lihat `STRATEGI_PERAK.md` §2.
- **Pohon keputusan baru (menggantikan §2 untuk submisi berikut)**:
  | Hasil klon S6 (57051680) | Tindakan |
  |---|---|
  | ≈0,954 | final A = S6; buat final B = S6+reader (`s6reader`) atau S6+OF (`s6plus`); perkuat varian (merger DIBATALKAN 11 Okt) |
  | ≤0,952 (S6 gagal/tidak reproduksi) | selidiki log (OOM/fallback >3 % → ERROR?), pakai `lenient`; kembali ke Apex-keluarga (merger DIBATALKAN) |
  | ERROR/Timeout | diagnosis lewat sonda `rsna-knee-s6probe`/rehearsal; perbaiki (RAM/waktu) |
- **Bahan siap (build_s6plus.py; semua lulus kompilasi)**: `s6reader` (reader goodpjw difusikan ke perna Raptor dgn bobot per-temuan Apex; tahap compact "lenient" tidak menjatuhkan notebook), `s6plus` (+ perna OF dgn supervisor v4.1 sebagai tahap TERAKHIR, STOP/KILL absolut 7,9/8,25 j sejak NB_T0 — hanya memakai waktu sisa; bobot 0,55/0,30/0,15), `s6plusr` (rehearsal pseudo-uji 160 studi dgn log terbaca + monitor RAM).
- **Zoologi bundel heliosli (semua publik)**: V4 `ofl_ft8` (OrthoFoundation-L fine-tune), V7 cnx_s5 (0,951), V8 `dino_s1` (DINOv3-ViT-B), V9 dual (convnext+dino), V10 cnx_s6 (0,954) — kandidat bagging; kendala: tiap bundel ≈3,5–4,5 j GPU di stride 1 (S6 sudah memakai ±4 j dari 9 j).
- **Rehearsal `s6cleanr` (11 Okt 04:20 WIB; compact + reader, tanpa Raptor/OAI, 160 studi pseudo-uji)**: LULUS — reader 211 dtk; compact `CONTROLLER_DONE PASS` 160/160 (0 fallback; decode 6,4 + render 2,7 + GPU 18,8 dtk/studi; ±610 jendela/studi) dlm 1.812 dtk = 11,3 dtk/studi agregat → 1.322 studi ≈4,2 j (sonda sebelumnya 9,6 dtk/studi ≈3,5 j; variasi ±20 % antar mesin); `NOTEBOOK_COMPLETE` blend bersih; MemAvailable min 15,6 GB (stabil ±16–17 GB selama compact: tak ada kebocoran). S6 penuh ≈4,5–5,3 j.
- **Rehearsal `s6plusr` (11 Okt 04:35 WIB; S6 + reader ×1,0 + OF terakhir 0,15, pseudo-uji 160, OF STOP 0,4 j sejak awal tahap)**: LULUS end-to-end — Raptor 230 dtk, reader 153 dtk, compact PASS 160/160 (1.452 dtk; decode 7,6 + render 2,9 + GPU 16,6 dtk/studi), OF 4 pekerja: 104/160 (coverage 0,65; 57 dtk/studi/pekerja; 0 pekerja mati), blend 3 perna `NOTEBOOK_COMPLETE` (total 3.522 dtk), MemAvailable min 16,2 GB. Produksi `s6plus`: OF mendapat ±3,3 j (sisa waktu hingga STOP 7,9 j) ≈ 750 studi ≈ 55–60 % coverage.
- **Final bersih (asuransi OAI)**: compact S6 + reader (tanpa Raptor/OAI) bila host memutuskan OAI terlarang; kualitas belum terukur (kemungkinan ±0,950–0,952).
- **Merger** tetap tuas terbesar; S6 membuat "pemegang bundel" (Kelompok A) tak istimewa lagi. Target baru: tim independen ≥0,956 ≤30 submisi (Kelompok B) + tim yg mencari rekan (Tanuki_boosting rank 37 0,955; Prateek Grover 0,950; Nikita Ageev 0,949 label LLM) + heliosli sendiri (solo 0,954; 29 submisi).

## 0. UPDATE 10 Okt 23:10 WIB — HASIL 57044355 = 0,950 TETAPI TIDAK ADA VONIS UNTUK OF
- 57044355 (Apex 0,70 + OF fold0 0,30, apexof2 v3) **COMPLETE 22:45 WIB, publik 0,950** — hanya **±1 j 35 mnt** setelah submit (21:10). Run Apex-only/varian lain butuh 1,0–1,6 j (57012143 1j37m, 57015077 1j08m, 57025414 1j03m), sedangkan OF utk 1.322 studi butuh ±6–7 j (uji skala: 194 studi/jam). ⇒ **kaki OF hampir pasti TIDAK jalan di run penilaian** (liputan <15 % → blend jatuh ke Apex murni; skor identik 0,950). Log run penilaian tak terbaca, jadi penyebab pasti belum diketahui.
- Dugaan utama: **RAM** — pada v3 OF dijalankan SESUDAH Apex; proses induk (notebook) masih memegang memori Apex + 4 pekerja OF (±5–6 GB tiap pekerja) di mesin 29 GB; catatan nartaa di kode Raptor: "On the full hidden test it drove a system-RAM OOM". Uji stub (3 studi) dan uji skala (OF saja) tak bisa menunjukkan masalah ini. Dugaan lain: perbedaan lingkungan run penilaian (jumlah GPU/CPU), pekerja gagal cepat.
- **Akibat bagi pohon keputusan (§2)**: baris "0,950 = netral" TIDAK berlaku untuk 57044355. Gerbang G1 ditahan: OF belum teruji; jangan menutup OF. Aturan baru: skor submisi OF hanya dianggap vonis bila **durasi submisi ≥ 5 j** (OF benar-benar jalan) ATAU run memakai mode `loud` (gagal KERAS → status ERROR bila liputan OF <15 %).
- **Perbaikan (v4, `of_stage_v4.py` + `build_apexof4.py`)**: OF dijalankan PERTAMA sebelum sel Apex (proses induk kecil, tanpa torch), antrean potongan 8 studi dgn klaim atomik, supervisor (reaping, pelepasan klaim yatim, restart anggaran 8, turunkan konkurensi bila rc −9), daur-ulang pekerja tiap 12 potongan/RSS>9 GB, `oom_score_adj` pekerja 500, keluaran per potongan atomik, pemutus lunak 6,0 j & keras 6,4 j (Apex ±1 j sesudahnya → total ≤ 7,9 j). Uji lokal (stub) skenario normal+injeksi-kill+daur-ulang, potongan racun, crash saat impor, berhenti lunak: LULUS (`test_of_v4_local.py`).
- **HASIL REHEARSAL `apexof4r` (10 Okt 23:47 WIB; log terbaca via Chrome + `kernels output --file-pattern '_leg_of_w.*log'`)**: mekanisme v4 jalan end-to-end (OF dulu → Apex 96 studi 135 dtk + reader 143 dtk → blend `[BLEND SUCCESS]` coverage 0,50, corr(apex,OF)=0,865). **Temuan kunci: RSS tiap pekerja OF = 8–10 GB** (4,0 GB @5 mnt → 7,3–8,5 GB akhir potongan-1 → 10,6 GB akhir potongan-2) di mesin 30 GB / 4 vCPU. **Dua pekerja mati serentak (rc −9, OOM nyata, bukan injeksi) pada ±17 mnt dgn hanya 3 pekerja hidup dan proses induk 0,2 GB** ⇒ dugaan RAM TERBUKTI: v3 (4 pekerja + induk Apex besar) pasti di-OOM di run penilaian ⇒ liputan <15 % ⇒ jatuh ke Apex. Kecepatan: 1 pekerja sendirian 44–52 dtk/studi; 3–4 pekerja bersamaan 62–80 dtk/studi/pekerja (±200 studi/jam agregat bila tak ada tekanan memori). Headroom cgroup naif menyesatkan (menyertakan page-cache) → guard memakai MemAvailable/headroom efektif.
- **HASIL DIAGNOSTIK `apexof4d` (11 Okt 00:27 WIB; urutan LAMA = OF sesudah Apex, pseudo-uji 1.322 studi) — PENYEBAB 57044355 TERKONFIRMASI**: setelah tahap Apex selesai, **proses induk notebook memegang RSS 18,0 GB** (MemAvailable tinggal 12,0 GB, headroom cgroup 3,9 GB); satu pekerja OF (puncak ±9,6 GB) mati OOM (rc −9) pada +0,31 j dan koverasi akhir 0,012 (16/1.322) ⇒ pada v3 (4 pekerja, 8–10 GB masing-masing, hanya ±12 GB bebas) semua pekerja di-OOM, liputan <15 %, blend jatuh ke Apex murni ⇒ skor 0,950 identik. Waktu Apex pada 1.322 studi (commit-run): Raptor 1.567 dtk (pre 1.477 dtk, CPU-bound) + reader ConvNeXt 1.755 dtk = **3.323 dtk ≈ 0,92 j**.
- **PROFIL MEMORI `ofmem1/ofmem3` (1 pekerja, 8 studi/potongan; 11 Okt 00:37–01:25 WIB)**: RSS puncak potongan-1 / potongan-2 (GB) · dtk/studi: base 8,8 / 11,7 · 48,1; mallocthr 8,7 / 10,5 · 52,4; chunk4 6,5 / 10,1 (tumbuh +1,8 GB per potongan) · 41; workers=0 9,2 / 12,1 · 47,7; tcmalloc 9,7 / 12,9 · 42,3 (lebih cepat 12 %, +1 GB); **nopin 6,3 / 9,0 · 44,9**; **nopin+mmap 5,1 / 6,5 · 38,7 (anon 2,3 GB)**. Penyebab lonjakan: (a) `pin_memory=True` pada DataLoader predict → host-allocator CUDA menahan ±2,5 GB shmem per pekerja (diatasi dgn `dl.pin_memory=False`), (b) `torch.load` checkpoint ViT-L (1,2 GB, dimuat 2× di run_inference+predict) → `mmap=True` memindahkannya ke page-cache bersama, (c) anon tumbuh +2–3 GB per potongan (fragmentasi) → daur-ulang tiap potongan. **CPU: tiap pekerja ≈ 1 core penuh (cpu_self ≈ wall)** → 4 vCPU = 4 pekerja. Fase per potongan 8 studi (±385 dtk): prepare 87 dtk (23 %), geometri 41 dtk (11 %), anatomi 178 dtk (46 %), predict ViT-L 64 dtk (17 %).
- **REHEARSAL r2 `apexof4r2` (11 Okt 02:47 WIB; v4.1: nopin+mmap, recycle per potongan, 4 pekerja, kill-injection +6 mnt, 240 studi pseudo-uji)**: LULUS — injeksi kill ditangani (potongan dilepas & diulang; target turun ke 3 sesuai desain), **tidak ada OOM nyata**, RSS pekerja di akhir potongan 3,0–4,8 GB (anon 2,0–3,9), **MemAvailable min 19,2 GB / median 22,4 GB**, 26 spawn, 0 potongan bad, 25/30 potongan selesai dlm 3.713 dtk (**194 studi/jam dgn 3 pekerja**; per pekerja 31–73, rata-rata ±52 dtk/studi) → koverasi 0,833 pd batas waktu; Apex 240 studi 320 dtk + reader 293 dtk; `[BLEND SUCCESS]` coverage 0,833, corr(apex,OF)=0,775. Ekstrapolasi produksi: 4 pekerja ≈ 235–250 studi/jam ⇒ 1.322 studi ≈ 5,5 j (3 pekerja ≈ 6,8 j). Rehearsal r3 (4 pekerja tanpa injeksi) memverifikasi ini.
- **REHEARSAL r3 `apexof4r3` (11 Okt 03:45 WIB; v4.1, 4 pekerja, TANPA injeksi, 200 studi pseudo-uji, stop 0,75 j)**: LULUS — 23/25 potongan dlm 2.899 dtk = **228 studi/jam dgn 4 pekerja** (rata-rata 54 dtk/studi/pekerja), 0 pekerja mati, 0 potongan bad, **MemAvailable min 13,8 GB / median 20,9 GB**; Apex 200 studi + reader + blend OK (coverage 0,92; corr(apex,OF)=0,736). Produksi: 1.322 studi ≈ 5,8 j dgn 4 pekerja.
- **Desain v4.1 (berdasar RAM)**: ≤3 pekerja (default `nproc=3`), daur-ulang tiap potongan (`maxchunks=1`, `rssmax=8`), spawn hanya bila MemAvailable ≥ 9,5 GB, uji profil memori `rsna-knee-ofmem1/2` (chunk 4, malloc thresholds, tcmalloc, `workers=0`, `torch.load(mmap=True)`) utk menurunkan RSS agar 4 pekerja muat. Koverasi target ±85–95 % dlm 6,2 j (≈190–200 studi/jam); Apex sesudahnya ±1,1 j (Raptor ±29 mnt + reader ±22–33 mnt per ekstrapolasi 96 studi) ⇒ total ≈7,8 j.
- **Uji bertingkat (commit-run, log terbaca; GPU akun utama, tanpa submisi)**: `rsna-knee-apexof4r` (OF dulu, pseudo-teruji 96 studi dari TRAIN, injeksi kill, daur-ulang tiap 2 potongan) dan `rsna-knee-apexof4d` (diagnostik: urutan LAMA OF sesudah Apex pada pseudo-uji 1.322 studi — mereproduksi kondisi 57044355 dgn log). Lalu **percobaan `apexof4t` (loud=1) 11 Okt setelah 07:00 WIB** (1 submisi agen); bila OF jalan ±7 j ⇒ skor = vonis nyata; bila gagal ⇒ ERROR cepat (±1,5 j) ⇒ kita tahu lingkungan penilaian berbeda.

## 1. Peta peluang (simulasi `silver_prob_sim.py`, model regresi-ke-rata-rata + noise; rentang = [s=0,3/σ=0,003 | s=0,5/σ=0,004 | s=0,7/σ=0,005])
| Skill sejati (publik 0,950 + naik δ) | P(perak) | P(PERUNGGU) |
|---|---|---|
| δ = 0 (Apex sekarang) | 13–18 % (tengah 18 %) | **25–41 % (tengah 35 %)** |
| +0,001 | 19–27 % | 33–53 % |
| +0,002 | 25–39 % | 40–66 % |
| +0,003 | 33–52 % | 50–77 % |
| +0,004 | 42–64 % | 58–85 % |
- **KOREKSI**: perunggu BUKAN 85–95 % lagi. Dataran 0,950 kini 582 tim dan garis perunggu (top 10 % ≈ rank 567) tepat di ujungnya (kita rank 551) → perunggu ≈ koin yg condong kalah bila tak ada perbaikan. Setiap +0,002 sejati ≈ +15–25 poin perunggu.
- **Dua final menguntungkan bila galat privatnya TIDAK berkorelasi** (maks dari 2 tarikan): ρ=0,9 → perak 17→21 %, perunggu 34→41 %; ρ=0,8 → 22 % / 42 %; ρ=0,6 → 26 % / 48 %; ρ=0,4 → 29 % / 51 %. Murah: pilih 2 final dgn skor publik setara tetapi mekanisme berbeda.

## 2. Pohon keputusan berdasar skor 57044355 (Apex 0,70 + OF fold0 0,30)
| Hasil | Arti | Tindakan (agen) |
|---|---|---|
| **≥0,953** | OF membawa pengetahuan nyata | final A = Apex+OF; final B = Apex+OF bobot 0,45 ATAU Apex (dekorelasi); fokus varian (merger DIBATALKAN) |
| **0,952** | kandidat aman-tipis | A = Apex+OF; B = Apex; boleh 1 varian bobot (0,45) bila jatah (merger DIBATALKAN) |
| **0,951** | cadangan (lotre) | A/B = {Apex+OF, Apex} (dekorelasi); perkuat varian & dekorelasi (merger DIBATALKAN) |
| **0,950** | netral (OF tak menambah) | OF = hedge dekorelasi saja; finals {Apex, Apex+OF}; seluruh tenaga ke varian & final (merger DIBATALKAN) |
| **≤0,949** | OF merugikan | buang OF; finals {Apex, varian Apex 0,950 (mis. fork r224/rabari)}; merger |
| **Timeout/ERROR lagi** | pipeline OF gagal di skala | JANGAN submit ulang tanpa log; kembali ke Apex-only; merger |
Aturan pakai: finals dipilih dari submisi yang SUDAH terskor (jangan mengandalkan run baru pada 22 Okt). Bila ada dua kandidat setara, pilih pasangan beda mekanisme.

## 3. Jalur B1 — MERGER dengan pemegang bundel / pipeline kuat (aksi USER; batas 15 Okt 23:59 UTC)
**Mengapa tepat.** Bundel `heliosli/rsna-knee-compact-v1` diunduh ±**1.974** kali sebelum ditarik (9 Okt 13:50 UTC). Notebook 0,951 publik memakai SALINAN PRIVAT-nya: kode `jenillangaliya/dual-stage-sota-ensemble` memuat konstanta `C_SLUG='rsna-knee-compact-v1'`, `W_SLUG='rsna-knee-ofl-ssl-weights'`, `BUNDLE_ID='c7ebab1eb953f1a6'`, `CODE_SLUG='rsna-orthofoundation-fold0-epoch7-20261004'`; sumber datasetnya `['', '', nartaa]` (dua kosong = privat). `kiyoshiohno/template` = peluncur yg menjalankan `rsna_knee_pipeline.py` dari dataset privat; `matterhorn3838/V2` & `naganumajun/*` punya satu sumber kosong.
**Isi dataset privat itu** (dari kode; "diduga" = kesimpulan):
- `rsna-knee-compact-v1` (bundel heliosli): `manifest.json` (kunci: `backbone`, `size`/`crop` 352 px, `slices`, `phys_mm` 3,5 mm, `views`, `eval_stride`, `models`, `state_spec`, `side_model` + `side_threshold/side_max_x/side_position_mm`, `bundle_id`), **4 ckpt fold `E_lrprior_s0_f0..f3.pt` (ConvNeXt-Small 2.5D)**, `state_spec` (prior report-state), model-sisi (verifikasi lateralitas), `pca.npz`/`feature_norm.npz`/`heads` (arm probe).
- `rsna-knee-ofl-ssl-weights`: bobot OrthoFoundation-L SSL (`weight_parts.json` + `weights.part*`, ±1,2 GB). **Catatan: bobot dasar OF-L juga PUBLIK** (`omkars5/orthofoundation` → `OrthoFoundation-L.pth` 1,21 GB; juga pritom2357, tsuyu122, leogamertetudo).
- Bundel "Takakura" (naganumajun): model 384 px "corrected-laterality" fold0 (`takakura-code-competition-inference-v1`) → arm tambahan di V2/V7.
- Tim 0,955–0,964 (solo/kecil): TIDAK mempublikasikan apa pun; dugaan: ckpt fold sendiri + tabel pseudo-label/soft-label + (mungkin) backbone berpengetahuan eksternal.
**Kelompok A — pemegang bundel (papan 10 Okt ~21:45 WIB; semua ≤2 anggota):**
| rank | skor | tim | pengguna | submisi | catatan |
|---|---|---|---|---|---|
| 112 | 0,954 | Heliosli | heliosli | 29 | SUMBER bundel (S6 0,954, bundel `209a32c8` 4-fold) |
| 128 | 0,954 | qtx202 | xianhan | 26 | keluarga fast-parent (DINO+RadImageNet+Raptor) |
| 133 | 0,954 | Netram Faran | netramfaran, rabari9999 | 63 | 2 anggota |
| 210 | 0,952 | Hackers | jenillangaliya, ritesh381 | 14 | notebook 0,951 pakai salinan bundel |
| 229 | 0,951 | samanyu1808 | samanyu1808 | 6 | blend3 (bundel) |
| 283 | 0,951 | Дворкин Е. В. | evgendvorkin | 19 | versia-7 (bundel) |
| 294 | 0,951 | kiyoshiohno (kiyoc) | kiyoshiohno | 14 | Template (logika privat + bundel) |
Tidak bisa digabung: `mattiaangeli` (282 submisi → gabungan > batas).
**Kelompok B — pipeline kuat sendiri, sedikit submisi (bukan sondir LB), ≤2 anggota, 0,955–0,958:** colum2131 (rank 48; 45 sub), monnu (52; 55), UEHA/triumphtrump (56; 22), jjakimoto (62; **6**), madmax0404 (64; 14), DoyMed (60; 2 anggota; 21), LP/lihu6666 (65; 18), Yakup Mert Karasulu (70; 21), LI YIZHOU (75; 31), wishmeluck (78; 26), Ray/faykudbq (86; 20), Robson (90; 16), dst. (daftar lengkap + profil: `docs/KANDIDAT_MERGER.md`).
**Prioritas pendekatan:** (1) Kelompok A dgn skor ≥0,952 & submisi sedikit (heliosli, xianhan, jenillangaliya/ritesh381) — gabungan "Apex + bundel compact + reader + OF fold0" ≈ resep 0,954+; (2) Kelompok B dgn <30 submisi (pipeline kuat, bukan probing); (3) sisanya. Kirim banyak permintaan sekaligus; jangan terima dua.
**Pitch yang jujur:** kita membawa (a) ckpt OAI nartaa (pengetahuan label berbeda dari model berbasis laporan), (b) reader goodpjw, (c) cabang OrthoFoundation fold0 (skor pada 57044355), (d) GPU 30 j/minggu, (e) pipeline cepat (<1 j). Hindari berbagi kode/data sebelum merger resmi (Rules 5.d/6.a).
**Risiko:** menggabungkan 3 akun kita → ukuran gabungan ≤5; selesaikan kepastian akun kedua ke Kaggle dulu; bundel heliosli ditarik pemiliknya (kemungkinan alasan kepatuhan data eksternal) → perlakukan sbg risiko yg sama dgn ckpt nartaa; pastikan satu final bersih.

## 4. Jalur B2 — DUA FINAL TERDEKORELASI (biaya nol)
- Syarat: skor publik setara (±0,001), mekanisme beda: Apex (nartaa+reader) vs Apex+OF (+ pengetahuan eksternal) vs keluarga lain (fork r224/rabari 0,950). Hindari dua titik bobot dari blend yg sama.
- Estimasi manfaat: +4–12 poin perak, +7–17 poin perunggu (§1). Final dipilih MANUAL di UI (auto-fill §3.18.c), verifikasi via daftar submisi.

## 5. Jalur B3 — tuas teknis cadangan (diurut EV; semua murah-menengah)
| # | Tuas | Bukti/EV | Status |
|---|---|---|---|
| T1 | Varian bobot OF (0,45) / OF hanya utk temuan tertentu | hanya bila OF 0,951–0,952; Δ≈+0,0005 | siapkan builder (`wo=`), 1 submisi |
| T2 | **Hipotesis lateralitas** (model "corrected-laterality" Takakura; `side_model` heliosli; `decide_study_side` di kode OF): kesalahan sisi lutut membalik medial↔lateral (6 label) | belum terukur; sumber gain S6 yg mungkin; EV ±0,001–0,003 bila ada studi uji tertukar | riset lanjut: ukur kesepakatan sisi (tag DICOM vs gambar) pada train |
| T3 | Probe fitur beku OF-L (bobot dasar PUBLIK + embedding publik `prithvirajrp/...raw-embeddings` 2,9 GB utk train) | EV rendah: fold0 yg di-fine-tune lebih kuat; bila fold0 gagal probe hampir pasti gagal | TIDAK dikerjakan kecuali OF fold0 ≥0,952 |
| T4 | Rekonstruksi compact heliosli sendiri (`compact_train.py` tertanam di V2 + OF-L publik) | label laporan sama → ±0 (terbukti reader/MaVIT); `state_spec`/bundel hilang | ditutup |
| T5 | OAI mentah | tak layak <10 hari (TB data, label OAI→12 temuan, akun NDA user) | ditutup |
| T6 | Pseudo-label/anak-OAI/mirror | terukur ≈ induk | ditutup |

## 6. Perlindungan perunggu (jangan kehilangan lantai)
- Selalu simpan satu final berbasis Apex yang sudah terskor 0,950. Jangan menjadikan run OF baru satu-satunya final (risiko Timeout; selalu pilih dari submisi terskor).
- Perunggu sensitif thd perbaikan kecil (+0,002 ≈ +15–25 poin) → dua final terdekorelasi + merger adalah pelindung terbaik.

## 7. Jadwal & pemicu
| Waktu (WIB) | Pemicu | Tindakan |
|---|---|---|
| 11 Okt ±04:00 | skor 57044355 keluar | agen: jalankan pohon keputusan §2; perbarui MEMORY/HANDOFF/laporan; usulkan final ke user |
| 11–14 Okt | user mengirim permintaan merger | agen: segarkan daftar kandidat tiap hari (`scan`/papan), siapkan notebook gabungan setelah merger |
| 14 Okt malam | keputusan akhir merger | bila ada mitra → protokol merger (rakit ensembel, gerbang bobot); bila tidak → Plan B2/B3 saja |
| 15 Okt 23:59 UTC | batas merger/entry | semua kandidat ter-push |
| 16–20 Okt | iterasi (agen ≤2 submisi/hari) | hanya varian beralasan; variasi TERAKHIR 20 Okt |
| 21–22 Okt | pilih 2 final manual | dari submisi terskor; verifikasi via daftar |

## 8. Risiko Plan B
- Mitra menolak (mayoritas) → B2 saja; peluang tetap ±17–29 % perak / 34–51 % perunggu.
- Skor publik bukan prediktor privat di dataran → jangan mengejar selisih 0,001.
- Bundel heliosli (dan ckpt OAI nartaa) = pengetahuan eksternal tak terklarifikasi host (743416: 35 komentar, host diam). Jalur bersih 0,944 BUKAN lagi zona perunggu (6 Okt: perunggu 0,944; 8 Okt: 0,947; 10 Okt: 0,950 — rank 0,944 ≈1.387) → hanya perlindungan kepatuhan, nilai medali ≈0. Putuskan 21–22 Okt setelah cek ulang 743416; selama host diam dan OAI 'accessible' (741819), kedua final tetap dari keluarga Apex.
