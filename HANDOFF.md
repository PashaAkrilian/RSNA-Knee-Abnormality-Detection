# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 18 Okt 2026 ~21:55 WIB (sesi Claude Code, mesin Linux; sebelumnya 14 Okt 15:20; sebelumnya 12 Okt 23:58 sesi OpenCode). Baca urut: `MEMORY.md` (aturan+fakta) → **`STRATEGI_0960.md`** (rencana aktif 12 Okt) → `STRATEGI_PERAK.md` (§0.0) → `docs/PLAN_B.md` (§00) → file ini (apa yang dikerjakan SEKARANG).
> Setelah tiap unit kerja: perbarui §0c + MEMORY.md §3 (commit terpisah "memory: ...") + `laporan_push.md`.

## 0. Untuk USER (manusia) — yang hanya Anda bisa kerjakan (15 Okt malam)
1. **Baca penanda auto-select** di halaman Kaggle *Submissions*: bila final belum dipilih manual, Kaggle menandai 2 submisi = **2 skor publik tak-dibulatkan tertinggi**. Kirim nomor/deskripsinya ke agen → kita tahu komponen mana yang benar-benar menaikkan skor (r7v3c? r7v4c40? r7v3s?) tanpa memakai jatah.
2. **Biarkan PC menyala** 16–17 Okt: jadwal otomatis (systemd --user, linger) submit 16 Okt 07:01, latih 17 Okt 07:05, unduh fold 17 Okt 11:00. Bila PC mati, tugas menyusul saat boot (bisa terlambat).
3. Jatah submisi: agen ≤2/hari (sisakan 2–3 utk Anda & rekan); jatah ekstra hanya atas izin Anda (15 Okt: 1 jatah dipakai utk `r7v4c40`).
4. **21–22 Okt: pilih 2 final** (manual di UI Submissions → select 2). Agen akan merekomendasikan A (resep terlengkap: S6F + murid) dan B (lindung nilai) pada 21 Okt.
5. Ketik **"lanjut"/"eksekusi"** untuk melanjutkan; **"push"** untuk `git push origin master`. Merger & pesan support DIBATALKAN (keputusan Anda 11 Okt; entry/merger ditutup 15 Okt 23:59 UTC).
- Harapan jujur: perak ±35–50 % bila murid + S6F memberi +0,0005–0,001 nyata; hanya salinan publik ±15–25 %; emas ≈0–2 %. **100 % tidak mungkin.**

## 0b. Quickstart AGEN BARU (Claude Code / OpenCode / lain)
1. `export PATH="$HOME/.local/bin:$PATH"` · `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`
2. Baca `MEMORY.md` → `STRATEGI_0960.md` §0.3 → file ini (§0c + §H). Cek: `kaggle quota` (+ `bash kg2.sh quota`, `bash kg3.sh quota`) · `kaggle competitions submissions rsna-knee-abnormality-detection -v | head` · `tail data/rank_watch.log` · `systemctl --user list-timers 'rsna-*'` · `python3 scan_public_updates.py`.
3. Akun: alver (`kg2.sh`) & nxtirfan (`kg3.sh`) = LATIH (keputusan user 12 Okt); akun utama = commit-run inferensi + submisi saja. Jangan cetak `kaggle.json`.
4. Bangun varian: `python3 build_r7murid.py base=v224m w=<murid> slot=compact folds=dimaspashaakrilian/rsna-knee-student-r2a [flip=1 fstride=1] [s6f=0.35] name=<nama>` → `kaggle kernels push -p kaggle_kernel_<nama>` → cek log commit (`GOLD58`, `MURID_LEG used`, `S6F_LEG used`, tanpa error) → submit. Setiap hasil: update MEMORY §3, `laporan_push.md`, §0c di sini.

## 0c. Status (18 Okt ±21:55 WIB — sesi Claude Code mesin Linux; status lama 15 Okt → arsip git)
| Hal | Keadaan |
|---|---|
| **Papan 18 Okt 17:45** | 6.168 tim; ≥0,957 = 1.408; **kita 0,958 rank 424** (terbaik publik = `g6a`; `g8a`/`g9a` tak melampauinya). Perak = rank ±308 → di DALAM blok 0,958 (salinan goodpjw v27 & heliosli "0,9586"). |
| **Submisi 18 Okt (UTC)** | `g9a` 03:36 UTC = 0,958 · **`g6c` 14:36 UTC** (21:36 WIB; murid r2a 8 model .4 + S6F .5) PENDING · sisa jatah 3 → agen boleh 2 lagi (sisakan 1 utk user). |
| **Bergerbang malam ini** | `g10a` (g6c + wc=h2 + R6 stride 2) & **`g12`** (R6s2 .36 / R7 .24 / murid8 .4 + S6F .5, wc=h) → `data/run_submit_gate2.sh` (SPECS `nama:versi:flip:s6f:r7`; versi gagal → coba v1) → submit bila lulus & sisa ≥2; log `data/submit_18okt.log`. |
| **Penjaga antrean** | `rsna-requeue.timer` tiap 2 mnt → `data/requeue.py`: kernel di WATCH yang **QUEUED ≥5 mnt** (perintah user 18 Okt) → hapus + push ulang (versi kembali v1). WATCH hanya kernel BELUM tersubmit (kini g10a, g12) — tambahkan tiap kernel baru. Log `data/requeue.log`. |
| **Bukti terbaru** | OOF lokal 3.480 studi: + R7 di compact **+0,0016–0,0020 (P 1,00)**; murid r2a~r3 ρ 0,970 (guru beda → murid nyaris sama); R6~R7 ρ 0,889 (lokal) vs 0,98 (gold). Gold-58: g6a 0,9398 · g6c 0,9408 · g9a 0,9408 · g10a 0,9403 · g12 0,9404 (semua derau). |
| **Kuota GPU** (reset 24 Okt 07:00 WIB = setelah final) | main 21,2 j (commit-run/submit saja) · alver 18,2 j · nxtirfan 10,3 j (sering dipakai proyek lain rekan — jangan disentuh). |
| **Riset** | agen latar menulis `reports/Riset 18 Okt malam - model publik baru.md` (papan, notebook ≥0,958 baru, dataset model/label baru: Mycelis 4-fold ConvNeXt 384px, DINO 5-fold, label omerzalman/haungjiawei). |
| **Menunggu user** | pilih 2 final 22 Okt (rekomendasi agen 21 Okt); push git hanya atas perintah. |

## 1. Urutan kerja
### A. 11 Okt 03:35–07:03 — SELESAI
1. (SELESAI) `s6plusr` LULUS → `s6reader` & `s6plus` ter-push; pewaktu men-submit keduanya 07:03 WIB (57055673 + 57055781).
2. Baca `apexof4r3` (4 pekerja tanpa injeksi; throughput/MemAvail) → putuskan `nproc` produksi OF.
### B. 07:00+ (jatah baru)
- Agen ≤2 submisi: pilih varian S6+ (urutan prioritas: `s6reader` → `s6plus`); bandingkan dgn klon S6 (57051680). LB resolusi 0,001 → hanya selisih ≥0,002 yg jelas; selebihnya putuskan berdasar prior/kekokohan.
### C. Tiap sesi
- `python3 scan_public_updates.py` + papan + `kaggle datasets list --user heliosli` / `kaggle kernels list --user heliosli` (S7?) + thread 748141/735304/743416. Syarat arm baru = skor ≥0,951 TERVERIFIKASI (halaman Code) **dan** input publik runnable.
### D. Merger — DIBATALKAN (user 11 Okt: fokus naikkan skor sendiri)
- Tanpa aksi; arsip kandidat tetap di `docs/KANDIDAT_MERGER.md`. Agen fokus J2 (varian S6+) dan J3 (final).
### E. Final (21–22 Okt)
- Variasi baru terakhir ≤20 Okt. Pilih 2 final dari submisi TERSKOR; beda mekanisme (A = S6/varian terbaik; B = varian S6+ atau bersih); verifikasi lewat daftar submisi; cek ulang 743416.
- **Riset 11 Okt SELESAI** (`reports/Riset diskusi + best-score code 11 Okt.md`): Fase B pseudo-label = prioritas naik (validator OOF-vs-weak-label r=0,87); pantau `kaggle datasets list -s r3cnx`.
- **Pipeline 0,960 (perintah user; riset bedah tim top — `reports/Riset murid+forum 11 Okt sore.md`)**: azamat1ch rank 4 = ConvNeXt-T 5-fold SOLO **0.959**; colum2131 = ConvNeXt 0.953 + DINOv2 0.951 → blend 0.955 → **0.959**; datnt114 0.961 (OOF 0.937 — train-on-all pseudo). Resep: pseudo-label quality + train-on-all + 2 keluarga blend. **Eksekusi (semua akun alver)**: prep v3 RUNNING (fix filename-find); train-a/b/c CONVNEXT siap (warm-start ckpt goodpjw, internet OFF); **student-dino ViT-S one-shot siap** (PyTorch-murni DinoSmall, key-strict 223, 30–60 mnt); student-infer siap (murid solo submission + student_test_preds.csv). Gelombang: prep→(a+b)→(c+dino)→push folds ke dataset→infer→submit. Blend S6+murid = builder leg CSV offline. OAI: host masih diam (748276) — risiko kepatuhan, keputusan user.

### F. 12 Okt siang (sesi Claude Code) — jalur 0,960
1. Tunggu hasil `s6eq`/`s6eqr` (watcher) → catat di `laporan_push.md` §2; bila ≥0,955 jadikan basis (gerbang G0 `STRATEGI_0960.md`).
2. Push `compact-oof-f0..f3` + `compact-gold` (±7 j GPU, 2 sesi paralel) → rakit label `L1 = 0,5 konsensus + 0,5 OOF-compact` → gerbang G1.
3. Revisi pipeline murid (`STRATEGI_0960.md` §6: cache CPU, 2 GPU paralel, deadline per epoch, 224 px) → murid A/B → notebook final S6-eq + murid.

### G. 12 Okt sore (sesi OpenCode, mesin Linux) — EKSEKUSI 2 submisi test
1. (SELESAI) `git pull` + merge PR #1 (scrape peta perak + STRATEGI_0960); papan segar 16:00 WIB: rank **499**/5.812, ≥0,954 = 835.
2. (SELESAI 16:12) Pindah `s6eq`/`s6eqr` ke akun utama; v1+v2 sempat 400 "non-versioned datasets" → pin versi metadata (TERBUKTI TIDAK menolong — server strip).
3. (SELESAI 17:00) **Akar 400 = setting "All Versions Accessible" dataset `heliosli/rsna-knee-compact-r6cnxfsss` (milik heliosli, mati)** → **mirror** `dimaspashaakrilian/rsna-knee-compact-r6cnxfsss` v1 (887 MB) → `build_s6eq.py` ganti sumber → **v3 push + submit 16:50 DITERIMA** (ref 57103358/57103359) → skor penuh **±20:35 WIB**; jatah 3/5.
4. (SELESAI 18:10) **Skor: `s6eq` v3 = 0,954, `s6eqr` v3 = 0,954** → **G0 ≥0,955 TIDAK lulus**; reader net-zero → sudah dicatat di `laporan_push.md` §2 (baris 20/21) + MEMORY §3. Jatah 3/5; watcher skor sudah dimatikan.
5. (SELESAI 23:00) train-a/b COMPLETE → tarik `fold0–3.pt` + `oof_fold0–3.csv` ke `data/student_out_a|b` (471 MB) + log; `dino_full.pt` (95 MB) tarik dari nxtirfan (`nxtirfan/rsna-knee-student-dinov2-s-one-shot`); **dataset `dimaspashaakrilian/rsna-knee-student-folds` v1 (privat) dibuat & LIVE**; `data/mirror_upload` dihapus (hemat 846 MB).
6. **RENCANA EKSEKUSI 17 Okt 07:00 WIB (kuota reset +30 j/akun; PERSIAPAN SUDAH SELESAI 12 Okt 23:30 — akun 2/3 kuota 0 s/d reset)**, perintah user: pakai akun 2 (alver) & 3 (nxtirfan):
   - **Sudah dibangun & siap push**: `kaggle_kernel_student-train-c` (fold4, alver), `kaggle_kernel_compact-oof-f0..f3` (owner alver; f2/f3 boleh rebuild `owner=nxtirfan` bila dijalankan di akun 3), `kaggle_kernel_compact-gold`, `kaggle_kernel_student_infer` (alver; mount `alverlfarfaraway/rsna-knee-student-folds` + `rsna-knee-student-dino` + cache256/tonylica/reader), `kaggle_kernel_s6student` (alver; s6prb base compact-v1 + student-preds, wmurid=0.35).
   - **Dataset ALVER LIVE**: `alverlfarfaraway/rsna-knee-student-folds` v1 (fold0–3+oof) + `alverlfarfaraway/rsna-knee-student-dino` v1 (`dino_full.pt`) — privat, milik akun sendiri (mount aman); salinan main `dimaspashaakrilian/rsna-knee-student-folds` v1 = cadangan.
   - Urutan 07:00: (1) kg2 push `student-train-c` + `compact-oof-f0` (retry bila "Maximum batch GPU session count"); kg3 push `compact-oof-f2` + `f3`; (2) setelah f0/f1 lengkap, kg2 push `f1` → `compact-gold` bila kuota sisa; (3) train-c COMPLETE → kg2 `kernels output ... --file-pattern "runs_student/(fold|oof)"` → `datasets version -p student_folds_alver` (v2 = +fold4; ulangi versi folds main bila perlu); (4) kg2 push `student_infer` → COMPLETE → tarik `submission.csv` + `student_test_preds.csv` → `datasets version -p` untuk `alverlfarfaraway/rsna-knee-student-preds`; (5) submit **murid solo** (kg2) → submit **`s6student`** (kg2) — jatah harian 5, agen ≤2; (6) catat skor di `laporan_push.md` §2 + MEMORY §3. Log tiap langkah di HANDOFF §0c.
   - **Catatan**: jangan push ke alver/nxtirfan SEBELUM 07:00 (kuota 0 → run ERROR); `build_student_infer.py`/`build_s6student.py` jangan diubah mount-nya tanpa rebuild; ganti ke basis `s6eq` utk final (`build_s6student.py nb=`) = putusan terpisah.

### H. 14 Okt (sesi Claude Code) — basis R7 + murid; target user 0,965 (jujur: top-7, <2 %)
1. (SELESAI) Baca md + cek: 13 Okt user submit 2 salinan R7 = 0,956 (rank 558). Papan 14 Okt: ≥0,957 = 201; dataran 0,956 ±750 tim.
2. (RUNNING) `r7murid` v1 (akun utama) → baca log commit (GOLD58 & MURID_LEG) → submit 1× (agen ≤2/hari).
3. (RUNNING) Guru OOF R7 di nxtirfan (f0,f1 → f2,f3 → gold). Setelah lengkap: `make_labels_v1.py` (lokal) → `labels_student_v1.csv` → dataset privat akun pelatih.
4. (17 Okt 07:00) Murid putaran-2 (alver; 2 fold paralel/sesi bila builder 2-GPU siap, warm-start dari r1) → `r7murid` dgn fold r2 → probe.
5. Final 21–22 Okt: A = varian R7 terbaik yg terskor (δ>0), B = salinan R7 murni (57146631) atau varian beda mekanisme.
9. (15 Okt 20:40) Izin user 1 jatah → **57231466 `r7v4c40`** (flip stride 1 + murid 0,40) 20:08. Riset: bliverrigpdoge = R7 + model knee-centred PRIBADI 0,2 (bobot privat); **leoprovorov r224m + S6F 0,4 compact = 0,957** (rank 212, 4 submisi); heliosli "0,9576" = R6 + flip (tak terbukti; salinan 0,9573 = 0,956 di 2 tim). **S6F terintegrasi** (`build_r7murid.py s6f=`; skrip publik nartaa komponen B saja, `kaggle_base/s6fall_solo_nartaa_20261015.py`); commit LULUS `r7v6` & `r7v3s`. Jadwal 16 Okt 07:01 diubah: **`r7v3s` + `r7v3c40`**. Laporan: `reports/Riset 15 Okt malam - komponen 0,957 dan jalan perak.md`. Minta user baca penanda auto-select (UI Submissions) = 2 skor tak-dibulatkan tertinggi.
8. (15 Okt 10:45) **`r7r224m` 57198411 = 0,956, rank 506→403** (cermin r224 menaikkan kita) → basis `v224m`. Notebook publik baru heliosli "=0.9573" = resep SAMA dgn `r7r224` (R7 + r224 A224 0,35, tanpa flip; dataset `r6-frozen-fe055406` v2 = bundel R7) → 0,9573 = salinan beruntung, tak ada komponen baru; `dkim54` = tim agent 0,946; papan 03:23 UTC: ≥0,957 = 255, tepat 0,956 = 916, perak rank 300 = 0,956. Commit LULUS: **`r7v4c`** (v224m + flip stride 1 + r2a compact 0,25; GPU compact 18 dtk/studi), **`r7v3c40`** (r2a compact 0,40), `r7v3t` (r2a kaki ke-3 0,2) commit LULUS. Colab TIDAK layak (cache latih dibangun dari DICOM kompetisi). Timer **17 Okt 11:00 `rsna-after-train-17okt`** → `data/run_after_train_17okt.sh` (tunggu r2a-n2 + r2c-a/b → unduh fold + gerbang gold ≥0,90/0,88 → dataset r2a v2 [fold 0–3] + dataset baru `dimaspashaakrilian/rsna-knee-student-r2c`; TIDAK submit) → log `data/after_train_17okt.log`. r2c bisa digabung ke slot murid sebagai `fold4–7.pt` (skrip murid merata-rata semua `fold[0-9].pt`).
6. (15 Okt 08:35) r7v3f/r7v3fs (57194021/35) = 0,956 tanpa naik rank → flip stride 2 dibuang. Submit `r7r224m` (57198411, 07:01) + `r7v3c` (57201659, 08:35; murid r2a 2 fold di slot compact). Murid r2a-n selesai: gold-58 0,9282. 17 Okt 07:05: alver `student-r2a-n2` (fold 2–3, resep sama) + nxtirfan r2c-a/b. Pemantau rank: `data/rank_watch.log`.
7. (15 Okt 03:05) user: "masih ada 2 submission, coba tingkatkan sampai perak" → submit r7v3f + r7v3fs; murid r2a-n (nxtirfan, fold 0–1, label v1) mulai latih; `r7v4` (flip stride 1) commit. Siang: r2a-n → dataset main → `r7v3c` (slot compact) → submit.
7. (15 Okt 02:30) rank 447 (r7r224 efektif); OOF R7 lengkap + label v1 diunggah (alver); timer systemd 07:03 & 17 Okt 07:05; builder `folds=` + `slot=compact` utk inferensi murid r2 (dataset fold r2 dibuat setelah latih: `dimaspashaakrilian/rsna-knee-student-r2a|r2c`).
7. (22:40) riset mendalam → `reports/Riset 14 Okt malam - jalan ke perak.md`; jadwal 07:03 diganti `r7v3f`/`r7v3fs` (base `v224m`); murid r2a (alver) + r2c (nxtirfan, konsensus saja) disiapkan utk 17 Okt.
7. (21:33) `r7r224` disubmit (57185636); `r7r224f` + `r7m224s` dibangun utk 15 Okt; pelajaran: anggota baru masuk di slot keluarganya.

### I00000000. 18 Okt 21:55 — `g9a` = 0,958 (tak melampaui g6a, rank 424). Gerbang lama salah menolak `g6c` (cek kaku w=0,3) → **g6c tersubmit manual 21:36**. `g10a` v1 salah dataset murid (r1) → v2 r2a. OOF lokal: R7 di compact +0,002 (P 1,00) → kernel **`g12`**. Runner baru `data/run_submit_gate2.sh`; penjaga antrean 5 mnt (perintah user). Latih akun 2/3 ditunda sampai riset label/model publik selesai (murid guru-baru ρ 0,97 dgn r2a → keragaman kecil).

### I0000000. 17 Okt 19:10 — heliosli 0,9586 = goodpjw + bobot V2 (probe papan). Kernel `g8a`/`g8b` (wc=h2 + murid 8 model, ± S6F) → `rsna-submit-g8` (log `data/submit_g8_17okt.log`). r7v6a = 0,957; g6a pending. Murid r3: r3-a (nxtirfan) + r3-bm (akun utama) jalan → dataset r3 MANUAL (pasca-latih dihentikan 19:05).

### I000000. 17 Okt 12:30 — PC mati 02:45–12:11; `r7v6a`/`g6a` tersubmit 12:11 (pending). OOF R6 f0–f1 bebas bocor; f2–f3 jalan (akun utama) → label v3 → r3 di nxtirfan (push script menunggu label s/d ±14:41, selain itu r2c). Gold setia g6a 0,9400.

### I00000. 17 Okt 02:30 — Target user 0,962 → jawaban jujur (< 5 %; perak cukup 0,959–0,960 unik). **Murid r3** (guru OOF R6+flip + R7): `rsna-oof-r6` (unit sementara) mendorong `compact-oof-r6-f1..f3` saat slot kosong → `data/r6oof/` → label v3 → dataset label nxtirfan → `data/labels_v3_ready`; 07:05 `data/run_push_17okt.sh` memilih r3 (atau r2c) untuk nxtirfan (`data/nxt_pushed.txt`); 11:00 `data/run_after_train_17okt.sh` (urutan baru: r2a → r7v7a + g7a commit → murid nxtirfan → dataset `rsna-knee-student-<X>`). Log: `data/oof_r6_17okt.log`, `data/labels_v3_make.log`, `data/push_17okt.log`, `data/after_train_17okt.log`. CATATAN: uji DRY 02:18 sempat membuat versi dataset r2a (isi sama fold0–3) — mode DRY kini dijaga.

### I0000. 17 Okt 03:00 — **Tim 0,958, rank 317/6.103** lewat salinan user 57269378 (notebook publik goodpjw2008 v27: Raptor 384m + 224m, **R6+flip**, bobot wc heliosli, pembaca 1/10; ±6 j). Perak publik = rank 305; blok 0,958 = 488 tim (kebanyakan salinan R6+flip). Gold setia: goodpjw 0,9384 < **r7v6a 0,9414**; R6flip + S6F .5 + murid4 .3 = 0,9398. Basis baru **`base=g958`** di `build_r7murid.py` (`kaggle_base/r6_goodpjw_0958_20261016.ipynb` = goodpjw v27 tanpa Part B) → kernel **`g6a`** (commit-run 02:35). Jadwal: 07:01 `r7v6a`, 07:03 `g6a` bergerbang (`rsna-submit-g6a-17okt`, log `data/submit_g6a_17okt.log`), `r7v7a` 11:00 dibangun+commit TANPA submit. Simulasi `silver_sim_1017.py`: salinan murni ≈0–12 % perak, resep unik δ +0,0006 → 57–86 %. **Baca rank tim setelah tiap skor**: naik = skor tak-dibulatkan > 57269378.

### I000. 16 Okt 20:40 — Garis perak publik kini **0,958** (≥0,958 = 446); kita rank 461 = ke-15 di blok 0,957. Murid 4 fold (gold 0,9334) siap; **`r7v6a`** (gold setia 0,9414) submit otomatis **17 Okt 07:01** (`rsna-submit-17okt`); `data/run_after_train_17okt.sh` (11:00) kini juga membangun **`r7v7a`** (murid 8 model) → commit → gerbang (gold ≥0,930, kaki murid & S6F, jumlah fold) → submit otomatis. R6+flip (heliosli) diuji di gold: tak menambah bila murid 4 fold ada. Flip DIBUANG.

### I00. 16 Okt 14:40 — `r7v3sr` = 0,957 **rank 317** (terbaik). Gold setia (`build_gold_full.py` + `gold_eval_full.py`): S6F 0,5 di separuh Raptor + murid 0,3 + bobot heliosli = 0,9410; flip DIBUANG. Tersubmit: `r7v3sw` (12:25), **`r7v5a`** (14:37). 17 Okt 07:05: alver `student-r2a-full` + `student-r2a-full2` (4 model full-data), nxtirfan r2c; 11:00 dataset murid 8 model → bangun `r7v5*` final (budget murid 6000 dtk).

### I0. 16 Okt 12:00 — HASIL: `r7v3s` = **0,957** (S6F + murid). `r7v3sr` pending ±13:15. Latih `dimaspashaakrilian/rsna-knee-student-r2a-m` (fold 2–3, akun utama, izin user) ±17:00. `r7v3sw` (bobot per-temuan heliosli) commit-run. 17 Okt: alver hanya `student-r2a-full`; skrip pasca-latih ambil fold 2–3 dari akun utama (`data/kg1.sh`).

### I. 16–22 Okt — URUTAN KERJA AKTIF (rencana `STRATEGI_0960.md` §0.3)
1. **16 Okt ±01:00**: skor `r7v4c40` 57231466 → `data/rank_watch.log` (rank saat SELESAI vs saat submit 403).
2. **16 Okt 07:01 & ±08:30 (otomatis)**: `r7v3s` (S6F 0,35 di slot compact + murid 0,30) lalu `r7v3sr` (S6F 0,35 di separuh Raptor + murid 0,30); tanpa flip. Skor ±11:45 & ±13:15. Gold-58: compact +0,0015, separuh Raptor +0,0034 → papan memutuskan struktur. Catat di `laporan_push.md` §2.
3. **17 Okt 07:05 (otomatis)**: push latih `student-r2a-n2` (fold 2–3) + `student-r2a-full` (fold 8/9) di alver, `student-r2c-a/b` di nxtirfan (`data/push_17okt.log`). **11:00 (otomatis)**: `data/run_after_train_17okt.sh` → dataset `rsna-knee-student-r2a` versi baru (fold 0–3 + 8–9) + `rsna-knee-student-r2c` (`data/after_train_17okt.log`).
4. **17 Okt siang (agen)**: bangun `r7v5*` = basis terbaik 16 Okt + murid 6 model (± r2c sbg `fold4–7.pt`, ± S6F) → commit (cek `GOLD58` ≥0,928 & detik/studi; naikkan `budget=` bila 6–10 model) → submit 2 jatah agen.
5. **18–20 Okt**: gabungkan komponen yang terbukti tak merugikan; variasi terakhir 20 Okt; pastikan durasi run < ±7 j (S6F + flip + murid).
6. **21 Okt**: rekomendasi 2 final ke user (A = resep terlengkap; B = lindung nilai berbeda risiko); **22 Okt 23:59 UTC** user memilih manual di UI.

## 2. Alat & perintah
- Builder baru 12 Okt: **`build_s6eq.py`** (`name=s6eq|s6eqr owner= reader=1`), **`build_compact_oof.py`** (`fold=0..3 | gold=1 owner= stride=`), **`scripts/submit_watch.py`** (push saat slot kosong → submit → pantau skor).
- Builder: **`build_s6plus.py`** (`name= reader=1 of=1 wr= wc= wo= pseudo= t0=nb|now stop_h= kill_h= deadline= lenient=1`), `build_s6probe.py` (sonda S6), `build_apexof4.py` (Apex+OF v4.1), `of_stage_v4.py` (+`test_of_v4_local.py`), `build_ofmem.py` (profil memori), lama: `build_submit_legs.py`/`of_stage.py`/`build_scale_of2.py`.
- Status/keluaran: `kaggle kernels status <slug>` · `kaggle kernels push -p <dir>` (commit-run; maks 2 sesi GPU; sesi antre tetap dihitung) · `kernels output --file-pattern '<regex>'` (JANGAN tanpa pola pada kernel OF).
- **Baca log lewat Chrome**: `https://www.kaggle.com/code/dimaspashaakrilian/<slug>/log` (log langsung terlambat ±5 mnt; baru lengkap setelah selesai) → JS: `[...document.body.innerText.matchAll(/^\d+(\.\d+)?s\s+\d+\s+.*$/gm)].map(x=>x[0].replace(/[?&=]/g,' '))` lalu saring. Panel "View Active Events" (sidebar) → ⋯ → Stop Session utk membatalkan sesi antre.
- Papan/Rules: `kaggle competitions leaderboard ... -d -p <dir>`; `kaggle competitions submission-limits ...`; forum: `kaggle forums topics show <id> --page-size 200` (daftar topik via CLI 403 → pakai Chrome `.../discussion?sort=recent-comments`).
- Versi dataset lama: venv kagglehub di scratchpad (`dataset_download('owner/slug/versions/N', path='manifest.json')`).
- Simulasi peluang: `python3 silver_prob_sim.py <csv papan>`. Colab GPU (cadangan): `colab new -s rsna --gpu T4` → `colab stop`.

## 3. Peta file
- Aturan/memori: `CLAUDE.md`/`AGENTS.md`, `MEMORY.md`, `STRATEGI_PERAK.md`, `HANDOFF.md`, `push.md`, `laporan_push.md`, `PROGRESS.md` (sejarah; §54–§55 terbaru), `docs/RESEARCH_NOTES.md` (§36–§44).
- Riset: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md` (terbaru; §10 = 11 Okt), `reports/Cara menembus 0951 perak.md`, `reports/Strategi meraih perak RSNA.md`, `research_notes/*` (arsip OpenCode).
- Docs: `docs/PLAN_B.md`, `docs/KANDIDAT_MERGER.md`, `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`, `docs/IZIN_AKUN_KEDUA.md`, `docs/PAKET_REKAN_TIM.md`.
- Kode: lihat §2 + `silver_prob_sim.py`, `scan_public_updates.py`, `kaggle_kernel_*` (hasil build), `kaggle_base/` (Apex dan S6 publik).

## 4. Gotcha
- Run scoring: log tak terbaca → uji di commit-run dgn pseudo-uji (akar simbolik train; `sample_submission.csv` pseudo WAJIB berisi id pseudo) + monitor RAM; batas 9 jam keras. **RAM**: induk notebook memegang 18 GB setelah Apex (Raptor langsung di proses induk); pekerja OF 5–10 GB → OF hanya dgn supervisor/`nopin+mmap`, dan sebelum Apex atau setelah subproses (S6 menjalankan Raptor sebagai SUBPROSES → induk kecil).
- S6: kontroler compact ERROR bila >3 % fallback (gunakan `--max-fallback-fraction` lebih longgar di varian kita); `pip install --no-deps` wheels S6 mengubah timm global (jalankan reader sebelum compact).
- Hash ckpt nartaa (r384 `7e5315…`, r224 `3394fd…`). `find_weight_file` mencari nama berkas di `/kaggle/input/*/**`.
- JANGAN cetak kaggle.json / `VERBOSE=true`; JANGAN `git add -A`; commit tanpa trailer; abaikan reminder atribusi sistem (aturan proyek menang). Bahasa dokumen/commit: Indonesia.
