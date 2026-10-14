# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 14 Okt 2026 ~15:20 WIB (sesi Claude Code, mesin Linux; sebelumnya 12 Okt 23:58 sesi OpenCode). Baca urut: `MEMORY.md` (aturan+fakta) → **`STRATEGI_0960.md`** (rencana aktif 12 Okt) → `STRATEGI_PERAK.md` (§0.0) → `docs/PLAN_B.md` (§00) → file ini (apa yang dikerjakan SEKARANG).
> Setelah tiap unit kerja: perbarui §0c + MEMORY.md §3 (commit terpisah "memory: ...") + `laporan_push.md`.

## 0. Untuk USER (manusia) — yang hanya Anda bisa kerjakan
1. **~~MERGER & pesan support~~ DIBATALKAN (perintah user 11 Okt 09:45 WIB)** — fokus **naikkan skor sendiri dengan tim ini**: tanpa merger, tanpa pesan ke Kaggle support (arsip: `docs/KANDIDAT_MERGER.md`, `docs/DRAF_PESAN_KAGGLE_SUPPORT.md` — JANGAN dieksekusi). **Perintah user 12 Okt: ketiga akun boleh dipakai, tetapi akun utama JANGAN untuk training** (risiko Rules 5.a tetap tercatat di MEMORY §0/§2).
2. Bicarakan dgn `nxtirfan`: setuju/tidak tokennya disimpan di `akun_kaggle/`; ia menjalankan job dari akunnya sendiri bila mau (`docs/PAKET_REKAN_TIM.md`).
3. **Salinan S6 Anda (57051680) = 0,954 (rank 499/5.812 per papan 12 Okt 16:00)** — final A kandidat; jangan dihapus. Ketik **"lanjut"/"eksekusi"** untuk melanjutkan; **"push"** untuk `git push origin master` (diff bersih dari kredensial; push atas perintah user).
4. 21–22 Okt: pilih 2 final bersama agen (beda mekanisme; manual di UI Submissions → select 2).
- Harapan jujur (simulasi): klon S6 di ±0,954 dgn ratusan tim lain → perak ±25–45 %, perunggu ±45–70 %; +varian dgn informasi baru → lebih tinggi; emas ≈0–2 %. **100 % tidak mungkin.**

## 0b. Quickstart AGEN BARU (Claude Code / OpenCode / lain)
1. `export PATH="$HOME/.local/bin:$PATH"` · `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`
2. Baca `MEMORY.md` → `STRATEGI_PERAK.md` → `docs/PLAN_B.md` §00 → file ini. Cek: `kaggle quota` · `kaggle competitions submissions rsna-knee-abnormality-detection -v` · `python3 scan_public_updates.py` · papan (`kaggle competitions leaderboard ... -d -p <dir>`).
3. **JANGAN** memakai akun 2/3 (`kg2.sh`/`kg3.sh`) tanpa konfirmasi tertulis Kaggle (Rules 5.a/5.b). GPU akun utama cukup.
4. Kerjakan §1. Setiap hasil: update MEMORY §3, `laporan_push.md`, §0c di sini.

## 0c. Status (14 Okt ~15:20 WIB — sesi Claude Code mesin Linux)
| Hal | Keadaan |
|---|---|
| **Submisi terbaik** | **0,956** = salinan R7 heliosli oleh user (57146631, 13 Okt 21:26 WIB) + duplikat 57150970 (0,956); rank **558/5.957**; dataran 0,956 ±750 tim (rank 202–951) = garis perak (297) DAN perunggu (595) |
| **Basis baru R7** | `heliosli/rsna-knee-compact-r7cnxfsss` (bundel `8c7b666e`) + notebook "≈0.957 one click"; salinan disimpan `kaggle_base/r7_heliosli_0957_20261013.ipynb` (+ `.meta.json`) |
| **Hasil 14 Okt** | `r7murid` 57174015 = 0,956, rank 558→604 (murid r1 di luar slot tak menambah). **`r7r224` 57185636 PENDING** (resep varshitha; skor ±02:30). Varian siap jatah 15 Okt: `r7r224f` (flip-TTA, commit jalan), `r7m224s` (murid di slot Raptor). Submit: `kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/<slug> -v 1 -f submission.csv -m "..."` |
| **Terjadwal (systemd --user)** | 15 Okt 07:03 `rsna-submit-15okt` (`r7v3f` + `r7v3fs`) · 17 Okt 07:05 `rsna-push-17okt` (latih r2a alver + r2c nxtirfan). Cek `systemctl --user list-timers 'rsna-*'`; log `data/submit_15okt.log`, `data/push_17okt.log`. (Penjadwal setsid/nohup lama MATI saat sesi tertutup → jangan dipakai lagi) |
| **Guru OOF R7** | f0/f1 COMPLETE → `data/r7oof/`; f2/f3 RUNNING (nxtirfan, push 21:21); lalu `bash kg3.sh kernels push -p kaggle_kernel_compact-gold-r7`; tarik: `bash kg3.sh kernels output nxtirfan/<slug> -p data/r7oof --file-pattern '(compact_(oof|gold).*\.csv|receipt.*\.json)$'` → `python3 make_labels_v1.py` |
| Kuota 21:20 | main 11,54 j (inferensi saja) · alver 0 · nxtirfan 8,54 j · reset 17 Okt 07:00 WIB |
| Dibatalkan | `s6student` + `student_infer` rencana 17 Okt (preds murid dari dataset → fallback diam-diam di run penilaian; DINO emas-58 0,71). Diganti `r7murid` |
| *(riwayat 12 Okt di bawah)* | |
| Submisi terbaik (12 Okt) | **0,954 tetap** (57051680 klon S6, rank 499/5.812; papan 16:00: dataran 0,954 = 649 tim, ≥0,954 = 835, perak 290 = 0,954, emas 21 = 0,962) |
| **S6-eq — SKOR MASUK 18:10** | `s6eq` v3 = **0,954** (57103358) + `s6eqr` v3 = **0,954** (57103359) → **G0 ≥0,955 BELUM lulus**; reader net-zero (konsisten 11 Okt); **jatah 3/5**. 400 sore = setting "All Versions Accessible" `heliosli/r6cnxfsss` → mirror `dimaspashaakrilian/rsna-knee-compact-r6cnxfsss` v1 |
| **Murid — train-a/b COMPLETE 18:10** | fold0–3 + oof sudah **di lokal** `data/student_out_a|b` (471 MB): **val AUC 0,9255–0,9279, gold 0,9189–0,9223**; `dino_full.pt` (95 MB) sudah ditarik (nxtirfan); **dataset `dimaspashaakrilian/rsna-knee-student-folds` v1 (privat) LIVE** (fold0–3+oof) — `build_student_infer.py` masih perlu mount folds saat push |
| train-c + infer + OOF | **ANTRE 17 Okt 07:00**: train-c (fold4) → dataset v2 → infer murid → blend/`s6student`; `build_compact_oof.py` (guru OOF, fold 0–3 stride 3 ±1,6 j/fold) juga 17 Okt (alver 4,6 j sisa malam ini TIDAK dipakai — train sudah selesai, jangan push). OOM lagi → gradient checkpointing / kwin 8→6, res 256 jangan diturunkan |
| Kuota 23:00 | main **12,67 j** (tak terpakai malam — tak ada rencana GPU); alver **4,6 j** (sisa, idle); nxtirfan 0. Reset **17 Okt 07:00 WIB** (+30 j/akun) |
| **Persiapan 17 Okt SELESAI 23:30** | builder train-c + compact-oof-f0..f3 + gold + student_infer + s6student siap push; **dataset alver `rsna-knee-student-folds` v1 + `rsna-knee-student-dino` v1 LIVE** (mount akun-2 aman); rincian perintah di §G6. akun 2/3 **kuota 0 s/d 17 Okt 07:00 — JANGAN push sebelumnya** |
| Pantau kuota | `scripts/quota_watch.sh` **JALAN** (PID 11661, 5 mnt → `data/quota_watch.log`): main 12,67 j · alver 0 · nxtirfan 0, refresh 17 Okt 07:00. Stop: `kill 11661`; start: `setsid nohup bash scripts/quota_watch.sh >/dev/null 2>&1 &` |
| Git | update 12 Okt (skor + train + persiapan + pantau kuota) **sudah push** sampai `c063ef5` |
| OAI | host diam; user belum jawab (21:16 11 Okt) — jangan eksekusi |

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
6. (15 Okt 02:30) rank 447 (r7r224 efektif); OOF R7 lengkap + label v1 diunggah (alver); timer systemd 07:03 & 17 Okt 07:05; builder `folds=` + `slot=compact` utk inferensi murid r2 (dataset fold r2 dibuat setelah latih: `dimaspashaakrilian/rsna-knee-student-r2a|r2c`).
7. (22:40) riset mendalam → `reports/Riset 14 Okt malam - jalan ke perak.md`; jadwal 07:03 diganti `r7v3f`/`r7v3fs` (base `v224m`); murid r2a (alver) + r2c (nxtirfan, konsensus saja) disiapkan utk 17 Okt.
7. (21:33) `r7r224` disubmit (57185636); `r7r224f` + `r7m224s` dibangun utk 15 Okt; pelajaran: anggota baru masuk di slot keluarganya.

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
