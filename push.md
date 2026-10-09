# push.md — Arahan untuk agen/harness lain (OpenCode, dll.) melanjutkan misi SILVER RSNA Knee
> Berkas mandiri. Baca juga `MEMORY.md` (aturan+fakta, padat) dan `HANDOFF.md` (urutan kerja). Dibuat 9 Okt 2026 ~21:30 WIB. Bahasa kerja: **Indonesia**.

## 1. Misi
- Tim `Krazy KIller` (akun utama `dimaspashaakrilian`, rekan `nxtirfan`) di kompetisi `rsna-knee-abnormality-detection` (macro ROC-AUC 12 label). **Final 22 Okt 2026 23:59 UTC.** Pilih maks 2 submisi final.
- Target: **perak** (top ~5% → saat ini butuh skor publik ≥0,951; garis bergerak naik, proyeksi akhir 0,951–0,953). Emas/peringkat 10 TIDAK realistis (butuh ~0,961; 12 dari 20 teratas solo GM/juara RSNA).
- Syarat user untuk fase ini: **JANGAN memakai data OAI mentah.** (Bobot publik pra-latih, termasuk ckpt nartaa yg dilatih dgn OAI, adalah "pre-trained model" yg diizinkan Rules; tanpa itu kita mentok 0,944 = tak dapat medali. Pertahankan satu submisi final yg sepenuhnya bersih sbg cadangan.)

## 2. Aturan keras (jangan dilanggar)
1. **3 akun Kaggle tim yg SAH dipakai** (perintah user): utama `dimaspashaakrilian` (`~/.kaggle/`), kedua `alverlfarfaraway` (`~/.kaggle-acct2/`, mulai 9 Okt), ketiga `nxtirfan` (`akun_kaggle/`, mulai 10 Okt — DIPAKAI BILA GPU AKUN2 HABIS). **DILARANG** akun Kaggle di luar tim (Rules: berbagi kode/data kompetisi privat di luar tim). Risiko Rules 3.5.a (multi-account) = `laporan_push.md` §6, **izin panitia TERARSIP 10 Okt di `docs/IZIN_AKUN_KEDUA.md`**. Jangan baca/cetak/salin isi `kaggle.json`; folder `akun_kaggle/` di-`.gitignore`.
2. Jangan membuat akun/menyetujui perjanjian atas nama user. Jangan mengirim pesan ke pihak ketiga tanpa persetujuan user.
3. Komitmen git: **tanpa trailer `Co-Authored-By`/`Claude-Session`**; pesan multi-baris via `git commit -F <berkas>`; commit `MEMORY.md` terpisah ("memory: …"). Jangan commit kredensial/berkas besar.
4. Jangan sentuh kernel/berkas proyek lain di akun yg sama (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`).
5. Sisakan 2–3 submisi/hari (batas 5/hari utk SELURUH tim, reset 07:00 WIB) untuk user & rekan.
6. Jujur melaporkan hasil; keputusan akhir hanya dari skor LB nyata (gold-58 SE ≈0,01, tak bisa membedakan ±0,005).

## 3. Status (10 Okt 03:00 WIB)
- Papan (10 Okt 01:25 WIB, 5.606 tim): perak ≤279. ≥0,951: **296 tim**; ≥0,952: 183. Blok 0,951-persis (113 tim) > kapasitas perak → **0,951 = lotre, 0,952 = target aman**. **Kita 0,950** di dataran (Apex v2 = 56943646).
- Terbaik bersih (tanpa bobot OAI): replika goodpjw 0,944 (56872456), d4-blend 0,943. Stack 7 jam; Apex ±30 menit.
- Kuota (10 Okt 03:00): **GPU utama 0,00 h sampai 10 Okt 07:00 WIB** lalu 30 h/minggu; **akun2 ~30h; akun3 ~27,7h** (reset juga 07:00 WIB → total ~90 jam); TPU ~54h total (XLA kita crash — hindari); submisi 9 Okt terpakai 3/5 → reset 5/hari. Push kernel GPU sebelum reset bisa ditolak ("Maximum weekly GPU quota"). Cek: `bash kg_quota.sh`.
- **Izin multi-akun TERARSIP** 10 Okt (`docs/IZIN_AKUN_KEDUA.md`). Riset mendalam 10 Okt → `reports/Cara menembus 0951 perak.md`: **PEMISAH 0,951+ = PSEUDO-LABELING** (735304: colum2131 LB 0,958, azamat1ch 0,960, hengck "it is your labels").
- Garis TUTUP (jangan ulangi): varian r224 & bobot halus (0,950 placebo), resep 746792 (identik stack kita; all-plane flip minus), MaVIT (gold-58 0,9202), heliosli S6 rekonstruksi (mati permanen), SWA manual/seed-avg (nartaa rilis cuma 2 ckpt), d5e-goldw & d5f (placebo), bagging tanpa OAI (0,944), KneeNet, soup, metadata DICOM, anak fine-tune ep1 (≈induk).
- Aset siap: cache `rsna-knee-volcache-z{0,1,2}` (4.407 studi), ckpt induk, OOF machinery, pola eval gratis `alverlfarfaraway/mavit-288-gold58-eval` (MODE=local). Alat: `scan_public_updates.py`, `kg2.sh`/`kg3.sh`/`kg_quota.sh`, `bash kg2.sh forums topics show <id>`.

## 4. Strategi perak (urut prioritas, REVISI 10 Okt)
**A. Fase A — harvest mirror GRATIS** (10 Okt pagi; ~0,5 GPU-jam; TANPA submisi): dump prob gold-58 utk incumbent ± mirror (H1 r224+mirror, H2 reader+mirror, H3 mirror sbg arm terpisah) via commit-run `MODE=local` (pola `alverlfarfaraway/mavit-288-gold58-eval`) → sweep bobot rank/prob CPU → **gate: Δ ≥+0,005 & bootstrap CI95>0** (tuned>5 iter → +0,007; <0,003=noise). Lulus → 1 submisi kandidat.

**B. Fase B — PSEUDO-LABEL SELF-TRAINING** (10–14 Okt; 10–25 GPU-jam; jalur berbukti — 735304): pseudo-label 4.407 train pake stack incumbent (OOF-aware, jangan bocor gold) → fine-tune ckpt induk → OOF gate → commit-run → submisi bila lulus. Jadwal & protokol lengkap: `reports/Cara menembus 0951 perak.md` §4.

**C. Fast-follower (tiap sesi):** `python3 scan_public_updates.py` lalu baca `~/.cache/rsna_scan_log.txt` (loop `scan_loop.sh` mengisinya tiap 20 mnt bila hidup: `nohup bash scan_loop.sh &`). Cari: (i) ckpt/dataset bobot baru dari peserta lain, (ii) notebook publik ≥0,951. Pasang sbg arm di `build_submit_oai_family.py` (daftar ARMS berbobot, rank-mean) atau fork lalu uji. Kandidat yg muncul 8–9 Okt (belum dievaluasi): `sameerk2004/rsna-knee-sil-d5b-dual0950`, `...d5c-triple224`, `hitarthjain0/rsna-knee-spatial-coatnet-swa-v6`, `rabari9999/rsna-knee-sota-2-5d-convnext-dual-reader-0950`. Tarik dgn `kaggle kernels pull <ref> -p <dir> -m`, baca sumber dataset di `kernel-metadata.json`. Replika murni boleh (kita pernah replika goodpjw = 0,944 persis); jangan klaim sbg karya sendiri.

**D. Tim (batas 15 Okt):** satu-satunya cara sah menambah jumlah model independen. Hanya opsi untuk user (lihat `docs/KANDIDAT_MERGER.md`): agen tak boleh menghubungi siapa pun. Dilarang berbagi kode/data kompetisi sebelum penggabungan resmi.

**E. Ketahanan privat LB (penting utk perak):** public & private test terpisah; dataran 0,950 akan teracak. Untuk 2 pilihan final: (1) satu notebook CEPAT & kuat (keluarga Apex ≈30–60 mnt), (2) satu cadangan bersih (replika 0,944 atau perbaikannya; stack 7 jam berisiko timeout bila set privat diganti). Tandai manual di UI Kaggle (Submissions → select 2); bila tidak, Kaggle memilih 2 terbaik publik.

## 5. Cara kerja teknis ringkas
- Persiapan: `export PATH="$HOME/.local/bin:$PATH"`; `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`; CLI terautentikasi akun utama (`kaggle ...`), akun2 (`bash kg2.sh ...`), akun3 (`bash kg3.sh ...`). Cek: `bash kg_quota.sh`, `kaggle competitions submissions rsna-knee-abnormality-detection`.
- Papan: `kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>` (CSV; kolom `TeamMemberUserNames` memuat `dimaspashaakrilian`).
- Kernel submisi harus punya versi COMPLETE; `-f submission.csv` WAJIB di `kaggle competitions submit`. Log run submisi nyata tak bisa ditarik; dev-run stub dulu. Slug = turunan judul; hapus `id_no` dari metadata bila menyalin kernel orang lain.
- Batas sesi: GPU 2 / TPU 1 / CPU 5. Satu sesi GPU = 2×T4. Jangan `pkill -f` dgn pola yg ada di perintah sendiri. Tanpa torch lokal; semua eksekusi ML di Kaggle/Colab.
- Colab GPU (cadangan saat GPU Kaggle habis): `colab new -s rsna --gpu T4`, unggah kaggle.json akun utama, `colab/bootstrap_child.py`; WAJIB `colab stop` + hapus kredensial di VM sesudah pakai. Skrip resumable ada di `colab/oai_child/`.
- Git: `git fetch origin && git status -sb` (rekan bisa push) → `git add <berkas spesifik>` (JANGAN `git add -A` tanpa path; hindari `kaggle.json`) → `git commit -F pesan.txt` → `git push origin master`. Remote: github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection (privat).

## 6. Kriteria sukses & kapan berhenti
- Sukses: submisi nyata ≥0,952 (aman perak) atau minimal ≥0,951 (tepi perak). Catat tiap hasil di `MEMORY.md` §3 dan `docs/RESEARCH_NOTES.md`.
- Bila setelah 14–15 Okt tak ada perbaikan ≥0,951: jangan buang kuota; fokus pada seleksi final yang tahan guncangan (poin 4E) dan pantau pemindai publik sampai 21 Okt.
- Peluang jujur: perak ±35–45%, perunggu ±85–90% (kita di tepi), emas ≈0%.

## 7. Laporan ke user (Indonesia, ringkas)
Setelah tiap unit kerja: hasil + bukti angka, status kuota, langkah berikut. Jangan menjanjikan emas/peringkat 10. Minta user hanya untuk tindakan yg memang hanya bisa dia lakukan (UI Kaggle, pendaftaran akun, menghubungi pihak lain).
