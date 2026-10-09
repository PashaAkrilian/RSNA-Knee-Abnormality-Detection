# push.md — Arahan untuk agen/harness lain (OpenCode, dll.) melanjutkan misi SILVER RSNA Knee
> Berkas mandiri. Baca juga `MEMORY.md` (aturan+fakta, padat) dan `HANDOFF.md` (urutan kerja). Dibuat 9 Okt 2026 ~21:30 WIB. Bahasa kerja: **Indonesia**.

## 1. Misi
- Tim `Krazy KIller` (akun utama `dimaspashaakrilian`, rekan `nxtirfan`) di kompetisi `rsna-knee-abnormality-detection` (macro ROC-AUC 12 label). **Final 22 Okt 2026 23:59 UTC.** Pilih maks 2 submisi final.
- Target: **perak** (top ~5% → saat ini butuh skor publik ≥0,951; garis bergerak naik, proyeksi akhir 0,951–0,953). Emas/peringkat 10 TIDAK realistis (butuh ~0,961; 12 dari 20 teratas solo GM/juara RSNA).
- Syarat user untuk fase ini: **JANGAN memakai data OAI mentah.** (Bobot publik pra-latih, termasuk ckpt nartaa yg dilatih dgn OAI, adalah "pre-trained model" yg diizinkan Rules; tanpa itu kita mentok 0,944 = tak dapat medali. Pertahankan satu submisi final yg sepenuhnya bersih sbg cadangan.)

## 2. Aturan keras (jangan dilanggar)
1. **3 akun Kaggle tim yg SAH dipakai** (perintah user): utama `dimaspashaakrilian` (`~/.kaggle/`), kedua `alverlfarfaraway` (`~/.kaggle-acct2/`, mulai 9 Okt), ketiga `nxtirfan` (`akun_kaggle/`, mulai 10 Okt — DIPAKAI BILA GPU AKUN2 HABIS). **DILARANG** akun Kaggle di luar tim (Rules: berbagi kode/data kompetisi privat di luar tim). Risiko Rules 3.5.a (multi-account) = `laporan_push.md` §6, izin panitia BELUM diarsip di `docs/IZIN_AKUN_KEDUA.md`. Jangan baca/cetak/salin isi `kaggle.json`; folder `akun_kaggle/` di-`.gitignore`.
2. Jangan membuat akun/menyetujui perjanjian atas nama user. Jangan mengirim pesan ke pihak ketiga tanpa persetujuan user.
3. Komitmen git: **tanpa trailer `Co-Authored-By`/`Claude-Session`**; pesan multi-baris via `git commit -F <berkas>`; commit `MEMORY.md` terpisah ("memory: …"). Jangan commit kredensial/berkas besar.
4. Jangan sentuh kernel/berkas proyek lain di akun yg sama (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`).
5. Sisakan 2–3 submisi/hari (batas 5/hari utk SELURUH tim, reset 07:00 WIB) untuk user & rekan.
6. Jujur melaporkan hasil; keputusan akhir hanya dari skor LB nyata (gold-58 SE ≈0,01, tak bisa membedakan ±0,005).

## 3. Status (9 Okt 21:00 WIB)
- Papan (5.584 tim): batas emas ≤rank 21 (0,960), perak ≤279 (0,950 bawah), perunggu ≤558. ≥0,951: 257 tim; ≥0,952: 169; ≥0,953: 128. **Kita rank 563 @0,950** (submisi 56943646 = notebook publik Apex v2 oleh rekan; dataran 0,950 ≈456 tim).
- Terbaik bersih (tanpa bobot OAI): replika goodpjw 0,944 (56872456), d4-blend 0,943. Stack 7 jam; Apex ±30 menit.
- Kuota (10 Okt 00:50): **GPU akun utama 0,00 h sampai Sabtu 10 Okt 07:00 WIB** lalu 30 h/minggu; **akun2 29,9h; akun3 27,7h** (reset juga 07:00 WIB); TPU utama ±14 h (tak berguna; skrip kita crash di XLA); CPU sesi 5. Push kernel GPU sebelum reset bisa ditolak ("Maximum weekly GPU quota"). Cek: `bash kg_quota.sh`.
- Garis yg SUDAH DITUTUP (jangan ulangi): anak fine-tune ckpt OAI pada label fusi (gold-58 ≈ induk/lebih rendah), bagging model tanpa OAI, KneeNet reader, soup bobot, metadata DICOM, OrthoDiffusion fitur beku, **varian r224 & tweak bobot halus** (57012143 = 0,950 = placebo), **replika arm heliosli OF** (dataset compact-v1 retired 9 Okt + ofl-ssl-weights 403 = blokir).
- Aset siap: `kaggle_kernel_oai_fam_r224/` (notebook Apex + sibling r224 nartaa; lolos uji CPU, belum dijalankan di GPU). Alat: `scan_public_updates.py`, `scan_loop.sh`, `build_submit_oai_family.py`, `docs/PAKET_REKAN_TIM.md`, `docs/KANDIDAT_MERGER.md`.

## 4. Strategi perak TANPA data OAI mentah (urut prioritas)
**A. Probe sibling r224 (hari pertama GPU)** — Sabtu 10 Okt ≥07:05 WIB.
1. `kaggle quota` (GPU remaining ≥25). Bangun ulang bila perlu: `python3 build_submit_oai_family.py name=r224 r224=1 children= w224=0.25 src=`.
2. `kaggle kernels push -p kaggle_kernel_oai_fam_r224` (commit-run di 3 studi stub, ≈10–20 mnt GPU). Pantau: `kaggle kernels status dimaspashaakrilian/rsna-knee-oai-fam-r224`; bila ERROR, log ada di `kaggle kernels output <slug> -p <dir>` (berkas `.log`), perbaiki `oai_family_r224_patch.py` (uji logika dulu di kernel CPU `build_smoke_r224_cpu.py`).
3. COMPLETE → `kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/rsna-knee-oai-fam-r224 -v <versi> -f submission.csv -m "Apex: r384 .75 + r224 .25 + reader goodpjw"`. Skor ±1–1,5 jam. **Terima bila ≥0,951.**
4. Alternatif sama persis hari ini: minta rekan `nxtirfan` menjalankan dari kuotanya (ikuti `docs/PAKET_REKAN_TIM.md`).

**B. Variasi murah di atas Apex (1 submisi per uji, maks 2/hari):** bobot r224 0,35; skala bobot reader per-label ×0,7 / ×1,3 (edit `TARGET_WEIGHTS` di sel 9 notebook Apex); crop 304; tambah/ganti reader. Tiap varian = commit-run GPU ±0,2 jam. Hentikan bila tak ada yg ≥0,951 setelah 4 uji (noise LB ±0,002).

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
