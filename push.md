# push.md — Arahan untuk agen/harness lain (Claude Code, OpenCode, dll.) melanjutkan misi PERAK RSNA Knee
> Berkas mandiri. Baca juga `MEMORY.md` (aturan+fakta), `STRATEGI_PERAK.md` (rencana+gerbang) dan `HANDOFF.md` (langkah harian). Diperbarui 10 Okt 2026 ~21:30 WIB. Bahasa kerja: **Indonesia**.

## 1. Misi
- Tim `Krazy KIller` (akun utama `dimaspashaakrilian`; rekan `nxtirfan`) di `rsna-knee-abnormality-detection` (macro ROC-AUC 12 label). **Final 22 Okt 2026 23:59 UTC**; pilih maks 2 submisi final (manual).
- Target: **perak** (top 5 % privat ≈ rank 283/5.666). Papan publik 10 Okt 19:37 WIB: ≥0,952 = 217, ≥0,951 = 348, =0,950 = 564; kita **0,950 rank 530** → kandidat final harus ≥0,952 publik (garis naik ≈ +20 tim/12 jam). Emas (~0,960) ≈0–2 %. **100 % tidak bisa dijamin.**
- Syarat user: tanpa data OAI MENTAH (ckpt publik pra-latih, termasuk nartaa & OrthoFoundation, adalah "pre-trained model" yg diizinkan Rules). Pertahankan satu final dgn risiko berbeda.

## 2. Aturan keras
1. **Hanya akun utama** untuk komputasi dan SUBMISI. Akun 2 `alverlfarfaraway` (milik user sendiri) & akun 3 `nxtirfan` (token tersimpan di `akun_kaggle/`): CLAUDE.md menyebutnya "SAH", tetapi Rules 5.a ("disqualified if ... more than one Kaggle account") dan 5.b ("each Team member must be a single individual") + izin tertulis yg hanya satu kalimat = risiko diskualifikasi tim. Jangan dipakai tanpa konfirmasi tertulis Kaggle (draf pesan: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`). GPU akun utama (30 j/minggu) cukup.
2. Jangan baca/cetak/salin isi `kaggle.json`; jangan `VERBOSE=true`. Jangan membuat akun/menyetujui perjanjian atas nama user. Jangan menghubungi pihak ketiga (merger = tindakan user).
3. Commit tanpa trailer `Co-Authored-By`/`Claude-Session`; `git commit -F <berkas>`; MEMORY.md commit terpisah ("memory: …"); `git add <path>` (bukan `-A`); `git fetch` dulu; push hanya atas perintah user.
4. Jangan sentuh kernel proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Submisi agen ≤2/hari (5/hari tim, reset 07:00 WIB); sisakan 2–3 utk user & rekan.
5. Rules: dilarang berbagi kode/data secara PRIVAT antar-tim (5.d/6.a); berbagi publik di forum boleh. Jujur melaporkan hasil; keputusan dari skor LB nyata (emas-58: SE ≈0,01 dan terkontaminasi utk OF).

## 3. Status (10 Okt 21:30 WIB)
- Terbaik: **0,950** (Apex 56943646). Dibuang: r224 0,950; fork rabari 0,950; **Apex+MaVIT 0,949**; klaim "0,954" (rabari/emresariduman/sameerk) = kode Apex. **57028008 (Apex+OF) = Notebook Timeout** (9 j).
- Tak ada notebook publik ≥0,951 yg runnable (heliosli mati; template/jenillangaliya = dataset privat).
- Uji berjalan: `ofscale3`/`ofscale4` (throughput OrthoFoundation, 4 proses) → gerbang G0 → submit `rsna-knee-apexof2` (stub lulus). Kuota: GPU 27,1 h (reset 17 Okt), submisi hari ini sisa 3.

## 4. Strategi (ringkas; rincian & gerbang di `STRATEGI_PERAK.md`)
- **J1 Merger** dgn tim ≥0,955 (EV terbesar; batas 15 Okt 23:59 UTC; aksi user; `docs/KANDIDAT_MERGER.md`). Agen menyiapkan bahan, TIDAK menghubungi siapa pun.
- **J2 Apex 0,70 + OrthoFoundation-L fold0 0,30** (resep heliosli): `python3 build_submit_legs.py name=apexof2 legs=of wo=0.30` → push (commit-run) → submit. Tahap OF v2 (`of_stage.py`): potongan 8 studi + akar-data kecil per potongan, 4 proses (2/GPU), penjaga disk (330 MB/studi), thread=1, prioritas studi, berhenti lunak +6,6 j / bunuh +7,0 j, reuso runtime anatomi, fallback rank Apex. **G1**: ≥0,952 → kandidat utama; 0,951 → cadangan; ≤0,950 → tutup OF.
- **J3 Final tahan-guncangan**: 2 final beda mekanisme; variasi terakhir ≤20 Okt; pilih dari submisi terskor; verifikasi via daftar submisi.
- **J4 Kepatuhan**: kepastian tertulis Kaggle soal akun ganda; satu final dgn risiko berbeda (OAI/eksternal; host diam di 743416).
- Ditutup (jangan ulangi): varian r224/bobot/mirror, MaVIT, pseudo-label/anak-OAI, kompilator aturan heliosli (0,784 vs emas), resep forum 746792 (sudah identik), klon "0,951+".

## 5. Cara kerja teknis ringkas
- Persiapan: `export PATH="$HOME/.local/bin:$PATH"`; `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`. Cek: `kaggle quota`; `kaggle competitions submissions rsna-knee-abnormality-detection -v`; `kaggle competitions submission-limits rsna-knee-abnormality-detection`.
- Papan: `kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>` (CSV; `TeamMemberUserNames`). Simulasi peluang: `python3 silver_prob_sim.py <csv>`.
- Kernel submisi harus punya versi COMPLETE; `-f submission.csv` WAJIB; log run submisi tak terbaca → uji di commit-run (akar simbolik). Estimasi waktu dari uji SKALA (bukan 58 studi). Batas 9 jam keras (Timeout = tanpa skor).
- **Log kernel lewat Chrome** (halaman `/code/<owner>/<slug>` → Logs; JS saring baris; hapus `? & =`). `kaggle kernels output` pada kernel OF mengunduh best.pt 1,2 GB → pakai `--file-pattern` kecil. Hindari pencarian forum tanpa `--category competitions`; `kaggle forums topics show <id> --page-size 200`.
- Batas sesi: GPU 2 / TPU 1 / CPU 5; 1 sesi GPU = 2×T4; `machine_shape: NvidiaTeslaT4`. Tanpa torch lokal. Jangan `pkill -f` pola sendiri. Colab (cadangan): `colab new -s rsna --gpu T4` → `colab stop` + hapus kaggle.json.
- Git: `git fetch origin && git status -sb` → `git add <path>` → `git commit -F pesan.txt` → push hanya atas perintah user. Remote: github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection (privat).

## 6. Kriteria sukses & kapan berhenti
- Sukses: submisi nyata ≥0,952 (aman) atau ≥0,951 (tepi). Catat tiap hasil di `MEMORY.md` §3, `laporan_push.md`, `docs/RESEARCH_NOTES.md`.
- Bila OF ≤0,950: tutup; fokus merger (user) + finals (Apex-keluarga + hedge). Jangan membuang kuota untuk variasi kecil (Δ<0,002 = noise).
- Peluang jujur (simulasi): perak ±15–20 % tanpa perbaikan; ±40–67 % bila skill sejati +0,004; merger dgn tim ≥0,956 ±23–82 %; emas ≈0–2 %.

## 7. Laporan ke user (Indonesia, ringkas)
Setelah tiap unit kerja: hasil + bukti angka, status kuota, langkah berikut. Jangan menjanjikan emas atau 100 %. Minta user hanya untuk tindakan yg hanya bisa dia lakukan (merger, pesan ke Kaggle, UI Kaggle, pendaftaran akun).
