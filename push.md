# push.md — Arahan untuk agen/harness lain (Claude Code, OpenCode, dll.) melanjutkan misi PERAK RSNA Knee
> Berkas mandiri. Baca juga `MEMORY.md` (aturan+fakta), `STRATEGI_PERAK.md` (rencana+gerbang) dan `HANDOFF.md` (langkah harian). Diperbarui 11 Okt 2026 ~04:00 WIB. Bahasa kerja: **Indonesia**.

## 1. Misi
- Tim `Krazy KIller` (akun utama `dimaspashaakrilian`; rekan `nxtirfan`) di `rsna-knee-abnormality-detection` (macro ROC-AUC 12 label). **Final 22 Okt 2026 23:59 UTC**; pilih maks 2 submisi final (manual).
- Target: **perak** (top 5 % privat ≈ rank 285/5.699). Papan publik 11 Okt 02:59 WIB: ≥0,954 = 141, ≥0,952 = 229, ≥0,951 = 367, =0,950 = 618; kita **0,950 rank 599** (di bawah garis perunggu publik rank 570). **S6 heliosli (Raptor 0,7 + compact 0,3 = 0,954) publik lagi** → klaster 0,954 akan membengkak. Emas (~0,960+) ≈0–2 %. **100 % tidak bisa dijamin.**
- Syarat user: tanpa data OAI MENTAH (ckpt publik pra-latih, termasuk nartaa/compact/OrthoFoundation, adalah "pre-trained model" yg diizinkan Rules 2.6.b). Pertahankan satu final dgn risiko berbeda (mis. bersih tanpa Raptor/OAI).

## 2. Aturan keras
1. **Hanya akun utama** untuk komputasi dan SUBMISI. Akun 2 `alverlfarfaraway` (milik user sendiri) & akun 3 `nxtirfan` (token tersimpan di `akun_kaggle/`): CLAUDE.md menyebutnya "SAH", tetapi Rules 5.a ("disqualified if ... more than one Kaggle account") dan 5.b ("each Team member must be a single individual") + izin tertulis yg hanya satu kalimat = risiko diskualifikasi tim. Jangan dipakai tanpa konfirmasi tertulis Kaggle (draf pesan: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`). GPU akun utama (30 j/minggu) cukup.
2. Jangan baca/cetak/salin isi `kaggle.json`; jangan `VERBOSE=true`. Jangan membuat akun/menyetujui perjanjian atas nama user. Jangan menghubungi pihak ketiga (merger = tindakan user).
3. Commit tanpa trailer `Co-Authored-By`/`Claude-Session`; `git commit -F <berkas>`; MEMORY.md commit terpisah ("memory: …"); `git add <path>` (bukan `-A`); `git fetch` dulu; push hanya atas perintah user.
4. Jangan sentuh kernel proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Submisi agen ≤2/hari (5/hari tim, reset 07:00 WIB); sisakan 2–3 utk user & rekan.
5. Rules: dilarang berbagi kode/data secara PRIVAT antar-tim (5.d/6.a); berbagi publik di forum boleh. Jujur melaporkan hasil; keputusan dari skor LB nyata (emas-58: SE ≈0,01 dan terkontaminasi utk OF).

## 3a. Status (12 Okt ±14:45 WIB) — terbaru
- Terbaik **0,954** (57051680, klon S6), rank 466/5.809; perak & perunggu publik sama-sama 0,954 (dataran ±600 tim). Rencana aktif: **`STRATEGI_0960.md`**.
- Perintah user 12 Okt: 2 submisi (`s6eq` = notebook publik #1 heliosli 0,5/0,5; `s6eqr` = + reader goodpjw) dari akun alver (satu-satunya kredensial di mesin Windows) lewat `scripts/submit_watch.py` (menunggu slot GPU); push ke branch + PR tanpa atribusi Claude.
- Siap pakai: `build_s6eq.py`, `build_compact_oof.py` (guru OOF compact 4 fold), scrape lengkap (`reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md`).

## 3. Status (11 Okt 04:00 WIB)
- Terbaik: **0,950** (Apex 56943646). Dibuang: r224 0,950; fork rabari 0,950; Apex+MaVIT 0,949; klaim "Tri-Reader 0,954" (rabari/jcgweb/sameerk = Apex+r224). 57028008 (Apex+OF) Timeout; **57044355 (Apex+OF v3) = 0,950 karena OF mati OOM** (bukan vonis OF). **57051680 = salinan S6 oleh USER (PENDING; ±07:00–08:30 WIB).**
- Koreksi: resep heliosli = Raptor + **compact** (bukan OF); bobot OF tak pernah dipakai di notebook heliosli.
- Alat siap: `build_s6plus.py` (reader goodpjw ke perna Raptor; OF dgn supervisor v4.1 sbg tahap terakhir; blend 3 perna), `build_apexof4.py` + `of_stage_v4.py` (Apex+OF v4.1), `build_s6probe.py`, `build_ofmem.py`. Rehearsal berjalan: `s6plusr`, `apexof4r3`. Kuota GPU ±19–20 h (reset 17 Okt).

## 4. Strategi (ringkas; rincian & gerbang di `STRATEGI_PERAK.md`, `docs/PLAN_B.md` §00)
- **J0 Dasar S6**: klon user 57051680; bila ≈0,954 → final A; bila ERROR/≤0,952 → diagnosis (RAM/fallback >3 %) dgn sonda `s6probe` + rehearsal.
- **J1 Merger** (EV terbesar; batas 15 Okt 23:59 UTC; aksi user; `docs/KANDIDAT_MERGER.md` §3b): Kelompok B (15 tim ≥0,955 ≤30 submisi), Kelompok C (mencari rekan), heliosli. Agen menyiapkan bahan, TIDAK menghubungi siapa pun.
- **J2 Tuas teknis di atas S6 (EV kecil; LB resolusi 0,001)**: `s6reader` (reader fusi pada perna Raptor), `s6plus` (+OF terakhir, coverage ±60–70 %), bagging bundel heliosli lain (mahal), final bersih (compact + reader).
- **J3 Final tahan-guncangan**: 2 final beda mekanisme; variasi terakhir ≤20 Okt; pilih dari submisi terskor; verifikasi via daftar submisi.
- **J4 Kepatuhan**: satu final dgn risiko berbeda; host diam di 743416/744056/747455; pesan ke support = DIBATALKAN (user 11 Okt).
- Ditutup (jangan ulangi): varian r224/bobot/mirror, MaVIT, pseudo-label/anak-OAI, kompilator aturan heliosli, "Tri-Reader 0,954", klon DINOsaur.

## 5. Cara kerja teknis ringkas
- Persiapan: `export PATH="$HOME/.local/bin:$PATH"`; `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`. Cek: `kaggle quota`; `kaggle competitions submissions rsna-knee-abnormality-detection -v`; `kaggle competitions submission-limits rsna-knee-abnormality-detection`.
- Papan: `kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>` (CSV; `TeamMemberUserNames`). Simulasi peluang: `python3 silver_prob_sim.py <csv>`.
- Kernel submisi harus punya versi COMPLETE; `-f submission.csv` WAJIB; log run submisi tak terbaca → uji di commit-run (akar simbolik). Estimasi waktu dari uji SKALA (bukan 58 studi). Batas 9 jam keras (Timeout = tanpa skor).
- **Log kernel lewat Chrome** (halaman `/code/<owner>/<slug>` → Logs; JS saring baris; hapus `? & =`). `kaggle kernels output` pada kernel OF mengunduh best.pt 1,2 GB → pakai `--file-pattern` kecil. Hindari pencarian forum tanpa `--category competitions`; `kaggle forums topics show <id> --page-size 200`.
- Batas sesi: GPU 2 / TPU 1 / CPU 5; 1 sesi GPU = 2×T4; `machine_shape: NvidiaTeslaT4`. Tanpa torch lokal. Jangan `pkill -f` pola sendiri. Colab (cadangan): `colab new -s rsna --gpu T4` → `colab stop` + hapus kaggle.json.
- Git: `git fetch origin && git status -sb` → `git add <path>` → `git commit -F pesan.txt` → push hanya atas perintah user. Remote: github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection (privat).

## 6. Kriteria sukses & kapan berhenti
- Sukses: submisi nyata ≥ klon S6 (≈0,954) dgn informasi baru. Catat tiap hasil di `MEMORY.md` §3, `laporan_push.md`, `docs/RESEARCH_NOTES.md`.
- Bila varian S6+ ≤ klon S6 dlm noise: jangan menambah variasi kecil (Δ<0,002 = noise); fokus final A/B + varian beda mekanisme.
- Peluang jujur (simulasi): klon S6 di klaster ratusan tim → perak ±25–45 %, perunggu ±45–70 %; perna dgn informasi baru lebih tinggi; emas ≈0–2 %.

## 7. Laporan ke user (Indonesia, ringkas)
Setelah tiap unit kerja: hasil + bukti angka, status kuota, langkah berikut. Jangan menjanjikan emas atau 100 %. Minta user hanya untuk tindakan yg hanya bisa dia lakukan (UI Kaggle final, pendaftaran akun).
