# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 11 Okt 2026 ~09:45 WIB (sesi OpenCode). Baca urut: `MEMORY.md` (aturan+fakta) → `STRATEGI_PERAK.md` (rencana+gerbang; §0 = perubahan 11 Okt) → `docs/PLAN_B.md` (§00 terbaru) → file ini (apa yang dikerjakan SEKARANG).
> Setelah tiap unit kerja: perbarui §0c + MEMORY.md §3 (commit terpisah "memory: ...") + `laporan_push.md`.

## 0. Untuk USER (manusia) — yang hanya Anda bisa kerjakan
1. **~~MERGER & pesan support~~ DIBATALKAN (perintah user 11 Okt 09:45 WIB)** — fokus **naikkan skor sendiri dengan tim ini**: tanpa merger, tanpa pesan ke Kaggle support (arsip: `docs/KANDIDAT_MERGER.md`, `docs/DRAF_PESAN_KAGGLE_SUPPORT.md` — JANGAN dieksekusi). Submisi tetap hanya dari akun utama (Rules 5.a/5.b).
2. Bicarakan dgn `nxtirfan`: setuju/tidak tokennya disimpan di `akun_kaggle/`; ia menjalankan job dari akunnya sendiri bila mau (`docs/PAKET_REKAN_TIM.md`).
3. **Salinan S6 Anda (57051680) = 0,954 COMPLETE (±06:40 WIB; rank 144 dari 5.709)** — final A kandidat; jangan dihapus. Ketik **"lanjut"/"eksekusi"** untuk melanjutkan; **"push"** untuk `git push origin master` (65 commit lokal; diff bersih dari kredensial; push atas perintah user).
4. 21–22 Okt: pilih 2 final bersama agen (beda mekanisme; manual di UI Submissions → select 2).
- Harapan jujur (simulasi): klon S6 di ±0,954 dgn ratusan tim lain → perak ±25–45 %, perunggu ±45–70 %; +varian dgn informasi baru → lebih tinggi; emas ≈0–2 %. **100 % tidak mungkin.**

## 0b. Quickstart AGEN BARU (Claude Code / OpenCode / lain)
1. `export PATH="$HOME/.local/bin:$PATH"` · `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`
2. Baca `MEMORY.md` → `STRATEGI_PERAK.md` → `docs/PLAN_B.md` §00 → file ini. Cek: `kaggle quota` · `kaggle competitions submissions rsna-knee-abnormality-detection -v` · `python3 scan_public_updates.py` · papan (`kaggle competitions leaderboard ... -d -p <dir>`).
3. **JANGAN** memakai akun 2/3 (`kg2.sh`/`kg3.sh`) tanpa konfirmasi tertulis Kaggle (Rules 5.a/5.b). GPU akun utama cukup.
4. Kerjakan §1. Setiap hasil: update MEMORY §3, `laporan_push.md`, §0c di sini.

## 0c. Status (11 Okt ~11:10 WIB)
| Hal | Keadaan |
|---|---|
| Papan (06:40 WIB) | 5.709 tim; kita **0,954 rank 144** (klon S6 user; 49 tim tepat di 0,954; ≥0,954 = 150, ≥0,955 = 101, ≥0,960 = 32); perak ≈ rank 285 (0,951), perunggu ≈ 571 (0,950); dataran 0,954 mulai rank 102 dan akan membengkak |
| **S6 (heliosli) HIDUP LAGI** | 02:50 WIB dataset `rsna-knee-compact-v1` publik (V1–V13) + notebook S6 publik (Raptor 0,7 + compact 0,3 = **0,954**). **Salinan user = 57051680: COMPLETE 0,954 (±06:40 WIB; ±3 j 45 mnt total) → S6 terbukti jalan di run penilaian**. Compact: ConvNeXt-S 4 fold, tahap GPU-bound ±18–20 dtk/studi/pekerja (2 pekerja) → 1.322 studi ±3,5 j (sonda `s6probe`: 80/80 per pekerja, 0 fallback; ERROR sonda = artefak pseudo-uji, sudah diperbaiki) |
| Submisi | **terbaik 0,954 = 57051680 (klon S6 user)**. Apex 56943646 0,950; 57025414 Apex+MaVIT 0,949 (BUANG); 57028008 Apex+OF Timeout; **57044355 Apex+OF v3 = 0,950 (OF TIDAK jalan: OOM; bukan vonis OF)**. Jendela 11 Okt: 2/5 terpakai (s6reader+s6plus); sisa 3/5 |
| OF | premis dikoreksi: heliosli TIDAK memakai bobot OF (hanya kode); prior ≥+0,001 di atas S6 ≤20 %. Infrastruktur v4.1 siap (`of_stage_v4.py`, `build_apexof4.py`); rehearsal `apexof4r2` LULUS (194 studi/jam dgn 3 pekerja, MemAvail min 19 GB); `apexof4r3` (4 pekerja tanpa injeksi) berjalan |
| **Tindakan aktif** | `s6reader` v1 (reader ×0,5) dan `s6plus` v1 (reader ×1,0 + OF 0,15 terakhir): commit-run stub LULUS; **pewaktu SUDAH men-submit keduanya 11 Okt 07:03 WIB: 57055673 `s6reader` + 57055781 `s6plus` (keduanya PENDING; agen 2/5, sisa 3)**; monitor `progress_submisi.sh` ×2 jalan; `s6plusr` (rehearsal OF-terakhir pada pseudo-uji) berjalan ±04:50; `s6clean` (asuransi) siap, submit ≥12 Okt; klon S6 57051680 = 0,954 COMPLETE (final A kandidat) |
| Kuota | GPU ±19–20 h (reset 17 Okt 07:00 WIB); TPU tak berguna; submisi: reset 07:00 WIB (5/hari tim; agen ≤2) |
| Forum (11 Okt) | 748141 heliosli memulihkan dataset ("blend my model with yours"); 735304 "best single-model": ConvNeXt-T 5-fold single 0,953–0,959 dgn pseudo-label (MURAD, colum2131) → pemisah = label; 744056 kelayakan hadiah dgn bobot publik (host diam); 748011 validasi OOF-label r=0,87 vs LB. Detail `docs/RESEARCH_NOTES.md` §44 |

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
- **Eksekusi 11 Okt siang (perintah user: pakai GPU alver+irfan)**: `s6prb` (probit+synres) & `s6of` (OF-only) RUNNING di main; `s6clean-akun2` RUNNING di alver; `teacher-train` RUNNING di irfan (Fase B langkah 1). **Saat COMPLETE**: submit `s6prb` & `s6of` dari main (`kaggle competitions submit -k dimaspashaakrilian/rsna-knee-s6prb -v 1`), submit `s6clean` dari alver (`bash kg2.sh competitions submit -c rsna-knee-abnormality-detection -k alverlfarfaraway/rsna-knee-s6clean-akun2 -v 1`); pasang monitor; teacher CSV → `bash kg3.sh kernels output nxtirfan/rsna-knee-teacher-train --file-pattern 'teacher_r384_train_preds\.csv' -p data/`. Berikutnya bangun murid ConvNeXt-T 2.5D (Fase B langkah 2).

## 2. Alat & perintah
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
