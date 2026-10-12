# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 12 Okt 2026 ~11:00 WIB (sesi OpenCode). Baca urut: `MEMORY.md` (aturan+fakta) → `STRATEGI_PERAK.md` (rencana+gerbang; §0 = perubahan 11 Okt) → `docs/PLAN_B.md` (§00 terbaru) → file ini (apa yang dikerjakan SEKARANG).
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

## 0c. Status (12 Okt ~11:00 WIB)
| Hal | Keadaan |
|---|---|
| Submisi terbaik | **0,954** (57051680 klon S6 user; rank 144). Varian 11 Okt semua ≤0,954 — plafon S6+reader; varian S6 TUTUP kecuali murid/blend. Belum ada submisi 12 Okt |
| **Pipeline murid — 3 ERROR, fix v6 RUNNING** | Semua kernel train-a/b/dino ERROR 3× (12 Okt): (1) tulis cache ke `/kaggle/input` read-only → fix `/tmp/cache256_train`; (2)(3) OOM CUDA bs=8→4→**2** (backbone ConvNeXt jalan per bs×series×kwin gambar; T4 16GB; goodpjw asli bs=8 di GPU lokal); dino: `_AttnOut` tanpa `forward` → ditambah. **v6/v4/v6 di-push ~10:45 WIB, status RUNNING**. Tiap retry = ±70 mnt rebuild cache on-demand |
| Keputusan bila v6 OOM lagi | gradient checkpointing (monkey-patch `model.enc`) ATAU kwin 8→6; jangan turunkan res (resep goodpjw = 256) |
| Setelah a/b/dino COMPLETE | jalankan ulang `scripts/night_student.sh` (PID lama exit 00:35 dgn fold=0): push train-c → tarik fold ckpt → dataset folds → infer → submit murid solo → s6student |
| Kuota 12 Okt 08:23 | alver 17,3 j; main 15,8 j; **nxtirfan 3,5 j ⚠️** (1 run dino tersisa; gagal = dino pindah main/alver). Reset 17 Okt 07:00 WIB |
| OAI | host diam; user belum jawab (21:16 11 Okt) — jangan eksekusi |
| Git | push 12 Okt atas perintah user |

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
