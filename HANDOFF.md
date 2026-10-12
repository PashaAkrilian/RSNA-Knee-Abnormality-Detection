# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 12 Okt 2026 ~17:05 WIB (sesi OpenCode, mesin Linux; sebelumnya ~14:45 WIB sesi Claude Code mesin Windows). Baca urut: `MEMORY.md` (aturan+fakta) → **`STRATEGI_0960.md`** (rencana aktif 12 Okt) → `STRATEGI_PERAK.md` (§0.0) → `docs/PLAN_B.md` (§00) → file ini (apa yang dikerjakan SEKARANG).
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

## 0c. Status (12 Okt ~17:05 WIB — sesi OpenCode mesin Linux)
| Hal | Keadaan |
|---|---|
| Submisi terbaik | **0,954** (57051680 klon S6 user); **papan 12 Okt 16:00 WIB: rank 499/5.812**; dataran 0,954 = **649 tim tepat** (≥0,954 = 835); perak 290 = 0,954, perunggu 581 = 0,954, emas 21 = 0,962; ≥0,960 = 39 |
| **S6-eq — 2 submisi TERKIRIMA 16:50 (ref 57103358 + 57103359)** | Di **AKUN UTAMA** (alver 2/2 slot train + kuota mati ±20:00; main dilarang training tapi inferensi/submisi boleh). v1 & v2 sempat **400 "non-versioned datasets"** → **akar masalah: setting "All Versions Accessible" di halaman dataset `heliosli/rsna-knee-compact-r6cnxfsss` milik heliosli (mati)**; pin versi di metadata TIDAK menolong (server strip). **Fix: mirror dataset** `dimaspashaakrilian/rsna-knee-compact-r6cnxfsss` v1 (887 MB, file identik, kredit heliosli) → `build_s6eq.py` ganti sumber → **v3 push + submit 16:50 DITERIMA** → skor penuh **±20:35 WIB**. Jatah **3/5** |
| Dataset mirror | `dimaspashaakrilian/rsna-knee-compact-r6cnxfsss` v1 — dipakai s6eq/s6eqr v3; REKAN/PUNYA ORANG bermasalah → mirror pola ini. Dataset kita sendiri tampaknya lolos pengecekan (bukti: v3 diterima) |
| Pipeline murid | **train-a + train-b RUNNING di alver** (v6 bs=2; sudah ±5 j → titik OOM ±30 mnt sudah lewat, AMAN); **dino COMPLETE** (nxtirfan; `dino_full.pt` siap ditarik via `kernels output --file-pattern`); 3 ERROR dini hari = read-only cache + OOM bs 8→4→2 + `_AttnOut.forward` |
| train-c + infer | **ANTRE reset kuota 17 Okt 07:00** — alver habis ±20:00 malam ini, nxtirfan 0, main DILARANG training (perintah user). Setelah train-a/b COMPLETE malam ini: tarik fold ckpt + oof (CPU, gratis) → dataset `rsna-knee-student-folds` |
| Kuota 16:13 | alver **6,89 j** (±1,9 j/jam → mati ±20:00); main **~12,9 j** (2 commit-run s6eq v3 saja); nxtirfan **0 ⚠️**. Reset **17 Okt 07:00 WIB** (+30 j/akun) |
| Guru OOF compact | `build_compact_oof.py` siap (fold 0–3: 1 model, stride 3, ±1,6 j/fold; gold: 4 model); push 17 Okt — lihat `STRATEGI_0960.md` §3 |
| Keputusan bila train OOM lagi | gradient checkpointing ATAU kwin 8→6; jangan turunkan res (resep goodpjw = 256) |
| Git | PR #1 (scrape 12 Okt + STRATEGI_0960 + builders) sudah **merged**; update MD sesi ini (resolusi 400 + v3) menunggu commit/push |
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
4. (BERJALAN) Pantau skor `s6eq`/`s6eqr` v3 — **catat di `laporan_push.md` §2 + MEMORY §3**; G0 ≥0,955. (Watcher `--resume` lama mati sendiri setelah 400; skor bisa dicek `kaggle competitions submissions -v`.)
5. Malam: pantau train-a/b (alver); bila COMPLETE → tarik `fold*.pt`+`oof_fold*.csv` (`kernels output --file-pattern`) → dataset `rsna-knee-student-folds` (CPU); dino sudah COMPLETE → tarik `dino_full.pt`. **JANGAN push apa pun ke alver** (kuota mati ±20:00).
6. 17 Okt 07:00+ (kuota baru +90 j): push train-c + `compact-oof-f0..f3` + `compact-gold` (alver/nxtirfan) → rantai infer → `s6student`; putaran-2 murid (§F + `STRATEGI_0960.md` §4).

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
