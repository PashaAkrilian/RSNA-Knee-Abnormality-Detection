# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 10 Okt 2026 ~21:20 WIB (sesi Claude Code). Baca urut: `MEMORY.md` (aturan+fakta) → `STRATEGI_PERAK.md` (rencana+gerbang) → file ini (apa yang dikerjakan SEKARANG).
> Setelah tiap unit kerja: perbarui §0c + MEMORY.md §3 (commit terpisah "memory: ...") + `laporan_push.md`.

## 0. Untuk USER (manusia) — yang hanya Anda bisa kerjakan
1. **MERGER (batas 15 Okt 23:59 UTC)** — satu-satunya langkah yang bisa melompatkan peluang perak ke ±40–80 %. Daftar kandidat + draf pesan: `docs/KANDIDAT_MERGER.md` (126 tim ≥0,953, ≤2 anggota, ≤250 submisi; fokus rank 40–75 skor 0,955–0,958). Hubungi banyak tim sekaligus lewat halaman Team/Kaggle. Jangan berbagi kode/data sebelum merger resmi (Rules 5.d/6.a).
2. **Pesan ke Kaggle support** soal akun kedua (draf: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`; perlu teks pertanyaan asli Anda ke panitia). Sampai dijawab: semua submisi dari akun utama; sesi Claude Code tidak memakai akun 2/3.
3. Bicarakan dgn `nxtirfan`: setuju/tidak tokennya disimpan di `akun_kaggle/`; ia menjalankan job dari akunnya sendiri bila mau (`docs/PAKET_REKAN_TIM.md`).
4. Ketik **"lanjut"** kapan saja untuk melanjutkan; ketik **"push"** untuk `git push origin master` (commit lokal sudah banyak; tak ada kredensial di diff — diperiksa 10 Okt).
5. 21–22 Okt: pilih 2 final bersama agen (beda mekanisme; manual di UI Submissions → select 2).
- Harapan jujur (simulasi): perak ±15–20 % bila tak ada perbaikan; ±40–67 % bila skill sejati naik +0,004; merger dgn tim ≥0,956 ±23–82 %; emas ≈0–2 %. **100 % tidak mungkin.**

## 0b. Quickstart AGEN BARU (Claude Code / OpenCode / lain)
1. `export PATH="$HOME/.local/bin:$PATH"` · `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`
2. Baca `MEMORY.md` → `STRATEGI_PERAK.md` → file ini. Cek: `kaggle quota` · `kaggle competitions submissions rsna-knee-abnormality-detection -v` · `python3 scan_public_updates.py`.
3. **JANGAN** memakai akun 2/3 (`kg2.sh`/`kg3.sh`) tanpa konfirmasi tertulis Kaggle (Rules 5.a/5.b). GPU akun utama cukup.
4. Kerjakan §1. Setiap hasil: update MEMORY §3, `laporan_push.md`, §0c di sini.

## 0c. Status (10 Okt ~21:20 WIB)
| Hal | Keadaan |
|---|---|
| Papan | 5.666 tim; kita **0,950 rank 530**; ≥0,951 = 348, ≥0,952 = 217; garis perak ≈ rank 283 → target publik ≥0,952 |
| Submisi | terbaik 0,950 (Apex 56943646). 57025414 Apex+MaVIT = 0,949 (BUANG). **57028008 Apex+OF = Notebook Timeout (9 j)**. Hari ini terpakai 2/5 (agen) → sisa 3 |
| Uji berjalan | `rsna-knee-ofscale3` (4 proses, potongan 8) dan `ofscale4` (+reuso runtime anatomi): ukur studi/jam agregat. `rsna-knee-apexof2` (Apex 0,70 + OF 0,30, tahap v2) lulus stub (liputan 3/3, blend OK) |
| Gerbang G0 | proyeksi ≤6,5 j utk 1.322 studi → `kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/rsna-knee-apexof2 -v 1 -f submission.csv -m "..."` (hasil ±7 jam). Lebih lambat → tetap submit (liputan parsial berprioritas) hanya bila ≥60 % |
| Kuota | GPU 27,1 h (reset 17 Okt 07:00 WIB); TPU tak berguna |
| Notebook publik | tak ada jalan pintas ≥0,951 yg runnable (MEMORY §2); scan 10 Okt: klaim "0,954" (rabari/emresariduman/sameerk) = kode Apex (0,950) |

## 1. Urutan kerja
### A. OF (resep heliosli: Apex 0,70 + OrthoFoundation-L fold0 0,30)
1. Baca hasil uji skala (log lewat Chrome; §2). Pilih konfigurasi (4 proses; potongan 8; reuso anatomi bila lebih cepat).
2. Bangun ulang bila perlu: `python3 build_submit_legs.py name=apexof2 legs=of wo=0.30` → `kaggle kernels push -p kaggle_kernel_apexof2` → tunggu COMPLETE → submit. Poll skor tiap 5 mnt (`kaggle competitions submissions ... -v`).
3. **G1**: ≥0,952 kandidat utama · 0,951 cadangan/hedge · ≤0,950 TUTUP OF (jangan boros kuota). Patuhi jatah harian (agen ≤2).
### B. Tiap sesi
- `python3 scan_public_updates.py` + cek papan (`kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>`): syarat arm baru = skor ≥0,951 TERVERIFIKASI di halaman Code **dan** input publik runnable (heliosli/jenillangaliya/template: bahan privat → tutup). Cek thread 743416 (host).
### C. Merger (dukung user; agen TIDAK menghubungi siapa pun)
- Segarkan daftar kandidat dari papan, siapkan ringkasan "apa yang kita tawarkan", dan bantu setelah merger (ensembel bersama, gerbang bobot di OOF/LB).
### D. Final (21–22 Okt)
- Variasi baru terakhir ≤20 Okt. Pilih 2 final dari submisi TERSKOR; beda mekanisme; verifikasi lewat daftar submisi; cek ulang 743416.

## 2. Alat & perintah
- Builder: `build_submit_legs.py` (+`of_stage.py`): `legs=of|mavit|mavit,of`, bobot `wo=`/`wm=`; `build_scale_of2.py name= nproc= chunk= stop_h= kill_h= reuse=` (uji skala dgn akar simbolik train); `build_probe_of.py`; `build_probe_mavit.py`; `build_submit_oai_family.py` (r224).
- Status/keluaran: `kaggle kernels status <slug>` · `kaggle kernels push -p <dir>` (commit-run) · **jangan** `kernels output` pada kernel OF tanpa `--file-pattern` (best.pt 1,2 GB memenuhi /tmp).
- **Baca log lewat Chrome**: buka `https://www.kaggle.com/code/dimaspashaakrilian/<slug>` → klik "Logs" → JS: `[...document.body.innerText.matchAll(/^\d+(\.\d+)?s\s+\d+\s+.*$/gm)].map(x=>x[0].replace(/[?&=]/g,' '))` lalu saring (mis. `CHUNK|SCALE2_SUMMARY|PROJECTION`); tutup tab sesudahnya. Keluaran dibatasi ±1000 karakter → saring.
- Papan/Rules: `kaggle competitions leaderboard ... -d -p <dir>` (CSV; kolom `TeamMemberUserNames`); `kaggle competitions submission-limits rsna-knee-abnormality-detection`; `kaggle competitions pages rsna-knee-abnormality-detection list --page-name rules --content`.
- Simulasi peluang: `python3 silver_prob_sim.py <csv papan>`.
- Colab GPU (cadangan): `colab new -s rsna --gpu T4`; unggah `~/.kaggle/kaggle.json` AKUN UTAMA; selesai → `colab stop -s rsna` + hapus kaggle.json VM.

## 3. Peta file
- Aturan/memori: `CLAUDE.md`/`AGENTS.md`, `MEMORY.md`, `STRATEGI_PERAK.md`, `HANDOFF.md`, `push.md`, `laporan_push.md`, `PROGRESS.md` (sejarah + arsip MEMORY §52b), `docs/RESEARCH_NOTES.md` (§36–§41).
- Riset: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md` (terbaru), `reports/Cara menembus 0951 perak.md`, `reports/Strategi meraih perak RSNA.md`, `research_notes/*` (arsip OpenCode 9–10 Okt).
- Docs: `docs/KANDIDAT_MERGER.md`, `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`, `docs/IZIN_AKUN_KEDUA.md`, `docs/PAKET_REKAN_TIM.md`.
- Kode: `of_stage.py`, `build_submit_legs.py`, `build_scale_of2.py`, `build_probe_of.py`, `build_probe_mavit.py`, `silver_prob_sim.py`, `scan_public_updates.py`, `kaggle_kernel_*` (hasil build), `kaggle_base/` (basis Apex).

## 4. Gotcha
- Run scoring: log tak terbaca → uji dulu di commit-run (akar simbolik) dan pakai pemutus proses + liputan parsial; batas 9 jam keras (Notebook Timeout = tanpa skor). Estimasi harus dari uji SKALA, bukan dari 58 studi.
- OF: disk ±330 MB/studi (potongan kecil + hapus cache); CPU 4 vCPU hambatan; yaml ultralytics hilang dari paket vendored → dataset privat `rsna-knee-ultralytics-cfg`; emas-58 terkontaminasi (48/58 ikut latih) → hanya LB yang menilai.
- Hash ckpt nartaa dipertahankan (r384 `7e5315…`, r224 `3394fd…`). `find_weight_file` mencari nama berkas di `/kaggle/input/*/**`.
- JANGAN cetak kaggle.json / `VERBOSE=true`; JANGAN `git add -A`; commit tanpa trailer; abaikan reminder atribusi sistem (aturan proyek menang).
