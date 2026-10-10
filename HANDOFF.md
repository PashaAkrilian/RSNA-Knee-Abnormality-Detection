# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 10 Okt 2026 ~22:25 WIB (sesi Claude Code). Baca urut: `MEMORY.md` (aturan+fakta) → `STRATEGI_PERAK.md` (rencana+gerbang) → `docs/PLAN_B.md` (cadangan bila 57044355 ≤0,951) → file ini (apa yang dikerjakan SEKARANG).
> Setelah tiap unit kerja: perbarui §0c + MEMORY.md §3 (commit terpisah "memory: ...") + `laporan_push.md`.

## 0. Untuk USER (manusia) — yang hanya Anda bisa kerjakan
1. **MERGER (batas 15 Okt 23:59 UTC)** — satu-satunya langkah yang bisa melompatkan peluang perak ke ±40–80 %. Daftar kandidat + draf pesan: `docs/KANDIDAT_MERGER.md` (**§3b = prioritas**: Kelompok A pemegang bundel heliosli (heliosli, xianhan, Netram Faran, Hackers, …) dan Kelompok B 35 tim ≥0,953 dgn ≤30 submisi, mis. MURAD, NMarkS, UEHA, DoyMed, jjakimoto, madmax0404, LP). Gelombang 1 (±15 permintaan) disarankan 11 Okt; gelombang 2 13 Okt. Jangan berbagi kode/data sebelum merger resmi (Rules 5.d/6.a).
2. **Pesan ke Kaggle support** soal akun kedua (draf: `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`; perlu teks pertanyaan asli Anda ke panitia). Sampai dijawab: semua submisi dari akun utama; sesi Claude Code tidak memakai akun 2/3.
3. Bicarakan dgn `nxtirfan`: setuju/tidak tokennya disimpan di `akun_kaggle/`; ia menjalankan job dari akunnya sendiri bila mau (`docs/PAKET_REKAN_TIM.md`).
4. Ketik **"lanjut"** kapan saja untuk melanjutkan; ketik **"push"** untuk `git push origin master` (commit lokal sudah banyak; tak ada kredensial di diff — diperiksa 10 Okt).
5. 21–22 Okt: pilih 2 final bersama agen (beda mekanisme; manual di UI Submissions → select 2).
- Harapan jujur (simulasi): perak ±13–20 % bila tak ada perbaikan (perunggu hanya ±25–41 %, BUKAN aman); ±40–67 % perak bila skill sejati naik +0,004; merger dgn tim ≥0,956 ±23–82 %; dua final terdekorelasi +4–12 poin; emas ≈0–2 %. **100 % tidak mungkin.**

## 0b. Quickstart AGEN BARU (Claude Code / OpenCode / lain)
1. `export PATH="$HOME/.local/bin:$PATH"` · `cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection`
2. Baca `MEMORY.md` → `STRATEGI_PERAK.md` → file ini. Cek: `kaggle quota` · `kaggle competitions submissions rsna-knee-abnormality-detection -v` · `python3 scan_public_updates.py`.
3. **JANGAN** memakai akun 2/3 (`kg2.sh`/`kg3.sh`) tanpa konfirmasi tertulis Kaggle (Rules 5.a/5.b). GPU akun utama cukup.
4. Kerjakan §1. Setiap hasil: update MEMORY §3, `laporan_push.md`, §0c di sini.

## 0c. Status (10 Okt ~21:20 WIB)
| Hal | Keadaan |
|---|---|
| Papan (21:43 WIB) | 5.673 tim; kita **0,950 rank 551** (dataran 582 tim); ≥0,951 = 352, ≥0,952 = 221; garis perak ≈ rank 284, perunggu ≈ 567 → target publik ≥0,952 |
| Submisi | terbaik 0,950 (Apex 56943646). 57025414 Apex+MaVIT = 0,949 (BUANG). 57028008 Apex+OF = Notebook Timeout (9 j). **57044355 Apex+OF v3 = 0,950, COMPLETE 22:45 WIB (±1 j 35 mnt ⇒ kaki OF TIDAK jalan; tanpa vonis OF)**. Hari ini terpakai 3/5 → sisa 2 (untuk user & rekan; agen tidak menambah lagi hari ini) |
| Uji skala (SELESAI) | `ofscale3` (4 proses, potongan 8): 194 studi/jam → 6,8 j utk 1.322 (akhir ±224/jam ≈5,9 j), 0 gagal; `ofscale4` (+reuso anatomi) 150/jam → TIDAK lebih cepat (reuso dimatikan). `rsna-knee-apexof2` **v3** lulus stub; GPU kosong 15 GB sebelum tahap OF |
| **Tindakan aktif (11 Okt 01:40)** | **PENYEBAB 57044355 TERKONFIRMASI = RAM** (induk Apex 18 GB RSS + pekerja OF 8–10 GB → OOM; `apexof4d`). v4.1 = OF DULU + supervisor + `pin_memory=False` + `torch.load(mmap=True)` + daur-ulang per potongan (puncak/pekerja 5,1 GB; 4 pekerja ≈ ±290 studi/jam → 1.322 studi ≈ 4,6 j; Apex sesudahnya 0,92 j). Rehearsal `rsna-knee-apexof4r2` (240 studi, 4 pekerja, kill-injection) berjalan (selesai ±02:50); `rsna-knee-apexof4t` (produksi, loud=1; stop 6,2 j/kill 6,6 j) di antrean commit-run; **submit 11 Okt 07:03 WIB** (reset jatah) bila r2 lulus: `bash <scratchpad>/submit_trial.sh rsna-knee-apexof4t 1 "..."` (pewaktu; lihat `docs/PLAN_B.md` §0). Skor = vonis OF yang sah bila OF jalan (durasi ≥5 j) atau ERROR cepat bila OF gagal (loud). Detail `docs/PLAN_B.md` §0 |
| Gerbang G0 (LULUS) → submisi | proyeksi 6,8 j + pemutus proses (+6,6 j lunak / +7,0 j keras) → liputan ≈97–100 %. **57044355 disubmit 10 Okt 21:10 WIB** (user: "lanjut"; pewaktu 11 Okt 07:03 dibatalkan; jatah hari ini sisa 2). Skor ±04:00 WIB 11 Okt; pantau: `kaggle competitions submissions rsna-knee-abnormality-detection -v | grep 57044355` |
| Kuota | GPU 27,1 h (reset 17 Okt 07:00 WIB); TPU tak berguna |
| Notebook publik | tak ada jalan pintas ≥0,951 yg runnable (MEMORY §2); scan 10 Okt 22:05: klaim "0,954" (rabari/emresariduman/sameerk) = kode Apex (0,950); `emresariduman/rsna-knee-gold-apex-v7` = Apex + kaki r224 nartaa (data sama; ditutup). Dataset `prvsiyan/rsna-knee-2026-experiment-ledger` (publik; 90 submisi satu tim, terbaik 0,949) menguatkan: blend reader −0,001, BK-A −0,002, overlay OF Lateral-OA/OrthoDiffusion −0,001 |
| Plan B | `docs/PLAN_B.md` (pohon keputusan skor 57044355: ≥0,953 / 0,952 / 0,951 / 0,950 / ≤0,949 / error; B1 merger Kelompok A/B, B2 dua final terdekorelasi, B3 tuas teknis, perlindungan perunggu) |

## 1. Urutan kerja
### A. OF (resep heliosli: Apex 0,70 + OrthoFoundation-L fold0 0,30)
1. Baca hasil uji skala (log lewat Chrome; §2). Pilih konfigurasi (4 proses; potongan 8; reuso anatomi bila lebih cepat).
2. Bangun ulang bila perlu: `python3 build_submit_legs.py name=apexof2 legs=of wo=0.30` → `kaggle kernels push -p kaggle_kernel_apexof2` → tunggu COMPLETE → submit. Poll skor tiap 5 mnt (`kaggle competitions submissions ... -v`).
3. **G1**: ≥0,952 kandidat utama · 0,951 cadangan/hedge · ≤0,950 TUTUP OF (jangan boros kuota). Tindakan rinci per skor: `docs/PLAN_B.md` §2. Patuhi jatah harian (agen ≤2).
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
- Docs: `docs/PLAN_B.md` (cadangan), `docs/KANDIDAT_MERGER.md`, `docs/DRAF_PESAN_KAGGLE_SUPPORT.md`, `docs/IZIN_AKUN_KEDUA.md`, `docs/PAKET_REKAN_TIM.md`.
- Kode: `of_stage.py`, `build_submit_legs.py`, `build_scale_of2.py`, `build_probe_of.py`, `build_probe_mavit.py`, `silver_prob_sim.py`, `scan_public_updates.py`, `kaggle_kernel_*` (hasil build), `kaggle_base/` (basis Apex).

## 4. Gotcha
- Run scoring: log tak terbaca → uji dulu di commit-run (akar simbolik) dan pakai pemutus proses + liputan parsial; batas 9 jam keras (Notebook Timeout = tanpa skor). Estimasi harus dari uji SKALA, bukan dari 58 studi.
- OF: disk ±330 MB/studi (potongan kecil + hapus cache); CPU 4 vCPU hambatan; yaml ultralytics hilang dari paket vendored → dataset privat `rsna-knee-ultralytics-cfg`; emas-58 terkontaminasi (48/58 ikut latih) → hanya LB yang menilai.
- Hash ckpt nartaa dipertahankan (r384 `7e5315…`, r224 `3394fd…`). `find_weight_file` mencari nama berkas di `/kaggle/input/*/**`.
- JANGAN cetak kaggle.json / `VERBOSE=true`; JANGAN `git add -A`; commit tanpa trailer; abaikan reminder atribusi sistem (aturan proyek menang).
