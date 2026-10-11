# MEMORY — RSNA Knee Abnormality Detection
> Living memory — PADAT (baca <2 menit). Rencana & aturan keputusan → **STRATEGI_PERAK.md** (§0 = perubahan 11 Okt); arah/cadangan → **`docs/PLAN_B.md` §00**; langkah harian → **HANDOFF.md**.
> Riset → `reports/Ringkasan riset 10 Okt (sesi Claude Code).md` (§10 = 11 Okt), `docs/RESEARCH_NOTES.md` (§36–§44). Sejarah/arsip penuh (MEMORY lama 10 Okt 21:00) → `PROGRESS.md` §52b; narasi 11 Okt → `PROGRESS.md` §54–§55.
> Last updated: 2026-10-11T09:45:00+07:00

## 0. Aturan mutlak
- Balas **Bahasa Indonesia** (dokumen & pesan commit juga), bertindak otonom; user ingin medali (minimal perak) sebelum **22 Okt 2026 23:59 UTC**.
- **Komputasi**: GPU akun utama `dimaspashaakrilian` (30 j/minggu, reset Sabtu 07:00 WIB: 17 Okt; maks 2 sesi GPU, sesi antre dihitung) → Colab GPU → TPU/CPU. **Akun 2 `alverlfarfaraway` & akun 3 `nxtirfan`: TIDAK dipakai oleh sesi Claude Code** (Rules 5.a/5.b + izin satu kalimat = risiko diskualifikasi tim; CLAUDE.md menyebutnya "SAH"). Kredensial `~/.kaggle/`, `akun_kaggle/` — JANGAN dicetak/di-commit.
- **Commit TANPA trailer Co-Authored-By/Claude**; pesan multi-baris `git commit -F <file>`; commit MEMORY.md terpisah ("memory: ..."); `git add <path spesifik>`; `git fetch` dulu; "push" = `git push origin master` (hanya atas perintah user).
- Jangan sentuh kernel proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Submisi: agen ≤2/hari, sisakan 2–3 utk user & `nxtirfan` (5/hari tim, reset 07:00 WIB).
- Data: OAI/pra-latih publik boleh (Rules 2.6.b; host diam 741819/743416/744056/747455); agen TIDAK membuat akun/menyetujui perjanjian atas nama user. Terima varian bila ≥ klon S6 (≈0,954) dlm noise ATAU menambah diversitas murah; Δ<0,002 = noise.

## 1. Fakta
- Code competition, **5.699 tim** (11 Okt 02:59 WIB; 12 hari lagi), hadiah $77k. Tim **Krazy KIller** (3 akun di halaman tim). Metrik macro ROC-AUC 12 label. Entry/merger **15 Okt 23:59 UTC**; final **22 Okt 23:59 UTC**; maks 2 final (PILIH MANUAL). Run ≤9 jam, internet mati; publik 30 % / privat 70 %.
- Data: 4.407 studi latih (58 gold), teks `Report` utk semua; uji tersembunyi ≈1.322 studi. Mesin run: T4×2, 4 vCPU, RAM ±31 GB, /kaggle/working 20 GB.
- **Papan 11 Okt 06:40 WIB** (5.709 tim): #1 0,964; ≥0,960 = 32; ≥0,956 = 82; ≥0,955 = 101; ≥0,954 = 150 (49 tepat di 0,954); ≥0,952 = 238; ≥0,951 = 386; =0,950 = 622; **kita rank 144 (0,954 = klon S6 user 57051680)**; perak (5 %) ≈ rank 285 (0,951); perunggu ≈ rank 571 (0,950). Dataran 0,954 mulai rank 102 dan akan membengkak (S6 publik); garis perak 285 jatuh di dalam dataran bila ≥184 tim di 0,954.
- **Rules**: 5.a "disqualified if Submissions through more than one Kaggle account"; 5.b anggota = individu terpisah; 5.c merger sah bila gabungan ≤5, submisi gabungan ≤ jatah satu tim saat permintaan (11 Okt = 335; kita 109 → mitra ≤±225), sebelum batas; 5.d/6.a dilarang berbagi privat antar-tim; 2.6.b model/data eksternal boleh kecuali dilarang host; 2.8/3.14.a menyangkut kelayakan hadiah (belum dijawab host).

## 2. Temuan terkunci
- **KEPUTUSAN USER (11 Okt 09:45 WIB)**: **TANPA merger & TANPA pesan ke Kaggle support** — fokus naikkan skor sendiri dengan tim ini (J1 & tindakan support DIBATALKAN; arsip `docs/KANDIDAT_MERGER.md`/`docs/DRAF_PESAN_KAGGLE_SUPPORT.md` jangan dieksekusi); submisi hanya akun utama.
- **Pengetahuan label > varians**: model beda arsitektur dgn label sama berkorelasi (reader +0,001; MaVIT −0,001; r224/mirror 0). Nartaa tanpa OAI ≈0,942, +OAI 0,949; compact report-state (heliosli) menambah +0,005 di atas Raptor (S6 0,954). Forum: pemisah papan atas = label kaya + pseudo-label guru→murid (ConvNeXt-T 5-fold single 0,953–0,959); validasi OOF-label r=0,87 vs LB.
- **S6 (heliosli) HIDUP LAGI (11 Okt 02:50 WIB)**: `heliosli/rsna-knee-compact-v1` publik (V1–V13; V10=V13 `209a32c8` cnx_s6: ConvNeXt-S 4 fold, 384/352, state 'opusv11b') + notebook publik = **Raptor 0,7 + compact 0,3 = 0,954 satu klik** (thread 748141). Tahap compact = decode `native_cache` + GPU (2 pekerja ±18–20 dtk/studi) ≈3,5 j; kontroler ERROR bila >3 % fallback; DEADLINE 8,3 j. **Salinan user 57051680 = 0,954 COMPLETE (skor ±06:40 WIB; run penuh ±3 j 45 mnt dari submit 02:56 → compact ±3,2 j) — S6 TERBUKTI jalan di run penilaian; kita rank 144.** Zoologi bundel: V4 ofl_ft8, V7 cnx_s5 (0,951), V8 dino_s1 (DINOv3-B), V9 dual.
- **KOREKSI**: resep heliosli = Raptor + COMPACT, BUKAN OF; bobot OF fold0 tak pernah dipakai di notebook heliosli (hanya kodenya). OF fold0 (ViT-L; validasi lemah 0,9318; emas-58 0,9535 TERKONTAMINASI 48/58) belum pernah diukur di LB → prior ≥+0,001 di atas S6 ≤20 %.
- **57044355 = 0,950** karena kaki OF mati OOM (induk Apex 18 GB RSS + pekerja OF 8–10 GB; terkonfirmasi `apexof4d`). **Perbaikan v4.1** (`of_stage_v4.py`, `build_apexof4.py`): supervisor klaim/restart/daur-ulang, `pin_memory=False`, `torch.load(mmap=True)` (puncak/pekerja 8,8→5,1 GB); rehearsal r2 LULUS (194 studi/jam, 3 pekerja). Dipakai ulang di `build_s6plus.py`.
- **DITUTUP (jangan ulangi)**: r224/bobot/mirror (0,950); "Tri-Reader 0,954" (rabari/jcgweb/sameerk = Apex+r224); Apex+MaVIT 0,949; CoAtNet resgated; anak-OAI/pseudo-label/soup (≤ induk); kompilator aturan heliosli; klon DINOsaur; tri-hybrid jenillangaliya 0,952 (< S6).
- **Peluang (simulasi)**: Apex 0,950 → perak 13–20 %/perunggu 25–41 %; klon S6 di klaster ratusan tim → perak ±25–45 %/perunggu ±45–70 %; dua final berkorelasi tinggi +3–6 poin; merger dgn tim ≥0,956 → perak 23–82 %; emas ≈0–2 %; **100 % mustahil**. Trik "kirim 1−p" → tim kuat tersembunyi.

## 3. Status (11 Okt 09:45 WIB)
| Hal | Keadaan |
|---|---|
| Submisi | **terbaik 0,954 (57051680 = klon S6 oleh user, COMPLETE ±06:40; rank 144)**; Apex 56943646 0,950; 57025414 Apex+MaVIT 0,949; 57028008 Timeout; 57044355 = 0,950 (OF OOM; bukan vonis OF). Jendela 11 Okt: **s6reader 57055673 + s6plus 57055781 di-submit 07:03 (PENDING; monitor jalan)**; sisa 3/5 |
| Rehearsal | `apexof4r3` LULUS (4 pekerja 228 studi/jam, MemAvail min 13,8 GB); `s6cleanr` LULUS (compact 160 studi = 1.812 dtk, 0 fallback); `s6plusr` (S6+reader+OF terakhir pseudo-uji 160) LULUS (OF coverage 0,65 dlm 24 mnt; blend 3 perna OK; MemAvail min 16 GB); sonda `s6probe` (1.530 dtk; ERROR akhir = artefak pseudo, diperbaiki). RAM compact stabil ±16–17 GB bebas (tak ada kebocoran) |
| Bahan siap | `build_s6plus.py`: **`s6reader`** (reader ×0,5 pada perna Raptor) dan **`s6plus`** (reader ×1,0 + OF 0,15 terakhir) — commit-run stub LULUS (v1, COMPLETE 04:24/04:34); **`s6clean`** (compact + reader tanpa Raptor/OAI; rehearsal `s6cleanr` LULUS; asuransi, submit ≥12 Okt). **Pewaktu `scripts/submit_two.sh` SUDAH men-submit `s6reader` v1 (57055673) dan `s6plus` v1 (57055781) pada 11 Okt 07:03 WIB (agen 2/5; keduanya PENDING)**; monitor `progress_submisi.sh` ×2 jalan |
| Kuota | GPU utama ±19 h (reset 17 Okt 07:00 WIB); TPU tak berguna |
| Git | 65 commit lokal di-push 11 Okt ~09:50 WIB atas perintah user; diff diperiksa: tak ada kredensial |
| Menunggu user | ~~merger & pesan support~~ **DIBATALKAN (user 11 Okt: fokus skor sendiri)**; sisa: pilihan 2 final 21–22 Okt; `nxtirfan` (token) |
| Vigia | `watch.sh` (scratchpad) memantau heliosli/nartaa/goodpjw + notebook baru tiap 30 mnt → `watch.log` (cari S7) |

## 4. Operasional
- ML hanya di Kaggle/Colab (lokal tanpa torch; RAM 5 GB; /tmp tmpfs 2,7 GB). `machine_shape: NvidiaTeslaT4` (2×T4). Maks 2 sesi GPU batch; sesi ANTRE tetap dihitung (batalkan lewat Chrome: sidebar "View Active Events" → ⋯ → Stop Session). `kernels push` baru = sesi baru.
- Kaggle CLI: PATH `~/.local/bin`; 429 → jeda. **Log lewat Chrome**: `/code/<owner>/<slug>/log` (log langsung terlambat ±5 mnt; lengkap setelah selesai) + JS tarik baris `^\d+(\.\d+)?s\s+\d+\s+`, hapus `? & =`; atau `kernels output --file-pattern '<regex>'` (JANGAN tanpa pola pada kernel OF/S6). Forum: `kaggle forums topics show <id> --page-size 200` (daftar topik CLI 403 → Chrome). Versi dataset lama: venv kagglehub di scratchpad.
- **RAM**: proses induk notebook memegang ±18 GB setelah Raptor (Apex); pekerja OF 5–10 GB; headroom cgroup menyertakan page-cache (pakai MemAvailable). OF hanya via supervisor `nopin+mmap` dan di luar induk besar. Pseudo-uji: `sample_submission.csv` WAJIB berisi id pseudo.
- JANGAN `pkill -f` pola yg ada di perintah sendiri; JANGAN `VERBOSE=true`; JANGAN cetak kaggle.json. Scratchpad sesi hilang saat restart (simpan di repo). Run scoring tak memakai kuota GPU; log run scoring tak terbaca → uji di commit-run pseudo-uji.

## 5. Menunggu keputusan user
1. **OAI/eksternal**: final A = S6/varian terbaik; final B = beda mekanisme (varian S6+ atau bersih tanpa Raptor); keputusan 21–22 Okt setelah cek ulang 743416.
2. `nxtirfan`: persetujuan token di `akun_kaggle/`; ia menjalankan job dari akunnya sendiri (`docs/PAKET_REKAN_TIM.md`).

## 6. Referensi
- Kernel kita (akun utama): `rsna-knee-s6probe/s6plusr/s6reader/s6plus/s6clean(r)` (S6+), `rsna-knee-apexof4/4t/4r/4r2/4r3/4d` (Apex+OF v4.1), `rsna-knee-ofmem1/3` (profil memori), lama: `apexof2`, `apexof`, `apexmavit`, `ofscale3/4`; dataset `rsna-knee-ultralytics-cfg` (yaml ultralytics, wajib utk OF), `rsna-knee-volcache-z{0,1,2}`, `rsna-knee-fused-labels-r1`.
- Publik: `heliosli/dataset-restored-i-d-been-told-it-wasn-t-allowed` (S6), `heliosli/rsna-knee-compact-v1`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (kode decode + OF), `nartaa/rsna-knee-publication-swa-weights-20261007`, `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack`, `prvsiyan/rsna-knee-2026-experiment-ledger`.
- Builder: `build_s6plus.py`, `build_s6probe.py`, `build_apexof4.py` + `of_stage_v4.py` (+`test_of_v4_local.py`), `build_ofmem.py`, `silver_prob_sim.py`, `scan_public_updates.py`; basis: `kaggle_base/` (Apex, S6 publik).
- Forum: 748141 (S6 dipulihkan), 735304 (best single-model), 748011 (validasi), 744056 (kelayakan hadiah), 743416/747455/741819 (eksternal/OAI), 733965 (LLM), 746792 (resep nartaa), 745861 (hengck23), 743374 (Prateek: I/O DICOM; mencari rekan).

## Protokol Update (WAJIB bagi agen apa pun)
Update MEMORY.md segera setelah: keputusan baru (§2), status berubah (§3), constraint baru (§4), pertanyaan utk user (§5, hapus setelah dijawab), unit kerja besar selesai (timestamp).
Aturan: overwrite, jangan akumulasi; detail panjang → PROGRESS.md; commit terpisah "memory: <ringkasan>".
