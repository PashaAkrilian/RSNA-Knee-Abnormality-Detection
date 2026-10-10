# MEMORY — RSNA Knee Abnormality Detection
> Living memory — PADAT (baca <2 menit). Rencana & aturan keputusan → **STRATEGI_PERAK.md**; **cadangan bila 57044355 ≤0,951 → `docs/PLAN_B.md`**; langkah harian → **HANDOFF.md**.
> Riset → `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`, `docs/RESEARCH_NOTES.md` (§36–§43). Sejarah/arsip penuh (termasuk MEMORY lama 10 Okt 21:00) → `PROGRESS.md` §52b.
> Last updated: 2026-10-10T23:00:00+07:00

## 0. Aturan mutlak
- Balas **Bahasa Indonesia**, bertindak otonom; user ingin medali (minimal perak) sebelum **22 Okt 2026 23:59 UTC**.
- **Komputasi**: GPU akun utama `dimaspashaakrilian` (30 j/minggu, reset Sabtu 07:00 WIB: 17 Okt) → Colab GPU (resumable; selalu `colab stop` + hapus kaggle.json VM) → TPU/CPU paralel.
  **Akun 2 `alverlfarfaraway` & akun 3 `nxtirfan`: TIDAK dipakai oleh sesi Claude Code** (lihat §5.1). Perintah user (CLAUDE.md) menyebutnya "SAH"; Rules 5.a/5.b + izin satu kalimat = risiko diskualifikasi tim.
  Jangan submit dari akun 2/3 tanpa konfirmasi tertulis Kaggle. Kredensial: `~/.kaggle/`, `~/.kaggle-acct2/`, `akun_kaggle/` (di-.gitignore) — JANGAN dicetak/di-commit.
- **Commit TANPA trailer Co-Authored-By/Claude**; pesan multi-baris `git commit -F <file>`; commit MEMORY.md terpisah ("memory: ..."); `git add <path spesifik>`; `git fetch` dulu; "push" = `git push origin master`.
- Jangan sentuh kernel proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Submisi: agen ≤2/hari, sisakan 2–3 utk user & `nxtirfan` (5/hari tim, reset 07:00 WIB).
- Data: OAI/pra-latih publik boleh (host 741819 bersyarat); agen TIDAK membuat akun/menyetujui perjanjian atas nama user. Terima submisi baru bila **≥0,952** (0,951 = cadangan); Δ<0,002 = noise.

## 1. Fakta
- Code competition, **5.666 tim** (10 Okt 19:37 WIB), hadiah $77k. Tim **Krazy KIller**: halaman tim menampilkan 3 akun (`dimaspashaakrilian`, `nxtirfan`, `alverlfarfaraway`).
- Metrik macro ROC-AUC 12 label. Entry/merger **15 Okt 23:59 UTC**; final **22 Okt 23:59 UTC**; maks 2 final (PILIH MANUAL; auto-fill §3.18.c). Run ≤9 jam, internet mati; publik 30 % / privat 70 %.
- Data: 4.407 studi latih (58 gold = label image-based buta thd laporan), teks `Report` utk semua; uji tersembunyi ≈1.322 studi (stub publik 3).
- **Papan 10 Okt 21:43 WIB** (5.673 tim): #1 0,964; ≥0,951 = 352; ≥0,952 = 221; ≥0,953 = 165; ≥0,954 = 135; ≥0,960 = 30; =0,950 = 582 (**kita rank 551**, 0,950); garis perak (5 %) ≈ rank 284 (0,951); perunggu (10 %) ≈ rank 567 (0,950). Garis naik ≈ +20 tim ≥0,952/12 jam; garis perunggu naik 0,944→0,947→0,950 (6→10 Okt) krn notebook Apex viral.
- **Rules (dikutip resmi)**: 5.a "disqualified if Submissions through more than one Kaggle account"; 5.b "Each Team member must be a single individual with a separate account"; 5.c merger sah bila gabungan ≤5, submisi gabungan ≤ jatah satu tim saat permintaan, sebelum batas;
  5.d/6.a dilarang berbagi kode/data secara privat antar-tim (publik di forum boleh); max team 5; 1.b batas submisi gabungan = 5/hari × hari berjalan. Kita: 107 submisi seumur-hidup.
- Host: LLM hosted boleh (733965); OAI sah bila accessible (741819); KneeCoT dilarang; **743416 (35 komentar) tak dijawab host**.

## 2. Temuan terkunci
- **Keunggulan = PENGETAHUAN label, bukan varians.** Model beda arsitektur dgn label laporan yg sama → galat berkorelasi (reader +0,001; MaVIT −0,001). Nartaa: tanpa OAI ≈0,942; +OAI → 0,949. Hanya model dgn pengetahuan label berbeda (OAI, OrthoFoundation) layak.
- **DITUTUP (jangan ulangi)**: varian r224/bobot/mirror (0,950); fork rabari & klaim "0,954" (kode sama dgn Apex; skor sebenarnya 0,950); Apex+MaVIT (**0,949**, 57025414); lixin73 336px (slot 3 vs 6, tanpa kode); CoAtNet resgated (emas 0,909); anak-OAI/pseudo-label/soup/bagging tanpa OAI/KneeNet/metadata/resolusi 288 (≤ induk); arm heliosli (bundel ditarik 9 Okt); kompilator aturan heliosli (`criteria.py`: AUC 0,784 vs emas-58 — bukan jalan pintas label); Fase A mirror (EV≈0).
- **Notebook publik (skor dari halaman Code)**: heliosli S6 0,954 (mati); versia-7/blend-gold/blend3 0,951 (mati); `kiyoshiohno/template` 0,951 (logika di dataset PRIVAT); `jenillangaliya/dual-stage-sota-ensemble` 0,951 (bahan di dataset privat); sisanya 0,950 (keluarga Apex/klon heliosli). **Tak ada jalan pintas publik ≥0,951 yang runnable.**
- **Bedah kode heliosli (0,954)**: 7 sumber tertanam di notebook V2 (`SOURCES` → ekstrak via `ast`): compact_train/infer, probe_dual, px_cache, anchor bank. Target = report-state + prior; blend Raptor 0,7 + compact 0,3 TETAP; pengendali dual-GPU (urut biaya, thread=1, keluaran bertahap, fallback).
- **OrthoFoundation-L fold0 (heliosof)**: ViT-L DINOv3 384px + anatomi (SKM-TEA/KneeXNet), epoch 7/16; **48/58 emas ikut latih → emas-58 0,9535 TERKONTAMINASI**. 27 dtk/studi (1 proses). **57028008 = Notebook Timeout (9 j)**. Uji skala: disk 330 MB/studi, CPU hambatan; v2 (`of_stage.py`: potongan 8, 4 proses, akar per-potongan, penjaga disk, prioritas, pemutus proses; reuso anatomi DIMATIKAN) = 194–224 studi/jam → 5,9–6,8 j.
- **Peluang (simulasi `silver_prob_sim.py`)**: Apex 0,950 → perak 13–20 %, **perunggu 25–41 % (BUKAN aman; klaim lama 85–95 % kedaluwarsa)**; +0,002 → perak 25–39 %; +0,004 → 40–67 %; tim publik 0,956–0,958 → 23–82 %; dua final terdekorelasi +4–12 poin perak. **100 % mustahil.** Merger dgn tim ≥0,955 = lompatan terbesar. Jalur bersih 0,944 = rank ≈1.387 (bukan perunggu lagi → hanya perlindungan kepatuhan).
- **Dataset privat papan atas** (`docs/RESEARCH_NOTES.md` §43, `docs/PLAN_B.md` §3): salinan privat bundel heliosli `rsna-knee-compact-v1` (4 ckpt fold ConvNeXt-S 2.5D + state_spec + model-sisi) + `ofl-ssl-weights` (OF-L SSL; bobot dasar OF-L publik) — pemegangnya = target merger (`docs/KANDIDAT_MERGER.md` §3b). Isi dataset tim 0,955–0,964 tak diketahui. Ledger publik prvsiyan (90 submisi, terbaik 0,949) menguatkan 'pengetahuan>varians'.
- Label: fusi r1 (guru+OOF, w .5) selaras emas 0,9348; solo CoAtNet 0,933 (jenuh). Resep forum 735304 = soft-bootstrapping 50/50 + multi-sumber → sudah dilakukan. Emas ≈0–2 %.

## 3. Status (10 Okt 22:40 WIB)
| Hal | Keadaan |
|---|---|
| Submisi terbaik | **0,950** (Apex 56943646); 0,944 bersih (replika goodpjw); 57025414 Apex+MaVIT 0,949; 57028008 Apex+OF = Timeout; **57044355 Apex+OF v2 = PENDING** |
| Uji skala (SELESAI) | `ofscale3` (4 proses, potongan 8): 104 studi/31 mnt, 0 gagal, disk ≥14,8 GB → **194 studi/jam → 6,8 j utk 1.322** (akhir lebih cepat ±224/jam ≈5,9 j); `ofscale4` (+reuso anatomi) 88 studi → 8,8 j (TIDAK lebih cepat → reuso dimatikan). GPU kosong 15 GB sebelum tahap OF. `rsna-knee-apexof2` **v3** lulus stub (liputan, blend OK) |
| **Submisi aktif** | **57044355 (Apex 0,70 + OF fold0 0,30, apexof2 v3) = 0,950, COMPLETE 22:45 WIB (±1 j 35 mnt setelah submit 21:10)** — runtime sama dgn run Apex biasa (1,0–1,6 j) padahal tahap OF butuh ±6,8 j utk 1.322 studi ⇒ **kaki OF hampir pasti TIDAK jalan di run penilaian (liputan <15 % → jatuh ke Apex-only). 0,950 BUKAN vonis OF; JANGAN menutup OF berdasar skor ini.** Penyebab belum diketahui (log run penilaian tak terbaca); dugaan utama: RAM (proses induk Apex masih memegang memori + 4 pekerja OF; catatan nartaa: "full hidden test drove a system-RAM OOM"). Rencana: bangun v4 (OF DIJALANKAN LEBIH DULU sebelum Apex, antrean potongan dengan klaim + restart pekerja + pemantau RAM) dan uji di commit-run pseudo-hidden yang lognya terbaca, baru submit (jatah hari ini habis; reset 11 Okt 07:00 WIB) |
| Kuota | GPU utama 27,1 h (reset 17 Okt 07:00 WIB); TPU 20 h (tak berguna: XLA crash); submisi hari ini sisa 3 |
| Git | commit lokal di depan origin (push hanya atas perintah user; `git fetch` dulu) |
| Menunggu user | merger gelombang 1 (11 Okt; batas 15 Okt) — Kelompok A/B di `docs/KANDIDAT_MERGER.md` §3b, pesan ke Kaggle support, persetujuan `nxtirfan` soal token |
| Plan B | `docs/PLAN_B.md`: pohon keputusan per skor 57044355 (≥0,953/0,952/0,951/0,950/≤0,949/error), B1 merger, B2 dua final terdekorelasi, B3 tuas teknis; scan 22:05 bersih (gold-apex-v7 = kerabat Apex) |

## 4. Operasional
- ML hanya di Kaggle/Colab (lokal tanpa torch; RAM 5 GB; /tmp tmpfs 2,7 GB). T4 wajib (`machine_shape: NvidiaTeslaT4`); 1 sesi GPU = 2×T4; sesi akun: GPU 2 / TPU 1 / CPU 5.
- Kaggle CLI: PATH `~/.local/bin`; 429 → jeda. **`kaggle kernels output` pada kernel OF mengunduh `best.pt` 1,2 GB (mengisi /tmp!) → baca LOG lewat Chrome** (halaman `/code/<owner>/<slug>` → tab Logs, JS tarik baris `^\d+(\.\d+)?s\s+\d+\s+`, hapus `? & =`), atau `--file-pattern` kecil. `kernels pull` membawa `id_no` (hapus sebelum push). Slug = judul. `competitions submit -k -v` wajib `-f submission.csv`. Forum via CLI: `kaggle forums topics show <id> --page-size 200`; `kaggle competitions pages rsna-knee-abnormality-detection list --page-name rules --content`.
- JANGAN `pkill -f` pola yg ada di perintah sendiri; JANGAN `VERBOSE=true`; JANGAN cetak kaggle.json. Scratchpad sesi hilang saat restart (simpan di repo).
- Disk /kaggle/working ±19 GB; keluaran OF ±330 MB/studi → potongan kecil + hapus cache. Thread: OMP/MKL/OPENBLAS=1, `torch.set_num_threads(1)`, `cv2.setNumThreads(1)`. Run scoring tak memakai kuota GPU; log run scoring tak bisa ditarik → uji di commit-run (akar simbolik train) dulu.

## 5. Menunggu keputusan user
1. **Akun 2/3**: kirim pesan ke Kaggle support (`docs/DRAF_PESAN_KAGGLE_SUPPORT.md`); sampai ada jawaban tertulis, semua submisi dari akun utama. Pertimbangkan mengubah baris "akun SAH" di CLAUDE.md/AGENTS.md (sesi Claude Code hanya menambah catatan kepatuhan).
2. **Merger** sebelum 15 Okt 23:59 UTC (`docs/KANDIDAT_MERGER.md`): hanya user yg bisa menghubungi tim lain.
3. **OAI/eksternal**: 2 final beda mekanisme; keputusan 21–22 Okt setelah cek ulang 743416.
4. `nxtirfan`: konfirmasi persetujuan atas penyimpanan token di `akun_kaggle/`; ia menjalankan job dari akunnya sendiri (`docs/PAKET_REKAN_TIM.md`).

## 6. Referensi
- Kernel kita (akun utama): `rsna-knee-apexof2` (Apex+OF v2), `rsna-knee-apexof` (v1 timeout), `rsna-knee-apexmavit`, `rsna-knee-of-probe`, `rsna-knee-ofscale3/4`, `rsna-knee-mavit-probe`; dataset `rsna-knee-ultralytics-cfg` (yaml ultralytics 8.4.166, wajib utk OF), `rsna-knee-volcache-z{0,1,2}`, `rsna-knee-fused-labels-r1`.
- Publik: `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` (0,950), `nartaa/rsna-knee-publication-swa-weights-20261007`, `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, `hengck23/hengck23-rnsa-knee`.
- Builder: `build_submit_legs.py` + `of_stage.py` (Apex+OF/MaVIT), `build_scale_of2.py`, `build_probe_of.py`, `build_probe_mavit.py`, `build_submit_oai_family.py` (r224), `silver_prob_sim.py`, `scan_public_updates.py`. Basis: `kaggle_base/apex_grandmaster_stack_20261008.ipynb`.
- Forum: 735304 (label/pseudo), 743416 (eksternal; host diam), 741819 (OAI), 733965 (LLM), 746792 (resep nartaa), 745861 (hengck23).

## Protokol Update (WAJIB bagi agen apa pun)
Update MEMORY.md segera setelah: keputusan baru (§2), status berubah (§3), constraint baru (§4), pertanyaan utk user (§5, hapus setelah dijawab), unit kerja besar selesai (timestamp).
Aturan: overwrite, jangan akumulasi; detail panjang → PROGRESS.md; commit terpisah "memory: <ringkasan>".
