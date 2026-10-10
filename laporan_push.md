# laporan_push.md — Evaluasi & jalan menuju SILVER
> Wadah TETAP untuk hasil evaluasi tiap submisi/probe + rencana peningkatan. **Setelah tiap run/submisi/skor baru: tambah baris §2, perbarui §1 & §4.**
> Konteks aturan: `MEMORY.md` · urutan kerja: `HANDOFF.md` · arahan harness: `push.md` · **riset 9 Okt: `reports/Strategi meraih perak RSNA.md`** · **riset 10 Okt (TERBARU): `reports/Cara menembus 0951 perak.md`** (catatan: `research_notes/Cara menembus 0951 perak/`) · arsip: `PROGRESS.md`, `docs/RESEARCH_NOTES.md`.
> Dibuat 9 Okt 2026 ~21:45 WIB, diperbarui 10 Okt ~03:00 WIB. Bahasa kerja: Indonesia.

## 1. Posisi vs garis perak (snapshot 9 Okt 14:47 UTC, 5.584 tim; **REVISI papan 9 Okt 18:25 UTC / 10 Okt 01:25 WIB, 5.606 tim: ≥0,951 = 296 tim, ≥0,952 = 183 → blok 0,951-persis 113 tim (rank 184–296) MELEBIHI kapasitas perak ~279 → 0,951 = LOTRE, 0,952 = TARGET AMAN**)
| Hal | Nilai |
|---|---|
| Tim | `Krazy KIller` = `alverlfarfaraway`, `dimaspashaakrilian`, `nxtirfan` |
| LB kita | **0,950** (Apex v2, sub 56943646) — rank ~563→571 & MENURUN (dalam dataran) |
| **Dataran 0,950** | **455 tim** (rank 265–719); urutan dalam dataran = tiebreak submisi, bukan kualitas → **0,950 TIDAK bisa dipertahankan** |
| Garis perak hari ini | rank 279 = posisi ke-299 di dalam dataran; kita di ke-~299 → **DI BAWAH garis** |
| Peta blok (10 Okt 01:25 WIB) | 0,951 = 296 (margin 17 di bawah garis rank 279 = LOTRE) · **0,952 = 183 (margin 96) = TARGET AMAN** · 0,953 ≈130 · emas ≈26 |
| Proyeksi garis akhir 22 Okt | 0,951–0,953 → **target minimum 0,951, kerja untuk 0,952** |
| Deadline | merger/entry **15 Okt 23:59 UTC** (+ `kernelsPublishingDisabled` 15 Okt); final **22 Okt 23:59 UTC**; maks **2 final** (slot kosong diisi otomatis Kaggle — PILIH MANUAL); publik = 30% test set |
| Aturan terima uji | **≥0,951 → kandidat** · 0,950 → buang/tanpa efek · **Δ <0,002 = noise** (konfirmasi arah OOF/gold dulu) |

## 2. Rekap evaluasi (diisi tiap hasil — JANGAN hapus baris lama)
| # | Tanggal | Submisi | Konfigurasi | LB | Δ vs 0,950 | Keputusan | Catatan |
|---|---|---|---|---|---|---|---|
| 1 | 13 Sep | 56196772 | v3 hybrid (fork + arm v3 α=0,15) | 0,941 | −0,009 | BUANG | byte-identik dgn base; arm tak terbaca |
| 2 | 4 Okt | 55795592 | c5 sendirian (CoAtNet5, label fusi) | 0,927 | −0,023 | BUANG sbg stack | solo ≠ stack; CV lokal tak pindah ke LB |
| 3 | 7 Okt | 56897590 | d4-blend (in-arm r2c5) | 0,943 | −0,007 | cadangan bersih | jalur tanpa OAI |
| 4 | 7 Okt | 56872456 | replika goodpjw (reader ConvNeXt) | 0,944 | −0,006 | cadangan bersih TERBAIK | 100% tanpa ckpt OAI |
| 5 | 8 Okt | 56943646 | **Apex v2**: nartaa r384 solo + reader goodpjw | **0,950** | — | **BASELINE / terbaik** | stack cepat (±30 mnt) |
| 6 | 9 Okt 20:34 | 57010586 | template akun kedua (smoke) | — | — | n/a | uji jalur submisi akun2 |
| 7 | 9 Okt 21:39 | **57012143** | **probe r224**: Apex r384 .75 + nartaa r224 .25 + reader | **0,950** | 0,000 | **BUANG** (noise; varian r224/tweak bobot TUTUP) | commit-run OK; skor 23:16 WIB; monitor 30 mnt: `progress_submisi.log` |
| 8 | 9 Okt 23:37 | **57015077** | **fork rabari dual-engine**: raptor r384 + MIRROR_TTA ⊕ reader goodpjw ⊕ r224@0,3 (input publik) | **0,950** | 0,000 | **BUANG** (klaim "0,951+" author TIDAK terbukti; keluarga r384+reader+r224 = plafon 0,950) | skor 10 Okt ~00:45; sisa jatah 2 (reset 07:00 WIB) |
| 9 | 10 Okt 07:27 | **57025414** | **Apex + MaVIT (hengck23 288px) @0,20** rank-blend (`build_submit_mavit.py`; kernel `rsna-knee-apexmavit` v1) | **0,949** | −0,001 | **BUANG** | pembaca ketiga arsitektur independen (solo LB 0,935; gold-58 0,9202) pada bobot 0,20 TIDAK menambah (galat berkorelasi dgn label laporan yg sama); akun UTAMA; skor 08:30 WIB |
| 10 | 10 Okt 09:10 | **57028008** | **Apex 0,70 + OrthoFoundation-L fold0 (heliosli) 0,30** rank-blend (`build_submit_legs.py legs=of wo=0.30`; kernel `rsna-knee-apexof` v1; inferensi OF per 30 studi, ≈27 dtk/studi, ≈5–6 jam total) | PENDING | — | menunggu (±15:30 WIB) | satu-satunya kaki publik dgn pengetahuan eksternal (OrthoFoundation-L + anatomi SKM-TEA/KneeXNet); resep heliosli (Raptor 0,7 + OF 0,3 = 0,951–0,954); terima bila ≥0,951 |

## 3. Temuan riset (sumber lengkap: `reports/Strategi meraih perak RSNA.md`)
### 3a. Sudah TERBUKTI (jangan diulang sia-sia)
- Keunggulan = **pengetahuan/data**, bukan varian: bagging tanpa OAI ≠ menolong; leg lemah MENURUNKAN; DITUTUP: KneeNet, soup init sama, metadata DICOM, resolusi 288, backbone besar (DINOv2-base +0,001 = noise; DINOv3 lebih buruk), label smoothing.
- **Bukti berhenti di 0,949**: TIDAK ADA sumber publik (label/preprocessing/bobot) yang melaporkan gain di atas stack 0,949 → relabeling/resep bukan jalur perak.
- **Optimisasi bobot halus = placebo** (+0,0002 bahkan hill-climb penuh); model sekeluarga & klon tak menambah.
- **PEMISAH 0,950 vs ≥0,951 (riset 10 Okt, thread 735304, 9 Okt) = PSEUDO-LABELING** — bukan arsitektur/TTA/bobot: **colum2131** (LB 0,958; single ConvNeXt-T 0,953 dgn pseudo labels, "no external data used yet"), **azamat1ch** (LB 0,960, rank 20, cuma 12 submisi; single 5-fold ConvNeXt-T 0,959), **hengck** +5: "model is not the key. it is your labels", **Raymond Yuen**: pseudo+blend label 0,949 → tim 0,958.
- Yang berangka & terbukti: **arm baru** (internal: 30% independen → 0,943→0,944; publik: blend-gold), **TTA/seed-avg +0,001–0,003** (writeup juara SIIM), per-label include/exclude (konvergen 2 notebook, TANPA angka → wajib triangulasi dulu).

### 3b. Kandidat arm publik — SUDAH DIEVALUASI 9 Okt
| Kandidat | Hasil evaluasi | Keputusan |
|---|---|---|
| `sameerk2004/sil-d5b-dual0950` | fork `hitarthjain0/apex-v5`; `TARGET_WEIGHTS_V1` identik persis dgn Apex kita; tak terskor 200 besar | **BUKAN arm baru (lift ~0)** |
| `sameerk2004/sil-d5c-triple224` | salinan `kozykappa/gold-gated-triple-reader` = Apex + r224 (sudah kita ukur +0,001 ≈ noise); guard 8,6 jam | **BUANG** |
| `rabari9999/...dual-reader-0950` | 2 ckpt sama + routing di-fit di gold58 (overfit); klaim "0.950" tak terskor | **BUANG** |
| `hitarthjain0/spatial-coatnet-swa-v6` | terhapus (403/404), tak ada fork | tak bisa dinilai |
| **`heliosli/rsna-knee-blend-gold` (#3, 126 vote)** | klaim 0,951 ×2 notebook; **TAPI BLOKIR 9 Okt**: `heliosli/rsna-knee-compact-v1` RETIRED 13:50 UTC ("files not accessible", README) + `heliosli/rsna-knee-ofl-ssl-weights` **403** → manifest/bundle S5 tak tersedia; fold0 tetap publik (best.pt parts + code + anatomy.pt). Fork `evgendvorkin/rsna-versia-7`, `samanyu1808/blend3` ikut mati. | **BLOKIR** — jangan buang waktu; pantau bila heliosli buka lagi |
| `rabari9999/...lb-0-951` (klaim 0,951+) | fork kami (57015077) = **0,950** → **SELESAI & BUANG** (10 Okt 00:45); klaim penulis tak terbukti; MIRROR_TTA+r224@0,3 tak mengangkat | **BUANG**; keluarga r384+reader+r224 = plafon 0,950 |
| `sameerk2004/rsna-knee-sil-d5f-ft7030`, `rainycloudss/...v6` | d5f = fine-tune dgn 5 dataset label LLM (resep placebo per riset); v6 = fork Apex kita (input sama) | pantau; jangan submisi dulu |
| `hengck23/mavit-288-baseline` (MaVIT) | **SUDAH DIEVALUASI 10 Okt 00:56** via commit-run `MODE=local` (GRATIS; `alverlfarfaraway/mavit-288-gold58-eval`): **gold-58 MEAN AUC 0,9202** (ACL .962 / Effusion .980, tapi Lateral OA .826 / Synovitis .823) — di bawah induk nartaa (.9214) & reader (.93) = di bawah ambang leg penambah | **DITOLAK** (tak buang submisi); `lixin73/...mavit-336px` = tanpa kernel publik → tak bisa disubmit |

### 3c. Aksi layak (REVISI 10 Okt 03:00, sumber: `reports/Cara menembus 0951 perak.md`)
1. **Fase A — harvest mirror GRATIS** (commit-run `MODE=local`, ~0,5 GPU-jam): H1 = r224+mirror, H2 = reader+mirror, H3 = mirror sbg arm rank terpisah. Protokol: tarik prob per-studi gold-58 → sweep bobot CPU → **sah bila Δ ≥+0,005 vs incumbent & paired-bootstrap CI95>0** (tuned >5 iter → ambang +0,007); <0,003 = noise.
2. **Fase B — PSEUDO-LABEL SELF-TRAINING** (jalur berbukti ke ≥0,951; 10–25 GPU-jam; muat sblm 15 Okt): pseudo-label 4.407 train pake stack incumbent → fine-tune/rerank → OOF gate → submit bila lulus gate. Bahan: cache `rsna-knee-volcache-z{0,1,2}`, ckpt induk, OOF machinery.
3. Fast-follower scan harian: `python3 scan_public_updates.py` — syarat arm baru: bukti skor ≥0,951 **dan input publik runnable**. Arm menengah (eval gratis dulu): `tbongc` v8/v10geo, `royalacecat` coatnet384.
4. TERTUTUP / jangan ulangi: r224-variant & bobot halus (0,950 placebo), resep 746792 (identik stack kita; all-plane flip merugikan), MaVIT (0,9202), heliosli S6 rekonstruksi (mati permanen), SWA manual/seed-avg (nartaa rilis cuma 2 ckpt), d5e-goldw & d5f (placebo), blending tanpa OAI (0,944).

### 3d. Struktur & final
- **Merger tim (15 Okt 23:59 UTC)** — hanya user; cek dulu `kaggle competitions submission-limits` (batas gabungan ~355–385); SEMUA kandidat final harus ter-push sebelum 15 Okt.
- **2 final = beda MEKANISME** (bukan 2 titik bobot): slot A = Apex+heliosli (agresif/OAI), slot B = hedge (jalur bersih 0,944 / leg OF / varian lolos gate). Pilih **manual** lalu **verifikasi via API** (auto-fill §3.18.c).
- Stack ±7 jam: aman utk skor private (file sama), tapi mepet limit 9 jam → submit 22 Okt dgn margin ≥6–7 jam.

## 4. Rencana peningkatan (urut waktu, tiap aksi punya gate)
### Fase 0 — MALAM INI (9 Okt)
- [x] Pantau skor **57012143** → **0,950 = BUANG** (23:16 WIB, monitor selesai).
- [x] Fork rabari ter-push v1 (23:33) → commit-run RUNNING.
- [x] Fork rabari di-submit (57015077) → **0,950 = BUANG** (10 Okt ~00:45 WIB). Keluarga r384+reader+r224 (dgn/dgn tanpa MIRROR_TTA) = plafon 0,950 — TUTUP.
- [x] **Izin panitia terarsip 10 Okt** di `docs/IZIN_AKUN_KEDUA.md` ("Oh, of course, that's allowed." + kronologi 10 submisi tim).

### Fase A — 10 Okt pagi (GPU pulih 07:00 WIB; ~0,5 GPU-jam; TANPA submisi)
- [ ] Harvest prob gold-58 utk incumbent + r224 + reader ± mirror (1 commit-run MODE=local dgn dump CSV per-studi).
- [ ] Sweep bobot rank/prob CPU (H1–H3); terapkan gate Δ≥0,005 + CI95.
- [ ] Bila lulus gate → 1 submisi kandidat (utama); kalau tidak, Fase B saja.
### Fase B — 10–14 Okt (pseudo-label self-training; 10–25 GPU-jam)
- [ ] Baca `reports/Cara menembus 0951 perak.md` §4 (jadwal, protokol).
- [ ] Pseudo-label 4.407 train (stack incumbent, OOF-aware) → fine-tune → OOF gate → commit-run → submisi bila lulus.
- [ ] Scan harian tetap jalan; arm menengah (tbongc, royalacecat) = eval gratis dulu.
- [ ] Anggaran: ≤2 submisi/hari utk agen, sisakan 2–3 utk user & `nxtirfan`; keputusan dari OOF/gold-58 + gate LB, BUKAN selisih 0,001.

### Fase 2 — 15 Okt (batas keras)
- [ ] `kaggle competitions submission-limits` sebelum merge; semua kandidat final ter-push; tulis kriteria pemilihan 2 final di repo.

### Fase 3 — 16–21 Okt
- [ ] Siklus gate bergilir (2–3 submisi/hari); susun hedge beda mekanisme; **cek harian thread 743416** (belum dijawab host — keputusan agresif-vs-bersih bergantung ini); audit runtime <6–7 jam.
- [ ] Bila garis merayap ke 0,953: perkuat pseudo-label (fold/konsistensi) — BUKAN rekayasa bobot.

### Fase 4 — 22 Okt
- [ ] Submit keduanya 16:00–17:00 WIB (margin ≥6–7 jam); cek ulang 743416 → sikap slot A; **pilih manual + verifikasi via daftar submisi** sebelum 23:59 UTC.

## 5. Daya & batas (perbarui tiap sesi)
| Sumber | Sisa | Reset |
|---|---|---|
| GPU akun utama | 0,00 h | **10 Okt 07:00 WIB** (30 h/minggu) |
| GPU akun kedua | ~29,92 h | 10 Okt 07:00 WIB |
| **GPU akun ketiga `nxtirfan`** | **27,68 h** (pakai BILA akun2 habis; `bash kg3.sh`) | 10 Okt 07:00 WIB |
| TPU akun utama | 14,41 h (XLA kita crash — hindari) | 10 Okt 07:00 WIB |
| Submisi tim hari ini (9 Okt) | **sisa 2** (terpakai: template, probe r224, fork rabari 57015077) | 10 Okt 07:00 WIB, kuota **5/hari** |
| Deadline | entry/merger 15 Okt · final 22 Okt 23:59 UTC | — |
- Cek: `bash kg_quota.sh` (3 akun). Wrapper: `kg2.sh` (akun2), `kg3.sh` (akun3, isi `akun_kaggle/` — di-.gitignore). Monitor submisi: `progress_submisi.log` (PID di `progress_submisi.pid`; ulang: `setsid nohup bash progress_submisi.sh [ID] >> progress_submisi.log 2>&1 < /dev/null &`).
- ML hanya di Kaggle (lokal tanpa torch).

## 6. Risiko & jalur mundur
- **RISIKO #1 — Rules multi-akun**: Rules 3.5.a eksplisit: *"You will be disqualified if you make Submissions through more than one Kaggle account…"* — 9–10 Okt ada **submisi dari `alverlfarfaraway`** (anggota tim) atas **izin panitia yang kini terarsip** (`docs/IZIN_AKUN_KEDUA.md`, 10 Okt); akun ketiga `nxtirfan` (juga anggota tim) kini aktif utk run GPU bila akun2 habis. Mitigasi: (1) ~~arsipkan~~ **izin sudah terarsip** 10 Okt (`docs/IZIN_AKUN_KEDUA.md`, teks + kronologi); (2) jaga kronologi tetap akurat; (3) §3.5.b mengizinkan tiap anggota tim punya akun sendiri — risiko muncul bila SATU orang mengoperasikan kedua; (4) konservatif: hentikan submisi dari akun2 (run GPU di akun2 masih oke — "Submissions" = ke leaderboard), arahkan semua submisi ke akun utama; (5) bila butuh kepastian: klarifikasi resmi panitia dengan lampiran izin.
- **OAI / 743416**: host jawab 741819 (OAI sah bila generally accessible & gratis) tapi **743416 (ckpt bekas-OAI & daftar data) 30 pesan, 0 jawaban host, 81 vote** — jalur bersih kita mentok 0,944 (di luar perak) → keputusan 2 final = user pada 21–22 Okt (1 bersih + 1 agresif). Cek harian.
- **Noise & overfit publik**: Δ<0,002 = noise; tiap probe = overfit adaptif √(k/n); publik hanya 30% test set → steering final dari OOF, bukan papan harian.
- **Timeout**: limit 9 jam, stack ±7 jam → 1 final wajib cepat (Apex ±30 mnt) atau pastikan runtime final <6–7 jam.
- **Jangan**: cetak kredensial / `VERBOSE=true`, `git add -A`, sentuh kernel proyek lain, `pkill -f` pola sendiri, submit varian identik utk "mengukur noise".
