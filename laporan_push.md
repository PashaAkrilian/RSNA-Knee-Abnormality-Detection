# laporan_push.md — Evaluasi & jalan menuju SILVER
> Wadah TETAP untuk hasil evaluasi tiap submisi/probe + rencana peningkatan. **Setelah tiap run/submisi/skor baru: tambah baris §2, perbarui §1 & §4.**
> Konteks aturan: `MEMORY.md` · urutan kerja: `HANDOFF.md` · arahan harness: `push.md` · **riset lengkap 9 Okt: `reports/Strategi meraih perak RSNA.md`** (catatan: `research_notes/Strategi meraih perak RSNA/`) · arsip: `PROGRESS.md`, `docs/RESEARCH_NOTES.md`.
> Dibuat 9 Okt 2026 ~21:45 WIB, diperbarui ~22:20 WIB. Bahasa kerja: Indonesia.

## 1. Posisi vs garis perak (snapshot 9 Okt 14:47 UTC, 5.584 tim)
| Hal | Nilai |
|---|---|
| Tim | `Krazy KIller` = `alverlfarfaraway`, `dimaspashaakrilian`, `nxtirfan` |
| LB kita | **0,950** (Apex v2, sub 56943646) — rank ~563→571 & MENURUN (dalam dataran) |
| **Dataran 0,950** | **455 tim** (rank 265–719); urutan dalam dataran = tiebreak submisi, bukan kualitas → **0,950 TIDAK bisa dipertahankan** |
| Garis perak hari ini | rank 279 = posisi ke-299 di dalam dataran; kita di ke-~299 → **DI BAWAH garis** |
| Peta blok (tim ≥skor) | 0,951 = 264 (margin 15) · **0,952 = 171 (margin 108) = TARGET AMAN** · 0,953 = 129 · 0,960 = 26 (emas) |
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
| 8 | 9 Okt 23:33 | (belum) | **fork rabari dual-engine**: raptor r384 + MIRROR_TTA ⊕ reader goodpjw ⊕ r224@0,3 (input publik) | commit-run **RUNNING** | — | gate: **≥0,951 → perak** | dir `kaggle_kernel_rabari_dual_a2/` → `alverlfarfaraway/rsna-knee-dual-engine-tta-fork` v1; sisa jatah 3 |

## 3. Temuan riset (sumber lengkap: `reports/Strategi meraih perak RSNA.md`)
### 3a. Sudah TERBUKTI (jangan diulang sia-sia)
- Keunggulan = **pengetahuan/data**, bukan varian: bagging tanpa OAI ≠ menolong; leg lemah MENURUNKAN; DITUTUP: KneeNet, soup init sama, metadata DICOM, resolusi 288, backbone besar (DINOv2-base +0,001 = noise; DINOv3 lebih buruk), label smoothing.
- **Bukti berhenti di 0,949**: TIDAK ADA sumber publik (label/preprocessing/bobot) yang melaporkan gain di atas stack 0,949 → relabeling/resep bukan jalur perak.
- **Optimisasi bobot halus = placebo** (+0,0002 bahkan hill-climb penuh); model sekeluarga & klon tak menambah; **yang membedakan 0,950 vs ≥0,951 = ARM, bukan formula**.
- Yang berangka & terbukti: **arm baru** (internal: 30% independen → 0,943→0,944; publik: blend-gold), **TTA/seed-avg +0,001–0,003** (writeup juara SIIM), per-label include/exclude (konvergen 2 notebook, TANPA angka → wajib triangulasi dulu).

### 3b. Kandidat arm publik — SUDAH DIEVALUASI 9 Okt
| Kandidat | Hasil evaluasi | Keputusan |
|---|---|---|
| `sameerk2004/sil-d5b-dual0950` | fork `hitarthjain0/apex-v5`; `TARGET_WEIGHTS_V1` identik persis dgn Apex kita; tak terskor 200 besar | **BUKAN arm baru (lift ~0)** |
| `sameerk2004/sil-d5c-triple224` | salinan `kozykappa/gold-gated-triple-reader` = Apex + r224 (sudah kita ukur +0,001 ≈ noise); guard 8,6 jam | **BUANG** |
| `rabari9999/...dual-reader-0950` | 2 ckpt sama + routing di-fit di gold58 (overfit); klaim "0.950" tak terskor | **BUANG** |
| `hitarthjain0/spatial-coatnet-swa-v6` | terhapus (403/404), tak ada fork | tak bisa dinilai |
| **`heliosli/rsna-knee-blend-gold` (#3, 126 vote)** | klaim 0,951 ×2 notebook; **TAPI BLOKIR 9 Okt**: `heliosli/rsna-knee-compact-v1` RETIRED 13:50 UTC ("files not accessible", README) + `heliosli/rsna-knee-ofl-ssl-weights` **403** → manifest/bundle S5 tak tersedia; fold0 tetap publik (best.pt parts + code + anatomy.pt). Fork `evgendvorkin/rsna-versia-7`, `samanyu1808/blend3` ikut mati. | **BLOKIR** — jangan buang waktu; pantau bila heliosli buka lagi |
| **`rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951` (9 Okt 22:55, klaim 0,951+)** | input publik PENUH (nartaa + goodpjw + comp); raptor r384 **+ MIRROR_TTA anatomis** ⊕ reader goodpjw per-finding ⊕ r224@0,3; **commit-run author sukses** (log bersih, dual-T4, spearman kaki 0,50–0,79); fork v1 kami ter-push 23:33 & RUNNING | **→ KANDIDAT AKTIF (slot 8)**; gate ≥0,951 |
| `sameerk2004/rsna-knee-sil-d5f-ft7030`, `rainycloudss/...v6` | d5f = fine-tune dgn 5 dataset label LLM (resep placebo per riset); v6 = fork Apex kita (input sama) | pantau; jangan submisi dulu |
| `lixin73/rsna-knee-mavit-336px-v51-weights` + `hengck23/mavit-288` | arsitektur independen (MaVIT), belum ada angka | opsi arm-2 bila fork rabari gagal |

### 3c. Aksi murah yg MASIH layak (urut bukti, REVISI 9 Okt 23:40)
1. **Fork rabari dual-engine (AKTIF, slot 8)** — commit-run → submit → gate ≥0,951; kalau lolos, ini peraknya. Inti beda vs kita = **MIRROR_TTA anatomis** pada raptor r384 + geometri blend (r224@0,3 coat-side).
2. **Kalau fork ≥0,951**: replikasi TTA-nya ke Apex kita (eval dulu di gold-58: induk+cermin sudah tercatat 0,9214→0,9221) → varian kedua; lalu blend rank-pct 3-leg (r384-TTA ⊕ reader ⊕ r224) utk slot agresif.
3. **Kalau fork =0,950**: jalur berikutnya = MaVIT (`lixin73`/`hengck23`, arsitektur independen) sbg arm baru; cek dulu commit status + input publiknya.
4. Rekonstruksi arm OF heliosli: **BLOKIR** (manifest/ckpt S5 retired) — hanya jika heliosli membuka lagi atau muncul salinan legal; jangan investasi waktu dulu.
5. TTA/seed-avg/variance-reduction lain: nilai di OOF+gold-58 (gratir), submit hanya bila lulus 2/3 (bukti +0,001–0,003).
6. Fast-follower scan harian: `python3 scan_public_updates.py` — syarat arm baru: bukti skor ≥0,951 **dan input publik runnable**.

### 3d. Struktur & final
- **Merger tim (15 Okt 23:59 UTC)** — hanya user; cek dulu `kaggle competitions submission-limits` (batas gabungan ~355–385); SEMUA kandidat final harus ter-push sebelum 15 Okt.
- **2 final = beda MEKANISME** (bukan 2 titik bobot): slot A = Apex+heliosli (agresif/OAI), slot B = hedge (jalur bersih 0,944 / leg OF / varian lolos gate). Pilih **manual** lalu **verifikasi via API** (auto-fill §3.18.c).
- Stack ±7 jam: aman utk skor private (file sama), tapi mepet limit 9 jam → submit 22 Okt dgn margin ≥6–7 jam.

## 4. Rencana peningkatan (urut waktu, tiap aksi punya gate)
### Fase 0 — MALAM INI (9 Okt)
- [x] Pantau skor **57012143** → **0,950 = BUANG** (23:16 WIB, monitor selesai).
- [x] Fork rabari ter-push v1 (23:33) → commit-run RUNNING.
- [ ] **Setelah commit-run COMPLETE → SUBMIT** (sisa jatah 3) → skor ±1–1,5 jam → gate **≥0,951 = PERAK**; restart monitor `progress_submisi.sh`.
- [ ] **User: tempel izin panitia ke `docs/IZIN_AKUN_KEDUA.md`** (Risiko #1, lihat §6).

### Fase 1 — 10–14 Okt (GPU utama pulih 10 Okt 07:00 WIB)
- [ ] Bila fork ≥0,951: bawa MIRROR_TTA ke Apex kita (gold-58: cermin +0,0007) → varian kedua + blend 3-leg; kunci 2 kandidat final beda mekanisme.
- [ ] Bila fork =0,950: uji arm MaVIT (cek input publik + commit status dulu), lalu TTA/variance-reduction (triangulasi dulu).
- [ ] Rekonstruksi heliosli OF: **tunda** (blokir); pantau bila dataset dibuka lagi.
- [ ] Anggaran: ≤2 submisi/hari utk agen, sisakan 2–3 utk user & `nxtirfan`; keputusan dari OOF/gold-58 + gate LB, BUKAN selisih 0,001.

### Fase 2 — 15 Okt (batas keras)
- [ ] `kaggle competitions submission-limits` sebelum merge; semua kandidat final ter-push; tulis kriteria pemilihan 2 final di repo.

### Fase 3 — 16–21 Okt
- [ ] Siklus gate bergilir (2–3 submisi/hari); susun hedge beda mekanisme; **cek harian thread 743416** (belum dijawab host — keputusan agresif-vs-bersih bergantung ini); audit runtime <6–7 jam.
- [ ] Bila garis merayap ke 0,953: naikkan varian-reduksi (TTA +0,001–0,003) + arm kedua (MaVIT), BUKAN rekayasa bobot.

### Fase 4 — 22 Okt
- [ ] Submit keduanya 16:00–17:00 WIB (margin ≥6–7 jam); cek ulang 743416 → sikap slot A; **pilih manual + verifikasi via daftar submisi** sebelum 23:59 UTC.

## 5. Daya & batas (perbarui tiap sesi)
| Sumber | Sisa | Reset |
|---|---|---|
| GPU akun utama | 0,00 h | **10 Okt 07:00 WIB** (30 h/minggu) |
| GPU akun kedua | ~29,96 h | 10 Okt 07:00 WIB |
| TPU akun utama | 14,41 h (XLA kita crash — hindari) | 10 Okt 07:00 WIB |
| Submisi tim hari ini | **sisa 3** (terpakai: template + probe r224) | 10 Okt 07:00 WIB, kuota **5/hari** |
| Deadline | entry/merger 15 Okt · final 22 Okt 23:59 UTC | — |
- Cek: `bash kg_quota.sh`. Monitor submisi: `progress_submisi.log` (PID di `progress_submisi.pid`; ulang: `setsid nohup bash progress_submisi.sh >> progress_submisi.log 2>&1 < /dev/null &`).
- ML hanya di Kaggle (lokal tanpa torch).

## 6. Risiko & jalur mundur
- **RISIKO #1 — Rules multi-akun**: Rules 3.5.a eksplisit: *"You will be disqualified if you make Submissions through more than one Kaggle account…"* — hari ini **2 submisi dari `alverlfarfaraway`** (anggota tim resmi) atas klaim izin tertulis panitia **yang belum diarsipkan**. Mitigasi urut murah: (1) **arsipkan izin ke `docs/IZIN_AKUN_KEDUA.md`**; (2) catat kronologi (akun, waktu, ID submisi); (3) §3.5.b mengizinkan tiap anggota tim punya akun sendiri — risiko muncul bila SATU orang mengoperasikan kedua; (4) konservatif: hentikan submisi dari akun2 (run GPU di akun2 masih oke — "Submissions" = ke leaderboard), arahkan semua submisi ke akun utama; (5) bila butuh kepastian: klarifikasi resmi panitia dengan lampiran izin.
- **OAI / 743416**: host jawab 741819 (OAI sah bila generally accessible & gratis) tapi **743416 (ckpt bekas-OAI & daftar data) 30 pesan, 0 jawaban host, 81 vote** — jalur bersih kita mentok 0,944 (di luar perak) → keputusan 2 final = user pada 21–22 Okt (1 bersih + 1 agresif). Cek harian.
- **Noise & overfit publik**: Δ<0,002 = noise; tiap probe = overfit adaptif √(k/n); publik hanya 30% test set → steering final dari OOF, bukan papan harian.
- **Timeout**: limit 9 jam, stack ±7 jam → 1 final wajib cepat (Apex ±30 mnt) atau pastikan runtime final <6–7 jam.
- **Jangan**: cetak kredensial / `VERBOSE=true`, `git add -A`, sentuh kernel proyek lain, `pkill -f` pola sendiri, submit varian identik utk "mengukur noise".
