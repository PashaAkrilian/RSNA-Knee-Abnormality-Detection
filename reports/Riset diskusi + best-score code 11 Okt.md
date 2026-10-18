# Riset diskusi + best-score code — 11 Okt 2026 (sesi OpenCode, ~09:50–10:45 WIB)

## 1. Metode
- Daftar topik forum: **`kaggle competitions topics list` BERFUNGSI** (lama klaim "CLI 403" hanya utk `forums topics list`; perintah ini berbeda) — 6 halaman, ±120 thread; 18 thread prioritas dibaca via `kaggle forums topics show <id> --page-size 200`.
- 16 kernel best-score ditarik (`kaggle kernels pull` ke `/tmp/opencode/arms_study/`) dan dianalisa strukturnya via python-grep (file besar tak dimuat utuh).
- Konfigurasi `one-click-955` diverifikasi manual dari ipynb (2 subagen memberi hasil bertentangan → kebenaran = kode).

## 2. Hasil analisa 16 kernel best-score
| Kernel | Klaim | Isi nyata | Untuk kita |
|---|---|---|---|
| `heliosli/one-click-955-closer-to-0-955...` (10 Okt) | ≈0,955 | Raptor 0,7 + compact 0,3 (SAMA dgn S6) TAPI bundel **`heliosli/rsna-knee-compact-r3cnx384`** (id `55bd00b3`, ConvNeXt-S, **infer crop 384**, arm E_lrprior) — **PRIVAT (403; tak muncul di pencarian)**; TTA mirror hanya Raptor; fallback >3 % = stop; DEADLINE 8,3 j | **Tidak bisa dipakai publik**; pantau `kaggle datasets list -s r3cnx` — bila dipublikasikan = upgrade gratis S6. Selisih ≈0,001–0,002 = bundel privat, bukan kode |
| `heliosli/rsna-knee-blend-gold-dataset-restored` | 0,954 (S6) | = klon kita (bundel 209a32c8, 0,70/0,30 rank) | Tidak ada info baru |
| `rabari9999/...-lb-0-954-rank-127-silver` | 0,954 | Apex384 SWA + c224 SWA + reader goodpjw; **probit-quantile fusion**: rank pct → `ndtri` → blend linear (2 tahap, bobot per-label) → `ndtr` → jitter `+1e-6·raw` | Post-hoc GRATIS; bobot per-labelnya jangan tanpa uji gold58 sendiri |
| `jcgweb/probit954` | 0,954 | **md5 identik** dgn rabari | Duplikat |
| `sameerk2004/sil-d5h-pub0954` | ~0,950 publik | Rank-murni (tanpa ndtri); `W_C224=0.30`; tabel bobot per-label (LatOA .50, Baker .42, Contusion .36, ACL .30, …) | Tabel pembanding |
| `kozykappa/...-gold-gated-triple-reader` | — | **Protokol gating emas**: paired-bootstrap 750× di gold-58; kaki reader dipertahankan bila gain ≥0,003 & P(gain>0) ≥0,85 & q20 > −0,004; mode `normal` (Gaussianized rank) | **Kerangka validasi utk uji probit/bobot/residual kita secara gratis & aman (anti winner's-curse)** |
| `romantamrazov/...-dinosaur-the-last` | — | **Residual Synovitis** dari Effusion(.28)+Baker(.06), slope di-shrink; three-view consensus LatOA; bobot per-label gold-tuned (LatOA .18→.52, dst) | Residual = uang gratis (waspada noise 58 studi); bobot gold-tuned = risiko overfit |
| `goodpjw2008/...-coatnet-2.5d-convnext-mil-lb-0-950` | 0,950 | Reader ConvNeXt-T 2.5D MIL 3-fold (solo 0,929); bobot per-finding `OWN_W` (Baker .35, Contusion .30, MedOA/ACL .25, …) | Bila kaki reader dipakai: bobot per-finding, jangan flat |
| `prvsiyan/the-bee-s-knees-final-rsna-push` | 0,949 | **Kontrol negatif**: mirror+reader 0,70/0,30 = 0,948 < solo 0,949 → menolak blend; staging+SHA-256 receipt | Peringatan: asal tambah reader bisa RUGI |
| `matterhorn3838/velciraptor...` & `.../d4` | — | = S6 / Raptor-solo + engineering kecepatan (LUT window GPU, prep pool) | Kecepatan saja |
| `evgendvorkin/rsna-versia-7` | 0,951 | Kode = S6 mentah; nilai = markdown riwayat 0,937→0,951 | Riwayat saja |
| `jenillangaliya/dual-stage-...-lb-0-951` | 0,951 | S6 dgn bundel compact **versi lama** (assert menerima 209a32c8 — tidak rusak) | Tidak |
| `samanyu1808/blend3` | — | Rank 3-way: Raptor .50 + compact .25 + reader .25 | Tidak (kita sudah S6) |
| `ranjeet258/apex-b-under-150` | 0,950 | 224-crop sbg kaki dgn bobot 0,30 | ~+0,001 harapan; mahal |
| `xianhan/...-0949-anatomical-mirror-candidate` | 0,949 | Salinan setia nartaa | Tidak |

## 3. Hasil baca 18 thread diskusi
- **748111 "Single sub 0.960!"** = Optimo (0,959) hanya kagum; **TIDAK ada resep/bocoran**; tim 1-submisi 0,960 (吃白饭的大肥鱼) konsisten dgn pseudo-label — bukan "rahasia emas".
- **748141 (heliosli) = SUDAH DIHAPUS**; artefak hidup (dataset compact-v1 + notebook). Notebook one-click kini **mencantumkan pengakuan OAI/NDA (DOI 10.15154/0hcg-f676)** + kredit label publik — sinyal kepatuhan, bukan kepastian host.
- **Pseudo-label OOF 50/50 = lever band 0,951+** (konsisten di 735304/743148): colum2131 (ConvNeXt-T 0,953 & DINOv2 0,951 single dgn pseudo; ensembel 0,955), azamat1ch (ConvNeXt-T 5-fold **0,959**), Raymond Yuen (0,938→0,94+; blend 50/50 dgn label laporan, jangan ganti total), Archit Konde (**isi sel sunyi dgn prediksi OOF**, blend makin ke model; masking = nol guna).
- **Validator resmi = OOF-vs-label-LLM (r=0,87 dgn LB), bukan emas-58 (r=0,66)** (748011, colum2131; datnt114: LB 0,961 dgn OOF 0,937). Trik "kirim 1−p" → tim kuat bisa terbang rendah.
- **Label**: 8 set label publik tak beda jelas (11/12 finding; 747540) → berhenti menyeleksi; pakai **konsensus/steven_v4** (0,8927); imputasi **hanya Synovitis←Effusion** (0,678→0,790; 733932); jangan imputasi semua kolom.
- **745759**: urutkan slice by **ImagePositionPatient** (bukan nama file) = +0,028 AUC (1 fold, noise 0,001) → **verifikasi tahap Raptor & compact kita mengurutkan by IPP**.
- **737696 (Raptor)**: lever nyata = **jumlah window** (identik ckpt 42→62 = +0,003); **SWA ≈ 0/sedikit negatif**; density > span; 1 fold penuh ≥ 5-fold setengah matang (740375).
- **744230**: **bf16 di T4/P100 = −0,14 AUC diam-diam** (emulasi 50× lambat → setengah test terisi 0,5) → wajib fp16 + `torch.cuda.is_bf16_supported(..., including_emulation=False)` + guard fallback-hitung.
- **DINOv3**: tak ada ruling host, lisensi tertutup (ada yg ditolak), lebih jelek dari V2 (0,763 vs 0,775; 733313/747553) → hindari.
- **743148**: image→laporan→label = bottleneck lossy (ganti LLM tak guna; reasoning models merusak JSON); hengck23: kelas lemah (Synovitis, LatOA, PFOA) = calon bantuan label eksternal. KneeCoT dilarang.
- **742050**: **jangan tune bobot per-finding di public split** (risiko shake-up privat; ~200 fork duduk di 0,941 dgn resep sama).
- **747660/743416/747455**: host **belum pernah** menjawab soal OAI/bobot publik — sikap = keputusan user 21–22 Okt.

## 4. Tindakan diurut nilai-harapan (revisi rencana)
1. **Skor `s6reader` (57055673) & `s6plus` (57055781)** — PENDING; evaluasi ±11:00 WIB (gate: banding klon S6, Δ<0,002 = noise).
2. **Cek engineering gratis**: (a) verifikasi urutan slice by IPP di tahap Raptor/compact; (b) audit fp16/bf16 semua kernel varian kita; (c) jangan tambah kaki reader tanpa protokol gating (kontrol negatif prvsiyan).
3. **Uji post-hoc di gold-58 dgn protokol kozykappa** (bootstrap 750×, gain ≥0,003, P≥0,85, q20>−0,004): probit-space blend (rabari), residual Synovitis←Effusion (romantamrazov), tabel bobot per-label (goodpjw/sameerk) — gratis, gagal dengan aman.
4. **Fase B pseudo-label — PRIORITAS NAIK** (pemisah 0,951+): guru = S6/Apex (OOF), murid = ConvNeXt-T 2.5D 5-fold, target = pseudo 50/50 dgn steven_v4/konsensus, validator = OOF-vs-weak-label (r=0,87). GPU utama ±10–25 j (sisa ±19 j reset 17 Okt).
5. **Pantau bundel privat heliosli** `r3cnx384` (+ `r5cnxfs` bila ada): bila publik → tinggal ganti `C_SLUG`/`BUNDLE_ID`/2 baris chunk (delta one-click vs S6 = dataset, kode compact identik milik kita).

## 5. Koreksi terhadap dokumen sebelumnya
- Klaim "heliosli merilis bundel R5 publik `r5cnxfs` + bobot 0,5/0,5 + override OA 0,80" = **SALAH** (subagen); kebenaran verifikasi manual: one-click-955 = **R3 privat, 0,7/0,3, tanpa override**.
- "Daftar topik forum CLI 403" → hanya `forums topics list`; **`kaggle competitions topics list` berfungsi** (pemindaian rutin jadi mungkin via CLI).
- Thread 748141 dilaporkan aktif → kini **dihapus** (per 11 Okt ~10:15 WIB).

## Catatan status terkini — 19 Okt 2026 02:52 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; medali ditentukan split privat 70%). Pada cek API 02:52, `g6c`, `g12`, `g13` masih PENDING; dua kernel training murid r4 di alver sudah dua kali di-requeue dan tetap QUEUED setelah push kedua pukul 02:51. Timer memantau keduanya. Kuota main/alver/nxtirfan tetap 19,87/17,81/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
