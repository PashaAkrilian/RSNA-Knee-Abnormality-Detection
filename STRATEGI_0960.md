# STRATEGI 0,960 — RSNA Knee (disusun 12 Okt 2026 ±14:15 WIB)

> Dasar: scrape kompetisi 12 Okt (`reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md`), kode repo (`build_s6plus.py`, `build_student_*.py`), bundel S6 (`manifest.json` compact-v1) dan kode latih compact heliosli yang tertanam di `kaggle_base/s6_heliosli_public_20261011.ipynb`.
> Status awal: LB terbaik **0,954** (klon S6 57051680), peringkat 455/5.807. Submisi hari ini: 0/5 terpakai. Kuota GPU: alver 12,0 j (terverifikasi), main ±15,8 j dan nxtirfan ±3,5 j (catatan repo); +30 j/akun pada 17 Okt 07:00 WIB.
> Batas: variasi baru terakhir **20 Okt**; final **23 Okt 06:59 WIB** (pilih 2 manual). Merger tidak dipakai (keputusan user 11 Okt).

## 0. Ringkasan
- **0,960 publik = papan atas ±38 tim hari ini** (garis emas 0,961). Itu berarti +0,006 dari posisi kita. Blend ulang S6 tidak akan sampai ke sana: semua varian S6 yang diuji repo dan publik berhenti di 0,954–0,956.
- **Satu-satunya jalur yang terbukti** di forum: **model murid 2.5D milik sendiri yang dilatih dengan pseudo-label OOF** (ConvNeXt-T / EfficientNetV2-S / ResNet), lalu di-blend dengan basis publik. Contoh yang dilaporkan peserta (belum terverifikasi): ConvNeXt-T 5-fold tunggal 0,959 dan ensembel 0,964 (azamat1ch, tanpa data eksternal); ConvNeXt-T 0,953 + DINOv2-S 0,951 → 0,955 (colum2131); ResNet-50 5-fold 224 px 0,954.
- **Ide kunci baru (terverifikasi dari kode + manifest):** bundel compact S6 dilatih dengan **4 fold hash deterministik** `fold = int(sha1(StudyInstanceUID)[:8], 16) % 4` atas 4.344 studi non-gold, dan gold tidak pernah dilatih. Jadi kita bisa membuat **prediksi OOF yang bersih dari model compact (ConvNeXt-S kuat) untuk semua studi latih**, cukup dengan menjalankan satu model fold per studi. Ini guru pseudo-label terbaik yang tersedia untuk kita (Raptor nartaa in-sample, jadi tidak bisa dipakai sebagai guru OOF).
- **Peluang jujur (perkiraan saya):** ≥0,955 ±80 %; ≥0,957 ±50–60 %; **≥0,960 ±20–30 %**. Bahkan 0,960 publik tidak menjamin perak (simulator repo: P(perak) 27–76 % di 0,960). Target realistis yang sudah sangat memperbaiki peluang perak: **0,957–0,958**.

## 1. Dari mana +0,006 harus datang
| Komponen | Bukti | Perkiraan tambahan (publik) | Peluang berhasil |
|---|---|---|---|
| A. S6 bobot 0,5/0,5 + rank sadar-gagal ("S6-eq") | notebook publik #1 heliosli; tim heliosli 0,956, salinan 0,955 | +0,001 (→ 0,955) | ±65 % |
| B. Murid keluarga 1: ConvNeXt-T 2.5D, label ⊕ OOF-compact | forum: tunggal 0,950–0,959 dengan pseudo-label; S6 + 1 model independen 0,2 → 0,955 | +0,001 s.d. +0,003 dalam blend | ±50 % |
| C. Murid keluarga 2: EfficientNetV2-S atau ResNet-50 224 px | colum2131: dua keluarga +0,002 di atas tunggal terbaik | +0,001 s.d. +0,002 | ±40 % |
| D. Putaran-2 noisy-student (label ⊕ OOF murid ⊕ OOF compact) | repo sendiri: label fusi +0,005 LB pada CoAtNet tunggal (PROGRESS §49) dan putaran-2 r2c5 +0,006 LB walau emas-58 tidak naik (RESEARCH_NOTES §37b); forum: +0,01–0,02 pada tunggal | +0,001 s.d. +0,002 dalam blend | ±40 % |
| E. TTA cermin anatomis pada murid | nartaa +0,001 pada Raptor | +0,0005 | ±50 % |
| **Total** | | **pesimis 0,955 · realistis 0,957–0,958 · optimis 0,960–0,962** | |

Yang **tidak** menyumbang (sudah diukur repo/publik): reader goodpjw, OF fold0, r224, MaVIT, probit, residual Synovitis, tuning bobot per temuan di LB, DINOv3, soup.

## 2. Bentuk notebook final yang dituju
```
[murid A: ConvNeXt-T 5 fold, 224–256 px, cermin TTA]  ─┐   (jalan PERTAMA, ±40–60 mnt, 2 GPU)
[murid B: EffNetV2-S / ResNet-50 224 px]               ─┤
[S6-eq: Raptor (subproses) → compact 4 fold]           ─┤   (±3,7–4,5 j)
                                                        └→ rank blend per temuan, bobot TETAP sebelum submit:
                                                           S6-eq 0,55 · murid A 0,25 · murid B 0,20 (awal; diubah hanya lewat gerbang §5)
                                                           rank sadar-gagal; fallback otomatis ke S6-eq bila murid tidak lengkap
```
Total ±5–6 j dari batas 9 j. Murid dijalankan duluan karena ringan dan tidak bergantung RAM induk. Bila murid melewati batas waktunya, notebook tetap menghasilkan S6-eq.

## 3. Guru OOF compact — cara membuatnya (pekerjaan teknis terpenting)
1. Pseudo-test = studi latih non-gold dengan `sha1(uid)%4 == k` (±1.086 studi per fold). Mekanisme symlink `pseudo` sudah ada di `build_s6plus.py`; ganti pemilihan acak dengan daftar fold-k.
2. Salin **hanya** `E_lrprior_s0_f{k}.pt` ke folder BUNDLE. `compact_infer.py` memuat model lewat `glob(arm + "_s*_f*.pt")`, jadi cukup satu file. Pemeriksaan sha256 per file tetap lolos.
3. `--eval-stride 3` (bawaan validasi latih; ±3× lebih cepat di GPU) dan jalur `noraptor` (tanpa tahap Raptor).
4. Perkiraan biaya: decode+render ±9 dtk + GPU ±1,6 dtk per studi per pekerja, 2 pekerja → ±1,6 j per fold → **±6,5 jam GPU untuk 4 fold** (2 sesi × 2 fold).
5. Jalankan juga compact penuh (4 model) pada 58 studi gold. Ini prediksi bersih karena gold tidak pernah dilatih, sekaligus mengukur kualitas guru.
6. Label baru: `L1 = 0,5 × konsensus-laporan + 0,5 × OOF-compact` per temuan. Konsensus = 4 set publik; Synovitis memakai prior dari Effusion; "tidak disebut" = prior, bukan 0. Varian kedua: tambah Raptor in-sample 0,2. Pilih lewat emas-58 (bootstrap berpasangan) dan korelasi dengan S6.

Catatan kepatuhan: compact memakai jangkar anatomi dari kode/ckpt dataset OF heliosli. Risikonya sama kelasnya dengan S6 yang sudah dipakai, tidak menambah kelas risiko baru.

## 4. Fase dan tugas
| Waktu (WIB) | Tugas | Akun / GPU | Keluaran |
|---|---|---|---|
| **12 Okt** | (a) Baca log `student-train-a` v6 (dtk/iterasi, OOM?) → hentikan bila jelas tidak selesai ≤ 9 j. (b) Bangun **S6-eq** (fork notebook heliosli #1 atau `build_s6plus.py wr=0.5 wc=0.5` + `pct_on`) → commit stub → **submit 1×** dari akun utama. (c) Bangun kernel **cache CPU** 3 shard (224/256 px, ≤28 irisan, urut IPP, npz terkompresi ≤ 20 GB per shard) | utama (stub ±0,3 j); CPU tanpa kuota | skor S6-eq (malam); cache sebagai `kernel_sources` |
| **12–13 Okt** | **Guru OOF compact** fold 0–3 (2 sesi × 2 fold) + gold-58 penuh | utama 2 sesi (±6,5 j) | `oof_compact.csv` (4.344) + AUC emas compact |
| **13 Okt** | Rakit `L1` (lokal, CPU) + evaluasi emas-58; unggah dataset label | — | `labels_student_v1.csv` |
| **13–15 Okt** | **Murid A putaran-1** ConvNeXt-T 5 fold (2 fold paralel per sesi, satu per GPU), 224–256 px, 10–12 epoch, fp16, EMA, cek deadline per epoch + resume | alver (12 j) + sisa utama | 5 ckpt + OOF + emas |
| **14–16 Okt** | **Murid B** EffNetV2-S/ResNet-50 224 px 5 fold (bobot ImageNet offline dari Models/dataset) | sisa kuota / nxtirfan | 5 ckpt + OOF |
| **15–16 Okt** | Notebook final v1 = S6-eq + A (+B); commit stub; **probe LB** 1–2 submisi | utama | skor S6-eq+murid |
| **17 Okt 07:00+** | Kuota baru (+90 j). **Putaran-2**: `L2 = 0,5 laporan + 0,5 rata-rata(OOF A, OOF B, OOF compact)` → latih ulang A dan B (5 fold, atau train-on-all 2 seed untuk final) | semua akun | murid r2 |
| **18–19 Okt** | Notebook final v2 (murid r2 + cermin TTA); 1–2 probe/hari | utama | kandidat final A |
| **20 Okt** | Variasi baru terakhir | — | — |
| **21–22 Okt** | Pilih 2 final: **A** = S6-eq + murid r2 (skor terbaik); **B** = mekanisme berbeda (S6-eq murni, atau murid-berat) | — | final terverifikasi via daftar submisi |

## 5. Gerbang keputusan (angka, bukan perasaan)
| Gerbang | Syarat lulus | Bila gagal |
|---|---|---|
| G0 (12 Okt malam) | S6-eq ≥ 0,955 | tetap jadi basis (lebih robust), target turun ke 0,957 |
| G1 (13 Okt) | compact OOF emas-58 ≥ 0,92 **dan** `L1` emas-58 ≥ konsensus + 0,02 | pakai `L0` (konsensus ⊕ Raptor in-sample 50/50, sudah ada di repo) |
| G2 (15 Okt) | murid A: AUC OOF-vs-konsensus ≥ reader goodpjw + 0,01 **dan** emas-58 ≥ 0,92 **dan** korelasi rank dengan S6 di emas < 0,9 | perbaiki label/resolusi; jangan masuk final |
| G3 (16 Okt) | LB S6-eq + A ≥ 0,956 | turunkan bobot murid ke 0,15 atau tahan; fokus putaran-2 |
| G4 (19 Okt) | LB ≥ 0,958 | final A = skor terbaik yang sudah terskor; jangan memaksa variasi baru |
Aturan tetap: bobot ditentukan sebelum submit; **jangan menyetel bobot per temuan di LB publik**; Δ < 0,002 = noise; maksimal 2 probe/hari dari agen.

## 6. Perbaikan teknis wajib pada pipeline murid repo
1. **Cache sekali di CPU** (3 kernel CPU, keluaran jadi `kernel_sources`). Sekarang setiap run GPU membuang ±72 menit untuk membangun cache.
2. **Pakai kedua T4**: dua proses (satu fold per GPU, `CUDA_VISIBLE_DEVICES`). Kode sekarang hanya memakai `cuda:0`, jadi separuh kuota terbuang.
3. **Deadline per epoch** + simpan state + resume. Sekarang `DEADLINE` hanya dicek antar-fold, sehingga fold kedua bisa terpotong batas 9 jam.
4. **224 px** (bukti forum: 224–288 cukup) atau gradient checkpointing untuk mengatasi OOM; bs 2–4, kwin 6–8.
5. **Guru OOF** (bagian 3), bukan Raptor in-sample.
6. Simpan **OOF + prediksi emas** setiap fold. Validator utama = OOF-vs-label-lemah (r=0,87 dengan LB menurut forum); emas-58 hanya pemeriksa.
7. fp16 (bukan bf16) di T4; log dtk/iterasi; uji inferensi pada ±1.300 studi latih sebelum submit.

## 7. Anggaran GPU (perkiraan, jam kuota)
| Pekerjaan | Jam |
|---|---:|
| Stub S6-eq + notebook final (beberapa) | 2 |
| Guru OOF compact (4 fold + gold) | 7 |
| Murid A putaran-1 (5 fold, 2 GPU paralel) | 5 |
| Murid B putaran-1 | 4 |
| Putaran-2 A + B | 9–10 |
| Cadangan (gagal/ulang) | 8 |
| **Total** | **±35 j** dari ±31 j sekarang + 90 j setelah 17 Okt → cukup, **asal** tiap run tidak gagal berulang |

Kendala sebenarnya adalah **waktu kalender** (2 sesi GPU per akun; run 4–9 jam), bukan total kuota. Karena itu urutan di §4 menempatkan guru OOF compact dan murid A di jalur kritis.

## 8. Plan B (bila murid tidak lulus G2 sampai 16 Okt)
- Pertahankan S6-eq sebagai final A.
- Kaki tambahan murah: 5 ckpt `jessky/rsna-knee-local-cv-cnx50` (compact `d50`, gold dikeluarkan dari latihan). Nilai dulu di emas-58 lewat commit-run gratis; masukkan berbobot 0,2 hanya bila emas ≥ 0,93 dan korelasinya dengan compact S6 rendah.
- Dengan Plan B, target realistis 0,955–0,956 (zona perak publik hari ini, tetapi masih rawan di papan privat).

## 9. Tidak dikerjakan
Blend tambahan tanpa informasi baru; tuning bobot di LB; DINOv3; VLM untuk melabel gambar (mahal dan lambat); pseudo-label test di dalam notebook (tidak muat di 9 jam bersama S6); unduh OAI mentah; merger (keputusan user).

## 10. Risiko utama
| Risiko | Mitigasi |
|---|---|
| Murid lemah (< 0,94 tunggal) → blend ±0 | gerbang G2; Plan B |
| Run gagal berulang (OOM/timeout/mount) | rehearsal kecil (1 fold, 1 epoch, 200 studi) sebelum run penuh; cache CPU; deadline per epoch |
| Notebook final > 9 j | murid duluan dengan batas waktu sendiri; fallback S6-eq; stub + uji 1.300 studi latih |
| Overfit papan publik | bobot tetap; ≤ 2 probe/hari; final B beda mekanisme |
| Kepatuhan akun ganda / OAI | keputusan user; submisi hanya dari akun utama |

## 11. Yang perlu dilakukan USER
1. Setujui **submit S6-eq hari ini** dari akun utama (1 submisi).
2. Setujui perombakan pipeline murid (§6) dan pembuatan guru OOF compact (§3). Agen membangun builder; push kernel hanya atas perintah.
3. Tetapkan akun mana yang boleh dipakai untuk **komputasi** (repo mencatat keputusan 11 Okt: akun 2/3 untuk eksperimen). Saran: submisi hanya dari akun utama.
4. Tetapkan jatah probe: agen ≤ 2/hari (butuh ±10 probe sampai 20 Okt).
5. 21–22 Okt: pilih 2 final bersama agen.

## 12. Checklist agen (urut)
- [ ] Baca log `student-train-a` (alver) → putuskan lanjut/stop.
- [ ] `build_s6eq` (fork heliosli #1 atau `build_s6plus.py` + `pct_on`) → stub → submit (setelah izin).
- [ ] `build_cache_cpu.py` 3 shard → jalankan (CPU).
- [ ] `build_compact_oof.py` (pseudo = fold-k, 1 model, stride 3, noraptor) + versi gold-58 → jalankan 2 sesi.
- [ ] `make_labels_v1.py` (lokal) → evaluasi emas → dataset label.
- [ ] Revisi `build_student_train.py` (kernel_sources cache, 2 GPU, deadline per epoch, 224 px, opsi backbone).
- [ ] Murid A → gerbang G2 → notebook final v1 (`build_s6eq_student.py`) → probe.
- [ ] Murid B; putaran-2 setelah 17 Okt; notebook final v2; final 21–22 Okt.
- [ ] Setiap hasil: catat di `laporan_push.md` §2 dan `MEMORY.md` §3 (sesuai protokol repo).
