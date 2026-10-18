# Scrape kompetisi Kaggle 12 Okt 2026 — peta menuju perak

> Diambil 12 Okt 2026 ±13:25–13:55 WIB (papan: snapshot 06:28 UTC) lewat Kaggle CLI 2.2.4 (baca-saja, akun `alverlfarfaraway`).
> Cakupan: Overview/Rules/Data (10 halaman resmi), papan peringkat penuh (5.807 tim), 182 topik diskusi (seluruh komentar + isi postingan pembuka), indeks 1.100 notebook publik + kode 169 notebook, 233 dataset + 157 model terkait.
> Data mentah: `C:\Users\LE\01- project\kaggle-scrape-rsna-knee-20261012\` (di luar repo; lihat `INDEX.md` di sana). Laporan ini hanya sintesis.
> **Yang tidak bisa dipastikan:** skor publik tiap notebook (API hanya memberi urutan skor, bukan angkanya), skor privat, dan isi model privat tim lain. Semua angka "perkiraan" di bawah ditandai.

## 0. Ringkasan (baca ini dulu)

1. **Garis perak sudah masuk ke dalam dataran klon.** Papan 12 Okt 06:28 UTC (5.807 tim): perak = peringkat 290 = **0,954**, perunggu = peringkat 581 = **0,954**, emas = peringkat 21 = 0,961. Skor 0,954 dipegang **604 tim** (peringkat 177–780). Tim kita (Krazy KIller) ada di **peringkat 455**, 0,954, 114 submisi. Pada 8 Okt garis perak masih 0,950; pada sore 11 Okt dataran 0,954 baru 244 tim.
2. **Hampir seluruh kode publik papan atas adalah satu keluarga (S6):** Raptor (nartaa) + compact ConvNeXt-S (heliosli) + kode OF heliosli. Dari 25 notebook teratas menurut urutan skor, 25 memakai bobot nartaa dan sekitar 22 juga memakai compact heliosli. Tidak ada informasi baru di dalamnya, hanya variasi bobot dan kaki tambahan.
3. **Temuan baru yang belum ada di dokumen repo:** notebook publik **#1** (`heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`, dijalankan 11 Okt 14:16 UTC) memakai **bobot 0,5/0,5** (bukan 0,7/0,3), rank "sadar-gagal" (rank hanya atas studi yang berhasil diskor), dan bundel `rsna-knee-compact-r6cnxfsss` (berkasnya sama dengan `compact-v1` V13). Skor **tim** heliosli 0,956 dan skor tim penulis salinan `r6cnxfsss 0.5 blend` 0,955 (skor tim, bukan skor per notebook, jadi ini petunjuk, bukan bukti). Dugaan: bobot 0,5/0,5 memberi ≈ +0,001.
4. **Pembeda sejati adalah model independen yang dilatih pada label laporan + pseudo-label OOF**, bukan blend ulang. Bukti dari pengakuan tim lain di diskusi (klaim peserta, belum terverifikasi): ConvNeXt-T 2.5D 5-fold tunggal 0,953–0,959 (colum2131, azamat1ch peringkat 4 dengan 0,964), ResNet-50 5-fold 224 px 0,954 (CoolinLai, 28 Sep), DINOv2-S satu fold 224 px 0,952, Qwen3.5-2B LoRA 0,950. S6 + satu model privat berbobot 0,2 (notebook `velci-kcd25-20`) berasosiasi dengan skor tim 0,955 (BliverRigPåDoge).
5. **Peluang jujur:** simulator repo (`silver_prob_sim.py`) dijalankan ulang di papan baru: P(perak) untuk skor publik **0,954 ≈ 16–25 %**, 0,956 ≈ 19–42 %, 0,958 ≈ 26–60 %, 0,960 ≈ 27–76 % (rentang = tiga set parameter model; model ini mengabaikan struktur blok klon, jadi arah saja). Untuk klon murni S6 saya perkirakan 15–25 % (kasar). **100 % tidak bisa dijamin.**
6. **Risiko non-teknis yang paling besar:** akun ganda (Rules 5.a, kalimat tebal paling atas halaman Rules: *"You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts"*) dan data OAI/eksternal (host belum menjawab soal bobot publik yang dilatih dengan OAI). Keduanya sudah dicatat repo; saya hanya menegaskan kembali.

## 1. Overview, aturan, data (sumber: `competitions pages`)

### 1.1 Fakta resmi
| Hal | Isi |
|---|---|
| Tugas | Prediksi 12 temuan biner per studi MRI lutut: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Metrik | rata-rata ROC-AUC 12 kolom (macro); format `StudyInstanceUID` + 12 probabilitas |
| Jenis | Code competition: notebook ≤ 9 jam (CPU atau GPU), internet mati, `submission.csv`, data eksternal publik dan model pra-latih boleh |
| Jadwal | mulai 30 Jul; **entry dan merger 15 Okt 23:59 UTC**; **final 22 Okt 23:59 UTC**; dokumen pemenang 5 Nov |
| Submisi | maks 5/hari, pilih maks 2 final; tim ≤ 5 orang |
| Skor | set uji publik dan privat terpisah (± 1.300 studi uji, tanpa teks laporan); rasio 30 %/70 % berasal dari catatan repo, halaman Rules hanya menyebut dua set terpisah |
| Hadiah | $77.000: top-10 papan utama ($9k, 7k, 6,5k, 6k, 5,5k, lalu 5 × $5k) + 3 hadiah efisiensi ($7k/6k/5k). **Medali** (perak = 5 % teratas) tidak berhadiah uang |
| Lisensi pemenang | CC-BY-NC 4.0 (dengan catatan bahwa data/model masukan berlisensi tak kompatibel dikecualikan); wajib kode latih+inferensi+bobot sebagai dataset publik |
| Efisiensi | skor = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400 (minimalkan); 2 final dinilai di kedua jalur |
| Data | ±4.407 studi latih (`train.csv` lokal di repo memuat 4.426 baris), 24.371 seri, ±820 rb DICOM, ±570 GB (angka dari forum); **hanya 58 studi berlabel**; sisanya hanya teks laporan (≥ 9 bahasa); 7 studi latih memuat dua lutut |
| Peringatan data | prevalensi temuan **tidak dijamin sama** antara train, publik, privat; DICOM dibersihkan ke 86 tag; sintaks transfer campuran (JPEG Lossless/2000 dll.) |

### 1.2 Cara label uji dibuat (jawaban host, utas 733826/733491/733343) — penting untuk desain label
- Label dibuat **dari citra** oleh dua radiolog MSK + satu pengadil, **bukan dari laporan**; sama untuk set uji. Bila laporan dan citra bertentangan, **citra yang otoritatif**.
- Ambang **tinggi** dan kasus abu-abu = negatif ("favor specificity"): ACL = robekan derajat tinggi/penuh (>50 % serat); MCL derajat tinggi akut; meniskus = sinyal yang jelas menyentuh permukaan pada ≥ 2 gambar; OA = kehilangan tulang rawan >50 % tebal pada area ≥ ±1 cm; **efusi dan Baker's = hanya "sedang atau besar"**; sinovitis, kontusio, fraktur akut didefinisikan terpisah.
- Akibatnya laporan klinis "over-call": efusi "kecil" di laporan sering label 0, tetapi di 58 studi emas efusi dengan kata "kecil/ragu" justru positif 14/18 (colum2131, utas 745949). Kolom paling sulit: **Synovitis** (laporan diam pada 82–84 % studi, padahal 27/58 emas positif), Lateral OA, PF OA, Fracture (di laporan 7 %, di emas 31 %).

### 1.3 Aturan yang relevan untuk medali
- 5.a satu akun per orang (diskualifikasi bila submisi lewat >1 akun); 5.b anggota tim = individu terpisah; 5.c merger sah bila submisi gabungan ≤ jatah sehari×hari berjalan; **6.a dilarang berbagi kode/data kompetisi secara privat antar-tim** (berbagi publik boleh); 3.4.b dilarang labeling-tangan atas data valid/uji (tidak berlaku untuk data latih).
- Peringkat akhir hanya dari papan privat; **imbang → submisi yang masuk lebih dulu menang**.
- Eksternal: boleh bila "reasonably accessible to all" dan biaya minimal; kewajiban pemenang (2.8) tetap berlaku.

### 1.4 Putusan/pernyataan host di diskusi (Po-Hao "Howard" Chen)
| Topik | Putusan | Utas |
|---|---|---|
| LLM berbayar untuk ekstraksi label | **Boleh** ("You can use LLM API, such as OpenAI") | 733965 |
| Dataset eksternal berlisensi non-komersial | Tidak dikecualikan hanya karena NC; klik-setuju/registrasi sederhana umumnya memenuhi aksesibilitas; persetujuan institusional/IRB/perjanjian tertulis bisa gagal | 733965 |
| KneeCoT | **Dilarang** (25 Agu) | 734109 |
| OAI | Tidak melanggar **jika** bisa diakses semua orang tanpa tinjauan institusional/tim hukum dan gratis; host tidak mengomentari syarat pemilik data (21 Sep) | 741819 |
| Bobot publik yang dilatih dengan OAI; MRNet/fastMRI/SKM-TEA; DINOv3; lisensi NC bagi pemenang; bobot yang ditarik penulis | **Belum dijawab** (utas 743416: 37 komentar, 93 vote; 747455, 744056, 748276, 735121, 739905) | — |
| Laporan di set uji | Tidak ada | 734118, 733592 |
| Dua lutut dalam satu studi | Sudah ditinjau per studi; teks/metadata disesuaikan; pertanyaan 7 Okt/11 Okt belum dijawab | 733826 |
| Pemilihan 2 final | Keduanya dinilai di jalur utama dan efisiensi (jawaban peserta, bukan host) | 743356 |

Pernyataan tim lain yang berguna: **azamat1ch (peringkat 4, 0,964) — "haven't used any external datasets so far, only the competition data"** (utas 748295). Beberapa tim kuat lain juga menyatakan tidak memakai data eksternal (YYama, Cà Rốt, diet1236364).

## 2. Papan peringkat (snapshot 12 Okt 06:28 UTC, `leaderboard/`)

| Ambang | Tim | 8 Okt (top-3000 prvsiyan) | Selisih |
|---|---:|---:|---:|
| ≥ 0,964 | 6 | 1 | +5 |
| ≥ 0,962 | 20 | 8 | +12 |
| ≥ 0,960 | 38 | 18 | +20 |
| ≥ 0,958 | 66 | 36 | +30 |
| ≥ 0,956 | 116 | 63 | +53 |
| ≥ 0,955 | 176 | 73 | +103 |
| ≥ 0,954 | 780 | 82 | +698 |
| ≥ 0,951 | 926 | 151 | +775 |
| ≥ 0,950 | 1.294 | 436 | +858 |

- **Garis medali publik:** 8 Okt (≈5.490 tim): emas 0,959 (rank 20), perak 0,950 (rank 274), perunggu 0,948 (rank 549). 12 Okt (5.807 tim): emas **0,961** (rank 21), perak **0,954** (rank 290), perunggu **0,954** (rank 581). Perak dan perunggu kini berada di **dataran yang sama**.
- Isi dataran 0,954: 604 tim; **529** mengirim submisi terakhir dalam 24 jam terakhir, 344 punya ≤ 30 submisi, 145 punya ≤ 10 (gelombang salin-tempel S6). Urutan di dalam dataran tampaknya ditentukan angka desimal keempat/kelima (peringkat 285–295 tidak berurutan menurut waktu submisi).
- Di atas dataran: 0,955 = 60 tim (median 60 submisi), 0,956 = 30 (median 90), 0,957 = 20 (median 82), ≥ 0,958 = 66 (median 116). Median submisi tim ≥ 0,960 ≈ 122: mereka mengiterasi banyak.
- 79 % tim di top-290 adalah solo; 46 tim di top-290 punya ≤ 20 submisi. Tim ≥ 0,960 dengan ≤ 30 submisi: MURAD (0,964; 29), NMarkS (0,961; 30), "吃白饭的大肥鱼" (0,960; **2**), Nischay Dhankhar (0,960; 18) — pipeline offline yang kuat, bukan probing papan.
- Teratas: CentralyDS 0,966; joinT fusion 0,965 (3 orang, 242 submisi); Sergey Bryansky 0,965 (46); MURAD 0,964; Ilya Kolb 0,964; kosuke123 0,964; YOKKOISHO 0,963 (3 orang, termasuk tennogh); "GM-Enhanced Kaggle Agent" 0,963 (5 orang); RB & Tucker & Tony 0,961 (tonylica, tuckerarrants).
- Tim kita: **Krazy KIller, peringkat 455, 0,954, 114 submisi, submisi terakhir 11 Okt 05:03 UTC**; tiga akun terdaftar di halaman tim.
- Trik "kirim 1−p" (utas 748011) dapat menyembunyikan tim kuat; garis perak privat bisa lebih tinggi daripada yang tampak.
- Studi 11 kompetisi lalu (Georgy Mamarin, utas 742327): tim persis di dalam garis perunggu publik berakhir dengan medali hanya ±24 %, sama dengan tim persis di luarnya. **Posisi publik dekat garis hampir tidak berarti.**

## 3. Kode publik (indeks 1.100 notebook; 169 ditarik)

### 3.1 Struktur ekosistem
- Urutan skor: 25 teratas semuanya memakai `nartaa/…swa-weights` (Raptor) + kode OF heliosli (24/25) + compact heliosli (20 dengan `compact-v1`, 3 dengan `r6cnxfsss`) ; reader goodpjw hanya 7/25. Di 120 teratas: nartaa 88 kali, reader 52, compact 28, OF-kode 32.
- Pembeda antar-notebook hanya: bobot blend, cara rank, kaki tambahan. Ini bukti keluarga klon.
- Peringkat skor (urutan, bukan angka) — kolom "skor tim penulis" = skor tim di papan, sebagai petunjuk batas atas:

| # | Notebook | Resep | Skor tim penulis |
|---|---|---|---|
| 1 | `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (177 vote) | S6 dengan **0,5/0,5**, rank sadar-gagal (`pct_on`), Medial/Lateral/PF OA juga 0,5, berhenti bila >50 % studi gagal di tahap kita; bundel `r6cnxfsss` V2 | 0,956 (rank 89) |
| 2 | `bliverrigpdoge/velci-kcd25-20` (50) | S6 0,7/0,3 **tidak berubah** + KCD25 (model privat: tampilan terpusat-lutut, 5 anggota ImageNet, target lunak "d25") berbobot 0,20, proses anak dengan batas waktu | 0,955 (rank 155) |
| 3 | `romantamrazov/rsna-knee-dinosaur-the-last` (110) | S6 + rank sadar-gagal + sisa-rank terbatas untuk Synovitis/Baker's/Contusion/Lateral OA dari reader ConvNeXt publik dan CoAtNet-224 | 0,954 |
| 4–5 | `evgendvorkin/rsna-versia-7` (367), `matterhorn3838/…velciraptor…` (148) | salinan S6 (± rekayasa kecepatan) | 0,954 |
| 6 | `sanketnandkumarpatil/kaggle-rsna-knee` | 0,85 Velciraptor + 0,15 DINOsaur V21 | 0,954 |
| 7, 13 | `goodpjw2008/…lb-0-954…` | S6 + reader 2.5D ConvNeXt berbobot per temuan 3–35 %; emas-58 0,9313 → 0,9331 tetapi LB tetap 0,954 | 0,954 |
| 14 | `wocenmma/rsna-e030-compact35-native` | S6 0,65/0,35 | 0,954 |
| 20 | `sujanmajhisuzan/…v10-tri-engine-probit` | probit + bobot per temuan (Raptor 0,50 / compact 0,30 / reader 0,20) | 0,954 |
| (kopian) | `jenillangaliya/…dual-sota-pipeline`, `goodjane/…r6-cnx-eq-fs-ss`, `sushanthtiruvaipati/…r6cnxfsss-0-5-blend` | salinan 0,5/0,5 | 0,954 / 0,954 / **0,955** |

- Klaim judul tidak bisa dipercaya: `yuvrajlamba01/…tri-hybrid (0.957+)` skor tim 0,954.
- Pelajaran yang berulang di kode: urutan fp16 (bukan bf16 pada T4), penjaga waktu, kegagalan bersuara (`BLEND_FALLBACK`), subproses terpisah per tahap, batas ERROR bila >3 % studi gagal.

### 3.2 Anatomi model yang dipakai semua (dari kode Raptor, nartaa/dreaddevelopment)
- CoAtNet-rmlp-2 pada 384 px; per studi 5 slot seri (sagittal fluid 26, sagittal 22, coronal fluid 18, coronal 12, axial 18 = 96 irisan); irisan merata 6–94 % tumpukan (tepi penting untuk kolateral dan meniskus lateral); crop fisik 140 mm; 3 irisan bertetangga sebagai kanal RGB; 94 jendela; atensi MIL **terpisah per temuan**.
- Latihan: 4.349 studi berlabel-laporan (lunak) + 2.399 lutut OAI (hanya PF OA, Lateral OA, Synovitis yang diberi label; sisanya dimasker); 16 epoch, lr 4e-5/2e-3, wd 0,059, SWA 3 epoch; penambahan OAI tercatat **+0,005** di tabel penulis (tidak terisolasi: jumlah langkah optimizer ikut naik); cermin anatomis +0,001 (flip lebar di semua bidang merugikan).
- Compact heliosli: ConvNeXt-S 4 fold, jangkar sendi lutut, klasifikasi sisi kiri/kanan, target "report-state"; kode latihnya publik di dataset `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` dan salinan `jessky/rsna-knee-local-cv-cnx50` (lihat 4).

## 4. Dataset dan model

| Aset | Catatan |
|---|---|
| `heliosli/rsna-knee-compact-r6cnxfsss` (742 unduhan, 823 MB, 11 Okt 13:18 UTC) | **Baru.** Berkas sama dengan `compact-v1` V13 (bundel `209a32c8b50a6b1c`); sudah disalin pihak lain (`goodjane`, `shareaccc3a`, `lojhhggg/r6data`) |
| `heliosli/rsna-knee-compact-v1` (4.226 unduhan) | Bundel S6; V1–V13 publik |
| `nartaa/…swa-weights-20261007` (2.899), `goodpjw2008/…convnext-reader` (6.040) | Pondasi hampir semua notebook |
| `jessky/rsna-knee-local-cv-cnx50` (12 Okt 01:15 UTC, Apache-2.0) | **5 ckpt compact `d50` + seluruh kode latih "mainline" heliosli** (`compact_train.py` 83 KB, `engine.py`, `distill.py`, `criteria.py`…). Dilatih pada `labels_distill50` (label yang memasukkan guru OOF eksternal; nama menyiratkan campuran 50 %), **gold-58 dikeluarkan** dari latihan (jadi emas-58 bisa dipakai menilainya). Penulis menandainya "provisional" dan mengakui kemungkinan kebocoran orde-dua |
| `gperdrizet/…pseudolabeled-training-data` (12 Okt 04:25 UTC, CC-BY-NC-SA) | Tabel 4.405 studi: label + `*_agreement` + `*_evidence` per temuan + teks laporan |
| `shogomiyawaki/rsna-knee-llm-labels` (12 Okt), `karttikjangid05/…label-sets-compared` (konsensus CC0), `stevenleehans/…llm-report-labels`, `pilkwang/…llm-labels`, `lixin73/…sol56`, `laymond/…qwen3-8b` | Delapan set label publik **tidak berbeda jelas** pada 11/12 temuan (utas 747540); pakai konsensus/steven_v4; imputasi Synovitis ← Effusion |
| `prvsiyan/…public-notebook-scores-and-assets`, `…experiment-ledger` | Snapshot 8–10 Okt: papan top-3000 (anonim), skor 24 notebook, 90 submisi satu tim (terbaik 0,949) |
| Hub Models Kaggle | Hampir kosong untuk tugas ini: `fabotelli/pillar0-knee-finetune` (0,797 pada emas, jauh di bawah CNN 2.5D), `ramitnandi/orthofoundation`, `yaolinghan/rsna-knee-pool-models`; bobot ImageNet offline: `ashishkubade/timm-tf-efficientnetv2-s-in21k`, `shogomiyawaki/rsna-knee-tvw` (torchvision), `sebgarciavillalobos/rsna-knee-imagenet-backbones` |
| Dataset bobot terbaru tanpa notebook publik (`syouyatobita/…g231-nf1`, `easonyanyan/…c12`, `tensorhydra/…meta-folds-v4`, `apoorvadixit/…v4ssl`) | Skor tak diketahui; nilai rendah |

## 5. Diskusi: apa yang terbukti dan apa yang gagal (182 topik)

### 5.1 Label dan pseudo-label — pembeda 0,951+
- **Target lunak (probabilitas), jangan dibulatkan** (Archit Konde, Tucker Arrants, colum2131). "Tidak disebut" ≠ negatif: pakai prior/0,5 atau mask (stevenleehans: 25,4 % sel = "tidak dibahas").
- **Student mengalahkan teacher**: model citra pada label laporan melampaui ekstraktornya, terutama Effusion (0,835 → 0,986 pada emas, colum2131) dan Medial OA; Synovitis tetap terlemah (≈0,86).
- **Pseudo-label OOF dicampur ±50/50 dengan label laporan, bukan diganti** (Raymond Yuen 0,938 → 0,949 → tim 0,958; Archit Konde: makin berat ke model makin baik; Chris Deotte: "dengan pseudo-label setiap model tunggal menyamai CV ensemble terbaik"). **Wajib out-of-fold**; in-fold hanya menggemakan label. Peringatan: Nicolai Karcher melihat kenaikan CV lokal tanpa kenaikan LB; menimpa total label lebih buruk daripada mencampur.
- Synovitis: kolom Effusion lebih prediktif terhadap Synovitis emas (0,7115) daripada kolom Synovitis sendiri (0,678); P(sin|efusi)=0,63 vs 0,22. Imputasi hanya Synovitis ← Effusion, jangan semua kolom. Encoder RadImageNet menaikkan Synovitis 0,62–0,72 → 0,78 (stevenleehans/starkhushi).
- Biaya label bukan masalah: Raymond Yuen < $5 total; tidak ada beda antara LLM mahal dan murah (Sanjib Biswas: ganti LLM = noise ±0,01).
- Hasil tim lain yang diumumkan: ConvNeXt-T 2.5D tunggal 0,953/0,959; EfficientNetV2-S 0,956; CoAtNet 0,952 (azamat1ch, 5-fold); DINOv2 ViT-S 224 px satu fold 0,951–0,952; ResNet-50 5-fold 224 px 0,954; satu fold CoAtNet 224 px 0,950; Qwen3.5-2B LoRA 384 px satu fold 0,950 (±3,5 jam); ResNet kecil satu fold 5 menit inferensi 0,949 (Scott Willis, 900 model dicoba).

### 5.2 Validasi
- Di 15 submisi satu peserta: AUC OOF-vs-label-lemah vs LB **r = 0,87**; AUC emas-58 vs LB **r = 0,66** (utas 748011). Emas-58 hanya 9–35 positif per temuan, SE ≈ 0,01; pilihan checkpoint di emas-58 bisa menyesatkan (epoch 14 naik di emas, turun di LB — nartaa).
- Lebih kuat: bandingkan perubahan berpasangan antar-versi per temuan (OOF lemah, emas, LB) searah atau tidak; bootstrap berpasangan; jangan menyetel bobot pada split publik (utas 742050: ±200 fork duduk di 0,941 karena menyetel bobot).
- Papan menampilkan 3 desimal: perubahan < 0,0005 tak terbedakan (utas 740439).
- Uji secara skala nyata: **jalankan inferensi pada ±1.300 studi latih di commit-run** (Chris Deotte); dua penyebab umum "Notebook Threw Exception": NaN pada prediksi dan DICOM rusak (dua berkas latih terpotong separuh, utas 737163).

### 5.3 Teknik gambar yang terbukti
| Teknik | Bukti |
|---|---|
| Urutkan irisan menurut ImagePositionPatient (bukan nama berkas) | satu baseline: +0,028 AUC satu fold (utas 745759); nama berkas cocok urutan anatomi hanya ±5 % kasus |
| Crop fisik 140–160 mm, bukan piksel | konsisten banyak tim; 100 mm memotong informasi |
| Resolusi 224–288 cukup | Tucker, tennogh, Raymond; 384 tidak otomatis lebih baik |
| 2.5D (jendela 3 irisan) + atensi per temuan | pooling rata-rata inferensi pada bobot terlatih atensi: −0,02 |
| Cakupan irisan 6–94 % dan kerapatan | efek jenuh (±31 jendela cukup pada satu pipeline; lebih banyak tidak membantu) |
| Cermin anatomis TTA | +0,001; flip umum merugikan |
| Laterality hilang pada ±50 % seri | tentukan sisi dari geometri; jika flip, tukar medial/lateral |
| Backbone | tidak menentukan (ConvNeXt, CoAtNet, EffNet, ResNet, DINOv2-S semuanya ±0,95 dengan label baik); **DINOv3 tidak lebih baik** dan lisensinya tidak jelas |
| RadImageNet | +0,019 hanya bila kontrak kanal dihormati (irisan tengah ×3, bukan 3 irisan bertetangga: +0,020 lagi) |
| Gagal/tutup | laporan sebagai loss bantu (merugikan), embedding laporan (kalah dari 12 angka ekstraksi), pemetaan label dari VLM per gambar, model soup dari init sama, flip semua bidang, bf16 di T4 (−0,14 diam-diam karena 50× lebih lambat lalu masuk fallback 0,5) |

### 5.4 Data eksternal (OAI)
Bobot nartaa dilatih dengan 2.399 lutut OAI (+0,005); setiap notebook S6 memuat pengakuan wajib NDA. Akses OAI: login.gov + klik-setuju bagi sebagian orang, **gagal LG22 bagi peserta di Tiongkok**; ±450 GB paket. Tucker Arrants: kemungkinan hanya top-10 yang diaudit. Host tidak menjawab soal bobot publik hasil OAI. Tim peringkat atas yang menyatakan tidak memakai data eksternal: azamat1ch (peringkat 4), YYama. Keputusan ada pada Anda; dokumen repo (CLAUDE/MEMORY) sudah merekomendasikan satu final yang risikonya berbeda.

## 6. Peta pembelajaran → rencana perak (urut nilai-harapan)

Prinsip dari bukti di atas: klon S6 berbagi prediksi nyaris identik, jadi hasil privatnya bergerak sebagai satu blok. Blok berisi ratusan tim yang menempati sebagian peringkat di sekitar garis perak (290) dan perunggu (581). Yang menentukan nasib bukan selisih 0,001 antar-klon, melainkan (a) apakah blok itu sendiri di atas garis dan (b) apakah kita punya **informasi yang tidak dimiliki blok**.

| # | Tindakan | Alasan/bukti | Biaya | Risiko | Nilai-harapan (perkiraan kasar) |
|---|---|---|---|---|---|
| 1 | Pindah ke varian S6 **0,5/0,5 + rank sadar-gagal** (fork `heliosli/blend-mine-in…` atau `build_s6plus.py wr=0.5 wc=0.5` + logika `pct_on`) | tim heliosli 0,956; salinan 0,955; kode jauh lebih jujur terhadap kegagalan | 1 commit-run (stub, ±10 mnt GPU) + 1 submisi; run penilaian ±3,7–4,5 jam, tanpa kuota GPU | tetap anggota blok klon; bobot disetel heliosli di papan publik | +0,001 publik; mengangkat ke ±peringkat 100–176 (belum terverifikasi); privat ≈ netral |
| 2 | Latih **student independen** pada label laporan + pseudo-label OOF 50/50 (ConvNeXt-T/ViT-S 2.5D, 224–256 px, 5 fold) — jalur repo yang sedang berjalan | 6 tim berbeda melaporkan tunggal 0,950–0,959; S6 + satu model independen 0,2 = 0,955; ini satu-satunya tuas di luar blok | GPU: repo sudah mulai (3 gelombang ERROR); kuota alver 12,0 j (terverifikasi 12 Okt 13:26), main ≈ 15,8 j dan nxtirfan 3,5 j (catatan repo 08:23); reset 17 Okt | OOM/waktu (lihat 7), guru in-sample membuat pseudo-label kurang berguna, validasi OOF terhadap label sendiri agak sirkular | +0,001 hingga +0,004 bila berhasil; ini yang menggeser peluang perak ke 30–60 % |
| 3 | **Jalur cadangan murah**: student ResNet-34/50 atau EffNet 224–288 px, jendela sederhana, mean-pooling, 5 fold (≈ 20–30 mnt/fold di T4) | Tucker: "ResNet34/EffNetB0 sederhana 224–288 sudah 0,94+"; CoolinLai ResNet-50 5-fold 224 px 0,954 (28 Sep) | rendah | tunggal mungkin < S6, tetapi independen → manfaat di blend | +0,000–0,002; asuransi bila jalur 2 gagal |
| 4 | Evaluasi `jessky/…cnx50` (5 ckpt compact distill50) pada **emas-58 via commit-run `MODE=local`** (gratis); bila ≥ 0,93 dan berkorelasi rendah dengan S6, jadikan kaki tambahan | gold-58 dikeluarkan dari latihannya; kode latih heliosli ikut publik | ±1 j GPU | ckpt bukan dari kita; "provisional" | tidak pasti; murah untuk dicek |
| 5 | Label: konsensus 4 set publik ⊕ prediksi OOF, **kolom Synovitis ← Effusion**, soft target, "tidak disebut" = prior | utas 733932, 745949, 747540 | rendah | — | sudah ada di pipeline repo |
| 6 | Validator: OOF-vs-label-lemah sebagai utama, emas-58 sebagai pemeriksa; keputusan berpasangan per temuan | utas 748011 | rendah | — | mencegah memilih final yang salah |
| 7 | Final A = varian terbaik (S6-0,5/0,5 + student bila lulus gerbang); final B = mekanisme berbeda (student-berat atau klon S6 yang sudah terskor) | Rules: maks 2; blok klon berkorelasi tinggi | — | — | +3–6 poin peluang (perkiraan repo) |

Jangan dikerjakan (bukti negatif): blend tambahan tanpa informasi baru (reader, OF, r224, MaVIT, probit, residual Synovitis — semuanya ±0 di LB menurut `laporan_push.md` dan notebook publik), tuning bobot per temuan pada split publik, DINOv3, model soup dari init sama, generasi laporan dari citra, bf16 di T4.

### 6.1 Garis waktu (WIB)
| Tanggal | Hal |
|---|---|
| 12 Okt (hari ini) | Submisi tersisa 5; tidak ada submisi 12 Okt. Putuskan #1; pantau `student-train-a` (sedang RUNNING; titik rawan OOM ±30 mnt setelah warm-start) |
| 13–15 Okt | Student fold 0–4 + inferensi + `s6student`; **merger/entry berakhir 16 Okt 06:59 WIB** (keputusan user: tanpa merger) |
| 16–20 Okt | Iterasi bergerbang; kuota GPU baru 17 Okt 07:00 WIB; variasi terakhir ≤ 20 Okt |
| 21–22 Okt | Pilih 2 final **manual**; verifikasi lewat daftar submisi; batas 23 Okt 06:59 WIB |

## 7. Catatan risiko untuk jalur student repo (hasil membaca `build_student_*.py`)
- Waktu: per fold ±1.740 iterasi/epoch (bs=2) × 14 epoch; bila ≳ 0,5 dtk/iterasi, dua fold dalam satu kernel (ditambah cache ±72 mnt) melewati 9 jam; `DEADLINE` hanya dicek sebelum fold dimulai. Log kernel mencetak dtk/iterasi tiap 100 langkah, jadi bisa dipantau.
- Label guru r384 in-sample (guru sudah melihat studi itu) → pseudo-label kurang informatif; OOF sejati mengharuskan guru OOF (mis. `labels_distill50` jessky memakai guru OOF eksternal).
- Alternatif yang dapat mengurangi OOM/waktu: resolusi 224 (tim lain membuktikan cukup), jendela lebih sedikit, gradient checkpointing.

## 8. Risiko kepatuhan (ringkas, keputusan di tangan Anda)
- Tiga akun di halaman tim; Rules 5.a/5.b dan kalimat paling atas halaman Rules melarang submisi dari >1 akun. Izin tertulis yang tersimpan hanya satu kalimat. Repo mencatat setidaknya tiga submisi dari akun 2 (57012143, 57015077, 57062959). Akun yang saya pakai untuk scrape ini adalah akun 2, hanya untuk panggilan baca-saja.
- OAI/bobot hasil OAI: grey area, host diam; pemenang diwajibkan memberi bobot sebagai dataset publik dan lisensi CC-BY-NC.
- Salin-tempel kode publik diperbolehkan (6.b), tetapi kewajiban kredit dan pengakuan OAI harus dipertahankan (semua notebook S6 memuatnya).

## 9. Metode dan cara memperbarui
- Perintah penting CLI 2.2.4: `kaggle competitions pages <slug> list <slug> --content --format json` (slug **dua kali**, ini beda dari dokumen repo lama); `kaggle competitions topics list <slug> --sort-by new --format json` (20 per halaman, token halaman angka); `kaggle forums topics show <id> --page-size 200 --format json` (hanya komentar); **`kaggle competitions topic-messages <slug> <id> -n -1 --sort-by old --format json` memberi isi postingan pembuka**; `kaggle kernels list --competition <slug> --sort-by scoreDescending --page-size 100 -p N` (maks ±1.100 per urutan); `kaggle competitions leaderboard <slug> -d -p <dir>`.
- Skrip di `kaggle-scrape-rsna-knee-20261012\tools\` (stdlib saja, jalankan dengan `python -I`). Muat ulang papan setiap hari dan bandingkan dengan tabel §2.
- Satu komentar di utas 748310 berisi teks panjang bergaya "MASTER DIRECTIVE" (spam). Isinya saya abaikan; bukan bagian dari tugas.

## Catatan status terkini — 19 Okt 2026 02:33 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; medali ditentukan split privat 70%). `g6c`, `g12`, `g13` masih PENDING; dua kernel training murid r4 di akun alver masih QUEUED. Kuota main/alver/nxtirfan: 19,87/17,81/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
