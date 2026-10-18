# Riset 18 Okt 2026 malam: model publik baru (sesi Claude Code, hanya baca)

> Sumber: `kaggle competitions leaderboard --download` (papan 18 Okt 14:34 UTC = 21:34 WIB), `kaggle kernels list` (dateRun 5 hal, voteCount/hotness/dateCreated/scoreDescending 3 hal), `kernels pull` 14 notebook, tab Code diurutkan "Public Score" (Chrome), `datasets metadata/files` + unduhan file KECIL saja (log, OOF, label, JSON; tanpa bobot). Evaluasi gold-58 memakai `reference_metadata/train.csv` dan prediksi gold kita (`data/r6gold_s2`, `data/r7oof`, `data/gold_full/gold-g6a`).
> Berkas unduhan dan skrip ada di scratchpad sesi (`.../scratchpad/riset18`, `.../scratchpad/scripts18`) dan tidak di-commit.
> Tanda **[dugaan]** = kesimpulan tanpa bukti langsung.

## 1. Papan publik (18 Okt 21:34 WIB)
| | Nilai |
|---|---|
| Total tim | **6.178** |
| ≥ 0,959 | **168 tim** (rank 1–168): 0,969 (1), 0,968 (3), 0,967 (1), 0,966 (6), 0,965 (4), 0,964 (7), 0,963 (11), 0,962 (21), 0,961 (18), 0,960 (37), 0,959 (59) |
| Tepat 0,958 | **1.089 tim** (rank 169–1.257) |
| Tepat 0,957 | 197 tim (rank 1.258–1.454); total ≥0,957 = 1.454 |
| Batas emas (10 + 0,2 %) | rank 22 = 0,964 |
| **Batas silver (5 %)** | **rank 308 = 0,958** (di tengah blok 0,958) |
| Batas perunggu (10 %) | rank 617 = 0,958 |
| **Krazy KIller** | **rank 448, 0,958** (139 submisi) = urutan ke-280 dari 1.089 di blok 0,958; perlu naik 140 peringkat, atau mencapai 0,959 (rank ≤168 = pasti silver) |

Catatan: blok 0,958 tumbuh dari 488 tim (17 Okt pagi) menjadi 1.089. Di dalam blok, urutan ditentukan skor yang tidak dibulatkan atau waktu submit, jadi menyalin resep publik tidak menaikkan rank.

## 2. Notebook publik baru sejak 16 Okt yang berskor ≥0,958
Tab Code yang diurutkan Public Score menunjukkan **skor publik tertinggi 0,958 (8 notebook)**. **Tidak ada notebook publik ≥0,959.** Notebook 0,957 ada 7.

| Notebook (tgl run, vote) | Skor | Isi / komponen | Beda dari resep kita |
|---|---|---|---|
| `kaiwalyaatulraut/rsna-knee-competition-solution` (17 Okt, 95) | 0,958 | **Salinan logika goodpjw2008 v27**: diff kode yang dinormalisasi hanya berbeda di komentar (lihat §5) | Tanpa S6F, murid, atau R7. Ada Part B (pembaca goodpjw 1/10) dan bobot per-temuan V1 |
| `kaiwalyaatulraut/rsna-knee` (18 Okt, 14) | 0,958 | Identik byte-per-byte dengan notebook di atas | sama |
| `kiyoshiohno/rsna-knee-template` (17 Okt, 48) | 0,958 | Peluncur tipis; logika ada di dataset PRIVAT (slug kosong). Output: `compact_r6_validity`, `compact_r7_predictions`, `raptor224_sub`, `ours_sub` → **R6 + R7 + 384 + 224 + pembaca** (gaya goodpjw v26) | Tidak ada komponen baru; logikanya tidak terlihat |
| `lavinwins/rsna-gold-medal` (17 Okt, 104) | 0,958 | Label "copied with edits from Heliosli". Versi terbaru = pipeline lama 0,943 + "combination layer" (dataset Raptor dreaddevelopment/mattiaangeli) | Lencana skor berasal dari versi salinan heliosli **[dugaan]**; tidak ada model baru |
| `lavinwins/top-1-gold` (17 Okt, 78) | 0,958 | Sama (salinan heliosli). Versi terbaru = templat LATIH (internet ON, 16 irisan sagital 224, 5 fold), tanpa bobot | Tidak ada |
| `heliosli/...-0-9586` (17 Okt, 151) | 0,958 | Sudah diketahui (bobot V2) | — |
| `goodpjw2008/...-lb-0-958` v27 (16 Okt, 356) | 0,958 | Sudah diketahui (basis kita) | — |
| `goodpjw2008/...does-s6-help...` (17 Okt) | 0,957 | **Bukti negatif**: R6flip + S6 compact-v1 0,3 = 0,957 (−0,001). Bobot per-temuan disetel di gold-58: v6 0,957, v7 0,955 | Mendukung aturan "jangan setel bobot di gold" |

Kesimpulan §2: **tidak ada notebook publik ≥0,959. Semua notebook publik 0,958 hanya memakai komponen yang sudah ada di resep kita** (Raptor 384/224 nartaa, R6, R7, pembaca goodpjw). Tim ≥0,959 memakai komponen privat. Contohnya: `okilireaya` (tim *Coffeethefifth*, 0,959 rank 150) yang dataset bobotnya `okilireaya/rsna-knee-e1-3c-weights` kini **403 (privat/dihapus)**, dan `bliverrigpdoge` (0,959 rank 164, KCD25 privat).

## 3. Dataset model publik baru
| Dataset | Isi | Lisensi | Bukti skor | Kesimpulan |
|---|---|---|---|---|
| `neronrx/mycelis-rsna-knee-4fold-hive` ("Mycelis 4-Fold ConvNeXt 384px", 17 Okt) | 4× ConvNeXt-S 384 (210 MB/fold), `knee.py` (KneeNet: 6 slot × jendela 3-irisan → transformer → atensi per temuan, turunan pembaca goodpjw), log + OOF. Init backbone dari `E_lrprior_s0_f0.pt` (R6), label `labels_pseudo_v1`, 14 epoch, k_eval 12 | CC0 | Metadata: **1 vote, ±630 unduhan** (angka "580" kemungkinan jumlah unduhan **[dugaan]**). **Tidak ada notebook publik** yang memakainya. Tim neronrx 0,958 rank 1.192. **Log sendiri: gold 0,67 → 0,58** saat val (pseudo) naik ke 0,92. **Hitungan kita dari OOF mereka: gold-58 rerata 4 fold = 0,596** (Effusion 0,49, MM 0,44), OOF 3.480 studi vs label v3 kita 0,671, ρ vs R7 0,25 (tanpa pola permutasi label). Blend di gold: R7 + 0,1 Mycelis = 0,9273 (< R7 0,9312) | **RUSAK, jangan dipakai.** Tidak ada kode decode/cache DICOM, jadi perlu ditulis ulang. Biaya inferensi ±0,5–1 j/1.284 studi **[perkiraan: 288 forward 384² per studi]** |
| **`kritikabenjwal/rsna-knee-bleg-s0-f01` + `sarthakgupta1801/...-bleg-s0-f23` + `xhaitanya/...-bleg-s1-f01`** ("B-leg", 18 Okt 12:42–14:19, tim *Ghutnoon mein dimaag*) | 6× ConvNeXt-S `E_prior` (seed 0 fold 0–3 lengkap, seed 1 fold 0–1), **warm-start dari R6 fe055406**, fine-tune 2 epoch dengan label `rsna-knee-labels-soft-v2` memakai `compact_train.py` heliosli (fp16). Pool 3.325 studi, gold tidak dilatih, ±2–2,2 j/fold di T4 | "other" (tidak jelas) | Gold per fold (dari `gold_logits` di JSON mereka, sesuai klaim): 0,911 / 0,927 / 0,924 / 0,914 / 0,912 / 0,923. **Rerata 4 fold = 0,9245** (R6flip 0,9377, R7 0,9312). ρ gold: vs R6 0,895, vs R7 0,885, vs g6a 0,916. **g6a + B-leg 0,1/0,2 = 0,9407/0,9412 (+0,0007/+0,0012, P 0,84/0,78)**. Dibanding R7 sebagai tambahan: +0,0003…+0,0011 (P 0,66–0,72) = **setara R7, dalam derau**. Gold sedikit optimistis karena label soft-v2 dikalibrasi di gold. **Tidak ada skor LB**: submisi terakhir tim 17 Okt 09:01, sebelum B-leg dibuat | **Satu-satunya model publik baru yang kredibel**, tetapi potensinya kecil (≤ +0,001, tanpa bukti LB). Jalur inferensinya sama dengan R6 (`compact_infer`). Biaya ≈ tahap R6 tanpa flip, sekitar 75 % tahap R6-flip-s2 bila berdiri sendiri; ±25 % bila dimuat di kontroler R6 yang sama sehingga decode/render dipakai bersama **[perkiraan dari gold R6s2: decode+render 13,8 dtk + GPU 13,0 dtk per studi per pekerja]**. Perlu patch arm `E_prior` |
| `riadmohamed42/rsna-knee-dino-5fold-weights` (16 Okt) | 5× `dino_best_fold*.pt` 91,5 MB (ukuran setara ViT-S/14 **[dugaan DINOv2-small]**), tanpa kode/deskripsi | CC0 | Tidak dipakai notebook publik mana pun. Tim 0,958 rank 569 (kemungkinan dari resep publik) | Tidak bisa dipakai tanpa arsitektur dan prapemrosesan; tidak ada bukti |
| `abhiiiish/rsna-knee-grandmaster-ensemble-3m` (18 Okt) | ConvNeXt-T f0 + ResNet34 f0/f1, tanpa kode | unknown | **Tim abhiiiish LB 0,762** (rank 5.191) | Lemah, abaikan |
| `okilireaya/rsna-knee-e1-3c-weights` | — | — | 403 (privat/dihapus). Pemiliknya 0,959 rank 150 | Tidak tersedia |
| Lain-lain: `arumpuri/weigh-knee-v5-tiny` (3× ConvNeXt-T) · `shantanuedgaonkar/best-knee-model` | OOF arumpuri: gold (36 studi) 0,856, non-gold 0,81–0,83 | Apache / MIT | Tim 0,871 / 0,830 | Lemah, abaikan |

Bukan model: `ayodejiibrahimlateef/rsna-knee-r6-frozen-v1-public-20261018` (salinan R6), `stefan87/...s6-render-shard-*` (cache render S6, ±14 GB/shard), `dkim54/rsna-knee-blend-code` (kode compact_infer heliosli v2/v4c tanpa bobot).

## 4. Set label publik baru
| Dataset | File / kolom | Cakupan | Cara dibuat | 12 temuan lunak? | Gold-58 (kita hitung) | Lisensi |
|---|---|---|---|---|---|---|
| `haungjiawei/rsna-knee-train-v2-code-and-labels` | `weak_labels_llm_v2.csv`: UID + 12 temuan + `label_source` | 4.407 | LLM dari laporan: StepFun 2.708 + Gemini 1.699. Ada juga kode DINOv2-S + SlotHead (tanpa bobot; tim LB 0,891) | **Tidak**: keras 0/1, NaN 41 % (tidak disebut) | 0,798 (sel non-NaN; Synovitis 0,50, Effusion 0,62) | Apache-2.0 |
| `omerzalman/rsna-knee-11-label-evidence-fusion` ("97,13 %") | `soft_labels.csv` (UID + 11), `strict_mri_oof_soft_labels.csv`, calibrated/accuracy/confidence/weights, `study_metadata.csv` (fold, grup laporan duplikat) | 4.407 | Fusi >10 sumber: JEV, Steven, Pilkwang, Lixin GPT-5.6, TypeSafe JEV, Qwen2.5-32B, OpenJev lokal, aturan, **Perplexity Decider via API OpenRouter**, plus **5 pembaca MRI ConvNeXt-S** milik mereka (OOF). Fusi dikalibrasi dengan CV luar di gold | **11 lunak, tanpa Synovitis** | **0,9287** (= klaim; sumber pernah dipilih pada panel gold). Strict-MRI-OOF 0,9158. vs label v3 kita: AUC 0,971, ρ 0,85 | CC BY-NC 4.0 |
| `omerzalman/rsna-knee-98-accuracy-11-soft-labels` | `soft_labels.csv` (UID + 11) + `confidence` / `accepted_mask` / `training_weights` | 4.407 | Varian keluarga yang sama (ambang dipilih dari 35 kandidat di panel gold) | 11 lunak, tanpa Synovitis | 0,9260; vs v3 kita ρ 0,81 | CC BY-NC 4.0 |
| `kritikabenjwal/rsna-knee-labels-soft-v2` | `labels_soft_v2_targets.csv`: UID, `is_gold`, `review_holdout`, `fold`, per temuan `F|strict` (KOSONG) dan `F|prior` | 4.407 (baris gold kosong) | Qwen2.5-7B-Instruct offline di Kaggle → status ordinal keparahan per temuan → peluang per (temuan, status) dikalibrasi di gold-58 dengan shrinkage. Klaim gold 0,831 vs 0,764 untuk label biner | **Ya, 12 temuan (termasuk Synovitis)**, tetapi kasar (≤9 level per temuan) | Tidak bisa dinilai (baris gold kosong; kalibrasinya memakai gold). vs v3 kita AUC 0,825, ρ 0,58 | "other" |

Catatan: (a) Label omerzalman paling dekat dengan "kebenaran" gold karena memuat prediksi model MRI, tetapi **ρ 0,85 dengan label v3 kita**, jadi tambahan informasinya kecil. (b) Mereka mengirim teks laporan ke API eksternal (OpenRouter), yang berisiko melanggar aturan pemakaian data kompetisi **[dugaan, cek Rules]**. (c) Varian lain yang belum diperiksa: `omerzalman/...-99-accuracy...`, `abdullahmathar/rsna-knee-argon4-soft-labels`, `lingyu07/rsna-knee-v53-master-labels`, `siamrahman29/rsna-knee-w5-labels`.

## 5. `kaiwalyaatulraut/rsna-knee-competition-solution`: resep persisnya
- Logikanya **identik dengan goodpjw2008 v27**. Kode kedua notebook dinormalisasi (komentar dibuang) lalu dibandingkan dengan diff: dari 1.830 baris, yang berbeda hanya baris komentar sebaris. Markdown dan gambar penjelasnya dihapus. Datasetnya identik: reader goodpjw, `heliosli/rsna-knee-compact-r7cnxfsss` (**terpasang tetapi TIDAK dipakai**, tidak ada referensi di kode), `rostigor/rsna-knee-r6-fe055406`, `nartaa/...swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`.
- Urutan resep:
  1. Separuh Raptor = 0,65·rank(CoAtNet 384, biasa+cermin) + 0,35·rank(224 crop 0,8, biasa+cermin) (`A224 = 0.35`).
  2. Separuh compact = **R6 + flip TTA** (`FLIP_TTA = True`, `C_SLUG = rsna-knee-r6-fe055406`), tanpa R7.
  3. Blend 0,5/0,5 dengan bobot Raptor per temuan **V1** `{Medial OA .5, Lateral OA .2, PF OA .6, Synovitis .8, Baker's .35}`.
  4. **Part B aktif**: pembaca goodpjw 3 fold (`cnxt_v0_fold[012].pt`), `OWN_MODE="end"`, bobot per temuan 1/10 `{Baker's .035, Contusion .03, Medial OA .025, ACL .025, MM .02, MCL .02, LM/Fracture/LatOA/PFOA .008, Effusion .004, Synovitis .003}`.
- Tim KAIWALYA RAUT 0,958 rank 858 (9 submisi). Ini resep yang sama dengan salinan user 57269378 (0,958).

## 6. Implikasi untuk final 22 Okt
1. **Tidak ada bahan publik ≥0,959 yang bisa disalin.** Semua 0,958 publik adalah keluarga R6+flip (klaster ±1.089 tim). Pembeda kita (S6F, murid, R7) tetap satu-satunya jalan.
2. **Satu-satunya model publik baru yang layak dipertimbangkan: B-leg** (R6 + label soft-v2). Di gold nilainya setara R7 sebagai tambahan compact (+0,001, dalam derau, P ≤0,84). Belum ada bukti LB, lisensinya "other", dan biayanya ≈ tahap R6 tanpa flip. Masuk akal hanya bila (a) menggantikan R7 di g9a karena kita butuh keragaman label (ρ B-leg~R6 0,895 vs R7~R6 ±0,98 di gold), atau (b) dimuat di kontroler R6 yang sama dengan anggaran waktu cukup. Ekspektasi ≤ +0,001 **[dugaan]**.
3. **Abaikan** Mycelis (gold 0,596, rusak), abhiiiish (tim 0,762), arumpuri, shantanuedgaonkar, dan DINO riadmohamed42 (tanpa kode/bukti).
4. Label baru tidak mengubah rencana: melatih murid baru butuh 2–3 hari, dan label omerzalman ≈ label v3 kita (ρ 0,85).

## Catatan status terkini — 19 Okt 2026 02:56 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_SILVER.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; cutoff ini sementara karena medali memakai split privat 70%). Pada cek API 02:55, `g6c`, `g12`, `g13` masih PENDING. Dua kernel murid r4 di alver sudah dua kali di-requeue dan kini RUNNING pada cek 02:56. Kuota main/alver/nxtirfan: 19,87/17,75/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
