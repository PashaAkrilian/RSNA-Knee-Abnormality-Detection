# Kandidat Arm Publik untuk Ensemble (riset 9 Okt 2026)

Sumber metode: `kaggle kernels list/pull` (baca saja, jeda 3 s, tanpa VERBOSE), dump LB resmi
`kaggle competitions leaderboard -d` (snapshot **2026-10-09 14:47:05 UTC**), `scan_public_updates.py`
+ `~/.cache/rsna_scan_log.txt`, dan kode notebook yang ditarik ke `/tmp/opencode/kernels_scan/` (bukan repo).
Tidak ada kernel yang di-push/run/di-submit.

**Konteks LB (dump resmi 9 Okt 14:47 UTC, 5.584 tim):** ≥0,950 = 719 tim · **≥0,951 = 264 tim (batas perak)** ·
≥0,952 = 171 · ≥0,953 = 129 · ≥0,960 = 26. Tim `Krazy KIller` = **rank 571 @0,950** (103 submisi).
→ Naik 1 desimal (0,950→0,951) = menyalip 148 tim.

---

## Q1 · Evaluasi teknis 4 kandidat: arsitektur, preprocessing, blend, independensi

### Takeaway
Tiga dari empat kandidat **bukan arm baru** — semuanya replika dua bobot yang sudah kita pakai
(nartaa r384 + reader ConvNeXt goodpjw) dengan bobot routing per-label yang **persis sama** dengan
Apex kita; kandidat keempat (`hitarthjain0/...spatial-coatnet-swa-v6`) **sudah dihapus/diprivatkan** dan
tidak bisa dinilai. Yang benar-benar independen justru muncul di Q3: leg **OrthoFoundation milik heliosli**.

### Cited Findings

**A. `sameerk2004/rsna-knee-sil-d5b-dual0950` (run 8 Okt 21:14 UTC, 0 vote)**
- Bukan kode orisinal: sel isi notebook = **fork `hitarthjain0/rsna-knee-apex-grandmaster-stack-v5`**
  (judul sel "Apex Grandmaster Stack V5 · Author: @hitarthjain0"); sameerk hanya menambah header & override akhir. — [kaggle.com/code/sameerk2004/rsna-knee-sil-d5b-dual0950](https://www.kaggle.com/code/sameerk2004/rsna-knee-sil-d5b-dual0950) (kode ditarik via CLI)
- Input **hanya 2 dataset**: `goodpjw2008/rsna-knee-2-5d-convnext-reader` + `nartaa/rsna-knee-publication-swa-weights-20261007` (`kernel-metadata.json`). — sumber CLI di atas
- Bobot: `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (arch `coatnet_rmlp_2_rw_384`, 384 px, SHA diserahkan) = **ckpt nartaa 0,949**; plus `cnxt_v0_fold*.pt` (reader ConvNeXt goodpjw). — kode, L133/L303/L1280
- Output final: **override sameerk memaksa Tier-2** (`[sameerk2004] submission.csv <- Tier-2 V3 dual-reader (claimed 0.950)`), Tier-3 "Routing 2.0" hanya aspirasi. Tier-2 = `rank_sota*(1-w) + rank_own*w` per label lalu rank-pct ulang. — kode L1310–1421
- Bobot routing Tier-2 (`TARGET_WEIGHTS_V1`): Baker's 0,35 · Contusion 0,30 · MedOA 0,25 · ACL 0,25 · MedMen 0,20 · MCL 0,20 · LatMen 0,08 · Fracture 0,08 · LatOA 0,08 · PF OA 0,08 · Effusion 0,04 · Synovitis 0,03 — **identik huruf-per-huruf dengan `TARGET_WEIGHTS` di basis kita `kaggle_base/apex_grandmaster_stack_20261008.ipynb`**. — kode d5b L1326 vs repo kita
- Preprocessing (warisan nartaa): volume kanonik **0,4 mm/px, FOV 153,6 mm (SIZE=384)**, span irisan **2–98%**, **96 irisan dgn slot [26,22,18,12,18]**, **K_EVAL=94 jendela 3 irisan**, crop tengah 384→**320 px**, TTA cermin anatomi kiri/kanan; ConvNeXt reader = **6-slot kanonik** (sagittal×2/coronal/axial), 3 fold. — kode d5b/d5c L76–L83, L601–670, L784
- Skor: klaim "claimed public 0.950"; **tidak muncul di 200 besar `--sort-by scoreDescending`** → kemungkinan belum terskor/di bawah ~0,93x. — CLI `kaggle kernels list --competition ... --sort-by scoreDescending`

**B. `sameerk2004/rsna-knee-sil-d5c-triple224` (run 8 Okt 21:39 UTC, 3 vote)**
- Isi = salinan **`kozykappa/rsna-knee-gold-gated-triple-reader`** (judul & teks sel identik: "Gold-Gated Triple Reader V2"). — [kozykappa/.../rsna-knee-gold-gated-triple-reader](https://www.kaggle.com/code/kozykappa/rsna-knee-gold-gated-triple-reader) vs d5c (dua-dua ditarik via CLI)
- Input **hanya 2 dataset** lagi: goodpjw reader + nartaa weights (`kernel-metadata.json`).
- Tiga pembaca: (1) CoAtNet-384 nartaa (0,949), (2) ConvNeXt 3-fold goodpjw (bobot incumbent **0,22**), (3) **opsional** `raptor_..._c08_r224_swa.pt` (224-crop, 94 jendela, **satu view tanpa cermin**, klaim publik 0,945) dengan bobot uji 0,08/0,16/0,20. — kode d5c L123–124, L181–182, L1048–1051, L1253–1256
- Strategi blend: **rank-pct (mode `rank`) atau Gaussianized-rank (mode `normal`)**, plus uji "clinical prior" 0,025; semua **di-gate di gold-58** dgn paired-bootstrap 750× (syarat: gain ≥0,002 · P(boot>0) ≥0,93 · q20 >0) — kalau gagal, kembali ke dual incumbent. — kode d5c L1360–1440
- Menurut notebooknya sendiri: "Gold58 score is **not independent validation** … no verified leaderboard increase yet"; runtime guard `TIME_LIMIT=8,6 jam`. — kode d5c L22, L41
- Skor: **peringkat #18** pada `scoreDescending` (di bawah blok klaim 0,950 di #11–#12) → ≈0,949–0,950, belum >0,950. — CLI

**C. `hitarthjain0/rsna-knee-spatial-coatnet-swa-v6` (scan 9 Okt 03:21 WIB)**
- **SUDAH HAPUS.** `kaggle kernels pull` → **403 Forbidden**, halaman web → **404**, `kaggle kernels list --parent ...` → **404**, dan tidak ada lagi di `kaggle kernels list --user hitarthjain0` (sisa rsna hanya `apex-grandmaster-stack-v5`, `apex-grandmaster-blend-gold`, `58-labels-4407-reports`). — CLI 9 Okt ~21:47 WIB + [kaggle.com/code/hitarthjain0/rsna-knee-spatial-coatnet-swa-v6](https://www.kaggle.com/code/hitarthjain0/rsna-knee-spatial-coatnet-swa-v6)
- Tidak ditemukan fork/copy publik (pencarian `kernels list -s "spatial coatnet"` tidak menghasilkan apa pun). — CLI

**D. `rabari9999/rsna-knee-sota-2-5d-convnext-dual-reader-0950` (run 8 Okt 14:15 UTC, 1 vote)**
- Input **3 dataset**: goodpjw reader + nartaa weights + **`mattiaangeli/rsna-knee-coat-resgated-ep10-top3`** (pakai `gold58_top3_predictions.npz` berisi truth+rank 3 ckpt di gold58). — `kernel-metadata.json`
- Arsitektur: **Stage 1** CoAtNet-384 nartaa (0,949, "native 320px center-crop, 94 slices, anatomical mirror TTA") ⊕ **Stage 2** ConvNeXt 2.5D MIL goodpjw (klaim "5 fold", di kode: semua `cnxt_v0_fold*.pt` yang ditemukan) ⊕ **Stage 3** routing per-pathology + "anti-tie micro-rank calibration" (jitter ±1e-4 — efeknya numerik, bukan sinyal). — kode L157–168, L1277–1309, L1338–1415
- Strategi: Tier-1 `_sota_0949.csv` (nartaa solo) → Tier-2 `_apex_0950_verified.csv` (routing V1, klaim 0,950) → **Tier-3 = bobot hasil grid-search 56 titik per label di atas gold-58 npz** (diadopsi bila `auc_opt >= auc_v3`) + jitter; fallback otomatis ke Tier-2/1. — kode L1315–1415
- Klaim keras di markdown: "Official Public LB: **0.950** (Verified Silver/Gold Medal Tier)", runtime "~7,5 menit" (tidak masuk akal untuk ~1.322 studi penuh → kemungkinan klaim marketing). — kode L1–L20
- Skor aktual: **tidak ada di 200 besar scoreDescending** → notebook ini sendiri kemungkinan belum terskor (kernel `rabari9999/rsna-knee-lb-0-95` lain — triple-reader 384+224+ConvNeXt dgn "probit quantile fusion" — yang terskor, peringkat **#12**). — CLI

### Inferences
- **Independensi vs Apex kita ≈ 0 untuk ketiganya.** d5b = resep Apex 0,950 kita dengan bobot V1 identik; d5c = Apex + r224 (sudah kita ukur sendiri: +0,001 ≈ noise); rabari = dua ckpt sama + routing di-CV di gold58. Korelasi prediksi mendekati 1 → menambahkannya ke ensemble praktis **tidak menaikkan skor**, hanya menambah runtime/jam GPU.
- Judul `0950`/`0.950` pada ketiganya adalah **klaim resep, bukan bukti skor kernel**; satu-satunya bukti skor ordinal = posisi `scoreDescending` (lihat batasan di Gaps).
- Kehilangan `spatial-coatnet-swa-v6` tidak merugikan banyak: dari pola lain milik Hitarth (Apex V5), lineage-nya sama-sama nartaa+goodpjw.

### Gaps
- **Skor LB per-kernel TIDAK bisa dibaca** via CLI/API: `kernel-metadata.json` & `kaggle kernels list --format json` hanya mengembalikan ref/title/author/lastRunTime/totalVotes; endpoint internal `KernelsService/GetKernel` → 403 `kernels.get` (baik anonim maupun dgn token); halaman `kaggle.com/code/...` dirender JS (curl → shell 8,7 KB, termasuk dgn UA Googlebot; `r.jina.ai` juga gagal). Field "Best Score" hanya muncul di cuplikan mesin pencari (contoh terverifikasi: `romantamrazov/rsna-knee-dinosaur-v4` = "Best Score 0.937 V32").
- Urutan `--sort-by scoreDescending` diasumsikan menurut skor LB publik; anchor yang konsisten: #11 `hitarthjain0/apex-v5` (klaim 0.950), #12 `rabari9999/lb-0-95` (klaim 0.950), #22 `nartaa/0949-anatomical-mirror` (0.949), #24 `goodpjw2008/...lb-0-949` (0.949), #168 `romantamrazov/dinosaur-v4` (0.937). Tapi ada anomali tie → jangan dipakai sebagai angka pasti.
- Bobot berapa fold ConvNeXt yang benar-benar dipakai rabari (3 vs 5) tak bisa dipastikan tanpa menjalankan notebook.

---

## Q2 · Layak-tidaknya jadi arm: potensi lift vs risiko korelasi

### Takeaway
Ketiga kandidat yang tersedia **tidak layak ditambahkan** (korelasi ~1, lift ~0); kandidat yang layak
justru **gabungan dengan leg OrthoFoundation heliosli** (0,951), yang menghasilkan arm baru yang setengahnya
independen. Perkiraan kasar keunggulan kita bila arm itu dipasang: **+0,001 → 0,951–0,952** (tepat di/lewat
garis perak), dengan risiko utama = runtime & korelasi kaki nartaa yang masih sama.

### Cited Findings
| Kandidat | Bukti skor | Komponen | Estimasi lift di atas Apex 0,950 | Risiko |
|---|---|---|---|---|
| d5b-dual0950 | klaim 0,950 (tak terskor di 200 besar) | nartaa r384 + goodpjw, bobot **sama persis** dgn Apex kita | **+0,000** | buang kuota GPU/submisi |
| d5c-triple224 | peringkat #18 scoreDescending (≤0,950-an) | Apex + r224 (gold-gate) | **+0,000…+0,001** | winner's curse gold58; runtime ~8,6 jam; r224 sudah kita ukur +0,001≈noise |
| rabari dual-reader-0950 | tak terskor di 200 besar; klaim "0.950 verified" | nartaa + goodpjw + npz gold58 mattiaangeli | **+0,000…+0,001** | routing di-fit di gold58 → overfit; jg anti-tie jitter = kosmetik |
| spatial-coatnet-swa-v6 | — (dihapus) | — | **tidak bisa dinilai** | — |
| **heliosli `rsna-knee-blend-gold`** (baru, Q3) | **0,951** (dua sumber independen) | nartaa r384 + **OrthoFoundation convnext_small 4-fold** (bobot lain) | **+0,001 (0,951–0,952)** | kaki nartaa masih sama → korelasi ~0,6–0,7; runtime 2 pipeline |
- Klaim & reproduksi 0,951: header `sameerk2004/rsna-knee-sil-d5d-compact0951` — "**Public 0.951 recipe (heliosli `rsna-knee-blend-gold`)** … Different second model than our scored 0.950 goodpjw dual"; dan `evgendvorkin/rsna-versia-7` — "Evolution of Public Solutions (0.937 → 0.951) … **Current reproduced pipeline: Version 20 — Heliosli's 0.951 public leaderboard blend**", tabel "Work 6 — 0.950 | Work 7 — 0.951". — [sameerk2004/.../sil-d5d-compact0951](https://www.kaggle.com/code/sameerk2004/rsna-knee-sil-d5d-compact0951), [evgendvorkin/.../rsna-versia-7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7)
- Blending 0,950 ⊕ 0,951 dgn korelasi tinggi secara historis di stack ini memberi +0,000…+0,002 (analogi: reader goodpjw 30% menambah +0,001 pada stack 0,943→0,944; leg keluarga independen kuat hanya +0,001) — [MEMORY.md §2 "Terukur (LB solo)"](../../../MEMORY.md)
- Aturan praktis kita sebelumnya: "Pakai hanya bila LB publik ≥0,951" → arm heliosli-lah yang memenuhi syarat itu; tiga kandidat utama tidak. — [MEMORY.md §2](../../../MEMORY.md)

### Inferences
- **Prioritas 1**: replika `heliosli/rsna-knee-blend-gold` (atau `heliosli/rsna-knee-abnormality-detection` = varian S6) → submit sbg arm; kalau ≥0,951, campur rank-pct dgn Apex kita (bobot awal 50/50 atau 60/40 condong ke 0,951).
- **Prioritas 2 (lebih murah)**: pakai leg OF sahaja bila runtime penuh 2 pipeline terlalu berat — blend-gold sudah membuktikan bobot OF + nartaa, tapi kita tak punya pred OF tersendiri tanpa menjalankan Stage-2-nya.
- **Tidak ditambahkan**: d5b/d5c/rabari (korelasi ~1) kecuali sebagai cadangan "filler" bila slot submisi habis — nilainya nol tapi aman.

### Gaps
- Belum ada angka LB aktual untuk blend-gold dari sumber primer (hanya klaim sameerk + reproduksi evgendvorkin; keduanya notebook publik, bukan halaman leaderboard).
- Runtime blend-gold di 2×T4 tak terukur (perlu lihat log run); kalau >7 jam, arm ini berisiko kena limit 9 jam saat RAM/HDD penuh.
- Apakah bobot OF dilatih memakai data OAI/NDA (perlu kepatuhan seperti ckpt nartaa) → **tidak ditemukan**: kredit notebook hanya menyebut label lemah kompetisi (`stevenleehans`, `pilkwang`, `riadmohamed42`); paragraf OAI di header tampak salinan boilerplate dari notebook nartaa. Perlu dicek lagi sebelum 2 submisi final.

---

## Q3 · Notebook/ckpt publik BARU ≥0,951 & dataset bobot baru (7–9 Okt 2026)

### Takeaway
Sekelompok kecil notebook **≥0,951** berpusat pada **satu resep**: Raptor nartaa 0,949 ⊕ compact
`convnext_small` milik **heliosli** (bobot publik dataset `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`)
dgn bobot tetap 0,70/0,30 — direplikasi minimal oleh 4 penulis berbeda. Dataset bobot baru 7–9 Okt yang
layak dicatat: **bobot MaVIT 336px (lixin73)** + kernel **hengck23/mavit-288**; sisanya label/codec/cache.

### Cited Findings

**Notebook (diurutkan `kaggle kernels list --competition ... --sort-by scoreDescending`, 9 Okt ~21:45 WIB):**
| # | ref | votes | catatan |
|---|---|---|---|
| 1 | `heliosli/rsna-knee-abnormality-detection` | 10 | blend nartaa + OF **bundle S6** `209a32c8b50a6b1c`, 4 fold; run 8 Okt 18:51 |
| 2 | `kiyoshiohno/rsna-knee-template` | 11 | template tipis; logika ada di dataset, input = goodpjw+nartaa+**OF heliosli** |
| 3 | `heliosli/rsna-knee-blend-gold` | 126 | **0,951** (klaim×2); Raptor 0,70 ⊕ OF `E_lrprior` bundle `c7ebab1eb953f1a6` |
| 4 | `evgendvorkin/rsna-versia-7` | 343 | "reproduksi Heliosli 0.951"; Work6 0,950 → Work7 0,951 |
| 5 | `romantamrazov/rsna-knee-dinosaur-v5` | 146 | klaim "V14 ~0,950" (resep nartaa+goodpjw) — kemungkinan 0,950, bukan 0,951 |
| 6–7 | `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed`, `.../rsna-knee-d4` | 97/15 | posisi tinggi, skor pasti tak terbaca |
| 8 | `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` | 129 | = basis Apex 0,950 kita |
| 10 | `kozykappa/rsna-knee-gold-gated-triple-reader` | 19 | asal-usul d5c |
| 11 | `hitarthjain0/rsna-knee-apex-grandmaster-stack-v5` | 34 | klaim "0.950 confirmed" |
| 12 | `rabari9999/rsna-knee-lb-0-95` | 25 | triple-reader 384+224+ConvNeXt, "probit quantile fusion", klaim 0,950 |
| 18 | `sameerk2004/rsna-knee-sil-d5c-triple224` | 3 | (Q1-B) |
| 91 | `hitarthjain0/rsna-knee-apex-grandmaster-blend-gold` | 0 | fork blend-gold, klaim "0.952+", **belum terskor/rendah** |
| 92 | `guruprasaathas111/...dual-paragon` | 1 | clone blend-gold (nartaa+OF), belum terskor |
| — | `sameerk2004/rsna-knee-sil-d5d-compact0951` (9 Okt 11:50) | 2 | klaim **0,951** via resep heliosli; **tak terskor** |
| — | `sameerk2004/rsna-knee-sil-d5e-goldw` (9 Okt 11:56) | 2 | routing emas gold58 (LatOA 0,40/Contusion 0,32/Baker 0,30/ACL 0,20) di atas nartaa+goodpjw |
| — | `rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-95` (9 Okt 11:46) | 3 | varian dual-engine baru |
| — | `hengck23/mavit-288-baseline-01-1-5sec-study-1gpu` (8 Okt) | 26 | keluarga **MaVIT** — arsitektur berbeda dari nartaa/OF |
| — | `omn1v3r53/omniverse-sota-...` (9 Okt) | 1 | tanpa `dataset_sources`; klaim target 0,955 → berbau AI-generated, jangan dipercaya |
Sumber seluruh baris: CLI `kaggle kernels list` (dua kali: `scoreDescending` & `dateRun`) + `kaggle kernels pull` untuk yang diambil.

**Resep inti 0,951 (dari kode `rsna-knee-blend-gold` yang ditarik):**
- Stage 1 = notebook `nartaa/rsna-knee-0949-anatomical-mirror` dijalankan apa adanya (SHA ckpt `bcd864cb…` diverifikasi): 96 irisan (slot 26/22/18/12/18), 94 jendela, 384→320 px, cermin anatomi, fp16.
- Stage 2 = `convnext_small` **4 fold**, arm `E_lrprior`, bundle `c7ebab1eb953f1a6` dari dataset heliosli; preprocessing beda total: **irisan 6–94% tumpukan, crop berbasis milimeter → resize 336 px, 42 jendela/studi**, kode SHA-bound (`native_cache.py`, `native_data.py`, …) + `anatomy.pt` (112.496.037 B). — [heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004)
- Stage 3 = **rank-pct per finding, bobot tetap Raptor 0,70 / OF 0,30**, "nothing fitted on labels or on the test set"; studi yang gagal di satu model diambil rank pasangannya.
- Dataset: 1.343.091.054 B, 4 Okt, 1.445 unduhan (isi: `anatomy.pt`, `cloud_code/**`, bundle tail fold). — CLI `kaggle datasets files/list`

**Dataset bobot/baru lain 7–9 Okt (CLI `scan_public_updates.py` + `kaggle datasets list`):**
- `lixin73/rsna-knee-mavit-336px-v51-weights` (9 Okt 04:23, 71.403.526 B): `mavit_336px_seed48_fold0.pt` + `oof_mavit_fold0.csv` → **bobot + OOF arsitektur MaVIT** (arm keluarga berbeda; pasangkan dgn kernel hengck23).
- `heliosli/rsna-knee-compact-v1` (9 Okt 13:50, hanya README 183 B, 1.642 unduhan) → stub/penera, bukan bobot.
- `arumpuri/rsna-knee-weak-labels` (9 Okt, 340.794 B), `glorysing/rsna-knee-offline-codecs-r09` (9 Okt, 5.327.514 B), `karttikjangid05/rsna-knee-public-llm-label-sets-compared` (8 Okt, 180.569 B) → label/codec, bukan bobot.
- `shikhageekachad/rsna-knee-native384-chunk-000…004` (8–9 Okt; chunk-003 = 14.712.034.073 B) → cache volume 384 px, menarik utk speed tetapi bukan bobot.
- (Di luar jendela, 6 Okt): `tunminhhunh/rsna-knee-dinov2-labels` 82 MB; `codeandodev/rsna-knee-run-bases`; `prvsiyan/rsna-knee-*-student` & `orthodiffusion-official-knee-mri-foundation`.

**Leaderboard publik snapshot** (sumber: dump resmi `rsna-knee-abnormality-detection-publicleaderboard-2026-10-09T14:47:05.csv`): 5.584 tim; distribusi 0,951 = 93 tim, 0,950 = 455 tim, 0,949 = 129; garis perak = rank ≤264.

### Inferences
- **Arm publik terbaik & termurah saat ini = `heliosli/rsna-knee-blend-gold` (0,951)** — satu-satunya klaim ≥0,951 yang didukung dua notebook independen + menempati #1/#3 daftar skor. Ambil sebagai arm, jangan mengejar d5b/d5c/rabari.
- `heliosli/rsna-knee-abnormality-detection` (bundle S6, #1) berpotensi ≥0,951 tapi **belum ada angka** — pantau, jangan dipakai sebelum ada bukti.
- Sumber lift jangka pendek lain yang belum kita coba: **MaVIT** (bobot fold0 publik 9 Okt + kernel hengck23) — arsitektur independen, tapi catatan MEMORY: "resolusi 288 = DITUTUP" dan track record arm lemah kita menurunkan skor → nilai ekspektasi rendah kecuali solo-nya sudah ≥0,94.

### Gaps
- Nilai "Best Score" blend-gold/S6/template/versia-7/dinosaur-v5 dari halaman Kaggle tidak bisa diambil (JS + API 403) — untuk angka pasti perlu dibuka manual di browser (pola yang sama dengan yang dipakai untuk membaca log kernel di MEMORY §4) atau cek cuplikan mesin pencari tiap halaman.
- Tidak ditemukan sumber diskusi/forum (7–9 Okt) yang membahas skor notebook ≥0,951 selain dua header di atas; `kaggle forums` tidak dieksplor lebih lanjut dalam sesi ini.
- `kiyoshiohno/rsna-knee-template` (#2) bergantung pada *dataset* logikanya yang tidak ditarik — perlu `kaggle datasets list --user kiyoc` bila ingin tahu isi pipeline-nya.
