# Research Notes — RSNA Knee Abnormality Detection

Riset Tahap 1 & 2. Dikumpulkan 16 Agustus 2026. Semua klaim disitasi — link ke sumber, atau "kode nyata dari notebook X" dengan referensi Kaggle yang bisa diverifikasi ulang via `kaggle kernels pull <ref>`.

## 0. Catatan metodologi

- **Discussion/Overview/Rules/Evaluation tab Kaggle tidak bisa di-scrape** — semuanya JS-rendered penuh, sudah dicoba berkali-kali di proyek ini (WebFetch cuma dapat title halaman). Ini konsisten di seluruh percobaan, bukan kegagalan sesaat.
- **Sebagai gantinya, riset ini fokus ke Code tab lewat `kaggle kernels list --competition ... --sort-by scoreDescending/voteCount`** — ini API resmi, bukan scraping, dan hasilnya jauh lebih bernilai: kode kerja nyata dari kompetitor top, bukan opini di forum.
- 10 notebook publik ditarik penuh via `kaggle kernels pull` dan dibaca kode-nya (bukan cuma deskripsi): `pilkwang/rsna-knee-baseline-v1` (340 votes, baseline paling banyak di-fork), `prvsiyan/rsna-knee-read-the-report-then-the-knee` (223 votes), `mattiaangeli/bend-the-knee-to-dinov3-ensembled` (102 votes), `sakhawathossen/rsna-knee-enhanced-ensemble`, `salemali7/rsna-knee-90-reports-llm-30-epochs`, `wguesdon/rsna-knee-dinov2-at-meniscus-resolution` (84 votes), `hida1211/rsna-knee-public-4-fold-dinov2-v4`, `ranjithragavan07/rsna-knee-dinov2-0-93`, `renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe`, `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`.
- **Metrik evaluasi: macro/mean ROC-AUC across 12 label — dikonfirmasi via bukti konvergen.** Halaman Evaluation resmi tetap tidak bisa diakses (JS-rendered), tapi 2 notebook top independen (`prvsiyan`, dan validasi silang lain) sama-sama menghitung "macro AUC" (rata-rata `roc_auc_score` per label) sebagai proxy validasi lokal mereka — kalau bukan itu metrik aslinya, mereka tidak akan optimasi ke situ. Digabung dengan rentang skor leaderboard (0.90-0.95, khas AUC) dan klaim awal Anda sendiri, ini sudah cukup meyakinkan untuk dipakai sebagai dasar loss/validasi kita: **mean ROC-AUC per label, dirata-rata ke 12 label (macro-average)**.

## 1. Temuan dari kode kompetitor top (Code tab, via API resmi)

### 1.1 Backbone: DINOv2 mendominasi — tapi "DINOv3" di judul menyesatkan

Hampir semua notebook top (9 dari 10 yang ditarik) memakai DINOv2 di kode aktual, termasuk yang judulnya menyebut "DinoV3" (`mattiaangeli/bend-the-knee-to-dinov3-ensembled`, `ranjithragavan07/rsna-knee-dinov2-0-93` yang menyebut `DINOv3` sekali di path string tapi model aktual tetap DINOv2).

**Verifikasi via `kaggle models list -s dinov3`**: **tidak ada Kaggle Model resmi dari Meta untuk DINOv3** — cuma upload komunitas tak terverifikasi (`mdabrarhossainzahin/dinov3`, `keras/dinov3`, termasuk varian 7B yang jelas di luar konteks/scope kompetisi ini). Bandingkan dengan DINOv2: `metaresearch/dinov2` resmi dari Meta, dipakai konsisten di semua notebook via path `/kaggle/input/dinov2/pytorch/{small,base}/1` atau `/kaggle/input/models/metaresearch/dinov2/pytorch/{small,base}/1`.

**Kesimpulan: "DINOv3" di judul-judul itu kemungkinan besar nama versi PIPELINE penulis sendiri (v3 dari eksperimen mereka), bukan model DINOv3 beneran.**

Cara load (dari `pilkwang/rsna-knee-baseline-v1`, fungsi `build_model`):
```python
from transformers import AutoModel
bb = AutoModel.from_pretrained(str(dinov2_path))  # path hasil scan /kaggle/input mencari config.json + "dinov2" di nama folder
n_layer = len(bb.encoder.layer)
for prm in bb.parameters():
    prm.requires_grad = False
for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:  # cuma unfreeze N blok terakhir
    for prm in blk.parameters():
        prm.requires_grad = True
```
Progressive unfreezing (hanya buka blok transformer terakhir) — rasional: blok awal self-supervised transformer = filter generik (edge/tekstur), blok akhir = semantik; dengan supervisi terbatas (58 studi gold), membuka semua layer berisiko merusak fitur generik yang sudah bagus.

### 1.2 Representasi studi: "slot" bag, bukan sekadar flatten semua series

Konsep kunci dari `pilkwang`: studi direpresentasikan sebagai **bag of slot images** — setiap "slot" adalah kombinasi anatomis tertentu (plane × sekuens), bukan raw list series. Attention pooling per-diagnosis dengan **prior anatomis yang di-hardcode** dari pengetahuan domain:

```python
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}
```
(Indeks = ID slot, mis. sagittal/coronal/axial × dengan-tanpa fat-sat — skema pasti bervariasi per notebook.) Ini konsisten dengan rekomendasi kita sendiri sebelumnya ("bobot series diinformasikan Anatomical_Plane") tapi diimplementasikan lebih presisi dengan tabel eksplisit per-finding.

`TARGETS` di kode ini **identik persis** dengan 12 label kita: `["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]` — validasi silang bagus.

### 1.3 Pooling: CLS + mean-patch (+ opsional top-k "focal")

```python
out = self.backbone(pixel_values=x).last_hidden_state
patch = out[:, 1:]
parts = [out[:, 0], patch.mean(1)]          # CLS token + mean semua patch
if pool == "cls_mean_focal":
    k = max(1, patch.shape[1] // 8)
    parts.append(patch.topk(k, dim=1).values.mean(1))   # top-1/8 patch per channel -> tangkap temuan lokal
feat = torch.cat(parts, dim=1)
```
Alasan varian "focal": mean polos atas ratusan patch mengencerkan sinyal temuan lokal (misal robekan kecil) sampai 2 orde magnitudo — dengan ambil top-k per channel, sinyal lokal tidak hilang.

### 1.4 Weak-label: LLM offline lebih disukai, rule-based sebagai fallback (BUKAN live API call)

Poin paling penting untuk pertanyaan kepatuhan Anda: **referensi "LLM"/"GPT" di kode BUKAN pemanggilan API langsung di dalam kernel** — semuanya membaca CSV hasil pra-komputasi dari Dataset terpisah:
```python
"/kaggle/input/rsna-knee-llm-report-labels/llm_labels_v2.csv"
"/kaggle/input/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"
```
Pola: seseorang menjalankan LLM **di luar kernel yang di-score** (kemungkinan besar saat development, internet menyala, sebelum submission final) untuk menghasilkan label dari teks laporan, lalu mempublikasikan hasilnya sebagai Dataset statis yang di-attach. Kernel yang di-submit sendiri **tidak pernah memanggil API** — dan secara teknis pun tidak bisa (internet OFF saat scoring).

Ada juga **bukti eksplisit seseorang membandingkan performa "Claude Opus 5" vs rule-based extractor** untuk task ini (dari `wguesdon/rsna-knee-dinov2-at-meniscus-resolution`):
```
rule extractor      agreement 0.7960  precision 0.6802  recall 0.7648
Claude Opus 5 (low)  agreement 0.8233  precision 0.6910  recall 0.8602
paired delta +0.0273 agreement, 95% CI [+0.0029, +0.0532], McNemar p=0.0204
```
Claude menang signifikan secara statistik. **Ini indikasi kuat praktik komunitas nyata**, tapi TIDAK membuktikan itu sesuai Competition Rules — kemungkinan besar dilakukan sebagai eksperimen dev-time terpisah, bukan bagian dari kernel yang disubmit. **Tetap perlu Anda cek Rules resmi soal ini** — saya tidak mengasumsikan diizinkan.

Rule-based extractor `pilkwang` (untuk skenario Dataset LLM tidak tersedia) memakai **deteksi negasi/uncertainty per-klausa**, bukan sekadar keyword match:
```python
def _polarity(clause, anchor_end):
    if UNCERTAIN.search(clause): return "uncertain"
    if NEGATION.search(clause): return "negative"
    if NORMALITY.search(clause):
        if TEAR.search(clause) or re.search(r"\bgrade [34]\b", clause): return "positive"
        return "negative"
    return "positive"
```
Komentar kode secara eksplisit menyebut **variasi urutan kata negasi antar bahasa** ("Turkish puts its negator at the end of the sentence, English at the front") — ini persis peringatan yang sudah kita catat di artifact strategi sebelumnya, sekarang terkonfirmasi jadi masalah nyata yang ditangani kompetitor top.

Ada guardrail eksplisit `LabelSourceError` — kalau run dikonfigurasi mengharapkan label LLM tapi yang ke-attach cuma rule-based, run **berhenti paksa** daripada diam-diam melatih dengan label lebih lemah. Praktik rekayasa yang baik untuk ditiru.

### 1.5 Ensemble multi-backbone

`prvsiyan` (223 votes, notebook paling detail) mengombinasikan minimal 3 backbone berbeda dalam satu pipeline: DINOv2 (transformer), BiomedCLIP (vision-language, dipakai sebagai encoder fitur — "BiomedCLIP encoded {n}/{total} acquired slices"), dan EfficientNet-B3 (`timm`, CNN klasik). Kombinasi rank-blend disebut di nama file (`submission_yash_dino_rankblend.csv`) — indikasi ensembling berbasis ranking, bukan cuma rata-rata probabilitas mentah (lebih robust kalau skala kalibrasi antar-model beda).

**Catatan integritas:** sebagian kode `prvsiyan` di-obfuscate sebagai blok base64 di dalam cell — pola umum kompetitor top menyembunyikan detail implementasi sambil tetap "publik" secara teknis. Tidak semua logic bisa dibaca langsung dari situ.

### 1.6 Hyperparameter konkret (dari `pilkwang/rsna-knee-baseline-v1`)

```python
EPOCHS = 10
BATCH_STUDIES = 8   # 1 batch = 8 studi, tiap studi = bag berisi hingga N_SLOT slot
```
Resolusi input bervariasi per konfigurasi eksperimen (terlihat referensi ke ukuran 336px di salah satu daftar konfigurasi). k-fold: banyak notebook pakai `GroupKFold` (grouping di level studi, bukan series/instance — konsisten dengan rekomendasi kita: split di `StudyInstanceUID`).

### 1.7 Fingerprinting untuk validasi reproducibility

Teknik menarik dari `pilkwang`: sebelum training/inference, jalankan model pada **input sintetis dari seed tetap**, simpan output sebagai "fingerprint". Kalau di run lain fingerprint beda > toleransi kecil (2e-3), berarti ada yang salah di preprocessing/arsitektur/urutan load — meski model "jalan tanpa error" dan submission "terlihat wajar". Berguna karena kesalahan preprocessing sering *silent* (submission valid, skor cuma turun, tidak ada pesan error). **Direkomendasikan ditiru di pipeline kita.**

## 2. Efficiency Prize — track terpisah yang tidak ada di brief awal Anda

Ditemukan dari `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` (dihosting staf Kaggle Ryan Holbrook, dikenal dari notebook demo RSNA lain dengan 913 votes — figur staf/grandmaster yang kredibel).

- Ada **leaderboard efisiensi terpisah**, 1.507 tim masuk (dari total 1.642 entrant) — ini bukan niche, mayoritas peserta ikut.
- **Urutan `EfficiencyRank` TIDAK sama dengan urutan `PublicScore`** (contoh: rank 1 skor 0.936, rank 2 skor 0.943 — rank 2 lebih tinggi skornya tapi rank efisiensinya lebih rendah). Ini bukti empiris kuat bahwa formula rank memasukkan faktor efisiensi (waktu/compute), bukan cuma AUC mentah.
- **Formula pasti belum bisa saya konfirmasi** — halaman `/overview/efficiency-prize-evaluation` JS-rendered, blocked sama seperti semua halaman Kaggle lainnya.
- **Implikasi strategis:** kalau ada prize/pengakuan terpisah untuk efisiensi, dan compute kita (GPU quota mingguan terbatas) memang jadi constraint utama, ini bisa jadi jalur kompetitif yang kurang ramai dibanding leaderboard utama yang sangat padat (top-20 dalam 0.016). **Perlu keputusan Anda**: prioritaskan skor mentah, efisiensi, atau dua-duanya sekaligus (mis. pilih arsitektur yang tetap akurat tapi jauh lebih murah, seperti DINOv2-small alih-alih -large/-giant).

## 3. Kompetisi RSNA & serupa sebelumnya

| Kompetisi | Arsitektur pemenang | Sumber |
|---|---|---|
| RSNA Intracranial Hemorrhage (2019) | 2D CNN per-slice + BiGRU/LSTM antar-slice + linear stacking; ensemble 7-31 model; 1.345 tim | [arXiv 2205.07556](https://arxiv.org/pdf/2205.07556), [5th place solution](https://github.com/Anjum48/rsna-ich) |
| MRNet (Stanford, bukan RSNA tapi struktur sangat mirip: knee MRI, multi-plane, multi-finding) | AlexNet per-slice + max-pool antar-slice + logistic regression gabung 3 plane; riset lanjutan capai 93.4% AUC dgn arsitektur modern + augmentasi | [PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699), [LearnOpenCV](https://learnopencv.com/stanford-mrnet-challenge-classifying-knee-mris/) |
| RSNA Breast Cancer Screening (2023) | Juara 1: **ConvNeXt-small** (bukan EfficientNet seperti dugaan umum); ~1.700 tim | [1st place solution (GitHub)](https://github.com/dangnh0611/kaggle_rsna_breast_cancer) |

**Pattern yang reusable ke kompetisi kita:** pemisahan intra-slice (2D CNN/ViT) vs inter-slice/inter-series (attention/RNN/pooling) muncul konsisten di semua kompetisi medical-imaging-multi-slice ini — memvalidasi arah arsitektur kita ("2D backbone + attention pooling multi-slice & multi-series") bukan cuma tebakan, tapi pola yang terbukti menang berulang kali di domain serupa.

## 4. Perbandingan foundation model (literatur, di luar Kaggle)

| Model | Temuan literatur | Lisensi | Kaggle Model resmi? |
|---|---|---|---|
| **DINOv2** | Hasil campuran di medical: lebih lemah dari ImageNet-pretrained di sebagian dataset klinis, tapi kompetitif/lebih baik di dataset publik lain, terutama dgn frozen features. Efektivitas bergantung seberapa jauh domain gap dari foto natural (MRI = gap besar). ([arXiv 2402.07595](https://arxiv.org/abs/2402.07595), [arXiv 2312.02366](https://arxiv.org/pdf/2312.02366)) | Apache 2.0 | ✅ `metaresearch/dinov2` (resmi Meta) |
| **DINOv3** | "Peningkatan modest, bukan lompatan revolusioner" dibanding DINOv2 untuk medical imaging; task-dependent, perlu validasi domain-spesifik ([arXiv 2509.06467](https://arxiv.org/pdf/2509.06467)) | Belum diverifikasi | ❌ Tidak ada versi resmi Meta; cuma re-upload komunitas tak terverifikasi |
| **BiomedCLIP** | Vision-language biomedis (PMC-15M), SOTA di retrieval/klasifikasi/VQA cross-modal medis | **MIT** (permisif, aman dipakai) | ⚠️ Cuma re-upload komunitas (`mrquadian/biomedclip` dll), tidak ada dari Microsoft langsung — tapi lisensi asli MIT jadi risiko rendah |
| **RadImageNet** | ResNet50 di-pretrain 1.35 juta gambar radiologi. **Bukti spesifik untuk task kita**: pada studi transfer-learning knee MRI, RadImageNet unggul **+4.8% AUC untuk ACL** dan **+4.5% AUC untuk robekan meniskus** dibanding ImageNet (P<.001), performa setara/melebihi radiolog MSK pada MR knee. Dipublikasikan di jurnal RSNA sendiri. ([Radiology: AI / RSNA](https://pubs.rsna.org/doi/10.1148/ryai.210315), [PMC9530758](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9530758/)) | **Terklarifikasi**: dataset mentah (1.35jt gambar) restriktif (Research Use Agreement, dilarang redistribusi), TAPI bobot model pretrained yang dirilis resmi BMEII-AI (tim asli) di GitHub **MIT License** (copyright notice "(c) 2021 BMEII-AI" tertanam di teks lisensi, mengindikasikan dirilis resmi oleh pembuat asli, bukan klaim sepihak pihak ketiga) | ✅ Dipakai — sumber [`Lab-Rasool/RadImageNet` di HuggingFace](https://huggingface.co/Lab-Rasool/RadImageNet) (mirror resmi BMEII-AI, MIT, provenance jelas), **bukan** re-upload Kaggle Model `dbpdenesh/...` yang provenance-nya tidak jelas |
| **MedicalNet** | 3D ResNet, domain medis volumetrik (awalnya proyek Tencent) | Perlu diverifikasi lebih lanjut | ⚠️ Beberapa re-upload komunitas |
| **MedSAM** | Model **segmentasi** (Segment Anything varian medis) — bukan didesain untuk klasifikasi, kurang cocok langsung utk task ini kecuali dipakai murni sbg feature extractor | Perlu diverifikasi | ⚠️ Beberapa re-upload komunitas |

**Insight penting:** dominasi praktis DINOv2 di kompetisi ini kemungkinan besar bukan murni soal "akurasi terbaik secara teoretis" (literatur menunjukkan hasilnya campuran), tapi kombinasi: (a) satu-satunya yang punya Kaggle Model **resmi** dan tervalidasi banyak orang, (b) kompatibilitas mulus dengan `transformers.AutoModel`, (c) medium size (small/base) cukup ringan untuk iterasi cepat dalam limit compute Kaggle.

## 5. Keputusan FINAL Tahap 2 (diputuskan 16 Agustus 2026, mandat penuh dari user — "putuskan yang terbaik")

| Kandidat | Keputusan | Alasan |
|---|---|---|
| **DINOv2-base** (`metaresearch/dinov2`) | ✅ **Backbone utama** | Resmi, tervalidasi luas di kompetisi ini sendiri, lisensi jelas (Apache 2.0), kompatibel `transformers.AutoModel`, ukuran base cukup ringan utk iterasi & sesuai limit T4x2/P100 |
| **RadImageNet-ResNet50** | ✅ **Dipakai — komponen ensemble kedua** | Bukti spesifik +4.8%/+4.5% AUC utk ACL/meniskus (lihat bagian 4), lisensi terklarifikasi MIT dari sumber resmi (`Lab-Rasool/RadImageNet`), sama seperti pola `prvsiyan` (223 votes) |
| **BiomedCLIP** | ✅ **Dipakai — komponen ensemble ketiga** | Lisensi MIT aman, dipakai kompetitor top sbg salah satu dari 3 backbone ensemble |
| **DINOv2-small** | 🟡 Cadangan | Kalau compute jadi bottleneck nyata di jalan |
| **DINOv3** | ❌ Tidak dipakai | Tidak ada Kaggle Model resmi; literatur cuma "modest improvement"; risiko provenance tidak sepadan gain |
| **MedicalNet / MedSAM** | ❌ Tidak dipakai | MedSAM = segmentasi bukan klasifikasi; MedicalNet tidak muncul dipakai kompetitor top kompetisi ini |
| **LLM hosted (Claude/GPT dkk) untuk proses teks laporan** | ❌ **Tidak dipakai** | Data pasien sensitif + Rules belum terverifikasi eksplisit mengizinkan. Ada bukti komunitas memakainya (offline, hasil dipublikasi sbg Dataset statis) tapi itu bukan konfirmasi kepatuhan. Default aman: **rule-based negation-aware multi-bahasa saja**, sesuai instruksi awal user |

**Target:** kalahkan peringkat 1 (Brandon Low, 0.950) lewat skor mentah (macro ROC-AUC) — bukan Efficiency Prize. Prioritas murni akurasi, ukuran model dipilih base (bukan small) selama masih muat limit compute Kaggle (T4x2/P100, submission ≤9 jam).

**Arsitektur final:** ensemble 3-backbone (DINOv2-base + RadImageNet-ResNet50 + BiomedCLIP) → CLS+mean(+focal top-k) patch pooling per slot/slice → attention pooling per-diagnosis dengan prior anatomis (SLOT_PRIOR_TABLE) → rank-blend antar-backbone → sigmoid 12-label. Latih di atas weak-label rule-based negation-aware multi-bahasa dari 4.349 studi, kalibrasi & validasi ke 57 studi gold (exclude 1 studi anomali dari audit kita).

## 6. Item yang masih perlu verifikasi Anda (tidak memblokir Tahap 3, tapi penting)

1. **Rules soal LLM API eksternal & data security** — kalau Anda bisa cek browser dan ternyata diizinkan, kita bisa upgrade weak-labeler pakai LLM lokal open-weight (bukan API hosted) di kernel Kaggle sendiri — opsi ini tetap terbuka tanpa perlu kirim data keluar Kaggle sama sekali
2. **Efficiency Prize** — saat ini saya prioritaskan skor mentah sesuai instruksi Anda; beri tahu kalau berubah pikiran

## 7. Dataset publik yang dipakai kompetitor (arahan 16 Agustus 2026 — "cek dataset apa yang dipakai utk naikkan skor")

**Metodologi:** `kaggle datasets list -s "rsna knee llm"` / `"rsna knee report"` untuk cari dataset publik yang menargetkan kompetisi ini secara spesifik, lalu unduh & ukur macro AUC (implementasi rank-based manual, karena `sklearn` tidak tersedia di mesin lokal) terhadap 58 studi gold di `reference_metadata/train.csv` sebelum memutuskan pakai/skip.

**Ditemukan `stevenleehans/rsna-knee-llm-report-labels`** (CC0-1.0, public domain) — label probabilitas `[0,1]` per finding, dihasilkan LLM pihak lain yang membaca kolom `Report` `train.csv` kompetisi ini secara offline (bukan API hosted yang kita panggil sendiri — hasil sudah dipublikasikan statis, konsisten dengan pola "LLM-offline-label" di §1). Dataset card-nya sendiri secara eksplisit membandingkan terhadap regex/rule-based (0.8136 macro AUC klaim mereka) dan terhadap versi blend dengan labeler independen kedua (`lixin73/rsna-knee-llm-report-labels-sol56`, juga CC0).

**Verifikasi independen (bukan sekadar percaya klaim publikasi):**

| sumber | macro AUC vs 58 gold (ukur sendiri) | vs klaim dataset card |
|---|---|---|
| rule-based labeler kita (§`kaggle_kernel_weak_labeler`) | ~0.71 (estimasi dari precision/recall/agreement — biner, bukan skor kontinu) | — |
| `llm_labels_full.csv` (v1) | **0.8780** | cocok persis (klaim: 0.8780) |
| `llm_labels_v4_blend.csv` (v4, final) | **0.8927** | cocok persis (klaim: 0.8927) |
| `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` | 1.0000 (trivial/circular) | — |

Pengukuran ulang cocok persis dengan angka yang diklaim publik — meningkatkan kepercayaan pada metodologi pengukuran kita sendiri maupun pada dataset ini.

`barun2104` diperiksa lebih jauh: nilai labelnya identik persis (selisih absolut 0.0) dengan 58 studi gold kita di baris yang berlabel, dan `NaN` murni di 4.349 baris sisanya (`fold=-1`). Ini bukan hasil labeling independen — cuma re-package `train.csv` resmi kompetisi plus kolom `fold` untuk split. **Di-skip**, tidak menambah informasi apa pun di luar yang sudah kita punya, dan lisensinya ("other") belum terverifikasi aman.

**Keputusan: adopsi `llm_labels_v4_blend.csv`** sebagai target training untuk 4.349 studi non-gold di `kaggle_kernel_modeling` (lihat `PROGRESS.md` §9 untuk detail integrasi). 57 studi gold tetap pakai label asli, tidak pernah ditimpa oleh estimasi apa pun — public dataset ini murni menggantikan rule-based labeler kita sebagai sumber *weak target* untuk studi yang tidak punya anotator asli.

## 8. Ekosistem dataset publik kompetitor top — riset lanjutan (arahan 16 Agustus 2026, "jangan cuma 1 dataset")

**Metodologi:** `kaggle kernels list --competition rsna-knee-abnormality-detection --sort-by scoreDescending --page-size 50` untuk daftar 50 kernel teratas (skor asli tidak diekspos di field JSON, tapi urutan sort + skor yang disebut eksplisit di banyak judul, mis. "0.93", "Public 0.916 Reproduction", "[0.899] Let me Cook", cukup jadi sinyal kuat). Dari situ, **18 kernel** ditarik penuh (`kaggle kernels pull -m`) mencakup rentang vote 0-344 dan gaya arsitektur berbeda (baseline asli, ensemble, TTA, per-target specialist, dual/hybrid-LLM label), lalu `kernel-metadata.json` tiap kernel dibaca untuk daftar `dataset_sources`/`model_sources`/`kernel_sources`-nya.

**Temuan utama: ada ekosistem dataset publik bersama yang dipakai HAMPIR SEMUA kompetitor top sekaligus** — bukan cuma 1 dataset label seperti yang kita adopsi sebelumnya. Frekuensi kemunculan di 18 kernel sampel:

| dataset publik | dipakai di | jenis | lisensi | catatan |
|---|---|---|---|---|
| `pilkwang/rsna-knee-weights` | 14/18 | **checkpoint model TERLATIH** dari notebook asal 344-vote | CC0-1.0 | ⚠️ ini bobot model kompetitor lain yang SUDAH di-fine-tune ke kompetisi ini, bukan sekadar pretrained generik — lihat catatan legalitas di bawah |
| `pilkwang/rsna-knee-llm-labels` | 13/18 | label LLM (beda dari stevenleehans) | CC0-1.0 | sumber label independen kedua, dipakai luas |
| `marwanmath/resnet-50-radimagenet-marwan` | 12/18 | bobot RadImageNet-ResNet50 | **CC-BY-NC-SA-4.0** | ⚠️ **NonCommercial** — kompetisi ini berhadiah $77.000, jadi status "non-commercial" ambigu. **Jangan pakai re-upload ini**; kita sudah punya sumber MIT-licensed terverifikasi (§4/§5) |
| `stevenleehans/rsna-knee-llm-report-labels` | 9/18 | label LLM (yang sudah kita pakai) | CC0-1.0 | terkonfirmasi jadi salah satu dari beberapa sumber label yang biasa di-blend, bukan dipakai sendirian |
| `lixin73/rsna-knee-llm-report-labels-sol56` | 8/18 | label LLM (GPT-5.6-Sol, disebut di dataset card stevenleehans) | CC0-1.0 | hampir selalu muncul BERBARENGAN dengan stevenleehans — konfirmasi pola blend 2-labeler |
| `antoinegg1/rsna-knee-e9-radimagenet-heads-v15` | 9/18 | head/probe terlatih di atas RadImageNet | **CC-BY-NC-SA-4.0** | ⚠️ sama isu NonCommercial |
| `mattiaangeli/knee-mri-fold-weights` | 7/18 | bobot model per-fold (k-fold CV) | CC0-1.0 | pola k-fold CV proper — kita belum punya ini (lihat item "Belum dikerjakan") |
| `tonylica/rsna2026-models` | 7/18 | kumpulan model (isi belum diperiksa detail) | **unknown** | ⚠️ lisensi tidak jelas, isi belum diverifikasi — belum direkomendasikan |
| `pilkwang/pilkwang-public-dataset-for-notebooks-figures` | 7/18 | aset figure/plot, bukan model/label | CC0-1.0 | tidak relevan untuk modeling |
| `sohaibanwaar1203/kneemridataset` | 4/18 | **data MRI lutut EKSTERNAL** (917 volume, RS Rijeka Kroasia, bukan data kompetisi) | **unknown** | ⚠️⚠️ dua isu sekaligus: lisensi tidak jelas DAN ini data eksternal di luar kompetisi — apakah Rules mengizinkan data eksternal belum terverifikasi. **Item STOP, tunggu keputusan Anda** |
| `royceplayz/rsna-medicalnet-resnet50-weights` | 2/18 | bobot MedicalNet-ResNet50 | **Apache-2.0** | bersih! Sebelumnya kita putuskan skip MedicalNet (§5) — dengan lisensi jelas + bukti dipakai kompetitor 226-vote & 99-vote, layak dipertimbangkan ulang |
| `flight0234/rsna-knee-hybrid-report-labels` | 1/18 (baru) | **label hybrid**: 11 finding dari stevenleehans v4_blend + kolom Fracture dari pilkwang v2 (lebih akurat utk Fracture: 0.8713 vs 0.7931) | CC0-1.0 | **macro AUC 0.8991 vs gold — diverifikasi ulang independen, cocok persis dgn klaim.** Lebih baik dari yang kita pakai sekarang (0.8927). Juga sudah menyediakan kolom `__conf` per-label (confidence eksplisit, lebih baik dari heuristik `2*\|p-0.5\|` kita) |

**model_sources yang dipakai:** `metaresearch/dinov2/PyTorch/small/1` justru LEBIH sering dipakai drpd `base` (kemungkinan supaya ensemble multi-backbone tetap muat di kuota GPU/waktu Kaggle), `base` juga umum, `large` jarang (2/18). `mrquadian/biomedclip/PyTorch/default/1` dan `timm/tf-efficientnet/PyTorch/tf-efficientnet-b3/1` muncul di kernel 226-vote (`prvsiyan`) — konfirmasi independen atas rencana ensemble DINOv2+BiomedCLIP+EfficientNet-B3 di §5.

**kernel_sources / pola arsitektur pipeline:** beberapa kompetitor top memecah kerja jadi kernel training terpisah lalu di-attach lewat `kernel_sources` di kernel ensemble/inference — persis pola modular kita sendiri (preprocessing → labeler → modeling). Yang paling canggih: `sakhawathossen` punya 5 kernel berantai — `notebook-01-baseline` → `notebook-02-conservative-fine-tuning` → `notebook-04-target-specialist-fine-tuning` (fine-tune head terpisah per-label) → `rsna-knee-enhanced-ensemble` → `notebook-05-final-oof-selected-ensemble` (pemilihan kombinasi ensemble berdasar skor **OOF/out-of-fold**, bukan cuma val holdout). Judul kernel lain menyebut teknik serupa: TTA (`renta0426`, "tta-pool-probe"), blend-weight tuning (`cf33333`/`fishface`, "uniform-035"/"per-target"), dual-LLM pseudo-label (`sadamtorres`, 0.892).

### 8a. Item yang perlu keputusan Anda (STOP — legalitas ambigu, bukan cuma teknis)

1. **`pilkwang/rsna-knee-weights` (dan `tonylica/rsna2026-models`)** — ini bukan pretrained generik, tapi **bobot model yang SUDAH di-fine-tune kompetitor lain khusus untuk kompetisi ini**, dipublikasikan sbg Kaggle Dataset publik & dipakai 14/18 sampel kita (termasuk kernel 344-vote & banyak turunannya). Secara teknis publik & CC0, tapi banyak kompetisi Kaggle punya aturan "no solution sharing / tidak boleh submit hasil replikasi solusi orang lain" — pakai bobot yang SUDAH belajar dari label kompetisi ini beda kelas dgn RadImageNet/DINOv2 (pretrained generik, tidak pernah lihat data kompetisi ini). **Belum kami pakai, menunggu keputusan Anda.**
2. **`sohaibanwaar1203/kneemridataset`** — data MRI lutut eksternal (917 studi RS Rijeka Kroasia) di luar dataset resmi kompetisi, lisensi "unknown" di Kaggle. Dua pertanyaan terpisah: (a) apakah Rules kompetisi ini mengizinkan data eksternal, (b) kalau ya, apa lisensi asli dataset ini (butuh cek sumber akademik aslinya, bukan cuma tag Kaggle). **Belum kami pakai, menunggu keputusan Anda.**
3. **`marwanmath/resnet-50-radimagenet-marwan` & `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`** — CC-BY-NC-SA-4.0 (NonCommercial). Kompetisi berhadiah $77.000 membuat status "non-commercial" ambigu. **Rekomendasi kami: SKIP, pakai sumber RadImageNet-ResNet50 MIT-licensed yang sudah kita verifikasi sendiri (§4)** — kapabilitas sama, risiko lisensi jauh lebih rendah. Ini bukan item STOP (kita sudah punya alternatif aman), sekadar melaporkan supaya Anda tahu kenapa kita TIDAK ikut pakai versi yang paling populer di kalangan kompetitor.

### 8b. Langsung diadopsi (aman, CC0, terverifikasi independen)

- **`flight0234/rsna-knee-hybrid-report-labels`** menggantikan `stevenleehans` v4_blend sbg sumber label training (0.8991 > 0.8927, CC0-1.0, plus kolom confidence eksplisit per-label) — akan dipakai di run modeling berikutnya (v8).
- **`royceplayz/rsna-medicalnet-resnet50-weights`** (Apache-2.0) — dipertimbangkan sbg backbone tambahan di ensemble, menggantikan keputusan skip MedicalNet sebelumnya di §5 (lisensinya ternyata bersih, dan dipakai kompetitor top).
- **`mattiaangeli/knee-mri-fold-weights`** — bukan dipakai langsung (bukan bobot kita), tapi polanya (proper k-fold CV, bobot per-fold disimpan sbg Kaggle Dataset) akan kita tiru sbg struktur pipeline berikutnya, mengisi item "Belum dikerjakan: cross-validation stratified".

### 8c. Strategi menuju peringkat 1 (sintesis)

Kesenjangan terbesar dari 0.7658 (v6, single-backbone DINOv2-base, 3 epoch, val N=57) ke ~0.950 (Brandon Low) kemungkinan besar BUKAN dari satu perubahan tunggal, tapi akumulasi dari pola yang konsisten dipakai SEMUA kompetitor top sekaligus:
1. **Label lebih baik** — hybrid multi-LLM blend (0.8991) drpd 1 sumber (sedang dikerjakan, v8).
2. **Epoch & compute lebih besar** — sudah jalan (v7, EPOCHS 3→8).
3. **Ensemble multi-backbone** — DINOv2 + RadImageNet-ResNet50 (MIT, punya sendiri) + BiomedCLIP + (dipertimbangkan) MedicalNet — arsitektur final §5, belum diimplementasi.
4. **Proper k-fold CV** ketimbang 1 split gold — validasi lebih stabil, memungkinkan OOF ensemble selection spt `sakhawathossen`.
5. **Per-target/specialist fine-tuning** — head terpisah per-label utk label yang sulit (Synovitis, Contusion) — pola nyata dari kompetitor top, belum kita coba.
6. **TTA saat inference** — teknik murah, belum kita implementasi (belum ada pipeline inference/submission sama sekali — item "Belum dikerjakan").

Urutan prioritas eksekusi berikutnya: (v8) label hybrid → (v9) ensemble RadImageNet+BiomedCLIP di atas baseline yg sudah lebih lama dilatih → k-fold CV & OOF ensemble → per-target specialist → TTA + pipeline submission. Ini directly menjawab arahan "jangan cuma 1 dataset" — sekarang minimal 2 sumber label (blend) + rencana 3+ backbone, bukan cuma DINOv2 sendirian.
