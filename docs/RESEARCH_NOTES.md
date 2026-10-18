# Research Notes — RSNA Knee Abnormality Detection

Riset Tahap 1 & 2. Dikumpulkan 16 Agustus 2026. Semua klaim disitasi — link ke sumber, atau "kode nyata dari notebook X" dengan referensi Kaggle yang bisa diverifikasi ulang via `kaggle kernels pull <ref>`.

## 0. Catatan metodologi

- **TEROBOSAN (16 Agustus 2026): `kaggle competitions pages <slug> list --content --format json` mengembalikan TEKS ASLI LENGKAP semua halaman kompetisi** (rules, description, evaluation, timeline, data-description, prizes, code requirements, dll) sebagai markdown, langsung dari API resmi Kaggle — bukan scraping, dan sepenuhnya melewati masalah JS-rendering yang selama ini memblokir WebFetch. Command butuh `kaggle config set -n competition -v <slug>` dulu (positional arg kernel/competition di CLI kaggle punya bug parsing kalau ditulis eksplisit). Ini mengganti seluruh pendekatan "Code tab sbg proxy" di bawah untuk pertanyaan spesifik soal Rules/Evaluation/Data — lihat §10 utk hasil lengkapnya (SEMUA item "menunggu verifikasi" sebelumnya sekarang terjawab pasti, bukan inferensi).
- **(Historis, sebelum terobosan di atas) Discussion/Overview/Rules/Evaluation tab Kaggle tidak bisa di-scrape** — semuanya JS-rendered penuh, sudah dicoba berkali-kali di proyek ini (WebFetch cuma dapat title halaman). Ini konsisten di seluruh percobaan, bukan kegagalan sesaat.
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

## 6. Item yang masih perlu verifikasi Anda (tidak memblokir Tahap 3, tapi penting) — **SEMUA TERJAWAB, lihat §10**

1. ~~Rules soal LLM API eksternal & data security~~ **TERJAWAB §10** — Data Security (2.4.b) melarang transmit Competition Data (termasuk teks Report) ke pihak yang tidak ikut kompetisi; API LLM hosted eksternal termasuk pihak itu. Larangan kita sendiri sudah tepat.
2. **Efficiency Prize** — saat ini saya prioritaskan skor mentah sesuai instruksi Anda; beri tahu kalau berubah pikiran. (Formula efficiency score & syarat kelayakan sekarang terverifikasi penuh, lihat §10.)

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

### 8d. Keputusan Anda atas 2 item STOP di §8a

- **`pilkwang/rsna-knee-weights`: "Pakai sebagai warm-start."** Diperiksa isinya lebih dalam: bukan sekadar checkpoint tunggal, tapi **paket inferensi lengkap** — `manifest.json` (20 anggota ensemble, tiap anggota berisi `config` arsitektur + `fingerprint` verifikasi) + 20 file `m_<hash>.pt` (tiap file = `{"model": state_dict, "fingerprint": array}`) + `oof.npz`/`merge_gain.npz`. Kode aslinya (`find_weights`/`infer_from_package`/`check_fingerprint` di `pilkwang/rsna-knee-baseline-v1`) dirancang untuk di-mount sbg dataset lalu langsung dipakai INFERENSI (tanpa training ulang): load tiap checkpoint → `check_fingerprint()` (jalankan model pada input sintetis tetap, tolak kalau output beda dari yang tersimpan — melindungi dari mismatch preprocessing/arsitektur yang senyap) → prediksi tiap anggota dirata-rata pakai **TTA sliding-window antar-slice** → seluruh 20 anggota digabung via **rank-mean** (persentil rank per model, dirata-ratakan — bukan rata-rata probabilitas mentah, supaya kalibrasi antar-model yang beda tidak mengacaukan pembobotan, karena metriknya cuma peduli urutan/rank).

  **Arsitektur checkpoint mereka (DINOv2-**small**, `pool="cls_mean"`) tidak match langsung dgn backbone kita (DINOv2-**base**)** — jadi tidak bisa asal `load_state_dict` ke model kita. Rencana warm-start yang aman & sesuai desain aslinya: dipakai sbg **anggota ensemble independen yang sudah lengkap terlatih** (bukan menyuntik bobotnya ke model kita) — jalankan inferensi mereka dgn kode asli mereka (kredit penuh + link, CC0 mengizinkan reuse), lalu **rank-mean**-kan hasilnya dgn prediksi model kita sendiri di tahap ensemble final. Ini butuh pipeline inferensi/submission kita sendiri berdiri dulu (belum ada — item "Belum dikerjakan") sebelum bisa digabung; direncanakan masuk di tahap v9+ (ensemble) supaya tidak terburu-buru mengintegrasikan tanpa `torch` lokal utk verifikasi.

- **`sohaibanwaar1203/kneemridataset`: "Riset lisensi asli dulu."** Sumber akademik: I. Štajduhar et al., *Semi-automated detection of anterior cruciate ligament injury from MRI*, Computer Methods and Programs in Biomedicine 140 (2017) 151–164, data dari Clinical Hospital Centre Rijeka, Kroasia (917 volume, disetujui Komite Etik CHC Rijeka 30 Agustus 2014, disetujui utk dipublikasikan 23 Mei 2017, consent pasien dianonimkan). **Satu-satunya syarat yang ditemukan (di deskripsi Kaggle uploader maupun hasil pencarian independen): wajib mencantumkan atribusi + sitasi paper di atas** — tidak ditemukan klausa NonCommercial atau larangan redistribusi eksplisit. Halaman resmi asli (`riteh.uniri.hr/~istajduh/projects/KneeMRI/`) saat ini **404/down**, jadi tidak bisa diverifikasi 100% tidak ada syarat tambahan di luar yang sudah ditemukan.

  **Masih ada 1 lapis ketidakpastian terpisah dari lisensi data itu sendiri: apakah Rules kompetisi RSNA Knee ini mengizinkan data eksternal di luar yang disediakan panitia sama sekali** — ini pertanyaan yang sama dgn item terbuka di §6 (halaman Rules tetap JS-blocked, belum bisa diverifikasi langsung). Bukti tidak langsung: dataset ini dipakai terbuka oleh 4/18 sampel kernel top (termasuk turunan dari kernel 344-vote) tanpa tanda-tanda kena moderasi/dihapus Kaggle — sinyal (bukan bukti pasti) bahwa penggunaannya dianggap wajar oleh komunitas & platform. **Keputusan akhir menunggu Anda**: lisensi datanya sendiri tampak aman (atribusi saja), tapi izin Rules kompetisi soal data eksternal tetap belum terverifikasi independen dari kami.

## 9. Riset mendalam: ekosistem dataset "model weights" kompetitor top (arahan lanjutan 16 Agustus 2026)

Memperdalam §8 khusus utk dataset berisi **bobot model terlatih** (bukan label) yang dipakai kompetitor top utk menunjang skor publik mereka. Semua dataset di bawah diperiksa: metadata lisensi, isi file, dan (kalau ada) `README.md`/dataset card.

| dataset | isi | lisensi | catatan legalitas/teknis |
|---|---|---|---|
| `pilkwang/rsna-knee-weights` | paket inferensi lengkap: `manifest.json` + 20 checkpoint `m_<hash>.pt` (tiap file = `{"model": state_dict, "fingerprint": array}`) + `oof.npz` + `merge_gain.npz` | CC0-1.0 | Lihat §8d — arsitektur DINOv2-small, perlu jalur inferensi terpisah (rank-mean ensemble), bukan warm-start langsung ke backbone kita |
| `tonylica/rsna2026-models` | 4 checkpoint `rsna_20260807_v1-v4.pt` (~358MB tiap file) | unknown, tanpa README/deskripsi | Dikonfirmasi via credit note kompetitor top (`sofiaanjenje`, lihat di bawah): "tonylica -- four more trained models that join the ensemble". Isi & format internal belum diverifikasi (tidak ada manifest/dokumentasi spt punya pilkwang) -- belum direkomendasikan tanpa verifikasi lebih lanjut |
| `mattiaangeli/knee-mri-fold-weights` | bobot per-fold (k-fold CV) | CC0-1.0 | pola proper k-fold CV -- teknik-nya (bukan bobotnya) layak ditiru |
| `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads` | 5 checkpoint `rad_head_f{0-4}.pt` (~12.7MB, head saja, bukan encoder penuh) -- README eksplisit: "no encoder weights included (attach marwanmath/resnet-50-radimagenet-marwan)" | CC-BY-NC-SA-4.0 | bergantung langsung pada encoder marwanmath yang sudah kita hindari (NonCommercial) -- skip, konsisten dgn §8a |
| `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` | 1 checkpoint head, dari kernel 226-vote v52 | "other" (custom) | READMEnya eksplisit memperingatkan sendiri: "downstream users remain responsible for complying with... upstream license, including its non-commercial/share-alike restrictions where applicable" -- konfirmasi independen dari kompetitor top sendiri bahwa isu NC RadImageNet ini nyata, bukan kekhawatiran berlebihan kita. Skip |
| `sadamtorres/rsna-ft-a`, `rsna-ft-b` | 5-fold checkpoint besar (~345MB/file) milik pribadi sadamtorres | unknown | dipakai cuma di kernel sadamtorres sendiri (bukan lintas-kompetitor spt pilkwang/tonylica) -- bukan bagian ekosistem bersama, tidak relevan utk kita |
| `sadamtorres/rsna-wheels` | bukan model weights -- cache wheel pip (numpy, pylibjpeg, gdcm) utk install offline | unknown | kategori beda (infra), dicatat saja |
| `royceplayz/rsna-medicalnet-resnet50-weights` | bobot MedicalNet-ResNet50 | Apache-2.0 | sudah dicatat di §8, bersih |
| `marwanmath/resnet-50-radimagenet-marwan` | bobot RadImageNet-ResNet50, dipakai sbg encoder dasar oleh SEMUA head dataset RadImageNet lain di atas | CC-BY-NC-SA-4.0 | sudah dicatat di §8 -- akar dari semua isu NC di tabel ini |

**Temuan paling berharga: kredit eksplisit dari kompetitor top sendiri.** Kernel `sofiaanjenje/rsna-knee-e11-train` (dipakai sbg `kernel_sources` oleh `kunaldesale2408` & `mattiaangeli`, dua entri teratas hasil sort skor kita) mencantumkan bagian "Thank you" yang menjelaskan susunan ensemble komunitas ini:

> - pilkwang -- dua puluh model terlatih yang jadi bagian ensemble ini, dan satu set label hasil baca laporan.
> - stevenleehans dan lixin73 -- dua set label lain hasil baca laporan, supaya satu bacaan bisa dicek silang yang lain.
> - tonylica -- empat model terlatih lagi yang bergabung ke ensemble.
> - marwanmath -- bobot resmi RadImageNet ResNet-50.
> - prvsiyan -- notebook asal fork ini (Apache 2.0).
> - cf696666 -- teknik menyisakan 2 finding dari blend RadImageNet.

Ini mengkonfirmasi: **skor tinggi di leaderboard publik (klaster ~0.89-0.93 di banyak judul kernel) sebagian besar BUKAN dari 1 arsitektur/dataset istimewa, tapi dari akumulasi ensemble 20+4 model terlatih siap-pakai + 2-3 sumber label yang di-blend** -- sebuah "meta-solusi" komunitas yang di-fork & diutak-atik ratusan kali (menjelaskan kenapa begitu banyak kernel dgn judul mirip: V38-V46, "frontier", "reproduction", dll -- semuanya varian tuning-parameter dari resep yang sama). Praktik hati-hati yang juga terlihat: beberapa versi (V40, V41) eksplisit menyebut "no visible-test tuning or identifier is used" / "hidden-test-safe" -- komunitas ini sadar & aktif menghindari overfitting ke leaderboard publik lewat pemilihan bobot blend yang tidak melihat data test.

**Implikasi utk strategi kita:** kita sudah memutuskan (§8d) untuk memakai `pilkwang/rsna-knee-weights` sbg anggota ensemble independen (bukan warm-start ke backbone kita) via rank-mean, setelah pipeline inferensi kita sendiri berdiri. `tonylica/rsna2026-models` (4 model tambahan) adalah kandidat sama tapi belum cukup terdokumentasi (tidak ada manifest/fingerprint spt punya pilkwang) utk diintegrasikan dgn aman -- perlu diperiksa manual isinya (perlu `torch` utk `torch.load`, cuma bisa dilakukan di kernel Kaggle, bukan lokal) sebelum diputuskan pakai/skip.

## 10. Rules resmi terverifikasi LANGSUNG dari API Kaggle (arahan lanjutan 16 Agustus 2026, "cek Rules soal data eksternal")

**Ditemukan cara membaca teks resmi lengkap semua halaman kompetisi** (lihat §0) -- ini bukan lagi inferensi/sinyal tidak langsung, tapi teks asli dari `kaggle competitions pages rsna-knee-abnormality-detection list --content --format json`. Semua item yang sebelumnya "menunggu verifikasi" di §6 sekarang **terjawab pasti**.

### 10a. External Data & pretrained model -- DIIZINKAN, eksplisit

Halaman **Code Requirements** (paling tegas, khusus kompetisi ini):
> - CPU Notebook <= 9 hours run-time
> - GPU Notebook <= 9 hours run-time
> - Internet access disabled
> - **Freely & publicly available external data is allowed, including pre-trained models**
> - Submission file must be named `submission.csv`

Rules umum §2.6 (EXTERNAL DATA AND TOOLS) menguatkan dgn syarat yg sama: publicly available + equally accessible + no cost ke peserta lain ("the use of external data and models is acceptable unless specifically prohibited by the Host" -- kompetisi ini TIDAK melarangnya, malah eksplisit mengizinkan di atas).

**Implikasi konkret:**
- **`sohaibanwaar1203/kneemridataset`** (item STOP di §8a/8d) -- **DIIZINKAN.** Dataset Kaggle publik, gratis, sama-sama bisa diakses semua peserta -- memenuhi syarat persis. Boleh dipakai (mis. sbg data pretraining/augmentasi tambahan), dgn tetap mencantumkan atribusi ke Štajduhar et al. 2017 sesuai permintaan sumber aslinya.
- **`pilkwang/rsna-knee-weights`, `tonylica/rsna2026-models`, dan seluruh dataset weight/label publik lain di §8-9** -- juga eksplisit **DIIZINKAN** ("including pre-trained models"), tidak perlu was-was lagi soal "solution sharing".
- Rules §3.5.d (Private Sharing) menegaskan dari sudut lain: **"It's okay to share code if made available to all Participants on the forums"** -- publikasi Kaggle Dataset/kernel publik = persis pola ini, dan Rules §3.6.b bahkan MEMAKSA lisensi OSI-approved pada kode yg dibagi publik di kompetisi ini, terlepas dari tag lisensi yg dicantumkan uploader-nya.

### 10b. Lisensi NC (marwanmath dkk.) -- bukan dilarang Rules, tapi tetap ada risiko terpisah dari lisensor asli

Rules §2.5.a.4 (WINNER LICENSE) eksplisit mengantisipasi kasus ini: *"In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license... for that data and/or model(s)."* Jadi Kaggle **tidak melarang** dependensi berlisensi NC/incompatible -- cuma tidak memaksa kita relisensi bagian itu kalau menang. **Tapi ini beda dgn pertanyaan apakah lisensi ASLI (CC-BY-NC-SA-4.0 dari `marwanmath`) sendiri mengizinkan dipakai di kompetisi berhadiah uang** -- itu urusan lisensor, di luar kendali Rules Kaggle. **Rekomendasi tidak berubah: tetap skip, pakai RadImageNet-ResNet50 MIT-licensed milik kita sendiri** -- bukan lagi krn Rules melarang, tapi krn ini pilihan lebih bersih & tidak perlu menanggung risiko itu sama sekali padahal ada alternatif setara.

### 10c. Data Security -- mengonfirmasi larangan API LLM hosted kita sendiri

Rules §2.4.b: *"You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition."* Teks laporan radiologi (`Report`) adalah Competition Data. API LLM hosted eksternal (OpenAI/Anthropic/dll di luar Kaggle) adalah "a party not participating in the Competition". **Larangan kita mengirim teks report ke API LLM hosted sudah tepat dan sekarang berdasar teks Rules langsung, bukan cuma kehati-hatian.**

### 10d. Evaluasi -- 100% terkonfirmasi, bukan lagi inferensi

Halaman Evaluation: *"Submissions are evaluated by the average AUC ROC between predicted confidence scores and observed targets across the twelve targets... macro-averaged AUC ROC."* Formula eksplisit: $\frac{1}{12}\sum_{i=0}^{11} AUC_i$ -- **persis** yang sudah dipakai `macro_auc()` di training loop kita sejak v1. Submission: 1 baris/studi test, header `StudyInstanceUID` + 12 kolom label, skor `[0,1]`.

### 10e. Format kompetisi -- code competition, submission via Notebook, batas ketat

- **Submission HARUS lewat Notebook** (bukan file upload biasa), nama file `submission.csv`.
- **Internet HARUS mati** saat run submission (beda dari kernel R&D kita yg `enable_internet:true` -- notebook submission final NANTI wajib pakai `dataset_sources`/`model_sources` utk semua aset, tidak boleh `pip install` on-the-fly dari internet).
- **<=9 jam runtime** (CPU maupun GPU) utk 1 run submission penuh -- termasuk decode ~1.300 studi test + seluruh inferensi ensemble. Ini batas keras yg jadi acuan desain pipeline inferensi kita nanti (mis. berapa banyak anggota ensemble & TTA window yg realistis muat dlm 9 jam).
- Test set: **~1.300 studi**, kolom `Report` **TIDAK disediakan saat testing** -- konfirmasi bahwa pendekatan report-text (weak-labeler/LLM label) HANYA bisa dipakai utk target training, tidak pernah sbg input model saat inferensi -- sudah sesuai desain kita (gambar-only di forward pass).
- **Deadline:** Final Submission 22 Oktober 2026, 23:59 UTC. Entry/Team Merger deadline 15 Oktober 2026.
- **Efficiency Prize**: formula $\text{Efficiency} = \frac{AUC}{Benchmark - \max AUC} + \frac{RuntimeSeconds}{32400}$ (minimalkan), syarat: submission itu ranking di atas benchmark `sample_submission.csv` di Private LB dan termasuk salah satu Final Submission yg dipilih. Track terpisah dari leaderboard utama, konsisten dgn temuan sebelumnya di §2.
- **Winner's obligation kalau menang**: wajib serahkan training+inference code, model weights (sbg Kaggle Dataset publik), deskripsi metodologi lengkap, & sanggup direproduksi dari deskripsi itu -- relevan utk kerapian dokumentasi kita ke depan (`PROGRESS.md`/`README.md` sudah mengarah ke sana).

## 11. Leaderboard sungguhan + riset mendalam strategi peringkat 1 (arahan 17 Agustus 2026, "riset dulu dengan mendalam")

### 11a. Leaderboard publik sungguhan (`kaggle competitions leaderboard`)

Top 20 (17 Agustus 2026):

| # | tim | skor |
|---|---|---:|
| 1 | Brandon Low | 0.950 |
| 2 | MKhlystun | 0.947 |
| 3 | CloseAI | 0.946 |
| 4 | Lukas Nissen Molvær | 0.945 |
| 5-6 | qkrtkddnjs, Pizza Boy | 0.942 |
| 7 | Matteo Vitali | 0.941 |
| 8-9 | DaviDavian, Aastik Rajan15 | 0.940 |
| 10-11 | Ignat, Siontist | 0.939 |
| 12 | JOLEE | 0.938 |
| 13-14 | Sida Zuo, Raihan | 0.937 |
| 15-16 | rhoskeri, Marcin Jarosz | 0.936 |
| 17-19 | Chikuwabu, codex rybaka, xnx | 0.935 |
| 20 | Pirhosseinlou | 0.934 |

**Temuan penting: top-20 sangat rapat (spread cuma 0.016).** Peringkat 1 BUKAN outlier liar — ini kompetisi ketat di puncak, bukan 1 solusi ajaib yang jauh di depan.

### 11b. Kalibrasi kritis: skor lokal gold-58 vs skor leaderboard sungguhan

Ditemukan di catatan `sadamtorres/dinov2-dual-llm-pseudo-labels-0-892` (dibaca lebih dalam): **skor macro AUC dari 58 studi gold lokal secara konsisten MENAKSIR TERLALU RENDAH skor leaderboard sungguhan**, dengan offset sekitar **+0.04** diamati di 3 sistem independen (2 punya sadamtorres sendiri + 1 sistem lain yg dipublikasikan dgn arsitektur berbeda, offset-nya sama sampai per-seribu). Kutipan: *"Both systems with measured gold-58 landed 0.044 above it on the leaderboard... Three independent systems, one constant."*

**Implikasi utk kita**: skor lokal v7 kita (0.7811) kemungkinan setara **~0.82-0.83 di leaderboard sungguhan** kalau disubmit -- BUKAN jaminan pasti (offset ini diukur utk pipeline/label mereka, bukan punya kita persis), tapi arah & besaran kasarnya masuk akal karena mekanismenya jelas: 58 gold studies dianotasi radiologis dari GAMBAR, sedangkan target training kita berasal dari LLM membaca TEKS laporan -- keduanya kadang tidak sepakat, dan gold-58 (N kecil) jadi ukuran yg lebih ketat/noisy drpd apa yg sebenarnya diukur leaderboard (~1.300 studi test). **Ini alasan kuat utk segera dapat 1 submission nyata** -- baru dari situ kita punya angka kalibrasi jujur, bukan tebak-tebak dari 57 studi.

### 11c. Skor tertinggi yang benar-benar terdokumentasi publik (bukan cuma judul kernel)

Dari `prvsiyan/rsna-knee-read-the-report-then-the-knee` (226 votes, scorecard eksplisit, tiap baris dikaitkan ke submission Kaggle yg selesai):

| recipe | skor publik terverifikasi |
|---|---:|
| Pilkwang V15 (20-checkpoint DINOv2-small ensemble, anchor publik) | 0.891 |
| + EfficientNet-B3 5-fold (V49) | 0.898 |
| V46 | 0.899 |
| + 20% RadImageNet rank-blend (V52, **skor tertinggi notebook ini**) | **0.906** |
| `yashbishnoi98/rsna-knee-infer-v1` (independen: LLM lokal Qwen3.6-35B baca SEMUA 4.349 laporan, bukan cuma yg bergold) | 0.903 |

**Temuan penting & agak menyadarkan**: skor publik tertinggi yg BENAR-BENAR terdokumentasi dgn rantai submission lengkap (bukan cuma disebut di judul kernel) ada di kisaran **0.90-0.906** -- masih ~0.03-0.05 di bawah batas bawah top-20 (0.934). Artinya tim-tim peringkat atas (termasuk Brandon Low) kemungkinan besar TIDAK cuma menjalankan resep publik yang beredar -- ada penyempurnaan tambahan (lebih banyak fold/member, tuning blend lebih cermat, spesialis per-label lebih banyak, atau teknik yang tidak dipublikasikan) di luar apa yang bisa kita amati dari Code tab manapun.

### 11d. Pelajaran teknis konkret (bukan cuma angka)

1. **Domain adaptation (fine-tune backbone) >> resolusi lebih tinggi.** Temuan `sadamtorres`: fine-tune di resolusi SAMA (224px) menggerakkan skor label fokal jauh lebih besar drpd menaikkan resolusi:

   | label | frozen | fine-tuned @224px | Δ |
   |---|---:|---:|---:|
   | Medial Meniscus | 0.679 | 0.850 | **+0.171** |
   | MCL | 0.708 | 0.825 | **+0.118** |
   | ACL | 0.727 | 0.840 | **+0.113** |

   Resolusi (224→336px) cuma +0.017. **Implikasi konkret**: arsitektur kita saat ini `unfreeze_last=2` (2 blok terakhir DINOv2-base) -- ini fine-tune ringan. Layak dicoba naikkan (unfreeze lebih banyak blok) sebelum berinvestasi ke resolusi lebih tinggi.

2. **Rank-mean, bukan rata-rata probabilitas, di SETIAP tahap ensembling** -- dikonfirmasi berulang (`pilkwang`, `prvsiyan`, `sadamtorres`): metrik AUC cuma peduli urutan, rata-rata sigmoid membiarkan model paling percaya diri mendominasi.

3. **Diversitas arsitektur asli > diversitas superfisial.** `sadamtorres`: model 224px & 336px (backbone SAMA) berkorelasi 0.901 meski akurasi lokal nyaris sama -- gain ensembling cuma +0.006. Backbone yang GENUINELY beda keluarga (DINOv2 vs RadImageNet-ResNet50 vs BiomedCLIP) akan lebih terdekorelasi -- inilah alasan rencana ensemble 3-backbone kita (§5/§8c) tepat, dan ensemble sesama-DINOv2-beda-hyperparameter kurang bernilai.

4. **Validasi 2-tingkat**: `sadamtorres` sempat SALAH gate 1 stage krn cuma percaya OOF-vs-pseudolabel (yg ternyata under-state perbaikan asli, krn makin bagus model makin "tidak setuju" dgn pseudo-label yg salah). Pelajaran: **pakai OOF/pseudo-label utk deteksi kerusakan murah (N besar), tapi keputusan arah pakai 58 gold** (N kecil tapi ground-truth asli) -- jangan buang perubahan cuma krn OOF turun tipis.

5. **Bahaya k-fold naif #1**: kalau label berubah lalu fold di-generate ulang dgn seed sama, partisi BISA berubah (re-stratifikasi bereaksi ke label baru) -- studi yang tadinya training utk 1 tahap bisa "bocor" jadi validation di tahap berikutnya kalau warm-start dipakai lintas tahap. **Fix: transplant kolom fold lama, jangan hitung ulang.**

6. **Bahaya k-fold naif #2**: sebagian laporan radiologi **identik persis** (template laporan "knee normal" dst.) di banyak studi berbeda -- kalau split fold naif (per studi), studi kembar-laporan bisa kepisah training/validation, dan model "curang" krn sudah pernah lihat teks yang sama persis. `prvsiyan`: split berdasar **hash teks laporan**, bukan per-studi, supaya grup laporan identik selalu utuh di 1 sisi. **Wajib diterapkan saat kita bangun k-fold CV sendiri.**

7. **Per-target specialist terbukti kasih gain nyata, tervalidasi ketat**: spesialis Synovitis (`blacklions`, frozen DINOv2-base RTA-HMIL) naikkan Synovitis AUC 0.742→0.826 di 58 gold (dipilih via 4-fold stratified 500x resampling, positif di 99.4% partisi). Spesialis PCA per-label naikkan Lateral Meniscus +0.0273, Synovitis +0.0185, divalidasi leave-one-out semua positif. Bukan asal tambah kompleksitas -- semua diverifikasi ketat sebelum dipakai.

### 11e. Sintesis: apa yang realistis bisa kita lakukan

**Jujur**: skor publik TERDOKUMENTASI tertinggi yang bisa kita verifikasi (~0.906) masih di bawah batas bawah top-20 (0.934), apalagi peringkat 1 (0.950). Tidak ada resep publik manapun yang terbukti sampai ke sana. Tapi ada jalur konkret, berbasis-bukti, yang belum kita jalankan penuh:

1. **Dapatkan 1 submission nyata SECEPATNYA** setelah baseline+label terbaik siap -- supaya kalibrasi lokal-vs-leaderboard bukan tebakan lagi (§11b).
2. **Unfreeze lebih banyak blok DINOv2** (bukan cuma naikkan resolusi) -- domain adaptation adalah pengungkit terbesar per bukti §11d.1.
3. **Ensemble 3-backbone SUNGGUHAN berbeda keluarga** (DINOv2 + RadImageNet-ResNet50 + BiomedCLIP) + gabung dgn `pilkwang/rsna-knee-weights` (20 model, legal dipakai) via **rank-mean**, bukan cuma tambah lebih banyak DINOv2.
4. **k-fold CV yang benar** — grouping per hash-laporan (bukan per-studi), transplant fold kalau label berubah, validasi 2-tingkat (OOF besar utk deteksi rusak, gold-58 utk keputusan arah).
5. **Per-target specialist tervalidasi ketat** utk label yang secara konsisten lemah di semua sumber kita (Synovitis, Fracture, Contusion) — dgn ambang validasi statistik yg jelas sebelum dipakai (bootstrap/permutation, bukan 1 angka).
6. Setelah dapat submission nyata & kalibrasi jujur, **iterasi berdasar gap yang benar-benar terlihat**, bukan cuma angka lokal N=57.

Ini bukan jaminan pasti sampai peringkat 1 — bahkan tim top-20 kemungkinan pakai penyempurnaan yang tidak bisa kita amati dari Code tab publik manapun. Tapi jalur di atas: (a) berbasis bukti nyata dari kompetitor top, bukan tebakan, (b) membawa kita realistis ke rentang 0.90-0.93 (ceiling publik yg terdokumentasi) sbg checkpoint jujur, lalu (c) dari situ baru masuk akal menutup gap ke 0.934+.

## 12. Resep persis 0.912-0.916: bagaimana kombinasi dataset benar-benar menaikkan skor (arahan lanjutan 17 Agustus 2026)

Ditemukan & dibaca penuh 3 kernel "reproduction" (`ieshanmeghani/rsna-knee-v38-public-0-912-reproduction`, `.../v39-public-0-916-reproduction`, `sofiaanjenje/rsna-knee-frontier-0-916-reproduction`) — kernel ini secara eksplisit dibuat untuk MEREPRODUKSI submission Kaggle yang sudah selesai & bernilai persis 0.912 dan 0.916, bukan cuma klaim di judul. Ini kasih rantai sebab-akibat yang jelas, bukan cuma tabel skor.

### 12a. Rantai skor lengkap: dari 0.891 (1 dataset) ke 0.916 (kombinasi + tuning tervalidasi)

| tahap | resep | skor publik |
|---|---|---:|
| 1 | `pilkwang/rsna-knee-weights` SENDIRIAN — 20-model DINOv2-small rank-mean ensemble | 0.891 |
| 2 | + `tonylica/rsna2026-models` (4 model tambahan) + label diblend dari **3 sumber** (stevenleehans, lixin73, pilkwang) — ini "Bend the Knee Ensemble", resep komunitas yang sama persis dgn credit note di §9 | **0.912** (V38) |
| 3 | + arm independen `marwanmath/resnet-50-radimagenet-marwan` (encoder RadImageNet-ResNet50 beku + head terlatih), digabung ke ensemble tahap 2 lewat **blend rank bertingkat** dgn bobot α dipilih via **nested 5-fold grouped CV** di 58 studi gold, **dideploy konservatif** di α=0.35 (bukan α=0.60-0.70 yg sebenarnya dipilih nested-CV jujur) | **0.916** (V39, "E10 uniform_035") |

**Tepat di kisaran yang Anda tanyakan (0.914)** — antara tahap 2 (0.912) dan tahap 3 (0.916), beda 0.004 murni dari cara MENGGABUNGKAN arm RadImageNet, bukan dari model baru.

### 12b. Mekanisme blend yang sebenarnya (bukan cuma "gabungkan saja")

```python
blend = (1 - alpha) * base_rank + alpha * rad_rank   # base = rank ensemble 20+4 model DINOv2
                                                        # rad  = rank arm RadImageNet-ResNet50 independen
alpha_grid = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, ...]
# alpha dipilih via NESTED 5-fold grouped CV di 58 gold: latih pilihan alpha di 4 fold,
# UKUR di fold ke-5 yg tidak pernah dilihat -- bukan pilih alpha terbaik di seluruh 58 sekaligus
# (itu overfitting terhadap sample kecil, persis yg diwanti-wanti user riset ini sejak awal)
```

**3 hal penting yg bikin ini beda dari sekadar "coba-coba bobot":**
1. **Nested CV, bukan flat CV** — supaya bobot yang dipilih bukan hasil mengintip data yg nanti dipakai mengukur bobot itu sendiri. Catatan penulisnya sendiri: *"selecting the rung on four grouped folds and scoring the fifth picks 0.60... and never picks 0.20. So every deployable rung at or below 0.35 is below what honest selection would choose, which is the answer to 'you tuned on the 58 gold rows'."* — mereka SENGAJA deploy lebih konservatif (0.35) drpd pilihan "jujur" nested-CV (0.60-0.70), demi keamanan generalisasi krn N=58 kecil.
2. **Per-target gate**: 2 label (`Baker's`, `Fracture`) di-exclude dari blend RadImageNet krn terbukti regresi kalau di-blend — dipertahankan 100% dari ensemble DINOv2 murni. Bukti nyata bahwa blending itu tidak selalu menang di semua label, harus dicek per-label.
3. **Ditemukan juga** (di kernel lain, `sofiaanjenje/rsna-knee-e11-train`, versi "V41: alpha 0.60") — versi yang PAKAI bobot "jujur" nested-CV (0.60) alih-alih yang konservatif (0.35) juga ada & dipublikasikan, kemungkinan skornya LEBIH TINGGI dari 0.916 (belum kami verifikasi angka pastinya — worth dicoba).

### 12c. E11: lever belum dipakai siapa pun secara luas — diversitas di PIKSEL, bukan di bobot

Ditemukan potongan kode "E11" (mode training-only, belum masuk submission resmi manapun yg kami temukan) di kernel yg sama: menambah **arm ketiga** yang inputnya beda secara fisik, bukan cuma beda bobot model — 3 slot NON-fat-suppressed + 1 slot fat-suppressed, di-crop fisik ke 130mm (field-of-view lebih sempit & konsisten, di bawah FOV akuisisi 99.6% series). Alasan eksplisit penulisnya: *"the existing arm reads three fat-suppressed slots... nothing has looked at a non-fat-suppressed series, where meniscal and ligament morphology is conventionally read."* Ini levers yang BELUM dieksploitasi luas oleh komunitas — kandidat kuat utk diferensiasi kita sendiri drpd cuma mengikuti resep yang sudah dipakai ratusan fork.

### 12d. Implikasi konkret untuk kita

Kita SEKARANG (setelah §10-11) tahu persis resep 0.912-0.916, dan semua bahannya legal dipakai:
1. `pilkwang/rsna-knee-weights` (20 model) + `tonylica/rsna2026-models` (4 model, masih perlu diverifikasi isinya di kernel Kaggle sblm dipakai — §9) → basis rank-ensemble ~0.891-0.912 tanpa kita training apa pun.
2. Tambahkan arm RadImageNet-ResNet50 **milik kita sendiri** (MIT-licensed, §4) — bukan `marwanmath` (NC-licensed) — dilatih head-nya di atas fitur RadImageNet beku, di-blend via rank dgn bobot α dipilih **nested grouped-CV di 58 gold**, deploy konservatif di awal.
3. Tambahkan BiomedCLIP (§5) sbg arm ke-3 — diversitas arsitektur asli, bukan cuma DINOv2 varian lain.
4. **Coba E11-style**: arm ke-4 dgn piksel yg genuinely beda (non-fat-suppressed, crop fisik berbeda) — lever yang belum banyak dipakai kompetitor, potensi diferensiasi nyata drpd cuma mengulang resep yang sudah dipakai ratusan fork.
5. Per-target gate: cek tiap label, jangan blend kalau regresi (pola `Baker's`/`Fracture` di atas).

Ini realistis membawa kita ke **kisaran 0.90-0.92** relatif cepat (basis komunitas legal + arm tambahan milik sendiri), dan kombinasi dgn item lain di §11e (fine-tune lebih dalam, k-fold proper, per-target specialist tervalidasi, E11-style diversitas piksel) adalah jalur paling masuk akal utk mendekati/melewati klaster top-20 (0.934+).

## 13. Bug leakage kritis di percobaan pertama kita sendiri + cara benar (17 Agustus 2026)

**Percobaan pertama** (`kaggle_kernel_ensemble` v1): jalankan SEMUA 20 model paket `pilkwang/rsna-knee-weights` ke SEMUA 57 studi gold kita, rank-mean. Hasil: **macro AUC 0.9954** — hampir sempurna, banyak label persis 1.000.

**Ini SALAH, bukan pencapaian.** Root cause: tiap dari 20 model punya `fold` assignment (k-fold CV, field `fold: 0-4` di `manifest.json`) — artinya tiap model dilatih pakai ~4/5 data (termasuk sebagian studi gold, krn gold adalah SATU-SATUNYA ground truth asli yg tersedia utk siapa pun berlatih dgnnya) dan cuma held-out dari 1/5 fold-nya sendiri. Menjalankan SEMUA 20 model ke SEMUA 57 studi & rata-ratakan berarti tiap studi gold diskor oleh ~16 dari 20 model yg SUDAH PERNAH MELIHAT studi itu saat training mereka — bukan evaluasi, tapi menanyakan jawaban ke murid yg sudah hafal soalnya sendiri. Field `manifest.json` per-member `holdout: 0.838` (skor OOF jujur mereka sendiri) langsung membocorkan bahwa 0.9954 mustahil benar.

**Perbaikan**: paket `pilkwang/rsna-knee-weights` TERNYATA sudah menyediakan `oof.npz` — prediksi **out-of-fold resmi** (tiap studi diskor HANYA oleh model yg fold-nya meng-exclude studi itu dari training), utk seluruh 4.407 studi training, siap pakai (`ids`, `pred`, `gold_mask`, `targets`). Diukur ulang pakai `oof.npz["pred"]` (bukan live-inference 20-model kita sendiri) terhadap 57 gold kita:

| label | AUC (OOF resmi, leak-free) |
|---|---:|
| ACL | 0.8649 |
| MCL | 0.8750 |
| Medial Meniscus | 0.8462 |
| Lateral Meniscus | 0.6317 |
| Medial OA | 0.9492 |
| Lateral OA | 0.7055 |
| PF OA | 0.7976 |
| Effusion | 0.9403 |
| Synovitis | 0.7346 |
| Baker's | 0.9537 |
| Contusion | 0.8407 |
| Fracture | 0.8932 |
| **MACRO** | **0.8360** |

**Jauh lebih masuk akal** — dekat dgn klaim publik pilkwang (0.891 di leaderboard sungguhan), dan cocok dgn kalibrasi +0.04 dari §11b (0.836 + 0.04 ≈ 0.876, dalam jangkauan wajar 0.891 mengingat gold-N kecil & noisy). **Angka inilah yang valid dipakai sbg baseline "ensemble komunitas" utk dibandingkan/di-blend dgn model kita sendiri** — bukan 0.9954.

**Implikasi metodologi (langsung menegaskan lagi pelajaran §11d.4 dari sadamtorres, sekarang dibuktikan lewat kesalahan kita sendiri, bukan cuma dibaca):**
1. **Jangan pernah evaluasi model pihak lain terhadap sample yg overlap dgn training mereka**, kecuali punya bukti eksplisit exclusion (fold assignment, holdout flag) DAN benar-benar dipakai saat scoring — bukan diasumsikan.
2. **Kalau paket eksternal menyediakan OOF resmi (`oof.npz` di sini) — PAKAI ITU, jangan re-infer sendiri.** Lebih murah (tidak perlu decode+forward-pass ulang di Kaggle), dan otomatis benar krn metodologi exclusion-nya sudah mereka desain dgn benar sejak awal.
3. Utk v2 ensemble (blend dgn model kita + arm RadImageNet), sumber prediksi pilkwang yg dipakai adalah **`oof.npz["pred"]` utk 4.349 studi non-gold** (target training/blend-weight-selection) **bukan** hasil live-inference kernel v1 kita yg dibuang krn bocor.

## 14. Mencari bukti skor publik di atas 0.916 (arahan lanjutan 17 Agustus 2026, "saya mau melewati skor 0.950")

Dicari lebih jauh kernel dgn klaim skor 0.92-0.95 di judul, ditarik & dibaca 3 kandidat: `nikhil00718/rsna-knee-0-95-dinov2-v2` ("DINOsaur V2"), `ieshanmeghani/rsna-knee-v41-dinov3-e10-alpha060`, `ranjithragavan07/rsna-knee-dinov2-0-93`.

**Temuan penting soal keandalan klaim skor di judul kernel**: kernel "reproduction" (V38=0.912, V39=0.916, §12) semuanya mencantumkan **sitasi eksplisit** ke submission Kaggle yang sudah selesai (mis. "submission 55399732", SHA-256 kode sumber) — bisa diverifikasi. Sebaliknya, `ranjithragavan07/...-0-93` dan `nikhil00718/...-0-95-...` **TIDAK** mencantumkan sitasi serupa di badan notebook — 0.93 dan 0.95 di judul tampak **aspirasional/klaim penulis**, bukan skor submission yang tervalidasi. `nikhil00718`-nya sendiri malah menulis eksplisit: *"A leaderboard improvement is plausible but cannot be guaranteed without executing against the hidden test set."* — penulisnya sendiri tidak mengklaim kepastian.

**Yang genuinely baru & berguna (terlepas dari validitas angka judul):**
- **`ranjithragavan07`'s V40/V41** (turunan `sofiaanjenje/rsna-knee-e11-train`): menambah **anggota DINOv3 ViT-S/16** ke ensemble 20-model DINOv2 — "self-supervised dan pretrained tanpa label, lalu di-fine-tune di data kompetisi ini", digabung via rank-blend. Beda dgn asumsi awal kita ("tidak ada DINOv3 resmi di Kaggle Model" — §1) — di sini penulisnya melatih DINOv3 sendiri dari sumber lain (bukan Kaggle Model resmi Meta), jadi keabsahan sumber bobot dasarnya perlu dicek terpisah sebelum ditiru, tapi tekniknya (DINOv3 sbg arm tambahan yg genuinely beda arsitektur) valid & konsisten dgn prinsip diversitas §11d.3.
- **Agregasi lebih canggih**: token dari SEMUA series dalam 1 studi digabung jadi 1 sequence key/value (bukan diringkas per-slot dulu), dgn embedding tipe (plane × fat-suppression) per token, lalu 12 query per-diagnosis attend ke seluruh sequence itu (`segment_softmax` per-studi). Ini strictly lebih ekspresif drpd slot-mean-lalu-attention yang kita (dan pilkwang) pakai — kandidat upgrade arsitektur head kita.
- **`nikhil00718`'s "DINOsaur V2"** (independen, bukan turunan pilkwang): kumpulan teknik matang — kalibrasi monotonic per-fold, **fold dikelompokkan per hash-laporan duplikat** (persis pelajaran §11d.6, dikonfirmasi independen lagi), unfreezing progresif dgn learning-rate per-layer, **greedy checkpoint soup** (rata-rata beberapa checkpoint terbaik, bukan cuma pilih 1), gamma TTA, dual-GPU DataParallel, fallback submission.csv langsung di awal (pola sama dgn `write_benchmark_submission()` pilkwang, dikonfirmasi jadi praktik umum yg baik).

**Kesimpulan jujur**: kami tidak menemukan bukti tervalidasi (sitasi submission selesai) untuk skor publik di atas **0.916** dari sisi Code tab manapun yang bisa diverifikasi. Klaim 0.92-0.95 di beberapa judul kernel tampak aspirasional. Tim top-20 leaderboard (0.934-0.950) kemungkinan besar mencapainya lewat kombinasi teknik-teknik di atas (E10 alpha lebih tinggi, DINOv3 arm, agregasi cross-attention penuh, checkpoint soup, fold-hash-grouping) yang dijalankan & di-tuning lebih jauh drpd versi publik mana pun yang bisa kami baca — bukan 1 trik rahasia tunggal, tapi akumulasi eksekusi yang lebih matang dari resep yang sama. **Tidak ada jalan pintas terverifikasi ke atas 0.950** — jalur realistis tetap: bangun ensemble kita sendiri setinggi mungkin (0.90-0.92+ dari §12-13), dapat submission nyata, lalu iterasi pakai teknik-teknik tambahan di atas (checkpoint soup, agregasi cross-attention, fold-hash-grouping, arm ke-4 genuinely beda) berdasar gap yang benar-benar terlihat.

## 15. Discussion tab — sumber baru, jauh lebih berharga dari Code tab (arahan lanjutan 17 Agustus 2026)

**Terobosan metodologi**: `kaggle competitions topics list` + `topic-messages` (belum pernah dipakai sepanjang sesi ini) membuka akses ke Discussion tab — dan isinya jauh lebih bernilai drpd Code tab utk pertanyaan strategis, krn ada **komentar host resmi** dan diskusi teknis jujur (termasuk orang mengaku skor & metode asli mereka, sesuatu yg tidak ada di kernel publik yg cenderung "kudos farming").

### 15a. RULING RESMI HOST: API LLM hosted TERNYATA DIIZINKAN (mengubah konstrain awal kita)

Topik resmi host **"Use of Commercially Hosted LLMs"** (733965, diposting 9 Agustus 2026):

> *"Use of commercially hosted LLMs and other external inference services is **permitted**, provided that the service and method of use otherwise comply with the Competition Rules... submitting Competition Data, including report text, to an external LLM or API for inference or other computational processing (for example, extracting labels from reports) will **not, by itself, be considered prohibited PRIVATE SHARING** of Competition Data outside the Team."*

**Ini membalik asumsi/aturan kerja kita sejak awal proyek** ("JANGAN pakai API LLM hosted untuk memproses report text" — dulu berdasar kehati-hatian data pasien, sekarang host eksplisit bilang itu BUKAN pelanggaran Data Security asalkan layanannya "reasonably accessible & minimal cost" ke semua peserta). Diskusi komunitas di 733652 (sebelum ruling ini keluar) sampai pada bacaan yang SAMA dgn kita sendiri (dilarang) berdasar §2.4.b — host mengoreksi bacaan itu secara eksplisit lewat pengumuman resmi.

**Item keputusan Anda (STOP — ini konstrain yang Anda tetapkan sendiri di awal proyek, bukan urusan legalitas lagi krn sudah jelas diizinkan)**: apakah restriksi "jangan pakai API LLM hosted" mau dilonggarkan sekarang? Ini relevan krn ada bukti kuat kualitas label masih jadi pengungkit besar (§15b) — model LLM yang lebih mumpuni (dgn prompting hati-hati) berpotensi kasih label lebih baik drpd yang dipakai komunitas (flight0234 dkk., 0.8991).

### 15b. Single model bisa capai 0.92-0.934 — ensemble 20+ model BUKAN keharusan

Thread "Best single-model score" (735304) — pengakuan skor asli dari peserta lain (bukan klaim di judul kernel, jadi lebih kredibel meski tidak tersitasi formal):

| klaim | metode |
|---|---|
| 0.887 | DINOv2, single model (starting point penanya) |
| 0.915 | DINOv2, single model |
| **0.934** | single model, **224px** (resolusi sama persis dgn yang kita pakai!) |
| 0.92 | 5-fold ensemble (bukan 20+) |
| 0.92 | single model, "OOF AUC = LB AUC within noise" |
| 0.826 | pembanding: 192px/9 slice/4 slot, masih mencari tahu ceiling-nya input pipeline atau model |

**Komentar penting** dari salah satu peserta skor tinggi: *"Highly doubt the top results are ensembles... Public notebooks just ensemble early on to try and get a high leaderboard score and exposure/kudos on their notebooks. Waste of time IMO."* — mendukung hipotesis bahwa **ekosistem 20-model publik yang kita riset di §8-9 & §12 mungkin BUKAN jalan yang dipakai tim top-20 sungguhan**, melainkan efek viral "kudos farming" di kalangan publik. Petunjuk lain: skor tinggi berkorelasi dgn **runtime rendah** (lihat efficiency leaderboard) — konsisten dgn single/few-model, bukan ensemble besar.

**Implikasi besar utk strategi kita**: mungkin lebih bernilai investasi ke **1 model yang benar-benar kuat** (fine-tune dalam, k-fold proper, resolusi 224px sudah cukup terbukti org lain capai 0.934 di situ) drpd terus mengejar ensemble 20+ model komunitas yang levelnya "cuma" ~0.89-0.916.

### 15c. Teknik sampling slice yang jauh lebih baik dari cara kita ("3+3+3")

Thread "Scaling the encoder bought us nothing (+0.0011)" (735154) — org yang sama juga membagi resep sampling slice detail:
- **DINOv2-small → base cuma +0.0011** (noise floor 0.0020) — kapasitas model BUKAN pengungkit, konfirmasi independen lagi dari temuan kita/sadamtorres.
- **Urutan fisik slice WAJIB** (filename-order cuma benar ~5% dari waktu di dataset ini) — kita sudah tahu ini, dikonfirmasi lagi.
- **Sampling "3+3+3"**: taruh 3 anchor merata (linspace, bukan acak) di jendela tengah stack per series, ambil 3 slice fisik BERDEKATAN di sekitar tiap anchor (jadi 3×3=9), dan tiap grup-3 jadi channel R/G/B 1 input 2.5D — representasi ~10mm depth neighbourhood asli, BUKAN 3 pandangan ~20mm terpisah kalau 9 slice disebar rata. **Ini lebih canggih dari K_SLICE=4 subsample rata-rata yang kita pakai sekarang** (`kaggle_kernel_modeling` §4) — kandidat upgrade konkret.

### 15d. Konfirmasi independen dari komunitas: isu lisensi NC RadImageNet masih terbuka

Thread "Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation?" (735121) — **belum ada jawaban host** (commentCount: 0 saat kami cek). Menguatkan lagi keputusan kita (§10b) utk skip `marwanmath` & pakai sumber MIT sendiri — bahkan komunitas sendiri tidak yakin & masih menunggu klarifikasi resmi yang belum datang.

### 15e. Peringatan kualitas label dari sumber LLM (relevan langsung ke pilihan kita)

Dari thread yang sama (735304): *"Do research on the labels. Learn what the model needs to see. LLMs are very wrong a lot of the time here without careful guidance."* — konsisten dgn temuan kita sendiri (flight0234 0.8991 vs gold, masih ada label lemah spt Synovitis 0.79, Lateral Meniscus di kisaran 0.7-an di beberapa evaluasi) — ruang perbaikan label MASIH ADA, bukan sudah maksimal.

## 16. Kenapa skor "mentok" + bedah forensik resep 0.920 tim sendiri (arahan 18 Agustus 2026, "score mentok, apa strategi ke top 1")

**Konteks baru penting: leaderboard sungguhan tim kita SUDAH DI 0.920** (team "Krazy KIller", `dimaspashaakrilian`+`nxtirfan`, 47 submission), naik dari 0.914 sebelumnya — via submission notebook `tonylica/rsna-knee-dino-radimagenet-rank-ensemble` (bukan model kita sendiri, kernel publik yang di-fork). **Peringkat: #130 dari 1.886 tim.**

### 16a. Leaderboard sungguhan 18 Agustus 2026 (snapshot lengkap, bukan cuma top 20)

| rank | tim | skor |
|---|---|---:|
| 1 | Brandon Low | **0.951** (naik dari 0.950, submit HARI INI) |
| 2 | MKhlystun | 0.947 |
| 3 | CloseAI | 0.946 |
| 5 | Pizza Boy | 0.942 |
| 10 | Matteo Vitali | 0.941 |
| 20 | YT | 0.937 |
| 50 | Ivy | 0.929 |
| 100-140+ | **puluhan tim termasuk kita (#130)** | **0.920 (persis sama, plateau)** |

**Temuan kunci: skor 0.920 bukan pencapaian kita — itu "commodity ceiling".** Dari rank ~100 sampai ~140+ berjejer PULUHAN tim dengan skor 0.920 **identik sampai 3 desimal** — semua nge-fork notebook publik yang sama (`tonylica/rsna-knee-dino-radimagenet-rank-ensemble`, 73 votes, dan turunan²nya spt `amanatar/rsna-knee-super-ensemble-0-920`). Siapa pun yang copy-paste notebook itu otomatis dapat 0.920 — **tidak mendiferensiasi kita, dan bukan jalan ke top 1**, karena top 1-50 semuanya SUDAH melewati commodity ceiling ini (0.929-0.951) dgn sesuatu yang tidak dibagikan publik.

**Gap konkret ke target**: 0.920 → top 50 (0.929) = **+0.009**. → top 20 (0.937) = **+0.017**. → #1 (0.951) = **+0.031**.

### 16b. Bedah forensik: apa isi resep 0.920 (`tonylica/rsna-knee-dino-radimagenet-rank-ensemble`)

Ditarik penuh via `kaggle kernels pull`. Ini "meta-solusi" hasil rantai kredit 12 kontributor publik berbeda (pilkwang, stevenleehans, lixin73, tonylica, marwanmath, prvsiyan, cf696666, romantamrazov, ieshanmeghani, sofiaanjenje, saidmohamedomary, sakhawathossen/ranjithragavan07) — **35 checkpoint + 1 encoder bersama**:

- 20 checkpoint DINOv2-small (5-fold, milik pilkwang) — sama dgn `oof.npz` yang sudah kita evaluasi (0.8360 leak-free)
- 5 checkpoint DINOv3-small (5-fold, milik tonylica)
- 5 attention-head RadImageNet "reference" (E10, 3-plane layout)
- 5 attention-head RadImageNet "E13" (4-slot fat-suppressed-crop layout, dipakai 2x di 2 slot-layout beda)
- 1 encoder ResNet-50 RadImageNet BERSAMA (`marwanmath` — **lisensi CC-BY-NC**, item yang kita SENGAJA skip di proyek ini)

**Pipeline blend (rank-space, bukan probability-space)**:
1. DINOv2 branch: 6 slot @ **336px** (lebih tinggi dari 224px kita), 20 checkpoint → rank-avg
2. DINOv3 branch: 6 slot @ 336px, 5 fold → rank-avg
3. `transformer_parent = 0.55*DINOv2_rank + 0.45*DINOv3_rank`
4. Rad branch: `0.50*reference_head_rank + 0.50*E13_head_rank` → rank lagi
5. **E10** (10 target, exclude Baker's & Fracture): `0.50*transformer_parent + 0.50*Rad_branch` — **persis pola "per-target regression gate" yg sudah kita riset di §12b**, sekarang terkonfirmasi dipakai kernel nyata yang skor tim kita pakai
6. E13 head dijalankan LAGI di slot-layout ke-3 ("E11") → second-pass rank
7. **Final** (12 target): `0.85*E10_branch + 0.15*E11_second-pass`

**Konfirmasi teknis lain via grep kode**: pakai **TTA (flip)**, crop berbasis **laterality**, dan urutan slice fisik dari **`ImagePositionPatient`** DICOM tag (bukan filename-order) — semua ini SEKARANG jadi standar di kernel publik top, bukan lagi trik tersembunyi.

### 16c. Bukti independen baru dari Discussion (18 Agustus 2026) — kenapa "tambah checkpoint" bukan lagi jalan maju

- **Thread "What is the real bottleneck now?" (735826)**: eksperimen terkontrol asli (bukan klaim) — satu kompetitor menukar backbone ResNet34→DINOv2-Small dgn PIPELINE & LABEL SAMA PERSIS, hasilnya **DINOv2 JUSTRU LEBIH BURUK** (OOF BCE 0.674 vs ResNet34 0.664). Kesimpulan kerja mereka: **"image geometry / slice selection + label quality > backbone size"** — validasi independen kuat utk arah yang sudah kita curigai (§15c: DINOv2-small→base cuma +0.0011).
- **Thread "What Could the Final Ceiling Be?" (735767)**: kompetisi baru 12 hari jalan, sudah 1.832 tim (RSNA 2025 total cuma 1.149 tim) — leaderboard bergerak SANGAT cepat. Komentar penting: *"It's pretty easy to spot what looks like conflicting results in the 58 gold labels we have received"* — **58 studi gold ITU SENDIRI ada inkonsistensi internal** (mis. teks "small effusion" dilabel positif walau heuristik brief bilang cuma "moderate/large" yg positif) — validasi lokal kita vs gold-58 punya batas akurasi bawaan yang tidak bisa dihilangkan model manapun.
- **Thread PSA (735854)**: 2 jebakan submission yang WAJIB dicek sebelum kita benar-benar submit kernel sendiri — (1) **codec DICOM `pylibjpeg`/`gdcm` HARUS di-bundle offline** (scored run tanpa internet, dataset ini punya series JPEG Lossless/JPEG2000 yang butuh codec ini, gagal SILENT kalau tidak ada), (2) path competition data saat submission sungguhan ada di `/kaggle/input/competitions/<slug>/...`, BUKAN path flat spt saat dev — kode discovery kita harus tangani dua-duanya.
- **Thread "Two knees are better than one?" (735639)**: ~7 studi train punya KEDUA lutut dalam 1 DICOM — crop naif bisa 100% miss lutut (nge-flag fracture palsu krn crop kena tulang lain). Edge-case kecil tapi nyata utk pipeline crop kita.

### 16d. Kesimpulan & strategi ke top 1 (bukan cuma re-fork commodity ensemble)

**Kenapa "mentok"**: skor 0.8195 (model sendiri) dan 0.920 (tim, via fork) sama-sama representasi TITIK JENUH dari 2 pendekatan berbeda — model sendiri jenuh krn epoch tambahan dgn LR flat cuma overfit (v12), tim jenuh krn 0.920 adalah **lantai commodity** yang otomatis didapat siapa pun yang fork notebook publik yang sama — bukan lantai kemampuan kita. **Untuk lewat 0.920 dan mendekati 0.951, dibutuhkan sesuatu yang TIDAK dibagikan publik** — konsisten dgn §14 ("tidak ada jalan pintas terverifikasi ke atas 0.916" tapi sekarang ceiling publik naik jadi ~0.920).

**Prioritas konkret (diurutkan by leverage & biaya, GPU kuota masih 0 sampai 22 Agustus)**:
1. **[Bisa mulai SEKARANG, CPU-only, TIDAK butuh kuota GPU]** Bangun ulang geometri slice preprocessing: urutan fisik `ImagePositionPatient` + crop sadar-laterality + handle kasus 2-lutut-1-DICOM (735639) + eksplorasi sampling "3+3+3" (§15c). Ini pengungkit dgn BUKTI TERKUAT hari ini (735826, independen dari 2 sumber terpisah) dan belum kita implementasikan sama sekali.
2. **Begitu kuota GPU refresh (~22 Agustus)**: lanjutkan fine-tuning model kita sendiri dari checkpoint v11 (0.8195) dgn **LR scheduler/decay** (bukan LR flat spt v12 yg overfit) — perbaikan langsung & murah utk masalah v12.
3. **k-fold CV proper** (bukan 1 split gold N=57) — kurangi noise dari inkonsistensi label gold-58 yang sekarang terkonfirmasi nyata (735767).
4. **Baru setelah model kita sendiri > ~0.85-0.88 standalone**: masukkan sbg anggota BARU yang genuinely independen ke ensemble (bukan cuma re-fork punya tonylica) — kombinasikan dgn arm RadImageNet MIT-licensed kita sendiri (bukan `marwanmath` NC), via rank-blend nested-CV (resep §12b).
5. **TTA (flip)** — item murah, terbukti dipakai semua kernel top, belum ada di pipeline kita.
6. **Kuota GPU 0 sampai 22 Agustus, tapi kuota TPU 20h BELUM DIPAKAI SAMA SEKALI** — opsi utk tidak menganggur 4 hari: eksperimen kecil `torch_xla` di Kaggle TPU utk lanjutkan progres tanpa nunggu refresh GPU (belum diverifikasi kompatibilitasnya dgn kode kita, perlu uji coba kecil dulu).

## 17. TPU dieksekusi (v13/v14): 12x lebih cepat terkonfirmasi, tapi skor plafon ~0,80-0,82 (arahan 19 Agustus, "kalo sukses lanjutkan eksperimen")

Item 6 di §16d dieksekusi penuh, bukan cuma probe. Ringkas (detail lengkap di `PROGRESS.md` §20-22):

- **Probe TPU** (`rsna-knee-tpu-diagnostic-torch-xla-check`): `torch_xla` 2.8.0 pre-installed, `xm.xla_device()`+matmul+`transformers` semua jalan. Antrian terukur **~28,6 menit**.
- **v13 push 1 (sanity, ERROR)**: `AttributeError: module 'torch' has no attribute 'xla'` — HF gradient checkpointing (`torch.utils.checkpoint._get_device_module`) belum support device `'xla'` di versi torch/transformers ini. Try/except di `gradient_checkpointing_enable()` TIDAK cukup krn error muncul saat forward pass, bukan saat enable dipanggil.
- **v13 push 2 (sanity, sukses mekanis, subset 400 studi)**: fix = matikan grad checkpointing khusus di device XLA. Jalan lancar TANPA OOM walau grad checkpointing off, unfreeze_last=6. Ketemu temuan operasional terpisah: `CKPT_DIR=None` — checkpoint v11 (0,8195) TIDAK bisa diakses via `kernel_sources` krn kernel v11 = kernel yg sama dipakai self-reference v12, dan v12 tidak pernah menang lawan guard `best_auc=0,8195`-nya sendiri sehingga tidak pernah `torch.save()` — versi TERBARU kernel itu (yg selalu di-mount `kernel_sources`) jadi tidak punya `best_model.pt` sama sekali. **Pelajaran umum**: pola self-reference+guard checkpoint aman utk proteksi bobot DI DALAM kernel yg sama, tapi bikin file itu TIDAK terjangkau kernel LAIN via `kernel_sources` begitu ada versi baru yg tidak menimpanya. Kalau butuh akses lintas-versi yg aman, simpan checkpoint sbg Kaggle Dataset terpisah, bukan andalkan `kernel_sources` (yg selalu ambil versi terbaru).
- **v14 (run penuh, dari nol, TPU, 12 epoch, `CosineAnnealingLR`)**: SELESAI ~78 menit. **Kecepatan steady-state ~370 detik/epoch (~6,2 menit) utk data penuh (4.349 studi) — v11 di GPU T4 makan ~74 menit/epoch. TPU ~12x lebih cepat**, melebihi estimasi awal (~10x) dari sanity run. **Tapi val_macro_auc terbaik cuma 0,8159 (epoch 6) — TIDAK melampaui checkpoint v11 (0,8195)**, plateau/turun tipis di epoch 7-12 walau LR terus meluruh ke ~1e-5/1e-7.

**Kesimpulan strategis (bukti independen ke-4, setelah v7/v9/v11/v12)**: arsitektur+config saat ini (DINOv2-base, unfreeze 6 blok, slot attention, label hybrid flight0234) punya **plafon keras ~0,80-0,82 macro AUC** — variasi jadwal training (epoch banyak/sedikit, LR flat/decay, dari-nol/resume, GPU/TPU) semuanya konvergen ke kisaran yg sama. Ini SELARAS PERSIS dgn temuan independen Discussion (735826, §16c) bahwa "geometri slice + kualitas label > ukuran backbone/tuning training". **TPU tervalidasi sbg infrastruktur berharga (12x lebih murah waktu) utk eksperimen berikutnya, tapi BUKAN pengungkit skor dgn sendirinya** — prioritas mutlak sekarang preprocessing v2 (geometri slice) + kualitas label, bukan lagi tuning training-schedule di config yang sama.

## 18. Preprocessing v2 + v15: hipotesis geometri slice TERBUKTI lewat eksperimen sendiri (19 Agustus 2026)

Prioritas §17 dieksekusi penuh sampai hasil terukur, bukan cuma dibangun.

**Preprocessing v2** (`kaggle_kernel_preprocessing_v2/`): `process_one_series` diubah SATU hal — file DICOM per series diurutkan pakai posisi fisik (proyeksi `ImagePositionPatient` ke sumbu normal slice dari `ImageOrientationPatient`) sebelum pilih 16 titik linspace, bukan urutan nama file (v1). Fallback berjenjang: `SliceLocation` → `InstanceNumber` → urutan file asli. Diagnostik `pos_source`/`n_pos_fallback` ditambahkan ke manifest.

**Hasil run penuh (24.371 series, ~72,5 menit, CPU-only): 100% `pos_source=ipp`, 0,00 rata-rata fallback** — dataset kompetisi ini ternyata konsisten menyediakan tag posisi DICOM lengkap, fix terpakai penuh tanpa pengecualian berarti (24.369 OK, 2 ERROR).

**v15** (`kaggle_kernel_modeling_v2geom/`, TPU): eksperimen isolasi variabel murni — arsitektur/hyperparameter identik dgn v14, SATU-SATUNYA beda adalah `kernel_sources` menunjuk preprocessing v2 (bukan v1).

**Hasil: val_macro_auc 0,8304 (epoch 6/8) — MODEL TERBAIK BARU PROYEK**, naik dari checkpoint v11 (0,8195, GPU, cache v1, +0,0109) dan v14 (0,8159, TPU, cache v1, +0,0145). **Ini menembus plafon ~0,80-0,82 yang mentok di 4 eksperimen berturut (v7/v9/v11/v12/v14, semuanya pakai cache v1)** — bukti langsung dari eksperimen sendiri (bukan cuma laporan Discussion) bahwa geometri slice adalah pengungkit nyata, konsisten dgn temuan §17/§16c.

**Implikasi**: (1) item geometri lain yg belum dikerjakan (sampling "3+3+3", crop sadar-laterality, handle 2-lutut-1-DICOM) makin layak diprioritaskan; (2) skor 0,8304 mendekati tapi belum capai ambang "~0,85-0,88" yg direncanakan sbg syarat masuk ensemble (§8c); (3) pola kurva v15 mirip v11 (naik terus sampai epoch 6, dip tipis di 7, parsial pulih di 8) — indikasi epoch lebih banyak atau LR schedule yg disetel ulang (T_max lebih presisi ke titik puncak) berpotensi masih ada ruang naik lagi, belum tentu sudah plateau penuh spt v14.

## 19. v16: sampling "3+3+3" sbg channel RGB — NEGATIF, penyebab teridentifikasi jelas (19 Agustus 2026)

Item "3+3+3" (Discussion 735154) diimplementasikan penuh: preprocessing v3 (3 anchor merata di jendela tengah stack + 2 tetangga fisik/anchor = 9 slice/series, `get_sorted_files` v2 tidak berubah) SELESAI PENUH (24.371/24.371 OK, 0 error, 100% `n_slices_cached=9`, 100% `pos_source=ipp`, ~67,7 menit). v16 (modeling) mengonsumsi cache ini — `KneeStudyDataset._load_slot` reshape (9,224,224)→(3 anchor,3 RGB,H,W), `KneeModel.forward` hapus `.repeat(1,3,1,1)` (channel RGB sekarang genuine dari 3 slice fisik berbeda, bukan grayscale-di-repeat).

**Hasil: val_macro_auc 0,7737 (epoch 8/8, masih naik) — LEBIH BURUK dari v15 (0,8304) DAN v11 (0,8195).** Trainloss juga turun jauh lebih lambat drpd v15/v11 (stuck ~0,48-0,53 selama beberapa epoch), belum konvergen saat `EPOCHS=8` habis.

**Diagnosis (bukan spekulasi kosong, berbasis arsitektur nyata)**: `unfreeze_last=6` di semua eksperimen (v9-v16) berarti **patch embedding DINOv2 (layer paling awal) SELALU tetap beku**. Layer itu di-pretrain di ImageNet mengharapkan channel R/G/B = korelasi warna foto natural. Skema lama (grayscale-di-*repeat*, dipakai v1-v15) aman krn 3 channel identik = backbone tetap efektif melihat citra abu-abu biasa (praktik umum & robust utk fine-tuning backbone RGB-pretrained ke domain grayscale). Skema baru (3 slice fisik genuinely berbeda dipasang sbg R/G/B) menyodorkan pola channel yg sangat jauh dari statistik warna natural yg pernah dilihat patch embedding — krn layer itu beku (tidak trainable), model tidak punya cara belajar menafsirkan pola channel baru ini.

**Pelajaran metodologi**: "3+3+3" sbg TEKNIK SAMPLING (jendela tengah + tetangga fisik) dan "3+3+3" sbg CARA PACKING (jadi channel RGB) adalah 2 keputusan desain terpisah yg BISA dipisah — Discussion 735154 menggabungkan keduanya, tapi implementasi ke arsitektur KITA (unfreeze parsial, backbone ImageNet-pretrained) ternyata sensitif ke keputusan kedua. Kandidat lanjutan kalau mau coba lagi: (a) unfreeze patch embedding juga (risiko OOM/instabilitas lebih tinggi), atau (b) pakai jendela-sampling 3-anchor tapi SEBAGAI 9 view independen grayscale-repeat (spt skema v15), bukan RGB-packing — isolasi manfaat "jendela tengah + tetangga fisik" tanpa risiko channel-mismatch. **Model terbaik proyek TETAP v15 (0,8304)**, tidak tergantikan oleh eksperimen ini.

## 20. v17: "3+3+3" sbg 9 view independen — MASIH negatif, line riset ditutup (19 Agustus 2026)

Kandidat (b) dari §19 dieksekusi: `kaggle_kernel_modeling_v3indep/` — cache v3 (9 slice = 3 anchor jendela-tengah x 3 tetangga fisik) dipakai sbg 9 view independen grayscale-repeat (`K_SLICE=9`, pola identik dgn v15/K_SLICE, `.repeat(1,3,1,1)` TIDAK dihapus). Run ~1 jam 43 menit (K_SLICE=9 = ~2,25x lebih banyak forward-pass gambar drpd v15/K_SLICE=4).

**Hasil: val_macro_auc 0,7999 (epoch 8/8, masih naik)** — lebih baik dari v16 (0,7737, RGB-packing) TAPI TETAP di bawah v15 (0,8304). `train_loss` epoch 8 = 0,4493 (v17) vs 0,2985 (v15) — konvergensi jauh lebih lambat, sinyal training kurang efektif.

**Kesimpulan diperjelas lewat perbandingan v16 vs v17**: v17 > v16 mengonfirmasi channel-packing MEMANG sebagian penyebab kegagalan v16 (§18/19), TAPI v17 < v15 menunjukkan ada masalah kedua yg independen: **pembatasan sampling ke jendela TENGAH stack (20-80%) kemungkinan memotong informasi anatomi di ~40% ujung stack** yg justru terjangkau sampling penuh v15 (4 titik linspace tersebar 0-100%). Ditambah, 3 slice per anchor v17 berdekatan sangat rapat secara fisik (nyaris duplikat) — rata-rata dari 9 slice yg efektifnya cuma ~3 sudut pandang benar-benar beda (bukan 9), lebih sempit drpd 4 sudut pandang v15 yg tersebar jauh satu sama lain di seluruh rentang stack.

**Keputusan: line riset "3+3+3" (Discussion 735154) DITUTUP utk proyek ini — 2 dari 2 implementasi (RGB-packing & independent-view) GAGAL mengalahkan v15.** Bukan berarti tekniknya salah secara umum, tapi utk arsitektur/pipeline kita saat ini (backbone unfreeze parsial, label training saat ini) tidak terbukti jadi pengungkit tambahan di atas sekadar urutan-fisik (v15) + sampling-sebar-penuh. Kandidat kalau mau dibuka lagi nanti: skema sampling yg TETAP sebar penuh 0-100% (bukan jendela-tengah) tapi menambah smoothing/interpolasi antar-slice fisik-berdekatan (beda dari packing-jadi-RGB maupun window-restriction yg sudah terbukti gagal keduanya). **Model final proyek TETAP v15 (0,8304)** — TIDAK tergantikan oleh v16 maupun v17. **(Update pasca-§21: magnitude perbedaan v15 vs v16/v17 ternyata berada dekat noise-floor k-fold yg baru terkuantifikasi — diagnosis channel-mismatch v16 tetap valid, tapi lihat §21 sebelum terlalu yakin soal besaran bedanya.)**

## 21. v18: k-fold CV ungkap validasi 1-split selama ini cukup berisik (20 Agustus 2026)

Arahan user pasca-§20 (line "3+3+3" ditutup): "k-fold CV proper". Motivasi: validasi selama ini cuma 1 split tetap (semua gold->val N=57, semua weak->train) — cukup rawan noise utk perbandingan antar-eksperimen yg bedanya tipis (0,01-0,06), spt yg berulang kali terjadi di v15/v16/v17.

**Implementasi** (`kaggle_kernel_modeling_kfold/`, TPU, cache preprocessing v2 spt v15): 57 studi gold dibagi 5 fold. Per fold, (K-1)/K studi gold (~45-46) DITAMBAHKAN ke training (ekstra sinyal label asli, bukan cuma dibuang ke validasi spt skema lama) + 1/K jadi validasi OOF. Tiap fold dilatih dari NOL, independen (bukan saling resume). Arsitektur/hyperparameter identik v15 (unfreeze 6, LR terpisah, `CosineAnnealingLR`, EPOCHS=8).

**Hasil (run ~4,5 jam)**:

| fold | best val_macro_auc | n_val |
|---|---:|---:|
| 1 | 0,7915 | 12 |
| 2 | 0,8400 | 11 |
| 3 | 0,8670 | 11 |
| 4 | 0,7949 | 11 |
| 5 | 0,8698 | 12 |

Rata-rata per-fold: **0,8326 ± 0,0338** (rentang 0,7915–0,8698, **selisih 0,078 antar-fold**). **OOF macro AUC gabungan (57/57 studi gold, metrik paling terpercaya krn tiap studi dinilai model yg tak pernah melihatnya): 0,8068** — LEBIH RENDAH dari v15 single-split (0,8304).

**Temuan metodologis besar**: selisih antar-fold (0,078) jauh melampaui selisih² yg selama ini dikejar antar-eksperimen (v15 vs v11: +0,0109; v16: -0,0567; v17: -0,0305 — semuanya berada DALAM rentang noise antar-fold yg baru terukur). Ini artinya **validasi 1-split N=57 yg dipakai sejak v6 CUKUP BERISIK utk membedakan perubahan sekecil itu dgn percaya diri**. v15 kemungkinan besar mendapat split yg kebetulan menguntungkan model itu, bukan murni krn model itu objektif terbaik.

**Bukan berarti semua kesimpulan sebelumnya salah** — diagnosis v16 (channel-mismatch krn patch embedding beku) tetap berdasar alasan arsitektur yg independen dari noise split, dan geometri urutan-fisik (v2, dari v11=0,8195 GPU single-split ke v15=0,8304 TPU single-split) tetap perubahan besar (+0,011, sementara typical fold-to-fold noise ±0,034 satu arah bisa lebih besar dari itu — jadi bahkan lompatan v11->v15 pun perlu dipandang dgn kehati-hatian serupa, walau arah dan mekanismenya masuk akal secara independen: preprocessing v2 tervalidasi 100% pos_source=ipp di skala penuh, bukan artefak split).

**Implikasi ke depan**: (1) **OOF/k-fold sekarang metrik yg lebih dipercaya** drpd single-split N=57 — dipakai utk eksperimen besar berikutnya kalau memungkinkan (walau ~4-5x lebih mahal waktu/kuota drpd 1 split); (2) **5 checkpoint fold tersimpan** (`best_model_fold0-4.pt`) — kandidat ensemble 5-model yg murah (tidak perlu training baru, cuma rata-rata prediksi saat inferensi), berpotensi lebih stabil dari model tunggal manapun krn menggabungkan 5 model yg dilatih di subset training berbeda; (3) target "0,85-0,88 sebelum masuk ensemble komunitas" (roadmap §8c) sekarang diukur relatif ke OOF 0,8068 (bukan 0,8304) — gap ke ambang itu lebih besar dari yg dikira sebelumnya.

## 22. Submission v2 tervalidasi + BLEND OOF kita+pilkwang = 0,8462 tanpa training tambahan (20 Agustus 2026)

**Submission v2** (`kaggle_kernel_submission_v2/`): pipeline ensemble 5-fold model KITA SENDIRI (bukan reuse kode komunitas) — decode DICOM `test_series` mentah on-the-fly (tidak ada cache preprocessing utk test), pakai fungsi preprocessing v2 PERSIS SAMA (`get_sorted_files`, `preprocess_slice`, `group_norm_stats.csv` dipinjam dari preprocessing v2). Dev-test CPU di stub 3 studi: **SUKSES PENUH, 0 error**, format `submission.csv` benar. **Temuan: CPU ~58 detik/studi → ~21 jam utk 1.300 studi test sungguhan, jauh lewat batas 9 jam** — konfirmasi GPU/TPU wajib utk submission real.

**Blend OOF kita + pilkwang** (arahan "lanjut" -> "sesuai rekomendasi"): dgn GPU masih terkunci, langkah bernilai tanpa kuota — blend `oof_predictions.csv` kita (v18, 5-fold) + `oof.npz` resmi pilkwang (diunduh ulang), murni numpy/pandas lokal.

| sumber | macro AUC (57 gold, leak-free) |
|---|---:|
| Kita sendiri (v18 OOF) | 0,8068 |
| Pilkwang (20-model OOF) | 0,8360 |
| **Blend rank-space α=0,60 (60% pilkwang, 40% kita)** | **0,8462** |

Sweep α 0,00-1,00: blend mengalahkan KEDUA sumber di rentang luas (α=0,30-0,90 semua >0,83) — bukan artefak titik tunggal, konfirmasi keragaman model kita bernilai nyata walau lebih lemah sendirian. **Skor macro AUC leak-free tertinggi proyek ini.**

Per-label: Effusion 0,869→0,940→**0,962**, MCL 0,801→0,875→**0,887** naik konsisten via blend. Lateral Meniscus TETAP lemah di ketiganya (~0,63-0,64) — bukan soal blend, kandidat perbaikan representasi/label tersendiri.

Validitas dicek: `true_` (kita, hard gold) vs `y_derived` (pilkwang, kemungkinan soft/derived) TIDAK identik (selisih 143,0) — bukan bug, evaluasi tetap konsisten krn SEMUA AUC (kita/pilkwang/blend) dihitung relatif ke `true_` yg sama; recompute pilkwang-alone via jalur ini cocok persis dgn 0,8360 yg sudah diverifikasi independen (§13).

**Langkah nyata berikutnya (butuh GPU T4, terkunci sampai ~22 Agustus)**: jalankan submission v2 (kita) + submission v1 (pilkwang) di `test_series` sungguhan, blend rank-space α≈0,60, submit hasil blend — bukan salah satu sendirian. Ini jalur konkret pertama menuju skor gabungan yg benar-benar terukur di leaderboard, dibangun dari 2 sumber yg SUDAH tervalidasi leak-free.

## 23. Riset ulang Code/Discussion/eksternal (20 Agustus 2026, arahan "scan code+diskusi+model, cari referensi utk naikkan skor")

Leaderboard bergerak lagi (#1 Brandon Low naik ke **0,952**). Topik baru minor (2 thread non-teknis), tapi thread lama "bottleneck" (735826) dapat balasan baru yg PENTING: kompetitor lain independen konfirmasi ulang **label gold sendiri kontradiktif** (2 report memakai template & kalimat effusion yg SAMA persis, tapi diberi label 0 dan 1 berbeda) — memperkuat lagi temuan noise label yg sudah kita ukur sendiri via k-fold (§21).

### 23a. TEMUAN UTAMA: `sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee` — tulisan teknis terbaik yg pernah ditemukan di kompetisi ini

Public LB **0,883**, single-family 5-fold DINOv2-B (bukan ensemble multi-sumber), TANPA prediksi eksternal dicampur. Menulis metodologi & alasan tiap keputusan secara eksplisit (langka utk kernel publik). Temuan kunci yg actionable:

1. **Konfirmasi independen ke-3 (setelah 735826 & temuan kita sendiri v11→v15)**: fine-tuning di resolusi SAMA (224px) menggerakkan label FOKAL (Medial Meniscus +0,171, MCL +0,118, ACL +0,113) jauh lebih besar drpd naikkan resolusi (+0,017 doang). "Domain adaptation does most of the work."

2. **Kalibrasi lokal→LB: klaim offset +0,044, TAPI ketemu inkonsistensi internal di sumbernya — dilaporkan apa adanya, bukan ditelan mentah**. Kutipan asli: *"Both scored systems landed 0.044 above their gold-58 macro AUC (0.771 → 0.776; 0.824 → 0.866). A published external system... reports 0.857 gold-58 against 0.903 LB."* Dicek manual: Stage A (0,824→0,866) = **+0,042** ✓ dekat klaim; sistem eksternal (0,857→0,903) = **+0,046** ✓ dekat klaim; TAPI baris frozen-baseline (0,771→0,776) cuma **+0,005** ✗ SAMA SEKALI TIDAK cocok dgn "+0,044" — kemungkinan besar salah tulis/salah pasang angka di sumber aslinya, bukan kesalahan baca kita (sudah di-cek ulang 2x langsung dari teks mentah notebook). **Kesimpulan kita**: offset **~+0,04-0,046 didukung cukup baik utk model YANG SUDAH KUAT** (gold-58 di kisaran 0,82-0,86, 2 dari 3 contoh cocok), tapi JANGAN dianggap konstanta presis krn 1 dari 3 contoh sumbernya sendiri tidak konsisten. Diterapkan hati-hati ke skor kita (asumsi offset ~+0,04, BUKAN presisi): v18 OOF gold (0,8068) → estimasi LB kasar ~0,84-0,85; blend kita+pilkwang (0,8462, §22) → estimasi LB kasar **~0,88-0,89** — di bawah skor tim saat ini (0,920) tapi tidak jauh, dgn margin ketidakpastian yg jujur diakui.

3. **TEKNIK BARU BELUM KITA COBA — laterality canonicalization (bukan cuma crop, tapi transformasi geometris penuh)**: `Laterality`/`ImageLaterality` kosong di ~72% studi mereka (mirip kita, ~50,7% kosong). Solusi: hitung **CENTER** citra (bukan cuma corner dari `ImagePositionPatient`) di ruang pasien via rumus `centre = IPP + row_cosine·col_spacing·(Columns/2) + col_cosine·row_spacing·(Rows/2)`, pakai TANDA koordinat-x sbg penentu kiri/kanan. Lalu **SEMUA studi dikanonikalisasi seolah lutut kanan**: flip horizontal utk coronal/axial dari lutut kiri, balik urutan slice sagittal per lateralitas (slice 0 = paling medial SELALU). **4 dari 12 label (Medial/Lateral Meniscus, Medial/Lateral OA) rusak diam-diam kalau ini tidak dikerjakan** — "the model sees mirrored anatomy half the time... nothing crashes, the loss still goes down, and you lose a third of the score without a symptom." Ini MEMPERKUAT PRIORITAS item "crop sadar-laterality" yg selama ini kita taruh rendah — ternyata berpotensi dampak besar, bukan cuma nice-to-have.

4. **Arsitektur pooling lebih canggih drpd skema kita**: attention pooling per-slice (bukan mean sederhana spt kita) dgn 2 tambahan per slice-embedding: **posisi slice ternormalisasi** (0=medial, 1=lateral, cuma valid SETELAH kanonikalisasi di atas) + **embedding plane** (sagittal/coronal/axial) supaya 1 modul pooling melayani ketiga plane. Agregasi series→studi: rata-rata DALAM plane dulu baru concat 3 plane (bukan weighted merata semua series) — atasi ketidakseimbangan jumlah series per plane (mirip 6-slot kita tapi beda mekanisme).

5. **CLS+patch-mean concat**: PERSIS sama dgn `KneeModel.forward` kita — validasi independen bahwa pilihan kita sudah tepat.

6. **Fracture = masalah LABEL, bukan vision**: 97% studi mereka share 1 nilai target (prior "report tidak sebut fraktur") → hampir tanpa varians → hampir tanpa gradien. Tidak ada model yg bisa perbaiki tanpa perbaikan label. Relevan kalau kita mau prioritaskan label sisa.

7. **Diversity tetap bernilai walau model SANGAT mirip**: varian 224px vs 336px dari arsitektur SAMA berkorelasi 0,901 (tinggi!) tapi ensemble tetap +0,006 — validasi independen lagi bahwa temuan blend kita (§22, 0,8462 dari 2 sumber leak-free) bukan kebetulan.

8. **KOREKSI POTENSIAL ke submission v2 kita**: mereka eksplisit menyarankan **rank-mean, BUKAN probability-mean**, utk kombinasi model — "AUC reads order and averaging sigmoids lets the most confident member dominate." **Submission v2 kita saat ini pakai `np.mean(preds)` (probability-mean) utk 5-fold ensemble** — pertimbangkan ganti ke rank-mean sblm submit sungguhan.

9. **Pelajaran metodologi ttg OOF vs gold-58 (BUKAN kontradiksi temuan k-fold kita, tapi klarifikasi penting)**: OOF mereka (n=4407, vs weak/pseudo-label) SISTEMATIS meremehkan peningkatan vision asli — "when the model gets better at seeing the knee, it departs from the labels precisely on the studies where the report was wrong". Protokol yg disarankan: **OOF (vs weak label) pilih epoch & deteksi kerusakan run; gold-58 (vs label asli) putuskan apakah SATU ARAH layak dikejar** — BUKAN dibalik. **Catatan penting utk kita**: k-fold v18 kita BEDA kasus — OOF kita divalidasi vs GOLD ASLI (bukan weak label), jadi tidak kena bias sistematis yg sama; val_macro_auc dlm training loop kita jg sudah pakai gold (bukan weak-label OOF) utk checkpoint selection — praktik kita sudah sesuai protokol yg disarankan di sini, bukan perlu diubah.

## 24. `mattiaangeli/bend-the-knee-to-dinov3-ensembled` — bedah kode lengkap, 3 teknik arsitektur baru bernilai tinggi (21 Agustus 2026, arahan "cek progres model+riset, kalau kurang scraping artikel")

Ditemukan lewat `kaggle competitions submissions` — tim (kemungkinan kolaborator) sudah submit notebook ini (55650895, PENDING saat dicatat). Author = **Mattia Angeli**, sama dgn "V104" yg sudah dicatat di §23/MEMORY sblmnya, tapi ini versi kode LENGKAP (bukan cuma klaim dari Discussion) — di-pull penuh (`kaggle kernels pull`) dan dibedah cell-per-cell. 27 cell, ensemble besar (20×DINOv2 pilkwang + 1×DINOv3 ViT-S/16 + 2 stage RadImageNet E11/E13 + label triangulasi 3-sumber).

### 24a. Peringatan eksplisit dari author sendiri (cell 0) — VALIDASI KUAT thesis "commodity ceiling" kita

> "This notebook is now likely overfit to the public leaderboard: after users repeatedly forked and republished it with one or two parameter tweaks, I also followed movements of only 0.001–0.003 and changed the notebook in response. That user-driven fork-and-republish race is not meaningful model development."

Author sendiri menyarankan JANGAN fork versi terbaru, tapi pakai versi lebih lama sbg baseline, atau ambil **member DINOv3 mentah dari dataset `knee-mri-fold-weights` dan validasi lokal sendiri**. Ini konfirmasi independen dari org dalam komunitas kompetisi ttg apa yg sudah kita simpulkan sendiri di §2 MEMORY (skor 0,920 tim = commodity ceiling, bukan pencapaian unik) — memperkuat strategi kita: bangun model sendiri (v21 dst), JANGAN ikut race fork-tweak 0,001-0,003.

### 24b. Teknik arsitektur baru #1 — `DepthCompress`: solusi TERUKUR utk masalah yg menjatuhkan v16 kita

v16 kita gagal (0,7737) krn RGB-packing literal 3 slice fisik ke channel R/G/B — patch embedding DINOv2 yg beku tidak bisa menafsirkan korelasi channel non-foto-natural (§2 MEMORY, "pola berulang"). Notebook ini memecahkan masalah PERSIS ini dgn cara berbeda & lebih prinsipil: `DepthCompress` module — ambil **N_SLICE=16 slice mentah sbg channel input**, lalu learnable **gated 1×1 conv block** (`_GatedDepthBlock`: `GroupNorm → conv "value" × silu(conv "gate") → conv "out"`, residual + **layer-scale** `gamma` diinisialisasi KECIL 0,1 shg training awal dekat identity/stabil) meng-kompres 16 channel → 3 channel "pseudo-RGB" yg SUDAH dilatih utk pola knee, baru dinormalisasi ImageNet & masuk backbone. Beda kunci vs v16 kita: **bukan berharap patch embedding belajar interpretasi RGB non-natural, tapi ada modul terpisah yg tugasnya KHUSUS belajar kompresi itu sebelum masuk backbone** — secara arsitektural lebih masuk akal drpd andalan unfreeze patch_embeddings (pendekatan v21 kita) ATAU RGB-packing literal (pendekatan v16 yg gagal). **Kandidat kuat utk v22** kalau v21 tidak cukup naik.

### 24c. Teknik arsitektur baru #2 — `ViTSlotToken`: injeksi info slot KE DALAM backbone, bukan cuma di head

Trik yg belum pernah kita coba: sisipkan token kategori (slot: plane × fat-suppression, 6 kategori + mask) LANGSUNG ke prefix token ViT (`num_prefix_tokens += 1`, semua block attention diberitahu prefix baru) SEBELUM masuk ke transformer blocks — jadi self-attention DI DALAM backbone sendiri bisa memakai info slot itu, bukan cuma ditambahkan setelah backbone di level pooling (spt `plane` embedding kita di `AnatomyQueryHead` v21). Ini strictly lebih ekspresif drpd pendekatan kita saat ini. Kandidat upgrade v22/v23, tapi butuh akses langsung ke internal `vit.blocks`/`vit.patch_embed` (kode contoh pakai `timm` ViT, bukan `transformers.AutoModel` spt kita — perlu adaptasi).

### 24d. Teknik arsitektur baru #3 — `TokenXAttnPool`: cross-attention atas SEMUA series (bukan 1-pilih-per-plane spt v21)

Beda dari v21 kita (pilih 1 series terbaik per plane via `pick_score`, buang sisanya): notebook ini **concat token dari SEMUA series dlm 1 studi** jadi satu key/value sequence (var-length, di-pad + `key_padding_mask` biar tidak ada series yg diimputasi/dibuang), lalu 12 query per-label cross-attend via `nn.MultiheadAttention` sungguhan (bukan einsum manual spt kita). Output per label = concat(hasil attention, mean CLS semua series, max CLS semua series) sebelum classifier — pooling attention DAN statistik agregat digabung, bukan cuma salah satu. **Implikasi utk v21 kita**: strategi "1 series terbaik per plane" mungkin membuang informasi dari series lain yg tetap relevan (mis. 2 sequence sagittal beda fat-suppression) — kalau v21 tidak menang meyakinkan, salah satu kandidat penyebab adalah pemilihan 1-per-plane ini, bukan cuma arsitektur head-nya.

### 24e. Dataset baru dgn nilai langsung — checkpoint SIAP PAKAI, bukan cuma ide

- **`mattiaangeli/knee-mri-fold-weights`** (dikonfirmasi CC0-1.0, akses publik) — **5 checkpoint fold DINOv3 ViT-S/16 SUDAH DILATIH** (`m_f0.pt`...`m_f4.pt`, ~94MB tiap, total ~470MB), SIAP diinferensi tanpa training tambahan sama sekali. Ini beda dari `mattiaangeli/rsna-knee-mri-fold-weights` yg tercatat sblmnya di MEMORY (slug SALAH/tidak ketemu — dikoreksi jadi `knee-mri-fold-weights`, tanpa prefix "rsna-"). **Butuh pipeline preprocessing SESUAI recipe mereka** utk hasil valid: `SIZE=336` (bukan 224 kita), `CROP_MM=130.0` (crop fisik dlm mm, bukan cuma resize), `N_SLICE=16`, 6 SLOTS (plane × fat-suppression). **Nilai potensial**: anggota ensemble independen baru TANPA training cost — tapi effort integrasi non-trivial (beda resolusi/crop/arsitektur dari kita), realistiknya dikerjakan SETELAH v21 selesai & kuota GPU/TPU longgar, bukan sekarang.
- Dataset sumber lain yg baru kekonfirmasi dipakai kernel ini (semua CC0/publik, belum kita pakai): `mattiaangeli/rsna-knee-rad-calibration`, `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`, `pilkwang/pilkwang-public-dataset-for-notebooks-figures`.
- **3-sumber label triangulation** dikonfirmasi dipraktikkan kompetitor top: `pilkwang` + `stevenleehans` + `lixin73` (label baru, belum tercatat di MEMORY) — "so one reading could be checked against another". Menguatkan prioritas jangka panjang kita ttg label-strengthening (§2 MEMORY, konsensus "bottleneck = kualitas label").

### 24f. Kesimpulan & rekomendasi prioritas

Tidak ada "satu trik ajaib" baru di sini — ini ensemble besar hasil kolaborasi berbulan² banyak org, PERSIS jenis "commodity" yg strategi kita sengaja hindari (§2 MEMORY). **Yang bernilai bukan buat di-fork, tapi 3 potongan teknik konkret** (DepthCompress §24b, ViTSlotToken §24c, TokenXAttnPool-atas-semua-series §24d) yg BISA diadaptasi ke arsitektur v21 kita SENDIRI kalau hasil v21 mentahnya tidak cukup kuat. Rekomendasi urutan: (1) tunggu OOF v21 keluar dulu — kalau v21 sudah menang jelas vs v18/v20, PRIORITASKAN submission real dulu drpd re-arsitektur lagi; (2) kalau v21 masih flat/negatif, `TokenXAttnPool` (pakai semua series, bukan 1-pilih-per-plane) adalah kandidat perbaikan PALING MURAH utk dicoba duluan (ubah logic pemilihan series, bukan bikin modul baru); (3) `DepthCompress` adalah upgrade lebih dalam (ganti cara triplet dibentuk) — worth dicoba kalau (2) belum cukup; (4) `knee-mri-fold-weights` (checkpoint DINOv3 siap pakai) adalah "cheap win" ensemble tapi butuh effort integrasi non-trivial, taruh di antrian setelah submission real pertama selesai.

**Implikasi prioritas**: (a) **laterality canonicalization NAIK prioritas signifikan** (bukti dampak besar, bukan dugaan); (b) offset kalibrasi +0,044 jadi alat estimasi LB tanpa buang submission; (c) submission v2 kita perlu dicek ulang: ganti probability-mean → rank-mean sblm submit sungguhan.

## 25. `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` — TEMUAN TERBAIK PROYEK: single model 0,924 public LB, lebih tinggi dari fork ensemble tim (0,922), arsitektur berbeda total dari kita (23 Agustus 2026, arahan user "ada notebook yang sudah mencapai 0,924, pelajari dengan mendalam")

Ditemukan via `kaggle kernels list --competition ... --sort-by scoreDescending` (bukan cuma cari manual di Code tab — cara ini langsung nemu notebook TERTINGGI skornya, bukan yg terpopuler/paling banyak vote). Ranking #1 di sort skor kompetisi ini. Kode di-pull penuh (459 baris, `.py` script bukan `.ipynb`) & dibaca lengkap — docstring pembuka sangat rinci & jujur soal metodologi (langka, kualitas tulisan setara `sadamtorres` §23a). **Klaim di docstring: "It scores 0.924 on the public leaderboard using one model, with no ensembling and no test-time augmentation."** Dataset bobotnya (`dreaddevelopment/raptor-knee-widedense`) **berlisensi CC0-1.0 eksplisit** — bebas dipakai, tanpa syarat atribusi.

**INI LEBIH TINGGI dari skor tim kita saat ini (0,922, fork ensemble commodity `bend-the-knee-to-dinov3-ensembled`)** — dan dicapai dgn SATU model, bukan ensemble 20+ — PERSIS strategi yg sudah kita putuskan dari awal (§2 MEMORY: "single model kuat > ensemble komunitas 20+ model") tapi belum pernah kita capai sendiri seefektif ini.

### 25a. Angka validasi lokal — data poin kalibrasi baru yg penting

Author melaporkan **0,9167 macro-AUC di 58 studi gold** (held-out, TIDAK PERNAH dilatih) → **0,924 di public LB**. Offset **cuma +0,0073** — JAUH lebih kecil dari klaim sadamtorres (+0,04-0,046, §23a). Kemungkinan penjelasan: model ini SANGAT KUAT & well-calibrated (gold-58 sendiri sudah 0,9167), jadi makin sedikit "headroom" utk pola kalibrasi sistematis yg sama munculnya. **Pelajaran utk kita**: offset kalibrasi TIDAK konstan — makin kuat modelnya, makin kecil kemungkinan offsetnya (bukan alasan utk berhenti menaikkan model lokal kita, malah sebaliknya: base gold-58 rendah kita, seperti v18 0,8068, mungkin under-representasi seberapa jauh kita dari LB drpd yg dikira).

### 25b. Arsitektur: CoAtNet (bukan DINOv2!) + attention pooling per-label — beda backbone total dari kita

- **Backbone: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`** via `timm`, bukan `transformers.AutoModel` DINOv2. CoAtNet = hybrid conv-attention (bukan ViT murni) — arsitektur BEDA KELUARGA dari yg kita pakai sepanjang proyek ini. Komentar kode mengungkap mereka SUDAH BANDINGKAN 7 arm/backbone secara sistematis (greedy forward selection + exhaustive subset search di 45-studi gold): **CoAtNet 384 = 0,9025 (single terbaik)**, SwinBase384 = 0,8825, EfficientNetV2L480 = 0,8716, lalu drop CNN336/ConvNeXtBase/ConvNeXtLarge/MaxViT384 sbg redundan. Blend 3-arm cuma +0,0043 di atas single terbaik (0,9068 vs 0,9025) — DAN di LB SUNGGUHAN blend cuma +0,001 (0,914-0,915) sementara single CoAtNet 0,914 — **konfirmasi independen ke-2 (setelah temuan blend kita sendiri) bahwa gain ensemble di gold-set kecil sering noise, bukan sinyal asli**. Keputusan akhir mereka: **1 arm CoAtNet SAJA**, bukan blend — 1/3 runtime kernel, skor GPU-jam-efisien tapi TIDAK korbankan akurasi.
- **`RaptorClassifier` head** (kode lengkap didapat): `LayerNorm` → attention MLP (`Linear(F,256)→Tanh→Dropout→Linear(256,12)`) → softmax atas K windows PER finding (attention terpisah per 12 label, mirip filosofi `SlotHead`/`AnatomyQueryHead` kita) → weighted-pool fitur via einsum → classifier linear per-label (`clsW`/`clsb`). Validasi independen ke-3 (setelah kita sendiri & sadamtorres) bahwa **attention pooling per-label, bukan pooling tunggal dibagi rata ke 12 label, adalah pola yg tervalidasi berulang kali di kompetisi ini**.
- **Full backbone fine-tuning (BUKAN cuma unfreeze last-N-block spt kita)**: komentar sebut "retrained on the EXPANDED 4.349-study corpus" — tidak eksplisit rincian freeze-schedule, tapi nama file `raptor_ft_coatnet_v4_full.pt` ("_full") & konteks menyiratkan fine-tune menyeluruh, bukan skema unfreeze-parsial kita.

### 25c. Teknik data — 3 hal LANGSUNG relevan ke titik lemah kita sendiri

1. **RGB-triplet 3-slice-berdekatan SEBAGAI CHANNEL — SAMA PERSIS teknik v16 kita yg GAGAL (0,7737)**. Bedanya: notebook ini BERHASIL (arm tunggalnya 0,9025 lokal). Konfirmasi LANGSUNG diagnosis kita sendiri (§2 MEMORY, "pola berulang" v16) — RGB-packing BUKAN ide yg salah, masalahnya khusus krn PATCH EMBEDDING KITA BEKU. Di sini backbone di-fine-tune PENUH ("_full"), jadi triplet RGB genuine BISA dipelajari. **Ini bukti tambahan (independen dari mattiaangeli `DepthCompress`, §24b) bahwa akar masalah v16 memang freeze-schedule, bukan RGB-packing itu sendiri** — 2 solusi berbeda (fine-tune penuh di sini, gated-conv compress di mattiaangeli) sama2 menyelesaikannya dgn cara yg BUKAN unfreeze-parsial kita/v21.
2. **Span sampling LEBAR (6%-94% dari stack), BUKAN window tengah**: "Within a series, slices are taken evenly across 6 to 94 percent of the stack rather than from the middle. The outer slices are where the collateral ligaments and the lateral meniscus sit, and cutting them was measurably costing accuracy on exactly those findings." — **INI LANGSUNG relevan ke titik lemah kita yg belum terpecahkan**: Lateral Meniscus lemah di SEMUA sumber kita (~0,63-0,64, MEMORY §2 item terbuka). Perlu audit: v15-v20 kita pakai sampling seperti apa persisnya (4-titik linspace di seluruh stack — perlu cek span-nya, mungkin sudah cukup lebar; tapi preprocessing v3 "3+3+3" v16/v17 SENGAJA pakai window tengah 20-80% yg SUDAH terbukti gagal 2x — sekarang ADA PENJELASAN MENGAPA: window tengah literally memotong anatomi (ligamen kolateral, lateral meniscus) yg justru ADA DI TEPI stack, bukan cuma "kehilangan diversity" spt diagnosis kita sebelumnya, lebih presisi dari itu).
3. **Crop fisik 140mm + resolusi 336px** — KONFIRMASI INDEPENDEN KE-2 (setelah mattiaangeli CROP_MM=130, SIZE=336, §24e) bahwa physical-mm-crop + resolusi >224px adalah pola yg terbukti berulang di top performer. **2 dari 2 notebook top yg kita bedah pakai teknik ini, 0 dari kita** — sinyal kuat utk diadopsi.

### 25d. Weak-label: SOFT target (probabilitas kontinu), bukan hard-label + confidence-weight

"A report that says a tear is suspected becomes a number near 0.8, not a 1" — target regresi-BCE kontinu langsung dari LLM, BUKAN hard 0/1 + bobot loss `2*|p-0,5|` spt kita. Secara matematis mirip tapi tidak identik: pendekatan kita men-downweight sampel low-confidence di loss (BCE tetap terhadap target biner keras); pendekatan mereka membiarkan TARGET ITU SENDIRI kontinu (BCE terhadap target lunak). Worth dicoba sbg eksperimen loss/label terpisah, tapi bukan prioritas #1 (dampak targetnya blm terukur terisolasi, disebutkan cuma sbg konteks metodologi bukan ablasi berdampingan).

### 25e. Engineering robustness (operasional, bukan akurasi) — worth ditiru ke submission kita

- fp16 inference dgn fallback otomatis ke fp32 per-studi kalau CUDA "no engine" error (T4 tidak support bf16 utk shape tertentu) — **studi TIDAK PERNAH gagal/di-drop dari submission**.
- Baseline fallback prediksi 0,5 per studi (SAMA pola dgn `write_benchmark_submission()` kita, §22/PROGRESS).
- **Load model SEQUENTIAL (bukan simultan)** kalau >1 arm — komentar eksplisit sebut versi simultan bikin OOM RAM sistem di FULL hidden test (~1.300 studi), padahal aman di dev-test kecil. **Pelajaran operasional penting utk `kaggle_kernel_submission_v2/` kita**: dev-test 3-studi kita TIDAK akan menangkap masalah RAM skala penuh ini — perlu diwaspadai kalau nanti nambah >1 model/arm di submission asli.

### 25f. Kesimpulan & rekomendasi

Ini bukan sekadar "referensi teknik" spt §24 — **ini checkpoint SIAP PAKAI (CC0), skor TERUKUR (0,924) LEBIH TINGGI dari submission tim kita saat ini (0,922)**, dari SATU model tanpa ensemble. Rekomendasi konkret (urutan, tunggu keputusan user krn menyentuh kuota submission tim):
1. **Opsi cepat/murah**: jalankan notebook ini apa adanya (atau fork) sbg submission tim BARU — kandidat langsung naikkan skor tim dari 0,922 ke ~0,924 TANPA bangun apa pun, cuma butuh 1 slot submission harian & koordinasi tim.
2. **Opsi gabung**: blend prediksi model raptor ini dgn OOF kita (v18) + pilkwang (pola sama spt blend §22, 0,8462) — berpotensi lebih tinggi lagi dari 0,924 kalau sumbernya cukup independen (beda backbone total: CoAtNet vs DINOv2 vs ensemble pilkwang — DIVERSITY tinggi, kandidat kuat utk blend).
3. **Opsi jangka panjang**: adopsi 3 teknik data (§25c: span sampling lebar, crop-mm+336px, RGB-triplet dgn fine-tune penuh) ke v22 kita sendiri — terutama span sampling lebar krn LANGSUNG menjelaskan & berpotensi memperbaiki titik lemah Lateral Meniscus yg belum terpecahkan.
4. **Backbone CoAtNet sbg kandidat baru**: worth dicoba sbg alternatif/tambahan DINOv2 kita, given bukti internal author (0,9025 CoAtNet > 0,8825 Swin > 0,8716 EffV2L) — tapi ini investasi rekayasa lebih besar (integrasi `timm`, bukan `transformers.AutoModel`; fine-tune penuh bukan unfreeze-parsial).

## 26. Scraping penuh Code-tab ≥0.936 + leaderboard terkini + strategi menuju top-1 (25 Agustus 2026, arahan user "pelajari notebook 0,936 model+dataset+diskusi sampai bisa mengalahkan top 1")

### 26a. Cara scraping
`kaggle kernels list --competition ... --sort-by scoreDescending` memberi URUTAN skor (tanpa nilai). Nilai skor dikonfirmasi dari markdown notebook + leaderboard resmi (`leaderboard/download`, format ZIP). Endpoint internal `/api/i/kernels.KernelsService/ListKernels` TIDAK memuat skor & filter kompetisinya diabaikan (body valid: `{"kernelFilterCriteria":{"listRequest":{...},"competitionName":...,"sortBy":...}}`) — jangan buang waktu lagi. `kaggle kernels output ... --file-pattern "log"` berguna utk ambil log kernel tanpa download semua output.

### 26b. Leaderboard publik per 25 Agustus 2026 (sumber: API resmi)
- **#1 Brandon Low (zxkyouma) = 0.952**; #2 MKhlystun 0.948; #3-4 0.947; #5-8 0.946; cutoff **#10=0.945, #15/#20=0.943, #30=0.941, #40=0.939, #50=0.937**.
- **Tim KITA (Krazy KIller) = 0.936, rank #91/2.372** — naik drastis dari 0.920 (#130, snapshot 18 Agu): kolaborator sudah submit sesuatu setara stack DINOsaur terbaru. Kita PERSIS di ceiling notebook publik.
- 137 tim ≥ 0.936; 36 tim ≥ 0.94. Artinya: 0.936 = skor "commodity" baru; semua fork notebook publik sama menumpuk di sini.

### 26c. Puncak Code-tab publik = keluarga DINOsaur/Raptor (Roman Tamrazov), proven **0.936**
- Urutan skor Code-tab (#1-#16): romantamrazov/rsna-knee-dinosaur-v4 (~0.935-0.936, V10 = scriptVersionId 344688577 proven 0.936 via port prvsiyan) > llccqq624 fusion > prvsiyan/head-and-shoulders-knees-and-toes (**port setia Apache-2.0 dari DINOsaur V10, public 0.936, mount distandarkan — INI BASIS REPLIKA TERBAIK**) > nk bend-the-knee fork > kunal repro fork > raptor v5/w065 forks > dreaddevelopment single CoAtNet 0.924.
- **Anatomi stack DINOsaur V10 @0.936** (hasil bedah 3 agen paralel atas dinosaur-v4, raptor v5/w065, nk, kunal, lixin-w10):
  1. **Stage A**: manifest DINOv2/DINOv3 ensemble "24-member" (20 pilkwang + 4 tonylica) + fingerprint-check tiap checkpoint + slot-prior attention + TTA sliding-window full-overlap + pooling per-target (max utk Fracture/Contusion/Menisci/Baker, top2 utk ACL/MCL) → promotion ke file "public-frontier ~0.899".
  2. **A5**: fold-family timm (`mattiaangeli/knee-mri-fold-weights`, ViTSlotToken+DepthCompress+TokenXAttnPool, 336px/crop130mm/16 slice×6 slot) blend rank @0.45.
  3. **RadImageNet chain**: ResNet50 GAP (marwanmath CC-BY-NC — CATATAN lisensi NC, kita skip sbg anggota aktif tapi stack publik pakai) E10/E13/E11 layouts + twin heads + adaptive α per-target (stability/diversity z-score) + **V18 calibrator linear 88-fitur** (payload zlib+base64 embedded) w=0.40 gated.
  4. **CoAtNet dual-checkpoint**: MaxSpan v5 SWA (span 0.02-0.98, K_EVAL=62, 336px/crop140mm, 64 slice 5-slot, RGB-triplet) primary + WideDense v4 complement di GPU#1, **per-diagnosis complement weights** dgn Pearson-correlation gating (>0.992→dilemahkan, <0.65→diperkuat).
  5. Final: percentile-RANK fusion semuanya (bukan prob-mean), per-target weight {MM .52, LM .54, Fracture .54 dst}, fail-safe backup/restore + benchmark-first + time-budget adaptive degradation ladder (budget 8-8.72h dari cap 9h).
- Klaim skor penting dalam kode: CoAtNet single = 0.924 LB / 0.9167 gold-58 / 0.9025 gold-45; arms-x SWA CoAtNet = 0.917 LB; "ensembling worth ~+0.001 on LB"; offset gold→LB ≈ +0.007-0.012. **Peringatan eksplisit author fork-race: LB bergerak cuma 0.001-0.003 per iterasi publik; hampir pasti overfit public.**
- Dataset pendukung baru yang dipakai stack ini: `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` (bundel semua asset), `dreaddevelopment/raptor-knee-maxspan` (checkpoint MaxSpan), `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` (twin heads), `sofiaanjenje/rsna-knee-e11-train`+`e13-train`, `antoinegg1/rsna-knee-e11-diverse-heads-v20`+`e9-radimagenet-heads-v15` (**deskripsi dataset E10 mengungkap metodologi nested-selection weight pada gold-58: nested macro 0.862-0.873 vs base 0.840** — pola evaluasi leak-free yg layak ditiru), `lixin73/rsna-knee-train-checkpoints` (DINOv2-Small SlotHead mirip arsitektur kita!, dipakai lixin fork @w=0.10).

### 26d. Sinyal diskusi/eksternal terbaru (agregat)
- Akteruzzaman Raihan (LB 0.937, ~#15 saat itu): ~90 eksperimen — backbone besar/TTA/members/pretraining corpora ≈ ZERO gain; progress datang dari ELIMINATION; noise LB desimal ke-3 sering tidak resolvable; "the gap to the top isn't another architecture — it's how you learn from a noisy teacher when the exam is graded by someone else".
- Andrej Velichkovski (post LinkedIn): resep label severity-aware — "small effusion" → target ~0.2 BUKAN 1.0 (borderline=graded negative per Overview); monotonic ordering + margin; hierarchical shrinkage ke prior; confidence floor utk zero-support; kalibrasi HARUS in-fold (58 gold ÷ 5 fold = 11/fold); bias labeler tidak uniform — menumpuk di middle ranks tempat AUC diputuskan.
- Discussion 735826 (poll bottleneck): konsensus B(geometri)+A(label) >> C(backbone); saran praktis: bandingkan extractor label vs gold-58, lalu image-model vs extractor — kalau image model kalah dr teachernya ada headroom modeling; kalau menang jauh, label yang menahan.
- LLM soft-labeling (lit.: ACL 2026 GEM): ensemble-of-LLM-annotations > single-annotator soft labels utk downstream CE/calibration — relevan utk relabeling multi-sample temperature.

### 26e. Hasil v22 (selesai sejak 23 Agu, ditarik 25 Agu): REGRESI — line DITUTUP
OOF macro AUC **0.6713** (N=46 gold, cuma 4/5 fold selesai — kena safety time-limit di ~7.1h TPU; runtime/studi naik drastis krn 336px+triplet). Per-fold avg 0.6878±0.0477. JAUH di bawah v18 (0.8068)/v20 (0.8073). Pola konsisten ke-3: **SlotHead mean-pooling tidak punya jalur utk manfaatkan triplet-RGB/patch-embed-unfreeze; kombinasi itu butuh head attention-pooling a la mattiaangeli/lixin73** (dan bahkan mereka butuh seluruh stack lain utk menang). Kesimpulan arsitektur proyek: STOP modifikasi data-side di SlotHead; kalau mau arm sendiri yang kompetitif, tiru arsitektur lixin73 (DINOv2-Small + SlotHead attention + 336px/crop130) yg terbukti layak jadi anggota w=0.10 di fork publik.

### 26f. STRATEGI MENUJU TOP-1 (gap: kita 0.936 #91 → Brandon Low 0.952, +0.016)
Prinsip: (1) jangan kejar desimal ke-3 public LB — 137 tim sudah numpuk di 0.936 dan public-fit akan kena shakeup; (2) dua pengungkit dengan bukti terkuat = LABEL ENGINEERING + FUSION NESTED-CV, keduanya TIDAK dimiliki notebook publik secara benar (publik pakai regex/LLM-label mentah + weight hasil public-fit); (3) pertahankan diversity utk private shakeup.
- **Fase A — kontrol & basis (hari 1-7)**: pegang KODE sumber stack 0.936 (fork prvsiyan toes/DINOsaur V10 dgn mount standar) supaya bisa dimodifikasi; simpan 2 profil submission (public-best & diverse-conservative). Tutup line v22/v21/v16 (arsitektur SlotHead mentok).
- **Fase B — label engineering (minggu 2-4)**: bangun labeler severity-aware: regex multilingual + LLM ensemble (allowed, ruling 733965) dgn HEURISTIK RESMI annotator dari Overview (moderate/large effusion, high-grade MCL tear, dsb.) sebagai prompt; output = soft target kontinu [0,1] dgn monotonicity, shrinkage ke prior, confidence floor; kalibrasi vs gold-58 IN-FOLD SAJA. Gate sukses: agreement/OOF label baru > flight0234 (0.8991). Kalau lolos gate → retrain arm sendiri (arsitektur lixin73-style) dgn label baru → tambah sbg anggota fusion.
- **Fase C — fusion nested-CV (berkelanjutan)**: optimasi weight per-target antar-arm (stack 0.936 + arm kita + pilkwang OOF) via nested grouped-CV di gold-58 + bootstrap (pola kontrak antoinegg1 E10), BUKAN grid di public LB. Ukur leave-one-arm-out.
- **Fase D — disiplin akhir (Okt)**: 2 final submissions profil beda; semua keputusan pakai gold-58 OOF + bootstrap CI; koordinasi kuota 5 submit/hari dgn nxtirfan.
- Realisme target: +0.002-0.004 dari fusion tuning (→0.938-0.940, top-50); +0.005-0.010 lagi kalau label engineering berhasil (→0.943-0.950, top-10); mengalahkan 0.952 butuh eksekusi sempurna + shakeup menguntungkan — bukan hal yg bisa dijamin, tapi jalur satu-satunya dgn expected value positif adalah label engineering, karena semua jalur lain sudah commodity.

## 27. Riset Discussion segar (26 Agustus, arahan "scraping seluruh strategi, beri strateginya") — 3 temuan actionable + 1 temuan negatif diuji langsung

Setelah c5-solo real-LB (0,927) membantah pivot 25 Agustus (§26), riset ulang fokus ke Discussion
terbaru (bukan Code-tab lagi — sudah dibedah habis, top score obfuscated/tidak bisa digali lebih
jauh) + 1 pengujian lokal langsung (tanpa GPU) dari klaim yg ditemukan.

### 27a. Root cause Synovitis (topic 737566, `starkhushi`) — konfirmasi independen + fix konkret

Synovitis (titik lemah kita, 0,79-0,815 di semua sumber) TERNYATA punya penjelasan tegas: **cuma
~16% dari 4.407 laporan menyebut istilah synovitis SAMA SEKALI** (dicek di 9 bahasa) — sisanya label
diinferensi, bukan dibaca. Label table TERBAIK pun cuma 0,79-0,893 setuju dgn radiolog di label ini
— model dilatih dari "guru" yg sendiri cuma ~79-89% benar. **Fix yg TERBUKTI membantu penulis**:
**encoder pretrained RadImageNet capai 0,78 di Synovitis vs 0,62-0,72 utk model ImageNet-pretrained**
— domain medical-pretraining spesifik membantu label yg sinyalnya halus/samar. Kita PUNYA akses ke
RadImageNet (MIT-licensed, sudah dicatat sbg opsi backbone alternatif sejak lama, TAPI belum pernah
dibangun sbg arm sendiri — cuma dipakai via komponen ensemble komunitas). Ide kedua dari penulis:
infer Synovitis dari co-mention Effusion (korelasi ~0,5 di studi ber-anotasi) saat istilah eksplisit
tidak ada — belum pernah dicoba siapa pun setahu penulisnya.

### 27b. Hipotesis pooling spasial utk temuan kecil/tipis (topic 737597) — LANGSUNG relevan ke kelemahan Lateral Meniscus kita

Post ablasi jujur & rinci (EfficientNet-B0, 0,7815 OOF/0,870 public) menemukan pola per-label yg
PERSIS sama dgn kita: temuan besar/diskret/posisi-tetap (Baker's 0,855, ACL 0,853, Effusion 0,843)
bagus; temuan tipis/kecil/berjenjang (Lateral Meniscus 0,740, PF OA 0,748, MCL 0,758) buruk.
**Hipotesis penulis (belum diuji, cuma diajukan)**: robekan meniscus ~3mm cuma ~0,01% dari frame
288px — **global average pooling (GAP)/mean-pooling MELUNAKKAN sinyal itu sampai hilang**. Kandidat
fix yg diajukan: ROI crop di sekitar joint line, ATAU **spatial attention pooling** (bukan GAP)
supaya model belajar KE MANA harus melihat, bukan rata-rata semua patch. **INI LANGSUNG relevan**:
arsitektur `SlotHead` kita (v11-v22 SEMUA varian) pakai `patch_mean = out[:,1:].mean(1)` — PERSIS
GAP yg dicurigai penulis sbg akar masalah. Kita BELUM PERNAH coba attention SPASIAL di level patch
(v21's `AnatomyQueryHead` attention-nya di level WINDOW/slice, bukan di level patch-dalam-gambar) —
ini kandidat arsitektur BARU & BEDA dari semua yg sudah dicoba (v11-v22), bukan pengulangan.

Post yg sama juga MENGONFIRMASI independen 2 temuan kita: (a) laterality mirror-effect diuji &
DITOLAK signifikan (selaras v20 kita); (b) deteksi lateralitas geometris dari center-image di ruang
pasien resolve 4274/4407 studi, cocok 98,4% dgn tag asli — HAMPIR PERSIS metodologi kita (97,4%).

### 27c. PERINGATAN METODOLOGIS (topic 737454, notebook `starkhushi/rsna-knee-which-report-labels-should-you-train-on`) — checkpoint selection di gold-58 bisa MENYESATKAN

Kutipan penting: *"In my own runs the model that ranked LAST on those 58 [gold] ranked FIRST on the
leaderboard."* Saran penulis: **gold-58 utk PELAPORAN, JANGAN utk SELEKSI CHECKPOINT** — pakai
holdout weak-label lebih besar (ratusan studi) utk seleksi, baru gold-58 utk laporan akhir. **Ini
selaras LANGSUNG dgn pengalaman kita sendiri hari ini**: c5 lokal (0,9206, "menang" di nested-CV
N=57) ternyata KALAH di real LB (0,927) dari ensemble tim (0,936) — persis pola yg diperingatkan.
**Pelajaran diserap**: jangan lagi bikin keputusan strategis BESAR (ganti submission tim, pilih
arsitektur final) murni dari perbandingan gold-57/58 — submission real adalah pemutus utk keputusan
besar, gold-58 cuma kompas AWAL.

**Temuan tambahan yg SAMA PENULIS ungkap & KITA VERIFIKASI LANGSUNG di data lokal kita** (bukan cuma
percaya klaimnya): **194 studi berbagi Report BYTE-IDENTICAL di 47 grup** (grup terbesar 37 studi) —
diverifikasi via `reference_metadata/train.csv` lokal (`pd.read_csv(...).Report.value_counts()`),
angka kita (194/47) dekat dgn klaim penulis (131/46, beda snapshot data mungkin). **1 dari 58 studi
gold kita TERNYATA berbagi report dgn studi lain (non-gold)** — risiko leakage kecil tapi nyata di
skema k-fold kita (v18-v22): studi weak-label kembarannya SELALU ada di training (WEAK_STUDIES tidak
di-fold), jadi 1 fold yg menahan studi gold itu utk validasi kemungkinan tetap "bocor" sedikit lewat
kembaran reportnya. **Belum diperbaiki** (dampaknya kecil, 1/58, tapi dicatat sbg utang teknis).

### 27d. Diuji LANGSUNG (lokal, tanpa GPU): blend tabel label — HASIL NEGATIF, flight0234 tetap terbaik

Post 737454 menyarankan blend stevenleehans+pilkwang lebih baik dari salah satu sendirian (utk
MEREKA, yg tidak punya flight0234). Diuji dgn data KITA (`blend_labels.py`, unduh 3 tabel via
`kaggle datasets download`, skor manual vs 58 gold, murni pandas lokal < 1 menit):

| Sumber | macro AUC vs gold-58 |
|---|---:|
| **flight0234 (dipakai sekarang)** | **0,8991** |
| mean(flight, pilkwang) | 0,8978 |
| mean(flight, steven) | 0,8952 |
| mean(flight, steven, pilkwang) | 0,8935 |
| stevenleehans v4 blend | 0,8927 |
| mean(steven, pilkwang) | 0,8930 |
| pilkwang v2 | 0,8700 |

**TIDAK ADA blend yg mengalahkan flight0234 sendirian** — beda dari kasus penulis krn flight0234
SUDAH lebih akurat dari steven/pilkwang individual, jadi mencampurnya dgn sumber lebih lemah
menariknya turun, bukan naik (kebalikan dari nested-CV OOF blend prediksi kita yg dulu BERHASIL,
§22 — di situ semua sumber sebanding kualitasnya, di sini tidak). **Kesimpulan: line ditutup,
flight0234 TETAP satu-satunya label source, TIDAK perlu blend tabel lagi.**

### 27e. Sintesis strategi (ranked, per urgensi/biaya)

1. **[Medium, GPU, belum dicoba]** Bangun arm RadImageNet-pretrained sendiri (kita punya lisensi
   MIT-nya) — target spesifik Synovitis & temuan halus lain, konfirmasi independen §27a.
2. **[Besar, riset+GPU, arsitektur baru genuin]** Ganti `patch_mean.mean(1)` (GAP) di `SlotHead`
   dgn spatial attention pooling atas grid patch token — kandidat langsung utk kelemahan Lateral
   Meniscus/PF OA/MCL, BELUM PERNAH dicoba (beda dari v21 yg attention-nya di level window bukan
   patch-dalam-gambar). Prioritas riset berikutnya kalau lanjut bangun model sendiri.
3. **[Kecil, gratis, utang teknis]** Perbaiki k-fold split supaya grup duplicate-report (194 studi/
   47 grup) tidak terpecah gold/weak lintas fold — kebersihan metodologi, dampak kecil tapi nyata.
4. **[Disiplin proses, gratis]** JANGAN lagi pilih arsitektur/model final murni dari gold-57/58 —
   submission real WAJIB jadi pemutus utk keputusan besar (pelajaran dari c5-solo hari ini + §27c).
5. **[Ditutup]** Blend tabel label — TIDAK membantu, jangan diulang (§27d).
6. **[Tetap prioritas tertinggi jangka panjang]** LLM API relabeling severity-aware — SEMUA sumber
   independen (kita, sadamtorres, starkhushi, k256.dev/PC Jimmmy) sepakat label ~0,89-0,90 adalah
   plafon single-model, arsitektur/pooling sebagus apa pun mentok di situ tanpa label lebih baik.

## 28. Riset ulang leaderboard + Code-tab (13 September, arahan "riset dengan mendalam gimana caranya mengalahkan peringkat pertama")

Sesi ini pertama kali kaggle CLI aktif lokal (lihat MEMORY.md §4) — riset dilakukan LANGSUNG via API
resmi (`kaggle competitions leaderboard -d`, `kaggle kernels list --sort-by scoreDescending`,
`kaggle kernels pull`), bukan scraping tidak langsung spt sebelumnya. Discussion tab TIDAK bisa
diakses (SPA, butuh JS; `claude-in-chrome` extension tidak terpasang di environment ini) — riset ini
murni dari leaderboard resmi + kode Code-tab publik, TIDAK ada input Discussion baru.

### 28a. Leaderboard bergeser signifikan dalam ~3 minggu (25 Agu → 13 Sep)

| | 25 Agustus | 13 September | Δ |
|---|---:|---:|---:|
| #1 | Brandon Low, 0,952 | **Scott Willis, 0,956** (125 submission) | +0,004 |
| #10 cutoff | 0,945 | 0,952 | +0,007 |
| #20 cutoff | 0,943 | 0,950 | +0,007 |
| #50 cutoff | 0,937 | 0,946 | +0,009 |
| #100 cutoff | ~0,936 | 0,943 | +0,007 |
| Jumlah tim | 2.372 | 3.633 | +1.261 |
| **Tim kita (Krazy KIller)** | **0,936, #91** | **0,941, #213** | **skor +0,005, rank turun** |

**Skor kita naik tapi rank turun** — bukti langsung "commodity ceiling terus naik" lebih cepat dari
kita: submission history tim (`kaggle competitions submissions`) menunjukkan SEMUA kenaikan 0,936→
0,941 berasal dari fork ulang notebook publik yang terus diupdate authornya (bukan arm riset sendiri
— arm sendiri, c5-solo 0,927 dan v18+v23 SlotHead 0,837, TETAP jauh di bawah fork commodity). Gap ke
#1 PRAKTIS TIDAK BERUBAH (+0,016 → +0,015) walau skor absolut kita naik — seluruh lapangan bergerak
bersama.

### 28b. Evolusi commodity 0,937→0,941 TERDOKUMENTASI SENDIRI oleh komunitas (langka, biasanya obfuscated)

Notebook `evgendvorkin/rsna-baseline` (125 votes, kini bagian dari fork chain tim kita) punya markdown
cell eksplisit "Evolution of Public Solutions: 0.937 → 0.94 → 0.941" — jarang sekali top-notebook
mendokumentasikan diff-nya sendiri sejelas ini:

- **0,937 — "Head and Shoulders, Knees and Toes"** (prvsiyan, = DINOsaur V10 port, sudah lama kita
  kenal): CoAtNet 4-arm (v5/v10/v5-reverse/v8), outer CoAtNet weight GLOBAL 0,60 (semua label sama).
- **0,937→0,94 (+0,003) — "Hybrid Raptor Lateral Meniscus R100"** (renta0426): SATU perubahan bedah —
  reproduksi persis parent 0,939, ganti outer-routing HANYA utk Lateral Meniscus jadi CoAtNet 100% /
  Transformer-ensemble 0% (semua 11 label lain tetap bit-for-bit 0,60/0,40). Insight: isolasi 1 label
  bermasalah, kasih bobot penuh ke arm yg TERBUKTI menang di label itu — BUKAN retraining apa pun.
- **0,94→0,941 (+0,001) — "Bend the Knee to the Dinosaurs (ALL PUBLIC)"** (mattiaangeli): tambah arm
  CoAtNet baru (checkpoint publik `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`) — 384px adjacent-
  slice triplet → CoAtNet RMLP-2 → **"finding-specific GATED SPATIAL head" + "protected global-
  average residual"** (persis hipotesis §27b kita: attention spasial utk temuan kecil, GAP dilindungi
  sbg fallback utk temuan difus) — dicampur ke arm Raptor via `private_alpha=0,40`.

**Kode blend final di-ekstrak langsung** (cell `_coat_substitute`/blend akhir, `bend-the-knee-to-the-
dinosaurs.ipynb`): default `_coatnet_weight = 0.60` semua label, di-override `{'ACL': 0.675,
'Medial Meniscus': 0.70, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.675, 'Fracture': 0.675}` — angka
ini HASIL PROBING PUBLIC LB manual penulis (bukan nested-CV), risiko overfit persis yg diperingatkan
§27c/§26f sendiri. **Nama file checkpoint arm CoAtNet mereka**: `raptor_ft_coatnet_v5_full_swa.pt`
(w=0,60 dominan), `_v10_full.pt` (w=0,10), `_v5_full_swa.pt`-reverse (w=0,10), `_v8_full_swa.pt`
(w=0,20) — **PERSIS checkpoint c5/v5_full_swa yg SUDAH kita harvest OOF-nya sendiri (0,9206-0,9213
gold-58, §22/GPU#1 harvest)**. Konfirmasi langsung: seluruh komunitas top memang berdiri di atas
dataset checkpoint publik yg SAMA (`dreaddevelopment/raptor-knee-maxspan`) yg sudah lama kita analisis.

### 28c. Celah nyata yang KITA punya, komunitas TIDAK: v3 CoAtNet fine-tune (0,9234) mengalahkan checkpoint stock yg dipakai SEMUA notebook di atas

Checkpoint `raptor_ft_coatnet_v5_full_swa.pt` yg dapat bobot dominan (0,60, bahkan sampai 1,00 utk
Lateral Meniscus) di SETIAP notebook top saat ini skornya 0,9206-0,9213 gold-58 — **v3 kita (fine-
tune di atas label blend raw-mean flight0234+Claude, lihat MEMORY.md §3/PROGRESS.md §47) = 0,9234,
LEBIH TINGGI dari checkpoint stock yg mereka semua pakai.** Tidak ada satu pun notebook publik yg
scrape sesi ini menerapkan relabeling LLM skala penuh (4.349 studi) sbg training signal — semua masih
pakai label eksternal mentah (flight0234/steven/pilkwang) atau regex. **Ini satu-satunya lever yg
BUKAN commodity** — satu-satunya cara realistis mengalahkan lapangan yang bergerak sama cepatnya
adalah bawa sesuatu yang tidak mereka punya, bukan mengejar tuning bobot yang sama.

### 28d. Strategi menuju top-1, versi diperbarui (gap sekarang: 0,941 → 0,956, tetap +0,015)

Prinsip tidak berubah dari §26f (jangan kejar desimal public LB, itu sudah dibuktikan overfit —
`Scott Willis 125 submission`), tapi rencana konkretnya diperbarui dgn celah v3 di atas:

1. **[Prioritas 1, murah, GPU sedikit]** Submit checkpoint v3 SENDIRIAN dulu ke LB real utk konfirmasi
   (WAJIB, sesuai pelajaran c5-solo §27c — gold-58 menang TIDAK menjamin LB menang). Kalau checkpoint
   c5 solo dulu = 0,9206 gold-58 → 0,927 LB (offset +0,006), v3 (0,9234 gold-58) REALISTIS diharapkan
   ~0,929-0,932 LB SENDIRIAN — bukan pemenang sendirian, tapi sinyal apakah layak masuk ensemble.
2. **[Prioritas 2, kalau (1) positif]** Ganti/tambah checkpoint v3 SEBAGAI arm CoAtNet di pipeline
   "Bend the Knee to the Dinosaurs" (source sudah ada lokal, `code_research/dinosaurs/`) — ganti
   `raptor_ft_coatnet_v5_full_swa.pt` dgn v3, atau tambah sbg arm ke-5 dgn bobot kecil dulu (diversity-
   safe) — ini gabungkan keunggulan label kita DENGAN seluruh stack DINOv2+DINOv3+RadImageNet+per-
   label-routing yg sudah proven, drpd coba kalahkan 0,941 pakai arm kita sendirian (sudah terbukti
   berkali-kali TIDAK cukup — SlotHead 0,837, c5-solo 0,927).
3. **[Prioritas 3, disiplin metodologi, gratis]** JANGAN tiru bobot per-label komunitas (0,675/0,70/
   1,00) mentah-mentah — itu hasil probing 125-submission, rawan overfit public LB (persis §26f/§27c).
   Kita PUNYA infra nested-CV gold-58 (pola antoinegg1 E10/E9) — pakai itu utk cari bobot per-label
   SENDIRI antara (arm publik + v3 + stock CoAtNet), lebih kemungkinan robust ke private LB.
4. **[Prioritas 4, sudah lama diantre]** Arm RadImageNet target Synovitis (§27a) — makin diperkuat:
   §27b/28b (gated spatial pooling) & Synovitis SAMA-SAMA masih titik lemah v3 kita (0,84) MAUPUN
   pola umum industri.
5. **[Disiplin akhir]** Pilih 2 final submission profil beda (agresif ikut fork terbaru vs konservatif
   diversity-safe) — proteksi shakeup private LB, sesuai §26f poin Fase D, makin relevan krn #1
   sekarang py 125 submission (sinyal fit yg sangat spesifik ke public LB).

**Realisme (diperbarui)**: gap +0,015 SAMA besarnya dgn 3 minggu lalu meski usaha maksimal — lapangan
bergerak bersama. Tidak ada jaminan #1 bisa dikalahkan; jalur dgn expected-value positif TETAP hanya
yang membawa sesuatu yg TIDAK dimiliki commodity crowd (label engineering kita, §28c) dikombinasikan
DENGAN (bukan pengganti) stack publik yg sudah proven — bukan arsitektur baru dari nol.

## 29. Riset mendalam pasca-hasil-null + keputusan mandiri (13 September malam, arahan user "riset dengan sangat mendalam dan putuskan sendiri")

### 29a. EKONOMI KUOTA — kendala sebenarnya, bukan jumlah submission (temuan operasional terpenting sesi ini)

Diukur langsung via `kaggle quota` sebelum/sesudah: **7,55h → 17,27h = ~9,7 GPU-jam untuk SATU
submission pipeline penuh** (plus 2 dev-test ~7 menit). Mesin T4×2, jadi ≈4,8 jam wall-clock —
artinya juga: **run submission kemarin TIDAK kehabisan waktu** (batas 9 jam, terpakai ~4,8 jam),
sehingga hipotesis "arm v3 di-skip oleh self-gating waktu" **hampir pasti SALAH** — arm v3
kemungkinan besar memang jalan, efeknya saja yang di bawah resolusi 3 desimal leaderboard.

**Implikasi strategis**: sisa kuota 12,73h (refresh 19 Sep) = cukup untuk SATU pipeline penuh lagi
(9,7h) ATAU satu training run (~7-9h), TIDAK KEDUANYA. Submission solo 1-arm jauh lebih murah
(~4,6 GPU-jam, terukur 6,2 detik/studi × 1.322 studi ≈ 2,3 jam wall). **Kuota, bukan kuota
submission harian (5/hari), yang jadi pembatas nyata.** Semua perencanaan eksperimen ke depan wajib
pakai angka ini.

### 29b. v3 vs c5: BUKAN redundan — justru lebih beragam dari arm yang sudah mereka blend

Hipotesis awal "v3 = c5 yang di-fine-tune, jadi terlalu mirip untuk menambah nilai" **DIUJI dan
DITOLAK**: korelasi Pearson rata-rata (58 gold, `oof_coatnet_v5_full_swa.csv` dari harvest kita
sendiri vs `train_coatnet_v3_rawmean_gold_preds.csv`):

| pasangan | korelasi |
|---|---:|
| **v3 vs c5 (kita vs stock)** | **0,938** |
| c5 vs c4_full (dua arm yg SUDAH sama-sama dipakai di ensemble publik) | 0,955 |

v3 **lebih tidak berkorelasi** dgn c5 dibanding c4 yang sudah mereka anggap layak jadi anggota.
Per label, divergensi terbesar: **Synovitis 0,751** (v3 melihat Synovitis sangat berbeda) —
menarik krn Synovitis titik lemah universal. Jadi v3 SAH sbg anggota ensemble, bukan duplikat.

Uji blend lokal di 58 gold (aproksimasi arm CoAtNet: 0,75·c5 + 0,25·c4 = 0,9220):
- tambah v3 α=0,15 → 0,9254 · α=0,25 → 0,9264 · α=0,40 → **0,9265 (puncak)**
- ganti c5 dgn v3 → 0,9264
- **v3 DAN c5 dua-duanya, v3 lebih berat (0,50·v3 + 0,25·c5 + 0,25·c4) → 0,9280 (terbaik)**

Artinya: struktur ideal = pertahankan c5, TAMBAH v3 dgn bobot ≥ c5 — bukan ganti, bukan α kecil.

### 29c. Sumber label publik BARU diuji (qwen, tonylin, zhen931) — naif menang, nested-CV MEMBALIKKAN

Ditemukan 4 dataset label baru yg belum pernah kita uji. Skor vs 58 gold:

| sumber | macro AUC |
|---|---:|
| **`tonylin1026/rsna-knee-refined-soft-labels`** | **0,9027** |
| `flight0234` (tulang punggung kita sekarang) | 0,8991-0,8999 |
| Claude pilot (punya kita) | 0,8953-0,8961 |
| `zhen931/rsna-knee-weak-labels-v8` | 0,8700 |
| `laymond/...qwen3-8b-weak-labels` (Qwen 32B) | 0,8429 |

`tonylin1026` ternyata **sumber label publik terkuat yang pernah kita ukur** — mengalahkan
flight0234. Grid-search 4-sumber menemukan blend 0,9069 (flight .3/claude .1/tony .6) vs blend kita
sekarang 0,9040 → tampak +0,0029.

**TAPI nested-CV (bobot dipilih di 4 fold, diukur di fold ke-5 yg tak terlihat) MEMBALIKKAN
kesimpulan**: blend baru **0,9010** vs blend lama (flight .7 + claude .3) **0,9033**; bootstrap
2.000 resample: **P(blend baru lebih baik) = 0,342** — lebih sering LEBIH BURUK. Bobot per-fold
juga tidak stabil (flight 0,0-0,3; tony 0,6-0,9). **Kesimpulan: "peningkatan" itu murni overfitting
ke 58 studi.** Label training kita sekarang TETAP yang terbaik; line "tambah sumber label publik"
sekarang sudah DITUTUP 2× (§27d dgn steven/pilkwang, §29c dgn tony/qwen/zhen) — **plafon label
report publik ≈ 0,903 OOF, jangan diulang lagi.** Temuan ini menyelamatkan 7-9 jam kuota yang
nyaris dipakai melatih ulang di atas label yang sebenarnya lebih buruk.

### 29d. Keputusan mandiri: eksperimen MURAH yang paling menentukan dulu

Menolak opsi "ulangi pipeline penuh dgn α lebih besar" (9,7 GPU-jam untuk efek yg kemungkinan besar
tetap tak terbaca di 3 desimal, dan TIDAK bisa didiagnosis krn log run submission tak pernah
diekspos Kaggle — §MEMORY §4). Sebagai gantinya: **submit v3 SENDIRIAN** (`rsna-knee-submit-v3-solo`,
~4,6 GPU-jam) — harness, K_EVAL=42, dan struktur SAMA PERSIS dgn c5-solo yang sudah pernah disubmit
real (0,927, 26 Agustus). Satu-satunya variabel beda = bobot checkpoint.

**Kenapa ini eksperimen paling bernilai**: memberi jawaban apple-to-apple atas pertanyaan inti yang
belum pernah terjawab — *apakah label engineering kita benar-benar menghasilkan model lebih baik di
data nyata, atau keunggulan +0,0028 di 58 gold cuma noise?* Hasilnya menentukan apakah investasi
training besar (v4, 7-9 jam kuota) layak sama sekali. Eksperimen murah yang meng-de-risk keputusan
mahal.

### 29e. HASIL DECISIVE: v3 solo = 0,932 vs c5 solo = 0,927 di LB NYATA (+0,005)

Eksperimen §29d terjawab: **label engineering kita TERBUKTI menghasilkan model lebih baik di data
nyata**, bukan artefak 58 studi gold. Selisih di LB nyata (+0,005) bahkan LEBIH BESAR dari selisih
di gold-58 (+0,0028). Ini juga berarti checkpoint `c5` — yang jadi **arm dominan (w=0,60) di SEMUA
notebook publik teratas** — sekarang punya pengganti yang lebih baik, dan cuma kita yang punya.

Koreksi estimasi kuota: run solo 1-arm cuma **1,24 GPU-jam** (bukan 4,6h spt diperkirakan §29a).
Pipeline penuh tetap ~9,7 GPU-jam. Submission solo = eksperimen murah, pakai sesering perlu.

### 29f. DIAGNOSIS GURU-vs-MURID per label — membalikkan asumsi lama "bottleneck = label"

Diagnosis yg disarankan §26d ("bandingkan extractor label vs gold, lalu image-model vs extractor")
akhirnya DIJALANKAN (58 gold, guru = label training v3 [flight .7 + claude .3], murid = prediksi
model v3):

| Label | Guru (label) | Murid (v3) | Selisih | Tafsir |
|---|---:|---:|---:|---|
| PF OA | 0,9003 | 0,8571 | **−0,0431** | model tertinggal → **headroom MODELING** |
| Lateral OA | 0,8491 | 0,8124 | **−0,0368** | model tertinggal → **headroom MODELING** |
| Lateral Meniscus | 0,8826 | 0,8472 | **−0,0354** | model tertinggal → **headroom MODELING** |
| ACL | 0,9865 | 0,9828 | −0,0037 | ~setara |
| MCL / Medial Meniscus / Fracture | 0,976 / 0,957 / 0,905 | 0,982 / 0,968 / 0,932 | +0,006…+0,027 | label mulai jadi plafon |
| Medial OA / Baker's | 0,934 / 0,924 | 0,980 / 0,971 | +0,046 / +0,047 | label jadi plafon |
| Synovitis | 0,7820 | 0,8399 | **+0,0579** | model MELAMPAUI guru |
| Contusion | 0,8758 | 0,9406 | **+0,0648** | model MELAMPAUI guru |
| Effusion | 0,8665 | 0,9689 | **+0,1025** | model MELAMPAUI guru |
| **RATA-RATA** | **0,9033** | **0,9234** | **+0,0202** | |

**Kesimpulan yg mengubah arah strategi**: di **8 dari 12 label, model SUDAH MELAMPAUI kualitas
labelnya sendiri** — artinya "bottleneck = kualitas label" (asumsi proyek sejak Agustus, §26d/§27e)
**TIDAK LAGI BERLAKU secara umum**. Itu juga menjelaskan kenapa 2 percobaan menambah sumber label
publik gagal (§27d, §29c): labelnya memang sudah bukan pengungkit utama.

**Pengungkit yg tersisa jelas & terukur**: cuma **3 label** (PF OA, Lateral OA, Lateral Meniscus)
di mana model masih tertinggal 0,035-0,043 dari gurunya sendiri. Kalau ketiganya ditarik naik
sampai setara gurunya saja → **≈ +0,010 macro AUC** — persis ordo besaran gap ke papan atas.

**Ketiganya satu keluarga**: temuan kecil/terlokalisir/berjenjang (osteoarthritis patellofemoral,
OA lateral, robekan meniscus lateral) — PERSIS kelas temuan yg §27b hipotesiskan butuh **spatial
attention pooling, bukan GAP**, yg v23 kita sendiri sudah buktikan menaikkan meniscus +0,06-0,10
vs GAP, dan yg komunitas baru saja adopsi lewat "gated spatial head" di arm 0,941 mereka (§28b).
Tiga jalur bukti independen menunjuk ke mekanisme yg sama. **Ini desain v4: CoAtNet + gated spatial
pooling head, dilatih di label kita yg sudah tervalidasi, dgn bobot loss ekstra utk 3 label itu.**

## 30. Simulasi ensemble: syarat kuantitatif untuk mengejar papan atas (26 September)

Konteks: dua percobaan menyisipkan v3 ke ensemble publik (§48 α=0,15 di luar; §50 w=0,35 di dalam
ARMS) SAMA-SAMA menghasilkan 0,941 persis. Sebelum membakar kuota untuk percobaan ke-3, dilakukan
simulasi Monte-Carlo (N=1.322 studi = ukuran test asli, prevalensi per label diambil dari 58 gold,
kekuatan sinyal dikalibrasi supaya 1 model = 0,932 = skor v3 solo di LB nyata).

### 30a. Menumpuk model SEJENIS mentok — keberagaman arsitektur yang menentukan

| jumlah model | ρ=0,98 | ρ=0,94 | ρ=0,88 | ρ=0,80 |
|---:|---:|---:|---:|---:|
| 1 | 0,9305 | 0,9312 | 0,9320 | 0,9326 |
| 4 | 0,9333 | 0,9371 | 0,9407 | 0,9468 |
| 12 | 0,9336 | 0,9373 | 0,9420 | 0,9511 |

ρ=0,94 adalah korelasi TERUKUR v3 vs c5 (§29b) — yaitu dua CoAtNet yang beda sumber label. Artinya
**melatih lebih banyak CoAtNet di label kita akan mentok di ~0,937**, berapa pun jumlahnya. Ini
sekaligus menurunkan nilai v4 (CoAtNet lagi, cuma beda bobot loss) — bukan pengungkit besar.

### 30b. TAPI model tambahan harus KUAT juga, bukan cuma beragam (koreksi penting)

Simulasi lanjutan dgn model tambahan yang LEBIH LEMAH (0,900, realistis utk backbone baru yang
dilatih dari ImageNet, bukan dari checkpoint kuat):

| komposisi | bobot SAMA | bobot DIOPTIMALKAN | w(model kuat) |
|---|---:|---:|---:|
| v3 sendirian | 0,9334 | 0,9334 | — |
| v3 + 1 lemah (ρ=0,85) | 0,9237 | 0,9335 | 0,98 |
| v3 + 4 lemah (ρ=0,85) | 0,9180 | 0,9331 | 0,97 |
| v3 + 4 lemah (ρ=0,75) | 0,9288 | 0,9373 | 0,68 |
| v3 + 6 lemah (ρ=0,75) | 0,9289 | 0,9377 | 0,65 |

**Dua pelajaran keras:**
1. **Rata-rata berbobot sama dgn model lebih lemah MERUSAK** (0,9334 → 0,9180). Bobot wajib
   dioptimalkan; pada ρ=0,85 bobot optimal utk model lemah nyaris NOL (w_kuat 0,97) — artinya
   model lemah yang tidak cukup beragam TIDAK LAYAK dimasukkan sama sekali.
2. Bahkan pada ρ=0,75 (beda arsitektur sungguhan), 6 model tambahan @0,900 cuma memberi **+0,004**
   — jauh dari +0,015 yang dibutuhkan utk mengejar 0,959.

**Konsekuensi strategi**: rencana "latih banyak backbone baru dari ImageNet" GUGUR — hasilnya
terlalu lemah untuk memberi lompatan. Anggota ensemble baru harus **kuat (~0,93) DAN beragam
(ρ≤0,8)** sekaligus. Kandidat yang memenuhi keduanya: **fine-tune checkpoint PUBLIK yang sudah kuat
tapi keluarga arsitekturnya beda dari CoAtNet** (DINOv2/DINOv3 SlotHead — kodenya `class SlotHead`,
`class Model`, `build_model()` semuanya TERLIHAT di notebook publik d4-blend, dan checkpoint
20-member pilkwang tersedia) **di atas label hasil relabeling kita** — kombinasi yang belum dimiliki
siapa pun.

### 30c. Catatan metodologis untuk diri sendiri

Teks kesimpulan pada percobaan simulasi pertama sempat ditulis SEBELUM melihat angkanya, dan
ternyata bertentangan dgn hasil (mengklaim "model lemah tetap menolong", padahal angkanya turun).
Dikoreksi di §30b. Pelajaran: jangan menyiapkan kalimat kesimpulan di dalam skrip sebelum hasil
keluar — tulis kesimpulan hanya setelah membaca angkanya.

### 30d. Sumber label publik baru (26 Sep) — DIUJI, NEGATIF LAGI (konfirmasi ke-3, line DITUTUP)

Lima dataset label baru muncul dlm 2 minggu terakhir. Diuji vs 58 gold:

| sumber | macro |
|---|---:|
| `riadmohamed42/jev-knee-labels` HYBRID_labels_scores | **0,9037** |
| blend kita (flight .7 + claude .3) | 0,9033 |
| flight0234 | 0,8991 |
| `riadmohamed42` FINAL2_labels_scores | 0,8745 |
| `saisahanas/...radiologist-report-labels` | 0,8333 |

**Perangkap yang hampir terlewat**: beberapa file (`yehezkielhaganta/psuedo-labeling-knee/train.csv`,
`data.csv`, `gold74_eval_set.csv`) skornya **1,0000** — setelah dicek, 100% nilainya IDENTIK dgn
label gold kompetisi, alias file itu cuma MENYALIN label gold, bukan memprediksinya. Bukan sumber
label lebih baik. (Pelajaran: selalu cek identik-atau-tidak sebelum percaya skor mendekati sempurna.)

JEV hybrid (0,9037) sekilas mengalahkan blend kita (0,9033), tapi **nested-CV membalikkannya**:
blend 3-sumber = **0,8992** vs blend kita **0,9033**; bootstrap 2.000 resample
**P(baru > lama) = 0,172**. Bobot per-fold sangat tidak stabil (jev 0,2-0,9) — persis pola
overfitting ke 58 studi yg sudah 2x terjadi.

**Kesimpulan: line "tambah sumber label publik" DITUTUP PERMANEN** — sudah 3 kali diuji
independen (§27d steven/pilkwang, §29c tony/qwen/zhen, §30d jev/saisahanas/xhaitanya) dan
SELALU kalah dari blend flight+Claude kita begitu diukur jujur. Plafon label report publik
≈ 0,903 OOF. Pengungkit yang tersisa ada di MODELING & ENSEMBLING, bukan label.

## 31. AKAR PENYEBAB dua hasil null akhirnya ketemu: bobot GLOBAL vs PER-LABEL (26 September)

Diuji dgn data yang SUDAH ada (tanpa GPU): prediksi gold-58 milik v3 kita vs proksi cabang CoAtNet
publik (0,75·c5 + 0,25·c4, keduanya dari harvest kita sendiri). Bobot dipilih **nested-CV 5-fold**
(dari 4 fold, diukur di fold ke-5), lalu diuji bootstrap 2.000 resample.

| skema | OOF macro | bootstrap P(> publik) |
|---|---:|---:|
| publik saja | 0,9220 | — |
| blend bobot **GLOBAL** | 0,9241 | 0,758 |
| blend bobot **PER-LABEL** | **0,9296** | **0,989** |

Bobot v3 yang terpilih per label (rata-rata lintas fold):

| v3 DOMINAN | w | v3 nyaris NOL | w |
|---|---:|---|---:|
| Baker's | 0,85 | Lateral Meniscus | 0,01 |
| PF OA | 0,80 | MCL | 0,03 |
| Synovitis | 0,75 | Lateral OA | 0,05 |
| ACL | 0,69 | | |
| Fracture | 0,67 | | |

**Ini menjelaskan tuntas kedua hasil null (§48 α=0,15 global; §50 w=0,35 global di ARMS)**: satu
bobot global memaksa kompromi antara label di mana v3 JAUH lebih baik (w optimal 0,7-0,85) dan
label di mana v3 MERUGIKAN (w optimal ~0). Rata-ratanya saling meniadakan -> 0,941 -> 0,941, dua
kali, persis seperti yang teramati.

**Konsekuensi langsung utk rencana:**
1. **Penggantian checkpoint menyeluruh (§build_submit_ourarms) BUKAN skema optimal** — itu setara
   memaksa w=1,0 di SEMUA label, padahal data bilang w≈0 utk Lateral Meniscus/MCL/Lateral OA.
   Pertahankan arm publik, TAMBAHKAN arm kita, lalu bobot per-label.
2. Skema yang benar = **§48 diulang TAPI per-label**: jalankan pipeline publik apa adanya, jalankan
   arm kita, lalu blend dgn bobot per-label hasil pengukuran (bukan tebakan). Implementasinya
   justru lebih sederhana krn tidak perlu menyentuh internal mereka.
3. Bobot itu HARUS diukur terhadap ensemble publik PENUH (bukan proksi CoAtNet saja) — inilah
   gunanya kernel panen gold-58 yang sedang antri.
4. Catatan silang §29f: v3 dapat bobot ~0 justru di 2 dari 3 label "headroom modeling"
   (Lateral Meniscus, Lateral OA) — konsisten, di situ v3 memang lemah (0,847 & 0,812), jadi arm
   publik lebih dipercaya. Perbaikan 3 label itu tetap pekerjaan modeling terpisah (v4/v5).

### 31a. Uji shrinkage bobot per-label — tidak diperlukan

Dgn 58 studi, bobot per-label berpotensi overfit, jadi diuji penyusutan ke arah bobot global
(w = s·w_perlabel + (1-s)·w_global), semuanya diukur nested-CV:

| s (1 = murni per-label) | OOF | P(> publik) |
|---:|---:|---:|
| 0,00 (global murni) | 0,9241 | 0,757 |
| 0,50 | 0,9272 | 0,949 |
| 0,75 | 0,9283 | 0,979 |
| **1,00 (per-label murni)** | **0,9296** | **0,989** |

Monoton — sinyalnya cukup kuat sehingga bobot per-label bertahan di fold yang tak terlihat, tanpa
perlu disusutkan. Dipakai apa adanya.

### 31b. Batas proksi: kenapa kernel panen WAJIB, bukan opsional

Dicoba menyusun tiruan ensemble publik tanpa GPU: cabang Transformer = OOF resmi 20-model DINOv2
`pilkwang` (diunduh HANYA `oof.npz`, 368 KB — bukan 1,65 GB bobotnya), cabang CoAtNet = proksi
0,75·c5+0,25·c4, digabung memakai **bobot routing per-label ASLI mereka** yang diekstrak dari kode
d4-blend (0,60 default; ACL/Lat OA/Fracture 0,675; Medial Meniscus 0,70; Lateral Meniscus 1,00).

| komponen | gold-58 |
|---|---:|
| cabang Transformer (pilkwang DINOv2) | 0,8400 |
| cabang CoAtNet (proksi) | 0,9220 |
| **tiruan ensemble publik** | **0,9131** |
| v3 kita | 0,9234 |

**Tiruan ini TERBUKTI meleset**: stack asli mencetak 0,943 di LB, sementara tiruannya cuma 0,9131
di gold-58. Bandingkan offset yang diketahui utk v3 (gold-58 0,9234 → LB 0,932, ≈ +0,01): kalau
offset serupa berlaku, tiruan ini setara ~0,923 LB — jauh di bawah 0,943 yang sebenarnya. Sebabnya
jelas: cabang Transformer asli bukan cuma 20-model DINOv2, tapi juga DINOv3 5-fold + RadImageNet
(E10/E13/V48) + kalibrator, yang semuanya tidak ada di proksi.

**Konsekuensi**: bobot per-label yang dihitung terhadap proksi ini akan TERLALU MEMIHAK model kita
(karena pembandingnya dibuat lebih lemah dari aslinya) — dipakai submit justru berisiko merugikan.
Terlihat juga dari gejalanya: terhadap proksi lemah ini, bobot global (0,9238) hampir menyamai
per-label (0,9241), padahal terhadap proksi CoAtNet yang lebih realistis selisihnya lebar
(0,9241 vs 0,9296). **Maka bobot final HARUS diukur dari output kernel panen (pipeline publik
sungguhan di 58 studi gold), bukan dari tiruan apa pun.** Ini sekaligus alasan kernel
`submit_perlabel` sengaja BELUM dipush.

## 32. NOISY-STUDENT: label fusi guru+murid jauh lebih baik dari keduanya (30 September)

Konteks: posisi 30 Sep = #462/4613, 0,943 (#1 rhoskeri 0,961, gap 0,018, 22 hari tersisa).
Notebook induk `ryokucha/rsna-knee-d4-blend-0946-ours10` (terdokumentasi oleh forker
`yamadan96/...d4-public0946`): **0,946 dicapai dgn mencampur ensemble ConvNeXt PRIVAT milik
penulisnya sebesar 10% ke pipeline publik 0,943** — bukti langsung +0,003 dari satu keluarga model
privat yang kuat & berbeda arsitektur (persis syarat "kuat DAN beragam" dari §30b).

### 32a. Uji: fusi label laporan (guru) + prediksi gambar (murid) terhadap gold-58

| w_murid | guru + v3 | guru + pilkwang (DINOv2, lemah) |
|---:|---:|---:|
| 0,0 (guru saja) | 0,9033 | 0,9033 |
| 0,2 | 0,9164 | 0,9110 |
| 0,4 | 0,9290 | 0,9083 |
| **0,5** | **0,9317** | 0,9037 |
| **0,6** | **0,9342** | 0,8943 |
| 0,8 | 0,9296 | 0,8699 |
| 1,0 (murid saja) | 0,9234 | 0,8400 |

- Murid KUAT (v3, gold-58 0,9234) + guru: fusi **0,9317-0,9342**, melampaui KEDUANYA
  (+0,030 di atas guru, +0,011 di atas murid). Kurva mulus & datar di w=0,4-0,8 → bukan puncak
  kebetulan. Penyebab: kesalahan label-laporan dan kesalahan model-gambar independen.
- Bahkan murid LEMAH (pilkwang, 0,84) menaikkan guru 0,9033 → 0,9110 (nested-CV per-label 0,9108,
  bootstrap P=0,935; global 0,9068, P=0,875). Korelasi guru~murid cuma 0,658.
- Makin kuat muridnya makin besar gain: 0,84 → +0,008; 0,923 → +0,031.

### 32b. Konsekuensi: satu putaran noisy-student
label_fusi = (1-w)·guru + w·murid_OOF  (raw-probability-mean, bukan rank — pelajaran §46 soal
kalibrasi BCE) → latih ulang → model lebih baik. SYARAT KERAS: prediksi murid utk tiap studi lemah
harus OUT-OF-FOLD; kalau model sudah melihat studi itu, ia cuma menggaungkan label guru. Makanya
K-fold CoAtNet (4 fold, fold sadar-GRUP krn 194 studi berbagi laporan identik, §27c).

**Risiko yang diketahui**: checkpoint c5 (init) sudah pernah dilatih penulisnya pada SEMUA studi
lemah dgn label mirip guru → prediksi held-out-nya bisa "bocor" sebagian. Kernel K-fold menyertakan
PROBE: model inisial (pra-latih) memprediksi 150 studi lemah vs gold (tak pernah dilihat siapa pun)
dan membandingkan korelasinya. Kalau kontaminasi berat → ganti init ke bobot ImageNet bersih
(dataset `mohamed3abdelrazik/coatnet-rmlp-2-rw-384-sw-in12k-ft-in1k` tersedia offline).

### 32c. Koreksi fakta operasional (PENTING)
Klaim saya sebelumnya "7 submission fork publik menghabiskan ~22 jam kuota (~3 jam/submission)"
**TIDAK TERBUKTI**: akun ini juga menjalankan proyek lain (`casmi26-*`, `fase-c-qlora-train`,
`fase-a-*`) yang memakai kuota & slot GPU yang sama. Biaya riil per submission RSNA tak diketahui;
estimasi lama (~9,7 jam) dan baru (~3 jam) sama-sama tak bisa dipegang. Kuota GPU (30h/minggu) dan
batas 2 sesi GPU batch DIBAGI dgn proyek lain → jangan rencanakan memakai semuanya.

Cache DICOM (tiap kernel GPU bayar ~0,75 jam hanya utk decode) dipindah ke 3 kernel CPU
(`rsna-knee-volcache-s{0,1,2}`, tanpa kuota GPU, jalan paralel) dan dimount via kernel_sources.

### 32d. Hipotesis prior-situs (site) — DIBANTAH, ditutup

Prevalensi label guru berbeda besar antar situs (proksi = bahasa laporan): PF OA Kroasia 0,62 &
Prancis 0,63 vs Bulgaria/Serbia 0,22; Lateral OA 0,52 vs 0,15; simpangan baku antar-situs 0,11-0,13
utk label OA — dan OA (PF OA, Lateral OA) adalah label terlemah kita. Diuji apakah menggeser logit v3
dgn log-odds prior situs menambah AUC terhadap GOLD (anotasi gambar, bebas dari gaya penulisan laporan):

| beta (kekuatan prior) | 0 | 0,25 | 0,5 | 1,0 |
|---|---:|---:|---:|---:|
| macro gold-58 | 0,9234 | 0,9218 | 0,9192 | 0,9140 |

Turun monoton, PF OA ikut memburuk (0,857 → 0,833). Model gambar sudah menangkap info situs
(penampilan scanner), ATAU perbedaan prevalensi itu sebagian gaya pelaporan, bukan prevalensi nyata.
(Caveat: n=58, bahasa cuma proksi kasar situs.) Line ini ditutup.

### 32e. Fusi adaptif-konfidensi — tidak lebih baik dari fusi konstan
Tabel guru ternyata penuh nilai DEFAULT konstan (~0,2325 utk label yang tak disebut laporan,
konfidensi rendah) — guru praktis tak informatif di sebagian besar pasangan (studi,label), dan di
situlah murid paling berguna. Tapi varian w_murid fungsi (1-conf) tidak mengalahkan konstan:
konstan w=0,6-0,7 -> 0,9348-0,9350; (1-conf)^0,5 -> 0,9342; 0,4+0,4(1-conf) -> 0,9340; 1-conf ->
0,9274. Dipakai: konstan w=0,5 (sisi konservatif: murid fold dilatih 75% data).

## 33. HASIL PANEN: gold-58 TERKONTAMINASI untuk stack publik — rencana bobot per-label vs pipeline penuh gugur (30 September)

Kernel panen (`rsna-knee-harvest-public-gold58`, pipeline d4-blend di 58 studi gold lewat env
`RSNA_COMP_ROOT` + direktori palsu ber-symlink) berjalan 21 menit, menyelesaikan semua tahap publik,
lalu crash di sel "OUR LEG 2/3" (kaki ensemble privat penulis induk: `_EFF6`/`_ENS14` = None krn
dataset-nya tidak terpasang di fork). **Keluaran TETAP tersimpan walau kernel ERROR**
(`kaggle kernels output` mengembalikan semua berkas yg ditulis sebelum crash) — dan itu cukup.

| keluaran antara | gold-58 macro |
|---|---:|
| `submission.csv` = `_public_probe22.csv` = `_pipeline_stage.csv` (pipeline publik penuh) | **0,9515** |
| `_coat_family_rank.csv` (resgated + D4 + …, rank) | 0,9315 |
| `_raptor.csv` | 0,9353 |
| `_raptor_public.csv` (Raptor 4-arm murni publik) | 0,9254 |
| `_coat_arm.csv`, `_coat_repairv1.csv` | 0,9101 / 0,9103 |
| `submission_checkpoint_rank_mean.csv` (tahap DINOv2 20-anggota) | 0,9088 |
| v3 kita (referensi) | 0,9234 |

### 33a. KONTAMINASI: gold-58 tidak bisa dipakai menimbang stack publik
Stack publik mencetak **0,9515 di gold-58 tapi 0,943 di LB** (selisih −0,008; model bersih spt v3
justru +0,009: gold 0,9234 → LB 0,932). Bukti kontaminasi di komponennya: tahap DINOv2 = 0,9088 di
sini padahal OOF resmi pilkwang cuma 0,840 (anggota dilatih dgn gold; persis bug kebocoran §13);
kalibrator RadImageNet (88 fitur) dan seleksi checkpoint "top3" pun memakai gold-58.

**Konsekuensi**: bobot blend per-label yg diukur TERHADAP OUTPUT PUBLIK di gold-58 akan bias
(publik tampak terlalu kuat → bobot arm kita terlalu kecil). Rencana §31 ("ukur bobot vs pipeline
penuh") **GUGUR**. Yang tetap BERSIH di gold-58: (i) model kita sendiri (gold tak pernah ikut latih),
(ii) cabang Raptor c5/v10/v8 (dreaddevelopment melaporkan gold-58 sbg held-out). Jadi:
- perbandingan ANTAR-ARM (mis. round-2 vs stock) di gold-58 = VALID;
- keputusan integrasi ke ensemble publik harus lewat LB nyata, bukan gold-58.

### 33b. Arah integrasi yang dipilih (dari bukti, bukan tebakan)
Jangan tempel arm di luar dgn bobot global (§48/§50 = nihil) dan jangan ukur bobot vs publik penuh
(§33a). Gunakan jalur yang sudah punya routing per-label tuning-an komunitas: **tukar SELURUH
keluarga arm CoAtNet di ARMS dgn versi berlabel-fusi kita** (`build_submit_ourarms.py`) — routing
luar per-label mereka (CoAtNet 0,60-1,00 tergantung label) tetap utuh dan otomatis menimbang arm baru
per label. Syarat masuk: round-2 terbukti > stock di gold-58 (bersih) DAN solo-LB > 0,932.

### 33c. Penyisiran checkpoint publik lain (30 Sep) — tidak ada "makan siang gratis"

Dicek 4 dataset checkpoint yang baru muncul (membaca struktur state-dict TANPA torch lewat
unpickler minimal, dan berkas receipt/JSON kecil):

| dataset | isi | temuan |
|---|---|---|
| `damoonshahhosseini/ct02-teacher-blend-coatnet-checkpoint` | CoAtNet, latih di "teacher blend" | **GAGAL dilatih**: gold AUC 0,52-0,58, loss mandek ~0,90 (receipt-nya sendiri) |
| `shingo257/rsna-knee-trained-checkpoints-v1` | ConvNeXt-small 5-fold + Swin-small | lemah (**gold 0,868**, 224px) & skema keluaran beda (head meniskus/tulang/cairan) — tak kompatibel |
| `ernestorafaelperez/angelus-rsna-knee-gold-weights-v5` | resgated "gold", matriks blend | dilatih dgn label GOLD (kontaminasi utk eval) & matriks blend-nya utk skema lain (PCL dst) |
| `kawaharataishi/rsna-knee-coatnet-distilled-checkpoint` | `coatnet_384_full_cohort_swa.pt` | drop-in Raptor (clsW 12×1024, att.*, res 384), `auc` 0,9231 — sekelas c5/v3; tambahan CoAtNet marginal |

Kesimpulan: tak ada keluarga arsitektur kuat-dan-beragam yang tinggal ambil. Keragaman yang kuat
harus dilatih sendiri di atas label yang lebih baik (§32) — bukan dipungut.

### 33d. Catatan operasional
- `/tmp` hanya 2,7 GB dan 2,0 GB-nya dipakai sesi proyek LAIN (`Vlm-Edge-Jetson`, bukan milik RSNA,
  tidak disentuh). Unduhan >200 MB pakai `~/.cache/rsna_scratch` (disk rumah), hapus setelah dipakai.
- Kolom `size` dari `kaggle kernels files` TIDAK bisa dipercaya: semua berkas npz cache tampil
  "1034" padahal aslinya 5,6 MB (diverifikasi dgn mengunduh satu berkas: (64,336,336), mean 86, 64/64
  irisan terisi). Jangan menyimpulkan cache "kosong" dari listing itu.

## 34. TPU terbukti bisa melatih CoAtNet (2 Okt) + hambatan antrian lintas-proyek

**Probe** (`rsna-knee-tpu-probe-coatnet`, torch 2.8 + torch_xla, TPU Kaggle): CoAtNet-RMLP-2 384px, K=12
window, forward+backward+AdamW di XLA. Kompilasi ~50 dtk dua kali (langkah 0 dan 1), lalu **0,5 dtk/langkah
stabil**, loss turun (0,74 → 0,65). Kuota TPU 20 jam/minggu terpisah dari GPU. Estimasi 1 fold (3.261 studi
x 3 epoch) ~80-90 mnt vs ~130 mnt di T4.

**Keputusan desain port (`build_train_coatnet_kfold_tpu.py`)**: bentuk tensor tetap (window diisi ulang sampai
12; bentuk berubah = rekompilasi ~50 dtk), autocast bf16 'xla' dgn bobot tetap fp32 (BUKAN `XLA_USE_BF16`:
bobot ikut bf16 dan update LR 2e-5 hilang di bawah resolusi bf16), tanpa GradScaler, `mark_step` per studi,
LR dikuantisasi 30 level sbg pengaman rekompilasi, checkpoint disimpan dari state_dict di CPU. Belum
terbukti di data nyata — run pertama akan menunjukkan apakah autocast xla dan kecepatan sesuai.

**Hambatan**: akun ini berbagi kuota & slot dgn proyek lain. Batas sesi: GPU batch 2, **TPU batch 1**. Kernel
TPU proyek lain (`fase-t-...-smoke-tpu` QUEUED, `fase-a-orchestrator-tpu` macet `CANCEL_ACKNOWLEDGED`)
menahan slot TPU; GPU habis (37,4/30 jam terpakai, dihabiskan `tuning-seimbang-*`). Kernel milik proyek
lain TIDAK dihapus. Penjaga otomatis (`auto_launch.sh`, proses terpisah) mencoba kirim fold 2 ke TPU dan,
setelah refresh GPU (3 Okt 07:00 WIB, sisa >=10 jam), meluncurkan fold 3 (dan fold 2 bila TPU belum jalan) di GPU.

### 34a. 4 Okt: fold di GPU+TPU, bug penjaga, dan cache yang cocok geometri
- Penjaga otomatis semalam TIDAK meluncurkan apa pun: ia membaca kolom `total` (30) dari `kaggle quota`
  alih-alih `remaining`, jadi kondisi ">=10 jam" tak pernah bermakna. Kerugian ±19 jam waktu proses (bukan
  kuota). Diluncurkan manual: fold 3 (GPU), fold 2 (GPU) dan fold 2 versi TPU (validasi port, konfigurasi sama).
- Batas sesi akun: GPU 2, TPU 1, **CPU 5**.
- **Ketidakcocokan geometri ditemukan**: cache awal (`volcache-s*`) = 336px, 64 irisan, rentang 6-94%. Pipeline
  publik memakai c5 di rentang 2-98% (336px), v10 384px/64 irisan, v8 384px/44 irisan rentang 6-94%.
  Penulis arm menekankan irisan tepi penting utk meniskus lateral (label terlemah kita). Putaran-2 dilatih pada
  cache yg cocok: `w` (336/2-98%) utk c5, `x` (384/64/2-98%) utk v10, `y` (384/44/6-94%) utk v8
  (`build_volcache_cpu2.py`, CPU saja). Fold OOF (murid) tetap di cache lama — murid hanya sumber pseudo-label.
- Medali: untuk ~4.600 tim, aturan Kaggle standar (bila berlaku di kompetisi ini): perunggu = top 10%, perak =
  top 5%, emas = top 10 + 0,2% tim. Kita 0,943 ≈ batas perunggu; 0,946 ≈ perak. (Ambang diperiksa dari leaderboard
  di langkah berikutnya, bukan diasumsikan.)

## 35. HASIL PUTARAN-2 noisy-student (4 Okt) — label lebih bagus TIDAK menghasilkan model lebih bagus; ensemble yang membantu

Rantai yang terbukti: label guru 0,9033 → 4 murid fold OOF → label fusi (w=0,5) **0,9348** terhadap gold
(+0,032), dgn murid yang bersih (gold tak pernah ikut latih). Tapi **model yang dilatih di label fusi
TIDAK lebih baik**:

| model (bersih thd gold, gold-58) | stok | putaran-2 | selisih |
|---|---:|---:|---:|
| c5 (harness 42 window) | 0,9213 | **0,9200** | −0,001 |
| v8 | 0,9116 | 0,9138 | +0,002 |
| v3 (label raw-mean, putaran-1) | 0,9213 | 0,9234 | +0,002 (LB solo: +0,005) |

Penyebab yang masuk akal: noise gold-58 (SE ~0,01 per model tunggal) jauh di atas selisih label; model sudah
melampaui kualitas guru (§29f) sehingga label yg sedikit lebih baik bukan lagi pembatas. Yang TERUKUR membantu
adalah **menjumlahkan beberapa fine-tune berbeda** (semuanya bersih thd gold):

| | gold-58 |
|---|---:|
| cabang Raptor stok (.6/.1/.1/.2, 94 window; dari panen gold) | 0,9254 |
| rata-rata 4 murid fold (tiap fold 75% data, label guru) | 0,9286 |
| stok + v3 (α=0,4) | 0,9299 |
| stok + 4 murid fold (α=0,6) | **0,9310** |
| stok + v3 + r2c5 + r2v8 | ~0,928 |

Semua gabungan jenuh di ~0,93 (+0,004–0,006 atas stok) — **sama persis dgn peringatan simulasi §30a**: model
sejenis (korelasi c5~c5-reverse 0,994, c5~v10 0,964, c5~v8 0,945) tak bisa menembus ~0,93. Keluarga CoAtNet
sudah habis diperas; langkah berikutnya yang bermakna harus keluarga BERBEDA yang kuat.

Perbaikan fakta: kuota GPU mingguan TIDAK berkurang oleh submission kompetisi (kuota tetap 0,10 jam setelah
submission rekan tim 3 Okt) — klaim lama "~9,7 jam GPU/submission" salah (itu pemakaian proyek lain di akun).
Satu-satunya batas submission adalah anggaran internal 8 jam di pipeline (melewatinya = tidak ada submission).
Biaya per arm Raptor tambahan ≈ +22 menit (log panen: 4 arm = 4,04 dtk/studi pada 2 GPU).

Uji berjalan: **model soup** (rata-rata bobot 4 fold + v3 + r2c5, semua dari init c5 yang sama) — bila satu
model hasil rata-rata mendekati ensemble-nya, keuntungan ensemble didapat dgn biaya inferensi satu arm.

### 35a. Model soup — NEGATIF (ditutup)
Rata-rata bobot dari fine-tune c5 yg sama (init identik): soup_folds4 0,9221/0,9233 (42/94 window), soup_folds4_v3
0,9213/0,9230, soup_all6 0,9221/**0,9245**, soup_v3_r2 0,9206/0,9222 — ≈ c5 stok (0,9221) dan jauh di bawah
ensemble 4 model terpisah (0,9286). Rata-rata bobot meruntuhkan keragaman antar-model yang justru sumber gain
ensemble; keuntungan harus dibayar dgn inferensi tiap model terpisah (+22 mnt/arm di pipeline publik).

## 36. Diskusi 735304 (4 Okt): jalur "pseudo-label + model kecil" menggeser prioritas

**Sumber**: thread "Best single-model score" (114 komentar), dibaca 4 Okt 2026 lewat Chrome. Semua angka di bawah adalah
klaim peserta (peringkat leaderboard saat komentar ditulis), BELUM terverifikasi, tetapi konsisten satu sama lain:
- ResNet-50 5-fold 224px = 0,954 (peringkat 13); CoAtNet 2.5D 224px single-fold = 0,950 (51); Qwen3.5-2B (LoRA +
  vision encoder) 384px single-fold = 0,950 (64); CoAtNet 288px 0,938 -> 0,949 HANYA dgn perbaikan label (47);
  ResNet kecil 224px single-fold 0,936-0,938, latih ~5 menit, 5-fold 0,947 (9); DINOv2 / EffNet-B4 / ResNet50 5-fold
  ~0,940-0,942 (119); 0,942 mentah / 0,943 dgn TTA di 288px (8).
- Resep yg berulang: **pseudo-label multi-guru lewat OOF** ("weak-label dari beberapa sumber (publik, API, lokal) ->
  model guru -> pseudo-label -> model akhir; biaya API < $5"), model kecil beresolusi rendah, fold. Kutipan inti:
  "On the 58 provided labels, my OOF predictions were already closer to the radiologists than the extracted labels the
  model was trained on... labels were the bottleneck" (51) -- temuan yg SAMA dgn §29f kita (model melampaui guru).
  Chris Deotte (118): model dgn pseudo-label "can match our best ensemble CV score". Satu tim 0,943 melaporkan 1 arm
  eksternal berkorelasi 0,83 bernilai +0,004 di LB.
- Hal yg sudah kita punya: label fusi (guru + murid OOF) = 0,9348 di gold-58. Yang BELUM kita punya: model murah
  beresolusi rendah yg dilatih di label itu, dan fold lengkap utk pseudo-label putaran-2.

**Verifikasi klaim "0946"**: fork `ryokucha/rsna-knee-d4-blend-0946-ours10` dan `yamadan96/rsna-knee-d4-public0946` bernama
"0946", tetapi yamadan96 sendiri mengoreksi: pipeline publiknya = 0,943; 0,946 hanya "bila" ditambah ensemble ConvNeXt-tiny
PRIVAT (dua checkpoint single-fold, bobot 10%). Dataset-nya privat (403) dan di leaderboard 3 Okt tim `ryo` = 0,943.
Jadi klaim itu TIDAK terverifikasi -- tidak dijadikan dasar keputusan. (Pipeline publik terbaik tetap 0,943.)

**Keputusan (4 Okt, ~11:15 WIB)**: (1) seed ke-2 CoAtNet-384 dari ImageNet (`inet-gpu-s2`) dihentikan -- 6 jam kuota utk
keluarga yg sama dgn arm stok tak sebanding dgn jalur murah; s1 dibiarkan sbg penentu. (2) Bangun `build_train_cheap_kfold.py`:
CNN kecil (ResNet-50 timm A1, 224px) 6 fold sadar-grup di label fusi r1, 4 studi/forward (BatchNorm), EMA, DataLoader
berpekerja. **Satu sesi Kaggle = 2 GPU T4**; trainer lama hanya memakai cuda:0 -> proses kedua di cuda:1 GRATIS
(2x hasil per jam kuota). Smoke test (48 studi) lolos end-to-end; run penuh `rsna-knee-cheap-resnet50-s1` berjalan.
(3) Rencana: bila gold-58 ensemble 6-fold >= ~0,92 -> (a) fusi pseudo-label putaran-2 dari OOF murah + OOF CoAtNet,
(b) keluarga CNN lain (ConvNeXt-tiny, EffNet-B3) utk keragaman, (c) submit solo utk ukur LB, (d) masukkan sbg arm tambahan.

### 36b. Bukti langsung "fleet murid" di papan atas + mekanisme blend yang bisa dipakai (4 Okt)
- Notebook publik `pjmathematician/rsna-knee-d4-blend` (versi terbaru) memuat sel 30-32 "OUR LEG": menyimpan keluaran publik,
  menjalankan "fleet murid" (`S_dist`, `S_coat`, `S_cnxl`; 8 + 43 bobot dari dataset PRIVAT `rsna-knee-eff6-assets` /
  `rsna-knee-ens14-assets`, skrip `infer_kernel.py`) lalu mem-blend di ruang RANK dgn **satu bobot global `RSNA_OURS_W`=0,45**
  (alasan penulis: bobot per-label di 870 studi memberi +0,0026 dari noise murni). Tanpa aset privat semua leg gugur ke
  keluaran publik (0,943). Pengaman: gerbang cakupan, gerbang fraksi-ikat 0,5, gerbang kolom konstan, pemulihan otomatis.
  Korelasi rank leg-vs-publik yg diukur penulis offline: **0,75-0,89**.
- Skor leaderboard 3 Okt para penulis publik yg menambah leg privat: pjmathematician **0,947** (#141, dasar publik 0,943 -> +0,004),
  mattiaangeli 0,950 (#100), jiweiliu/Theo Viel 0,951 (#82), dreaddevelopment 0,945 (#248). Penulis yg hanya menjalankan
  pipeline publik: 0,943 (romantamrazov, prvsiyan, evgendvorkin, yamadan96...). `ryo` (ryokucha) = 0,943 (klaim 0,946 tak terbukti).
- Kesimpulan: fleet murid (model kuat + beragam, dilatih di pseudo-label) yg di-blend ~0,3-0,45 memang bernilai ~+0,004 s/d +0,008.
  Itu persis jalur kita. Integrasi akan memakai pola sel yg sama (stash publik -> leg milik kita -> blend rank berbobot global ->
  pemulihan otomatis), bukan mengubah bagian dalam pipeline publik. Waktu: pipeline publik d4-blend ~4,5-6 jam (header notebook:
  DINOsaur ~4,5 jam, Speedy Raptors ~1,5 jam), jadi tersisa ~2-3 jam utk leg kita di bawah batas 9 jam.
- Data eksternal MRI lutut (MRNet, fastMRI+, OAI, KneeMRI, KneeXNet): status lisensi/akses BELUM diputuskan host (thread 743416);
  hanya KneeCoT yg dilarang jelas. KEPUTUSAN: tidak memakai data/bobot MRI lutut eksternal (risiko diskualifikasi > manfaat);
  bobot pra-latih citra umum/medis publik berlisensi permisif tetap dipakai (ImageNet; RadImageNet MIT di hulu).

### 36c. Bukti publik ke-2 + fleet gabungan (5 Okt)
- Notebook publik baru `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`: stack publik 0,943 + reader **ConvNeXt-Tiny 2.5D +
  transformer studi** yang dilatih mandiri (5 fold grup-laporan, soft target = rata-rata 4 tabel label publik, 14 epoch @256px,
  0,4 mm/px, FOV 154 mm, 6 series) -> **reader saja 0,929 (3 fold), blend 70/30 = 0,944**. Bobot + kode inferensi publik
  (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, train.py ada di dataset). Konsisten dgn §36b: keluarga independen ~0,93 pada bobot 0,3 = +0,001.
- Tab Code menampilkan skor versi PENULIS (aset privat ikut): d4-blend 0,946, d4-lite/s75 0,945; fork kita tanpa aset privat = 0,943.
- `build_submit_plus.py reader=goodpjw w2=...` menyisipkan sel sumber reader (kaggle_base/goodpjw_reader_cells_20261004.json) dan
  menjalankannya sbg leg-2 (slot RSNA_EFF_W sel blend pj) dgn batas waktu subprocess. Dev-run lolos.
- Submisi real 5 Okt: students4 w.35 (56846031), rad6 w.30 (56846312), **fleet2 = publik .60 + murid CoAtNet .20 + reader .20**.
- Operasional: unduhan ribuan berkas cache oleh VM Colab memakai token yg sama -> Kaggle API 429 berjam-jam; baca log kernel lewat
  halaman web (Chrome) bila API dibatasi. CLI Colab kadang mem-"prune" sesi saat exec gagal sesaat -> `colab/readopt.py`.

### 36d. HASIL: leg luar MENURUNKAN skor (6 Okt) -> pindah ke penggantian arm DI DALAM stack
| submisi (basis = pj d4-blend terbaru, identik sel-demi-sel dgn fork 72e30e = 0,943) | LB |
|---|---:|
| + leg 4 murid CoAtNet, w 0,35 | 0,942 |
| + leg RadImageNet-ResNet50 6 fold 6 epoch, w 0,30 | 0,942 |
| fleet2: publik .60 + murid .20 + reader ConvNeXt goodpjw .20 | 0,942 |
Ketiganya -0,001. Model yg lebih lemah dari stack (solo 0,929-0,934) diblend 0,30-0,40 mengencerkan stack; bukti +0,001 milik goodpjw
(reader 0,30) tidak mereplikasi di sini (dalam batas noise 3 desimal, tapi arahnya konsisten negatif utk kita). Fleet pjmathematician
(+0,004 pd w 0,45) pasti jauh lebih kuat dari punya kita -> JANGAN ulang leg luar lemah.
Arah baru: perkuat cabang DOMINAN dari dalam: `build_submit_inarm2.py` mengganti berkas di ARMS cell 27 (c5 & c5-reverse -> r2c5, v8 ->
r2v8; v10 stok), menambah akar ke ASSET_ROOTS (pencari aset tidak mem-pin hash arm Raptor), dan menetralkan sel 31 asli pj (crash
`_EFF6 None` bila aset privat tak ada). Dev-run lolos; disubmit 6 Okt.

### 36e. REPLIKA goodpjw = 0,944 (6 Okt 20:00 WIB) — zona perunggu tercapai
- Notebook publik goodpjw2008 disalin UTUH (`kaggle_kernel_goodpjw_replica/`, stack 70% + reader ConvNeXt-Tiny 2.5D 30%) -> **LB 0,944** (klaim penulis terverifikasi).
  Fleet2 kita (publik .60 + 4 murid CoAtNet .20 + reader .20) = 0,942: tambahan CoAtNet kita MENURUNKAN, reader mandiri MENAIKKAN. Pelajaran:
  yang berguna = keluarga INDEPENDEN (arsitektur + resep + pra-latih berbeda) yg cukup kuat (~0,93 solo), bukan varian CoAtNet.
- Peringkat: snapshot 6 Okt (5.278 tim): >=0,943: 1.483 tim; >=0,944: 484 (top 9,2%); >=0,945: 318; >=0,946: 254; >=0,947: 205; >=0,950: 121.
  Batas perunggu (top 10%) = rank 527 (~0,944), perak (5%) = 263 (~0,946), emas = rank 20 (~0,957). Kita sebelumnya rank 750 @0,943.
- Rencana: (1) kunci 0,944 (pilih sbg salah satu dari 2 submisi final); (2) variasi bobot reader (0,40) setelah kuota GPU pulih 10 Okt (commit-run butuh 2 GPU);
  (3) latih reader sendiri bergaya KneeNet (ConvNeXt-Tiny IN22k, token per slot + transformer studi) di label fusi, di TPU, utk leg-2 independen.

## 36f. Status 8 Okt 2026: tim 0,950 (rank 292), plafon publik, dan risiko OAI
- Submisi 56943646 (notebook publik "Apex Grandmaster Stack" v2, ItsmeSujan) = 0,950 -> rank 292/5.490. Isi: ckpt nartaa r384 (Raptor CoAtNet, 96 irisan, 94 jendela, 2 tampilan plain+mirror anatomis; solo 0,949) + reader ConvNeXt goodpjw dgn bobot per-label (0,03-0,35). Ckpt dilatih dgn 2.399 lutut OAI (NDA, akses terkontrol) + label blend (JEV/LLM/report) di SEMUA data kompetisi (termasuk 58 gold -> gold-58 tak bisa dipakai utk evaluasi ckpt ini).
- Ambang (snapshot 8 Okt, 5.490 tim): emas rank<=20 = 0,959; perak <=274 = 0,950; perunggu <=549 = 0,947; dataran 0,950 = rank 146-397 (252 tim); >=0,951:145; >=0,952:122; >=0,953:101; maks 0,964.
- goodpjw (blend ckpt nartaa + stack 0,943): 60/40 = 0,949; + reader sendiri 0,933/0,929 @30% = 0,948. "Sisa headroom ada di label & data eksternal, bukan arsitektur". Enam notebook publik berhenti di 0,950.
- Konsekuensi: blend/leg tambahan dari model berbasis data kompetisi saja TIDAK menambah di atas 0,949; jalur kita (reader KneeNet, murid CoAtNet) tidak lagi bernilai. KneeNet fold 0 TPU: gold-58 EMA 0,736/0,814/0,863/0,896 (ep1-4) lalu divergen ep5 (loss 7,5e10 -> 0,49 sampai ep14); penyebab dugaan: LR puncak + bf16/XLA tanpa grad-clip. Tidak diperbaiki (prioritas turun).
- Kepatuhan: thread 743416 belum dijawab host (OAI, MRNet, fastMRI, KneeMRI, KneeXNet = "unclear"; KneeCoT = dilarang). Panduan host 27 Agu: registrasi/click-through DUA umumnya OK, persetujuan institusional tidak. Bobot nartaa berlisensi 'other' (research/edu termasuk kompetisi ini). Mitigasi: 1 submisi final bersih + 1 agresif.

## 37. Riset mendalam 8 Okt malam: apa yang membawa ke emas, dan apa yang bisa kita jalankan tanpa GPU
Sumber: forum (746792 nartaa, 745861 hengck23, 745214, 743416, 733965), notebook publik (goodpjw 0,949 blend, Apex, bee's knees), dataset Kaggle.
- **Tangga skor nartaa** (LB publik, satu CoAtNet): all-data 80 irisan/78 jendela 0,939 -> 96 irisan 384 px 0,940 -> label blend+JEV 0,942 ->
  224 px 0,940 -> **+contoh OAI (label bermasker) 0,945 (+0,005)** -> resep 384 full-field + OAI 0,948 -> crop native 320 0,948 -> cermin anatomis 0,949.
  Tanpa OAI resep ini mentok ~0,942. Gagal: flip lebar semua bidang (0,940->0,939), crop 304/288 (-0,001/-0,002), epoch 14 tanpa SWA (-0,001),
  latih ulang BN "diperbaiki" (0,948->0,944), latih ulang crop 320 (0,949->0,946). Noise LB antar-latih ~0,002.
- **Pola atas (Tucker Arrants, peringkat 14)**: "sebagian besar model saya di kisaran 0,95 rendah" dgn ekstraksi label Qwen sederhana; ia ragu perbaikan label
  berarti. => 0,959-0,964 = ENSEMBEL banyak model ~0,951 (diduga data eksternal + banyak seed), bukan satu trik. Jarak emas = +0,009 dari plafon publik 0,950.
- Label emas dibuat BUTA terhadap laporan (komentar Prateek Grover); label laporan bisa salah (mis. Efusi=1 padahal "no knee effusion").
- Host MENGIZINKAN LLM hosted utk memproses Competition Data termasuk teks laporan (733965; Rules 2.6.b) selama layanan terjangkau & murah. Peserta (rank 65/81/594)
  memakai LLM utk relabel; "Claude-labels" -> satu model visi 0,942.
- **Data eksternal**: thread 743416 BELUM dijawab host (Okt 8). OAI = akses terkontrol NDA (login.gov + DUC click-through); peserta (hengck23) pakai ChatGPT utk
  mengunduh. Di Kaggle ada mirror publik MRNet (6 GB), fastMRI knee (45-128 GB) + anotasi fastMRI+ (155 KB), OAI-3D-DESS (9,5 GB), OAI-ZIB (26 GB) tanpa label 12-temuan.
  Label yg berguna (OA, efusi, sinovitis, Baker, kontusio) praktis hanya dari OAI/MOAKS (NDA) -> kita TIDAK memakai data eksternal mentah; hanya bobot publik.
- Plafon publik: enam notebook di 0,950 (semua turunan ckpt nartaa 0,949/0,945 + reader goodpjw). Probe goodpjw: stack 0,943 @40% -> 0,949; reader sendiri @30% -> 0,948.
  Soup fine-tune dgn init sama (c5) NEGATIF (§35a) -> "anak" harus punya info baru (label fusi + KD).
- **Program yg dijalankan (TPU/CPU, tanpa GPU)**: (1) cache `z` = geometri persis korpus ckpt nartaa (96 irisan: 26/22/18/12/18, 384 px, rentang 2-98%) via 3 kernel CPU
  `rsna-knee-volcache-z{0,1,2}`; (2) kernel TPU `rsna-knee-train-tpu-oai-child` (`build_train_tpu_oai_child.py`): induk = ckpt r384 nartaa; anak a (KD lambda 0,5) & b (tanpa KD)
  di label fusi r1, LR_BB 1e-5/head 1e-4, 3 epoch, BN dibekukan, SWA epoch 2-3; juga mengeluarkan prediksi in-sample induk (parent_soft_k16.csv).
  (3) Setelah GPU pulih (10 Okt 07:00 WIB): notebook ensembel induk+anak(+reader goodpjw) -> probe LB; terima hanya bila >=0,951.
- **Probe tag DICOM (8 Okt, kernel CPU `rsna-knee-dicom-tags-probe`, 455 studi / 2.537 seri)**: tidak ada PatientAge/Weight/Size/Birthdate/Institution/StudyDate; ADA PatientSex (M 47%/F 44%/O 4%/kosong 5%),
  Laterality (47% seri), Manufacturer (11 nilai)/ManufacturerModelName (34)/MagneticFieldStrength/koil/TR/TE/matriks. PatientID unik per-studi (0 pasien berulang) -> tak ada kebocoran lintas studi.
  Kesimpulan: fitur meta (jenis kelamin, vendor/model) terlalu lemah utk menambah di atas model citra 0,95 -> DITUTUP.

## 37b. Peta skor -> peringkat (papan 8 Okt, 5.490 tim) dan garis medali yang BERGERAK
- Skor publik -> peringkat tengah: 0,950 ~271 (dataran 146-397) | 0,951 ~134 | 0,952 ~112 | 0,953 ~91 | 0,954 ~77 | 0,955 ~68 | 0,956 ~54 | 0,957 ~40 | 0,958 ~32 | 0,959 ~23 | 0,960 ~16 | 0,961 ~10 | 0,962 ~6 | 0,963 ~3.
- Garis bergerak cepat: perak (5%) 0,945 (3 Okt) -> 0,946 (6 Okt) -> 0,950 (8 Okt); perunggu 0,943 -> 0,944 -> 0,947; emas (rank 20) 0,957 -> 0,959; #1 0,961 -> 0,964.
  Proyeksi penutupan 22 Okt (perkiraan kasar): perak ~0,951-0,953, emas ~0,961-0,963. Diam di 0,950 bisa turun ke perunggu.
- Skenario (perkiraan penilaian, bukan pengukuran): diam 0,950 (perak ~40-50%); induk+anak+reader 0,951-0,952 (perak ~70%); + bagging 6-8 model resep tanpa OAI 0,952-0,954 (perak ~80%, emas <1%);
  + model OAI buatan sendiri (butuh daftar NDA oleh user + pemetaan label OAI->12 temuan yg belum diketahui) 0,954-0,957 (perak ~90% bila selesai, emas ~3%). Emas butuh >= ~0,962.

- **KOREKSI 8 Okt 21:10 (atas §37b)**: skenario "bagging 6-8 model tanpa OAI -> 0,952-0,954" TERLALU OPTIMIS. Bukti goodpjw: stack 0,943 (bagging puluhan model tanpa OAI) @40% di atas ckpt OAI 0,949
  -> 0,949 (nol tambahan); reader sendiri 0,933/0,929 @30% -> 0,948. Keunggulan model OAI = PENGETAHUAN (data), bukan varians; rata-rata model tanpa OAI tak menutupnya.
  Perkiraan baru tanpa data OAI tambahan: plafon kita ~0,950-0,951 (anak + r224 + reader per-label: +0,000-0,001). Jalur lain = ckpt OAI BARU yg dipublikasikan peserta lain
  (pantau `kaggle datasets list -s "rsna knee" --sort-by updated` + notebook >=0,951 tiap hari) lalu pasang sbg arm di pipeline Apex. Model OAI buatan sendiri TIDAK layak:
  butuh akun NDA (hanya bisa dibuat user), unduhan ratusan GB, pemetaan label OAI->12 temuan tak diketahui, infrastruktur kita (RAM 5 GB, Kaggle 20 GB keluaran) tak cukup.
  Peluang medali yg dikoreksi (kasar): perunggu ~95%, perak ~35-50% (garis perak diproyeksikan 0,951-0,953), emas ~0-2%.

- **KOREKSI 8 Okt 22:25 (Colab T4, diagnostik induk)**: AUC gold-58 ckpt induk nartaa r384 pada cache z (94 window 384 px penuh, 1 tampilan, fp16) = **0,9214**. Itu BUKAN angka "dilatih pada gold" (akan ~0,98+): gold-58 hanya
  dipakai nartaa utk seleksi epoch (README: "reused for selection") -> mendekati out-of-sample. Maka gold-58 BISA jadi uji RELATIF anak vs induk (sinyal kasar, SE ~0,01; selisih <0,005 = noise). Anak c (Colab) dan a/b (TPU) mencetak gold per epoch.

- **Hasil pertama anak (Colab T4, 8 Okt 23:44 WIB)**: anak `c` (induk r384 nartaa; label fusi r1; KD lambda 0,5; LR_BB 1e-5, LR_HEAD 1e-4; seed 3) setelah 1 epoch (4.349 studi, ±78 mnt, 1,05 dtk/studi): loss label 0,5845 (400 studi pertama) -> 0,5586 (rata2 epoch), KD 0,551 -> 0,573,
  gold-58 0,9207 vs induk 0,9214 (selisih -0,0007 = noise). Anak hampir identik dgn induk -> sbg anggota ensembel nilainya ~0. Diganti anak `e` dgn resep r2 (LR_BB 2e-5, LR_HEAD 1e-3, WD 0,05, KD 0,25) utk menguji apakah label fusi bisa menggeser model berguna.
  Catatan metodologis: pada r2c5 dulu gold-58 TIDAK naik (0,9200 vs 0,9213) padahal LB naik +0,006 -> gold-58 buruk utk mendeteksi perbaikan kecil; keputusan akhir tetap dari LB.

## 37c. Evaluasi gold-58 induk/anak/r224 (Colab T4, 9 Okt 01:30) — garis anak-OAI ditutup
Induk nartaa r384 (cache z, 94 window, fp16): plain 384 = 0,9214; cermin 0,9221; crop 320 plain/cermin = 0,9219/0,9225 (konsisten dgn temuan nartaa: crop 320 ≈ 384 penuh, cermin +0,0005-0,001).
Sibling r224 (crop tengah 307 -> 224, 1 tampilan) = 0,9150. Anak c ep1 = 0,9207 (≈induk; KD kuat, LR kecil). Anak e ep1 (resep r2: LR_BB 2e-5, LR_HEAD 1e-3, WD .05, KD .25) = 0,9138 (turun).
Campuran rank terhadap induk(320+cermin ≈0,9225): +r224 (bobot .2/.35/.5) = 0,9239/0,9238/0,9223; +c1 = 0,9233/0,9232/0,9225; +e1 = 0,9226/0,9231/0,9204; semua 4 = 0,9232. Semua selisih <=0,0015 << SE 0,01.
Kesimpulan: (1) fine-tune ckpt OAI pada label fusi tidak menggeser model ke arah yg terukur-berguna: setelan lembut = salinan induk, setelan kuat = sedikit turun; (2) satu-satunya tambahan yg bernilai (kecil) = sibling r224 (diduga +0,001 di LB). (3) Fine-tune tanpa data OAI tidak bisa menambah pengetahuan OAI — konsisten dgn probe goodpjw.
Dipertimbangkan, TIDAK dikerjakan: OrthoDiffusion (3 U-Net difusi 3D pra-latih MRI lutut, MIT, bobot 527 MB/bidang; input 16x256x256, fitur blok mid_2 pada t=100; dataset Kaggle `prvsiyan/orthodiffusion-official-knee-mri-foundation`) — fitur beku + kepala linear pd label lemah diperkirakan <0,93 solo (keluarga lemah tak menambah di atas 0,949) dan biaya rekayasa tinggi.
Rencana 10 Okt: probe LB tunggal "Apex + r224 (bobot ~.25 di dalam campuran CoAtNet)"; terima bila >=0,951.

## 37d. Riset struktur papan & aturan (9 Okt 02:30)
- **Komposisi papan (8 Okt, 5.490 tim)**: 12 dari top-20 adalah tim SOLO; dari 68 tim >=0,955, 44 solo (ukuran maks tim = 5). Rata2 submisi top-50 ≈143 vs rank 201-400 ≈30. Pembuat artefak publik di top-30 = veteran RSNA/Kaggle GM
  (mis. `christofhenkel` (RSNA 2024), `vaillant` (Ian Pan, juara RSNA berulang), `tamotamo` (RSNA 2025 peringkat 3), `yosukeyama` (rank 9), `wadakoki` (RSNA 2024 spine), `tonylica`). => emas = pipeline level GM + komputasi besar + (diduga) data eksternal; bukan sesuatu yg bisa dikejar dgn 2xT4/minggu.
- **Aturan penting**: skor akhir = privat LB pada "private test set" TERPISAH dari public test set (Rules); Data page: "~1300 studi di test set"; prevalensi bisa beda antar set -> pergeseran peringkat di dataran 0,950 (252 tim) besar. Pilih final dgn alasan ketahanan (ensembel lebih banyak model, notebook cepat), bukan selisih 0,001 publik. Bila tak memilih, Kaggle otomatis ambil 2 terbaik publik.
  Catatan runtime: Apex ≈30 mnt; stack d4/replika ≈7 jam utk ~1.300 studi -> bila set privat diganti lebih besar saat akhir, notebook 7 jam berisiko lewat batas 9 jam; sertakan setidaknya satu notebook cepat di 2 pilihan final.
- **Tim**: batas penggabungan 15 Okt (syarat: total submisi gabungan <= batas harian*hari ≈385; kita 102). Jalur sah menambah keluasan model; mitra setara (tim <=2 anggota, 0,950-0,957) ada banyak, tetapi insentif mereka kecil; sekadar opsi bila user mau menghubungi.
- **Submisi Apex 0,950 dibuat rekan tim `nxtirfan`** (halaman Submissions), bukan oleh user/agen.
- Hub Models Kaggle: tidak ada ckpt kuat baru (hanya baseline 2.5D, ConvNeXt-L, 'Pillar-0 BreastMRI fine-tune'). OrthoDiffusion dipertimbangkan, tidak dikerjakan (lihat 37c).

- **Job TPU anak-OAI (9 Okt)**: setelah ±6 jam antre, berjalan 5 menit lalu MATI ("Kernel died") di `native_batch_norm_backward` XLA saat anak `a` mulai dilatih (BN dibekukan `eval()` + backward di XLA = crash fatal). Sebelum itu: induk di TPU = gold-58 **0,9212** (≈ Colab GPU 0,9214 → lintas-perangkat konsisten). Garis anak sudah ditutup, tidak diperbaiki.
- **Push kernel GPU 9 Okt 07:34**: `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached` → commit-run notebook probe r224 TIDAK mungkin di akun utama sebelum reset 10 Okt 07:00 WIB (jalan keluar sah: kuota `nxtirfan`, lihat docs/PAKET_REKAN_TIM.md).

## 37e. KOREKSI kepatuhan OAI (9 Okt malam)
Host (thread 741819, 21 Sep): OAI sbg data eksternal TIDAK melanggar aturan bila dapat diakses semua peserta tanpa tinjauan/persetujuan institusional atau tim hukum, dan gratis (host tak mengontrol syarat data OAI). Peserta peringkat 185 (thread 743416): daftar NDA sbg individu, Login.gov+email, disetujui seketika, grup OAI "Institutional Approval Required: No".
=> Koreksi §37b/penilaian sebelumnya ("akses butuh penandatangan institusi / abu-abu"): jalur OAI TERBUKA dan sah menurut syarat host (risiko tersisa: blokir negara, syarat data NDA tentang redistribusi -> jangan unggah data mentah ke Kaggle, hanya bobot). Ini mengubah rencana perak: model OAI buatan sendiri + ensembel dgn ckpt nartaa = pengungkit nyata (+0,003-0,006).

## 38. Riset mendalam menuju 0,951+ (10 Okt 02:00–03:00)
- Catatan penuh: `research_notes/Cara menembus 0951 perak/` (forensik_0951.md, inference_gratis_0001.md, arm_baru_realistis.md); sintesis: `reports/Cara menembus 0951 perak.md`.
- **Pemisah 0,950 vs ≥0,951 = PSEUDO-LABELING** (thread 735304, 9 Okt): colum2131 (LB 0,958; single ConvNeXt-T 0,953 pseudo; "no external data used yet"), azamat1ch (LB 0,960, rank 20, 12 submisi; single 5-fold 0,959), hengck +5 ("model is not the key. it is your labels"), Raymond Yuen (pseudo+blend 0,949→0,958).
- Papan 10 Okt 01:25 WIB (5.606 tim): ≥0,951=296, ≥0,952=183 → blok 0,951-persis (113 tim, rank 184–296) melebihi kapasitas perak (~279) → 0,951 = lotre, **0,952 = target aman**.
- Tertutup/ditolak (bukti 10 Okt): heliosli S6 mati permanen (manifest hilang), MaVIT hengck gold-58 0,9202, d5e-goldw (bobot fit di emas = overfit), d5f (FT on-the-fly label-LLM = placebo), SWA manual/seed-avg (nartaa rilis cuma 2 ckpt), `raptor-knee-arms`/`cnn336` (arm lemah + kemungkinan bocor gold), resep 746792 (identik stack kita).
- Hipotesis gratis (inference): H1 r224+mirror, H2 reader+mirror, H3 mirror sbg arm terpisah; gate sah gold-58: Δ≥+0,005 & paired-bootstrap CI95>0; tuned>5 iter → +0,007; <0,003 = noise.
- Jalur utama: **pseudo-label self-training** di 4.407 train (bahan: cache volcache-z, ckpt induk, OOF machinery; ≈10–25 GPU-jam).
- Alat baru: **baca forum via CLI** `kaggle forums topics show <topicId> --page-size 200` (743416 masih BELUM dijawab host per 10 Okt).



## 39. Notebook publik ber-skor tertinggi & bedah kode heliosli (10 Okt 2026)
- **Skor per notebook** (halaman Code → urut skor; CLI tak menampilkan skor): heliosli S6 0,954; versia-7, `kiyoshiohno/template`, blend-gold, blend3, `jenillangaliya/dual-stage-sota-ensemble` 0,951; DINOsaur V5, Velciraptor V2, Apex dan sisanya 0,950. Klaim "0,954" (rabari9999 `lb-0-954-rank-127-silver`, emresariduman `sota-0-954-breakthrough`, sameerk `d5h-pub0954`) = kode identik (215 KB, 17 sel; dataset goodpjw+nartaa) → skor sebenarnya 0,950.
- **kiyoshiohno/template** = peluncur 3 sel: memuat `rsna_knee_pipeline.py` + `rsna_knee_manifest.json` (bundle_id `rsna-knee-local-logic`) dari dataset PRIVAT; sumber publik: goodpjw, nartaa, heliosli OF fold0 (+2 dataset kosong/privat). **jenillangaliya**: sumber `['', '', nartaa]` (dataset privat; "ConvNeXt-S 4-Fold anatomically anchored" = bundel heliosli yg ditarik).
- **Ekstraksi sumber heliosli**: notebook V2 (`matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed`, klon S5) sel 2 (143 KB) memuat `SOURCES = {...}` (7 berkas) → `ast.literal_eval` pada Assign `SOURCES`: `px_cache.py` 20 KB, `compact_train.py` 45 KB, `compact_infer.py` 26 KB, `probe_infer.py` 17 KB, `probe_dual.py` 8 KB, `knee_anchor_bank_v2.py` 15 KB, `side_common.py` 1,6 KB. `DEADLINE_HOURS = 8.3`, `ARM='E_lrprior'`, `BUNDLE_ID='c7ebab1eb953f1a6'` (S5).
- **compact_train.py**: jendela 2.5D (3 irisan bertetangga = kanal) → encoder (resnet34 default; `convnext_small` utk E_lrprior; opsi `ofl_l` ViT-L dgn `--freeze-blocks 16 --llrd`) → perhatian per kelas atas jendela → 12 logit; K-fold atas pool NON-emas, "never reads official labels; gold never trained on"; target dari CSV laporan sendiri + `--state-targets/--state-spec` (report-state + vektor prior, `--state-weight 0,5`); arm S_strict/P_presence/E_prior/E_lrprior; aug `v2` (rotasi/skala/gamma, tanpa flip); `--phys-mm`, `--rare-boost`, EMA.
- **probe_dual.py**: urut studi menurut jumlah DICOM (−biaya) dan intersep antar-GPU; pekerja `--deadline` + keluaran bertahap; env `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`; studi gagal → rerata kelas studi lain (kita: rank Apex); `--max-fallback-fraction 0,03`; resi decode/render/gpu per studi.
- **Kompilator aturan** `cloud_code/training/criteria.py` (v7.4, regex multibahasa EN/ES/PT/FR/DE/TR/ZH; Fact(presence, criterion, certainty, reason, evidence)); modul pendukung `report_language_recovery`, `report_scope_recovery`, `report_consistency`, `verified_report_language`, `clinical_factors`, `criterion_trace`, `report_assertion_recovery`, `report_review_recovery` (stub `common.py` tanpa torch). 36 dtk utk 4.407 laporan. **AUC rata-rata vs emas-58 = 0,784** (ACL .955, MCL .907, Lat Menisc .878, Baker's .830, Medial Menisc .821, Fracture .799, Lat OA .787, PF OA .767, Effusion .719, Medial OA .689, Contusion .644, Synovitis .618); skor = criterion (1/0) lalu presence, "unknown" 0,4; liputan criterion≠None 9–53 %. → bukan label mentah yg layak; kekuatan heliosli = prior + model.

## 40. Model peluang perak (simulasi) — `silver_prob_sim.py` (10 Okt)
- Data: papan 10 Okt 19:37 WIB (5.666 tim; top-5 % = 283). Model: privat_i = pub_i − s·(pub_i − 0,945) + N(0, σ); tim kita +δ (skill sejati). Rentang s ∈ {0,3; 0,5; 0,7}, σ ∈ {0,003; 0,004; 0,005}.
- Hasil (tim publik 0,950): δ=0 → 14–20 %; +0,001 → 19–28 %; +0,002 → 26–39 %; +0,003 → 33–53 %; +0,004 → 40–67 %; +0,006 → 55–85 %; +0,008 → 70–96 %. Tim publik X (merger): 0,952 → 17–30 %; 0,954 → 21–51 %; 0,956 → 23–67 %; 0,958 → 28–82 %; 0,960 → 30–92 %.
- **Tweedie/empirical-Bayes TIDAK valid**: dataran 0,950 berisi 564 tim dgn prediksi identik (salinan notebook) → lonjakan densitas membuat turunan log-densitas tak bermakna (menghasilkan penyusutan absurd). Regresi linear + noise = pendekatan paling jujur; angka = arah, bukan ramalan.
- Implikasi: perbaikan +0,001 hampir tak berarti; hanya +0,004 atau masuk tim ≥0,956 yg menggeser peluang berarti; 100 % mustahil.

## 41. OrthoFoundation-L fold0: probe, timeout, uji skala (10 Okt)
- `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (837 file; `weights.part000–009` = best.pt 1.223.580.827 B sha `4f424cc7…`; `anatomy.pt`; `dependencies/kneexnet/*`; `cloud_code/training/*` ± 800 file; wheel timm 1.0.25/pydicom 3.0.1). `model_config`: `vit_large_patch16_dinov3_qkvb`, 384 px, hidden 256, `anatomy_pool`, `mcl_joint`, `partial_joint`, `local_oa_partial`, `oa_binding bound`, `external_mcl false`; train: 16 ep, batch1×accum24, 8 jendela/seri (eval 16), `views global_local`; epoch 7 EMA; val AUC lemah 0,93175; `label_source_policy own_reports_and_official_gold_only_v1`; **48/58 emas di train_ids fold0** (3.555 id).
- Probe (`rsna-knee-of-probe`, v1→v5) perbaikan: (1) `cloud_code` ditulis ke /kaggle/working (kode menulis `training/output/yolo_config`); (2) `anatomy.pt` & `kneexnet/` disalin (symlink di-resolve → path salah); (3) yaml `ultralytics` 8.4.166 tak ikut paket vendored (hanya 635 .py) → dataset privat `rsna-knee-ultralytics-cfg` dari wheel PyPI (132 berkas non-.py, 526 KB; akar dataset = isi folder `ultralytics/`). Hasil: stub 3 studi 127 dtk; emas-58 58 studi 1.571 dtk (27 dtk/studi, 1 proses); AUC 0,9535 (terkontaminasi). Waktu: native cache ±3 dtk/studi, anatomi ±70–75 %, ViT ±10–15 %.
- **57028008** (Apex 0,70 + OF 0,30) = Notebook Timeout (9 j). Uji skala (`build_scale_of.py`, akar simbolik 1.322 studi train): pemetaan `test_series` = 84–95 dtk/panggilan (248.514 berkas); disk ±9,6–10 GB per 30 studi (≈330 MB/studi; "No space left on device"/"Not enough free space"); 2 proses: 45–95 dtk/studi/proses; 4 proses: 56–62 dtk/studi/proses (≈15 dtk/studi agregat ≈5,5 j utk 1.322); GPU sering 0 % (CPU 4 vCPU hambatan), load ≈2,4–3.
- Perbaikan `of_stage.py` v2: potongan 8 studi + akar-data kecil per potongan (symlink), 4 proses (2/GPU), penjaga disk (`statvfs`), env thread=1 + `torch.set_num_threads`→1 + `cv2.setNumThreads(1)`, prioritas studi (disagreement nartaa vs reader), keluaran bertahap, berhenti lunak +6,6 j / bunuh +7,0 j, liputan minimum 15 %, reuso `AnatomyRuntime` antar-potongan (patch `infer.py`: `runtime=g['_RT']`, hemat ≈45 dtk/potongan), pembersihan `/kaggle/working` di akhir. Uji `ofscale3` (tanpa reuso) & `ofscale4` (reuso) → throughput agregat → gerbang G0.
- **Hasil uji skala v2 (10 Okt 20:15)**: `ofscale3` (4 proses, workers 1, thread 1, potongan 8, tanpa reuso): 104 studi dlm 0,58 j (setup ±0,04 j), 0 gagal, disk bebas 19,5→14,8 GB → **193,9 studi/jam agregat → 6,8 j utk 1.322** (interval akhir 224/jam ≈5,9 j). `ofscale4` (reuso `AnatomyRuntime`): 88 studi, 150/jam → 8,8 j → reuso TIDAK membantu (dimatikan; `__REUSE__`=0). Stub `apexof2` v3: GPU bebas [15,1; 15,4] GB sebelum & sesudah pembersihan (Apex tak menyisakan memori), 4 pekerja OK, blend sukses.

## 42. Thread 745861 (hengck23, 10 Okt 08:09 UTC) — resep label terbaru dari GM (tak ada pengetahuan baru)
- hengck23 (rank 1029 di papan; skor tim 0,949): "1) mulai dari label Claude Opus 4.5 (dataset sendiri); 2) latih model MRI dan model teks pada 4,3 rb label non-emas; 3) blend → CV lokal emas-58 0,938". Prioritas label bila memakai data eksternal: anotasi Kaggle (skor tertinggi) > anotasi eksternal > negatif (skor terendah). Tips lain: label termahal utk sampel paling tak pasti; "budget control has become part of the competition".
- Kesimpulan: sama dgn resep kita (fusi guru+OOF, emas ±0,935–0,938); tak ada celah baru. Host tetap diam di 743416 (35 komentar).

## 43. Dataset privat notebook papan atas + riset Plan B (10 Okt malam)
- **Pertanyaan user**: "kebanyakan yang top silver itu pakai dataset privat, isinya apa?" Jawaban terverifikasi dari KODE notebook publik 0,951 (jenillangaliya `dual-stage-sota-ensemble`, kiyoshiohno `template`, blend3/versia-7): konstanta `C_SLUG='rsna-knee-compact-v1'`, `W_SLUG='rsna-knee-ofl-ssl-weights'`, `BUNDLE_ID='c7ebab1eb953f1a6'`, `CODE_SLUG='rsna-orthofoundation-fold0-epoch7-20261004'`; `dataset_sources` mereka `['', '', nartaa]` → dua sumber kosong = dataset privat (salinan privat bundel heliosli yang ditarik 9 Okt 13:50 UTC setelah ±1.975 unduhan). Dataset publik `heliosli/rsna-knee-compact-v1` kini 296 byte (kosong).
- **Isi bundel compact (dari kode heliosli)**: `manifest.json` (backbone, size/crop 352, slices, phys_mm 3,5, views, eval_stride, models, `state_spec`, `side_model`+ambang, `bundle_id`), 4 ckpt fold ConvNeXt-Small 2.5D `E_lrprior_s0_f0..f3.pt`, `state_spec` (prior report-state), model-sisi (verifikasi lateralitas), `pca.npz`/`feature_norm.npz`/head probe. `rsna-knee-ofl-ssl-weights` = bobot OrthoFoundation-L SSL (potongan ±1,2 GB; bobot dasar OF-L PUBLIK di omkars5/orthofoundation, pritom2357, tsuyu122, leogamertetudo). Bundel logika: `rsna_knee_pipeline.py` + manifest (`rsna-knee-local-logic`). Takakura: model 384 px laterality-terkoreksi (`takakura-code-competition-inference-v1`).
- **Tak diketahui**: isi dataset privat tim 0,955–0,964 (hanya ada inferensi: ckpt fold sendiri, tabel pseudo/soft-label, mungkin backbone berpengetahuan eksternal). Jangan klaim lebih.
- **Ledger publik `prvsiyan/rsna-knee-2026-experiment-ledger`** (CC BY 4.0; 90 submisi satu tim, terbaik 0,949; dirilis 10 Okt 13:14 UTC): mirror-reader (resep nartaa) 0,941→0,949 (+0,008); + reader ConvNeXt 2.5D 0,70/0,30 = 0,948; + BK-A 0,55/0,45 = 0,947; MaVIT 288 solo 0,935; overlay OrthoFoundation Lateral-OA / OrthoDiffusion pada DINOsaur 0,935 vs kontrol 0,936 (−0,001); hanya "depth-index alignment" (+0,008) dan RadImageNet 20 % (+0,007) yang menaikkan pada pipeline lama. Kesimpulan: menguatkan "pengetahuan, bukan varians" — overlay pretrain generik tak membantu; OF fold0 heliosli (fine-tune pada label kompetisi) belum tentu mengikuti pola ini → tunggu 57044355.
- **Papan (21:43 WIB, 5.673 tim)**: ≥0,951 = 352; ≥0,952 = 221; =0,950 = 582 (rank 353–934); kita rank 551; perak (5 %) ≈ rank 284 (0,951); perunggu (10 %) ≈ rank 567 (0,950). Garis perunggu naik 0,944 (6 Okt) → 0,947 (8 Okt) → 0,950 (10 Okt). Jalur bersih 0,944 = rank ≈1.387 (bukan perunggu).
- **Simulasi (silver_prob_sim.py)**: P(perunggu | Apex 0,950) = 25–41 % (BUKAN 85–95 % — klaim lama kedaluwarsa sejak dataran 0,950 membengkak). Dua final dgn ρ galat 0,9→0,4: perak 17→21–29 %, perunggu 34→41–51 %.
- **Penyaringan mitra merger** (papan 21:43): Kelompok A (pemegang bundel; heliosli 0,954/29 sub, xianhan 0,954/26, Netram Faran 0,954/63, Hackers 0,952/14, samanyu1808 0,951/6, evgendvorkin 0,951/19, kiyoc 0,951/14) dan Kelompok B (35 tim ≥0,953, ≤2 anggota, ≤30 submisi; mis. MURAD 0,962/18, NMarkS 0,961/30, UEHA 0,957/22, DoyMed 0,957/21, madmax0404 0,957/14, LP 0,956/18) → `docs/KANDIDAT_MERGER.md` §3b. Batas submisi gabungan 5.c: 330 (10 Okt) – 355 (15 Okt) − 107 kita = mitra ≤ ±220–245.
- **Scan 22:05**: `emresariduman/rsna-knee-gold-apex-v7` = Apex + kaki c224 (kode `infer224.py` ala dreaddevelopment; dataset sama dgn Apex) — tak ada pengetahuan baru; ditutup.
- Rencana lengkap: `docs/PLAN_B.md`.

## 45. Riset diskusi + 16 kernel best-score (11 Okt pagi, sesi OpenCode)
- Daftar topik via **`kaggle competitions topics list`** (BERFUNGSI; 'CLI 403' hanya utk `forums topics list`) + baca 18 thread (`forums topics show --page-size 200`) + tarik/anilisa 16 kernel best-score. Laporan penuh: `reports/Riset diskusi + best-score code 11 Okt.md`.
- **`heliosli/one-click-955` (klaim ≈0,955) TIDAK bisa dipublik**: Raptor 0,7 + compact 0,3 SAMA dgn S6, tapi bundel **privat `rsna-knee-compact-r3cnx384`** (55bd00b3, infer crop 384; metadata 403). Delta vs S6 = dataset, bukan kode. Thread 748141 sudah DIHAPUS; notebook kini mencantumkan pengakuan OAI/NDA. Pantau publikasi bundel R3.
- **Kode**: rabari9999 0,954 & jcgweb probit954 = identik — probit-quantile fusion (rank pct→ndtri→blend→ndtr→jitter), post-hoc gratis; kozykappa = protokol gating emas (bootstrap 750×, gain ≥0,003, P≥0,85, q20>−0,004) sbg kerangka uji; romantamrazov = residual Synovitis←Effusion(.28)+Baker; goodpjw = bobot per-finding; prvsiyan = kontrol negatif (mirror+reader 0,948 < solo 0,949).
- **Diskusi**: 748111 tak ada bocoran (Optimo kagum); pseudo-label OOF 50/50 = lever 0,951+ (colum2131 0,953, azamat1ch 0,959, Raymond 0,938→0,94+, Archit: isi sel sunyi dgn OOF); validator = OOF-vs-weak-label (r=0,87), emas-58 rusak (r=0,66); 8 set label publik tak beda → steven_v4/konsensus + imputasi hanya Synovitis←Effusion; slice order by IPP = +0,028 (1 fold) → verifikasi di kernel kita; Raptor: window count lever (42→62=+0,003), SWA≈0; bf16 di T4 = −0,14 diam-diam → wajib fp16; DINOv3 = tanpa ruling + lebih jelek dari V2; jangan tune bobot per-finding di publik split.
- **Implikasi**: Fase B pseudo-label prioritas naik (guru S6/Apex → murid ConvNeXt-T 2.5D, 50/50 dgn steven_v4); cek IPP+fp16 gratis; uji post-hoc dgn protokol kozykappa; pantau bundel R3 heliosli.

## 44. S6 heliosli DIPULIHKAN + kebun-binatang bundel + forum "best score" (11 Okt 2026 dini hari, sesi Claude Code)
- **Fakta (02:50 WIB)**: heliosli memulihkan `heliosli/rsna-knee-compact-v1` (V13 = V10, bundel `209a32c8b50a6b1c`, 887 MB) dan menerbitkan ulang notebook S6 publik `heliosli/dataset-restored-i-d-been-told-it-wasn-t-allowed` ("satu klik = 0,954"). Thread 748141 (−2 suara): "feel free to blend my model with yours". User menyalin/mensubmit S6 (57051680, 02:56 WIB, PENDING).
- **S6 = Raptor 0,7 (nartaa 0,949 mirror) + compact 0,3** (ConvNeXt-Small in22k, 4 fold, 384/crop 352, 16 epoch, `views v4`, `eval_stride 1`, 3 irisan, `phys_mm 3.5`, side classifier; target report-state 'opusv11b' + prior; dilatih di D:\ dan AutoDL; `gold_in_training 0`). Rank-persentil per temuan, bobot tetap "sebelum run apa pun". Tahap 2 `probe_dual.py` = 2 pekerja (1/GPU, loader 2), decode via `training.native_cache` (kode yg sama dgn OF); kontroler MENOLAK (error, tanpa skor) bila >3 % fallback (`--max-fallback-fraction`); DEADLINE 8,3 j sejak NB_T0.
- **KOREKSI PREMIS (kesalahan saya di dokumen sebelumnya)**: S6 dan semua notebook heliosli TIDAK memakai bobot `best.pt` OrthoFoundation fold0 — hanya KODE dataset OF (`cloud_code`, `anatomy.pt`, wheels). Resep "Raptor 0,7 + X 0,3 = 0,951–0,954" adalah **Raptor + COMPACT** (blend-gold, bundel S5 `c7ebab1e` = 0,951; S6 `209a32c8` = 0,954), BUKAN Raptor + OF. Nilai LB OF fold0 (ViT-L; `validation_macro_auc_weak_reference` 0,9318 pada epoch 7) belum pernah diukur siapa pun; prior direvisi dari ±35 % menjadi ≤20 % untuk menambah ≥+0,001 di atas S6.
- **Kebun-binatang bundel (semua versi publik lagi; `kagglehub ... /versions/N` mengunduh manifest)**: V1 204ac38d resnet34 uji; V2 8f398157 r34 lean 5 fold; V3 1e93c0c5 r34 opusv11b 5 fold; **V4 ac0b1332 `ofl_ft8_opusv11b_lr` (fine-tune OrthoFoundation-L 8 blok, 288/256, 5 fold)**; V5 84076237 cnx_s1; V6 f26e2c7b cnx_s3; **V7 c7ebab1e cnx_s5 (0,951 di blend-gold; side_cls)**; **V8 305a85d1 dino_s1 (`vit_base_patch16_dinov3.lvd1689m` 384/352 v4, 4 fold)**; **V9 a5e76db5 `s5_dino_s1_dual` (convnext_small + models2 dino, bobot per kelas di bundel)**; V10 = V13 209a32c8 cnx_s6 (0,954). Kandidat bagging (biaya tahap compact ≈3,5–4,5 j per bundel pada stride 1).
- **Profil tahap 2 (sonda `rsna-knee-s6probe`, 160 studi pseudo-uji; sebagian)**: 20/80 per pekerja dlm ±500–530 dtk → ±25–27 dtk/studi/pekerja (decode ±14 dtk, render ±4 dtk, **GPU ±21 dtk** per studi) → tahap compact GPU-bound (T4), ±3,7–4,6 j untuk 1.322 studi dgn 2 pekerja; Raptor ±0,5 j → S6 lengkap ±4,5–5 j (hasil klon user ±07:30–08:30 WIB).
- **Forum — "Best single-model score" (735304, 128 komentar)**: azamat1ch (MURAD, rank 10, 0,962, 19 submisi): arsitektur tunggal 5-fold ConvNeXt-T **0,959**, EffNetV2-S 0,956, CoAtNet 0,952; colum2131 (rank ±45): ConvNeXt-T 2.5D 0,953, DINOv2 ViT-S 2.5D 0,951, ensembel 0,955 — keduanya dgn PSEUDO-LABEL; Tanuki_boosting 0,955 single 5-fold; CoolinLai (Attention Is All You Knee 0,961) ResNet-50 224 px 5-fold 0,954; hengck23: "model is not the key. it is your labels"; Chris Deotte: dgn pseudo-label model tunggal apa pun menyamai ensembel terbaik (CV); Raymond Yuen: beberapa sumber weak label → model guru → pseudo-label → model akhir (biaya API < US$5); Archit Konde: OOF sudah lebih dekat ke emas daripada label hasil ekstraksi → label = hambatan; Charles Savas: perna eksternal korelasi 0,83 = +0,004 publik. **Pemisah papan atas (0,959–0,964) = guru→pseudo-label→murid ConvNeXt-T 2.5D 5-fold + label kaya**, bukan arsitektur.
- **Forum — validasi (748011, colum2131)**: AUC OOF pada ~4.300 label LLM berkorelasi r=0,87 dgn LB publik; emas-58 hanya r=0,66; datnt114: LB 0,961 dgn OOF 5-fold 0,937 dan emas-58 0,938. Catatan: trik "kirim 1−p" memungkinkan tim kuat bersembunyi di papan (yu4u/tattaka di 0,949) → garis perak privat bisa lebih tinggi dari yang tampak.
- **Forum — kepatuhan**: 744056 (Shalini S): 2.6.b ("external data and models acceptable unless prohibited by the Host"), 2.8 (pemenang hadiah menyerahkan kode latih/inferensi/bobot), 3.14.a ("own original work") — host belum menjawab; zhao_hapi (10 Okt) menanyakan bobot publik yg ditarik (compact). 747455 (bobot OAI) & 743416 tanpa jawaban host. Medali tak bergantung pada kelayakan hadiah, tetapi risiko OAI bersifat kolektif (seluruh klaster Raptor).
- **Forum — lain-lain**: 748111 "Single sub 0.960! Wow!" (Optimo; hengck23 "one-shot learning"); 743374 (Prateek Grover, 0,950, mencari rekan): I/O DICOM — 1 pembaca 59 MB/dtk, 16 thread 372 MB/dtk di /kaggle/input; header DICOM langsung 0,307→0,058 ms; piksel via offset 1,22→0,33 ms; persentil pada sub-sampel; fp16 tanpa autocast 2,08× di T4; Nikita Ageev (0,949, solo, label LLM utk 4.407 laporan) mencari tim sebelum 15 Okt; 747658 benchmark ekstraksi LLM (Claude 4.5 AUC 0,8919 vs emas).
- **Notebook publik baru**: `jenillangaliya/...-0-952-tri-hybrid-multi-scale-mil` (Raptor 384 + compact + r224 crop; 0,952 < S6); `rabari9999/...-lb-0-954-rank-127-silver` dan salinannya (`jcgweb/probit954`, `sameerk2004/sil-d5l-probit`): "Tri-Reader 0,954" dgn masukan publik saja (goodpjw + nartaa) = keluarga Apex+r224 (diuji 0,950; 0,954 milik tim itu = S6 privat) — JANGAN dipakai; `romantamrazov/dinosaur-*` (DINOsaur; ±0,950).
- **Papan (02:59 WIB, 5.699 tim)**: 31 ≥0,960; 82 ≥0,956; 141 ≥0,954; 229 ≥0,952; 367 ≥0,951; 618 di 0,950; rank kita 599 (0,950) — di bawah garis perunggu publik (rank 570, 0,950); perak (5 %) rank 285 (0,951). Dataran 0,954 akan membengkak karena S6 hidup lagi.
- **Implikasi**: (1) S6 = dasar; klon berimpit di ±0,954 dan urutan privat = undian + selisih kecil; (2) perna dgn informasi baru bernilai, tetapi OF/reader/MaVIT/r224 memakai label yg sama → ekspektasi ≤ +0,001; (3) tuas besar tetap merger/label kaya; (4) final bersih (tanpa Raptor/OAI: compact + reader) = asuransi bila host memutuskan soal OAI.

## 46. Scrape mendalam seluruh kompetisi (12 Okt 2026, sesi Claude Code) — sintesis di `reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md`
- **Cakupan**: 10 halaman resmi (rules, evaluation, data-description, timeline, prizes, efficiency…), papan penuh 5.807 tim (06:28 UTC) + 07:29 UTC, 182 topik (seluruh komentar; postingan pembuka lewat `kaggle competitions topic-messages <slug> <id> -n -1`), indeks 1.100 notebook per 6 urutan + kode 169 notebook, 233 dataset + 157 model. Catatan CLI 2.2.4: `competitions pages <slug> list <slug> --content` (slug dua kali).
- **Label uji** (host 733826/733491/733343): dari citra, 2 radiolog MSK + pengadil, ambang tinggi (efusi/Baker = sedang–besar; OA >50 % tebal ≥1 cm; ragu = negatif); laporan klinis over-call → label lemah harus dikalibrasi ke definisi ini.
- **Papan**: 8 Okt → 12 Okt: garis perak 0,950 → 0,954; emas 0,959 → 0,961/0,962; ≥0,955 73 → 182; dataran 0,954 = 604 tim (529 submit dlm 24 j). Simulator `silver_prob_sim.py` (papan baru): P(perak) 0,954 ≈ 16–25 %, 0,956 ≈ 19–42 %, 0,958 ≈ 26–60 %, 0,960 ≈ 27–76 %.
- **Kode**: notebook #1 = S6-eq heliosli (0,5/0,5, `pct_on` sadar-gagal, bundel `r6cnxfsss` = compact-v1 V13); velci-kcd25-20 (S6 + KCD25 privat 0,2) = tim 0,955; goodpjw S6 + reader = 0,954 (emas-58 0,9313 → 0,9331). Raptor nartaa: CoAtNet-rmlp-2 384, 96 irisan (26/22/18/12/18), 140 mm, 94 jendela, atensi per temuan; OAI 2.399 lutut (PF OA/Lat OA/Synovitis dari MOAKS, sisanya dimasker) tercatat +0,005; cermin anatomis +0,001.
- **Forum (teknik)**: target lunak, "tidak disebut" ≠ negatif; pseudo-label OOF dicampur ±50/50 (jangan ganti total; in-fold = membeo label); student > teacher terutama Effusion (0,835 → 0,986 emas, colum2131); Synovitis ← Effusion (0,7115 vs 0,678); urut IPP (+0,028 pada satu baseline); 224–288 px cukup; RadImageNet butuh kontrak kanal (irisan tengah ×3); bf16 di T4 = −0,14 diam-diam; LLM mahal ≈ murah; 8 set label publik tak beda jelas.
- **Klaim tunggal papan atas**: ConvNeXt-T 5-fold 0,959, EffNetV2-S 0,956, CoAtNet 0,952 (azamat1ch, rank 4, tanpa data eksternal); ConvNeXt-T 0,953 + DINOv2-S 0,951 → 0,955 (colum2131); ResNet-50 5-fold 224 px 0,954 (CoolinLai); Qwen3.5-2B LoRA satu fold 0,950; datnt114 LB 0,961 dgn OOF 0,937.
- **Kode latih compact heliosli** (tertanam di S6): fold = `int(sha1(uid)[:8],16) % 4` atas 4.344 non-gold; `gold_in_training 0`; opsi `--holdout-fold` untuk putaran pseudo-label; init `convnext_small_in22k_ft_in1k`, 16 epoch, lr 1e-4/1e-3, wd 0,05, 176 jendela latih, crop 352, `freeze_blocks 16`, target `probe_targets_opusv11b_rr3_plv21_cdr`. → guru OOF compact (`build_compact_oof.py`).
- **Aset baru**: `heliosli/rsna-knee-compact-r6cnxfsss` (742 unduhan), `jessky/rsna-knee-local-cv-cnx50` (5 ckpt compact `d50` + kode mainline, label `labels_distill50` dgn guru OOF eksternal, gold dikeluarkan), `gperdrizet/…pseudolabeled-training-data` (label + agreement + evidence), `shogomiyawaki/rsna-knee-llm-labels`; hub Models hampir kosong untuk tugas ini.
- **Implikasi**: rencana `STRATEGI_0960.md` — basis S6-eq; murid ConvNeXt-T/EffNetV2-S 2.5D dgn label `0,5 konsensus + 0,5 OOF-compact`, putaran-2 dgn OOF murid; bobot blend tetap sebelum submit; jangan menyetel di papan publik.

## Catatan status terkini — 19 Okt 2026 02:45 WIB
Status operasional pusat ada di `MEMORY.md`, `HANDOFF.md` §0, dan rencana aktif di `STRATEGI_PERAK.md` §0; riset lengkap ada di `reports/Riset 19 Okt dini hari - evaluasi silver dan murid r4.md`. Papan publik terakhir: 0,958 rank 467/6.187 (cutoff top 5% publik rank 309; medali ditentukan split privat 70%). Pada cek API 02:44, `g6c`, `g12`, `g13` masih PENDING; dua kernel training murid r4 di alver, di-requeue v1 pukul 02:42, masih QUEUED. Timer memantau keduanya. Kuota main/alver/nxtirfan tetap 19,87/17,81/4,62 jam. Catatan historis di dokumen ini tetap sesuai tanggal asal; gunakan dokumen aktif untuk keputusan sekarang.
