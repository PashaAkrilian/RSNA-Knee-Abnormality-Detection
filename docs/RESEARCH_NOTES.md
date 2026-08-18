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
