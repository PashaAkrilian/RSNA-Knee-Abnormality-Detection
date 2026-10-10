# Strategi Meraih Perak RSNA Knee Abnormality Detection — Status Aturan & Pemilihan 2 Submisi Final

> **ARSIP/STATUS 10 Okt 21:30 WIB (sesi Claude Code)**: dokumen riset 9–10 Okt (OpenCode). Rencana Fase A/B (mirror/pseudo-label) DIGANTI oleh `STRATEGI_PERAK.md` (merger + Apex+OrthoFoundation + finals + kepatuhan). Fakta terbaru: `MEMORY.md`; riset terbaru: `reports/Ringkasan riset 10 Okt (sesi Claude Code).md`.


Konteks: 9 Okt 2026. Tim posisi 0,950 (rank 563/5.584). Deadline final 22 Okt 2026 23:59 UTC.
Semua sumber publik; data papan & thread diakses langsung 9 Okt 2026 (Kaggle CLI + API internal Kaggle).

---

## 1. Status thread host: 743416, 741819, 733965 (per 9 Okt 2026)

### Takeaway
Host (Po-Hao "Howard" Chen / @javaduke95, authorType=HOST) sudah menjawab thread **741819 (OAI, 21 Sep)** dan **733965 (LLM, sticky, 9 Agu + follow-up 27 Agu)**, tetapi thread **743416 (daftar dataset eksternal MRNet/fastMRI/OAI/KneeMRI, 26 Sep) BELUM dijawab host sama sekali** per 9 Okt 2026 — 30 pesan semuanya dari peserta.

### Cited Findings
**Thread 743416 — "Please clarify which external knee MRI datasets are allowed"** (OP: YYama/yosukeyama, 26 Sep 2026, 81 vote, 30 pesan):
- Verifikasi via API (`discussions.DiscussionsService/GetForumMessagesInTopic`, 9 Okt 2026): **0 pesan dengan authorType=HOST**. — [Discussion 743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- Percobaan ke-4 tanpa jawaban: *"4 days and still no answer. is that a yes?"* — madmax0404, 30 Sep 2026 — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- Tabel permintaan klarifikasi (YYama, 7 Okt 2026): KneeCoT = **"Prohibited (host ruling)"**; MRNet, SKM-TEA, fastMRI/fastMRI+, OAI, KneeMRI (CC BY-NC-ND), KneeXNet-2.5D = **"Unclear"**; meminta pin jawaban sebelum 22 Okt. — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- Permintaan yes/no eksplisit (Taeyang, 8 Okt 2026): OAI via NIMH Data Archive (login.gov + click-through), dan apakah **checkpoint publik yang dilatih sebagian dari OAI (Apache 2.0, dipakai notebook publik 0.949) boleh dipakai tim lain** — "before final selections on Oct 22". — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- Interpretasi peserta atas jawaban host 21 Sep (Tucker Arrants, di 741819): *"It is a no, as it requires institutional sign off. Virtually all external datasets that are remote..."* — [741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)

**Thread 741819 — "Is the OAI permitted as external data?"** (OP: osamurai, 17 Sep; 1 jawaban HOST 21 Sep 2026):
- Kutipan resmi host: *"Using OAI as an external dataset does not violate the RSNA Knee Abnormality Detection challenge rules **if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge**. We do not own or control the OAI dataset and cannot comment on whether this challenge complies OAI data-use terms from the dataset stewards' perspective."* — [Discussion 741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819), 21 Sep 2026
- Thread tersebut juga merujuk ruling KneeCoT (thread 734109): KneeCoT **tidak diizinkan** *"due to the required data agreement with dataset institution"*. — dikutip di [741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)

**Thread 733965 — "Use of Commercially Hosted LLMs"** (sticky; host = pembuat topik, 9 Agu 2026):
- Kutipan resmi: *"Use of commercially hosted LLMs and other external inference services is permitted, provided that the service and method of use otherwise comply with the Competition Rules... submitting Competition Data, including report text, to an external LLM or API for inference... will not, by itself, be considered prohibited PRIVATE SHARING... This is in accordance to Section 2.6.b, EXTERNAL DATA AND TOOLS."* — [Discussion 733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965), 9 Agu 2026
- Follow-up host 27 Agu 2026 (kutipan inti): *"Datasets that can be accessed through a **straightforward registration or click-through data-use agreement would generally satisfy** this requirement. By contrast, datasets requiring institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing... may present a meaningful accessibility barrier"* ; dan *"use of these datasets would be considered non-commercial; receipt of competition prize money does not, by itself, make their use a commercial sale."* — [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- Host 27 Agu 2026: label/fitur dari LLM (OpenAI dll.) boleh dipakai saat training **dan** inference: *"Yes. Just make sure you read some of the other discussion threads about what the model is provided as input at 'inference time.'"* ; dan *"You can use LLM API, such as those from OpenAI, to read the reports to generate the labels."* — [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- Catatan peserta soal batas host (Tucker Arrants, 30 Sep 2026, di 743416): *"the host has no reason to check solutions other than the prize eligible ones"* — hanya top-10 yang teraudit ketat. — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)

### Inferences
- Aturan main yang bisa dipegang sekarang: (a) dataset click-through + gratis = kemungkinan besar sah; (b) dataset yang butuh persetujuan institusi/legal = kemungkinan besar tidak sah (prinsip ruling KneeCoT + OAI); (c) OAI secara spesifik masih "conditional yes" tanpa jawaban pasti soal NDA/institutional sign-off — risiko tetap ada sampai host menjawab 743416.
- Karena 743416 belum dijawab menjelang deadline, keputusan submisi final sebaiknya **tidak bergantung** pada asumsi dataset eksternal yang belum terkonfirmasi (khususnya untuk slot final yang diandalkan ke private LB).

### Gaps
- Belum ada jawaban host di 743416 per 9 Okt 2026 (fakta negatif hasil verifikasi API langsung, bukan asumsi).
- Status pasti NDA-OAI (apakah "institutional sign-off" menurut host) belum diputuskan; peserta melampirkan screenshot balasan NDA (8 Okt) tapi host belum merespons.

---

## 2. Aturan multi-akun & tim (Rules page resmi)

### Takeaway
Rules resmi **melarang keras satu orang memakai 2 akun** (disangkal/didiskualifikasi), hanya boleh 1 tim, dan hanya 2 Final Submission yang dinilai.

### Cited Findings
Semua kutipan dari halaman Rules resmi, diakses 9 Okt 2026 via `kaggle competitions pages list --page-name rules` (CLI resmi):
- [Competition Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
- Pembuka Rules: *"**You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.**"*
- Section 3.5.a INDIVIDUAL ACCOUNT: *"You may make Submissions only under one, unique Kaggle.com account. **You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy.**"*
- Section 3.5.b TEAMS: *"you may **join or form only one Team**. Each Team member must be a single individual with a separate Kaggle account... Team membership may not exceed the Maximum Team Size."*
- Section 3.5.d PRIVATE SHARING: *"No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted."* (3.6.a: berbagi code antar-tim hanya sah jika terjadi merger.)
- Section 2.2 SUBMISSION LIMITS: *"a. You may submit a maximum of five (5) Submissions per day. b. **You may select up to two (2) Final Submissions for judging.**"*
- Section 2.1 TEAM LIMITS: max 5 anggota; merger boleh dengan syarat jumlah submission gabungan ≤ maksimum yang diizinkan saat Team Merger Deadline (*"number of Submissions per day multiplied by the number of days the competition has been running"*).
- Section 2.6 EXTERNAL DATA AND TOOLS: External Data harus *"publicly available and equally accessible to use by all Participants... at no cost"* ATAU memenuhi kriteria Reasonableness; *"their use must be 'reasonably accessible to all' and of 'minimal cost'"*.
- Section 3.7.a: *"potential winner(s) are determined solely by the leaderboard ranking on the **Private Leaderboard**... Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set."*
- Definisi 3.18.c: *"'Final Submission' is the Submission selected by the user, **or automatically selected by Kaggle in the event not selected by the user**"* — slot kosong diisi otomatis submisi terbaik.
- Metadata resmi kompetisi (Kaggle CompetitionService, 9 Okt 2026): `maxDailySubmissions=5`, `numScoredSubmissions=2`, `maxTeamSize=5`, `leaderboardPercentage=30`, `deadline=2026-10-22T23:59:00Z`, `teamMergerExplicitDeadline=2026-10-15T23:59:00Z`, `prohibitNewEntrantsExplicitDeadline=2026-10-15T23:59:00Z`, `maxGpuRuntimeMinutes=540` (9 jam), `onlyAllowKernelSubmissions=true`, metrik `Roc Auc Score`, `scoreTruncationNumDecimals=3`, hadiah USD 77.000 (10 leaderboard + 3 efficiency).
- Timeline resmi: Start 30 Jul 2026 (data di-enable 5 Agu), **Entry Deadline 15 Okt 2026**, **Team Merger Deadline 15 Okt 2026**, **Final Submission Deadline 22 Okt 2026**, Winners' Requirement 5 Nov 2026 — semua 23:59 UTC. — [Overview/Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)

### Inferences
- "Pakai 2 akun untuk submit 10 submisi/hari" = pelanggaran eksplisit → diskualifikasi; tidak ada ruang abu-abu.
- `leaderboardPercentage=30`: public LB ≈ 30% test set, private ≈ 70% — dasar statistik kenapa geser public→private wajar.

### Gaps
- Tidak ada bagian Rules yang secara khusus membahas "satu orang dengan 2 akun di berbagai kompetisi" selain kutipan di atas; hanya regulasi akun tunggal di kompetisi ini.

---

## 3. Strategi memilih 2 submisi final yang tahan public→private shift (post-mortem)

### Takeaway
Konsensus post-mortem: pilih 2 submisi dengan **peran berbeda** (satu terbaik di public LB/CV-OOF, satu dari CV/OOF yang jujur) dan **mekanisme kegagalan berbeda** (korelasi error rendah / beda lineage), jangan kejar selisih kecil public LB di papan yang padat; final score = **max dari 2 submisi di private LB**, jadi slot kedua adalah opsi gratis untuk lindung nilai.

### Cited Findings
**Mekanika pemilihan & auto-fill:**
- *"You pick two final submissions. Your better private score is what counts"* ; *"if you select only one of your two finals, the platform quietly fills the empty slot with your best-scoring unselected submission, which... would have silently inserted exactly the kind of model I was trying to avoid. The interface still reads '2 of 2'"* — verifikasi via API, bukan layar. — [Alan Scott Encinas, "The Hedge That Failed", 12 Agu 2026](https://alanscottencinas.com/kaggle-final-submission-selection/)
- *"selecting my two highest-public results from this audited window would have yielded zero"* — pasangan final yang dipilih manual (hedge rendah-public) menyelamatkan hasil. — [pengwang91, "111.29 Public, 0 Private", 6 Sep 2026](https://www.kaggle.com/writeups/pengwang91/111-29-public-0-private-measuring-two-kinds-of-d)
- Pemilihan manual wajib: *"Auto-selection would have picked our two best public submissions — both marker engines with private zero... Do not trust auto-selection by public score."* — [46th Place, AI Agent Security, 2026](https://kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/46th-place-solution-hedging-the-hidden-guardrail)
- Formula resmi umum Kaggle: satu sub terbaik di local validation + satu terbaik di public LB. — [DataCamp, "Winning a Kaggle Competition in Python"](https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/modeling?ex=12); pola sama: [Sudhanshu Tiwari playbook, 7 Mei 2026](https://uselessai.in/from-baseline-to-top-10-a-practical-kaggle-competition-playbook-702d1d9394f8) (*"Always select two final submissions. One that maximises your public LB score, and one that you trust based on your local CV. They should not be the same submission."*)

**Diversifikasi kegagalan (bukan cuma format):**
- *"For the final 2 models, pick very different models. Picking two very similar solutions means that your solutions either fail together or win together... You should not depend on the leaderboard score at all."* — [Chioka, "How to Select Your Final Models"](https://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)
- Hedge harus bebas dari lineage yang ditakuti: *"The check is not 'are these two things different from each other' but 'does my main position share anything with the thing that would hurt me.' Audit lineage and shared components, not just an error correlation."* — [Encinas, 12 Agu 2026](https://alanscottencinas.com/kaggle-final-submission-selection/)
- Dua slot = opsi gratis karena diambil maksimum: *"the second leg costs us nothing in expectation"* — [46th Place writeup](https://kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/46th-place-solution-hedging-the-hidden-guardrail)

**Risiko overfitting public LB & bukti geser private:**
- Greg Park: dari **rank 2 public → 52 private**; 5 besar public berakhir di rank 64/52/58/16/57; makin banyak submisi → makin jelek private (kurva-U setelah ±25 sub); *"It is easier to overfit the public leaderboard than previously thought. Be more selective with submissions."* — [Greg Park postmortem, 2012](https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem)
- Home Credit: **public 16 → private 127**; pelajaran *"don't trust LB, trust CV"* + pentingnya pemilihan 2 final. — [Wei Wu writeup](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/wei-wu-reflections-from-the-biggest-fall-on-leader)
- ROGII 22nd: **public rank 2 → private 22**; di band akhir, public↔private **Spearman −0.48** (antikorelasi), CV(773 well)↔private **+0.84**; *"We kept the gold because one final slot was reserved for pure CV."* — [22nd Place writeup, 7 Agu 2026](https://www.kaggle.com/c/rogii-wellbore-geology-prediction/writeups/22nd-place-solution-five-engines-and-a-hard-less)
- ROGII 3581→429: berhenti mengikuti notebook/LB publik 2 minggu sebelum akhir; final justru bukan yang public terbaik. — [3581st → 429th writeup](https://kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/3581st-429th-a-3152-place-shakeup)
- Papan sangat padat: public 0.95094 (rank 414) → private 0.95084 = **rank 4**; selisih 0.00010 membuat 410 rank. *"Public rank is a useful warning light, not always a useful steering wheel."* — [4th Place, Playground S6E7](https://www.kaggle.com/competitions/playground-series-s6e7/writeups/4th-place-from-414-to-4-trusting-oof-when-the)
- Gagal pilih sub ber-CV tinggi demi public LB = kalah medali: chaii. — [Adam Montgomerie, 2022](https://amontgomerie.github.io/2022/02/10/kaggle-journey.html)
- Konteks "public 0,95x turun di private": di kompetisi lain, publikasi leaderboard pascacompetition masih menampilkan public board (jebakan API) dan shu01 (leader public) jatuh ke rank 28 private; median tim memburuk saat private dihitung. — [Encinas, 7 Agu 2026](https://alanscottencinas.com/kaggle-private-leaderboard-api/)

**Timeout / stack lama (7 jam) — konteks spesifik RSNA:**
- Aturan: CPU/GPU notebook ≤ 9 jam (540 menit), internet mati, output `submission.csv`. — [Code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)
- Thread "9h runtime question": batas 9 jam hanya untuk **menghasilkan submission.csv** (bobot model boleh di-load dari Dataset, tidak perlu training di notebook); waktu yang dihitung untuk efficiency = dari submit sampai dapat skor; leaderboard efficiency tercepat ±5–6 menit, ada yang 5–6 jam. — [Discussion 740474, 9–20 Sep 2026](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740474)
- Rumus efficiency (resmi): `Efficiency = (AUC_benchmark − maxAUC + RuntimeSeconds) / 32400` — runtime 9 jam = 32400 detik, jadi stack 7 jam membuat skor efficiency praktis tidak bersaing. — [Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
- Mekanik penilaian: rules 3.18 mendefinisikan public/private sebagai **potongan test set yang berbeda untuk submission.csv yang sama** — skor private dihitung dari file yang sama, bukan rerun notebook, sehingga timeout bukan risiko saat penutupan; risiko timeout ada di (a) saat submit harian (margin ±2 jam untuk stack 7 jam), (b) rerun validasi pemenang (5 Nov), (c) kegagalan antrean GPU.
- Contoh nyata pilihan "lambat+kuat vs cepat+sedikit lebih buruk": Uni-Mol/XGB memilih satu model terbaik-LB ber-runtime panjang + satu lebih cepat sedikit lebih lemah; kekhawatiran kehabisan waktu ternyata tidak beralasan. — [38 public / 46 private writeup, 19 Sep 2025](https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/38-public-46-private-lb-solution-uni-molxgb)

### Inferences
- Formula praktis untuk tim 0,950: slot 1 = varian dengan public/CV terbaik yang sudah terbukti (cepat & stabil), slot 2 = varian mekanisme berbeda (mis. label/ensemble/backbone/preprocessing berbeda) dengan CV/OOF lokal setara — bukan varian incremental dari pipeline yang sama.
- Karena papan sangat padat (455 tim di 0,950), selisih public ±0.001 tidak bisa dipakai untuk memilih; gunakan OOF lokal + robustness check (beda fold/seed) seperti saran ROGII 22nd.
- Pastikan submisi final lolos margin waktu (ideal <6–7 jam) dan verifikasi pemilihan final via API/submission list, bukan hanya layar (pelajaran Encinas; rules 3.18.c auto-fill).

### Gaps
- Tidak ditemukan post-mortem khusus RSNA Knee 2026 yang sudah memuat hasil private (kompetisi masih berjalan).
- Tidak ada aturan resmi yang menyatakan Kaggle melakukan *rerun* notebook untuk skor private; asumsi "skor private = submission.csv yang sama" berdasar definisi rules 3.18.e–f — belum dikonfirmasi host secara eksplisit.

---

## 4. Dampak merger tim (sebelum 15 Okt 2026)

### Takeaway
Merger (tenggat 15 Okt 23:59 UTC, bersamaan Entry Deadline) **tidak menambah kuota 2 submisi final**, tapi memperbesar kumpulan kandidat/model & menggandakan quota submission harian gabungan; syaratnya ketat (jumlah submission gabungan harus ≤ batas harian × hari berjalan) dan maksimal 5 anggota.

### Cited Findings
- Team Merger Deadline & Entry Deadline: 15 Okt 2026 23:59 UTC. — [Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline); API kompetisi `teamMergerExplicitDeadline=2026-10-15T23:59:00Z` (9 Okt 2026)
- Rules 2.1.b: *"Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running."* — [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
- Rules 3.5.c: merger sah jika (i) gabungan ≤ maksimum tim, (ii) jumlah submission gabungan ≤ kuota satu tim saat itu, (iii) sebelum merger deadline, (iv) memenuhi syarat Rules lain. — [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
- Setelah merger, kode boleh dishare internal (anti "private sharing" 3.6.a hanya berlaku antar-tim terpisah). — [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
- Kuota akhir tetap: *"You may select up to two (2) Final Submissions for judging"* — dihitung per tim, bukan per anggota. — [Rules 2.2.b](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
- Konfirmasi pemilihan 2 final (peserta senior, BUKAN host — authorType=COMMENT): *"We select 2 final submissions. Both will be considered for main track and both will be consider for efficiency track."* — Chris Deotte, 25 Sep 2026 — [Discussion 743356](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743356)
- Pasar merger aktif: thread "[Team Merger] Looking for a 0.940+ Partner to Ensemble and Push for Gold" (15 Sep 2026). — [Discussion 741459](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741459)
- Field: 5.585 tim / 6.387 peserta / 99.373 submisi total (API Kaggle, 9 Okt 2026).

### Inferences
- Praktis sebelum 15 Okt: (1) cek dulu total submission gabungan vs batas (`kaggle competitions submission-limits`) sebelum merge — kalau melewati, merge gagal; (2) sepakati kriteria pemilihan 2 final di awal (2 slot = rebutan internal); (3) semakin banyak model di satu tim, semakin penting diversifikasi mekanisme antar 2 final, bukan menambah varian kecil; (4) setelah 15 Okt semua aktivitas entry/merge/publish notebook dibatasi (`kernelsPublishingDisabledDeadline=2026-10-15`), jadi semua kandidat final harus sudah terakumulasi sebelum tanggal itu.
- Merger tidak mengubah aturan "max 2 final", jadi manfaat utamanya: kumpul submission/kandidat terbaik + ensemble lintas tim + kuota harian gabungan (5/hari per tim).

### Gaps
- Rumus batas submission gabungan tidak dihitung presisi (perlu konfirmasi apakah "days the competition has been running" dihitung dari 30 Jul atau 5 Agu; rentang ≈355–385 submission gabungan).
- Maksud pasti `kernelsPublishingDisabledDeadline` (apakah melarang publish notebook baru) tidak dijelaskan di Rules — perlu verifikasi tambahan.

---

## 5. Proyeksi garis perak dari papan publik (9 Okt 2026)

### Takeaway
Papan publik 9 Okt (unduhan resmi 2026-10-09 14:48 UTC, 5.584 tim): garis perak (top 5%) hari ini berada di **0,950 (rank 279)**; untuk naik ke 0,951 hanya perlu +15 tim di atas 0,951 (kini 264), ke 0,952 perlu +108 (kini 171), ke 0,953 perlu +150 (kini 129). Rentang proyeksi 0,951–0,953 masuk akal mengingat 455 tim menumpuk persis di 0,950 dan sisa 13 hari.

### Cited Findings
- Ambang medali Kaggle untuk kompetisi 1.000+ tim: **Bronze = top 10%, Silver = top 5%, Gold = top 10 + 0,2%** (semakin banyak tim, makin banyak gold). — [Kaggle Progression (Competitions)](https://www.kaggle.com/progression/competitions)
- Data papan publik (CSV resmi `kaggle competitions leaderboard -d`, 9 Okt 2026 14:48 UTC, 5.584 tim, skor maks 0,964):
  - ≥ 0,953: **129 tim** | ≥ 0,952: **171 tim** | ≥ 0,951: **264 tim** | ≥ 0,950: **719 tim**
  - Tepat 0,950: **455 tim** (rank 265–719, termasuk posisi tim di rank 563)
  - ≥ 0,954: 100 | ≥ 0,955: 85 | ≥ 0,960: 26
- Garis medali pada papan hari ini (dihitung dari 5.584 tim): Gold rank 21 → 0,960; **Silver rank 279 (5%) → 0,950**; Bronze rank 558 (10%) → 0,950.
- Tim di 0,950 rank 563 = ~10,1% (tepi luar bronze, jauh dari silver 279) — sesuai posisi yang dilaporkan 9 Okt.
- Proyeksi akhir oleh peserta (informal, bukan resmi): Chris Deotte 26 Sep 2026: *"Gold threshold is probably 0.96 and first place is probably 0.975"* ; Quan Vu 30 Sep 2026: kenaikan kini hanya *"+0.001 to 0.002, which likely fall within the margin of noise... My last remaining hope lie in ensembling"* ; Tim Krige (Agu): *"private LB >0.94 would be gold"* (spekulasi awal). — [Discussion 735767](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767)
- Konteks kepadatan: public LB hanya 30% test set (`leaderboardPercentage=30`), skor ditampilkan 3 desimal (`scoreTruncationNumDecimals=3`), sehingga ratusan tim bisa berbagi skor identik.

### Inferences
- Butuh kenaikan nyata untuk masuk silver: dengan asumsi jumlah tim akhir ±5.600–6.000, garis perak akhir ≈ rank 280–300. Agar garis itu berada di 0,952, minimal ~110–130 tim harus melewati 0,951→0,952 dalam 13 hari; wajar karena 455 tim cuma butuh +0,002–0,003.
- Satu kenaikan +0,001 hingga 0,951 memindahkan tim dari rank 563 ke dalam blok ≥0,951 (rank 1–264) di papan publik — tetapi kualifikasi medali dihitung di **private** leaderboard dengan jumlah tim saat deadline, jadi yang harus dioptimalkan adalah selisih CV/OOF vs kandidat lain, bukan rank publik harian.
- Risiko terbesar proyeksi: overfitting public (lihat §3) — tim yang mengejar +0.001–0.002 di public 30% bisa turun di private 70%, menurunkan garis perak bagi yang konservatif dan menaikkannya bagi yang overfit.

### Gaps
- Garis perak final hanya bisa diketahui setelah private leaderboard dibuka (22 Okt+); perhitungan di atas berbasis jumlah tim hari ini, sedangkan Kaggle menghitung medali dengan jumlah tim pada deadline (bisa berbeda beberapa posisi — lihat dataset Meta Kaggle "Kaggle Shakeup Atlas").
- Tidak tersedia data historis harian resmi untuk mengukur laju harian perpindahan garis (hanya snapshot 9 Okt; snapshot lb/ 14:32 UTC vs lb2/ 14:47 UTC menunjukkan perubahan minimal dalam 15 menit).

---

## Sumber utama (diakses 9 Okt 2026)
- Rules resmi: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules (via Kaggle CLI `pages list --content`)
- Timeline: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline
- Thread host: /discussion/743416, /discussion/741819, /discussion/733965, /discussion/743356, /discussion/740474, /discussion/735767 (via API `discussions.DiscussionsService/GetForumMessagesInTopic`)
- Metadata & papan: Kaggle API `competitions.CompetitionService/GetCompetition`, `kaggle competitions leaderboard -d` (snapshot 2026-10-09T14:48Z)
- Medali: https://www.kaggle.com/progression/competitions
- Post-mortem: gregpark.io (2012), chioka.in, kaggle.com writeups (home-credit, playground-series-s6e7, rogii ×2, pengwang91, 46th ai-agent-security), alanscottencinas.com (7 & 12 Agu 2026), amontgomerie.github.io, datacamp, uselessai.in
