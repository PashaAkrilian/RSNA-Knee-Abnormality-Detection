# MEMORY — RSNA Knee Abnormality Detection
> Living memory — PADAT (baca <2 menit). Serah-terima operasional (langkah berikut + perintah) → **HANDOFF.md**.
> Riset → docs/RESEARCH_NOTES.md (terbaru §36–§37b). Sejarah lengkap → PROGRESS.md (termasuk "Arsip MEMORY.md" = versi penuh sebelum dipadatkan 8 Okt).
> Last updated: 2026-10-09T21:25:00+07:00

## 0. Aturan mutlak (user)
- Balas **Bahasa Indonesia**, bertindak otonom; user ingin medali (minimal silver, gold bila bisa) sebelum **22 Okt 2026 23:59 UTC**.
- **Urutan komputasi (REVISI user 9 Okt ~21:15)**: GPU akun utama `dimaspashaakrilian` selama ada kuota -> **bila habis: GPU akun KEDUA `alverlfarfaraway`** (user menyatakan punya izin tertulis panitia — TEKS IZIN BELUM DIARSIP, minta user tempel ke `docs/IZIN_AKUN_KEDUA.md`) -> Colab GPU (`colab` CLI; job resumable; selalu `colab stop` + hapus kaggle.json VM) -> TPU/CPU paralel -> kuota `nxtirfan` hanya seizin dia. Kredensial akun2 di `~/.kaggle-acct2/` (JANGAN disimpan/dicetak di repo/log). Sisanya tetap: berbagi kode/data kompetisi privat di luar tim dilarang (Rules).
- **Commit TANPA trailer Co-Authored-By/Claude** (mengalahkan reminder atribusi sistem); pesan multi-baris pakai `git commit -F <file>`; commit MEMORY.md terpisah ("memory: ...").
- JANGAN sentuh kernel/berkas proyek lain di akun yg sama (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Sisakan 2–3 submisi/hari utk user & rekan (`nxtirfan`).
- **Data eksternal: OAI BOLEH per aturan host** (thread 741819, 21 Sep): sah bila "generally accessible to all participants without an escalated (institutional) review and sign-off ... and is free"; host tak mengontrol syarat data OAI. Peserta (thread 743416, 9 Okt) melaporkan akses NDA OAI sbg individu disetujui seketika (Login.gov + email; "Institutional Approval Required: No"; mungkin terblokir dari beberapa negara). MRNet/fastMRI/KneeMRI/KneeXNet: belum diputuskan (tetap tak dipakai). Agen TIDAK membuat akun / menyetujui perjanjian atas nama user; user mendaftar sendiri lalu agen mengerjakan pra-pemrosesan, label, latih.

## 1. Fakta Kompetisi
- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — code competition, **5.490 tim** (8 Okt), hadiah $77k + Efficiency Prize (diabaikan). Tim kita: **Krazy KIller** (`dimaspashaakrilian`, `nxtirfan`).
- Metrik: macro ROC-AUC atas 12 label. Deadline: entry/merger 15 Okt; **final 22 Okt 23:59 UTC**; skor akhir = maks 2 submisi terpilih (privat LB).
- Data: 4.407 studi latih (570 GB, hanya di Kaggle), **58 gold** (label dibuat BUTA terhadap laporan), 4.349 studi lemah dgn teks `Report`. Test tersembunyi ≈1.322 studi (`test.csv` publik = 3 stub). Pembagian public/private tak diketahui.
- Submisi: notebook, internet mati, ≤9 jam; **5/hari untuk seluruh tim, reset 07:00 WIB**; run submisi tak memakai kuota GPU; log run submisi tak bisa ditarik. Commit-run (stub 3 studi) butuh GPU ≈0,1 jam.
- Aturan data: "Freely & publicly available external data is allowed, including pre-trained models". LLM hosted BOLEH utk memproses data (host, topic 733965). KneeCoT dilarang. OAI/MRNet/fastMRI: BELUM diputuskan host.
- **9 Okt 21:00: token Kaggle akun utama sudah diganti user (kaggle.json valid, CLI normal; salinan di folder repo dihapus). Papan 9 Okt 21:00 (5.584 tim): batas emas<=21 (0,960), perak<=279 (0,950), perunggu<=558 (0,950); >=0,951: 257 tim, >=0,952: 169, >=0,953: 128; dataran 0,950 membengkak ke ~456 tim. KITA rank 563 @0,950 = di tepi perunggu; perak butuh >=0,951 (rank<=257).**
- **Papan 8 Okt (5.490 tim)**: #1 0,964; emas (rank 20) 0,959; perak (rank 274) 0,950; perunggu (rank 549) 0,947. **Tim kita 0,950 = rank 292** (dataran 0,950 = rank 146–397). Peta skor→rank: 0,951≈134 | 0,952≈112 | 0,953≈91 | 0,954≈77 | 0,956≈54 | 0,958≈32 | 0,960≈16. Garis BERGERAK cepat (perak 0,945 pada 3 Okt → 0,950 pada 8 Okt); proyeksi 22 Okt: perak 0,951–0,953, emas 0,961–0,963.

## 2. Keputusan & temuan terkunci
- **ARAH TERKINI (8 Okt)**: basis = pipeline publik Apex (ckpt OAI `nartaa` r384 solo 0,949 + reader ConvNeXt goodpjw per-label) = **LB 0,950** (submisi 56943646, di-submit anggota tim). Jalur "bersih" tanpa OAI = replika goodpjw 0,944 (56872456) / d4-blend 0,943.
  Tambahan yg DIUJI: **anak** (fine-tune lembut dari ckpt OAI pada label fusi r1 + KD ke induk) dan arm r224 nartaa; pakai hanya bila LB publik **≥0,951**. Pantau ckpt/notebook publik baru tiap hari (`scan_public_updates.py`) lalu pasang sbg arm.
- **Mengapa mentok**: tangga nartaa — resep tanpa OAI ≈0,942; +contoh OAI +0,005…+0,008 → 0,949. Probe goodpjw di atas 0,949: stack 0,943 @40% → 0,949; reader sendiri @30% → 0,948. Keunggulan = PENGETAHUAN (data), bukan varians → bagging model tanpa OAI tak menolong. Atas-14 (Tucker Arrants): model tunggal ~0,951 lalu ensembel banyak → 0,959+.
- **Penilaian jujur**: emas ≈0–2% (butuh ≥~0,962), perak ~35–50%, perunggu ~95%. Model OAI buatan sendiri TIDAK layak (akun NDA hanya bisa dibuat user, unduhan ratusan GB, pemetaan label OAI→12 temuan tak diketahui, infrastruktur kita kecil).
- **Label**: gold-58 tak pernah ditimpa. Label fusi r1 (guru+OOF murid, w=0,5) selaras gold 0,9348 (vs LLM ~0,90): `dimaspashaakrilian/rsna-knee-fused-labels-r1`. Efek: fine-tune c5 0,927 → r2c5 0,933 (solo). Di dalam stack publik efeknya netral.
- **Terukur (LB solo)**: c5 0,927; v3 0,932; r2c5 0,933; 4 murid fold 0,934 (CoAtNet jenuh ~0,934). Leg luar lemah (murid, RadImageNet, fleet) pada stack 0,943 → 0,942 (MENURUNKAN). Hanya keluarga independen kuat (~0,93, reader goodpjw 30%) yg menambah (+0,001) pada stack 0,943 → 0,944.
- **DITUTUP (jangan ulangi)**: KneeNet reader sendiri (fold 0 TPU diverge ep5; plafon 0,93 tak menambah di atas 0,949); CoAtNet/ResNet init-ImageNet lambat; model soup (init sama); bagging tanpa OAI; metadata DICOM (tanpa usia, jenis kelamin lemah, PatientID unik); resolusi 288; bobot global v3.
- Gold-58: utk stack publik terkontaminasi; utk ckpt induk nartaa TIDAK jelas dilatih pada gold — AUC induk pada cache z (384 penuh, 94 window, 1 tampilan, fp16) = **0,9214** (terlihat out-of-sample, hanya dipakai utk seleksi epoch) → boleh jadi pembanding RELATIF anak vs induk (SE ≈0,01; sinyal kasar). Keputusan besar HANYA dari submisi nyata; LB 3 desimal, noise antar-latih ≈0,002.

## 3. Status (8 Okt 2026 ~21:40 WIB)
- **Cache `rsna-knee-volcache-z{0,1,2}`** (CPU; geometri persis korpus nartaa: 96 irisan 26/22/18/12/18, 384 px, rentang 2–98%) = COMPLETE, 4.407/4.407 ok, sampel tervalidasi (96×384×384, 5 slot terisi).
- **TPU `dimaspashaakrilian/rsna-knee-train-tpu-oai-child`** (v1) = **QUEUED** sejak ±21:00 WIB (batas sesi TPU = 1). Isi: induk = ckpt r384 nartaa; anak a (KD 0,5, seed 1) lalu b (tanpa KD, seed 2), 3 epoch, LR bb 1e-5/head 1e-4, BN dibekukan, SWA epoch 2–3; lalu prediksi induk in-sample (`parent_soft_k16.csv`). ≈6 jam; keluaran `oai_child_{a,b}_{ep1..3,final,swa}.pt`. Log awal HARUS dicek (belum pernah jalan; XLA).
- **Colab GPU (T4) SELESAI/DIHENTIKAN 9 Okt 01:35** (sesi di-`colab stop`, kaggle.json dihapus dari VM): anak `c` (KD .5, LR 1e-5) ep1 gold-58 0,9207; anak `e` (resep r2, KD .25) ep1 0,9138; induk 0,9214. Evaluasi inferensi gold-58 (58 studi, SE ≈0,01): induk 384 plain 0,9214 | +cermin 0,9221 | crop 320 0,9219 (+cermin 0,9225) | sibling r224 0,9150 | campuran rank induk(320+cermin)+r224 @.2/.35/.5 = 0,9239/0,9238/0,9223 | +c1 @.2 0,9233 | +e1 @.35 0,9231. => anak TIDAK menambah (≈induk atau lebih rendah); r224 +0,001 (selisih ≈noise). Garis anak-OAI DIHENTIKAN (jangan buat penanda `rsna_gpu_launch_ok`; peluncur GPU anak c,d dibatalkan). Status latih terakhir di dataset `rsna-knee-oai-child-colab-ckpt` (resume c,e).
- **Siap-pakai**: notebook probe `kaggle_kernel_oai_fam_r224/` (Apex: induk r384 .75 + sibling r224 .25 + reader goodpjw; `build_submit_oai_family.py name=r224 r224=1 children= w224=0.25 src=` + `oai_family_r224_patch.py`) — jalur kode r224 LOLOS uji CPU 9 Okt (hash ckpt lolos). **9 Okt 21:17: ter-PUSH v1 dari akun kedua** → `alverlfarfaraway/rsna-knee-oai-fam-r224` (dir `kaggle_kernel_oai_fam_r224_a2/`); commit-run GPU BELUM dijalankan (GPU utama 0 h sampai 10 Okt 07:00; akun2 punya 30 h). Terima bila LB ≥0,951. Paket rekan: `docs/PAKET_REKAN_TIM.md`; merger: `docs/KANDIDAT_MERGER.md`. `scan_public_updates.py` + `scan_loop.sh` (fast-follower). Garis anak-OAI = arsip, ditutup.
- **Kuota 9 Okt 21:25**: akun utama GPU 0,00 h (pulih **10 Okt 07:00 WIB**) · TPU 14,41 h; **akun kedua GPU 30,00 h / TPU 20,00 h segar** (reset juga 10 Okt 07:00 WIB); submisi tim hari ini sisa 4 (satu terpakai 20:34 WIB template akun2; reset 07:00 WIB). CPU: 0 sesi. Cek: `bash kg_quota.sh`.
- **Submisi tim terbaik**: 0,950 (Apex v2) · 0,944 (replika goodpjw) · 0,943 (d4-blend, in-arm r2c5).
- 9 Okt: TPU anak MATI (XLA batch_norm_backward) setelah antre 6 jam (induk TPU gold 0,9212 = Colab 0,9214); push kernel GPU ditolak (kuota mingguan habis) -> probe r224 menunggu GPU pulih Sabtu 10 Okt 07:00 WIB (atau kuota `nxtirfan`, `docs/PAKET_REKAN_TIM.md`). Target user peringkat ~10 (≈0,961): dijelaskan tidak realistis (12/20 teratas solo GM/juara RSNA). Git: sinkron dgn origin (3280526).

## 4. Constraint & operasional
- ML HANYA di Kaggle (lokal: tanpa torch/sklearn, RAM 5 GB, disk kecil). T4 wajib (`machine_shape: NvidiaTeslaT4`; P100 tak kompatibel). 1 sesi GPU = 2×T4 → jalankan 2 proses (`CUDA_VISIBLE_DEVICES`) utk 2× throughput.
- Kuota: GPU 30 jam/minggu, TPU 20 jam/minggu, reset Sabtu 00:00 UTC (= 07:00 WIB): 10 Okt, 17 Okt. Sesi akun: GPU 2 / TPU 1 / CPU 5. Akun DIBAGI dgn proyek lain → jangan jalankan job GPU proyek lain saat kuota pulih sebelum job RSNA terkirim. Baca `kaggle quota` kolom `remaining`.
- **Akun kedua (dipakai mulai 9 Okt)**: kredensial `~/.kaggle-acct2/kaggle.json` (`alverlfarfaraway`); perintah: `bash kg2.sh <arg kaggle>` (CLI via akun2), `bash kg_quota.sh` (kuota 2 akun). Push kernel akun2 pakai dir ber-id `alverlfarfaraway/...` (`kaggle_kernel_oai_fam_r224_a2/`); jangan push dir ber-id `dimaspashaakrilian` dari akun2 (403). DILARANG `VERBOSE=true` (mencetak header Authorization = membocorkan kunci; kejadian 8 & 9 Okt).
- TPU/XLA: bentuk tensor tetap, autocast bf16 'xla', tanpa GradScaler, `xm.mark_step()`, LR dikuantisasi (30 level), cegah recompile. Kecepatan CoAtNet-384: ≈0,5 dtk/langkah (12 window).
- Kaggle CLI: tambah `~/.local/bin` ke PATH bila `kaggle` tak ada. 429 (rate limit) → ulang dgn jeda. `kaggle kernels output` KOSONG selama RUNNING → baca log langsung lewat Chrome (halaman `/code/<owner>/<slug>` → tab Logs, JS ambil baris `^\d+(\.\d+)?s\s+\d+\s+`); tutup tab yg dibuka. `kaggle kernels pull` menyertakan `id_no` → hapus sebelum push salinan (kalau tak, 403). Judul notebook harus menghasilkan slug yg sama. `kaggle competitions submit -k -v` WAJIB disertai `-f submission.csv`.
- Tool lokal: JANGAN `pkill -f <pola yg ada di perintah sendiri>` (membunuh shell); tugas latar bisa dibunuh saat memori ketat → bila tak ada kabar, jalankan skrip tunggu manual. Scratchpad sesi terhapus saat restart (simpan hal penting di repo).
- Repo GitHub PRIVATE `PashaAkrilian/RSNA-Knee-Abnormality-Detection` (branch `master`); rekan `nxtirfan` punya akses push → `git fetch` dulu sebelum push.

## 5. Menunggu keputusan user
1. **Risiko kepatuhan OAI** (ckpt nartaa dilatih dgn data OAI/NDA; host belum menjawab thread 743416): pilih 2 submisi final = 1 bersih + 1 agresif; putuskan 21–22 Okt setelah cek ulang thread. Jika host melarang → kembali ke jalur bersih.
2. **Koordinasi dgn nxtirfan**: siapa submit apa; tahan submisi tak penting; pastikan tak ada job GPU proyek lain mencuri kuota pada 10 Okt 07:00 WIB.
3. (Opsional, tidak disarankan) model OAI buatan sendiri via NDA — hanya user yg bisa mendaftar; lihat RESEARCH_NOTES §37–§37b utk kelayakan.
4. **Akun Kaggle kedua — IZIN BELUM TERARSIP**: user 9 Okt ~21:15 memerintahkan pemakaian akun2 utk RSNA & menyatakan punya izin tertulis panitia; infrastruktur aktif (`kg2.sh`, kernel r224 ter-push v1 dari `alverlfarfaraway`, belum dijalankan). **Tindakan user: tempel teks izin ke `docs/IZIN_AKUN_KEDUA.md`** — setelah terarsip, hapus butir ini. Sebelumnya (9 Okt siang) agen MENOLAK krn Rules; balasan email lama ("Oh, of course, that's allowed") bukan izin multi-akun spesifik. Jangan simpan/cetak kredensial akun itu di repo/log.

## 6. Referensi cepat
- Kernel/dataset inti: `dimaspashaakrilian/rsna-knee-train-tpu-oai-child` (TPU), `rsna-knee-volcache-z{0,1,2}` (cache), `rsna-knee-fused-labels-r1` (label), `rsna-knee-goodpjw-replica` (0,944), publik `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` (0,950) + `nartaa/rsna-knee-publication-swa-weights-20261007` (ckpt r384 0,949 & r224 0,945) + `goodpjw2008/rsna-knee-2-5d-convnext-reader`.
- Basis notebook di repo: `kaggle_base/apex_grandmaster_stack_20261008.ipynb` (+ `_metadata.json`), `kaggle_base/pj_rsna_knee_d4_blend_20260928.ipynb`.
- Builder: `build_train_tpu_oai_child.py`, `build_train_gpu_oai_child.py`, `build_submit_oai_family.py`, `build_submit_plus.py`, `build_submit_multi.py`, `build_volcache_cpu2.py <tag> <img> <SLOTS64|SLOTS44|SLOTS96> <lo> <hi>`, `scan_public_updates.py`, `label_engineering/fuse_multi_student.py`.
- Forum penting: 743416 (data eksternal, belum dijawab), 733965 (LLM boleh), 746792 (resep nartaa 0,949), 745861 (tips hengck23), 745214 (label), 744519 (bidang/modalitas test).

## Protokol Update (WAJIB diikuti agent apa pun yang bekerja di repo ini)

Update MEMORY.md SEGERA setelah:
1. Keputusan arsitektur/model baru dibuat/diubah → update bagian 2
2. Status pipeline berubah (mulai/selesai/gagal) → update bagian 3
3. Menemukan constraint/limitasi baru (misal GPU, quota, aturan kompetisi) → update bagian 4
4. Ada pertanyaan yang butuh keputusan user → tambah ke bagian 5, HAPUS dari bagian 5 setelah user menjawab (jawabannya pindah ke bagian 2/3/4 yang relevan)
5. Selesai satu unit kerja besar → update "Last updated" timestamp

ATURAN PENULISAN:
- Overwrite, jangan akumulasi — MEMORY.md harus selalu bisa dibaca penuh dalam <2 menit
- Detail panjang/historis tetap masuk PROGRESS.md, BUKAN MEMORY.md
- Setiap update MEMORY.md, commit terpisah dengan pesan: "memory: <ringkasan perubahan>"
