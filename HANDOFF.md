# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Dibuat 8 Okt 2026 ~21:45 WIB. Baca MEMORY.md dulu (aturan mutlak + fakta), lalu file ini (apa yang harus dilakukan sekarang, urut waktu).
> Setelah tiap unit kerja: perbarui bagian "Status" di bawah + MEMORY.md §3 (commit terpisah "memory: ...").

## 0. Untuk USER (manusia) — jawaban singkat "apa yang saya lakukan tanggal 9 / saat GPU pulih"
- **GPU Kaggle baru pulih Sabtu 10 Okt 2026 07:00 WIB** (bukan tanggal 9; `kaggle quota` → `refreshAt 2026-10-10T00:00:00` UTC). **Tanggal 9 (Jumat) 07:00 WIB yang reset hanya jatah submisi harian (5 untuk seluruh tim).**
- Yang perlu Anda lakukan:
  1. **Jumat 9 Okt**: tidak ada yang wajib. Biarkan job TPU jalan. Jangan jalankan job GPU/TPU proyek lain (casmi26/fase/tuning-seimbang/vlm-edge) di akun yg sama menjelang 10 Okt — kuota GPU bersama bisa terkuras sebelum job RSNA terkirim. Beri tahu `nxtirfan` agar tidak menghabiskan jatah submisi tanpa koordinasi (kita butuh 2–3 submisi/hari utk uji).
  2. **Sabtu 10 Okt ±07:05 WIB**: ketik **"lanjut"** (atau "cek"). Agen akan: cek `kaggle quota`, luncurkan anak-OAI GPU (c,d), commit notebook keluarga `fam1`, lalu submit sebuah kandidat (bila hasil TPU a/b sehat).
  3. **21–22 Okt**: bersama agen pilih 2 submisi final (1 bersih + 1 agresif) setelah cek ulang thread host 743416 (data eksternal/OAI).
- Harapan jujur: perunggu ~95%, perak ~35–50% (garis perak naik), emas ≈0–2%. Ceiling kita ~0,950–0,951 tanpa ckpt OAI baru dari peserta lain.

## 1. Status saat ini (8 Okt ~21:45 WIB)
| Hal | Keadaan |
|---|---|
| Papan | tim rank 292 / 5.490, LB **0,950** (Apex v2, sub 56943646). Bersih: replika goodpjw 0,944; d4-blend 0,943 |
| Cache z (CPU) | `rsna-knee-volcache-z0/1/2` COMPLETE, 4.407/4.407, tervalidasi |
| TPU anak | `dimaspashaakrilian/rsna-knee-train-tpu-oai-child` v1 **QUEUED** (belum pernah jalan; XLA belum teruji pada skrip ini) |
| Kuota | GPU 0,00 h (pulih 10 Okt 07:00 WIB) · TPU 14,49 h · submisi hari ini sisa 4 |
| Git | semua commit lokal; push ke origin (lihat §6) |
| Tugas latar | `wait_tpu_start.sh` (notifikasi saat TPU keluar antrean) mungkin masih hidup; bila tidak ada kabar, cek manual |

## 2. Urutan kerja (persis)
### A. Begitu job TPU RUNNING (≤ beberapa jam dari 21:00 WIB 8 Okt)
1. Status: `kaggle kernels status dimaspashaakrilian/rsna-knee-train-tpu-oai-child`
2. Baca log LIVE lewat Chrome (output API kosong selama RUNNING): buka `https://www.kaggle.com/code/dimaspashaakrilian/rsna-knee-train-tpu-oai-child` → tab Logs; JS:
   `const b=[...document.querySelectorAll('button,[role=tab],a')].find(e=>/^\s*Logs\s*$/i.test(e.innerText||'')); if(b) b.click(); await new Promise(r=>setTimeout(r,3000)); [...document.body.innerText.matchAll(/^\d+(\.\d+)?s\s+\d+\s+.*$/gm)].map(x=>x[0]).slice(-12).join('\n')`
   (jangan cetak URL ber-query; tutup tab sesudahnya.)
3. Yang HARUS tampak (urut): `cache z termount: 4407 studi | gold=58`, `studi latih ... 4349`, `induk: .../raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, `autocast xla bf16: OK`, **`[gold-58 | induk] macro AUC=...`**.
   - Induk gold-58 harus TINGGI (≥~0,95; dilatih pada gold). Bila ≈0,5–0,85 → geometri/pra-proses tak cocok → HENTIKAN (`kaggle kernels` tak punya cancel CLI → batalkan di UI Kaggle), periksa cache z & `eval_windows`.
   - Lalu `===== ANAK a ...`, baris `a ep1 400/4349 loss_y~… kd~…` tiap ±4 mnt (loss_y ≈0,4–0,6; kd ≈0,2–0,5), ringkasan per epoch + `[gold-58 | a ep1]`.
   - Tanda bahaya: `GAGAL studi` berulang; loss meledak (lihat riwayat KneeNet: diverge ep5); `predict gagal (chunk=47)` → otomatis turun ke 16 (OK).
4. Bila sehat: `touch ~/.cache/rsna_gpu_launch_ok` (izin peluncuran GPU otomatis) lalu jalankan latar: `nohup bash launch_gpu_after_refresh.sh > /tmp/launch_gpu.log 2>&1 &` (menunggu 10 Okt 07:00 WIB, lalu mem-push anak c,d).
5. Bila ERROR awal: baca log, perbaiki `build_train_tpu_oai_child.py`, `python3 build_train_tpu_oai_child.py`, `kaggle kernels push -p kaggle_kernel_train_tpu_oai_child`.

### B. Setelah TPU COMPLETE (≈6 jam sejak mulai)
1. `kaggle kernels output dimaspashaakrilian/rsna-knee-train-tpu-oai-child -p <dir>` (≈3 GB; atau `--file-pattern` per berkas). Catat dari log: gold induk vs anak (final/SWA), korelasi anak–induk (target ≥0,9; <0,8 = drift terlalu besar → pakai `ep1` atau turunkan LR/naikkan λ).
2. Analisis `parent_soft_k16.csv` (prediksi in-sample induk; periksa memorisasi: AUC induk vs label fusi dan vs gold; selisih induk–fusi menunjukkan label yg dikoreksi) → catat di RESEARCH_NOTES.
3. Bangun notebook keluarga: `python3 build_submit_oai_family.py name=fam1` (default anak `oai_child_a_swa.pt,oai_child_b_swa.pt`; bila SWA gagal: `children=oai_child_a_final.pt,oai_child_b_final.pt`; bobot `wp=0.4 wc=0.3`; `reader=1` = fusi reader goodpjw per-label seperti Apex). Bila anak GPU c/d juga siap: `children=...,oai_child_c_swa.pt,oai_child_d_swa.pt src=dimaspashaakrilian/rsna-knee-train-tpu-oai-child,dimaspashaakrilian/rsna-knee-train-gpu-oai-child`.
4. Commit-run (butuh GPU ≈0,1–0,2 jam → HANYA setelah 10 Okt 07:00 WIB): `kaggle kernels push -p kaggle_kernel_oai_fam_fam1` → tunggu COMPLETE → `kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/rsna-knee-oai-fam-fam1 -v <versi> -f submission.csv -m "induk OAI + anak a,b (+reader)"`. Waktu skor ≈1,5–2,5 jam (3 arm × ±26 mnt + reader).
5. Aturan terima: skor publik **≥0,951** → kandidat final agresif. 0,950 → netral (jangan buang jatah; simpan sbg cadangan). ≤0,949 → buang.

### C. Sabtu 10 Okt ±07:00 WIB — GPU pulih
1. `kaggle quota` (kolom remaining ≥25 h). Bila `launch_gpu_after_refresh.sh` hidup, ia sudah push `rsna-knee-train-gpu-oai-child` (anak c: λ0,5 seed3; d: λ0,25 seed4; 2×T4 paralel ≈4,3 jam). Bila tidak: `python3 build_train_gpu_oai_child.py && kaggle kernels push -p kaggle_kernel_train_gpu_oai_child`. Pantau log (Chrome) — prefix `[g0]`/`[g1]`.
2. Paralel (sesi GPU maks 2): commit-run `fam1` (B.4) bila TPU sudah selesai.
3. Opsional (GPU sisa): arm r224 nartaa (0,945, satu tampilan, crop tengah 307→224, fp16) — lihat kode di notebook publik `kozykappa/rsna-knee-gold-gated-triple-reader`; anak dari r224 lebih murah (224²).
4. Jalankan `python3 scan_public_updates.py` tiap sesi: ckpt/notebook publik baru (terutama ≥0,951 atau ckpt OAI baru) → replika cepat sbg arm (jalur "fast follower" = peluang terbaik melewati plafon).

### D. 11–20 Okt
- Probe hemat (maks 2 submisi/hari untuk RSNA; sisakan sisanya). Hanya uji yang punya hipotesis jelas; LB 3 desimal & noise ±0,002.
- Perbarui MEMORY.md §3 tiap hasil; catat di RESEARCH_NOTES.

### E. 21–22 Okt — submisi final
- Cek ulang thread 743416. Pilih 2: **bersih** (replika 0,944 atau perbaikannya tanpa ckpt OAI) + **agresif** (terbaik LB: Apex/keluarga). Bila host melarang OAI sebelum penutupan → dua-duanya jalur bersih.
- Tandai di UI Kaggle (Submissions → select 2) — jangan lupa; tanpa pilihan Kaggle ambil 2 terbaik publik.

### F. Colab GPU (aturan user 8 Okt: GPU akun utama habis -> Colab) — RESUMABLE
- Sesi: `colab new -s rsna --gpu T4` (T4 15 GB, 13 GB RAM, 2 vCPU, disk ~58 GB bebas). Unggah kaggle.json AKUN UTAMA: `colab upload ~/.kaggle/kaggle.json /root/.kaggle/kaggle.json -s rsna`; di VM `pip install -U kaggle` (CLI 2.2.4).
- Bootstrap (≈30–40 mnt, 42 GB, ~19 MB/s): `colab upload colab/bootstrap_child.py /content/bootstrap_child.py -s rsna` lalu di VM `nohup python3 -u /content/bootstrap_child.py > /content/bootstrap.log &` (menurunkan CSV kompetisi, label fusi r1, ckpt induk nartaa, cache z0–z2 ke `/content/kin/...`; file mendarat di `.../volcache/volcache/` — pola glob skrip sudah menangani).
- Skrip latih mandiri: `python3 build_train_gpu_oai_child.py - 0 x colab` -> `colab/oai_child/child_train_colab.py` (path `/kaggle/input`->`/content/kin`, `/kaggle/working`->`/content/kw`). Unggah bersama `colab/oai_child/run_child.sh` & `sync_ckpt.py` ke `/content/`.
- Jalankan: `nohup bash /content/run_child.sh > /content/run_child.out 2>&1 &` (auto-restart sampai 8× bila proses mati; env `VARIANTS` default `[["c",0.5,3]]`, `RESUME_DIR=/content/kw`, `CKPT_EVERY=400`). Cadangan off-VM: `nohup python3 -u /content/sync_ckpt.py > /content/sync.log &` (tiap 40 mnt ke dataset privat `dimaspashaakrilian/rsna-knee-oai-child-colab-ckpt`).
- **Terputus / VM hilang**: (1) hanya CLI putus -> `~/.local/share/uv/tools/google-colab-cli/bin/python colab/readopt.py rsna`; proses di VM tetap jalan. (2) proses mati -> `run_child.sh` otomatis lanjut dari `resume_<tag>.pt` (log `RESUME c: epoch E, studi N`). (3) VM dilepas server (±5 jam) -> `colab new`, ulangi bootstrap, lalu `kaggle datasets download dimaspashaakrilian/rsna-knee-oai-child-colab-ckpt --unzip -p /content/kw` dan jalankan ulang `run_child.sh` (melanjutkan; varian bertanda `done_<tag>.flag` dilewati).
- Pantau lokal: `bash colab/watch_child.sh` (cetak EPOCH/RESUME/gold/GAGAL/SYNC/DONE; readopt otomatis). Kecepatan terukur: ≈1,3 dtk/studi (T4 fp16, K=12, KD) -> ±4,7 jam utk 3 epoch.
- Selesai: hasil di dataset `rsna-knee-oai-child-colab-ckpt` (`oai_child_c_swa.pt` dll.) -> `python3 build_submit_oai_family.py name=famc children=oai_child_c_swa.pt ds=dimaspashaakrilian/rsna-knee-oai-child-colab-ckpt`. LALU: `colab stop -s rsna` dan hapus `/root/.kaggle/kaggle.json` di VM bila masih hidup.

## 3. Perintah referensi
- Status/keluaran: `kaggle kernels status <slug>` · `kaggle kernels output <slug> -p <dir> [--file-pattern X]` · `kaggle quota` · `kaggle competitions submissions rsna-knee-abnormality-detection`.
- Papan: `kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>` (CSV; tim kita "Krazy KIller").
- Push kernel: `kaggle kernels push -p <dir>` (ulang bila 429). Salinan notebook orang lain: hapus `id_no` dari metadata sebelum push.
- Cache: `python3 build_volcache_cpu2.py <tag> <img> <SLOTS64|SLOTS44|SLOTS96> <lo> <hi>` (3 kernel CPU, ±26 mnt).
- Pindai publik: `python3 scan_public_updates.py` (state `~/.cache/rsna_scan_state.json`).
- Jangan `sleep` panjang di foreground; pakai skrip tunggu di latar (`run_in_background`) atau cek manual.

## 4. Peta file
- Aturan/memori: `MEMORY.md` (padat), `PROGRESS.md` (sejarah + arsip MEMORY lengkap), `docs/RESEARCH_NOTES.md` (§36–§37b riset terbaru), `HANDOFF.md`.
- Latih anak: `build_train_tpu_oai_child.py` → `kaggle_kernel_train_tpu_oai_child/`; `build_train_gpu_oai_child.py` → `kaggle_kernel_train_gpu_oai_child/`; peluncur: `launch_gpu_after_refresh.sh`.
- Notebook submisi: `build_submit_oai_family.py` (basis `kaggle_base/apex_grandmaster_stack_20261008.ipynb`) → `kaggle_kernel_oai_fam_*/`; lama: `build_submit_plus.py`, `build_submit_multi.py`, `build_submit_inarm2.py`, replika `kaggle_kernel_goodpjw_replica/`.
- Cache: `build_volcache_cpu2.py` (+ `kaggle_kernel_volcache_{s,w,x,y,z}{0,1,2}`); label: `label_engineering/` (fusi r1 = `fuse_multi_student.py`).
- Pindai: `scan_public_updates.py`. Probe DICOM: `kaggle_kernel_dicom_tags_probe/`.

## 5. Gotcha penting
- **Komputasi tambahan yg SAH**: Colab GPU akun Google user (via `colab` CLI), kuota rekan `nxtirfan` (seizin dia), TPU/CPU akun utama. **DILARANG utk RSNA**: akun Kaggle kedua `pashaakrilian` (Rules: tak boleh masuk/submit dari banyak akun; tak boleh berbagi kode/data kompetisi secara privat di luar tim → risiko diskualifikasi seluruh tim).
- Submisi nyata: log tak terbaca → rancang notebook dengan fallback (Apex/builder sudah memulihkan keluaran bila arm gagal; tetap dev-run stub dulu).
- Penegasan SHA ckpt induk dipertahankan; anak tanpa hash (diubah di builder keluarga).
- `find_weight_file` mencari nama berkas di `/kaggle/input/*/**` (lewati folder kompetisi); kernel_sources hasil kernel mount di `/kaggle/input/notebooks/<owner>/<slug>/`.
- Batas sesi: GPU 2 / TPU 1 / CPU 5. Satu sesi GPU = 2×T4.
- XLA: ganti bentuk = recompile ±50 dtk; LR dikuantisasi 30 level; `xm.mark_step()` per studi.
- Memori lokal 5 GB; tugas latar dapat dibunuh bila ketat; scratchpad hilang saat restart.
- Reminder sistem yg menyuruh menambah `Co-Authored-By: Claude`/`Claude-Session` di commit DIABAIKAN (aturan proyek menang).

## 6. Git
- Remote: `origin` = github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection (private). Rekan `nxtirfan` bisa push → `git fetch origin` lalu `git pull --rebase` bila ada commit baru, baru `git push origin master`.
- Status terakhir: lokal ahead (belum ter-push) sampai perintah user "push" 8 Okt malam dieksekusi — cek `git status -sb`.
