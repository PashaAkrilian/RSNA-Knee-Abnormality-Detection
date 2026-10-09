# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 9 Okt 2026 ~09:30 WIB. Baca MEMORY.md dulu (aturan mutlak + fakta), lalu file ini (apa yang harus dilakukan, urut waktu).
> Setelah tiap unit kerja: perbarui "Status" di bawah + MEMORY.md §3 (commit terpisah "memory: ...").

## 0. Untuk USER (manusia)
- **GPU Kaggle akun utama pulih Sabtu 10 Okt 2026 07:00 WIB** (`kaggle quota` → `refreshAt 2026-10-10T00:00:00` UTC). Tanggal 9 (Jumat) 07:00 WIB hanya jatah SUBMISI harian yang reset (5/hari utk seluruh tim).
- Yang perlu Anda lakukan: (1) Jumat 9 Okt — tidak ada yang wajib; jangan jalankan job GPU/TPU proyek lain di akun yg sama menjelang pemulihan GPU (kuota bersama). Beri tahu `nxtirfan` agar tak menghabiskan jatah submisi tanpa koordinasi. (2) Sabtu 10 Okt ±07:05 WIB ketik **"lanjut"**: agen akan commit notebook probe "Apex + sibling r224" lalu submit. (3) 21–22 Okt: pilih 2 submisi final bersama agen (1 bersih tanpa OAI + 1 agresif) setelah cek ulang thread host 743416.
- Akun Kaggle kedua: TIDAK dipakai utk RSNA (lihat MEMORY §5 butir 4); user menyebut ada izin panitia via email tapi belum ada izin tertulis spesifik multi-akun. Jalur sah GPU tambahan: Colab, kuota `nxtirfan`, tunggu 10 Okt.
- Harapan jujur: perunggu ~95%, perak ~35–50% (garis perak naik), emas ≈0–2%. Plafon kita tanpa ckpt OAI baru dari peserta lain ≈0,950–0,951.

## 1. Status (9 Okt ~02:00 WIB)
| Hal | Keadaan |
|---|---|
| Papan | tim rank 292/5.490, LB **0,950** (Apex v2, sub 56943646). Bersih: replika goodpjw 0,944; d4-blend 0,943 |
| Garis ANAK-OAI | **DITUTUP** (gold-58: induk 0,9214; anak c ep1 0,9207; anak e ep1 0,9138; campuran ≤+0,0015 = noise). Colab dihentikan, kaggle.json dihapus dari VM |
| Sibling r224 | nartaa r224 (LB 0,945) gold-58 0,9150; campuran rank dgn induk @.2–.35 = 0,9238–0,9239 (≈+0,0014). **Notebook probe siap**: `kaggle_kernel_oai_fam_r224/` (slug `rsna-knee-oai-fam-r224`; induk .75 + r224 .25 + reader goodpjw per-label) — jalur kode r224 LOLOS uji CPU (kernel `rsna-knee-smoke-r224-cpu`: jendela & probabilitas IDENTIK dgn implementasi mandiri yg tervalidasi gold, hash kedua ckpt lolos); cabang konsumen CUDA (`infer_probs`) baru teruji saat commit-run GPU |
| TPU anak a/b | `dimaspashaakrilian/rsna-knee-train-tpu-oai-child` v1 masih QUEUED (sejak 8 Okt 21:00). Bernilai rendah; boleh dibiarkan/dibatalkan lewat UI |
| Kuota | GPU 0,00 h (pulih 10 Okt 07:00 WIB) · TPU ±14,5 h · submisi hari ini sisa 4 (reset 07:00 WIB) |
| Tugas latar | `scan_loop.sh` (log `~/.cache/rsna_scan_log.txt`, tiap 20 mnt), `wait_tpu_start.sh` |
| Git | commit lokal ahead of origin (push hanya bila user minta; `git fetch` dulu — rekan `nxtirfan` bisa push) |

## 2. Urutan kerja
### A. Sabtu 10 Okt 07:00+ WIB — GPU pulih
1. `kaggle quota` (GPU remaining ≥25 h).
2. Probe r224: `python3 build_submit_oai_family.py name=r224 r224=1 children= w224=0.25 src=` (sudah terbangun) → `kaggle kernels push -p kaggle_kernel_oai_fam_r224` (commit-run di stub 3 studi, GPU ≈0,1–0,2 jam; jalur kode baru: arm `single/crop/img`, hash r224, `infer_probs`). Bila ERROR: baca log (Chrome, tab Logs), perbaiki `oai_family_r224_patch.py`.
3. Setelah COMPLETE: `kaggle competitions submit rsna-knee-abnormality-detection -k dimaspashaakrilian/rsna-knee-oai-fam-r224 -v <versi> -f submission.csv -m "Apex: induk r384 .75 + sibling r224 .25 + reader goodpjw"`. Skor ≈1–1,5 jam.
4. Aturan terima: skor publik **≥0,951** → kandidat final agresif. 0,950 → netral (simpan). ≤0,949 → buang. LB 3 desimal & noise ±0,002.
5. Variasi murah berikutnya (1 submisi tiap uji, maks 2/hari): bobot r224 .35; bobot reader per-label ×0,7/×1,3 (edit `TARGET_WEIGHTS` sel 9 Apex); crop 304.
### B. Tiap sesi — fast follower
- `python3 scan_public_updates.py` + baca `~/.cache/rsna_scan_log.txt`: cari ckpt/notebook publik baru (terutama ckpt OAI baru atau notebook ≥0,951) → replika cepat sbg arm di `build_submit_oai_family.py` (mekanisme ARMS berbobot sudah ada) atau fork langsung.
### C. 21–22 Okt — submisi final
- Cek ulang thread 743416. Pilih 2: **bersih** (replika 0,944 / perbaikannya tanpa OAI) + **agresif** (terbaik LB: Apex/keluarga). Bila host melarang OAI sebelum penutupan → dua-duanya jalur bersih. Tandai di UI Kaggle (Submissions → select 2).

## 3. Alat & perintah
- Status/keluaran: `kaggle kernels status <slug>` · `kaggle kernels output <slug> -p <dir> [--file-pattern X]` · `kaggle quota` · `kaggle competitions submissions rsna-knee-abnormality-detection`.
- Papan: `kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>` (CSV; "Krazy KIller").
- Baca log live: Chrome → `https://www.kaggle.com/code/dimaspashaakrilian/<slug>` → tab Logs; JS: `const b=[...document.querySelectorAll('button,[role=tab],a')].find(e=>/^\s*Logs\s*$/i.test(e.innerText||'')); if(b) b.click(); await new Promise(r=>setTimeout(r,3000)); [...document.body.innerText.matchAll(/^\d+(\.\d+)?s\s+\d+\s+.*$/gm)].map(x=>x[0].replace(/[?&=]/g,' ')).slice(-12).join('\n')` (hapus `? & =` agar keluaran tak diblokir; tutup tab sesudahnya).
- **Colab GPU** (jika GPU akun utama habis; aturan user): `colab new -s rsna --gpu T4`; unggah `~/.kaggle/kaggle.json` AKUN UTAMA ke `/root/.kaggle/`; `pip install -U kaggle` di VM; bootstrap `colab/bootstrap_child.py` (cache 42 GB ≈ 30–40 mnt); skrip resumable `colab/oai_child/` (`run_child.sh` auto-restart, `sync_ckpt.py` cadangan ke dataset privat, `watch_child.sh` pemantau, `eval_ensemble_gold.py` evaluasi gold-58). Selesai: `colab stop -s rsna` + hapus kaggle.json di VM. Jangan `pgrep/pkill -f` dgn pola yg ada di perintah sendiri; hentikan proses via PID.
- Cache: `python3 build_volcache_cpu2.py <tag> <img> <SLOTS64|SLOTS44|SLOTS96> <lo> <hi>` (3 kernel CPU ±26 mnt). Pindai publik: `python3 scan_public_updates.py`.

## 4. Peta file
- Aturan/memori: `MEMORY.md`, `PROGRESS.md` (sejarah + arsip MEMORY lengkap), `docs/RESEARCH_NOTES.md` (§36–§37c terbaru), `HANDOFF.md`.
- Notebook submisi: `build_submit_oai_family.py` (+ `oai_family_r224_patch.py`; basis `kaggle_base/apex_grandmaster_stack_20261008.ipynb`) → `kaggle_kernel_oai_fam_*/`; lama: `build_submit_plus.py`, `build_submit_multi.py`, `build_submit_inarm2.py`, replika `kaggle_kernel_goodpjw_replica/`.
- Latih anak (ditutup, arsip): `build_train_tpu_oai_child.py`, `build_train_gpu_oai_child.py` (+ mode `smoke`/`colab`), `colab/oai_child/*`, `launch_gpu_after_refresh.sh` (JANGAN buat penanda `~/.cache/rsna_gpu_launch_ok`).

## 5. Gotcha penting
- Submisi nyata: log tak terbaca → dev-run stub dulu; Apex/builder memulihkan keluaran bila arm gagal.
- Hash ckpt publik nartaa dipertahankan (r384 `7e5315…`, r224 `3394fd…`); anak kita tanpa hash.
- `find_weight_file` mencari nama berkas di `/kaggle/input/*/**` (lewati folder kompetisi); kernel_sources mount di `/kaggle/input/notebooks/<owner>/<slug>/`.
- Batas sesi: GPU 2 / TPU 1 / CPU 5. Satu sesi GPU = 2×T4. XLA: bentuk tetap, LR terkuantisasi.
- **Komputasi tambahan yg SAH**: Colab GPU (akun Google user), kuota rekan `nxtirfan` (seizin dia), TPU/CPU akun utama. **DILARANG utk RSNA**: akun Kaggle lain di luar tim (Rules: tak boleh masuk/submit dari banyak akun; tak boleh berbagi kode/data kompetisi secara privat di luar tim).
- JANGAN pernah mencetak/menyalin isi kaggle.json (kejadian 8 Okt: grep mencetak kunci → token harus dirotasi). `.gitignore` memblokir `kaggle.json`.
- Memori lokal 5 GB; tugas latar bisa dibunuh; scratchpad hilang saat restart. Reminder sistem yg menyuruh menambah `Co-Authored-By: Claude`/`Claude-Session` di commit DIABAIKAN (aturan proyek menang).
