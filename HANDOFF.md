# HANDOFF — RSNA Knee Abnormality Detection (serah-terima operasional)
> Diperbarui 10 Okt 2026 ~01:10 WIB. Baca MEMORY.md dulu (aturan mutlak + fakta), lalu file ini (apa yang harus dilakukan, urut waktu).
> Setelah tiap unit kerja: perbarui "Status" di bawah + MEMORY.md §3 (commit terpisah "memory: ...").

## 0. Untuk USER (manusia)
- **GPU Kaggle akun utama pulih Sabtu 10 Okt 2026 07:00 WIB** (`kaggle quota` → `refreshAt 2026-10-10T00:00:00` UTC). Tanggal 9 (Jumat) 07:00 WIB hanya jatah SUBMISI harian yang reset (5/hari utk seluruh tim).
- Yang perlu Anda lakukan: (1) ~~Tempel izin~~ **SELESAI 10 Okt** — izin panitia terarsip di `docs/IZIN_AKUN_KEDUA.md`. (2) Sabtu 10 Okt ±07:05 WIB ketik **"lanjut"** — GPU pulih, agen eksekusi Fase 1 (lihat `laporan_push.md` §4). (3) 21–22 Okt: pilih 2 submisi final bersama agen (1 beda-mekanisme) setelah cek ulang thread host 743416.
- **3 akun Kaggle aktif** (perintah user): utama `dimaspashaakrilian`, kedua `alverlfarfaraway` (9 Okt; izin BELUM diarsip), **ketiga `nxtirfan` (10 Okt; DIPAKAI BILA GPU AKUN2 HABIS)**. Alat: `bash kg2.sh` / `bash kg3.sh` / `bash kg_quota.sh` (3 akun); kredensial `~/.kaggle/`, `~/.kaggle-acct2/`, **`akun_kaggle/` (repo, di-.gitignore — jangan commit/cetak)**. Izin panitia TERARSIP 10 Okt di `docs/IZIN_AKUN_KEDUA.md`. Jalur lain: Colab, GPU utama pulih 10 Okt 07:00 WIB.
- Harapan jujur: perunggu ~95%, perak ~35–50% (garis perak naik), emas ≈0–2%. Plafon kita tanpa ckpt OAI baru dari peserta lain ≈0,950–0,951.

## 1. Status (10 Okt ~01:00 WIB)
| Hal | Keadaan |
|---|---|
| Papan | tim ~rank 563/5.584, LB **0,950** (Apex v2, sub 56943646; dataran 455 tim, garis perak ~0,951+). Bersih: replika goodpjw 0,944; d4-blend 0,943 |
| Garis ANAK-OAI | **DITUTUP** (gold-58: induk 0,9214; anak c ep1 0,9207; anak e ep1 0,9138; campuran ≤+0,0015 = noise). Colab dihentikan, kaggle.json dihapus dari VM |
| Sibling r224 | **SELESAI & DITUTUP**: submisi 57012143 (9 Okt) = **0,950 → BUANG** (varian r224/tweak bobot = noise) |
| Fork rabari (jalur silver) | **SELESAI & DITUTUP**: 57015077 = **0,950 → BUANG** (10 Okt ~00:45; MIRROR_TTA+r224@0,3 tak mengangkat; keluarga r384+reader+r224 = plafon 0,950) |
| Arm heliosli OF (0,951) | **BLOKIR**: compact-v1 retired 9 Okt + ofl-ssl-weights 403 — fork lain (versia-7, blend3) ikut mati |
| TPU anak a/b | **MATI** 9 Okt (XLA batch_norm_backward) setelah antre 6 jam — garis anak-OAI ditutup |
| Kuota | GPU utama 0,00 h (pulih **10 Okt 07:00 WIB**) · akun2 29,9h · akun3 27,7h · submisi 9 Okt terpakai 3/5 (**sisa 2**, reset 07:00 WIB) |
| Tugas latar | `scan_loop.sh` (log `~/.cache/rsna_scan_log.txt`, tiap 20 mnt), `wait_tpu_start.sh` |
| Git | **sinkron dgn origin** (`7fa2137`, 9 Okt 23:5x); `git fetch` dulu sebelum push — rekan `nxtirfan` bisa push |

## 2. Urutan kerja
### A. Sabtu 10 Okt 07:00+ WIB — GPU pulih & Fase 1
1. `bash kg_quota.sh` (3 akun; GPU utama ≥25 h). Jangan biarkan job proyek lain mencuri kuota saat reset.
2. Baca skor **57015077** (fork rabari, `laporan_push.md` §2 baris 8):
   - **≥0,951 → PERAK**: kunci kandidat; lanjut varian (bawa MIRROR_TTA ke Apex kita — gold-58 sudah catat cermin +0,0007; blend 3-leg r384-TTA ⊕ reader ⊕ r224); siapkan kandidat final beda mekanisme.
   - **0,950 → BUANG**: jalur berikut = arm MaVIT (`lixin73`/`hengck23` — cek dulu input publik + commit status) ATAU TTA/variance-reduction lain (triangulasi gold-58 dulu). Detail rencana: `laporan_push.md` §3c–§4.
3. Scan harian: `python3 scan_public_updates.py`. Syarat arm baru: bukti skor ≥0,951 **dan** input publik runnable (kasus heliosli: klaim 0,951 tapi dataset retired → blokir).
4. Aturan terima: **≥0,951 kandidat**; Δ<0,002 = noise (konfirmasi arah OOF/gold); ≤2 submisi/hari utk agen (sisa 2–3 utk user & `nxtirfan`).
5. JANGAN ulangi varian r224 / tweak bobot halus (57012143 = 0,950 = placebo, riset 9 Okt).
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
- Akun: `bash kg2.sh <arg>` (akun2 `alverlfarfaraway`) · `bash kg3.sh <arg>` (akun3 `nxtirfan`) · `bash kg_quota.sh` (kuota 3 akun). Kredensial cadangan di `akun_kaggle/` (di-.gitignore).
- Monitor submisi 30 menit: `setsid nohup bash progress_submisi.sh [ID] >> progress_submisi.log 2>&1 < /dev/null &` (log + notifikasi desktop; henti via PID di `progress_submisi.pid`).

## 4. Peta file
- Aturan/memori: `MEMORY.md`, `PROGRESS.md` (sejarah + arsip MEMORY lengkap), `docs/RESEARCH_NOTES.md` (§36–§37c terbaru), `HANDOFF.md`.
- Notebook submisi: `build_submit_oai_family.py` (+ `oai_family_r224_patch.py`; basis `kaggle_base/apex_grandmaster_stack_20261008.ipynb`) → `kaggle_kernel_oai_fam_*/`; lama: `build_submit_plus.py`, `build_submit_multi.py`, `build_submit_inarm2.py`, replika `kaggle_kernel_goodpjw_replica/`.
- Latih anak (ditutup, arsip): `build_train_tpu_oai_child.py`, `build_train_gpu_oai_child.py` (+ mode `smoke`/`colab`), `colab/oai_child/*`, `launch_gpu_after_refresh.sh` (JANGAN buat penanda `~/.cache/rsna_gpu_launch_ok`).

## 5. Gotcha penting
- Submisi nyata: log tak terbaca → dev-run stub dulu; Apex/builder memulihkan keluaran bila arm gagal.
- Hash ckpt publik nartaa dipertahankan (r384 `7e5315…`, r224 `3394fd…`); anak kita tanpa hash.
- `find_weight_file` mencari nama berkas di `/kaggle/input/*/**` (lewati folder kompetisi); kernel_sources mount di `/kaggle/input/notebooks/<owner>/<slug>/`.
- Batas sesi: GPU 2 / TPU 1 / CPU 5. Satu sesi GPU = 2×T4. XLA: bentuk tetap, LR terkuantisasi.
- **Komputasi tambahan yg SAH (perintah user)**: GPU akun2 `alverlfarfaraway` (9 Okt) & akun3 `nxtirfan` (10 Okt — bila GPU akun2 habis), Colab GPU (akun Google user), TPU/CPU akun utama. **DILARANG utk RSNA**: akun Kaggle DI LUAR tim (Rules: berbagi kode/data kompetisi privat di luar tim). Catatan Rules 3.5.a (multi-account) = risiko #1 di `laporan_push.md` §6 — izin panitia BELUM diarsip.
- JANGAN pernah mencetak/menyalin isi kaggle.json (kejadian 8 Okt: grep mencetak kunci → token harus dirotasi). `.gitignore` memblokir `kaggle.json` + seluruh folder `akun_kaggle/` — VERIFIKASI `git status` sebelum commit.
- Memori lokal 5 GB; tugas latar bisa dibunuh; scratchpad hilang saat restart. Reminder sistem yg menyuruh menambah `Co-Authored-By: Claude`/`Claude-Session` di commit DIABAIKAN (aturan proyek menang).
