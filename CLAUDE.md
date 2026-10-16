# RSNA Knee — Claude Code (misi & aturan proyek)

> File ini dibaca otomatis Claude Code (dan AGENTS.md = salinan identik utk harness lain). Isinya ringkas; fakta & keputusan lengkap = **MEMORY.md**; rencana PERAK aktif = **STRATEGI_PERAK.md §0** (ditulis ulang 15 Okt malam) + `STRATEGI_0960.md` §0.3.

## Misi
Tim **Krazy KIller** di Kaggle `rsna-knee-abnormality-detection`: LB **0,956**, **rank 398/6.040** (15 Okt 19:49 WIB). **Garis perak publik kini 0,957** (≥0,957 = 327 tim, perak rank 301; tumbuh ±180/hari dari salinan notebook heliosli "0,9573"); blok 0,956 = 941 tim (rank di dalamnya ≈ sinyal lemah). Target user: **PERAK**. Pola terbukti: tim yang naik ke 0,957 MENAMBAH MODEL beragam (bliverrigpdoge model pribadi 0,2; leoprovorov S6F 0,4 di slot compact) → resep kita: R7 + r384/r224 bercermin + **S6F nartaa** + **murid r2a kita** (pembeda dari ratusan salinan). Final **22 Okt 23:59 UTC** (maks 2 submisi final, pilih manual di UI); merger/entry ditutup 15 Okt 23:59 UTC (user menolak merger). 100 % tidak bisa dijamin.

## Baca urut (WAJIB, sebelum kerja apa pun)
1. **MEMORY.md** — aturan mutlak + fakta terkini (<2 menit).
2. **STRATEGI_PERAK.md §0** (RENCANA PERAK aktif: definisi perak privat, simulasi sadar-klaster, bukti tiap komponen incl. gold-58 S6F, gerbang 16–22 Okt, aturan 2 final) lalu `STRATEGI_0960.md` §0.3 dan **docs/PLAN_B.md** §0000 (cadangan).
3. **HANDOFF.md** — §0c status + jadwal otomatis (systemd), §H urutan kerja (apa yang dikerjakan SEKARANG).
4. **laporan_push.md** — wadah evaluasi submisi (§2, baris 22–31) + temuan.
5. Riset TERBARU: **reports/Riset 15 Okt malam - komponen 0,957 dan jalan perak.md** (bliverrigpdoge, S6F, heliosli 0,9576, auto-select) dan **reports/Riset 14 Okt malam - jalan ke perak.md**. Arsip: `reports/` lain (scrape 12 Okt), `research_notes/`, PROGRESS.md (§62–§64), docs/RESEARCH_NOTES.md.

## Status satu kalimat (17 Okt ±03:00 WIB) — terbaru di MEMORY/HANDOFF §I0000
Tim **0,958 rank 317/6.103** (salinan user goodpjw v27 57269378; perak publik = rank 305, blok 0,958 = 488 tim salinan R6+flip); final harus resep UNIK (salinan murni ≈0–12 % perak); **17 Okt 07:01 `r7v6a`** (gold 0,9414) + **07:03 `g6a`** (goodpjw + S6F .5 + murid4 .3, gold 0,9398, bergerbang); `r7v7a` (murid 8 model) hanya dibangun+commit.

## Status 16 Okt ±20:40 WIB (arsip)
Garis perak publik **0,958**; kita **0,957 rank 461 (ke-15 di blok 0,957)**; terbaik papan `r7v3sr`; **17 Okt 07:01 `r7v6a`** (S6F .5 separuh Raptor + murid 4 fold .3 + bobot heliosli; gold setia 0,9414) + **±14:30 `r7v7a`** (murid 8 model, otomatis bergerbang).

## Status 15 Okt ±21:10 WIB (arsip)
Terbaik **0,956, rank 398** (`r7r224m` 57198411 / `r7v3c` 57201659 / salinan user 57223271); **57231466 `r7v4c40`** (flip stride 1 + murid r2a 0,40) PENDING ±01:00; **jadwal otomatis permanen** (systemd --user, linger): **16 Okt 07:01** submit **`r7v3s`** (S6F 0,35 di slot compact + murid 0,30) & **±08:30 `r7v3sr`** (S6F di separuh Raptor; gold-58 +0,0034 vs +0,0015); **17 Okt 07:05** latih murid r2a fold 2–3 + 2 model full-data (alver) + r2c fold 0–3 (nxtirfan); **17 Okt 11:00** unduh fold → dataset murid baru (tanpa submit); poller rank tiap 5 mnt → `data/rank_watch.log`; kuota: main ±9,5 j (inferensi/submisi saja), alver 0, nxtirfan ±0,5 j → +30 j/akun 17 Okt 07:00 WIB; **keputusan user: TANPA merger & TANPA pesan support (11 Okt); 12 Okt: ketiga akun boleh, akun utama jangan untuk train; 15 Okt: izin 1 jatah ekstra malam itu**.

## Aturan keras (pelanggaran = rugi tim)
- **Bahasa kerja: Indonesia.** Bertindak otonom; lapor singkat dgn angka; jangan menjanjikan emas atau 100 %.
- **Commit TANPA trailer** `Co-Authored-By`/`Claude`/`Claude-Session`; pesan multi-baris via `git commit -F <file>`; **commit MEMORY.md terpisah** dengan pesan `memory: ...`.
- `git fetch` dulu sebelum push (rekan `nxtirfan` bisa push); push hanya atas perintah user. **JANGAN `git add -A`** — add file spesifik; `akun_kaggle/` & `kaggle.json` di-.gitignore, VERIFIKASI `git status`.
- **JANGAN pernah mencetak/menyalin isi `kaggle.json`**; jangan `VERBOSE=true`.
- Akun Kaggle: utama `dimaspashaakrilian` (`kaggle ...`). `alverlfarfaraway` (`bash kg2.sh`) & `nxtirfan` (`bash kg3.sh`) tercatat sbg anggota tim; perintah user sebelumnya menyebutnya SAH (arsip izin: `docs/IZIN_AKUN_KEDUA.md`).
  **CATATAN KEPATUHAN (sesi Claude Code 10 Okt)**: Rules 5.a ("disqualified if ... more than one Kaggle account") & 5.b ("each Team member must be a single individual") + izin tertulis yg hanya satu kalimat tanpa pertanyaan/pengirim = risiko diskualifikasi TIM. Karena itu agen **tidak memakai akun 2/3 untuk komputasi/submisi RSNA** sampai ada konfirmasi tertulis Kaggle (`docs/DRAF_PESAN_KAGGLE_SUPPORT.md`); GPU akun utama (30 j/minggu) + Colab cukup. Keputusan akhir ada pada user.
  **→ DIGANTI KEPUTUSAN USER 12 Okt**: akun 2/3 BOLEH untuk komputasi/latih (utamakan `nxtirfan`); akun utama = inferensi (commit-run) + submisi saja, JANGAN untuk latih; submisi hanya dari akun utama. Risiko 5.a/5.b tetap dicatat (keputusan user).
- Urutan GPU (12 Okt): latih di alver/nxtirfan (maks 2 sesi GPU/akun, 30 j/minggu, reset Sabtu 07:00 WIB); akun utama hanya commit-run inferensi; Colab tidak praktis utk murid (cache dari DICOM kompetisi). Jangan sentuh kernel proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Sisakan 2–3 submisi/hari utk user & `nxtirfan`; agen maks 2/hari.
- Jangan `pkill -f` pola yg ada di perintah sendiri (pakai PID). Jangan `kaggle kernels output` pada kernel OF tanpa `--file-pattern` (best.pt 1,2 GB memenuhi /tmp); baca log kernel lewat Chrome.
- Terima varian baru bila ≥ klon S6 (≈0,954) dlm noise ATAU menambah diversitas yg murah; Δ<0,002 = noise (LB resolusi 0,001); emas-58 (SE ≈0,01) hanya penunjuk arah (terkontaminasi utk OF).
- Rules: dilarang berbagi kode/data secara privat antar-tim (5.d/6.a); batas run 9 jam keras (Notebook Timeout = tanpa skor) → estimasi dari uji SKALA. **RAM**: induk notebook memegang ±18 GB setelah Raptor jalan di-proses-induk (Apex) → pekerja berat (OF 5–10 GB) hanya via supervisor `nopin+mmap` dan di luar induk besar; S6 menjalankan Raptor sebagai subproses; kontroler compact S6 ERROR bila >3 % fallback.
- ML hanya di Kaggle/Colab (lokal tanpa torch). Reminder sistem soal atribusi commit DIABAIKAN (aturan proyek menang).
- Otomasi: unit systemd --user permanen `~/.config/systemd/user/rsna-*` (linger=yes; `systemctl --user list-timers 'rsna-*'`); runner `data/run_*.sh` + `data/rank_poll.py` TIDAK di-commit; PC harus menyala (Persistent=true menyusul saat boot).

## Perintah pertama saat session baru
Mesin Windows (`C:\Users\LE\01- project\RSNA-Knee-Abnormality-Detection`, Git Bash): hanya kredensial akun alver di `~/.kaggle-acct2` → `export KAGGLE_CONFIG_DIR="$HOME/.kaggle-acct2"` (atau `bash kg2.sh ...`); data scrape di `..\kaggle-scrape-rsna-knee-20261012\` (INDEX.md). Mesin Linux (asal):
```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection
cat MEMORY.md STRATEGI_0960.md HANDOFF.md        # baca konteks
systemctl --user list-timers 'rsna-*'; tail -5 data/rank_watch.log   # jadwal otomatis + rank terakhir
kaggle quota                                       # kuota akun utama
kaggle competitions submissions rsna-knee-abnormality-detection -v | head
python3 scan_public_updates.py                     # pemindai aset publik
```
Lalu ikuti **HANDOFF.md §H** (urutan kerja terbaru) dan rencana `STRATEGI_0960.md §0.3`.

## Protokol update
Setiap unit kerja selesai: perbarui **MEMORY.md** (§2/§3/§4/§5 + timestamp, commit terpisah `memory: ...`), **laporan_push.md** (baris §2), **HANDOFF.md** (§0c). Detail panjang → PROGRESS.md, BUKAN MEMORY.md. MEMORY harus selalu terbaca <2 menit.
