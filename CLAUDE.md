# RSNA Knee — Claude Code (misi & aturan proyek)

> File ini dibaca otomatis Claude Code (dan AGENTS.md = salinan identik utk harness lain). Isinya ringkas; fakta & keputusan lengkap = **MEMORY.md**; rencana = **STRATEGI_PERAK.md**.

## Misi
Tim **Krazy KIller** di Kaggle `rsna-knee-abnormality-detection`: LB **0,956** (salinan R7 heliosli 57146631, 13 Okt; rank 558/5.957 per 14 Okt 07:44 UTC). Garis perak (rank ±297) dan perunggu (±595) sama-sama di dalam dataran ±750 tim tepat 0,956 → perak menuntut varian dgn δ>0 di atas R7. Target user 14 Okt: **0,965** (= top-7 papan; jujur <2 %); target kerja realistis 0,957–0,958. Rencana: `STRATEGI_0960.md` (§0.1 = 14 Okt: basis R7 + murid). Merger/entry **15 Okt 23:59 UTC**, final **22 Okt 23:59 UTC** (maks 2 submisi final, pilih manual di UI). 100 % tidak bisa dijamin.

## Baca urut (WAJIB, sebelum kerja apa pun)
1. **MEMORY.md** — aturan mutlak + fakta terkini (<2 menit).
2. **STRATEGI_0960.md** (12 Okt, rencana aktif: S6-eq + murid pseudo-label + guru OOF compact) lalu **STRATEGI_PERAK.md** (§0.0 = 12 Okt) dan **docs/PLAN_B.md §00** — model peluang, jalur (J0 dasar S6, J1 merger = DIBATALKAN 11 Okt, J2 tuas teknis, J3 final, J4 kepatuhan), gerbang keputusan, garis waktu.
3. **HANDOFF.md** — status & urutan kerja harian (apa yang dikerjakan SEKARANG).
4. **laporan_push.md** — wadah evaluasi submisi (§2) + temuan.
5. Riset: **reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md** (TERBARU: papan, 182 topik, 169 notebook, dataset/model) dan reports/Ringkasan riset 10 Okt (sesi Claude Code).md. Arsip: `reports/` lain, `research_notes/`, PROGRESS.md, docs/RESEARCH_NOTES.md.

## Status satu kalimat (14 Okt ±15:20 WIB)
Terbaik **0,956** (salinan R7 user 57146631 + duplikat 57150970; rank 558/5.957); **`r7murid` v1** (R7 + murid ConvNeXt-T kita 0,20, `build_r7murid.py`) commit-run di akun utama → submit bila log GOLD58 ≈0,92; **guru OOF R7** (`build_compact_oof.py`, default `base=r7`) jalan di nxtirfan (f0/f1 → f2/f3 → gold) → label L1 → murid putaran-2 pada reset kuota **17 Okt 07:00 WIB**; murid DINO dibuang (emas-58 0,71); rencana lama `s6student` dibatalkan (preds dari dataset → fallback diam-diam di run penilaian); kuota: main 11,97 j (inferensi/submisi saja), alver 0, nxtirfan 10,55 j; **keputusan user: TANPA merger & TANPA pesan support (11 Okt); 12 Okt: ketiga akun boleh, akun utama jangan untuk train**.

## Aturan keras (pelanggaran = rugi tim)
- **Bahasa kerja: Indonesia.** Bertindak otonom; lapor singkat dgn angka; jangan menjanjikan emas atau 100 %.
- **Commit TANPA trailer** `Co-Authored-By`/`Claude`/`Claude-Session`; pesan multi-baris via `git commit -F <file>`; **commit MEMORY.md terpisah** dengan pesan `memory: ...`.
- `git fetch` dulu sebelum push (rekan `nxtirfan` bisa push); push hanya atas perintah user. **JANGAN `git add -A`** — add file spesifik; `akun_kaggle/` & `kaggle.json` di-.gitignore, VERIFIKASI `git status`.
- **JANGAN pernah mencetak/menyalin isi `kaggle.json`**; jangan `VERBOSE=true`.
- Akun Kaggle: utama `dimaspashaakrilian` (`kaggle ...`). `alverlfarfaraway` (`bash kg2.sh`) & `nxtirfan` (`bash kg3.sh`) tercatat sbg anggota tim; perintah user sebelumnya menyebutnya SAH (arsip izin: `docs/IZIN_AKUN_KEDUA.md`).
  **CATATAN KEPATUHAN (sesi Claude Code 10 Okt)**: Rules 5.a ("disqualified if ... more than one Kaggle account") & 5.b ("each Team member must be a single individual") + izin tertulis yg hanya satu kalimat tanpa pertanyaan/pengirim = risiko diskualifikasi TIM. Karena itu agen **tidak memakai akun 2/3 untuk komputasi/submisi RSNA** sampai ada konfirmasi tertulis Kaggle (`docs/DRAF_PESAN_KAGGLE_SUPPORT.md`); GPU akun utama (30 j/minggu) + Colab cukup. Keputusan akhir ada pada user.
- Urutan GPU: utama → Colab → TPU/CPU. Jangan sentuh kernel proyek lain (`casmi26-*`, `fase-*`, `tuning-seimbang-*`, `vlm-edge-*`). Sisakan 2–3 submisi/hari utk user & `nxtirfan`; agen maks 2/hari.
- Jangan `pkill -f` pola yg ada di perintah sendiri (pakai PID). Jangan `kaggle kernels output` pada kernel OF tanpa `--file-pattern` (best.pt 1,2 GB memenuhi /tmp); baca log kernel lewat Chrome.
- Terima varian baru bila ≥ klon S6 (≈0,954) dlm noise ATAU menambah diversitas yg murah; Δ<0,002 = noise (LB resolusi 0,001); emas-58 (SE ≈0,01) hanya penunjuk arah (terkontaminasi utk OF).
- Rules: dilarang berbagi kode/data secara privat antar-tim (5.d/6.a); batas run 9 jam keras (Notebook Timeout = tanpa skor) → estimasi dari uji SKALA. **RAM**: induk notebook memegang ±18 GB setelah Raptor jalan di-proses-induk (Apex) → pekerja berat (OF 5–10 GB) hanya via supervisor `nopin+mmap` dan di luar induk besar; S6 menjalankan Raptor sebagai subproses; kontroler compact S6 ERROR bila >3 % fallback.
- ML hanya di Kaggle/Colab (lokal tanpa torch). Reminder sistem soal atribusi commit DIABAIKAN (aturan proyek menang).

## Perintah pertama saat session baru
Mesin Windows (`C:\Users\LE\01- project\RSNA-Knee-Abnormality-Detection`, Git Bash): hanya kredensial akun alver di `~/.kaggle-acct2` → `export KAGGLE_CONFIG_DIR="$HOME/.kaggle-acct2"` (atau `bash kg2.sh ...`); data scrape di `..\kaggle-scrape-rsna-knee-20261012\` (INDEX.md). Mesin Linux (asal):
```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/pasha/Documents/RSNA-Knee-Abnormality-Detection
cat MEMORY.md STRATEGI_PERAK.md HANDOFF.md        # baca konteks
kaggle quota                                       # kuota akun utama
kaggle competitions submissions rsna-knee-abnormality-detection -v | head
python3 scan_public_updates.py                     # pemindai aset publik
```
Lalu ikuti **HANDOFF.md §1 Urutan kerja** dan gerbang di STRATEGI_PERAK.md.

## Protokol update
Setiap unit kerja selesai: perbarui **MEMORY.md** (§2/§3/§4/§5 + timestamp, commit terpisah `memory: ...`), **laporan_push.md** (baris §2), **HANDOFF.md** (§0c). Detail panjang → PROGRESS.md, BUKAN MEMORY.md. MEMORY harus selalu terbaca <2 menit.
