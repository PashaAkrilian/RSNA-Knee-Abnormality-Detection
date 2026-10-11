# RSNA Knee — Claude Code (misi & aturan proyek)

> File ini dibaca otomatis Claude Code (dan AGENTS.md = salinan identik utk harness lain). Isinya ringkas; fakta & keputusan lengkap = **MEMORY.md**; rencana = **STRATEGI_PERAK.md**.

## Misi
Tim **Krazy KIller** di Kaggle `rsna-knee-abnormality-detection`: LB **0,950** (rank 599/5.699 per 11 Okt 02:59 WIB; di bawah garis perunggu publik rank 570), target **perak** = top 5 % privat ≈ rank 285. **S6 heliosli (Raptor 0,7 + compact 0,3 = 0,954) kini publik lagi** (11 Okt 02:50 WIB); user sudah men-submit salinannya (57051680). Merger/entry **15 Okt 23:59 UTC**, final **22 Okt 23:59 UTC** (maks 2 submisi final, pilih manual di UI). 100 % tidak bisa dijamin.

## Baca urut (WAJIB, sebelum kerja apa pun)
1. **MEMORY.md** — aturan mutlak + fakta terkini (<2 menit).
2. **STRATEGI_PERAK.md** (§0 = perubahan 11 Okt) dan **docs/PLAN_B.md §00** — rencana, model peluang, jalur (J0 dasar S6, J1 merger = DIBATALKAN 11 Okt, J2 tuas teknis, J3 final, J4 kepatuhan), gerbang keputusan, garis waktu.
3. **HANDOFF.md** — status & urutan kerja harian (apa yang dikerjakan SEKARANG).
4. **laporan_push.md** — wadah evaluasi submisi (§2) + temuan.
5. Riset: **reports/Ringkasan riset 10 Okt (sesi Claude Code).md** (TERBARU). Arsip: `reports/` lain, `research_notes/`, PROGRESS.md, docs/RESEARCH_NOTES.md.

## Status satu kalimat (11 Okt 09:45 WIB)
heliosli memulihkan bundel compact → **S6 (0,954) publik; salinan user 57051680 = 0,954 COMPLETE (rank 144/5.709; kandidat final A)**; 57044355 (Apex+OF) = 0,950 karena OF mati OOM (bukan vonis OF; resep heliosli = Raptor + compact); varian `s6reader` 57055673 + `s6plus` 57055781 di-submit 07:03 WIB (PENDING); **keputusan user 11 Okt: TANPA merger & TANPA pesan support — fokus naikkan skor sendiri** (akun 2/3 tetap tidak dipakai).

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
