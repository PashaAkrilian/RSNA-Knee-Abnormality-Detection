# STRATEGI PERAK — RSNA Knee 2026 (disusun 10 Okt 2026; DIPERBARUI 15 Okt 21:10 WIB — §0.-1 terbaru; sesi Claude Code)
> Baca setelah `MEMORY.md`. Dasar: riset 10–11 Okt (`reports/Ringkasan riset 10 Okt (sesi Claude Code).md`, `docs/RESEARCH_NOTES.md` §36–§44), Rules resmi, papan 11 Okt 02:59 WIB.
> Dokumen ini = RENCANA + ATURAN KEPUTUSAN; status harian ada di `HANDOFF.md` §0c dan `laporan_push.md`. Cadangan/pohon keputusan rinci: `docs/PLAN_B.md` (§00 = yang terbaru).

## 0.-1 Pembaruan 15 Okt malam (sesi Claude Code) — rencana aktif `STRATEGI_0960.md` §0.3
- Papan 15 Okt 19:49 WIB (6.040 tim): ≥0,958 = 132; **≥0,957 = 327 → garis PERAK (rank 301) = 0,957**; blok tepat 0,956 = 941 tim; perunggu (rank 603) = 0,956. **Kita rank 398 (0,956)**.
- Pola terbukti publik: tim naik ke 0,957 dgn MENAMBAH MODEL beragam (bliverrigpdoge model pribadi 0,2; leoprovorov S6F 0,4 di slot compact). Menyetel bobot saja tidak menaikkan.
- Resep kita menuju final: R7 + r384/r224 bercermin + **S6F** (publik) + **murid r2a 4 fold + 2 full-data** (milik kita) ± flip stride 1 ± r2c. Uji 16 Okt: `r7v3s` (S6F + murid) & `r7v3c40` (murid 0,40); 15 Okt malam `r7v4c40` (flip + murid 0,40).
- Peluang perak (perkiraan jujur): hanya salinan publik ±15–25 %; + S6F & murid memberi +0,0005–0,001 nyata ±35–50 %; emas ≈0–2 %. 100 % tidak mungkin. Papan privat (70 %, ±900 studi) mengacak ulang blok salinan.

## 0.0 Pembaruan 12 Okt (sesi Claude Code) — rencana aktif kini `STRATEGI_0960.md`
- Papan 12 Okt 07:29 UTC (5.809 tim): perak rank 290 = **0,954**, perunggu rank 581 = **0,954** (keduanya di dalam dataran ±600 tim tepat 0,954), emas rank 21 = 0,962; ≥0,955 = 182; ≥0,956 = 119; kita rank 466. Garis perak bergerak 0,950 (8 Okt) → 0,954 (12 Okt).
- Notebook publik #1 = **S6-eq** heliosli (Raptor 0,5 + compact 0,5, rank sadar-gagal; bundel `r6cnxfsss` = compact-v1 V13) → tim heliosli 0,956; dibangun ulang `build_s6eq.py` (`s6eq`, `s6eqr` + reader goodpjw), 2 submisi menunggu slot GPU (`scripts/submit_watch.py`).
- Pembeda nyata ke ≥0,957: murid sendiri (ConvNeXt-T/EffNetV2-S 2.5D) dgn pseudo-label OOF; guru = **OOF compact** (4 fold hash `sha1(uid)%4`, gold tak dilatih; `build_compact_oof.py`). Rincian, gerbang, anggaran: `STRATEGI_0960.md`; bukti forum/kode: `reports/Scrape kompetisi 12 Okt 2026 - peta menuju perak.md`.

## 0. Perubahan besar 11 Okt (baca ini dulu)
- **KEPUTUSAN USER (09:45 WIB): J1 merger & tindakan pesan support DIBATALKAN** — fokus naikkan skor sendiri dengan tim ini; semua submisi dari akun utama; arsip `docs/KANDIDAT_MERGER.md`/`docs/DRAF_PESAN_KAGGLE_SUPPORT.md` jangan dieksekusi.
- **S6 heliosli hidup lagi** (02:50 WIB): dataset `heliosli/rsna-knee-compact-v1` publik (V1–V13) + notebook publik "Dataset restored(...)" = **Raptor 0,7 + compact 0,3 = 0,954 satu klik** (thread 748141: "blend my model with yours"). **User men-submit salinannya (57051680, 02:56 WIB) → 0,954 COMPLETE ±06:40 WIB (rank 144 dari 5.709; run penuh ±3 j 45 mnt).**
- **Koreksi premis**: resep heliosli = Raptor + **compact** (bukan OF). OF fold0 belum pernah diukur di LB; prior ≥+0,001 di atas S6 ≤20 %.
- **Penyebab 57044355 = 0,950 (OF tidak jalan)** = OOM (induk Apex 18 GB + pekerja OF 8–10 GB); perbaikan v4.1 (OF dulu/terakhir, supervisor, `pin_memory=False`, `torch.load(mmap=True)`) lulus rehearsal; infrastrukturnya dipakai ulang di `build_s6plus.py`.
- **Papan**: kita rank 599 (0,950) — di bawah garis perunggu publik (rank 570, 0,950); perak rank 285 (0,951); dataran 0,954 akan membengkak (klon S6).

## 1. Sasaran dan ukuran
- Perak = 5 % teratas di papan **privat** (70 % data uji) ≈ rank 285 dari 5.699 tim. Papan publik (30 %) hanya penunjuk (11 Okt 02:59 WIB): ≥0,960 = 31; ≥0,956 = 82; ≥0,954 = 141; ≥0,952 = 229; ≥0,951 = 367; =0,950 = 618 (kita rank 599).
- **100 % tidak bisa dijamin.** Klon S6 berimpit di ±0,954; urutan privat di antara klon ≈ undian + selisih kecil (ketidakdeterminan GPU), jadi perbaikan nyata ≥+0,001 atau anggota tim kuat yang mengubah peluang.

## 2. Dasar kuantitatif (simulasi `silver_prob_sim.py`; rentang = s regresi 0,3–0,7 × σ privat 0,003–0,005)
| Kondisi (skill sejati) | P(perak) | P(perunggu) |
|---|---|---|
| Apex 0,950 | 13–20 % | 25–41 % |
| + naik 0,002 | 25–39 % | 40–66 % |
| + naik 0,004 (≈ S6 0,954) | 40–67 % (tanpa pembengkakan klaster) | 58–85 % |
| tim publik 0,956–0,958 (merger) | 23–82 % | — |
- **Koreksi untuk S6**: bila ratusan klon berimpit di 0,954, P(perak | klon) ≈ (irisan klaster dgn 5 %) / ukuran klaster ≈ 25–45 % (klaster 400–800 tim), BUKAN 40–67 %. Perunggu ≈ 45–70 %.
- **Model "blok klon" (11 Okt)**: ratusan klon S6 memiliki prediksi (hampir) identik → skor privat bersama; blok bergerak sebagai satu unit relatif terhadap tim non-klon, dan di DALAM blok urutan ≈ acak (ketidakdeterminan fp16 GPU ~1e-5; bila persis sama, urutan = waktu submisi → klon TERLAMBAT seperti kita kalah). Varian yg BERBEDA dan sedikit LEBIH BAIK (δ>0) melompati seluruh blok; yg sedikit LEBIH BURUK (δ<0) jatuh di bawah seluruh blok (contoh: tri-hybrid jenillangaliya 0,952 < S6 0,954). Maka yg menentukan = TANDA δ, bukan besarnya: pilih perna dgn prior δ>0 tertinggi (reader goodpjw ≈60–75 %; OF ≈50 %; r224/dino-sejenis ≈ <40 %), hindari tambahan "demi keragaman". Aturan: final A = varian dgn LB publik ≥ klon (selisih berpasangan SE ±0,0005–0,001); final B = klon atau varian ke-2 (peluang minimal satu δ>0 naik). Peluang kasar: klon ≈ (irisan blok dgn 5 %)/ukuran blok ≈ 20–30 %; varian δ>0 ≈ P(δ>0) × P(blok di atas garis) ≈ 45–60 %.
- Dua final dgn galat privat tak berkorelasi menambah ±4–12 poin perak (`docs/PLAN_B.md` §1, §4); S6 + variannya berkorelasi tinggi (ρ≈0,95) → tambahan ≈ +3–6 poin.
- Catatan: trik "kirim 1−p" membuat tim kuat bisa bersembunyi di papan (forum 748011) → garis perak privat bisa lebih tinggi dari yg tampak.

## 3. Jalur (paralel; EV menurun)
### J0 — DASAR S6 (user sudah menjalankan; agen memantau)
- **HASIL: 57051680 = 0,954 (S6 terbukti jalan di run penilaian; kita rank 144; 49 tim tepat di 0,954 → dataran 0,954 mulai rank 102).** Rincian: salinan S6 (Raptor ±0,5 j + compact ±3,5 j [GPU-bound, 2 pekerja ±18–20 dtk/studi/pekerja, terukur pada pseudo-uji 160 studi: 80/80 per pekerja, 0 fallback] → total ±4–4,5 j). Risiko: kontroler compact menolak (ERROR) bila >3 % fallback (OOM/pekerja mati/DEADLINE 8,3 j). Aturan: bila ≈0,954 → final A; bila ≤0,952/ERROR → diagnosis (sonda `rsna-knee-s6probe` + rehearsal), gunakan mode `lenient`.
### J1 — MERGER — **DIBATALKAN (perintah user 11 Okt 09:45; fokus naikkan skor sendiri)** — isi di bawah = arsip saja, jangan dieksekusi
- Rules 5.c: sah bila (i) gabungan ≤5 anggota, (ii) submisi gabungan ≤ jatah satu tim saat permintaan (11 Okt = 335; kita 109 → mitra ≤ ±225), (iii) sebelum batas merger. Rules 5.d/6.a: dilarang berbagi privat sebelum merger.
- Target baru (`docs/KANDIDAT_MERGER.md` §3b): Kelompok B (15 tim ≥0,955 ≤30 submisi, mis. MURAD 0,962/19, NMarkS 0,961/30, "吃白饭的大肥鱼" 0,960/**1 submisi**, UEHA, DoyMed, jjakimoto, madmax0404, LP …), Kelompok C (mencari rekan: Tanuki_boosting, Prateek Grover, Nikita Ageev, heliosli solo 0,954). Pemegang bundel (Kelompok A lama) tak istimewa lagi karena S6 publik.
### J2 — TUAS TEKNIS di atas S6 (semuanya EV kecil: +0,000–0,001; LB resolusi 0,001)
1. **S6 + reader** (`python3 build_s6plus.py name=s6reader reader=1`): perna Raptor = fusi Apex (reader goodpjw, bobot per-temuan); +0,5 j; risiko rendah; prior +0,0005.
2. **S6 + OF terakhir** (`name=s6plus reader=1 of=1 wr=0.55 wc=0.30 wo=0.15`): OF dgn supervisor v4.1 hanya memakai waktu sisa (STOP 7,9 j/KILL 8,25 j sejak NB_T0; coverage ±60–70 %); prior ≥+0,001 ≤20 %; rehearsal `s6plusr` memverifikasi rantai penuh.
3. **Bagging bundel heliosli lain** (V4 ofl_ft8, V7 cnx_s5 0,951, V8 dino_s1, V9 dual): biaya ±3,5–4,5 j GPU/bundel (S6 sudah ±4 j) → hanya bila stride dikurangi; prior +0,0005–0,001.
4. **Final bersih** (tanpa Raptor/OAI): compact + reader; kualitas belum terukur (±0,950–0,952) — asuransi bila host melarang OAI; nilai medali nyata hanya dlm skenario itu.
5. DITUTUP (jangan ulangi): varian r224/bobot/mirror (0,950), MaVIT (0,949), CoAtNet resgated, pseudo-label/anak-OAI (≈ induk), kompilator aturan heliosli, "Tri-Reader 0,954" (rabari/jcgweb/sameerk; keluarga Apex+r224 0,950), klon DINOsaur.
### J3 — FINAL TAHAN GUNCANGAN (21–22 Okt; pilih MANUAL & verifikasi via API; auto-fill §3.18.c)
- Final A = S6 (klon user atau varian terbaik); Final B = varian paling berbeda yg masih kuat (S6+reader/OF, atau bersih). Pilih dari submisi yg SUDAH terskor; variasi baru terakhir ≤20 Okt.
### J4 — KEPATUHAN (melindungi medali)
- Akun kedua/ketiga: Rules 5.a/5.b — semua submisi dari akun utama; pesan ke support = DIBATALKAN (user 11 Okt); submisi tetap akun utama.
- OAI/eksternal: risiko KOLEKTIF seluruh klaster Raptor/S6 (host diam di 743416/747455/744056); Rules 2.6.b: model/data eksternal boleh kecuali dilarang host; 2.8/3.14.a hanya menyangkut kelayakan hadiah. Pertahankan satu final dgn risiko berbeda bila murah.

## 4. Garis waktu dan gerbang (WIB)
| Waktu | Aksi | Gerbang / keputusan |
|---|---|---|
| 11 Okt 03:00–07:00 | rehearsal `apexof4r3` (OF 4 pekerja) dan `s6plusr` (reader+compact+OF); update dokumen | agen tidak mensubmit sebelum 07:00 (jatah hari-ini habis) |
| 11 Okt 07:00+ | jatah baru 5/hari (agen ≤2): submit varian S6+ terpilih; baca skor klon S6 (57051680) ±07:00–08:30 | bandingkan: klon S6 vs varian |
| 11–13 Okt | ~~merger & support~~ DIBATALKAN (user 11 Okt) — fokus varian S6+ & monitor skor | — |
| 13–15 Okt | semua kandidat ter-push sebelum 15 Okt (J1 dibatalkan) | 15 Okt 23:59 UTC = 16 Okt 06:59 WIB |
| 16–20 Okt | iterasi bergerbang (agen ≤2/hari); GPU baru 30 j 17 Okt 07:00 WIB; pantau publikasi baru (heliosli S7?) | variasi TERAKHIR 20 Okt |
| 21–22 Okt | cek ulang 743416; pilih 2 final manual; verifikasi via `kaggle competitions submissions` | 22 Okt 23:59 UTC |

## 5. Anggaran
- GPU akun utama: ±20 j tersisa (17 Okt reset +30 j). Run penilaian tak memakai kuota GPU. Submisi tim: 5/hari (reset 07:00 WIB), agen ≤2, sisakan 2–3 utk user & rekan.
- S6 ±4–4,5 j dari batas 9 j; sisa ±4,5 j dipakai perna tambahan (reader 0,5 j; OF terakhir ±3,4 j).
- Disk /kaggle/working ±19 GB efektif; /tmp & overlay besar.

## 6. Risiko utama
| Risiko | Mitigasi |
|---|---|
| S6 gagal di run penilaian (RAM/DEADLINE/fallback >3 % → ERROR) | sonda/rehearsal dgn log terbaca + monitor RAM; mode `lenient`; klon user = deteksi dini |
| Klaster 0,954 membengkak → perak = undian | varian dgn informasi baru; final B beda mekanisme |
| Host melarang OAI/model eksternal | final bersih (compact + reader); keputusan 21–22 Okt setelah cek ulang |
| Papan privat mengacak dataran | dua final; jangan menyetel ke papan publik; Δ<0,002 = noise |
| Diskualifikasi akun ganda | J4; semua submisi dari akun utama |
| Tim kuat tersembunyi (1−p) | tidak bisa dikendalikan; targetkan skill sejati, bukan hanya publik |
