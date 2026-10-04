#!/usr/bin/env python3
"""CoAtNet dari init IMAGENET (bukan dari c5) di label fusi -- keluarga model INDEPENDEN.

ALASAN (RESEARCH_NOTES §35): semua fine-tune c5 (v3, putaran-2, 4 murid fold, soup) berkorelasi tinggi dgn c5
stok (0,94-0,99) dan ensembel-nya jenuh di ~0,93 gold. Simulasi §30: tambahan harus KUAT dan TAK BERKORELASI.
Satu-satunya cara memutus ketergantungan pada c5 dgn sumber daya kita: melatih dari bobot ImageNet
(`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, dataset mohamed3abdelrazik/...) dgn label fusi 0,9348 dan waktu latih
panjang (6-8 epoch, SWA 3 epoch terakhir). Kuncinya apakah kualitasnya >= ~0,91-0,92; diukur di gold-58.

Dibuat dari builder putaran-2 (GPU/TPU) lewat penggantian teks yang di-assert:
  - init: bb ImageNet (strict=False tapi missing_keys WAJIB kosong), kepala Raptor baru
  - LR: head 2e-3, backbone 5e-5 (init ImageNet butuh LR lebih besar dari fine-tune c5), WD 0,05
  - seed dari env SEED; SWA (rata-rata bobot akhir 3 epoch terakhir, dihitung di CPU)
  - gold-58 dinilai tiap epoch HANYA utk diagnostik (tak dipakai seleksi: checkpoint final & SWA dua-duanya disimpan)
Pemakaian: python3 build_train_coatnet_imagenet.py <gpu|tpu> <epochs> <seed>
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEV = sys.argv[1] if len(sys.argv) > 1 else "gpu"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
assert DEV in ("gpu", "tpu")

# basis: skrip builder putaran-2 c5 (versi GPU atau TPU) -- dihasilkan ulang dulu
helper = "build_train_coatnet_round2_tpu.py" if DEV == "tpu" else "build_train_coatnet_round2.py"
subprocess.run([sys.executable, str(REPO / helper), "1", "c5", "4"], check=True, capture_output=True, cwd=REPO)
base_name = "build_train_tpu_round2_c5_r1.py" if DEV == "tpu" else "build_train_round2_c5_r1.py"
src = (REPO / base_name).read_text()


def must(text, old, new):
    assert old in text, f"pola tidak ketemu: {old[:100]!r}"
    return text.replace(old, new, 1)


# ---- (a) init ImageNet menggantikan pemuatan checkpoint c5 ----
INIT_NEW = '''import glob as _g2
_hits_w = [p for d in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/') for p in _g2.glob(d + 'model.safetensors')]
_hits_w = [p for p in _hits_w if 'coatnet' in p.lower()]
tlog('bobot ImageNet: ' + repr(_hits_w))
assert len(_hits_w) == 1, f'bobot ImageNet harus tepat 1: {_hits_w}'
from safetensors.torch import load_file as _lf
_bb = KE['build_backbone']('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False)
_res = _bb.load_state_dict(_lf(_hits_w[0]), strict=False)
tlog(f'init ImageNet: missing={list(_res.missing_keys)[:6]} unexpected={list(_res.unexpected_keys)[:6]}')
assert len(_res.missing_keys) == 0, 'bobot ImageNet tidak lengkap'
model = KE['RaptorClassifier'](_bb, F_dim=_bb.num_features); RES = 384
MODEL_TO = DEV_PLACEHOLDER
'''
if DEV == "tpu":
    old_init = ("wp = KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')\n"
                "model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, 'cpu')\n"
                "model = model.to(DEV)\n")
    init_new = INIT_NEW.replace("MODEL_TO = DEV_PLACEHOLDER\n", "model = model.to(DEV)\n")
else:
    old_init = ("wp = KE['find_weight_file']('raptor_ft_coatnet_v5_full_swa.pt')\n"
                "model, RES = KE['load_model'](wp, 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, DEV)\n")
    init_new = INIT_NEW.replace("MODEL_TO = DEV_PLACEHOLDER\n", "model = model.to(DEV)\n")
src = must(src, old_init, init_new)

# ---- (b) LR, epoch, seed ----
src = must(src, "LR_HEAD, LR_BB, WD = 1e-3, 2e-5, 0.05", "LR_HEAD, LR_BB, WD = 2e-3, 5e-5, 0.05")
src = must(src, "EPOCHS = int(os.environ.get('EPOCHS', '4'))", f"EPOCHS = int(os.environ.get('EPOCHS', '{EPOCHS}'))")
src = must(src, "DEV = 'cuda'" if DEV == "gpu" else "DEV = xm.xla_device()",
           (("DEV = 'cuda'" if DEV == "gpu" else "DEV = xm.xla_device()") +
            f"\nimport random as _rnd\nSEED = {SEED}\nnp.random.seed(SEED); _rnd.seed(SEED); torch.manual_seed(SEED)"))

# ---- (c) SWA + diagnostik gold per epoch: ganti baris penutup epoch ----
EPOCH_LINE = "    tlog(f'=== EPOCH {ep+1}/{EPOCHS} selesai | loss={ep_loss/max(1,len(order)):.4f}')"
SWA_NEW = EPOCH_LINE + '''
    # ---- SWA: rata-rata bobot akhir 3 epoch terakhir (di CPU) ----
    if ep >= EPOCHS - 3:
        _sd = {k_: v_.detach().cpu().float() if v_.is_floating_point() else v_.detach().cpu() for k_, v_ in model.state_dict().items()}
        if SWA_N == 0:
            SWA_SUM = {k_: v_.clone() for k_, v_ in _sd.items()}
        else:
            for k_, v_ in _sd.items():
                if v_.is_floating_point(): SWA_SUM[k_] += v_
        SWA_N += 1
        tlog(f'    SWA: akumulasi {SWA_N} snapshot')
    _model_mode = model.training
    model.eval()
    _pg = []
    with torch.no_grad():
        for _sid in sorted(GOLD):
            try:
                _v, _m = load_study(_sid)
                _xw = KE['eval_windows'](_v, _m, k=KE['K_EVAL'], res=RES, norm=KE['NORM'])
                _pg.append(KE['infer_probs'](model, _xw, DEV))
            except Exception:
                _pg.append(np.full(N_T, 0.5, np.float32))
    _Pg = np.stack(_pg)
    _yg = train.set_index('StudyInstanceUID').loc[sorted(GOLD), LAB].values.astype(float)
    def _a1(yy, ss):
        pos, neg = ss[yy == 1], ss[yy == 0]
        r = pd.Series(np.concatenate([pos, neg])).rank().values
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    tlog(f'    [diagnostik] gold-58 setelah epoch {ep+1}: {np.mean([_a1(_yg[:, j], _Pg[:, j]) for j in range(N_T)]):.4f}')
    if _model_mode: model.train()'''
src = must(src, EPOCH_LINE, SWA_NEW)
# inisialisasi akumulator SWA sebelum loop epoch
src = must(src, "step = 0\nfor ep in range(EPOCHS):", "SWA_N = 0; SWA_SUM = None\nstep = 0\nfor ep in range(EPOCHS):")

# ---- (d) ekor: evaluasi final + SWA, simpan dua checkpoint ----
a = src.index("# ===== evaluasi gold-58 (model FINAL, tanpa seleksi dr gold) =====")
b_mark = "tlog('TRAINING DONE')"
b = src.index(b_mark) + len(b_mark)
TAIL = '''# ===== evaluasi gold-58: model FINAL dan model SWA (keduanya dilaporkan, tak ada seleksi) =====
def eval_gold(tag):
    model.eval(); preds = []
    with torch.no_grad():
        for sid in sorted(GOLD):
            try:
                vol, mask = load_study(sid)
                xw = KE['eval_windows'](vol, mask, k=KE['K_EVAL'], res=RES, norm=KE['NORM'])
                preds.append(KE['infer_probs'](model, xw, DEV))
            except Exception as e:
                preds.append(np.full(N_T, 0.5, np.float32)); tlog(f'eval fail {sid[:20]}: {e}')
    P = np.stack(preds)
    out = pd.DataFrame(P, columns=LAB); out.insert(0, 'StudyInstanceUID', sorted(GOLD))
    out.to_csv(f'/kaggle/working/inet_{tag}_gold_preds.csv', index=False)
    ytrue = train.set_index('StudyInstanceUID').loc[sorted(GOLD), LAB].values.astype(float)
    def auc1(yy, ss):
        pos, neg = ss[yy == 1], ss[yy == 0]
        r = pd.Series(np.concatenate([pos, neg])).rank().values
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    aucs = [auc1(ytrue[:, j], P[:, j]) for j in range(N_T)]
    tlog(f'MACRO AUC gold-58 [{tag}] (init ImageNet, label fusi): {np.mean(aucs):.4f}   [acuan: c5 stok 0.9213, v3 0.9234, 4-murid-fold 0.9286]')
    for l_, v_ in zip(LAB, aucs): tlog(f'    {l_:18s} {v_:.4f}')

def _save(tag):
    torch.save({'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': RES,
                'model': {k_: v_.detach().cpu() for k_, v_ in model.state_dict().items()}},
               f'/kaggle/working/coatnet_inet_{tag}.pt')

eval_gold('final'); _save('final')
if SWA_N >= 2:
    _cur = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}
    _avg = {k_: ((SWA_SUM[k_] / SWA_N).to(_cur[k_].dtype) if SWA_SUM[k_].is_floating_point() else _cur[k_]) for k_ in SWA_SUM}
    model.load_state_dict({k_: v_.to(DEV) if False else v_ for k_, v_ in _avg.items()})
    model.to(DEV)
    eval_gold('swa'); _save('swa')
tlog('TRAINING DONE')'''
src = src[:a] + TAIL + src[b:]

# ---- identitas kernel & dataset bobot ImageNet ----
tag = f"inet-{DEV}-s{SEED}"
slug_old = "rsna-knee-train-round2-c5-r1" if DEV == "gpu" else "rsna-knee-train-tpu-round2-c5-r1"
dir_old = "kaggle_kernel_train_round2_c5_r1" if DEV == "gpu" else "kaggle_kernel_train_tpu_round2_c5_r1"
src = src.replace(slug_old, f"rsna-knee-train-{tag}").replace(dir_old, f"kaggle_kernel_train_{tag.replace('-', '_')}")
src = src.replace('"title": "RSNA Knee - Train round2-c5-r1"', f'"title": "RSNA Knee - Train {tag}"')
src = src.replace('"title": "RSNA Knee - Train TPU round2-c5-r1"', f'"title": "RSNA Knee - Train TPU {tag}"')
src = must(src, '"dreaddevelopment/raptor-knee-widedense",',
           '"dreaddevelopment/raptor-knee-widedense",\n        "mohamed3abdelrazik/coatnet-rmlp-2-rw-384-sw-in12k-ft-in1k",')
src = src.replace("coatnet_fused_r1_c5.pt", f"unused_{tag}.pt")

out_py = REPO / f"build_train_{tag.replace('-', '_')}.py"
out_py.write_text(src)
r = subprocess.run([sys.executable, str(out_py)], capture_output=True, text=True, cwd=REPO)
print(r.stdout.strip() or r.stderr.strip()[-1500:])
mp = REPO / f"kaggle_kernel_train_{tag.replace('-', '_')}" / "kernel-metadata.json"
if mp.is_file():
    m = json.loads(mp.read_text()); m["dataset_sources"] = list(dict.fromkeys(m["dataset_sources"])); mp.write_text(json.dumps(m, indent=2))
    print("dataset_sources:", m["dataset_sources"]); print("kernel_sources:", m.get("kernel_sources"))
