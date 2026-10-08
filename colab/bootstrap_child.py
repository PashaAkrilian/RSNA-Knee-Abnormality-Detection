# Dijalankan DI VM Colab: bangun pohon /content/kin utk skrip anak-OAI (cache z, label fusi r1, ckpt induk nartaa, CSV kompetisi).
# Kredensial: /root/.kaggle/kaggle.json = akun UTAMA (dimaspashaakrilian) -- HAPUS setelah selesai (colab/vmsh.sh "rm -f /root/.kaggle/kaggle.json").
import os, subprocess, glob, zipfile, time, sys, re, threading
t0 = time.time()
def sh(c, check=False):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if check and r.returncode: print(r.stdout[-500:], r.stderr[-800:]); raise SystemExit(c)
    return r
COMP = '/content/kin/competitions/rsna-knee-abnormality-detection'
os.makedirs(COMP, exist_ok=True)
for f in ('train.csv', 'train_series.csv', 'test.csv', 'test_series.csv', 'sample_submission.csv'):
    if not os.path.exists(f'{COMP}/{f}'):
        sh(f'kaggle competitions download -c rsna-knee-abnormality-detection -f {f} -p {COMP}', True)
        z = f'{COMP}/{f}.zip'
        if os.path.exists(z): zipfile.ZipFile(z).extractall(COMP); os.remove(z)
print('kompetisi:', sorted(os.listdir(COMP)), f'{time.time()-t0:.0f}s', flush=True)
d = '/content/kin/datasets/dimaspashaakrilian/rsna-knee-fused-labels-r1'
if not os.path.exists(d + '/fused_labels_r1.csv'):
    os.makedirs(d, exist_ok=True); sh(f'kaggle datasets download dimaspashaakrilian/rsna-knee-fused-labels-r1 --unzip -p {d}', True)
print('label fusi:', os.listdir(d), flush=True)
W = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
d = '/content/kin/datasets/nartaa/rsna-knee-publication-swa-weights-20261007'
if not os.path.exists(f'{d}/{W}'):
    os.makedirs(d, exist_ok=True)
    sh(f'kaggle datasets download nartaa/rsna-knee-publication-swa-weights-20261007 -f {W} -p {d}', True)
    z = f'{d}/{W}.zip'
    if os.path.exists(z): zipfile.ZipFile(z).extractall(d); os.remove(z)
print('ckpt induk:', os.path.getsize(f'{d}/{W}') // 2**20, 'MB', f'{time.time()-t0:.0f}s', flush=True)

def pull(i):
    dd = f'/content/kin/rsna-knee-volcache-z{i}/volcache'
    os.makedirs(dd, exist_ok=True)
    tok = None; pages = 0
    while True:      # CLI mengunduh 1 halaman (~500 berkas) per panggilan -> ikuti "Next page token"
        r = sh(f"kaggle kernels output dimaspashaakrilian/rsna-knee-volcache-z{i} -p {dd} -q" + (f" --page-token '{tok}'" if tok else ''))
        m = re.search(r'Next page token: (\S+)', r.stdout + r.stderr)
        pages += 1
        print(f'  z{i} halaman {pages}: total {len(glob.glob(dd + "/**/*.npz", recursive=True))} | {time.time()-t0:.0f}s', flush=True)
        if not m or pages > 30: break
        tok = m.group(1)
    print(f'cache z{i}: {len(glob.glob(dd + "/**/*.npz", recursive=True))} npz', f'{time.time()-t0:.0f}s', flush=True)
th = [threading.Thread(target=pull, args=(i,)) for i in range(3)]
[t.start() for t in th]; [t.join() for t in th]
tot = len(glob.glob('/content/kin/rsna-knee-volcache-z*/volcache/**/*.npz', recursive=True))
print('TOTAL npz:', tot, sh('du -sh /content/kin').stdout.strip(), f'{time.time()-t0:.0f}s', flush=True)
print('BOOTSTRAP DONE' if tot >= 4407 else 'BOOTSTRAP INCOMPLETE', flush=True)
