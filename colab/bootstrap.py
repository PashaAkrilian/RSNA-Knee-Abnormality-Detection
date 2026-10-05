# Dijalankan DI VM Colab: meniru pohon /content/kin spy worker Kaggle jalan tanpa ubahan.
import os, subprocess, glob, zipfile, time, sys
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
for ds in (sys.argv[1:] or ['dimaspashaakrilian/rsna-knee-fused-labels-r1', 'shigechan/radimagenet-resnet50-official']):
    own, slug = ds.split('/')
    d = f'/content/kin/datasets/{own}/{slug}'
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True); sh(f'kaggle datasets download {ds} --unzip -p {d}', True)
    print('dataset', ds, os.listdir(d)[:4], f'{time.time()-t0:.0f}s', flush=True)
import re
for i in range(3):
    d = f'/content/kin/rsna-knee-volcache-w{i}/volcache'
    os.makedirs(d, exist_ok=True)
    tok = None; pages = 0
    while True:      # CLI mengunduh 1 halaman (~500 berkas) per panggilan -> ikuti "Next page token"
        r = sh(f"kaggle kernels output dimaspashaakrilian/rsna-knee-volcache-w{i} -p {d} -q" + (f" --page-token '{tok}'" if tok else ''))
        m = re.search(r'Next page token: (\S+)', r.stdout + r.stderr)
        pages += 1
        print(f'  w{i} halaman {pages}: total {len(glob.glob(d + "/**/*.npz", recursive=True))}', flush=True)
        if not m or pages > 30: break
        tok = m.group(1)
    print(f'cache w{i}: {len(glob.glob(d + "/**/*.npz", recursive=True))} npz', f'{time.time()-t0:.0f}s', flush=True)
print(sh('du -sh /content/kin').stdout)
