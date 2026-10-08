#!/usr/bin/env python3
# Dijalankan DI VM Colab (latar): cadangkan status lanjut + hasil anak ke dataset Kaggle PRIVAT akun utama tiap SYNC_EVERY detik
# (default 2400 = 40 mnt) HANYA bila ada berkas berubah. Tujuan: bila VM dilepas server (~5 jam) status latih tidak hilang.
# Pulihkan di VM baru:  kaggle datasets download dimaspashaakrilian/rsna-knee-oai-child-colab-ckpt --unzip -p /content/kw  lalu jalankan ulang skrip latih (otomatis RESUME).
import os, time, glob, json, subprocess, shutil
OUT = '/content/kw'; STAGE = '/content/kw_sync'; DS = 'dimaspashaakrilian/rsna-knee-oai-child-colab-ckpt'
EVERY = int(os.environ.get('SYNC_EVERY', '2400'))
os.makedirs(STAGE, exist_ok=True)
json.dump({'title': 'rsna knee oai child colab ckpt', 'id': DS, 'licenses': [{'name': 'CC0-1.0'}]}, open(STAGE + '/dataset-metadata.json', 'w'))
def run(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); run.rc = r.returncode; return r.stdout + r.stderr
run.rc = 0
def files():
    return sorted({f for p in ('resume_*.pt', 'oai_child_*_swa.pt', 'oai_child_*_final.pt', 'done_*.flag', 'parent_soft_k16.csv') for f in glob.glob(f'{OUT}/{p}')})
def sig(): return {f: (os.path.getsize(f), int(os.path.getmtime(f))) for f in files()}
last, created = None, False
if 'ready' in run(f'kaggle datasets status {DS}').lower():       # dataset sudah ada -> pakai 'version' (bukan 'create')
    created = True
    if os.environ.get('ASSUME_SYNCED') == '1': last = sig()       # jangan unggah ulang berkas yang sama saat start ulang
    print(time.strftime('%T'), 'dataset sudah ada; created=True', flush=True)
while True:
    s = sig()
    if s and s != last and not glob.glob(OUT + '/*.tmp'):
        for f in os.listdir(STAGE):
            if f != 'dataset-metadata.json': os.remove(f'{STAGE}/{f}')
        for f in s:
            dst = f'{STAGE}/{os.path.basename(f)}'
            try: os.link(f, dst)
            except OSError: shutil.copy(f, dst)
        ok = False
        for attempt in range(6):
            txt = '' if not created else run(f'kaggle datasets version -p {STAGE} -m "sync {time.strftime("%H:%M")}"')
            if (not created) or 'not found' in txt.lower() or '404' in txt:
                txt = run(f'kaggle datasets create -p {STAGE}')
                if 'already in use' in txt.lower() or 'already exists' in txt.lower():
                    txt = run(f'kaggle datasets version -p {STAGE} -m "sync {time.strftime("%H:%M")}"')
            if '429' in txt or 'Too Many Requests' in txt:
                time.sleep(90); continue
            ok = (run.rc == 0) or ('being created' in txt.lower()) or ('successfully' in txt.lower())
            created = created or ok
            break
        print(time.strftime('%T'), 'SYNC', 'ok' if ok else 'GAGAL', '|', txt.replace('\r', '\n').strip().split('\n')[-1][-140:], flush=True)
        if ok: last = s
    time.sleep(EVERY)
