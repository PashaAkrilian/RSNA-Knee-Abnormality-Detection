#!/usr/bin/env python3
"""Pewaktu submit + pemantau skor (akun ditentukan oleh KAGGLE_CONFIG_DIR; kredensial tidak pernah dibaca/dicetak).

Untuk tiap kernel (berurutan):
  1. `kaggle kernels push -p <dir>`; bila slot GPU penuh ("Maximum batch GPU session count") coba lagi tiap RETRY_S detik
     (berhenti bila kuota GPU < MIN_QUOTA_H atau melewati batas waktu tunggu);
  2. tunggu status kernel COMPLETE (commit-run pada 3 studi uji publik);
  3. unduh hanya `submission.csv` versi itu dan cek bentuknya (1 baris header + >=1 baris, 13 kolom);
  4. cek jatah submisi harian, lalu `kaggle competitions submit -k <id> -v <versi> -f submission.csv -m <pesan>`.
Setelah semua tersubmit: pantau `kaggle competitions submissions` sampai setiap pesan kita punya publicScore atau ERROR.
Ringkasan akhir ditulis ke <log>.json.

Pakai:
  python -I scripts/submit_watch.py --log <file.log> "<kernel_dir>::<pesan submisi>" ["<kernel_dir>::<pesan>" ...]
  Lanjut tanpa push ulang (kernel sudah ter-push & berjalan): --resume <versi> tepat setelah <file.log>.
"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time

COMP = "rsna-knee-abnormality-detection"
RETRY_S = 120            # jeda coba push ulang saat slot GPU penuh
STATUS_POLL_S = 120      # jeda cek status commit-run
SCORE_POLL_S = 600       # jeda cek skor
MAX_WAIT_SLOT_H = 10.0   # batas menunggu slot GPU per kernel
MAX_COMMIT_H = 5.5       # batas menunggu commit-run selesai (S6-eq penuh ±3 j 45 mnt + reader)
MAX_SCORE_H = 10.0       # batas menunggu skor
MIN_QUOTA_H = 0.6        # jangan push bila sisa kuota GPU di bawah ini

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
args = sys.argv[1:]
assert args[0] == "--log", __doc__
LOG = args[1]
rest = args[2:]
RESUME = None
if rest and rest[0] == "--resume":
    RESUME = int(rest[1])
    rest = rest[2:]
ITEMS = [a.split("::", 1) for a in rest]
assert ITEMS and all(len(x) == 2 for x in ITEMS), __doc__


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kg(cmd, timeout=600):
    for i in range(6):
        try:
            p = subprocess.run(["kaggle"] + cmd, capture_output=True, env=ENV, timeout=timeout)
        except subprocess.TimeoutExpired:
            log("timeout:", " ".join(cmd[:3]))
            time.sleep(30)
            continue
        out = (p.stdout + p.stderr).decode("utf-8", "replace")
        if "429" in out or "Too Many Requests" in out:
            time.sleep(20 * (i + 1))
            continue
        return p.returncode, out
    return 1, "gagal setelah retry"


def gpu_quota_left():
    rc, out = kg(["quota"])
    m = re.search(r"^GPU\s+\S+\s+([\d.]+)h", out, re.M)
    return float(m.group(1)) if m else None


def kernel_id(kdir):
    return json.load(open(os.path.join(kdir, "kernel-metadata.json"), encoding="utf-8"))["id"]


def push(kdir):
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_SLOT_H * 3600:
        q = gpu_quota_left()
        if q is not None and q < MIN_QUOTA_H:
            log("STOP: kuota GPU tinggal", q, "jam")
            return None
        rc, out = kg(["kernels", "push", "-p", kdir])
        m = re.search(r"version (\d+) successfully pushed", out, re.I)
        if m:
            log("PUSH OK", kdir, "versi", m.group(1), "| kuota sisa", q)
            return int(m.group(1))
        if "Maximum batch GPU session count" in out:
            time.sleep(RETRY_S)
            continue
        log("PUSH GAGAL", kdir, out.strip()[-300:])
        return None
    log("STOP: slot GPU tidak tersedia dalam", MAX_WAIT_SLOT_H, "jam")
    return None


def wait_complete(kid):
    t0 = time.time()
    last = ""
    while time.time() - t0 < MAX_COMMIT_H * 3600:
        rc, out = kg(["kernels", "status", kid])
        m = re.search(r"(COMPLETE|ERROR|RUNNING|QUEUED|CANCEL\w*)", out)
        s = m.group(1) if m else out.strip()[-120:]
        if s != last:
            log("status", kid, s)
            last = s
        if s in ("COMPLETE", "ERROR") or s.startswith("CANCEL"):
            return s
        time.sleep(STATUS_POLL_S)
    return "TIMEOUT"


def check_output(kid):
    d = tempfile.mkdtemp(prefix="subcheck_")
    rc, out = kg(["kernels", "output", kid, "-p", d, "--file-pattern", r"^submission\.csv$", "-o", "-q"])
    p = os.path.join(d, "submission.csv")
    if not os.path.exists(p):
        log("submission.csv tidak ada di keluaran", kid, out.strip()[-200:])
        return False
    rows = list(csv.reader(open(p, encoding="utf-8")))
    ok = len(rows) >= 2 and all(len(r) == 13 for r in rows)
    log("cek submission.csv", kid, "baris", len(rows), "kolom", len(rows[0]) if rows else 0, "OK" if ok else "BURUK")
    return ok


def remaining_today():
    rc, out = kg(["competitions", "submission-limits", COMP])
    m = re.search(r"Remaining today:\s*(\d+)", out)
    return int(m.group(1)) if m else None


submitted = []
for kdir, msg in ITEMS:
    kid = kernel_id(kdir)
    ver = RESUME if RESUME is not None else push(kdir)
    if ver is None:
        continue
    st = wait_complete(kid)
    if st != "COMPLETE":
        log("commit-run tidak COMPLETE:", kid, st, "-> tidak disubmit")
        continue
    if not check_output(kid):
        continue
    rem = remaining_today()
    log("jatah submisi hari ini tersisa:", rem)
    if rem == 0:
        log("jatah habis -> tidak disubmit:", kid)
        continue
    rc, out = kg(["competitions", "submit", COMP, "-k", kid, "-v", str(ver), "-f", "submission.csv", "-m", msg])
    log("SUBMIT", kid, "v%d" % ver, "rc", rc, "|", out.strip()[-200:])
    if rc == 0:
        submitted.append(msg)

summary = {}
t0 = time.time()
while submitted and time.time() - t0 < MAX_SCORE_H * 3600:
    rc, out = kg(["competitions", "submissions", COMP, "--format", "json"])
    i = out.find("[")
    try:
        subs = json.loads(out[i:]) if i >= 0 else []
    except json.JSONDecodeError:
        subs = []
    for msg in submitted:
        for s in subs:
            if s.get("description") == msg:
                summary[msg] = {"ref": s.get("ref"), "status": s.get("status"), "publicScore": s.get("publicScore"), "date": s.get("date")}
                break
    log("skor:", json.dumps(summary, ensure_ascii=False))
    done = [m for m in submitted if m in summary and (summary[m]["publicScore"] or "ERROR" in str(summary[m]["status"]))]
    if len(done) == len(submitted):
        break
    time.sleep(SCORE_POLL_S)

json.dump(summary, open(LOG + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("SELESAI", json.dumps(summary, ensure_ascii=False))
