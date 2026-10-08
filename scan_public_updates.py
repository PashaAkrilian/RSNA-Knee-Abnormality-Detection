#!/usr/bin/env python3
"""Pindai pembaruan publik RSNA Knee yang mungkin membawa checkpoint/notebook lebih kuat (strategi 'fast follower').
Mencetak HANYA yang baru sejak pemanggilan sebelumnya (state: ~/.cache/rsna_scan_state.json).
Pakai: python3 scan_public_updates.py        # lalu cek judul bertanda 0.95x / 'checkpoint' / 'weights'
"""
import json
import subprocess
from pathlib import Path

STATE = Path.home() / ".cache" / "rsna_scan_state.json"
STATE.parent.mkdir(parents=True, exist_ok=True)
state = json.loads(STATE.read_text()) if STATE.exists() else {"datasets": [], "kernels": []}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout


def rows(txt):
    out = []
    for ln in txt.splitlines()[2:]:
        if ln.strip():
            out.append(ln.rstrip())
    return out


new = {"datasets": [], "kernels": []}
for q in ("rsna knee", "knee mri weights", "raptor knee", "knee swa", "oai knee"):
    for ln in rows(run(["kaggle", "datasets", "list", "-s", q, "--sort-by", "updated", "-p", "1"])):
        ref = ln.split()[0]
        if ref not in state["datasets"] and ref not in new["datasets"]:
            new["datasets"].append(ref); print("[DATASET]", ln[:170])
for key in ("dateRun", "scoreDescending"):
    for ln in rows(run(["kaggle", "kernels", "list", "--competition", "rsna-knee-abnormality-detection", "--sort-by", key, "--page-size", "20"])):
        ref = ln.split()[0]
        if ref not in state["kernels"] and ref not in new["kernels"]:
            new["kernels"].append(ref); print("[KERNEL ]", ln[:170])
state["datasets"] = sorted(set(state["datasets"]) | set(new["datasets"]))
state["kernels"] = sorted(set(state["kernels"]) | set(new["kernels"]))
STATE.write_text(json.dumps(state))
print(f"baru: {len(new['datasets'])} dataset, {len(new['kernels'])} notebook")
