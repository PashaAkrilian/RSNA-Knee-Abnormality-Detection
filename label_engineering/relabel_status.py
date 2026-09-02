#!/usr/bin/env python3
"""Checkpoint status checker for the relabeling scale-up (20 chunks, ~218 studi tiap chunk).
Cek chunk mana yg sudah SELESAI (ada di label_engineering/relabel_out_backup/, baris lengkap)
vs BELUM -- supaya kalau environment/scratchpad reset lagi, tidak perlu ulang dari nol.
"""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO / "label_engineering" / "relabel_chunks_input"
OUT_DIR = REPO / "label_engineering" / "relabel_out_backup"
OUT_DIR.mkdir(exist_ok=True)

# chunk_19's raw input has a known data-quality issue: 2 of its 209 rows are a
# corrupted CSV split of a single real study's Report text (broken quoting in the
# original dataset, not something we introduced). The true study count is 208,
# and out_19.csv correctly holds all 208 real studies (verified + recovered
# manually, see commit 00966bc). Special-case it here so status stays accurate.
EXPECTED_OVERRIDES = {19: 208}

done, missing = [], []
for i in range(20):
    inp = INPUT_DIR / f"chunk_{i:02d}.csv"
    if not inp.exists():
        continue
    n_expected = EXPECTED_OVERRIDES.get(i, len(pd.read_csv(inp)))
    out = OUT_DIR / f"out_{i:02d}.csv"
    if out.exists():
        try:
            n_got = len(pd.read_csv(out))
        except Exception:
            n_got = -1
        if n_got == n_expected:
            done.append((i, n_got))
            continue
        else:
            print(f"chunk_{i:02d}: PARTIAL/BAD -- expected {n_expected}, got {n_got}")
    missing.append(i)

print(f"\nDONE: {len(done)}/20 chunks, {sum(n for _,n in done)} studi berlabel")
print(f"MISSING: {missing}")
