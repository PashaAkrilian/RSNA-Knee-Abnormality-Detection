#!/usr/bin/env python3
"""Severity-aware soft labels utk RSNA Knee.

Lapisan-1: lexicon multilingual daur ulang dari notebook publik (extract()).
Lapisan-2: koreksi severity sesuai HEURISTIK RESMI Overview kompetisi --
borderline ("small/mild", sprain ringan, degenerasi intrasubstance) di-graded
NEGATIVE oleh annotator -> cap monotonik supaya tidak over-call di middle-rank
(tempat macro AUC diputuskan).

Output: label_engineering/severity_labels.csv + evaluasi vs gold-58.
Run: python3 severity_labels.py
"""
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/opencode/labels")
from lexicon import extract, normalize, unwrap, TARGETS  # noqa: E402

REPO = "/home/pasha-akrilian/Documents/RSNA Knee Abnormality Detection"

SEV_SMALL = r"\b(small|tiny|trace|mild|minimal|slight|minor|kecil|ringan|leicht|ligera|leve|piccolo|klein)\b"
SEV_LARGE = r"\b(moderate|moderately|large|severe|marked|significant|pronounced|besar|sedang|berat|schwer|grade\s+(ii|iii|iv|2|3|4))\b"
DEGEN_MENISCUS = r"\b(degenerat\w*|intrasubstance|intrameniscal|mucoid|mucinous)\b"


def sev_of(text):
    t = text.lower()
    return bool(re.search(SEV_SMALL, t)), bool(re.search(SEV_LARGE, t))


def context_window(t, kw, width=70):
    out = []
    for m in re.finditer(kw, t):
        lo, hi = max(0, m.start() - width), min(len(t), m.end() + width)
        out.append(t[lo:hi])
    return out


def severity_adjust(base, report):
    """Cap monotonik berbasis aturan resmi. base: {target: p} dr lexicon."""
    t = normalize(unwrap(report))
    adj = dict(base)

    # Effusion: cuma moderate/large yang positif resmi
    ctxs = context_window(t, r"\beffusion\b|\bjoint\s+fluid\b")
    if ctxs:
        small_any = all(s for s, _ in (sev_of(c) for c in ctxs))
        any_large = any(l for _, l in (sev_of(c) for c in ctxs))
        if small_any and not any_large:
            adj["Effusion"] = min(adj["Effusion"], 0.30)

    # OA: kualifier mild tanpa moderate/severe di konteks compartment -> cap
    for tgt, site in [("Medial OA", r"\bmedial\b"), ("Lateral OA", r"\blateral\b"),
                      ("PF OA", r"\bpatellofemoral\b|\bpatell\w*\b")]:
        if adj.get(tgt, 0) < 0.5:
            continue
        ctxs = context_window(t, site)
        if not ctxs:
            continue
        mild_only = all(s and not l for s, l in (sev_of(c) for c in ctxs))
        if mild_only:
            adj[tgt] = min(adj[tgt], 0.40)

    # Meniscus: degenerasi/intrasubstance tanpa kata surface-contact -> cap
    for tgt, side in [("Medial Meniscus", "medial"), ("Lateral Meniscus", "lateral")]:
        if adj.get(tgt, 0) < 0.5:
            continue
        ctxs = context_window(t, rf"\b{side}\b[^.]{{0,80}}menisc|\bmenisc[^.]{{0,80}}\b{side}\b")
        degen_only = bool(ctxs) and all(
            re.search(DEGEN_MENISCUS, c) and not re.search(r"\btear", c) for c in ctxs)
        if degen_only:
            adj[tgt] = min(adj[tgt], 0.45)

    # MCL: sprain ringan saja -> cap
    if adj.get("MCL", 0) >= 0.5:
        m = re.search(r"\bmcl\b|\bmedial collateral\b", t)
        if m:
            ctx = t[max(0, m.start() - 60): m.end() + 90]
            if re.search(r"\bsprain", ctx) and not re.search(
                    r"high.?grade|complete|full.?thickness|rupture|tear", ctx):
                adj["MCL"] = min(adj["MCL"], 0.40)
    return adj


def macro_auc(y, s):
    aucs = []
    for j in range(y.shape[1]):
        pos, neg = s[y[:, j] == 1, j], s[y[:, j] == 0, j]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = pd.Series(np.concatenate([pos, neg])).rank().values
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs))


def main():
    tr = pd.read_csv(f"{REPO}/reference_metadata/train.csv")
    reports = tr["Report"].fillna("").astype(str).values
    ext = [extract(r) for r in reports]  # SEKALI per report
    rows = []
    for d, rep in zip(ext, reports):
        base = {t: d.get(t, 0.28) for t in TARGETS}
        rows.append(severity_adjust(base, rep))
    lab = pd.DataFrame(rows)
    lab.insert(0, "StudyInstanceUID", tr["StudyInstanceUID"].astype(str).values)

    gold_mask = tr[TARGETS].notna().all(axis=1).values
    y = tr.loc[gold_mask, TARGETS].values.astype(float)

    raw = np.array([[d.get(t, 0.28) for t in TARGETS] for d in ext])
    sev = lab.loc[gold_mask, TARGETS].values.astype(float)

    fidx = pd.read_csv("/tmp/opencode/labels/report_labels_v4hybrid.csv"
                       ).set_index("StudyInstanceUID")
    fidx.index = fidx.index.astype(str)
    ids = tr.loc[gold_mask, "StudyInstanceUID"].astype(str).values
    flight = fidx.reindex(ids)[TARGETS].fillna(0.28).values

    print(f"gold: {gold_mask.sum()} studi")
    print(f"lexicon mentah      : {macro_auc(y, raw[gold_mask]):.4f}")
    print(f"severity-adjusted   : {macro_auc(y, sev):.4f}")
    print(f"flight0234 (gate)   : {macro_auc(y, flight):.4f}")

    for a in (0.3, 0.5, 0.7):
        blend = a * sev + (1 - a) * flight
        print(f"blend sev+flight a={a}: {macro_auc(y, blend):.4f}")

    out = f"{REPO}/label_engineering/severity_labels.csv"
    lab.to_csv(out, index=False)
    conf = pd.DataFrame([[d.get(f"{t}__conf", 0.05) for t in TARGETS] for d in ext])
    conf.insert(0, "StudyInstanceUID", tr["StudyInstanceUID"].astype(str).values)
    conf.to_csv(out.replace(".csv", "_conf.csv"), index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
