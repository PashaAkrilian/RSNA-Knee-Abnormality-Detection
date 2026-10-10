#!/usr/bin/env python3
"""Uji lokal supervisor OF v4 (of_stage_v4.OF4_CELL) dgn run_inference palsu (tanpa torch/GPU).
Skenario: A normal+injeksi-kill+daur-ulang, B potongan racun (2x gagal -> bad), C semua pekerja crash saat impor, D batas waktu lunak.
Pemakaian: python3 test_of_v4_local.py [A|B|C|D|all]"""
import importlib.util
import os
import shutil
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ofv4", ROOT / "of_stage_v4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

STUB = textwrap.dedent('''
    import os, time, random
    import pandas as pd
    {pre}
    LAB = %r
    def run_inference(root, ckpt, out, anatomy_weights=None, workers=1):
        ids = pd.read_csv(root + "/test.csv").StudyInstanceUID.tolist()
        for s in ids:
            assert os.path.isdir(root + "/test_series/" + s), "symlink ausente"
        if any("POISON" in s for s in ids):
            raise RuntimeError("poison study")
        time.sleep({sleep} * len(ids))
        os.makedirs(out, exist_ok=True)
        return pd.DataFrame([[s] + [random.random() for _ in LAB] for s in ids], columns=["StudyInstanceUID"] + LAB)
''' % (LAB,))


def run(name, n, chunk, nproc, stop_h, kill_h, maxchunks, inject, sleep, pre="", poison=0, expect_cov=None, expect_ok=True, maxsec=120, force=0):
    wd = tempfile.mkdtemp(prefix="of4test_")
    (Path(wd) / "cloud_code" / "training").mkdir(parents=True)
    (Path(wd) / "cloud_code" / "training" / "__init__.py").write_text("")
    (Path(wd) / "cloud_code" / "training" / "infer.py").write_text(STUB.format(pre=pre, sleep=sleep))
    root = Path(wd) / "data"
    (root / "test_series").mkdir(parents=True)
    ids = [f"1.2.3.{i:04d}" + ("POISON" if i < poison else "") for i in range(n)]
    import pandas as pd
    pd.DataFrame({"StudyInstanceUID": ids, "other": None}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [i for i in ids for _ in range(2)], "SeriesInstanceUID": [f"s{j}" for _ in ids for j in range(2)]}).to_csv(root / "test_series.csv", index=False)
    for i in ids:
        (root / "test_series" / i).mkdir()
    cell = (m.OF4_CELL.replace("__WD__", wd).replace("__NPROC__", str(nproc)).replace("__CHUNK__", str(chunk))
            .replace("__STOP_H__", str(stop_h)).replace("__KILL_H__", str(kill_h)).replace("__MINCOV__", "0.15")
            .replace("__MAXCHUNKS__", str(maxchunks)).replace("__RSSMAX__", "9.0").replace("__INJECT__", str(inject))
            .replace("__INJECT_AFTER__", "3").replace("__POLL__", "0.5").replace("__PRINTEVERY__", "5").replace("__MEMPERW__", "0.5").replace("__MEMFREE__", "0.0").replace("__DISKPER__", "0.0").replace("__DISKBASE__", "0.0").replace("__FORCEW__", str(force)).replace("__MMAP__", "0").replace("__EXTRAENV__", "{}")
            .replace("__SETUP__", "0").replace("__DATA_PRESET__", f'_of_data = "{root}"'))
    ns = {"__name__": "__of4test__"}
    t0 = time.time()
    exec(compile(cell, f"of4-{name}", "exec"), ns)
    dt = time.time() - t0
    st = ns["_OF_STATE"]
    leg = Path(wd) / "_leg_of.csv"
    cov = None
    if leg.exists():
        d = pd.read_csv(leg, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(ids)
        cov = 1.0 - float(d.isna().all(axis=1).mean())
    print(f"\n=== [{name}] state={st} cov={cov} dt={dt:.1f}s spawned={ns.get('_spawned')} restarts_left={ns.get('_restarts')}")
    good = (st["ok"] == expect_ok) and (expect_cov is None or (cov is not None and abs(cov - expect_cov) < 0.011)) and dt < maxsec
    # nenhum processo de pekerja remanescente
    left = 0
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                if wd + "/_of_worker.py" in open(f"/proc/{pid}/cmdline").read().replace("\0", " "):
                    left += 1
            except OSError:
                pass
    print(f"=== [{name}] {'PASS' if good else 'FAIL'} | leftover procs matching wd: {left}")
    shutil.rmtree(wd, ignore_errors=True)
    return good


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("A", "all"):
        res["A"] = run("A normal+inject+recycle", n=60, chunk=4, nproc=3, stop_h=0.05, kill_h=0.08, maxchunks=3, inject=1, sleep=0.4, expect_cov=1.0, force=3)
    if which in ("B", "all"):
        res["B"] = run("B poison", n=40, chunk=4, nproc=3, stop_h=0.05, kill_h=0.08, maxchunks=3, inject=0, sleep=0.2, poison=4, expect_cov=0.9, force=3)
    if which in ("C", "all"):
        res["C"] = run("C crash-on-import", n=16, chunk=4, nproc=2, stop_h=0.05, kill_h=0.08, maxchunks=3, inject=0, sleep=0.2, pre="raise ImportError('boom')", expect_ok=False, force=2)
    if which in ("D", "all"):
        res["D"] = run("D soft-stop", n=200, chunk=4, nproc=3, stop_h=0.004, kill_h=0.02, maxchunks=50, inject=0, sleep=0.5, expect_ok=True, maxsec=90, force=3)
    print("\nRESUMO:", res)
    sys.exit(0 if all(res.values()) else 1)
