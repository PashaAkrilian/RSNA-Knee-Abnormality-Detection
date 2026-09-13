#!/usr/bin/env python3
"""Build rsna-knee-submit-v3-hybrid: fork of the current best public pipeline ("Bend the
Knee to the Dinosaurs", our copy `bend-the-knee-to-the-dinosaurs-7d6471`, real LB 0.941)
with ONE new appended cell that blends in OUR OWN v3 CoAtNet arm (fine-tuned on the
raw-probability-mean blend of flight0234 + Claude report relabeling, gold-58 macro=0.9234 --
beats the stock raptor_ft_coatnet_v5_full_swa.pt checkpoint, 0.9206-0.9213, that dominates
every arm in the public pipeline above it, see PROGRESS.md §47 / RESEARCH_NOTES.md §28).

Design goals (this is bolted onto someone else's time-budget-sensitive pipeline, so it must
never be able to make things WORSE than not touching it at all):
  1. Self-gating on wall-clock time (reuses the notebook's own `T0` global) -- only attempts
     the v3 arm if there is still real headroom before the competition's 9h hard cap, and
     the loop itself carries an internal deadline so a slow run degrades gracefully (partial
     coverage, remaining studies just keep neutral rank 0.5) instead of risking a timeout.
  2. Wrapped in one broad try/except -- ANY failure (missing checkpoint, OOM, unexpected
     shape) is caught, logged, and leaves the existing (already proven, 0.941) submission.csv
     completely untouched. Adding this cell can only help or no-op, never break the base run.
  3. Conservative blend weight (alpha=0.15 rank-mean over the WHOLE final output) -- v3 is a
     diversity add-on on top of a proven stack, not a replacement; nested-CV-tuned per-label
     weights are a follow-up once this is validated on the real leaderboard, not before.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-1000/-home-pasha-Documents-RSNA-Knee-Abnormality-Detection/56a96f16-dd4d-450e-802e-b23bc47ce017/scratchpad")
FORK_DIR = SCRATCH / "code_research/our_fork"
OUT = REPO / "kaggle_kernel_submit_v3_hybrid"

base_nb = json.loads((FORK_DIR / "bend-the-knee-to-the-dinosaurs-7d6471.ipynb").read_text())
base_meta = json.loads((FORK_DIR / "kernel-metadata.json").read_text())

ke_src = (REPO / "build_train_coatnet_ke_src.txt").read_text()

# ---- patch 1: point ARMS at our v3 checkpoint instead of stock c5, trim K_EVAL for speed ----
old_arms = "ARMS = [{'file': 'raptor_ft_coatnet_v5_full_swa.pt', 'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384, 'w': 1.0}]"
new_arms = "ARMS = [{'file': 'coatnet_rawmean_ft_v3.pt', 'arch': 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 'res': 384, 'w': 1.0}]"
assert old_arms in ke_src, "ARMS line not found verbatim -- ke_src changed shape, re-check before patching"
ke_src = ke_src.replace(old_arms, new_arms)

old_keval = "K_EVAL = 42"
new_keval = "K_EVAL = 16"  # cheaper per-study eval; acceptable trade for a conservative-weight diversity arm
assert old_keval in ke_src
ke_src = ke_src.replace(old_keval, new_keval)

# ---- patch 2: internal wall-clock deadline inside the per-study loop, so a slow run degrades
# gracefully (remaining studies simply keep their pre-initialized neutral rank 0.5) instead of
# ever risking pushing the WHOLE submission over the competition's time limit. ----
old_loop = """        for i, sid in enumerate(test_ids):
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)"""
new_loop = """        for i, sid in enumerate(test_ids):
            if time.time() > globals().get('_V3_DEADLINE_ABS', float('inf')):
                print(f'[v3-arm] internal deadline reached at study {i}/{N}, stopping early '
                      f'(remaining studies keep neutral rank 0.5)', flush=True)
                break
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)"""
assert old_loop in ke_src, "per-study loop opening not found verbatim -- re-check before patching"
ke_src = ke_src.replace(old_loop, new_loop)

# ---- the new cell appended after the base pipeline's own final cell (per-label CoAtNet blend) ----
new_cell_src = '''\
# ===== v3 arm (ours): CoAtNet fine-tuned on raw-probability-mean blend of flight0234 + Claude
# report relabeling (4,349 studies, gold-58 macro=0.9234) -- beats the stock
# raptor_ft_coatnet_v5_full_swa.pt (0.9206-0.9213) used as the dominant arm above.
# Self-gating: only runs with real time headroom left, any failure/timeout falls back to
# leaving the existing (proven 0.941) submission.csv completely untouched.
import time as _v3_time
try:
    _V3_T0 = globals().get('T0', _v3_time.time())
    _V3_ELAPSED_S = _v3_time.time() - _V3_T0
    _V3_REMAINING_S = 9 * 3600 - _V3_ELAPSED_S
    _V3_ARM_BUDGET_S = min(3600, _V3_REMAINING_S - 1200)  # cap 1h for this arm, keep 20min buffer under the 9h line
    print(f"[v3-arm] elapsed so far {_V3_ELAPSED_S/3600:.2f}h, "
          f"budget for v3 arm: {_V3_ARM_BUDGET_S/60:.0f} min", flush=True)

    if _V3_ARM_BUDGET_S < 600:
        print("[v3-arm] SKIPPED -- not enough time headroom left, "
              "keeping existing submission.csv as-is", flush=True)
    else:
        import pandas as _v3pd, numpy as _v3np, os as _v3os

        _V3_KE_SRC = ''' + repr(ke_src) + '''
        _V3_NS = {'__name__': '_v3ke'}
        exec(compile(_V3_KE_SRC, '<v3arm>', 'exec'), _V3_NS)
        _V3_NS['_V3_DEADLINE_ABS'] = _v3_time.time() + _V3_ARM_BUDGET_S

        if _v3os.path.exists('/kaggle/working/_raptor.csv'):
            _v3os.replace('/kaggle/working/_raptor.csv', '/kaggle/working/_raptor_pre_v3.csv.bak')

        _V3_NS['main']()
        _v3_raw = _v3pd.read_csv('/kaggle/working/_raptor.csv', dtype={'StudyInstanceUID': str})

        _v3_current = _v3pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
        _V3_LAB = [c for c in _v3_current.columns if c != 'StudyInstanceUID']
        _v3_raw_aligned = _v3_raw.set_index('StudyInstanceUID').reindex(
            _v3_current['StudyInstanceUID']).reset_index()
        assert _v3_raw_aligned[_V3_LAB].notna().all().all(), 'v3 arm missing studies after align'

        _V3_ALPHA = 0.15  # conservative diversity weight -- proven stack keeps 85% say
        _cur_rank = _v3_current[_V3_LAB].rank(method='average', pct=True)
        _v3_rank = _v3_raw_aligned[_V3_LAB].rank(method='average', pct=True)
        _v3_out = _v3_current.copy()
        _v3_out[_V3_LAB] = (1.0 - _V3_ALPHA) * _cur_rank + _V3_ALPHA * _v3_rank
        assert _v3np.isfinite(_v3_out[_V3_LAB].to_numpy(_v3np.float64)).all(), 'non-finite after v3 blend'
        _v3_out.to_csv('/kaggle/working/submission.csv', index=False)
        print(f"[v3-arm] blended OUR CoAtNet (raw-mean flight+Claude relabel FT) into "
              f"submission.csv at alpha={_V3_ALPHA}; total elapsed now "
              f"{(_v3_time.time()-_V3_T0)/3600:.2f}h", flush=True)
        print(_v3_out.head().to_string(index=False), flush=True)
except Exception as _v3_exc:
    import traceback as _v3_tb
    print(f"[v3-arm] FAILED, keeping existing submission.csv untouched: "
          f"{type(_v3_exc).__name__}: {_v3_exc}", flush=True)
    _v3_tb.print_exc()
'''

new_cell = {"cell_type": "code", "source": new_cell_src.splitlines(keepends=True),
            "metadata": {}, "outputs": [], "execution_count": None}
base_nb["cells"] = base_nb["cells"] + [new_cell]

OUT.mkdir(exist_ok=True)
nb_filename = "submit_v3_hybrid.ipynb"
(OUT / nb_filename).write_text(json.dumps(base_nb, indent=1))

meta = dict(base_meta)
meta["id"] = "dimaspashaakrilian/rsna-knee-submit-v3-hybrid"
meta["title"] = "RSNA Knee - Submit v3 hybrid"
meta["code_file"] = nb_filename
meta.pop("id_no", None)
meta["kernel_sources"] = list(base_meta.get("kernel_sources", [])) + [
    "dimaspashaakrilian/rsna-knee-train-coatnet-v3-rawmean",
]
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

print("built:", OUT)
print("cells:", len(base_nb["cells"]), "(base had", len(base_nb["cells"]) - 1, ")")
print("kernel_sources:", meta["kernel_sources"])
print("dataset_sources:", meta["dataset_sources"])
