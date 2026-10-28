<div align="center">

# RSNA Knee Abnormality Detection

Our code for the Kaggle RSNA knee MRI competition: weak labels from radiology reports, a three-family ensemble, and a distilled student model.

![Public LB](https://img.shields.io/badge/public%20LB-0.958-blue)
![Private LB](https://img.shields.io/badge/private%20LB-0.944-lightgrey)
![Rank](https://img.shields.io/badge/public%20rank-757%20%2F%206299-orange)
![Metric](https://img.shields.io/badge/metric-macro%20ROC--AUC-success)

![Sample MRI slices](eda_results/eda_outputs/10_sample_slices_grid.png)

</div>

## The task

Predict 12 knee abnormalities per MRI study (ACL, MCL, both menisci, three OA sites, effusion, synovitis, Baker's cyst, contusion, fracture). The metric is macro ROC-AUC, scored 30% public and 70% private. It was a code competition: no internet, 9 hours per run.

## Why it was hard

- About 98-99% of the structured labels are missing. Only 58 studies are fully labeled, and most of the rest have just a report, written in 9 languages.
- Laterality is missing for about half of the series, so a horizontal flip has to swap medial and lateral labels.
- The 58 gold studies are too few to rank models: the standard error is about 0.01, while the leaderboard differences were 0.001.

| Label availability | Class balance | Scanner heterogeneity |
|:---:|:---:|:---:|
| ![](eda_results/eda_outputs/01_label_availability.png) | ![](eda_results/eda_outputs/02_label_class_balance.png) | ![](eda_results/eda_outputs/13_scanner_heterogeneity.png) |

## Approach

```
DICOM -> audit -> 16 slices per series at 0.5 mm/px -> 224x224 cache
Reports -> rule-based weak labeler (negation, uncertainty, 9 languages)

 Raptor family (384 + mirrored 224)   compact R6 + flip      our student (ConvNeXt-T 2.5D)
 public checkpoints, S6F in half      public checkpoints     distilled from OOF teachers
                  \                          |                        /
                   +------ per-finding weights, one blend ---------+
```

A new model goes inside the slot of its own family. Every time we added the student as a separate leg, the public score dropped.

## Results

| | |
|---|---|
| Public LB | 0.958, rank 757 of 6,299 |
| Private LB (both final picks, `g6a` and `g8a`) | 0.944 |

Most of the 0.958 comes from public notebooks, which about 1,300 teams also used. Our additions (S6F placement, per-finding weights, the student) did not move the public score above 0.958. Things that did not help: a student in the Raptor slot, the B-leg, a 24-channel two-plane model, the Part B reader. Details are in the Kaggle writeup.

## Layout

```
build_*.py                 kernel generators (student prep/train/infer, OOF, blends)
make_labels_v1.py          report -> weak labels
local_eval_oof.py          local OOF evaluation
gold_eval_full.py          gold-58 evaluation
kaggle_kernel*/            the pushed Kaggle kernels that matter (EDA, audit, preprocessing,
                           weak labeler, student, final submissions g6a and g8a)
label_engineering/         label versions and blending code
eda_results/  audit_results_exhaustive/  weak_labeler_results/  blend_results/   figures and outputs
docs/WRITEUP_KAGGLE.md     the writeup
```

Training and inference ran on Kaggle. The scripts assume the Kaggle CLI is configured and the competition data is attached to the kernel.

```bash
pip install kaggle
cd kaggle_kernel && kaggle kernels push -p .
```

## Credits

The final recipes build on public work by goodpjw2008, heliosli, nartaa, evgendvorkin and kritikabenjwal. Thanks to them and to the RSNA organizers.

Research and competition code, not a clinically validated system.

Team Krazy KIller. [Competition page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
