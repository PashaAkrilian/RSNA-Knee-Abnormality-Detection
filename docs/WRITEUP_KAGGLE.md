# Weak report labels + a 3-family ensemble: what moved the score, and what didn't (public 0.958, private 0.944)

Team **Krazy KIller**. Final: public LB 0.958 (rank 757/6,299), private 0.944. Code: https://github.com/PashaAkrilian/RSNA-Knee-Abnormality-Detection.

This is not a gold solution. Most of our 0.958 came from public notebooks, and I credit them below. I'm posting because the negative results may save you time, and because a few findings about the data were useful.

## 1. The data problem

![Label availability](docs/writeup_images/01_label_availability.png)

About 98–99% of structured label cells in `train.csv` are missing. Only 58 "gold" studies have full labels, so gold-58 has SE ≈ 0.01 and cannot separate submissions that the public LB separates by 0.001.

![Class balance](docs/writeup_images/02_label_class_balance.png)
![Label correlation](docs/writeup_images/03_label_correlation.png)

Series come from very different scanners, which is why preprocessing was physical (0.5 mm/px) and sequence-aware:

![Scanner heterogeneity](docs/writeup_images/13_scanner_heterogeneity.png)

![Sample slices](docs/writeup_images/10_sample_slices_grid.png)

Laterality is missing for ~50% of series, so a horizontal flip is **not** label-preserving: medial and lateral labels must swap.

## 2. DICOM audit

819,078 instances audited; 94.1% clean. Anomalies (duplicate slices, orientation, constant images, a few corrupt files) were concentrated outside the gold studies.

![Audit](docs/writeup_images/audit_error_distribution_FULL.png)

## 3. Weak labels from reports

A rule-based labeler (negation, uncertainty, clause-level co-occurrence of finding and location, 9 languages) gave 4,407 labeled studies. Against gold-58: agreement 0.747, precision 0.686, recall 0.951. It over-calls, which is fine for a teacher that gets averaged with model OOF predictions.

![Per-label agreement](docs/writeup_images/validation_agreement_per_label.png)

Later label versions mixed in OOF predictions of strong models and soft labels from an offline LLM.

## 4. Final recipe

- **Raptor family**: public 384 px + mirrored 224 px checkpoints. S6F (nartaa) placed in **half** of this family.
- **Compact R6 + flip** checkpoints (goodpjw2008's notebook).
- **Our student**: ConvNeXt-T 2.5D, distilled from OOF teacher consensus.
- Per-finding weights from heliosli's public notebook.

Rule that held every time: a new model goes **inside the slot of its own family**. Adding our student in the Raptor slot or as a third leg always lowered the public score.

## 5. What did not work

| Idea | Result |
|---|---|
| Student in Raptor slot | public 0.957 (down) |
| B-leg (`g13`) | gold 0.9415, OOF +0.0033, public 0.957 |
| 24-channel two-plane model | gold 0.8346, hurt at every blend weight |
| Part B reader leg | neutral after rounding |
| Student trained on label v3 | no gain vs v2 (gold 0.9292 vs 0.9334) |

Takeaway: gold-58 and local OOF agreed with each other but not with the public LB for small effects. Below ~0.002, I'd treat every comparison as noise.

## 6. Credits

`goodpjw2008`, `heliosli`, `nartaa`, `evgendvorkin`, `kritikabenjwal`, and the RSNA organizers.

## 7. Limits

Public 0.958 was a plateau shared by ~1,300 teams; our unique additions didn't break it, and the private score fell to 0.944. Not a clinical system.
