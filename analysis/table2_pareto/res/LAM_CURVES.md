# F1–latency curves: SeMARC over λ at uniform cost, and the Table 2 controllers (Table 2 metric)

Request: the figure of `CTRL_CURVES.md`, with SeMARC's curve over λ at uniform cost (γ = 0) instead of over γ at Table 2's λ (`lam_curves.py`, `fig/pareto_lambda_controllers_mean.{pdf,png}`).

## SeMARC's curve

- **Knob**: λ ∈ {.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.25, 1.5, 2}, at uniform cost (γ = 0, `RLB_COST_MODE=uniform`). One λ is shared by all six datasets and all folds, as each controller's knob.
- **Everything else** is the deployed setting per fold (`sweep.py`):
  - the per-fold ARC seed, FQI schedule and availability augmentation;
  - the paper's decision rule: the prior priced at the training λ, stop margin 0.
- **Nets**:
  - λ = .05, .1, .2, .5, 1, 2 are the γ × λ sweep's uniform-cost runs (`res/sweep_*_g0.csv`, `res/verify_*_g0.csv`).
  - λ = .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.25, 1.5, 2 were fitted for this figure: 219 nets (6 datasets × 3 folds × 14 λ), with the unchanged `train_or_load` on GPU 2/3 (`lamu_launch.sh` → `res/lamu_*_g0.csv`, nets in `RL_AAAI_branch/checkpoints_t2p/<ds>/`).
- **Star** = Table 2's SeMARC: the deployed λ = .5 on every fold except MM-Fi fold 2 (λ = 1). It is not a λ shared by all folds, so it is not on the curve; λ = .5 shared differs from it only on MM-Fi fold 2.
- **Diamond** = the paper's weighted row (γ = .75 at Table 2's λ), as in the γ figure.
- **Controllers**: curves, Table 2 markers and latency estimate unchanged from `CTRL_CURVES.md`.

## Checks

- The star and the diamond recompute to the published rows of `tab:comp-replace-complete`:

| Row | dataset | F1 | paper F1 | latency (ms) | paper latency (ms) |
|---|---|---|---|---|---|
| SeMARC (Table 2, uniform cost) | IEMOCAP | 0.6689 | 0.669 | 744.71 | 745.00 |
| SeMARC (Table 2, uniform cost) | MM-Fi | 0.9503 | 0.950 | 186.85 | 187.00 |
| SeMARC (Table 2, uniform cost) | CZU-MHAD | 0.9137 | 0.914 | 1300.80 | 1301.00 |
| SeMARC (Table 2, uniform cost) | EAV | 0.6891 | 0.689 | 216.77 | 217.00 |
| SeMARC (Table 2, uniform cost) | CMI | 0.5558 | 0.556 | 12.62 | 12.60 |
| SeMARC (Table 2, uniform cost) | UTD-MHAD | 0.8269 | 0.827 | 659.30 | 659.00 |
| SeMARC (Table 2, uniform cost) | Mean of six datasets | 0.7675 | 0.768 | 520.17 | 520.27 |
| SeMARC (paper, weighted, γ = .75) | IEMOCAP | 0.6704 | 0.670 | 87.32 | 87.30 |
| SeMARC (paper, weighted, γ = .75) | MM-Fi | 0.9519 | 0.952 | 68.68 | 68.70 |
| SeMARC (paper, weighted, γ = .75) | CZU-MHAD | 0.9158 | 0.916 | 1313.79 | 1314.00 |
| SeMARC (paper, weighted, γ = .75) | EAV | 0.6939 | 0.694 | 114.03 | 114.00 |
| SeMARC (paper, weighted, γ = .75) | CMI | 0.5515 | 0.552 | 12.18 | 12.20 |
| SeMARC (paper, weighted, γ = .75) | UTD-MHAD | 0.7877 | 0.788 | 536.82 | 537.00 |
| SeMARC (paper, weighted, γ = .75) | Mean of six datasets | 0.7619 | 0.762 | 355.47 | 355.53 |

## Mean of six datasets

**SeMARC, uniform cost** (λ)

| λ | F1 | MU | latency (ms) | on the front |
|---|---|---|---|---|
| 2 | 0.697 | 0.23 | 311 | yes |
| 1.25 | 0.720 | 0.31 | 371 | yes |
| 1.5 | 0.709 | 0.30 | 415 |  |
| 1 | 0.724 | 0.35 | 423 | yes |
| .9 | 0.750 | 0.38 | 442 | yes |
| .7 | 0.758 | 0.44 | 514 | yes |
| .5 | 0.768 | 0.48 | 550 | yes |
| .6 | 0.763 | 0.45 | 565 |  |
| .8 | 0.747 | 0.39 | 663 |  |
| .4 | 0.751 | 0.51 | 813 |  |
| .3 | 0.754 | 0.54 | 823 |  |
| .2 | 0.758 | 0.62 | 976 |  |
| .1 | 0.764 | 0.73 | 1253 |  |
| .05 | 0.765 | 0.78 | 1284 |  |
| Table 2 (star) | 0.767 | 0.48 | 520 |  |
| weighted row (diamond) | 0.762 | 0.63 | 355 |  |

The controllers' mean points are in `CTRL_CURVES.md`.

## Why the Table 2 markers are not on the curves

A curve point runs every dataset, fold and seed at one knob value. Table 2's configurations do not:
- **EAMA** picks λ separately in each of its 54 runs (6 datasets × 3 folds × 3 value-model seeds) on the validation holdout, by val F1 − 0.2 × val extraction fraction (`baseline_eama_raw.py`). The picks range from .005 to 2.
- **GMS** uses one k per dataset, the same on every fold, selected on validation according to the paper: IEMOCAP 5, MM-Fi 1, CZU-MHAD 4, EAV 1, CMI 5, UTD-MHAD 2 (`build_acq_profiles.py`).
- **Joint-AFA**'s r_cost is fixed at .05 on every dataset and fold, not tuned. Its marker is the paper's GPU run; its curve is the CPU retrain.
- **SeMARC**'s λ is not tuned either: .5 on every fold except MM-Fi fold 2 (1.0). Only the per-fold seed is selected on validation (Appendix `tab:policy-config`).

An average of runs at different knob values is not a point of the one-value curve, so it can land on either side:
- **Uniform EAMA lands below its curve.** It also sits at or below each dataset's own curve, by up to .012 F1 at the same latency (UTD-MHAD). Its validation picks do not carry over to test.
- **Weighted EAMA lands above its curve.** Per dataset it sits on its own curves (IEMOCAP +.008, CMI −.004). The gain at the mean comes from tuning λ per run: EAMA prices a fraction of each dataset's own total extraction time, from 8 ms (CMI) to 4.2 s (MM-Fi), so one shared λ prices a millisecond very differently on each dataset.
- The latency formula is not the cause: at the same λ the marker's price (stored frequencies) and the curve's (mean set size) differ by at most .004 ms.

| Configuration | Table 2 F1 @ ms | its curve at that latency (linear between front points) | difference |
|---|---|---|---|
| SeMARC | 0.767 @ 520 | 0.759 | +0.008 |
| SeMA + EAMA | 0.744 @ 464 | 0.753 | -0.009 |
| SeMA + EAMA (weighted) | 0.761 @ 425 | 0.754 | +0.006 |
| SeMA + Joint-AFA | 0.749 @ 920 | 0.749 | -0.000 |
| SeMA + GMS | 0.733 @ 772 | 0.657 | +0.076 |

## Who beats whom on the mean

Beaten = some point has F1 ≥ and latency ≤, one strictly. 'SeMARC λ' = the blue curve.

| Controller | point | F1 | latency (ms) | beaten by SeMARC λ | SeMARC λ best F1 at ≤ its latency |
|---|---|---|---|---|---|
| SeMA + EAMA | λ = 2 | 0.496 | 307 | no | – |
| SeMA + EAMA | λ = 1 | 0.692 | 359 | λ = 2 | 0.697 |
| SeMA + EAMA | λ = .5 | 0.738 | 400 | no | 0.720 |
| SeMA + EAMA | λ = .2 | 0.757 | 482 | no | 0.750 |
| SeMA + EAMA | λ = .1 | 0.766 | 552 | λ = .5 | 0.768 |
| SeMA + EAMA | λ = .05 | 0.771 | 644 | no | 0.768 |
| SeMA + EAMA | λ = .02 | 0.772 | 740 | no | 0.768 |
| SeMA + EAMA | λ = .01 | 0.773 | 781 | no | 0.768 |
| SeMA + EAMA | λ = .005 | 0.773 | 830 | no | 0.768 |
| SeMA + EAMA | Table 2 | 0.744 | 464 | λ = .9 | 0.750 |
| SeMA + EAMA (weighted) | λ = 2 | 0.534 | 20.6 | no | – |
| SeMA + EAMA (weighted) | λ = 1 | 0.574 | 60.8 | no | – |
| SeMA + EAMA (weighted) | λ = .5 | 0.750 | 363 | no | 0.697 |
| SeMA + EAMA (weighted) | λ = .2 | 0.760 | 509 | no | 0.750 |
| SeMA + EAMA (weighted) | λ = .1 | 0.767 | 595 | λ = .5 | 0.768 |
| SeMA + EAMA (weighted) | λ = .05 | 0.774 | 673 | no | 0.768 |
| SeMA + EAMA (weighted) | Table 2 | 0.761 | 425 | no | 0.724 |
| SeMA + Joint-AFA | r_cost = .5 | 0.672 | 384 | λ = 2, λ = 1.25 | 0.720 |
| SeMA + Joint-AFA | r_cost = .3 | 0.724 | 432 | no | 0.724 |
| SeMA + Joint-AFA | r_cost = .1 | 0.743 | 517 | λ = .9, λ = .7 | 0.758 |
| SeMA + Joint-AFA | r_cost = 0 | 0.758 | 1459 | λ = .5 | 0.768 |
| SeMA + Joint-AFA | Table 2 | 0.749 | 920 | λ = .9, λ = .7, λ = .5 | 0.768 |
| SeMA + GMS | k = 1 | 0.649 | 753 | λ = 2, λ = 1.25, λ = 1, λ = .9, λ = .7, λ = .5 | 0.768 |
| SeMA + GMS | k = 2 | 0.754 | 993 | λ = .7, λ = .5 | 0.768 |
| SeMA + GMS | k = 3 | 0.767 | 1418 | λ = .5 | 0.768 |
| SeMA + GMS | k = 4 | 0.773 | 1438 | no | 0.768 |
| SeMA + GMS | k = 5 | 0.775 | 1454 | no | 0.768 |
| SeMA + GMS | Table 2 | 0.733 | 772 | λ = .9, λ = .7, λ = .5 | 0.768 |

| SeMARC λ (front) | F1 | latency (ms) | beaten by a controller point |
|---|---|---|---|
| 2 | 0.697 | 311 | none |
| 1.25 | 0.720 | 371 | EAMA (weighted) λ = .5 |
| 1 | 0.724 | 423 | EAMA λ = .5, EAMA (weighted) λ = .5 |
| .9 | 0.750 | 442 | EAMA (weighted) λ = .5 |
| .7 | 0.758 | 514 | EAMA (weighted) λ = .2 |
| .5 | 0.768 | 550 | none |

## Per dataset: SeMARC's λ front

| Dataset | SeMARC λ front: λ → F1 @ ms | Table 2 (star) | controller front points not beaten by SeMARC λ | SeMARC λ front points beaten by a controller |
|---|---|---|---|---|
| IEMOCAP | 2 → 0.598 @ 5.5; 1 → 0.624 @ 238; .9 → 0.631 @ 259; .8 → 0.652 @ 292; .7 → 0.663 @ 420; .6 → 0.667 @ 543; .5 → 0.669 @ 745; .1 → 0.675 @ 994 | 0.669 @ 745 | 18 of 26 | 8 of 8: λ = 2 by GMS k = 1; λ = 1 by EAMA λ = .5, EAMA (weighted) λ = .2 (+7 more); λ = .9 by EAMA (weighted) λ = .1, EAMA (weighted) λ = .2 (+7 more); λ = .8 by EAMA (weighted) λ = .1, EAMA (weighted) λ = .2 (+5 more); λ = .7 by EAMA (weighted) λ = .01, EAMA (weighted) λ = .02 (+4 more); λ = .6 by EAMA (weighted) λ = .01, EAMA (weighted) λ = .02 (+4 more); λ = .5 by EAMA (weighted) λ = .005, EAMA (weighted) λ = .05 (+2 more); λ = .1 by GMS k = 4, GMS k = 5 |
| MM-Fi | 2 → 0.941 @ 22.4; 1 → 0.944 @ 87.6; .9 → 0.946 @ 133; .7 → 0.955 @ 318; .5 → 0.956 @ 366; .6 → 0.960 @ 386 | 0.950 @ 187 | 8 of 24 | 1 of 6: λ = 2 by EAMA λ = 1, EAMA λ = 2 |
| CZU-MHAD | 2 → 0.889 @ 1294; 1.5 → 0.895 @ 1295; .8 → 0.900 @ 1295; 1.25 → 0.901 @ 1296; .5 → 0.914 @ 1301; .05 → 0.917 @ 1314 | 0.914 @ 1301 | 16 of 21 | 6 of 6: λ = 2 by EAMA λ = 1, Joint-AFA r_cost = .3 (+2 more); λ = 1.5 by EAMA λ = .2; λ = .8 by EAMA λ = .2, GMS k = 2; λ = 1.25 by EAMA λ = .2, GMS k = 2; λ = .5 by EAMA λ = .05, GMS k = 2; λ = .05 by EAMA λ = .005, EAMA λ = .01 (+12 more) |
| EAV | 2 → 0.688 @ 27.1; 1.5 → 0.688 @ 29.9; .5 → 0.689 @ 217; .4 → 0.690 @ 256; .2 → 0.692 @ 441; .05 → 0.692 @ 644 | 0.689 @ 217 | 6 of 12 | 4 of 6: λ = .5 by EAMA (weighted) λ = .5, EAMA (weighted) λ = 1 (+2 more); λ = .4 by EAMA (weighted) λ = .5, EAMA (weighted) λ = 1 (+1 more); λ = .2 by GMS k = 2; λ = .05 by GMS k = 2 |
| CMI | 2 → 0.485 @ 4.6; 1.5 → 0.507 @ 6.6; 1 → 0.540 @ 9.6; .6 → 0.542 @ 9.8; 1.25 → 0.547 @ 10.0; .4 → 0.557 @ 11.6; .2 → 0.565 @ 13.8; .3 → 0.567 @ 13.8 | 0.556 @ 12.6 | 26 of 28 | 7 of 8: λ = 2 by GMS k = 2; λ = 1.5 by EAMA λ = .5, GMS k = 3; λ = 1 by GMS k = 4; λ = .6 by EAMA λ = .1, EAMA (weighted) λ = .1 (+1 more); λ = 1.25 by EAMA (weighted) λ = .1, GMS k = 4; λ = .4 by EAMA λ = .05, EAMA (weighted) λ = .05 (+2 more); λ = .2 by GMS k = 5, GMS k = 6 |
| UTD-MHAD | 2 → 0.582 @ 515; 1.5 → 0.607 @ 518; 1.25 → 0.641 @ 526; 1 → 0.655 @ 527; .8 → 0.795 @ 568; .9 → 0.802 @ 573; .7 → 0.812 @ 595; .6 → 0.828 @ 614; .3 → 0.837 @ 703; .1 → 0.853 @ 987; .05 → 0.858 @ 1004 | 0.827 @ 659 | 16 of 23 | 4 of 11: λ = 2 by EAMA (weighted) λ = 1, Joint-AFA r_cost = .5; λ = 1.5 by EAMA (weighted) λ = 1, Joint-AFA r_cost = .15 (+2 more); λ = 1.25 by EAMA (weighted) λ = 1, Joint-AFA r_cost = .15 (+1 more); λ = 1 by Joint-AFA r_cost = .15, GMS k = 2 |

