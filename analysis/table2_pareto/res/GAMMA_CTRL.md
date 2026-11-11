# SeMARC and SeMA + EAMA over the same cost weights γ, at Table 2's λ (Table 2 metric)

Request: fix λ for SeMARC and SeMA + EAMA, vary γ for SeMARC, and change each EAMA modality cost by the same percentage as SeMARC's; the other variants as in `fig/pareto_lambda_controllers_mean` (`gamma_ctrl.py`, `fig/pareto_gamma_controllers_mean.{pdf,png}`).

## Setup

- **SeMARC**: γ ∈ {0, .25, .5, .75, 1, 1.5, 2} at Table 2's λ (.5; MM-Fi fold 2: 1), the paper's γ-study nets (as in `fig/pareto_controllers_mean`). Modality j is priced λ w_j / Σ_m w_m with w_m = (t_m / min t)^γ.
- **SeMA + EAMA**: λ held at Table 2's value, each run's (fold × value-model seed) λ picked on validation at uniform cost. Its uniform cost z_j = 1/M is multiplied by the factor by which γ changes SeMARC's price of modality j, M w_j / Σ_m w_m. So z_j(γ) = t_j^γ / Σ_m t_m^γ.
  - γ = 0 is Table 2's EAMA row; γ = 1 uses the weighted row's costs t_j / Σ t, but at the uniform row's λ.
  - EAMA's costs act at test time only, but its value models were not stored. They were retrained with `baseline_eama_raw.py`'s own functions and seeds (`ctrl/eama_gamma.py`, GPUs 2/3).
  - Check: the retrained models reproduce the paper's run exactly: holdout AUC, every stored test row (both costs, all nine λ: F1, MU, extraction ms, stages) and both selected λ, on all 54 runs; largest difference 8.9e-16.
- **SeMA + GMS**: each dataset's Table 2 k (selected on validation: IEMOCAP 5, MM-Fi 1, CZU-MHAD 4, EAV 1, CMI 5, UTD-MHAD 2) scaled by the same percentage, −75% to +400%, rounded half up and clipped to 1 … M. 0% is Table 2's GMS. The stored greedy chains have test F1 at every k, so nothing is refitted.
- **Joint-AFA** over r_cost (retrained) as in `fig/pareto_lambda_controllers_mean`, with the paper's run as its marker; MultiModN + ARC as a point. Weighted EAMA is left out, at the user's request.
- **Mean**: one knob value shared by all six datasets. Lines are Pareto fronts; EAMA's and GMS's also run through their Table 2 points (EAMA's γ = 0 is beaten by its γ = .25; GMS's 0% is on its front).

## Check: EAMA at γ = 0 is Table 2's EAMA

| Dataset | F1 at γ = 0 | Table 2 F1 | latency at γ = 0 (ms) | Table 2 latency (ms) |
|---|---|---|---|---|
| IEMOCAP | 0.6503 | 0.6503 | 338.11 | 338.11 |
| MM-Fi | 0.9449 | 0.9449 | 282.25 | 282.25 |
| CZU-MHAD | 0.8988 | 0.8988 | 1295.91 | 1295.91 |
| EAV | 0.6874 | 0.6874 | 215.96 | 215.96 |
| CMI | 0.5233 | 0.5233 | 6.77 | 6.77 |
| UTD-MHAD | 0.7570 | 0.7570 | 642.04 | 642.04 |
| Mean of six datasets | 0.7436 | 0.7436 | 463.51 | 463.51 |

## Mean of six datasets: SeMARC and EAMA at the same γ

| γ | SeMARC F1 | SeMARC latency (ms) | EAMA F1 | EAMA latency (ms) | at this γ, better on both |
|---|---|---|---|---|---|
| 0 | 0.767 | 520 | 0.744 | 464 | neither |
| .25 | 0.764 | 441 | 0.745 | 416 | neither |
| .5 | 0.763 | 393 | 0.734 | 388 | neither |
| .75 | 0.762 | 355 | 0.671 | 312 | neither |
| 1 | 0.760 | 358 | 0.671 | 342 | neither |
| 1.5 | 0.759 | 356 | 0.671 | 394 | SeMARC |
| 2 | 0.761 | 387 | 0.670 | 423 | SeMARC |

EAMA's γ front on the mean: γ = .75: 0.671 @ 312; γ = 1: 0.671 @ 342; γ = .5: 0.734 @ 388; γ = .25: 0.745 @ 416

## SeMA + GMS with Table 2's k scaled

| change in k | k per dataset (IEMOCAP, MM-Fi, CZU-MHAD, EAV, CMI, UTD-MHAD) | mean F1 | mean latency (ms) | on the front |
|---|---|---|---|---|
| −75% | 1 / 1 / 1 / 1 / 1 / 1 | 0.649 | 753 | yes |
| −50% | 3 / 1 / 2 / 1 / 3 / 1 | 0.689 | 765 | yes |
| −25% | 4 / 1 / 3 / 1 / 4 / 2 | 0.733 | 768 | yes |
| 0 | 5 / 1 / 4 / 1 / 5 / 2 | 0.733 | 772 | yes |
| +25% | 6 / 1 / 5 / 1 / 6 / 3 | 0.745 | 1099 | yes |
| +50% | 6 / 2 / 6 / 2 / 7 / 3 | 0.779 | 1328 | yes |
| +100% | 6 / 2 / 7 / 2 / 7 / 4 | 0.779 | 1329 |  |
| +200% | 6 / 3 / 7 / 3 / 7 / 4 | 0.774 | 1667 |  |
| +400% | 6 / 5 / 7 / 3 / 7 / 4 | 0.774 | 1695 |  |
| Table 2 | 5 / 1 / 4 / 1 / 5 / 2 | 0.733 | 772 | |

## Who beats whom on the mean

Beaten = some point has F1 ≥ and latency ≤, one strictly. 'SeMARC γ' = the blue curve.

| Controller | point | F1 | latency (ms) | beaten by SeMARC γ | SeMARC γ best F1 at ≤ its latency |
|---|---|---|---|---|---|
| SeMA + EAMA | γ = .75 | 0.671 | 312 | no | – |
| SeMA + EAMA | γ = 1 | 0.671 | 342 | no | – |
| SeMA + EAMA | γ = .5 | 0.734 | 388 | γ = .75 | 0.762 |
| SeMA + EAMA | γ = .25 | 0.745 | 416 | γ = .5, γ = .75 | 0.763 |
| SeMA + EAMA | Table 2 | 0.744 | 464 | γ = .25, γ = .5, γ = .75 | 0.764 |
| SeMA + Joint-AFA | r_cost = .5 | 0.672 | 384 | γ = .75 | 0.762 |
| SeMA + Joint-AFA | r_cost = .3 | 0.724 | 432 | γ = .5, γ = .75 | 0.763 |
| SeMA + Joint-AFA | r_cost = .1 | 0.743 | 517 | γ = .25, γ = .5, γ = .75 | 0.764 |
| SeMA + Joint-AFA | r_cost = 0 | 0.758 | 1459 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + Joint-AFA | Table 2 | 0.749 | 920 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + GMS | k −75% | 0.649 | 753 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + GMS | k −50% | 0.689 | 765 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + GMS | k −25% | 0.733 | 768 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + GMS | k 0 | 0.733 | 772 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + GMS | k +25% | 0.745 | 1099 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |
| SeMA + GMS | k +50% | 0.779 | 1328 | no | 0.767 |
| SeMA + GMS | Table 2 | 0.733 | 772 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 |

| SeMARC γ (front) | F1 | latency (ms) | beaten by a controller point |
|---|---|---|---|
| .75 | 0.762 | 355 | none |
| .5 | 0.763 | 393 | none |
| .25 | 0.764 | 441 | none |
| 0 | 0.767 | 520 | none |

## Per dataset: SeMARC and EAMA at the same γ

| Dataset | γ where SeMARC is better on both | γ where EAMA is better on both | neither (one faster, the other more accurate) |
|---|---|---|---|
| IEMOCAP | .5, .75 | – | 0, .25, 1, 1.5, 2 |
| MM-Fi | 0, .25, .5, .75 | – | 1, 1.5, 2 |
| CZU-MHAD | – | – | 0, .25, .5, .75, 1, 1.5, 2 |
| EAV | .25, .75, 1, 1.5, 2 | – | 0, .5 |
| CMI | – | – | 0, .25, .5, .75, 1, 1.5, 2 |
| UTD-MHAD | – | – | 0, .25, .5, .75, 1, 1.5, 2 |

| Dataset | SeMARC γ front: γ → F1 @ ms | EAMA γ front: γ → F1 @ ms |
|---|---|---|
| IEMOCAP | 1 → 0.653 @ 81.1; .75 → 0.670 @ 87.3; .25 → 0.672 @ 630 | 1 → 0.661 @ 247; 1.5 → 0.669 @ 257; 2 → 0.669 @ 257 |
| MM-Fi | .25 → 0.946 @ 58.4; .5 → 0.954 @ 62.1 | .25 → 0.944 @ 184; .5 → 0.946 @ 262; .75 → 0.947 @ 391; 1 → 0.950 @ 571; 1.5 → 0.953 @ 864 |
| CZU-MHAD | 0 → 0.914 @ 1301; .5 → 0.920 @ 1307; 1 → 0.920 @ 1313; 1.5 → 0.922 @ 1315 | .75 → 0.461 @ 588; 1 → 0.462 @ 591; .5 → 0.830 @ 1154; 0 → 0.899 @ 1296; .25 → 0.906 @ 1297 |
| EAV | .25 → 0.690 @ 29.1; 1.5 → 0.691 @ 55.4; 1 → 0.691 @ 56.3; 2 → 0.692 @ 99.6; .75 → 0.694 @ 114; .5 → 0.697 @ 156 | 1 → 0.690 @ 121 |
| CMI | .25 → 0.548 @ 10.1; .5 → 0.550 @ 10.4; 1 → 0.555 @ 12.1; 0 → 0.556 @ 12.6 | 2 → 0.494 @ 6.4; 1.5 → 0.497 @ 6.4; .75 → 0.505 @ 6.4; .5 → 0.513 @ 6.5; .25 → 0.514 @ 6.5; 0 → 0.523 @ 6.8 |
| UTD-MHAD | .75 → 0.788 @ 537; 1.5 → 0.791 @ 539; 2 → 0.797 @ 566; .5 → 0.800 @ 599; .25 → 0.817 @ 617; 0 → 0.827 @ 659 | 2 → 0.760 @ 514; 1.5 → 0.761 @ 515; 1 → 0.762 @ 516; .75 → 0.765 @ 518; .5 → 0.766 @ 521 |

