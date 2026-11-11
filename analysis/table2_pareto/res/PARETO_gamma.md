# SeMARC over γ at Table 2's λ: F1–latency (Table 2 metric)

SeMARC with λ held at Table 2's value (the deployed per-fold λ: .5, MM-Fi fold 2 = 1.0) and only γ varied (w_m = (t_m / min t)^γ). These are the paper's own γ-study nets (Appendix tab:wgamma-alldata), reproduced exactly (verify.py). γ = 0 is Table 2's SeMARC; γ = .75 is the paper's weighted row. Latency: estimated GPU latency in ms. F1: macro-F1, mean over three test folds. Comparison points: Appendix tab:comp-replace-complete (mean = average of the six datasets).

Published points vs γ = 0 / .75 here: max |ΔF1| = 0.0005, max relative |Δlatency| = 0.20% (rounding of the published values).

## Is the configuration beaten by SeMARC at some γ (F1 ≥ and latency ≤, one strictly)?

| Configuration | IEMOCAP | MM-Fi | CZU-MHAD | EAV | CMI | UTD-MHAD | Mean of six datasets |
|---|---|---|---|---|---|---|---|
| MultiModN + ARC | yes (γ .75, 1, 1.5) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | no | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | no | yes (γ .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) |
| MultiModN | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) |
| SeMA + EAMA | yes (γ .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | no | yes (γ .25, .5, .75, 1, 1.5, 2) | no | yes (γ .25, .5, .75, 1, 1.5, 2) | yes (γ .25, .5, .75, 1, 1.5, 2) |
| SeMA + Joint-AFA | yes (γ .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | yes (γ 0, .25, .5) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | no | yes (γ 0) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) |
| SeMA + GMS | no | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | no | no | no | no | yes (γ 0, .25, .5, .75, 1, 1.5, 2) |
| SeMA + EAMA (weighted) | yes (γ .75) | yes (γ 0, .25, .5, .75, 1, 1.5, 2) | no | yes (γ .5, .75, 2) | no | yes (γ 0, .25) | yes (γ .5, .75, 2) |

## IEMOCAP

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.669 | 745 | 0.65 | 3.90 |  |
| .25 | 0.672 | 630 | 0.79 | 4.75 | yes |
| .5 | 0.658 | 225 | 0.72 | 4.30 |  |
| .75 | 0.670 | 87.3 | 0.75 | 4.48 | yes |
| 1 | 0.653 | 81.1 | 0.71 | 4.26 | yes |
| 1.5 | 0.651 | 82.1 | 0.71 | 4.28 |  |
| 2 | 0.658 | 230 | 0.76 | 4.54 |  |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.535 | 199 | 0.670 (γ=0.75) | 81.1 (γ=1) | yes |
| MultiModN | 0.580 | 1530 | 0.672 (γ=0.25) | 81.1 (γ=1) | yes |
| SeMA + EAMA | 0.650 | 338 | 0.670 (γ=0.75) | 81.1 (γ=1) | yes |
| SeMA + Joint-AFA | 0.621 | 332 | 0.670 (γ=0.75) | 81.1 (γ=1) | yes |
| SeMA + GMS | 0.676 | 95.9 | 0.670 (γ=0.75) | – (none) | no |
| SeMA + EAMA (weighted) | 0.667 | 187 | 0.670 (γ=0.75) | 87.3 (γ=0.75) | yes |

## MM-Fi

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.950 | 187 | 0.21 | 1.06 |  |
| .25 | 0.946 | 58.4 | 0.21 | 1.05 | yes |
| .5 | 0.954 | 62.1 | 0.32 | 1.61 | yes |
| .75 | 0.952 | 68.7 | 0.35 | 1.75 |  |
| 1 | 0.949 | 139 | 0.56 | 2.80 |  |
| 1.5 | 0.947 | 134 | 0.51 | 2.57 |  |
| 2 | 0.949 | 97.0 | 0.40 | 2.02 |  |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.377 | 4004 | 0.954 (γ=0.5) | 58.4 (γ=0.25) | yes |
| MultiModN | 0.397 | 4234 | 0.954 (γ=0.5) | 58.4 (γ=0.25) | yes |
| SeMA + EAMA | 0.945 | 282 | 0.954 (γ=0.5) | 58.4 (γ=0.25) | yes |
| SeMA + Joint-AFA | 0.945 | 2485 | 0.954 (γ=0.5) | 58.4 (γ=0.25) | yes |
| SeMA + GMS | 0.759 | 2677 | 0.954 (γ=0.5) | 58.4 (γ=0.25) | yes |
| SeMA + EAMA (weighted) | 0.945 | 226 | 0.954 (γ=0.5) | 58.4 (γ=0.25) | yes |

## CZU-MHAD

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.914 | 1301 | 0.37 | 2.62 | yes |
| .25 | 0.914 | 1301 | 0.38 | 2.63 |  |
| .5 | 0.920 | 1307 | 0.55 | 3.82 | yes |
| .75 | 0.916 | 1314 | 0.70 | 4.88 |  |
| 1 | 0.920 | 1313 | 0.70 | 4.87 | yes |
| 1.5 | 0.922 | 1315 | 0.74 | 5.21 | yes |
| 2 | 0.922 | 1315 | 0.75 | 5.25 |  |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.485 | 1294 | – (none) | 1301 (γ=0) | no |
| MultiModN | 0.467 | 1324 | 0.922 (γ=1.5) | 1301 (γ=0) | yes |
| SeMA + EAMA | 0.899 | 1296 | – (none) | 1301 (γ=0) | no |
| SeMA + Joint-AFA | 0.911 | 1311 | 0.920 (γ=0.5) | 1301 (γ=0) | yes |
| SeMA + GMS | 0.917 | 1306 | 0.914 (γ=0) | 1307 (γ=0.5) | no |
| SeMA + EAMA (weighted) | 0.919 | 1305 | 0.914 (γ=0) | 1307 (γ=0.5) | no |

## EAV

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.689 | 217 | 0.37 | 1.11 |  |
| .25 | 0.690 | 29.1 | 0.37 | 1.12 | yes |
| .5 | 0.697 | 156 | 0.61 | 1.82 | yes |
| .75 | 0.694 | 114 | 0.60 | 1.81 | yes |
| 1 | 0.691 | 56.3 | 0.50 | 1.51 | yes |
| 1.5 | 0.691 | 55.4 | 0.50 | 1.51 | yes |
| 2 | 0.692 | 99.6 | 0.62 | 1.85 | yes |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.647 | 319 | 0.697 (γ=0.5) | 29.1 (γ=0.25) | yes |
| MultiModN | 0.647 | 2032 | 0.697 (γ=0.5) | 29.1 (γ=0.25) | yes |
| SeMA + EAMA | 0.687 | 216 | 0.697 (γ=0.5) | 29.1 (γ=0.25) | yes |
| SeMA + Joint-AFA | 0.683 | 603 | 0.697 (γ=0.5) | 29.1 (γ=0.25) | yes |
| SeMA + GMS | 0.688 | 26.7 | – (none) | 29.1 (γ=0.25) | no |
| SeMA + EAMA (weighted) | 0.692 | 163 | 0.697 (γ=0.5) | 99.6 (γ=2) | yes |

## CMI

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.556 | 12.6 | 0.75 | 5.26 | yes |
| .25 | 0.548 | 10.1 | 0.60 | 4.21 | yes |
| .5 | 0.550 | 10.4 | 0.63 | 4.38 | yes |
| .75 | 0.552 | 12.2 | 0.73 | 5.09 |  |
| 1 | 0.555 | 12.1 | 0.72 | 5.07 | yes |
| 1.5 | 0.554 | 12.2 | 0.74 | 5.15 |  |
| 2 | 0.548 | 12.4 | 0.75 | 5.25 |  |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.386 | 4.4 | – (none) | 10.1 (γ=0.25) | no |
| MultiModN | 0.447 | 15.5 | 0.556 (γ=0) | 10.1 (γ=0.25) | yes |
| SeMA + EAMA | 0.523 | 6.8 | – (none) | 10.1 (γ=0.25) | no |
| SeMA + Joint-AFA | 0.517 | 10.0 | – (none) | 10.1 (γ=0.25) | no |
| SeMA + GMS | 0.567 | 10.9 | 0.550 (γ=0.5) | – (none) | no |
| SeMA + EAMA (weighted) | 0.533 | 8.0 | – (none) | 10.1 (γ=0.25) | no |

## UTD-MHAD

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.827 | 659 | 0.52 | 2.07 | yes |
| .25 | 0.817 | 617 | 0.61 | 2.46 | yes |
| .5 | 0.800 | 599 | 0.65 | 2.59 | yes |
| .75 | 0.788 | 537 | 0.64 | 2.56 | yes |
| 1 | 0.791 | 544 | 0.64 | 2.56 |  |
| 1.5 | 0.791 | 539 | 0.64 | 2.58 | yes |
| 2 | 0.797 | 566 | 0.64 | 2.55 | yes |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.734 | 653 | 0.817 (γ=0.25) | 537 (γ=0.75) | yes |
| MultiModN | 0.660 | 1036 | 0.827 (γ=0) | 537 (γ=0.75) | yes |
| SeMA + EAMA | 0.757 | 642 | 0.817 (γ=0.25) | 537 (γ=0.75) | yes |
| SeMA + Joint-AFA | 0.817 | 780 | 0.827 (γ=0) | 659 (γ=0) | yes |
| SeMA + GMS | 0.793 | 516 | – (none) | 566 (γ=2) | no |
| SeMA + EAMA (weighted) | 0.807 | 662 | 0.827 (γ=0) | 617 (γ=0.25) | yes |

## Mean of six datasets (Table 2)

| γ | F1 | latency (ms) | MU | modalities / sample | on the front |
|---|---|---|---|---|---|
| 0 | 0.767 | 520 | 0.48 | 2.67 | yes |
| .25 | 0.764 | 441 | 0.49 | 2.70 | yes |
| .5 | 0.763 | 393 | 0.58 | 3.09 | yes |
| .75 | 0.762 | 355 | 0.63 | 3.43 | yes |
| 1 | 0.760 | 358 | 0.64 | 3.51 |  |
| 1.5 | 0.759 | 356 | 0.64 | 3.55 |  |
| 2 | 0.761 | 387 | 0.65 | 3.58 |  |

| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |
|---|---|---|---|---|---|
| MultiModN + ARC | 0.527 | 1079 | 0.767 (γ=0) | 355 (γ=0.75) | yes |
| MultiModN | 0.533 | 1695 | 0.767 (γ=0) | 355 (γ=0.75) | yes |
| SeMA + EAMA | 0.744 | 463 | 0.764 (γ=0.25) | 355 (γ=0.75) | yes |
| SeMA + Joint-AFA | 0.749 | 920 | 0.767 (γ=0) | 355 (γ=0.75) | yes |
| SeMA + GMS | 0.733 | 772 | 0.767 (γ=0) | 355 (γ=0.75) | yes |
| SeMA + EAMA (weighted) | 0.760 | 425 | 0.763 (γ=0.5) | 355 (γ=0.75) | yes |
