# F1–latency curves of the Table 2 controllers (Table 2 metric)

Request: for SeMA + EAMA, SeMA + Joint-AFA and SeMA + GMS, vary each controller's own parameter (retraining where needed) to get an F1–latency curve, and plot all curves with SeMARC's in one figure (`ctrl_curves.py`, `fig/pareto_controllers_mean.{pdf,png}`).

## What was varied

| Configuration | Knob | Values | New training? |
|---|---|---|---|
| SeMARC | γ (modality weights), λ held at Table 2's .5 (MM-Fi fold 2: 1) | 0, .25, .5, .75, 1, 1.5, 2 | no: the paper's γ-study nets |
| SeMA + EAMA | λ in J(q) = score(q) − λ Σ z_j, uniform z_j = 1/M (Table 2's row) | .005, .01, .02, .05, .1, .2, .5, 1, 2 | no: test-time; the paper's value models (3 seeds × 3 folds) were already run at every λ |
| SeMA + EAMA (weighted) | the same λ, z_j = extraction-time fraction (the weighted row) | same | no |
| SeMA + Joint-AFA | r_cost, the per-acquisition cost in the reward | 0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5 | yes: retrained per value with the paper's code and seed (0), 20000 episodes |
| SeMA + GMS | budget k of the greedy chain (selected on validation in Table 2) | 1 … M | no: the stored chains and test F1 at every k |

- **Latency** is Table 2's estimate (Appendix `app:comp-replace-complete`).
  - EAMA and Joint-AFA: `price_raw_selectors.py`, i.e. extraction of the acquired set, one fusion of it, and one selector-proxy call per decision.
  - GMS: extraction plus one fusion, with no selector (`reprice_controllers_e2e.py`).
  - The same per-modality idle-L40 tables throughout.
- **F1**: macro-F1, mean over the three test folds (EAMA: also over its 3 value-model seeds).
- **Mean of six datasets**: one knob value shared by all six datasets, as SeMARC's γ. GMS: k capped at each dataset's M.
- **Curve**: each method's Pareto front over its knob values. Points another value of the same knob beats are not drawn.
- **Table 2 configuration**: the published point, drawn as a black-edged marker.
  - EAMA: λ chosen per fold and seed on validation.
  - Joint-AFA: r_cost = .05, the paper's run.
  - GMS: k chosen per dataset on validation.
  - These are not knob values shared by all six datasets, so they need not lie on the curve.

## Checks

- **Prefix caches.** In this environment the caches are the ones the paper's runs read (`ctrl/check_cache.py`):
  - re-reading the paper's stored Joint-AFA per-sample sets gives its per-fold test F1 on all 18 folds;
  - re-scoring GMS's stored sets gives its F1 at every k (|ΔF1| ≤ .00005, the stored rounding).
- **Published points.** Every Table 2 point is recomputed from stored files and matches Table 2 / `tab:comp-replace-complete`:

| Configuration | F1 (mean) | latency ms (mean) | Table 2 |
|---|---|---|---|
| SeMA + EAMA | 0.7436 | 463.51 | .744, 463.50 |
| SeMA + EAMA (weighted) | 0.7605 | 425.15 | .761, 425.17 |
| SeMA + Joint-AFA | 0.7490 | 920.31 | .749, 920.32 |
| SeMA + GMS | 0.7333 | 772.11 | .733, 772 (48.4% above 520.17) |

- **EAMA's fusion term away from the selected λ.** Only the selected λ has stored per-modality frequencies. Elsewhere the fusion term uses the mean set size: per-modality fusion steps differ by < 1% within a dataset. At the selected λ the two prices differ by at most 0.004 ms.

- **Joint-AFA retraining device.** The paper's r_cost = .05 run was trained on a GPU. On the shared GPUs 2/3 (another user's jobs at 100%) the same code ran at < 10 episodes/s against ~100 on CPU, so every r_cost was retrained on CPU, .05 included. Same code, seed and data; only floating-point numerics differ. Table 2's marker is the paper's run; the curve uses the CPU runs.
- **GPU determinism.** A GPU rerun of the paper's run (IEMOCAP fold 0, r_cost = .05; `ctrl/run_jafa.sh`) replayed the paper's trajectory exactly: replay buffer 8221 / 17189 / 26386 transitions at episodes 2000 / 4000 / 6000, as in `logs_raw_sel/jafa_raw_iemocap.log`. It was stopped there because it ran at 3 episodes/s. The CPU retrain at .05 differs from the paper's run by about as much as Joint-AFA's r_cost values differ from each other (table below): a single Joint-AFA run per fold is noisy, as in the paper.

| Dataset | paper run F1 / MU / ms | CPU retrain at .05 F1 / MU / ms |
|---|---|---|
| IEMOCAP | 0.621 / 0.34 / 332 | 0.632 / 0.48 / 617 |
| MM-Fi | 0.945 / 0.49 / 2485 | 0.957 / 0.30 / 1097 |
| CZU-MHAD | 0.911 / 0.62 / 1311 | 0.891 / 0.43 / 1304 |
| EAV | 0.683 / 0.51 / 603 | 0.670 / 0.64 / 693 |
| CMI | 0.517 / 0.62 / 10.0 | 0.519 / 0.61 / 9.8 |
| UTD-MHAD | 0.817 / 0.72 / 780 | 0.771 / 0.75 / 816 |

## Mean of six datasets

**SeMARC** (γ)

| γ | F1 | MU | latency (ms) | on the front |
|---|---|---|---|---|
| 0 | 0.767 | 0.48 | 520 | yes |
| .25 | 0.764 | 0.49 | 441 | yes |
| .5 | 0.763 | 0.58 | 393 | yes |
| .75 | 0.762 | 0.63 | 355 | yes |
| 1 | 0.760 | 0.64 | 358 |  |
| 1.5 | 0.759 | 0.64 | 356 |  |
| 2 | 0.761 | 0.65 | 387 |  |

**SeMA + EAMA** (λ)

| λ | F1 | MU | latency (ms) | on the front |
|---|---|---|---|---|
| .005 | 0.773 | 0.68 | 830 | yes |
| .01 | 0.773 | 0.66 | 781 | yes |
| .02 | 0.772 | 0.64 | 740 | yes |
| .05 | 0.771 | 0.59 | 644 | yes |
| .1 | 0.766 | 0.52 | 552 | yes |
| .2 | 0.757 | 0.43 | 482 | yes |
| .5 | 0.738 | 0.33 | 400 | yes |
| 1 | 0.692 | 0.24 | 359 | yes |
| 2 | 0.496 | 0.13 | 307 | yes |
| Table 2 configuration | 0.744 | 0.39 | 464 | |

**SeMA + EAMA (weighted)** (λ)

| λ | F1 | MU | latency (ms) | on the front |
|---|---|---|---|---|
| .005 | 0.773 | 0.70 | 857 |  |
| .01 | 0.773 | 0.69 | 795 |  |
| .02 | 0.772 | 0.68 | 753 |  |
| .05 | 0.774 | 0.65 | 673 | yes |
| .1 | 0.767 | 0.61 | 595 | yes |
| .2 | 0.760 | 0.56 | 509 | yes |
| .5 | 0.750 | 0.50 | 363 | yes |
| 1 | 0.574 | 0.44 | 60.8 | yes |
| 2 | 0.534 | 0.39 | 20.6 | yes |
| Table 2 configuration | 0.761 | 0.54 | 425 | |

**SeMA + Joint-AFA** (r_cost)

| r_cost | F1 | MU | latency (ms) | on the front |
|---|---|---|---|---|
| 0 | 0.758 | 0.85 | 1459 | yes |
| .01 | 0.683 | 0.70 | 946 |  |
| .02 | 0.728 | 0.66 | 1170 |  |
| .03 | 0.742 | 0.65 | 921 |  |
| .05 | 0.740 | 0.54 | 756 |  |
| .075 | 0.738 | 0.40 | 655 |  |
| .1 | 0.743 | 0.40 | 517 | yes |
| .15 | 0.680 | 0.33 | 499 |  |
| .2 | 0.737 | 0.34 | 536 |  |
| .3 | 0.724 | 0.29 | 432 | yes |
| .5 | 0.672 | 0.25 | 384 | yes |
| Table 2 configuration | 0.749 | 0.55 | 920 | |

**SeMA + GMS** (k)

| k | F1 | MU | latency (ms) | on the front |
|---|---|---|---|---|
| 1 | 0.649 | 0.21 | 753 | yes |
| 2 | 0.754 | 0.41 | 993 | yes |
| 3 | 0.767 | 0.62 | 1418 | yes |
| 4 | 0.773 | 0.77 | 1438 | yes |
| 5 | 0.775 | 0.88 | 1454 | yes |
| 6 | 0.775 | 0.95 | 1694 |  |
| 7 | 0.774 | 1.00 | 1695 |  |
| Table 2 configuration | 0.733 | 0.53 | 772 | |

## Who beats whom on the mean

Beaten = some point has F1 ≥ and latency ≤, one strictly. 'SeMARC γ' = the blue curve (Table 2's λ). 'SeMARC γ × λ' = the supplement's whole sweep (λ varied too, as the controllers' knobs are).

| Controller | point | F1 | latency (ms) | beaten by SeMARC γ | SeMARC γ best F1 at ≤ its latency | beaten by SeMARC γ × λ |
|---|---|---|---|---|---|---|
| SeMA + EAMA | λ = 2 | 0.496 | 307 | no | – | yes |
| SeMA + EAMA | λ = 1 | 0.692 | 359 | γ = .75 | 0.762 | yes |
| SeMA + EAMA | λ = .5 | 0.738 | 400 | γ = .5, γ = .75 | 0.763 | yes |
| SeMA + EAMA | λ = .2 | 0.757 | 482 | γ = .25, γ = .5, γ = .75 | 0.764 | yes |
| SeMA + EAMA | λ = .1 | 0.766 | 552 | γ = 0 | 0.767 | yes |
| SeMA + EAMA | λ = .05 | 0.771 | 644 | no | 0.767 | yes |
| SeMA + EAMA | λ = .02 | 0.772 | 740 | no | 0.767 | no |
| SeMA + EAMA | λ = .01 | 0.773 | 781 | no | 0.767 | no |
| SeMA + EAMA | λ = .005 | 0.773 | 830 | no | 0.767 | no |
| SeMA + EAMA | Table 2 | 0.744 | 464 | γ = .25, γ = .5, γ = .75 | 0.764 | yes |
| SeMA + EAMA (weighted) | λ = 2 | 0.534 | 20.6 | no | – | yes |
| SeMA + EAMA (weighted) | λ = 1 | 0.574 | 60.8 | no | – | no |
| SeMA + EAMA (weighted) | λ = .5 | 0.750 | 363 | γ = .75 | 0.762 | yes |
| SeMA + EAMA (weighted) | λ = .2 | 0.760 | 509 | γ = .25, γ = .5, γ = .75 | 0.764 | yes |
| SeMA + EAMA (weighted) | λ = .1 | 0.767 | 595 | γ = 0 | 0.767 | yes |
| SeMA + EAMA (weighted) | λ = .05 | 0.774 | 673 | no | 0.767 | no |
| SeMA + EAMA (weighted) | Table 2 | 0.761 | 425 | γ = .5, γ = .75 | 0.763 | yes |
| SeMA + Joint-AFA | r_cost = .5 | 0.672 | 384 | γ = .75 | 0.762 | yes |
| SeMA + Joint-AFA | r_cost = .3 | 0.724 | 432 | γ = .5, γ = .75 | 0.763 | yes |
| SeMA + Joint-AFA | r_cost = .1 | 0.743 | 517 | γ = .25, γ = .5, γ = .75 | 0.764 | yes |
| SeMA + Joint-AFA | r_cost = 0 | 0.758 | 1459 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 | yes |
| SeMA + Joint-AFA | Table 2 | 0.749 | 920 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 | yes |
| SeMA + GMS | k = 1 | 0.649 | 753 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 | yes |
| SeMA + GMS | k = 2 | 0.754 | 993 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 | yes |
| SeMA + GMS | k = 3 | 0.767 | 1418 | γ = 0 | 0.767 | yes |
| SeMA + GMS | k = 4 | 0.773 | 1438 | no | 0.767 | no |
| SeMA + GMS | k = 5 | 0.775 | 1454 | no | 0.767 | no |
| SeMA + GMS | Table 2 | 0.733 | 772 | γ = 0, γ = .25, γ = .5, γ = .75 | 0.767 | yes |

| SeMARC γ (front) | F1 | latency (ms) | beaten by a controller point |
|---|---|---|---|
| .75 | 0.762 | 355 | none |
| .5 | 0.763 | 393 | none |
| .25 | 0.764 | 441 | none |
| 0 | 0.767 | 520 | none |

## Per dataset: each method's front

**IEMOCAP**

| Method | front: knob → F1 @ ms | Table 2 configuration |
|---|---|---|
| SeMARC | 1 → 0.653 @ 81.1; .75 → 0.670 @ 87.3; .25 → 0.672 @ 630 | γ = 0 and .75 on the curve |
| SeMA + EAMA | 2 → 0.279 @ 57.5; 1 → 0.548 @ 146; .5 → 0.627 @ 236; .2 → 0.659 @ 315; .1 → 0.659 @ 347; .05 → 0.660 @ 381; .02 → 0.663 @ 571; .01 → 0.665 @ 593; .005 → 0.667 @ 620 | 0.650 @ 338 |
| SeMA + EAMA (weighted) | 2 → 0.631 @ 39.3; 1 → 0.647 @ 56.6; .5 → 0.655 @ 115; .2 → 0.660 @ 205; .1 → 0.668 @ 258; .05 → 0.669 @ 310 | 0.667 @ 187 |
| SeMA + Joint-AFA | .5 → 0.452 @ 42.6; .3 → 0.620 @ 174; .1 → 0.627 @ 368; .03 → 0.628 @ 512; .05 → 0.632 @ 617; 0 → 0.680 @ 1411 | 0.621 @ 332 |
| SeMA + GMS | 1 → 0.598 @ 5.2; 2 → 0.657 @ 70.8; 3 → 0.661 @ 72.9; 4 → 0.676 @ 81.8; 5 → 0.676 @ 95.9 | 0.676 @ 95.9 |

- Controller front points not beaten by any SeMARC γ: 10 of 26.
- SeMARC γ front points beaten by some controller point: 3 of 3: γ = 1 (0.653 @ 81.1) by GMS k = 2, GMS k = 3; γ = .75 (0.670 @ 87.3) by GMS k = 4; γ = .25 (0.672 @ 630) by GMS k = 4, GMS k = 5

**MM-Fi**

| Method | front: knob → F1 @ ms | Table 2 configuration |
|---|---|---|
| SeMARC | .25 → 0.946 @ 58.4; .5 → 0.954 @ 62.1 | γ = 0 and .75 on the curve |
| SeMA + EAMA | 1 → 0.941 @ 22.2; .5 → 0.942 @ 42.8; .2 → 0.944 @ 177; .1 → 0.947 @ 280; .05 → 0.947 @ 495; .02 → 0.950 @ 799; .01 → 0.951 @ 965; .005 → 0.955 @ 1174 | 0.945 @ 282 |
| SeMA + EAMA (weighted) | 2 → 0.941 @ 29.9; .5 → 0.944 @ 91.3; .2 → 0.949 @ 614; .1 → 0.951 @ 753; .05 → 0.951 @ 866; .01 → 0.953 @ 1174; .005 → 0.954 @ 1327 | 0.945 @ 226 |
| SeMA + Joint-AFA | .5 → 0.941 @ 99.8; .3 → 0.946 @ 225; .1 → 0.951 @ 241; .2 → 0.955 @ 324; .05 → 0.957 @ 1097 | 0.945 @ 2485 |
| SeMA + GMS | 1 → 0.759 @ 2677; 2 → 0.959 @ 4026; 3 → 0.964 @ 4064; 5 → 0.968 @ 4236 | 0.759 @ 2677 |

- Controller front points not beaten by any SeMARC γ: 10 of 24.
- SeMARC γ front points beaten by some controller point: 0 of 2.

**CZU-MHAD**

| Method | front: knob → F1 @ ms | Table 2 configuration |
|---|---|---|
| SeMARC | 0 → 0.914 @ 1301; .5 → 0.920 @ 1307; 1 → 0.920 @ 1313; 1.5 → 0.922 @ 1315 | γ = 0 and .75 on the curve |
| SeMA + EAMA | 2 → 0.889 @ 1294; 1 → 0.890 @ 1294; .5 → 0.894 @ 1294; .2 → 0.905 @ 1295; .1 → 0.913 @ 1298; .05 → 0.918 @ 1301; .02 → 0.919 @ 1304 | 0.899 @ 1296 |
| SeMA + EAMA (weighted) | 2 → 0.074 @ 14.8; 1 → 0.075 @ 17.7; .5 → 0.919 @ 1305 | 0.919 @ 1305 |
| SeMA + Joint-AFA | .3 → 0.889 @ 1294; .1 → 0.892 @ 1296; .03 → 0.901 @ 1309; .02 → 0.905 @ 1311; .01 → 0.907 @ 1323; 0 → 0.916 @ 1324 | 0.911 @ 1311 |
| SeMA + GMS | 1 → 0.889 @ 1293; 2 → 0.919 @ 1295; 3 → 0.919 @ 1301; 5 → 0.920 @ 1312; 6 → 0.921 @ 1318 | 0.917 @ 1306 |

- Controller front points not beaten by any SeMARC γ: 16 of 21.
- SeMARC γ front points beaten by some controller point: 2 of 4: γ = 0 (0.914 @ 1301) by EAMA λ = .05, GMS k = 2; γ = 1 (0.920 @ 1313) by GMS k = 5

**EAV**

| Method | front: knob → F1 @ ms | Table 2 configuration |
|---|---|---|
| SeMARC | .25 → 0.690 @ 29.1; 1.5 → 0.691 @ 55.4; 1 → 0.691 @ 56.3; 2 → 0.692 @ 99.6; .75 → 0.694 @ 114; .5 → 0.697 @ 156 | γ = 0 and .75 on the curve |
| SeMA + EAMA | 2 → 0.067 @ 0.2; 1 → 0.687 @ 151; .5 → 0.688 @ 205 | 0.687 @ 216 |
| SeMA + EAMA (weighted) | 2 → 0.690 @ 34.5; 1 → 0.690 @ 38.5; .5 → 0.692 @ 126 | 0.692 @ 163 |
| SeMA + Joint-AFA | .3 → 0.683 @ 343; .15 → 0.686 @ 448; .2 → 0.687 @ 544; .01 → 0.688 @ 619 | 0.683 @ 603 |
| SeMA + GMS | 1 → 0.688 @ 26.7; 2 → 0.694 @ 43.3 | 0.688 @ 26.7 |

- Controller front points not beaten by any SeMARC γ: 5 of 12.
- SeMARC γ front points beaten by some controller point: 4 of 6: γ = 1.5 (0.691 @ 55.4) by GMS k = 2; γ = 1 (0.691 @ 56.3) by GMS k = 2; γ = 2 (0.692 @ 99.6) by GMS k = 2; γ = .75 (0.694 @ 114) by GMS k = 2

**CMI**

| Method | front: knob → F1 @ ms | Table 2 configuration |
|---|---|---|
| SeMARC | .25 → 0.548 @ 10.1; .5 → 0.550 @ 10.4; 1 → 0.555 @ 12.1; 0 → 0.556 @ 12.6 | γ = 0 and .75 on the curve |
| SeMA + EAMA | 2 → 0.268 @ 1.7; 1 → 0.467 @ 3.6; .5 → 0.513 @ 5.5; .2 → 0.537 @ 7.8; .1 → 0.547 @ 9.8; .05 → 0.558 @ 11.1; .02 → 0.559 @ 11.9; .01 → 0.561 @ 12.3 | 0.523 @ 6.8 |
| SeMA + EAMA (weighted) | 2 → 0.406 @ 2.2; 1 → 0.448 @ 3.4; .5 → 0.506 @ 5.3; .2 → 0.535 @ 7.7; .1 → 0.547 @ 9.7; .05 → 0.558 @ 11.1; .02 → 0.558 @ 11.9; .01 → 0.561 @ 12.3; .005 → 0.562 @ 12.6 | 0.533 @ 8.0 |
| SeMA + Joint-AFA | .5 → 0.444 @ 3.3; .3 → 0.464 @ 3.8; .2 → 0.484 @ 5.3; .075 → 0.499 @ 8.1; .1 → 0.526 @ 8.6; .03 → 0.550 @ 11.1 | 0.517 @ 10.0 |
| SeMA + GMS | 1 → 0.391 @ 2.0; 2 → 0.503 @ 4.0; 3 → 0.537 @ 6.3; 4 → 0.563 @ 8.6; 5 → 0.567 @ 10.9 | 0.567 @ 10.9 |

- Controller front points not beaten by any SeMARC γ: 28 of 28.
- SeMARC γ front points beaten by some controller point: 4 of 4: γ = .25 (0.548 @ 10.1) by GMS k = 4; γ = .5 (0.550 @ 10.4) by GMS k = 4; γ = 1 (0.555 @ 12.1) by EAMA λ = .02, EAMA λ = .05, EAMA (weighted) λ = .02, EAMA (weighted) λ = .05, GMS k = 4, GMS k = 5; γ = 0 (0.556 @ 12.6) by EAMA λ = .01, EAMA λ = .02, EAMA λ = .05, EAMA (weighted) λ = .005, EAMA (weighted) λ = .01, EAMA (weighted) λ = .02, EAMA (weighted) λ = .05, GMS k = 4, GMS k = 5

**UTD-MHAD**

| Method | front: knob → F1 @ ms | Table 2 configuration |
|---|---|---|
| SeMARC | .75 → 0.788 @ 537; 1.5 → 0.791 @ 539; 2 → 0.797 @ 566; .5 → 0.800 @ 599; .25 → 0.817 @ 617; 0 → 0.827 @ 659 | γ = 0 and .75 on the curve |
| SeMA + EAMA | 2 → 0.535 @ 464; 1 → 0.620 @ 535; .5 → 0.761 @ 616; .2 → 0.813 @ 785; .1 → 0.841 @ 972; .05 → 0.860 @ 1019; .02 → 0.860 @ 1023; .005 → 0.861 @ 1036 | 0.757 @ 642 |
| SeMA + EAMA (weighted) | 2 → 0.462 @ 3.2; 1 → 0.642 @ 214; .5 → 0.785 @ 534; .2 → 0.808 @ 670; .1 → 0.833 @ 843; .05 → 0.860 @ 1020 | 0.807 @ 662 |
| SeMA + Joint-AFA | .5 → 0.628 @ 353; .15 → 0.757 @ 517; .2 → 0.781 @ 593; .075 → 0.785 @ 727; .03 → 0.848 @ 940; 0 → 0.848 @ 959 | 0.817 @ 780 |
| SeMA + GMS | 1 → 0.569 @ 515; 2 → 0.793 @ 516; 3 → 0.861 @ 1034 | 0.793 @ 516 |

- Controller front points not beaten by any SeMARC γ: 18 of 23.
- SeMARC γ front points beaten by some controller point: 2 of 6: γ = .75 (0.788 @ 537) by GMS k = 2; γ = 1.5 (0.791 @ 539) by GMS k = 2

