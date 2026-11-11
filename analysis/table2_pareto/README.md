# Table 2 Pareto: SeMARC over γ or λ, EAMA over the same γ, and the controllers' curves (2026-11-09/10)

Request (user): Table 2 (`tab:component-swap`) compares configurations at uniform acquisition costs, where SeMA + EAMA has lower F1 and lower latency than SeMARC. Figure 10 (`fig:diag-reward`) shows that latency-based modality weights cut IEMOCAP latency sharply. Plot SeMARC's F1–latency Pareto curve on all datasets with Table 2's metric. Show the other configurations (SeMA + EAMA / Joint-AFA / GMS, MultiModN + ARC, MultiModN) as points.

Clarified by the user: vary γ (the weights) with λ held at Table 2's value. That is the main view below. The γ × λ sweep after it was my first reading (λ varied too); it is kept as a supplement.

## Main view: γ varied, λ as in Table 2 (`plot_gamma.py`)

- **Points.**
  - SeMARC at the deployed λ (.5; MM-Fi fold 2: 1.0, as in Table 2), γ ∈ {0, .25, .5, .75, 1, 1.5, 2}.
  - These are the paper's own γ-study nets (Appendix `tab:wgamma-alldata`), reproduced exactly by `verify.py`; no new fit.
  - γ = 0 is Table 2's SeMARC and γ = .75 the paper's weighted row. Both coincide with the published points: |ΔF1| ≤ .0005, latency within 0.2%.
- **Line.** The Pareto front over the seven γ values. Light dots are γ values that another γ beats on both F1 and latency.
- **Figures.**
  - `fig/pareto_gamma_mean.{pdf,png}`: the six-dataset mean on its own (Table 2's view), the figure the user wants.
    - At the user's request it draws only the front's γ values (0, .25, .5, .75). γ = 1, 1.5 and 2 are beaten by .75 and left out.
    - It also leaves out MultiModN without ARC.
  - `fig/pareto_gamma_f1_latency.{pdf,png}`: all six datasets plus the mean. Insets zoom MM-Fi, CZU-MHAD, CMI, UTD-MHAD and the mean.
- **Tables.**
  - `res/PARETO_gamma.md`: per-γ table and comparison verdicts.
  - `res/gamma_points.csv` and `res/dominance_gamma.csv`.
- **Mean of six datasets (Table 2's view).**
  - Every γ ≥ .25 beats SeMA + EAMA (.744 at 463 ms) on both F1 and latency:
    - γ = .25: .764 at 441 ms;
    - γ = .5: .763 at 393 ms;
    - γ = .75: .762 at 355 ms (23% faster).
  - Only γ = 0, Table 2's own row (.767 at 520 ms), is slower than EAMA.
  - The single paper value γ = .75 beats all six comparison rows on the mean, weighted EAMA (.7605 at 425 ms) included.
- **Per dataset: beaten by some γ (F1 ≥ and latency ≤, one strictly)?**
  - MultiModN: all six.
  - SeMA + Joint-AFA: five; not CMI (.517 at 10.0 ms against .548 at 10.06 ms).
  - SeMA + EAMA: four (IEMOCAP, MM-Fi, EAV, UTD-MHAD). γ = .75 alone gives the same four.
    - Not CMI: at this λ, SeMARC's fastest is 10.1 ms against EAMA's 6.8 ms.
    - Not CZU-MHAD: every SeMARC is ≥ 1301 ms against EAMA's 1296 ms, though SeMARC has higher F1 (≥ .914 against .899).
  - SeMA + EAMA (weighted): four; not CZU-MHAD (.919 at 1305 ms against .920 at 1307 ms) or CMI.
  - MultiModN + ARC: four; not CZU-MHAD or CMI, where it is faster but far lower in F1.
  - SeMA + GMS: MM-Fi only. On IEMOCAP, SeMARC's best F1 at this λ is .672, against GMS's .676 at 96 ms.
- **Caveat.** The per-dataset "some γ" verdicts pick γ on test. With the one paper value γ = .75, the mean verdicts and the EAMA verdicts are unchanged.

## Controller curves: each controller's own knob varied (`ctrl_curves.py`)

Request (user): for SeMA + EAMA, SeMA + Joint-AFA and SeMA + GMS, adjust each controller's parameter (retraining if needed) to get F1–latency curves; plot all curves with SeMARC's in one figure. Full tables: `res/CTRL_CURVES.md`; every point: `res/ctrl_points.csv`.

- **Figure.** `fig/pareto_controllers_mean.{pdf,png}`, the six-dataset mean.
  - Left: the full range (log latency). Middle: zoom on 300–700 ms. Right: legend.
  - Lines: each method's Pareto front over its knob.
  - Black-edged markers: each controller's Table 2 configuration.
- **Knobs.** Everything else stays at the paper's configuration.
  - SeMARC: γ at Table 2's λ (the main view above).
  - SeMA + EAMA: λ ∈ {.005, .01, .02, .05, .1, .2, .5, 1, 2}, at uniform cost (Table 2's row) and at weighted cost (the weighted row).
    - λ acts at test time only, and the paper's value models (3 seeds × 3 folds) were already run at every λ (`eama_raw_<ds>_idle_f*.csv`). No new fit.
  - SeMA + Joint-AFA: r_cost ∈ {0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5}.
    - r_cost is in the reward, so every value was retrained: 198 runs (11 × 18 folds).
    - The paper's code (`baseline_jafa_raw.py` train/evaluate, called unchanged by `ctrl/jafa_sweep.py`), seed 0, 20000 episodes, on CPU.
  - SeMA + GMS: budget k = 1 … M of the stored greedy chains (`greedy_submodular_*.json`). No new fit.
  - Mean: one knob value shared by all six datasets, as for SeMARC's γ. For GMS, k is capped at each dataset's M.
- **Checks.**
  - The prefix caches of this environment are the ones the paper's runs read. The paper's stored Joint-AFA and GMS sets re-score to their per-fold F1 on all 18 folds (`ctrl/check_cache.py`).
  - Every Table 2 point is recomputed from stored files: EAMA .7436 at 463.51 ms, weighted EAMA .7605 at 425.15, Joint-AFA .7490 at 920.31, GMS .7333 at 772.11.
  - EAMA's fusion term away from its selected λ uses the mean set size. Per-modality fusion steps differ by < 1% within a dataset; at the selected λ this is within 0.004 ms of the exact price.
- **Joint-AFA on CPU.**
  - On GPUs 2/3, shared with another user's jobs at 100%, the paper's code ran at 3 episodes/s against ~100 on CPU.
  - A GPU rerun of the paper's run (IEMOCAP fold 0, r_cost = .05) replayed its trajectory exactly through 6000 episodes: replay buffer 8221 / 17189 / 26386, as in the paper's log. It was then stopped.
  - The CPU retrain at r_cost = .05 differs from the paper's run by as much as neighbouring r_cost values differ: mean .740 at 756 ms against .749 at 920. Single-seed Joint-AFA is noisy, so its curve is jagged.
  - The curve uses the CPU runs; Table 2's marker is the paper's run.
- **Results, six-dataset mean (one shared knob value).**
  - No controller point beats any point of SeMARC's γ curve.
  - Every controller point between SeMARC's fastest point (355 ms) and 595 ms is beaten by it. Examples:
    - EAMA λ = .2: .757 at 482 ms, against γ = .25: .764 at 441.
    - Weighted EAMA λ = .5: .750 at 363 ms, against γ = .75: .762 at 355.
    - Weighted EAMA λ = .1: .767 at 595 ms, against γ = 0: .767 at 520.
  - Joint-AFA's whole front (.672 at 384 ms to .758 at 1459 ms) and GMS for k ≤ 3 are beaten.
  - Above SeMARC's range, some controller points reach higher F1 than SeMARC's curve, which ends at .767 (γ = 0, 520 ms):
    - EAMA with λ ≤ .05: .771–.774 at 644–857 ms;
    - GMS with k = 4–5: .773–.775 at ~1440 ms.
  - SeMARC's whole γ × λ sweep tops out at .771 (408 ms). So EAMA at λ ≤ .02, weighted EAMA at λ = .05 and GMS at k = 4–5 are not beaten by any SeMARC configuration either.
    - Against SeMARC's γ = 0 point, they gain .005–.008 F1 for 1.3–2.8× its latency.
  - The readout does not explain this. On the full modality set, the order-averaged readout of the pre-backbone controllers is .003 below SeMARC's single order (mean .774 against .777; `ctrl/readout_gap.py`).
  - Below SeMARC's fastest point (355 ms) the controllers have only low-F1 points: EAMA λ = 2 (.496 at 307 ms) and weighted EAMA λ = 1–2 (.574 at 61 ms, .534 at 21 ms).
- **Results, per dataset (each knob tuned to that dataset; `res/CTRL_CURVES.md`).** The mean advantage rests on one knob value having to serve all six datasets.
  - Per dataset, GMS beats some SeMARC γ points on IEMOCAP, CZU-MHAD, EAV, CMI and UTD-MHAD.
    - IEMOCAP: k = 4 gives .676 at 82 ms, against γ = .75 at .670 and 87 ms.
    - CMI: k = 4 gives .563 at 8.6 ms, against γ = 0 at .556 and 12.6 ms.
    - EAV: k = 2 gives .694 at 43 ms.
  - EAMA beats SeMARC's γ points on CMI (λ ≤ .05) and on CZU-MHAD (λ = .05).
  - Only MM-Fi has no SeMARC γ point beaten.
  - On the mean, GMS's shared budget picks MM-Fi's 4-second depth stream first on two folds, which puts every GMS mean point at ≥ 750 ms.
- **Caveats.**
  - Every curve, SeMARC's included, is a Pareto front over test-evaluated knob values.
  - The Table 2 markers are not shared-knob points, so they need not lie on the curves:
    - EAMA's λ is picked on validation in each run (per dataset, fold and value-model seed). The uniform marker lies .009 F1 below its curve, the weighted one .006 above.
    - GMS's k is one value per dataset.
    - Joint-AFA's marker is the paper's GPU run at the fixed r_cost = .05.

## SeMARC over λ at uniform cost, with the controller curves (`lam_curves.py`, 2026-11-10)

Request (user): the controller figure with SeMARC's curve replaced by λ varied at uniform cost (γ = 0). Full tables: `res/LAM_CURVES.md`; every point: `res/lam_points.csv`.

- **Figure.** `fig/pareto_lambda_controllers_mean.{pdf,png}`, the six-dataset mean, laid out as the controller figure (zoom on 280–720 ms). The controllers' curves and Table 2 markers are unchanged.
- **SeMARC's knob.** λ ∈ {.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.25, 1.5, 2} at γ = 0.
  - One λ is shared by every dataset and fold; everything else is the deployed setting per fold (`sweep.py`: seed, FQI schedule, availability augmentation, prior priced at the training λ, δ = 0).
  - λ = .05, .1, .2, .5, 1, 2 come from the γ × λ sweep below.
  - The eight values in between were fitted for this figure: 144 nets (6 datasets × 3 folds × 8 λ) with the unchanged `train_or_load`, on GPUs 2/3 (`lamu_launch.sh`; nets in `RL_AAAI_branch/checkpoints_t2p/<ds>/`).
  - Their results are in `res/lamu_*_g0.csv`, outside the `sweep_*` glob, so the earlier figures and tables do not change.
- **Markers.** The star is Table 2's SeMARC (λ = .5; MM-Fi fold 2: 1); the diamond is the paper's weighted row (γ = .75). Both recompute to the published rows (|ΔF1| ≤ .0005, latency within 0.3 ms).
- **Results, six-dataset mean.**
  - SeMARC's front: λ = 2 .697 at 311 ms; 1.25 .720 at 371; 1 .724 at 423; .9 .750 at 442; .7 .758 at 514; .5 .768 at 550.
  - The other λ are off the front:
    - Smaller λ acquire more without more F1 (λ = .05: .765 at 1284 ms).
    - λ = .6, .8 and 1.5 are beaten by neighbouring values. Latency is not monotone in λ because uniform cost cannot tell MM-Fi's 4 s depth stream from a 20 ms one: at λ = .8 the MM-Fi fits take depth on 35% of samples, against 7% at .7 and 3% at .9.
  - Against uniform EAMA, the two trade wins:
    - SeMARC beats EAMA's Table 2 point (λ = .9: .750 at 442 ms against .744 at 464), EAMA λ = .1 (λ = .5: .768 at 550 against .766 at 552) and EAMA λ = 1 (λ = 2: .697 at 311 against .692 at 359).
    - EAMA λ = .5 (.738 at 400 ms) beats SeMARC λ = 1 (.724 at 423). Nothing on SeMARC's curve beats EAMA λ = .2 (.757 at 482 ms); SeMARC's best at ≤ 482 ms is .750.
    - Above 600 ms, EAMA with λ ≤ .05 reaches .771–.773, above SeMARC's best on this curve (.768).
  - Weighted EAMA beats SeMARC's λ = 1.25, 1, .9 and .7 (e.g. .750 at 363 ms against .750 at 442).
    - Weighted EAMA prices extraction time, and uniform-cost SeMARC does not.
    - SeMARC's own weighted row (.762 at 355 ms) beats weighted EAMA's Table 2 point (.761 at 425) and its λ = .5 and .2 points.
  - Joint-AFA: every point is beaten except r_cost = .3 (.724 at 432 ms), which is level with λ = 1 (.724 at 423).
  - GMS: k ≤ 3 and its Table 2 point are beaten; k = 4–5 (.773–.775 at ~1440 ms) are above SeMARC's best.
  - Compared with the γ curve: there, SeMARC beat every controller point between 355 and 595 ms. At uniform cost its λ curve only trades wins with uniform EAMA and loses to weighted EAMA between 363 and 514 ms, so the cost weights carry the mean result.
- **Results, per dataset.** At uniform cost, some controller point beats SeMARC λ points on every dataset. Front points beaten: IEMOCAP 8 of 8, CZU-MHAD 6 of 6, CMI 7 of 8, EAV 4 of 6, UTD-MHAD 4 of 11, MM-Fi 1 of 6. The beaters are mostly GMS and EAMA. On IEMOCAP they are weighted EAMA and GMS, which rarely or never take the 1.4 s video (GMS's set: text, audio, mocap).
- **Caveats.**
  - SeMARC's curve, like the controllers', is a Pareto front over test-evaluated knob values.
  - In Table 2, SeMARC's λ is fixed (.5; MM-Fi fold 2: 1) and only the per-fold seed is selected on validation. Joint-AFA's r_cost is fixed too. EAMA's λ is selected on validation per run, and GMS's k per dataset (Appendix `tab:policy-config`, `app:baseline_adaptations`).

## SeMARC and SeMA + EAMA over the same cost weights γ, at Table 2's λ (`gamma_ctrl.py`, 2026-11-10)

Request (user): use the γ figure for SeMARC; fix λ for SeMARC and SeMA + EAMA; change each EAMA modality cost by the same percentage as SeMARC's cost of that modality changes with γ; the other variants as in `fig/pareto_lambda_controllers_mean`. Full tables: `res/GAMMA_CTRL.md`; every point: `res/gamma_ctrl_points.csv`.

- **Figure.** `fig/pareto_gamma_controllers_mean.{pdf,png}`, the six-dataset mean, zoom on 290–560 ms and F1 .66–.78.
  - Weighted EAMA (curve and Table 2 point) is left out, at the user's request.
  - EAMA's line also runs to its γ = 0 point (its Table 2 point, .744 at 464 ms), which its γ = .25 point (.745 at 416 ms) beats.
- **One-plot version (`pareto_single.py`, 2026-11-10): `fig/pareto_gamma_controllers_single.{pdf,png}`.** The user asked to combine the two panels into one plot, as each panel of the paper's Figure 5 does, and to keep it clean given the reviewers' comments on cluttered figures and small, mixed text.
  - Same curves as the figure above (`gamma_ctrl.load()` and `line_pts()`), plus MultiModN + ARC.
  - Layout:
    - The whole range on a linear latency axis.
    - The 296–540 ms × .655–.776 cluster magnified in an inset, joined to a dotted box by one connector. The connector runs from the box's lower-right corner to the inset's lower-left corner; every other pairing crosses the GMS column or the Joint-AFA line.
    - The inset has no tick numbers, like the revised Figure 5.
    - The legend is frameless method names under the plot, in the style of the revised Figure 9, in two columns (three rows).
  - Removed:
    - titles, the explanatory paragraph and the per-point γ / k labels;
    - at the user's request, the Table 2 markers (black-edged, SeMARC's star) and SeMARC's weighted-cost diamond, with their legend entries;
    - "Table 2" from the MultiModN + ARC entry, and the knob notes in brackets from the legend.
  - Consequences of removing the Table 2 markers:
    - The paper's Joint-AFA run (.749 at 920 ms), the one Table 2 point off its curve, is no longer shown.
    - The other Table 2 points remain as ordinary dots on their curves: SeMARC γ = 0, EAMA γ = 0, GMS 0%.
    - GMS's −25% and 0% points (.733 at 768 and 772 ms) now show as two overlapping triangles.
  - The knobs (γ for SeMARC and EAMA, r_cost for Joint-AFA, Table 2's k scaled for GMS) belong in the caption.
  - Text and size: the user asked for text as large as in the revised Figure 9 (7 pt in a 125 pt-wide drawing, 5.6% of its width).
    - The figure is drawn at 246 × 214 pt with all text at 11.5 pt, 4.7% of the width. That is as large as a two-column legend allows.
    - Ticks are thinned to 400 / 800 / 1200 ms and 0.5 / 0.6 / 0.7.
    - Lines are 2.9 pt (SeMARC) and 2.2 pt. Markers are 6.4 pt in the inset and 5.2 pt in the main view, where the cluster is small.
  - Printed text size = 11.5 pt × (width in LaTeX / 246 pt): about 7.4 pt at 0.4\textwidth (Figures 8–10 print at 7 pt), 9.3 pt at 0.5\textwidth, and 11.5 pt at 0.62\textwidth.
  - The plotted numbers: `res/pareto_gamma_controllers_single.csv` (`pareto_single_csv.py`).
    - 20 rows, one per point: SeMARC 4, EAMA 5, Joint-AFA 4, GMS 6, MultiModN + ARC 1.
    - Columns: method, knob, knob value, six-dataset mean Macro-F1 and latency, whether the point is also in the inset, and whether it is Table 2's setting.
    - Before writing, the script reads the marker positions back from the PDF and checks every row against them (largest difference 1e-6 ms). Every curve marker in the PDF belongs to a row.
    - Joint-AFA's Table 2 setting (r_cost .05) is not on the plotted front: its retrained point (.740 at 756 ms) is beaten by r_cost .1 (.743 at 517 ms).
- **SeMARC.** γ ∈ {0, .25, .5, .75, 1, 1.5, 2} at Table 2's λ (.5; MM-Fi fold 2: 1): the paper's γ-study nets, as in the controller figure.
- **SeMA + EAMA.** λ fixed at Table 2's value: each run's (fold × value-model seed) λ picked on validation at uniform cost.
  - SeMARC prices modality j at λ w_j / Σ_m w_m with w_m = (t_m / min t)^γ. That is λ / M at γ = 0, so γ multiplies modality j's price by M w_j / Σ_m w_m.
  - EAMA's uniform cost 1/M multiplied by the same factor is z_j(γ) = t_j^γ / Σ_m t_m^γ.
  - γ = 0 is Table 2's EAMA row. γ = 1 uses the weighted row's costs t_j / Σ t, but at the uniform row's λ.
- **EAMA's value models were retrained.** Costs act at test time only, but the models were not stored.
  - `ctrl/eama_gamma.py` retrains them with `baseline_eama_raw.py`'s own functions and seeds on the saved raw previews: 54 runs on GPUs 2/3, ~1–4 min each.
  - Check: every run reproduces the paper's: the holdout AUC, every stored test row (both costs × nine λ: F1, MU, extraction ms, stages) and both selected λ, to ≤ 8.9e-16.
  - At γ = 0, EAMA lands exactly on its Table 2 point (.7436 at 463.51 ms) on every dataset.
- **SeMA + GMS.** Every dataset's Table 2 k (IEMOCAP 5, MM-Fi 1, CZU-MHAD 4, EAV 1, CMI 5, UTD-MHAD 2) is scaled by the same percentage, −75% to +400%, rounded half up and clipped to 1 … M. 0% is its Table 2 point, so the curve passes through the marker. No refit: the stored chains have test F1 at every k.
  - Mean: −75% .649 at 753 ms (k = 1 everywhere); −50% .689 at 765; −25% .733 at 768; 0% .733 at 772 (Table 2); +25% .745 at 1099; +50% .779 at 1328. +100% to +400% are off the front.
  - SeMARC's γ curve beats every GMS point up to +25%. From +50% on, GMS gets more F1 than SeMARC's best (.767), at 2.6× its latency or more: MM-Fi and EAV then add a second modality, and MM-Fi's raises its F1 from .759 to .959.
  - The shared-k curve of the earlier figures (one k for all six datasets) could not pass through the marker: Table 2's GMS uses a different k per dataset.
- **Other variants** as in `fig/pareto_lambda_controllers_mean`: Joint-AFA over r_cost with the paper's run as its marker, and MultiModN + ARC.
- **Results, six-dataset mean.**
  - At the same γ, SeMARC is more accurate at every γ, by .019–.091 F1. EAMA is faster for γ ≤ 1, by 5–56 ms. At γ = 1.5 and 2, SeMARC is better on both.
  - No controller point beats any point of SeMARC's γ curve.
  - SeMARC's curve beats EAMA's γ = .5 (.734 at 388 ms) and .25 (.745 at 416) points and its Table 2 point (.744 at 464).
  - EAMA's only unbeaten points are γ = .75 and 1 (.671 at 312 and 342 ms): up to 43 ms faster than SeMARC's fastest point (.762 at 355 ms), but .09 F1 lower.
  - EAMA's drop at γ ≥ .75 is CZU-MHAD. Depth is 98% of its extraction time, so depth's cost approaches EAMA's whole budget and EAMA drops it on about half the samples: 1296 → 588 ms, F1 .899 → .461. SeMARC keeps depth at every γ (F1 .914–.922).
  - With λ fixed at Table 2's mostly small values, γ moves EAMA much less than SeMARC elsewhere. On MM-Fi, EAMA gets slower as γ grows (282 → 1025 ms): the cheap streams become almost free, and some runs then also take depth.
- **Results, per dataset, SeMARC against EAMA at the same γ.**
  - EAMA is never better on both axes at any γ on any dataset.
  - SeMARC is better on both at some γ on IEMOCAP (γ = .5, .75), MM-Fi (0–.75) and EAV (five of seven γ).
  - On CZU-MHAD, CMI and UTD-MHAD the two split at every γ: SeMARC more accurate, EAMA faster.
- **Caveats.**
  - Each curve is a Pareto front over test-evaluated γ (as every curve here). The same-γ comparison needs no selection.
  - EAMA's λ was picked for uniform costs. Re-picking it on validation for each γ is a different experiment, not run.

## Supplement: γ × λ sweep (`plot_pareto.py`)

## Design

- **Metric.** Latency is Table 2's estimated GPU latency (Appendix `app:comp-replace-complete`, the formula of `RL_AAAI_branch/gamma_ablation_ds.py`): idle-L40 extraction of every acquired modality, plus calibrated branch-encoder, head and selector times. F1 is macro-F1, mean over the three test folds. The comparison points carry their published values from `tab:comp-replace-complete`; the mean panel averages them over the six datasets, as Table 2 does.
- **Grid.**
  - γ ∈ {0, .25, .5, .75, 1, 1.5, 2}, which sets the modality weights w_m = (t_m / min t)^γ as in Figure 10.
  - λ ∈ {.05, .1, .2, .5, 1, 2}, the same grid as Appendix `tab:wgamma-lambda`.
  - The deployed λ (`dep`) is also included. It equals .5 everywhere except MM-Fi fold 2 (1.0), so `γ = 0, λ = dep` is Table 2's SeMARC and `γ = .75, λ = dep` is the paper's weighted SeMARC.
- **Held at the deployed settings, fold by fold** (`deployed_provenance.json`): ARC seed, FQI schedule, availability augmentation, and the paper's rule (prior priced at the training λ, stop margin 0).
- **Nets.** `RL_AAAI_branch/checkpoints_t2p/<ds>/` only; no shipped checkpoint folder is written. `seed_runset.py` copies the paper's own nets there: the deployed uniform nets, the γ-study nets at the deployed λ, and IEMOCAP's whole γ×λ grid (231 files). Every other net is a new fit by the unchanged `train_or_load`, on GPU 2 or 3, from the same val cache.
- **Pareto front.** The configurations that no other configuration beats on both F1 and latency. It is an envelope over the 42 (γ, λ) configurations of each dataset (49 on MM-Fi and the mean, which also keep the deployed-λ ones), each evaluated on the test folds; no configuration was selected on validation. The comparison methods were not swept: each is one published configuration.

## Verification (before any new fit)

`verify.py`: with the copied nets, `sweep.py` reproduces the paper's per-fold results exactly (max |ΔF1| = |ΔMU| = |Δlatency| = 0):
- all 126 fold rows of `results_baselines/gamma_ablation_samelam_perfold.csv` (6 datasets × 7 γ × 3 folds at the deployed λ; Appendix `tab:wgamma-alldata`, γ = 0 = Table 2's SeMARC);
- all 126 IEMOCAP fold rows of `results_baselines/gammaabl_grid_v2/` (Appendix `tab:wgamma-lambda`).

## Results (finished 2026-11-09 12:45)

- **Sweep.**
  - 630 new fold rows: 5 datasets × 42 configurations × 3 folds. IEMOCAP's grid comes from the verified copies.
  - 524 nets fitted on GPUs 2/3. Median fit time per fold: CMI 223 s, CZU-MHAD 69 s, the others about 3 s.
  - 106 nets loaded: the copies, plus one CMI net fitted before an out-of-memory retry of its walk.
  - No job failed for good; out-of-memory kills were requeued.
- **Mean of six datasets (Table 2's view).** SeMARC's front dominates every Table 2 configuration.
  - SeMA + EAMA (.744 at 463 ms) against SeMARC .771 at 408 ms (γ = 1, λ = .2), or .759 at 348 ms (γ = 1.5, λ = .5).
  - Weighted EAMA (.761 at 425 ms) against SeMARC .762 at 355 ms, the paper's γ = .75 row.
  - With uniform cost alone, the best SeMARC F1 at ≤ 463 ms is .724 (λ = 1). Raising λ alone does not beat EAMA's latency at EAMA's F1; the modality weights do.
- **Per dataset: does some SeMARC configuration have F1 ≥ and latency ≤ (one strictly)?**
  - MultiModN, MultiModN + ARC, SeMA + Joint-AFA: yes on all six datasets.
  - SeMA + EAMA: yes on five. Not on CMI: .523 at 6.8 ms, while SeMARC needs 8.8 ms for .525.
  - Weighted EAMA: yes on four. Not on CMI (SeMARC needs 9.4 ms for .535) or CZU-MHAD (.919 at 1305 ms against .917 at ≤ 1305 ms).
  - SeMA + GMS: yes only on MM-Fi.
    - IEMOCAP: .676 at 96 ms; SeMARC has .672 at ≤ 96 ms and needs 483 ms for .678.
    - EAV: .688 at 26.7 ms against SeMARC .688 at 27.1 ms.
    - CMI: .567 at 10.9 ms against 13.6 ms.
    - UTD-MHAD: .793 at 516 ms against 566 ms.
    - CZU-MHAD: a tie at about 1306 ms.
- **Caveat.** The front is a test-set envelope over many configurations, while each comparison method is one point. "Dominated" therefore means "some SeMARC configuration beats it on the test folds", not that a validation-selected SeMARC does.

## Files

| File | Role |
|---|---|
| `sweep.py` | One dataset and one γ: fit or load ARC at every λ and fold, walk the test set, price latency |
| `run_one.sh` | Per-dataset environment (deployed backbone, cache, provenance key, folds) and cost environment for one `sweep.py` process |
| `seed_runset.py` | Copies the paper's nets into `checkpoints_t2p` |
| `verify.py` | Reproduction check against the paper's per-fold files |
| `worker.sh`, `run_sweep.sh`, `chain_czu3.sh` | Queue workers on GPUs 2/3: wait for free memory, requeue CUDA OOM. CMI and CZU-MHAD jobs need about 8.4 and 2 GB, so CZU-MHAD ran after CMI (`chain_czu2.sh` is superseded) |
| `plot_gamma.py` | Main view (γ varied, λ as in Table 2): figures `fig/pareto_gamma_mean.{pdf,png}` (mean only) and `fig/pareto_gamma_f1_latency.{pdf,png}` (all panels), `res/PARETO_gamma.md`, `res/gamma_points.csv`, `res/dominance_gamma.csv`. Reads only the verified `res/verify_*.csv` rows at the deployed λ |
| `plot_pareto.py` | Supplement (γ × λ): figure `fig/pareto_f1_latency.{pdf,png}` (zoom insets on CZU-MHAD, UTD-MHAD and the mean), `res/PARETO.md` (fronts and the dominance table), `res/pareto_points.csv`, `res/dominance.csv` |
| `res/verify_*.csv`, `res/sweep_l<λ>_<ds>_g<γ>.csv` | Per-fold rows: F1, MU, GFLOPs, extraction/fusion/policy/total latency, acquisition frequencies, net file, fitted or loaded, fit seconds |
| `res/sn/` | Per-sample stop nodes of every walk (for re-pricing on another platform) |
| `ctrl_curves.py` | Controller curves: EAMA (stored λ grid), Joint-AFA (`ctrl/jafa/`), GMS (stored chains) and SeMARC γ, priced with Table 2's estimate; figure `fig/pareto_controllers_mean.{pdf,png}`, `res/ctrl_points.csv`, `res/CTRL_CURVES.md` |
| `ctrl/jafa_sweep.py` | Joint-AFA at several r_cost with the paper's `train`/`evaluate` unchanged. `MODE=previews` saves the raw previews (needs a GPU: the data loader puts batches on `cuda:0`); `MODE=train` trains and evaluates every r_cost |
| `ctrl/envrun.sh` | Runs a script in a dataset's deployed environment (backbone, prefix cache, provenance key, folds, idle table) on GPU 2, GPU 3 or CPU |
| `ctrl/drive_jafa.sh` | The sweep as run: previews on GPU 2 one dataset at a time, then 36 CPU processes (2 per fold) |
| `ctrl/run_jafa.sh` | GPU determinism test of the paper's Joint-AFA run (IEMOCAP fold 0), stopped at 6000 episodes |
| `ctrl/check_cache.py`, `ctrl/readout_gap.py` | Cache check against the paper's stored Joint-AFA / GMS sets; full-set F1 of the order-averaged vs single-order readout |
| `ctrl/jafa/*.json`, `ctrl/jafa/nets/*.pt` | Joint-AFA per (dataset, fold, r_cost): test and validation F1, MU, per-modality frequency, per-sample final sets; the trained nets |
| `ctrl/previews/`, `ctrl/logs/` | Raw previews per fold (val and test); preview, training and test logs |
| `lam_curves.py` | SeMARC over λ at uniform cost with the controllers' curves (imports `ctrl_curves.py`); figure `fig/pareto_lambda_controllers_mean.{pdf,png}`, `res/lam_points.csv`, `res/LAM_CURVES.md` |
| `lamu_launch.sh` | The 144 extra uniform-cost fits (λ = .3, .4, .6, .7, .8, .9, 1.25, 1.5): `worker.sh` queues on GPUs 2/3 running `run_one.sh` / `sweep.py` unchanged → `res/lamu_l<λ>_<ds>_g0.csv`; logs `logs/lamu_*`, `logs/w_lamu_*` |
| `gamma_ctrl.py` | SeMARC and EAMA over the same γ at Table 2's λ, other controllers as in `lam_curves.py`; figure `fig/pareto_gamma_controllers_mean.{pdf,png}`, `res/gamma_ctrl_points.csv`, `res/GAMMA_CTRL.md` |
| `pareto_single.py` | the same figure as one plot with an inset (Figure 5 style): `fig/pareto_gamma_controllers_single.{pdf,png}` |
| `pareto_single_csv.py` | the numbers plotted in that figure, checked against the PDF's marker positions: `res/pareto_gamma_controllers_single.csv` |
| `ctrl/eama_gamma.py`, `ctrl/drive_eama_gamma.sh` | EAMA's value models retrained with the paper's functions and seeds, checked against the paper's run, then run at Table 2's λ with z_j = t_j^γ / Σ t^γ; one process per dataset on GPUs 2/3 (IEMOCAP fold 0 run first as the test) |
| `ctrl/eama_gamma/*.json`, `ctrl/logs/eama_gamma_*.log` | Per (dataset, fold): per seed the reproduction check, Table 2's λ, and per γ the costs, test and validation F1 / MU / extraction ms / stages / per-modality frequency |

## GitHub copy

A copy of this folder is in the GitHub repo `MrYangHaodong/EfficientMM`, branch `main`, at `analysis/table2_pareto/`. It
holds the scripts, this README, the result tables (`res/`), the figures (`fig/`) and the EAMA / Joint-AFA run summaries
(`ctrl/eama_gamma/*.json`, `ctrl/jafa/*.json`).

The scripts do not run inside that repository. They use the server's experiment code and caches under
`/files1/haodong/test/new/` (`RL_AAAI_branch/`, `energy_adaptive/`), not the repository's `arc/` package.

Left on the server only (about 300 MB):
- the trained Joint-AFA nets (`ctrl/jafa/nets/`);
- the raw previews (`ctrl/previews/`);
- the per-sample stop nodes (`res/sn/`);
- the logs.
