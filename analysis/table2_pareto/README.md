# Table 2: F1–latency Pareto curves

`train_settings.py` trains and evaluates the four controllers at every setting of their trade-off knob on IEMOCAP.
`pareto.py` turns per-fold test results into the Pareto curves of SeMARC and the Table 2 controllers. It writes the plotted
points to `pareto_points.csv` and draws `pareto_gamma_controllers.pdf` and `.png`.

```bash
python analysis/table2_pareto/train_settings.py                                      # retrain every IEMOCAP setting
python analysis/table2_pareto/pareto.py                                              # curves from the shipped data/
python analysis/table2_pareto/pareto.py --replace analysis/table2_pareto/results/fold_results_iemocap.csv
```

| File | Content |
|---|---|
| `train_settings.py` | Training and test evaluation of every setting on IEMOCAP. Writes `results/fold_results_iemocap.csv` in the format of `data/fold_results.csv`; `--check` compares it with the IEMOCAP rows of `data/` |
| `controllers.py` | GMS, EAMA and Joint-AFA on SeMA's prefix cache, plus the raw previews and the order-averaged set readout. SeMARC is `arc/` |
| `pareto.py` | Latency estimate, fold and dataset means, Pareto fronts, figure. `--replace` swaps in retrained rows per (dataset, method) |
| `data/fold_results.csv` | One row per (method, knob value, dataset, test fold, run): test macro-F1, and `freq_j`, the share of the fold's test samples whose final acquired set contains the dataset's j-th modality. EAMA rows also give its mean number of selection rounds (`stages`) |
| `data/timings.json` | Per dataset: modality order, raw-extraction and SeMA branch times per modality, readout and selector times, and two calibration factors |
| `pareto_points.csv` | Output: the plotted points (six-dataset means) |
| `pareto_gamma_controllers.pdf`, `.png` | Output: the figure |

## What each curve varies

Each curve varies one knob, shared by all six datasets, and shows that method's Pareto front over it:

| Curve | Knob (`knob_value` in `pareto_points.csv`) |
|---|---|
| SeMARC | modality cost weights γ ∈ {0, .25, .5, .75, 1, 1.5, 2}, at Table 2's λ (.5) |
| SeMA + EAMA | the same γ applied to its modality costs, at Table 2's λ; value models retrained, 3 seeds per fold |
| SeMA + Joint-AFA | acquisition cost r_cost from 0 to .5, retrained for each value |
| SeMA + GMS | every dataset's Table 2 budget k changed by the same percentage, −75% to +400% (rounded, clipped to 1…M) |
| MultiModN + ARC | none: Table 2's configuration (published per-dataset values, in `pareto.py`) |

- `table2_setting` marks the knob value Table 2 uses.
- EAMA and GMS keep that point on their curve even when another of their own points beats it, as EAMA's γ = .25 point beats its γ = 0.
- Joint-AFA's Table 2 setting (r_cost .05) is not on its front.

## How F1 and latency are computed

- **Means.**
  - Each fold result is a mean over the fold's test samples.
  - A dataset's value is the mean over its three test folds (folds 3–5 on UTD-MHAD), and over EAMA's three seeds.
  - Every curve point weights the six datasets equally (IEMOCAP, MM-Fi, CZU-MHAD, EAV, CMI, UTD-MHAD).
  - F1 is macro-F1 on the full test fold.
- **Latency is estimated, not timed end to end for each point** (`latency()` in `pareto.py`).
  - It counts raw extraction of every acquired modality, SeMA branch passes and readouts, and selector calls, each weighted by how often it runs (`freq_j`).
  - The component times were measured on one NVIDIA L40 on an otherwise idle server, at batch size 1, with no feature caching.
  - **SeMARC** fuses each acquisition as it goes. It pays a readout after every acquisition and a selector call at every step, STOP included.
  - **EAMA and Joint-AFA** select first and then run SeMA once over the final set. Their selector calls are priced at ARC's selector time, because their native selector runtimes were not timed.
  - **GMS** pays extraction and one SeMA pass only.
  - The two calibration factors scale the summed components to directly timed runs: the sequential walk (`cal_walk`) and a one-shot forward (`cal_fixed`).
- **End-to-end check.** At Table 2's operating point, SeMARC and MultiModN + ARC were also timed end to end on the same L40 (full test folds, batch size 1). Their six-dataset means are 515.67 and 1058.82 ms, against estimates of 520.17 and 1078.90 ms.

`data/` holds the per-run results of the full study on all six datasets. Those results come from:
- the γ-study ARC nets;
- the retrained EAMA value models and Joint-AFA agents;
- the stored GMS selection chains.

This repository has the IEMOCAP pipeline only, so `train_settings.py` regenerates the IEMOCAP rows.

## Training the settings (`train_settings.py`, IEMOCAP)

| Method | What is trained at each setting | Settings |
|---|---|---|
| SeMARC | ARC (`arc/policy.py`); acquiring modality m costs λ w_m / Σ_{j∈A} w_j with w_m = (t_m / min t)^γ (`arc.core.cost_weights`). λ and seed per fold as deployed (`configs/arc.json`). γ = 0 is the uniform cost | γ ∈ {0, .25, .5, .75, 1, 1.5, 2} |
| SeMA + EAMA | Value model on raw previews, seeds 0, 1, 2 per fold. λ chosen on the validation holdout at uniform cost (Table 2's setting: validation F1 − 0.2 × validation extraction fraction), then modality costs z_j = t_j^γ / Σ_m t_m^γ | γ as above |
| SeMA + Joint-AFA | Set-encoder dueling DQN on raw previews, seed 0, 20,000 episodes on CPU (2 threads each) | r_cost ∈ {0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5} |
| SeMA + GMS | Greedy mutual-information chain on validation, no training | budget k = 1 … 6 |

- t_m is the extraction time of modality m (`configs/costs.json`).
- **Inputs:**
  - the prefix caches (`arc/build_cache.py` → `cache/`);
  - for EAMA and Joint-AFA, the input features (`IEMOCAP_ROOT`). Their time-means are the raw previews these two selectors read.
- **Change to `arc/`:** `core.step_cost`, `prior.q_prior`, `policy.train` and `policy.rollout` take optional cost weights. Without them they are unchanged.
- **Runtime per fold on one L40:**
  - GMS: seconds;
  - SeMARC: about 15 s per fit (7 fits);
  - feature loading for the previews: about 40 s;
  - EAMA: about 45 s per value model, including scoring (3 models);
  - Joint-AFA: about 3–4 min per DQN (11 DQNs; `--jafa_workers` runs them in parallel).

**Checked against the paper's runs (IEMOCAP):**
- **GMS:** all 18 rows are identical.
- **Raw previews:** bit-identical. They are computed on CPU; a GPU mean differs in the last bits.
- **SeMARC, fold 0:**
  - γ = .25 … 2: all six rows identical (F1 and acquisition frequencies).
  - γ = 0: identical F1. The acquisition frequencies differ by up to .008, because the paper's γ = 0 row is the shipped uniform net, which a refit from today's caches does not reproduce bit for bit.
- **EAMA, fold 0:** all 21 rows identical (3 seeds × 7 γ).
- **Joint-AFA:** not run here. The code was checked line by line against the paper's implementation (network construction order, ε schedule, random-number sequence, replay buffer, targets, rollout), and it uses the same CPU thread count.
- Folds 1 and 2 were not run for SeMARC, EAMA and Joint-AFA.
