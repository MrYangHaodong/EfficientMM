# EfficientMM — SeMARC: sequential multimodal fusion with learned acquisition

Code for the ICLR 2027 submission *"SeMARC: ..."* (title and link to be added).

SeMARC decides, per sample, which input modalities to acquire and when to stop:

- **SeMA** (backbone) fuses modalities one at a time through a shared bottleneck and gives a
  prediction after every prefix of any acquisition order.
- **ARC** (controller) is a residual fitted-Q policy over *acquire modality m* and *STOP*,
  fitted on SeMA's cached prefix predictions.

This repository contains the complete IEMOCAP pipeline: preprocessing of the raw release,
SeMA training, prefix-cache generation, ARC training, test-set evaluation (including
test-time missing modalities) and online per-sample inference. IEMOCAP has six modalities
(video, audio, text, three motion-capture streams) and four emotion classes.

## Layout

```
preprocess/                raw IEMOCAP -> utterance clips, video/audio/text features, labels
data/IEMOCAP/get_data.py   IEMOCAP loader;  make_splits.py: cross-subject split CSVs
common/data.py             loader -> per-modality batches {modality: [B, T, D]}
models/sema.py             SeMA
training/train_sema.py     step 1: train SeMA                    -> checkpoints/sema/
arc/prefix_cache.py        prefix-tree cache of a trained SeMA
arc/build_cache.py         step 2: generate the cache            -> cache/
arc/core.py, arc/prior.py  ARC's MDP (state, actions, reward, cost) and value prior Q_0
arc/policy.py              residual Q-network, fitted-Q training, deployment rollout
arc/train_arc.py           step 3: train ARC                     -> checkpoints/arc/
inference/evaluate.py      step 4: test-set evaluation from the caches  -> results/
inference/online.py        per-sample online inference (SeMA + ARC loop)
configs/sema.json          SeMA recipe;  arc.json: ARC settings + per-fold (seed, lambda)
configs/costs.json         per-modality feature-extractor GFLOPs and timings (reporting only)
scripts/                   0_preprocess.sh ... 4_evaluate.sh, run_all.sh
```

## Setup

```bash
conda env create -f environment.yml && conda activate semarc
export IEMOCAP_ROOT=/path/to/IEMOCAP_full_release    # after step 0
```

Run all commands from the repository root. `IEMOCAP_ROOT` must contain, per utterance,
`video_features/<dialog>/<utt>.pkl` (100 × 1024, DINOv3 ViT-L/16), `audio_features/...`
(1 × 599 × 1280, wav2vec2 XLS-R 1B), `text_features/...` (T × 768, bert-base-uncased) and the
release's `MOCAP_hand/`, `MOCAP_head/`, `MOCAP_rotated/` text files, either directly under
`IEMOCAP_ROOT` or under `Session<k>/sentences/` as in the release. It also needs the split
files `{train,val,test}_cs_split{0,1,2}.csv` (session-disjoint folds: test S5 / S4 / S3,
validation S4 / S1 / S2). `data/IEMOCAP/make_splits.py` writes them with exactly the
utterances of the paper (4,457 with all six modalities and a neutral / angry / happy / sad
label); their row order sets the training batch order, so bit-exact reproduction needs the
paper's own split files.

## Pipeline

```bash
bash scripts/0_preprocess.sh $IEMOCAP_ROOT /path/to/dinov3 /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth 0
bash scripts/run_all.sh 0        # steps 1-4 for folds 0, 1, 2 on GPU 0
```

**0. Preprocess** (`preprocess/`): `segment_video.py` cuts each dialog video into per-utterance
clips cropped to the speaker's half of the frame; `extract_video.py` (DINOv3 ViT-L/16, 100
frames), `extract_audio.py` (XLS-R 1B, 12 s) and `extract_text.py` (BERT-base tokens) write the
features with the utterance labels parsed from `dialog/EmoEvaluation/` and
`dialog/transcriptions/`. DINOv3 code: github.com/facebookresearch/dinov3 (weights from Meta).

**1. Train SeMA** — `python training/train_sema.py --fold 0`. Recipe in `configs/sema.json`:
3 encoder + 3 fusion layers per modality, d = 128, K = 16 bottleneck tokens, 40 epochs, AdamW
1e-4 with cosine decay, class-weighted cross-entropy, random modality order, batch-level
modality dropout ramped to 0.4, and the loss

    L = CE(p_S, y) + 1.0 · Σ_s a_s CE(p_s, y) + 0.5 · Σ_s a_s KL(p̄ ‖ p_s),   a_s = s / Σ_r r

where p_s is the prediction after the first s modalities and p̄ is the prediction of an EMA
copy (decay 0.999) on the same modalities in canonical order. Best validation macro-F1 is
kept: `checkpoints/sema/best_model_fold<N>.pth` + `results_fold<N>.json`.

**2. Build the cache** — `python arc/build_cache.py`. Runs SeMA on all 1,957 ordered modality
prefixes of every validation and test utterance and stores the posterior and GFLOPs of each
prefix: `cache/iemocap_fold<N>_{val,test}.npz` (a few minutes per fold).

**3. Train ARC** — `python arc/train_arc.py` (`--grid` for every seed × λ). Q = Q_0(λ) + R_θ:
Q_0(ψ, STOP) is the current confidence; Q_0(ψ, m) adds the expected correctness gain of
modality m given the acquired set (per-class tables estimated on validation, read through the
posterior) minus λ/|A|; R_θ is a 64-unit MLP initialised at zero. Fitted-Q iteration (8 × 40
epochs) on 80% of the validation cache with each modality available w.p. 0.7; the iteration
with the best macro-F1 on the other 20% is kept. Output: `checkpoints/arc/arc_fold<N>_seed<S>_lam<λ>.pt`.

**4. Evaluate** — `python inference/evaluate.py [--avail 1.0 0.9 0.8 0.7 0.6]`. Deployment
rule: Q_0 priced at the policy's own training λ, greedy acquisition, STOP as soon as
Q(STOP) ≥ max_m Q(m) (no stop margin). Reports per-fold and three-fold macro-F1, modality
usage, fusion and end-to-end GFLOPs and estimated extraction time; `--avail p` drops each
test modality independently with probability 1 − p.

**Online inference** — `python inference/online.py --fold 0 [--time] [--compare_cache]` runs
the SeMA + ARC loop one sample at a time: ARC reads the current posterior, picks a modality,
SeMA encodes and fuses only that modality, and so on until STOP. Unacquired modalities are
never read. Inputs are the pre-extracted features; `--time` reports per-sample latency.

## Checks against the paper's artefacts

- The paper's SeMA checkpoints load strictly; `train_sema.py` matches the original trainer
  weight-for-weight; caches rebuilt here are bit-identical to the paper's.
- `train_arc.py` reproduces the original ARC training code bit-for-bit; retrained policies
  reach test macro-F1 .682 / .681 / .643 on folds 0 / 1 / 2 (paper: .682 / .682 / .643).
- `inference/evaluate.py` with the paper's policies reproduces every per-fold value of the
  paper's IEMOCAP results, at full availability and at p = 0.9 … 0.6: F1, usage, GFLOPs and
  the per-sample selections. Three-fold mean at full availability: F1 .669 ± .022, MU .65.
- `inference/online.py` agrees with the cache rollout on 100% of the test samples of every
  fold, both in the acquisition sequence and in the prediction.

## Checkpoints

Pre-trained SeMA and ARC checkpoints for the three folds: (link to be added).

## Citation

(to be added)

## License

MIT (see `LICENSE`).
