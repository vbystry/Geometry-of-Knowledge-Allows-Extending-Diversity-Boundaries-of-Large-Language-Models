# Geometry of Knowledge Allows Extending Diversity Boundaries of Large Language Models

## Overview

We introduce a framework that increases LLM generative diversity by conditioning on continuous latent variables in semantic embedding space. A small set of anchor generations is embedded, interpolated on the resulting manifold, and projected into the LLM's token-embedding space via an [xRAG](https://github.com/Hannibal046/xRAG) projector -- requiring no modification of LLM parameters.

## Repository structure

```
vendor/                       # Git submodules (external dependencies)
  novelty-bench/              #   NoveltyBench benchmark
  g2/                         #   G2 guided generation (baseline)
  llm-discussion/             #   LLM Discussion multi-agent framework (baseline)
  xRAG/                       #   xRAG multimodal projector
# Note: vendor code is modified under experiments/ to add Mistral-7B-Instruct-v0.2 support
# (see experiments/g2/, experiments/llm-discussion/, experiments/src/)

experiments/
  augment_responses.py        # Latent-space augmentation for NoveltyBench
  augment_aut_responses.py    # Latent-space augmentation for AUT
  plot_lambda_ablation.py     # Lambda ablation figure (Figure 2)
  remove_newlines.py          # CSV preprocessing for OCSAI scoring
  src/                        # Modified NoveltyBench evaluation pipeline (Mistral support)
  g2/                         # Modified G2 code (Mistral support)
  llm-discussion/             # Modified LLM Discussion code (Mistral support)
  scripts/
    run_augmentation.sh       # NoveltyBench augmentation (configurable)
    run_evaluation.sh         # NoveltyBench evaluation pipeline (configurable)
    run_aut.sh                # AUT augmentation (configurable)

notebooks/
  plot_originality_by_gen.ipynb   # AUT originality curve (Figure 3)

```

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd geometry-of-knowledge

# Install dependencies
pip install uv
uv sync

# Set up API keys (for NoveltyBench evaluation with GPT-based classifier)
cp .env.example .env  # edit with your keys
```

**Models** (downloaded automatically on first run):
- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) -- base LLM
- [SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) -- semantic encoder
- xRAG projector checkpoint (see [xRAG repo](https://github.com/Hannibal046/xRAG) for download instructions)

## Reproducing experiments

All scripts are run from the repository root.

### NoveltyBench

**Augmentation** -- generate latent-conditioned responses:

```bash
# Paper configuration (Table 1, "Ours (G2 seeds)")
bash experiments/scripts/run_augmentation.sh \
    --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
    --output results/curated/my_run_k15/generations.jsonl \
    --seed-ratio 0.3 --lambda 6-10 --target-n 15

# Custom configuration
bash experiments/scripts/run_augmentation.sh \
    --input  <path/to/base/generations.jsonl> \
    --output <path/to/output/generations.jsonl> \
    --seed-ratio 0.3 --lambda 8 --target-n 20 --no-style --gpu 0,1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Input JSONL with base generations |
| `--output` | (required) | Output JSONL path |
| `--seed-ratio` | `0.3` | Fraction of input generations used as anchors |
| `--lambda` | `6-10` | Interpolation coefficient (scalar or range) |
| `--target-n` | `15` | Total number of output generations per prompt |
| `--seed` | `42` | Random seed |
| `--no-style` | off | Skip realignment step |
| `--gpu` | `0` | CUDA visible devices |
| `--mode` | `interp` | Sampling strategy: `interp` (proposed), `single`, `mean`, `medoid`, `gauss` (ablation) |
| `--sigma` | `0.05` | Stddev of Gaussian perturbations (used by `--mode gauss`) |
| `--max-anchors` | (none) | Cap on anchor count — weak-seed ablation (lower coverage) |
| `--anchor-noise` | `0` | Stddev of Gaussian noise added to anchor embeddings — weak-seed ablation (lower quality) |

**Evaluation** -- partition, score, summarize:

```bash
# Standard evaluation (Table 1)
bash experiments/scripts/run_evaluation.sh \
    --eval-dir results/curated/my_run_k15

# With mean scores for seed ablation (Table 3)
bash experiments/scripts/run_evaluation.sh \
    --eval-dir results/curated/my_run_k15 \
    --mean-scores --seed-ratio 0.3
```

### Alternative Uses Test

**Augmentation:**

```bash
bash experiments/scripts/run_aut.sh \
    --input  results/llm-discussion/AUT/Output/multi_agent/<discussion_output>.json \
    --output results/aut/my_aut_run.json \
    --target-n 500 --lambda 5
```

**Scoring:** submit the output CSV to [OCSAI](https://openscoring.du.edu/ocsai) (model: `ocsai-4o`, task: English Alternate Uses). Use `experiments/remove_newlines.py` to preprocess CSVs before upload.

### Sampling and weak-seed ablations (NoveltyBench)

These ablations isolate the contribution of the proposed continuous sampling
strategy. The projector, anchor source, `target_n`, and (per sub-grid)
realignment are held fixed across cells; only the sampling step changes.

Sub-grids:

- **(A) Sampling strategy, realignment ON.** Compare under matched anchors and
  matched realignment: single-point conditioning (`single` / `mean` / `medoid`),
  non-geometric Gaussian perturbations (`gauss`), and the proposed
  interpolation/extrapolation (`interp`).
- **(B) Sampling strategy, realignment OFF.** Same grid as (A) but with
  `--no-style`, so any Distinct/Utility gap is attributable to the sampling
  step alone (the styler is removed).
- **(C) Weak-seed robustness.** `interp` only, but the anchor set is degraded
  via `--max-anchors` (lower coverage) and/or `--anchor-noise` (lower quality)
  to probe behavior when the seed prior under-covers the solution space.

Run the full grid:

```bash
bash experiments/scripts/run_ablations.sh \
    --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
    --out-root results/ablations/ \
    --target-n 15 --seed-ratio 0.3 --lambda 6-10
```

Results land under `results/ablations/{A_sampling,B_sampling,C_weak_seed}/...`,
each cell evaluated with `run_evaluation.sh` (pass `--skip-eval` to defer).
A single cell can also be run directly through `run_augmentation.sh`, e.g.

```bash
# Single-point baseline (anchor centroid), realignment off
bash experiments/scripts/run_augmentation.sh \
    --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
    --output results/ablations/manual_mean_nostyle/generations.jsonl \
    --mode mean --no-style

# Weak-seed: interp with only 2 anchors and mild anchor noise
bash experiments/scripts/run_augmentation.sh \
    --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
    --output results/ablations/manual_weak_seed/generations.jsonl \
    --mode interp --max-anchors 2 --anchor-noise 0.10
```

### Lambda ablation (Figure 2)

```bash
for lam in 0.5 2 3 4 5 6 7 8 9 10; do
    bash experiments/scripts/run_augmentation.sh \
        --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
        --output results/curated/ablation_lam${lam}/generations.jsonl \
        --lambda $lam --target-n 15

    bash experiments/scripts/run_evaluation.sh \
        --eval-dir results/curated/ablation_lam${lam}
done

python experiments/plot_lambda_ablation.py
```

### Figures

- **Figure 2** (lambda ablation): `python experiments/plot_lambda_ablation.py`
- **Figure 3** (AUT originality curve): `notebooks/plot_originality_by_gen.ipynb`

