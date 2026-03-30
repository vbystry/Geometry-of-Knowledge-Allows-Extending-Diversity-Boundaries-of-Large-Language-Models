# Reproducibility Guide: Curated NoveltyBench Diversity Experiments

This document provides a complete and self-contained guide for reproducing
the **diversity experiments reported in the paper** on the **curated split of
NoveltyBench**. The repository extends the official NoveltyBench codebase with
additional generation and augmentation scripts; **all NoveltyBench evaluation
code and metrics are used without modification**.

---

## Experimental Overview

- **Benchmark**: NoveltyBench (curated split)
- **Generation budgets**: \( k \in \{10, 15, 20, 25, 30\} \)
- **Anchors**:
  - G2-generated responses (primary setting)
  - In-context generations (baseline)
- **Augmentation method**:
  - `augument_responses.py` implementing continous semantinc conditioned diverse generation
  - Optional format realignment to mitigate structural drift
- **Evaluation metrics**:
  - **Distinct**: number of classifier-derived equivalence classes
  - **Utility**: discounted cumulative utility with patience \( p = 0.8 \)


---

## Step 0: Preparing Anchor Generations

### G2 Anchors

This repository **does not generate G2 outputs**. G2 generations must already
be available at:

```
results/curated/g2_theta0.3_temp1_iter{k}/generations.jsonl
```

Each line must be a JSON object of the form:

```json
{
  "id": "...",
  "prompt": "...",
  "model": "...",
  "generations": ["...", "...", "..."]
}
```

### In-Context Anchors (Baseline)

To generate in-context baselines using the NoveltyBench inference pipeline:

```bash
uv run python src/inference.py   --mode transformers   --model "mistralai/Mistral-7B-Instruct-v0.2"   --data curated   --eval-dir "results/curated/mistralai-Mistral-7B-Instruct-v0.2_in-context_${k}gen"   --sampling in-context   --num-generations "$k"
```

---

## Step 1: Augmentation via Continous Semantic Conditioning

### Parameterization

- **Seed ratio**:
  - \( k \in \{10, 15, 20\} \): `--seed-ratio 0.3`
  - \( k \in \{25, 30\} \): `--seed-ratio 0.2`
- **Latent exploration strength**:
  - If `--lambda-value` is omitted, λ is sampled as:
    \[
    \lambda \sim \mathcal{U}([6,10] \cup [-10,-6])
    \]
- **Format realignment**:
  - Enabled via `--use-style-normalization`

### Running Augmentation

```bash
for k in 10 15 20 25 30; do
  if [ "$k" -le 20 ]; then
    seed_ratio="0.3"; s_label="s3"
  else
    seed_ratio="0.2"; s_label="s2"
  fi

  in="results/curated/g2_theta0.3_temp1_iter${k}/generations.jsonl"
  out_dir="results/curated/g2_theta0.3_temp1_iter${k}_${s_label}_t${k}"
  out="${out_dir}/generations.jsonl"

  mkdir -p "$out_dir"

  uv run python augument_responses.py     --input "$in"     --output "$out"     --target-n "$k"     --seed-ratio "$seed_ratio"     --seed 42     --sigma 0.05     --use-style-normalization
done
```

---

## Step 2: NoveltyBench Evaluation

For each augmented directory:

```bash
eval_dir="results/curated/g2_theta0.3_temp1_iter15_s3_t15"

uv run python src/partition.py --eval-dir "$eval_dir" --alg classifier
uv run python src/score.py --eval-dir "$eval_dir" --patience 0.8
uv run python src/summarize.py --eval-dir "$eval_dir"
```

This produces:
- `partitions.jsonl`
- `scores.jsonl`
- `summary.json`

To obtain mean generation and partition scores:

```bash
uv run python src/score_mean.py --eval-dir "$eval_dir" --patience 0.8
uv run python src/summarize_means.py --eval-dir "$eval_dir"
```

---

## Lambda Ablation Study (k = 15)

### Generation

```bash
bash run_iter15_lambda_experiments.sh
```

### Evaluation

```bash
bash run_iter15_lambda_evaluation.sh
```

Each run corresponds to a fixed λ value and is evaluated using the same
NoveltyBench pipeline as above.

---

## Anchor-Seed Ablation (G2 vs In-Context)

1. Generate in-context anchors for each \( k \).
2. Apply augmentation with identical seed ratios and λ settings.
3. Evaluate using the standard Distinct and Utility metrics.

---

## AUT Augmentation

AUT experiments are **not part of NoveltyBench** and do not affect any benchmark
results reported above.

Example invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run augment_aut_responses.py   --input "/path/to/AUT_*.json"   --output "augmented_aut_output.json"   --target-n 500   --lambda-value 5   --csv-output "augmented_aut_output.csv"   --seed 42
```

---

## G2 AUT Experiments (Separate Artifact)

AUT experiments require task-specific extensions to the G2 implementation.
To preserve benchmark integrity, these modifications are released separately:

https://github.com/vbystry/emnlp25-g2

This repository contains the exact AUT configurations used in the paper,
including both from-scratch and seeded settings.

---

## Final AUT Scoring (External Evaluator)

Final originality scoring for AUT is performed using the **OpenScoring** web
application:

https://openscoring.streamlit.app

### Procedure

1. Upload the generated AUT CSV (e.g., `augmented_aut_output.csv`).
2. If necessary, remove newline characters from text fields
   (a helper script `remove_newlines.py` is provided).
3. Select:
   - **Model**: `ocsai-4o`
   - **Column mapping**:
     - `item`
     - `idea`
4. Run scoring and download the resulting CSV.

The scored CSV constitutes the final scoring used for AUT analysis in the paper.
