#!/usr/bin/env bash
# Evaluate NoveltyBench generations: partition → score → summarize.
#
# Usage:
#   bash experiments/scripts/run_evaluation.sh --eval-dir results/curated/my_run
#
# Optionally compute mean scores for the seed-ablation table (Table 3):
#   bash experiments/scripts/run_evaluation.sh --eval-dir results/curated/my_run --mean-scores --seed-ratio 0.3

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────
EVAL_DIR=""
PATIENCE=0.8
MEAN_SCORES=false
SEED_RATIO=""

# ── parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --eval-dir)    EVAL_DIR="$2";    shift 2 ;;
    --patience)    PATIENCE="$2";    shift 2 ;;
    --mean-scores) MEAN_SCORES=true; shift ;;
    --seed-ratio)  SEED_RATIO="$2";  shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$EVAL_DIR" ]]; then
  echo "Error: --eval-dir is required."
  exit 1
fi

# ── environment ─────────────────────────────────────────────────────────
export TMPDIR="${TMPDIR:-/tmp}"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

# ── pipeline ────────────────────────────────────────────────────────────
echo "Evaluating: $EVAL_DIR"

echo "  [1/3] Partitioning..."
uv run python experiments/src/partition.py \
    --eval-dir "$EVAL_DIR" \
    --alg classifier

echo "  [2/3] Scoring (patience=${PATIENCE})..."
uv run python experiments/src/score.py \
    --eval-dir "$EVAL_DIR" \
    --patience "$PATIENCE"

echo "  [3/3] Summarizing..."
uv run python experiments/src/summarize.py --eval-dir "$EVAL_DIR"

# ── optional: mean scores for ablation (Table 3) ───────────────────────
if $MEAN_SCORES; then
  echo "  [+] Computing mean scores..."
  if [[ -n "$SEED_RATIO" ]]; then
    uv run python experiments/src/score_mean.py \
        --eval-dir "$EVAL_DIR" \
        --patience "$PATIENCE" \
        --seed-ratio "$SEED_RATIO"
  fi
  uv run python experiments/src/mean_scores2.py --eval-dir "$EVAL_DIR"
  uv run python experiments/src/summarize_means.py --eval-dir "$EVAL_DIR"
fi

echo "Done: $EVAL_DIR"
