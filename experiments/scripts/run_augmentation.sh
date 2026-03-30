#!/usr/bin/env bash
# Run latent-space augmentation on NoveltyBench generations.
#
# Usage:
#   bash experiments/scripts/run_augmentation.sh \
#       --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
#       --output results/curated/my_run/generations.jsonl \
#       --seed-ratio 0.3 --lambda 8 --target-n 15
#
# All flags have defaults matching the paper's main experiment (Table 1).
# Add --no-style to skip the realignment step.

set -euo pipefail

# ── defaults (paper configuration) ──────────────────────────────────────
INPUT=""
OUTPUT=""
SEED_RATIO=0.3
LAMBDA_VALUE="6-10"
TARGET_N=15
SEED=42
SIGMA=0.05
STYLE="--use-style-normalization"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# ── parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)       INPUT="$2";        shift 2 ;;
    --output)      OUTPUT="$2";       shift 2 ;;
    --seed-ratio)  SEED_RATIO="$2";   shift 2 ;;
    --lambda)      LAMBDA_VALUE="$2"; shift 2 ;;
    --target-n)    TARGET_N="$2";     shift 2 ;;
    --seed)        SEED="$2";         shift 2 ;;
    --sigma)       SIGMA="$2";        shift 2 ;;
    --no-style)    STYLE="";          shift ;;
    --gpu)         GPU="$2";          shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "Error: --input and --output are required."
  exit 1
fi

# ── environment ─────────────────────────────────────────────────────────
export TMPDIR="${TMPDIR:-/tmp}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

# ── run ─────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUTPUT")"

echo "Augmentation: lambda=${LAMBDA_VALUE}  seed-ratio=${SEED_RATIO}  target-n=${TARGET_N}"
echo "  input:  $INPUT"
echo "  output: $OUTPUT"

uv run python experiments/augment_responses.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --seed-ratio "$SEED_RATIO" \
    --seed "$SEED" \
    --sigma "$SIGMA" \
    --lambda-value "$LAMBDA_VALUE" \
    --target-n "$TARGET_N" \
    $STYLE

echo "Done."
