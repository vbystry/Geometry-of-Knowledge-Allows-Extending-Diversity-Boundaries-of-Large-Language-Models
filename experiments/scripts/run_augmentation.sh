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
MODE="interp"
MAX_ANCHORS=""
ANCHOR_NOISE=""

# ── parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)         INPUT="$2";         shift 2 ;;
    --output)        OUTPUT="$2";        shift 2 ;;
    --seed-ratio)    SEED_RATIO="$2";    shift 2 ;;
    --lambda)        LAMBDA_VALUE="$2";  shift 2 ;;
    --target-n)      TARGET_N="$2";      shift 2 ;;
    --seed)          SEED="$2";          shift 2 ;;
    --sigma)         SIGMA="$2";         shift 2 ;;
    --no-style)      STYLE="";           shift ;;
    --gpu)           GPU="$2";           shift 2 ;;
    --mode)          MODE="$2";          shift 2 ;;   # ablation: interp|single|mean|medoid|gauss
    --max-anchors)   MAX_ANCHORS="$2";   shift 2 ;;   # weak-seed: cap anchor count
    --anchor-noise)  ANCHOR_NOISE="$2";  shift 2 ;;   # weak-seed: noise stddev on anchors
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

echo "Augmentation: mode=${MODE}  lambda=${LAMBDA_VALUE}  seed-ratio=${SEED_RATIO}  target-n=${TARGET_N}"
echo "  input:  $INPUT"
echo "  output: $OUTPUT"
[[ -n "$MAX_ANCHORS"   ]] && echo "  max-anchors:   $MAX_ANCHORS"
[[ -n "$ANCHOR_NOISE"  ]] && echo "  anchor-noise:  $ANCHOR_NOISE"

EXTRA_ARGS=()
[[ -n "$MAX_ANCHORS"  ]] && EXTRA_ARGS+=(--max-anchors "$MAX_ANCHORS")
[[ -n "$ANCHOR_NOISE" ]] && EXTRA_ARGS+=(--anchor-noise "$ANCHOR_NOISE")

uv run python experiments/augment_responses.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --seed-ratio "$SEED_RATIO" \
    --seed "$SEED" \
    --sigma "$SIGMA" \
    --lambda-value "$LAMBDA_VALUE" \
    --sampling-mode "$MODE" \
    --target-n "$TARGET_N" \
    "${EXTRA_ARGS[@]}" \
    $STYLE

echo "Done."
