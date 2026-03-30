#!/usr/bin/env bash
# Run latent-space augmentation for AUT (Alternative Uses Test).
#
# Usage:
#   bash experiments/scripts/run_aut.sh \
#       --input  results/llm-discussion/AUT/Output/multi_agent/discussion_output.json \
#       --output results/aut/augmented_500_lam5.json \
#       --target-n 500 --lambda 5
#
# Scoring is done externally via https://openscoring.du.edu/ocsai

set -euo pipefail

# ── defaults (paper configuration, Table 2) ─────────────────────────────
INPUT=""
OUTPUT=""
TARGET_N=500
LAMBDA_VALUE=5
SEED=42
CSV_OUTPUT=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# ── parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)      INPUT="$2";        shift 2 ;;
    --output)     OUTPUT="$2";       shift 2 ;;
    --target-n)   TARGET_N="$2";     shift 2 ;;
    --lambda)     LAMBDA_VALUE="$2"; shift 2 ;;
    --seed)       SEED="$2";         shift 2 ;;
    --csv)        CSV_OUTPUT="$2";   shift 2 ;;
    --gpu)        GPU="$2";          shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "Error: --input and --output are required."
  exit 1
fi

# ── derive csv output path if not given ─────────────────────────────────
if [[ -z "$CSV_OUTPUT" ]]; then
  CSV_OUTPUT="${OUTPUT%.json}.csv"
fi

# ── environment ─────────────────────────────────────────────────────────
export TMPDIR="${TMPDIR:-/tmp}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

# ── run ─────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUTPUT")"

echo "AUT augmentation: lambda=${LAMBDA_VALUE}  target-n=${TARGET_N}"
echo "  input:  $INPUT"
echo "  output: $OUTPUT"

uv run python experiments/augment_aut_responses.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --target-n "$TARGET_N" \
    --lambda-value "$LAMBDA_VALUE" \
    --csv-output "$CSV_OUTPUT" \
    --seed "$SEED"

echo "Done. Score via https://openscoring.du.edu/ocsai"
