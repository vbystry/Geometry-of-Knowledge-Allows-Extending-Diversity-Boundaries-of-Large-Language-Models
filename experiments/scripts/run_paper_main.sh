#!/usr/bin/env bash
# Reproduce the main NoveltyBench numbers reported in the paper:
#
#   - Tab 1 (standalone):     random sampler, no G2 anchors  (seed-ratio = 0)
#   - Tab 2 (hybrid):         interp sampler on G2 anchors    (paper default)
#   - Companion (random+G2):  random sampler with G2 anchors kept (seed-ratio = 0.3)
#                             — isolates the contribution of seed-derived
#                               directional information vs.\ stochastic latent.
#
# Each method is run at five generation budgets k in {10, 15, 20, 25, 30}.
# All three methods share the same projector, encoder, decoding settings,
# and seed (default 42), so reruns at the same seed produce bit-identical
# generations.
#
# Usage:
#   bash experiments/scripts/run_paper_main.sh \
#       --input    results/curated/g2_theta0.3_temp1_iter30/generations.jsonl \
#       --out-root results/paper_main/
#
# The G2-seed JSONL is expected to contain >=30 generations per record
# (k=30 is the largest budget). For standalone (no-G2) cells the G2
# generations are not used as anchors but the JSONL is still consumed
# as the prompt source.

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────
INPUT=""
OUT_ROOT="results/paper_main"
LAMBDA_VALUE="6-10"
SEED=42
SIGMA_RANDOM=4.77   # SFR per-coord std => ||z|| matches natural anchor norm
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_EVAL=0
BUDGETS=(10 15 20 25 30)

# ── parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)       INPUT="$2";       shift 2 ;;
    --out-root)    OUT_ROOT="$2";    shift 2 ;;
    --lambda)      LAMBDA_VALUE="$2";shift 2 ;;
    --seed)        SEED="$2";        shift 2 ;;
    --sigma)       SIGMA_RANDOM="$2";shift 2 ;;
    --gpu)         GPU="$2";         shift 2 ;;
    --skip-eval)   SKIP_EVAL=1;      shift ;;
    --budgets)     IFS=',' read -ra BUDGETS <<<"$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Error: --input is required (G2 seed generations JSONL)."
  exit 1
fi

mkdir -p "$OUT_ROOT"

run_cell() {
  local cell_dir="$1"
  shift
  local extra_args=("$@")

  if [[ -s "$cell_dir/generations.jsonl" ]]; then
    echo "[skip] $cell_dir (generations.jsonl already exists)"
    return 0
  fi

  mkdir -p "$cell_dir"
  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  CELL: $cell_dir"
  echo "  args: ${extra_args[*]}"
  echo "════════════════════════════════════════════════════════════════════"

  bash experiments/scripts/run_augmentation.sh \
      --input  "$INPUT" \
      --output "$cell_dir/generations.jsonl" \
      --lambda "$LAMBDA_VALUE" \
      --seed   "$SEED" \
      --gpu    "$GPU" \
      "${extra_args[@]}"

  if [[ "$SKIP_EVAL" -eq 0 ]]; then
    bash experiments/scripts/run_evaluation.sh --eval-dir "$cell_dir"
  fi
}

# Three methods × five budgets = 15 cells. Method definitions:
#   interp_g2     : anchor-based interpolation on G2 seeds. seed-ratio=0.3
#                   keeps the first 30% of G2 outputs as seeds (paper default).
#   random_g2     : random isotropic sampler at SFR-natural sigma, but G2
#                   seeds kept (seed-ratio=0.3) so the cell isolates
#                   "stochastic latent" contribution while still benefiting
#                   from G2 directional seeds.
#   random_no_g2  : random isotropic sampler standalone, no G2 seeds kept
#                   (seed-ratio=0). Pure-method ceiling.
for k in "${BUDGETS[@]}"; do
  run_cell "$OUT_ROOT/k${k}/interp_g2" \
      --target-n "$k" --seed-ratio 0.3 --mode interp

  run_cell "$OUT_ROOT/k${k}/random_g2" \
      --target-n "$k" --seed-ratio 0.3 --mode random --sigma "$SIGMA_RANDOM"

  run_cell "$OUT_ROOT/k${k}/random_no_g2" \
      --target-n "$k" --seed-ratio 0 --mode random --sigma "$SIGMA_RANDOM"
done

echo
echo "All paper-main cells written under: $OUT_ROOT"
