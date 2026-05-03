#!/usr/bin/env bash
# NoveltyBench ablation grid.
#
# Isolates the contribution of the proposed continuous sampling strategy by
# holding the projector, anchor source, target_n, and (optionally) realignment
# fixed across cells, and varying ONE axis at a time:
#
#   (A) sampling strategy        : single-point baselines (single/mean/medoid),
#                                  non-geometric Gaussian noise (gauss), and
#                                  the proposed interpolation/extrapolation (interp).
#                                  Realignment ON for all cells (paper default).
#
#   (B) realignment toggle       : repeat (A) with realignment OFF (--no-style)
#                                  to attribute Distinct/Utility gains to the
#                                  sampling step rather than the styler.
#
#   (C) weak-seed robustness     : interp only, but vary anchor coverage
#                                  (--max-anchors) and anchor quality
#                                  (--anchor-noise) to probe what happens when
#                                  the seed prior under-covers the solution space.
#
# Each cell calls run_augmentation.sh + run_evaluation.sh.
#
# Usage:
#   bash experiments/scripts/run_ablations.sh \
#       --input  results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
#       --out-root results/ablations/ \
#       [--target-n 15] [--seed-ratio 0.3] [--lambda 6-10] [--gpu 0]
#       [--skip-eval]   # run augmentation only

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────
INPUT=""
OUT_ROOT="results/ablations"
TARGET_N=15
SEED_RATIO=0.3
LAMBDA_VALUE="6-10"
SEED=42
SIGMA=0.05
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_EVAL=0

# Sampling-strategy grid for (A) and (B).
MODES=(interp single mean medoid gauss)

# Weak-seed grid for (C). Each entry is "max_anchors:anchor_noise".
# 0 / "" means: do not constrain that axis.
WEAK_GRID=(
  "1:0"      # extreme: a single anchor  (low coverage)
  "2:0"      # 2 anchors                  (low coverage)
  ":0.10"    # all anchors, mild noise
  ":0.25"    # all anchors, heavy noise
  "2:0.10"   # combined: few + noisy anchors
)

# (D) Sigma-calibration grid for the random sampler (no anchors).
# Reference: SFR-Embedding-Mistral on NoveltyBench has per-coord std ~4.77,
# so sigma=4.77 gives ||z|| matching the natural anchor norm. The plateau
# across orders of magnitude is the RMSNorm-magnitude-invariance result.
RANDOM_SIGMA_GRID=(4.77 50 500 5000 10000)

# (D') Anchor-noise grid for interp: cosine(orig, perturbed) drops to
# 0.98/0.92/0.69/~0.18/~0.07 for sigma in {1,2,5,20,50}. Noise far above
# the natural scale effectively turns anchors into random vectors.
ANCHOR_NOISE_GRID=(1.0 5.0 20.0 50.0 200.0)

# (E) bypass-projector sigma grid. Probe values from
# probe_projector_distribution.py:
#   0.0027 = LLM embed_tokens per-coord std (||z|| ~ 0.17, natural Mistral)
#   1.0    = intermediate (||z|| ~ 64)
#   200    = projector-image per-coord std (||z|| ~ 12.8k, what xRAG was
#            fine-tuned to receive when the projector is in the loop)
BYPASS_SIGMA_GRID=(0.0027 1.0 200.0)

# ── parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)       INPUT="$2";       shift 2 ;;
    --out-root)    OUT_ROOT="$2";    shift 2 ;;
    --target-n)    TARGET_N="$2";    shift 2 ;;
    --seed-ratio)  SEED_RATIO="$2";  shift 2 ;;
    --lambda)      LAMBDA_VALUE="$2";shift 2 ;;
    --seed)        SEED="$2";        shift 2 ;;
    --sigma)       SIGMA="$2";       shift 2 ;;
    --gpu)         GPU="$2";         shift 2 ;;
    --skip-eval)   SKIP_EVAL=1;      shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Error: --input is required (path to seed generations JSONL, e.g. G2 seeds)."
  exit 1
fi

mkdir -p "$OUT_ROOT"

run_cell() {
  local cell_dir="$1"
  shift
  local extra_args=("$@")

  mkdir -p "$cell_dir"
  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  CELL: $cell_dir"
  echo "  args: ${extra_args[*]}"
  echo "════════════════════════════════════════════════════════════════════"

  bash experiments/scripts/run_augmentation.sh \
      --input  "$INPUT" \
      --output "$cell_dir/generations.jsonl" \
      --seed-ratio "$SEED_RATIO" \
      --lambda     "$LAMBDA_VALUE" \
      --target-n   "$TARGET_N" \
      --seed       "$SEED" \
      --sigma      "$SIGMA" \
      --gpu        "$GPU" \
      "${extra_args[@]}"

  if [[ "$SKIP_EVAL" -eq 0 ]]; then
    bash experiments/scripts/run_evaluation.sh --eval-dir "$cell_dir"
  fi
}

# ── (A) sampling-strategy ablation, realignment ON ──────────────────────
for mode in "${MODES[@]}"; do
  run_cell "$OUT_ROOT/A_sampling/style_on/$mode"  --mode "$mode"
done

# ── (B) sampling-strategy ablation, realignment OFF ─────────────────────
# Cleanest attribution: any Distinct/Utility gap between modes here is from
# the sampling step alone (the styler is removed).
for mode in "${MODES[@]}"; do
  run_cell "$OUT_ROOT/B_sampling/style_off/$mode" --mode "$mode" --no-style
done

# ── (C) weak-seed robustness, interp only ───────────────────────────────
for entry in "${WEAK_GRID[@]}"; do
  ma="${entry%%:*}"
  an="${entry##*:}"

  cell_name="ma${ma:-NA}_an${an:-NA}"
  extra=(--mode interp)
  [[ -n "$ma" ]] && extra+=(--max-anchors "$ma")
  [[ -n "$an" && "$an" != "0" ]] && extra+=(--anchor-noise "$an")

  run_cell "$OUT_ROOT/C_weak_seed/$cell_name" "${extra[@]}"
done

# ── (D) sigma calibration for the random sampler ────────────────────────
# Tests RMSNorm magnitude invariance: Distinct/Utility should plateau across
# orders of magnitude in sigma because the projector RMSNorm absorbs ||z||.
for sig in "${RANDOM_SIGMA_GRID[@]}"; do
  run_cell "$OUT_ROOT/D_calibrated/random_s${sig}" --mode random --sigma "$sig"
done

# (D') Anchor-noise sweep for interp at increasing sigma. Companion to (D):
# tests whether anchor identity matters once noise is large enough to
# randomise anchors.
for sig in "${ANCHOR_NOISE_GRID[@]}"; do
  run_cell "$OUT_ROOT/D_calibrated/an_s${sig}" --mode interp --anchor-noise "$sig"
done

# ── (E) bypass projector ────────────────────────────────────────────────
# Insert z directly into the LLM input-embedding sequence at the xRAG token
# position, skipping the projector. Three sigma scales: natural Mistral
# embedding, intermediate, and projector-image scale.
for sig in "${BYPASS_SIGMA_GRID[@]}"; do
  run_cell "$OUT_ROOT/E_bypass/bypass_s${sig}" --mode bypass_random --sigma "$sig"
done

# ── (H) standalone (G2-independence) ─────────────────────────────────────
# Same random sampler but seed-ratio=0 means no G2 anchors are kept;
# all 15 outputs come from latent + realignment. This is the fairer
# "method works without G2" cell.
run_cell "$OUT_ROOT/H_standalone/random_no_g2" \
    --mode random --sigma 4.77 --seed-ratio 0

echo
echo "All ablation cells written under: $OUT_ROOT"
echo
echo "Note: realignment-prompt ablation (F/G in the paper) is run via"
echo "      experiments/probe_refinement_prompts.py (small grid harness)."
