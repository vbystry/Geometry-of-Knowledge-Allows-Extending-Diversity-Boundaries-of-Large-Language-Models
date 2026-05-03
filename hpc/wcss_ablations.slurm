#!/usr/bin/env bash
# NoveltyBench ablation grid on WCSS (lem-gpu, 4× H100).
#
# Reproduces every ablation cell that appears in the paper:
#   A. sampling strategy (interp/single/mean/medoid/gauss) × realignment
#      ON  -> A_sampling/style_on/{mode}
#   B. realignment OFF                                         (--no-style)
#      -> B_sampling/style_off/{mode}
#   C. weak-seed robustness (interp + max-anchors / anchor-noise grid)
#   D. sigma calibration: random sampler at sigma in
#      {4.77, 50, 500, 5000, 10000} and anchor-noise sweep at
#      sigma in {1, 5, 20, 50, 200} (RMSNorm-magnitude-invariance)
#   E. bypass projector at three probe-derived sigmas
#      {0.0027, 1.0, 200}  (E_bypass/bypass_s{sigma})
#   H. standalone (G2-independent): random sampler with seed-ratio=0
#      so all generations come from latent + realignment, no G2 anchors
#      kept (H_standalone/random_no_g2).
#
# The realignment-prompt ablation (paper §F/G) is run via
#   experiments/probe_refinement_prompts.py  +  hpc/wcss_probe_refinement.slurm
# and is *not* part of this grid.
#
# Submit from the repo root with $WCSS_ACCOUNT set, e.g.
#
#   sbatch \
#     --extra=FORCE_RM_TMPDIR \
#     --account="$WCSS_ACCOUNT" \
#     --partition=lem-gpu \
#     --exclude=r10-12,r16-4 \
#     --chdir="$PWD" \
#     --gres=gpu:hopper:4 \
#     --cpus-per-task=32 \
#     --mem=512G \
#     --time=24:00:00 \
#     --job-name=gok-abl \
#     --output=logs/gok-abl_%j.out \
#     --error=logs/gok-abl_%j.err \
#     hpc/wcss_ablations.slurm
#
# Override defaults via env, e.g.:
#   INPUT=results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \
#   OUT_ROOT=results/ablations \
#   SEED=42 sbatch ... hpc/wcss_ablations.slurm

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── env ─────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
# Datasets cache lives on TMPDIR (local node scratch) so its disk-space
# self-check doesn't trip on the near-quota lustre PD.
export HF_DATASETS_CACHE="${TMPDIR:-$HF_HOME}/hf-datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
# Keep tool caches off the small $HOME quota.
export XDG_CACHE_HOME="$PWD/.cache"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export UV_CACHE_DIR="$PWD/.uv_cache"
export PIP_CACHE_DIR="$PWD/.pip_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

if [[ ! -d .venv ]]; then
  echo "[setup] creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[setup] python: $(which python)  $(python --version)"

echo "[setup] uv pip install -r requirements.txt"
uv pip install --quiet -r requirements.txt
uv pip install --quiet hf_transfer

# augment_responses.py imports `xRAG.src.model` directly — make vendor importable.
# experiments/src/*.py uses `from src.common ...` — needs experiments/ on path.
export PYTHONPATH="$PWD/vendor:$PWD/experiments:${PYTHONPATH:-}"

# Helper scripts call `uv run python ...`; shim it to plain `python ...` so
# the .venv we just activated is used (no uv installed on the cluster).
uv() {
  if [[ "${1:-}" == "run" ]]; then
    shift
    "$@"
    return $?
  fi
  command uv "$@"
}
export -f uv

# ── inputs / config ─────────────────────────────────────────────────────
INPUT="${INPUT:-results/curated/g2_theta0.3_temp1_iter15/generations.jsonl}"
OUT_ROOT="${OUT_ROOT:-results/ablations}"
TARGET_N="${TARGET_N:-15}"
SEED_RATIO="${SEED_RATIO:-0.3}"
LAMBDA="${LAMBDA:-6-10}"
SEED="${SEED:-42}"
N_GPUS="${N_GPUS:-4}"
SKIP_EVAL="${SKIP_EVAL:-0}"

if [[ ! -f "$INPUT" ]]; then
  echo "FATAL: input not found: $INPUT" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" logs

# ── ablation cell list ─────────────────────────────────────────────────
# Each entry: "subdir | extra args to run_augmentation.sh"
CELLS=(
  # (A) sampling strategy, realignment ON
  "A_sampling/style_on/interp|--mode interp"
  "A_sampling/style_on/single|--mode single"
  "A_sampling/style_on/mean|--mode mean"
  "A_sampling/style_on/medoid|--mode medoid"
  "A_sampling/style_on/gauss|--mode gauss"
  # (B) sampling strategy, realignment OFF
  "B_sampling/style_off/interp|--mode interp --no-style"
  "B_sampling/style_off/single|--mode single --no-style"
  "B_sampling/style_off/mean|--mode mean --no-style"
  "B_sampling/style_off/medoid|--mode medoid --no-style"
  "B_sampling/style_off/gauss|--mode gauss --no-style"
  # (C) weak-seed robustness (interp only)
  "C_weak_seed/ma1|--mode interp --max-anchors 1"
  "C_weak_seed/ma2|--mode interp --max-anchors 2"
  "C_weak_seed/an010|--mode interp --anchor-noise 0.10"
  "C_weak_seed/an025|--mode interp --anchor-noise 0.25"
  "C_weak_seed/ma2_an010|--mode interp --max-anchors 2 --anchor-noise 0.10"
  # (D) sigma calibration for random sampler (RMSNorm-magnitude-invariance):
  # SFR per-coord std = 4.77 → ||z|| matches natural anchor norm. The plateau
  # across orders of magnitude is the load-bearing claim.
  "D_calibrated/random_s4.77|--mode random --sigma 4.77"
  "D_calibrated/random_s50|--mode random --sigma 50.0"
  "D_calibrated/random_s500|--mode random --sigma 500.0"
  "D_calibrated/random_s5000|--mode random --sigma 5000.0"
  "D_calibrated/random_s10000|--mode random --sigma 10000.0"
  # (D') anchor-noise sweep for interp: cosine(orig, perturbed) drops to
  # 0.98/0.69/~0.18/~0.07/~0.02 at sigma in {1, 5, 20, 50, 200}.
  "D_calibrated/an_s1|--mode interp --anchor-noise 1.0"
  "D_calibrated/an_s5|--mode interp --anchor-noise 5.0"
  "D_calibrated/an_s20|--mode interp --anchor-noise 20.0"
  "D_calibrated/an_s50|--mode interp --anchor-noise 50.0"
  "D_calibrated/an_s200|--mode interp --anchor-noise 200.0"
  # (E) bypass_random: feed z directly into the LLM input-embedding sequence
  # at the xRAG token position (no projector). Probe-derived sigmas:
  #   0.0027 = LLM embed_tokens per-coord std (||z|| ~ 0.17)
  #   1.0    = intermediate (||z|| ~ 64)
  #   200    = projector-image per-coord std (||z|| ~ 12.8k)
  "E_bypass/bypass_s0.0027|--mode bypass_random --sigma 0.0027"
  "E_bypass/bypass_s1.0|--mode bypass_random --sigma 1.0"
  "E_bypass/bypass_s200|--mode bypass_random --sigma 200.0"
  # (H) standalone (G2-independence): random sampler, seed-ratio=0 so no
  # G2 outputs are kept; all 15 generations come from latent + realignment.
  "H_standalone/random_no_g2|--mode random --sigma 4.77 --seed-ratio 0"
)

run_cell() {
  local gpu_id="$1" spec="$2"
  local subdir="${spec%%|*}" extra="${spec##*|}"
  local out="$OUT_ROOT/$subdir"
  local log="logs/cell_${subdir//\//_}_${SLURM_JOB_ID:-local}.log"
  mkdir -p "$out"

  if [[ -s "$out/generations.jsonl" ]]; then
    echo "[$(date +%T)] [GPU $gpu_id] SKIP   $subdir   (output exists)"
    return 0
  fi

  echo "[$(date +%T)] [GPU $gpu_id] START  $subdir   ($extra)"
  # shellcheck disable=SC2086
  bash experiments/scripts/run_augmentation.sh \
       --input  "$INPUT" \
       --output "$out/generations.jsonl" \
       --seed-ratio "$SEED_RATIO" \
       --lambda     "$LAMBDA" \
       --target-n   "$TARGET_N" \
       --seed       "$SEED" \
       --gpu        "$gpu_id" \
       $extra \
       > "$log" 2>&1
  local rc=$?
  echo "[$(date +%T)] [GPU $gpu_id] DONE   $subdir   rc=$rc   log=$log"
  return $rc
}
export -f run_cell
export INPUT OUT_ROOT TARGET_N SEED_RATIO LAMBDA SEED

# ── 4-GPU worker pool via FIFO ──────────────────────────────────────────
FIFO=$(mktemp -u "${TMPDIR:-/tmp}/gok_gpu_XXXXXX")
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm "$FIFO"
for ((g=0; g<N_GPUS; g++)); do echo "$g" >&9; done

failed=0
for spec in "${CELLS[@]}"; do
  read -u 9 gpu
  (
    if run_cell "$gpu" "$spec"; then :; else
      echo "FAILED: $spec" >> logs/failed_${SLURM_JOB_ID:-local}.txt
    fi
    echo "$gpu" >&9
  ) &
done
wait
exec 9>&-

if [[ -f logs/failed_${SLURM_JOB_ID:-local}.txt ]]; then
  echo "Some cells failed:" >&2
  cat logs/failed_${SLURM_JOB_ID:-local}.txt >&2
  failed=1
fi

# ── evaluation (sequential, one GPU is enough) ──────────────────────────
if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "[$(date +%T)] starting evaluation pass"
  for spec in "${CELLS[@]}"; do
    subdir="${spec%%|*}"
    out="$OUT_ROOT/$subdir"
    [[ -f "$out/generations.jsonl" ]] || { echo "skip eval (no gens): $out"; continue; }
    if [[ -s "$out/summary.json" && "${FORCE_EVAL:-0}" != "1" ]]; then
      echo "skip eval (summary exists): $out"
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 bash experiments/scripts/run_evaluation.sh \
        --eval-dir "$out" \
        > "logs/eval_${subdir//\//_}_${SLURM_JOB_ID:-local}.log" 2>&1 || \
      echo "EVAL FAILED: $subdir" >&2
  done
fi

echo "[$(date +%T)] all done. results in: $OUT_ROOT"
exit $failed
