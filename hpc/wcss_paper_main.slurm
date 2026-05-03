#!/usr/bin/env bash
# Reproduce the main NoveltyBench numbers reported in the paper:
#   - interp on G2 anchors                (Tab 2 hybrid)
#   - random sampler with G2 anchors kept (companion isolating "stochastic
#                                          latent" contribution)
#   - random sampler standalone (no G2)   (Tab 1 standalone)
# at five generation budgets k in {10, 15, 20, 25, 30}.
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
#     --job-name=gok-paper \
#     --output=logs/gok-paper_%j.out \
#     --error=logs/gok-paper_%j.err \
#     hpc/wcss_paper_main.slurm
#
# Override defaults via env, e.g.:
#   INPUT=results/curated/g2_theta0.3_temp1_iter30/generations.jsonl \
#   OUT_ROOT=results/paper_main \
#   SEED=42 sbatch ... hpc/wcss_paper_main.slurm

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── env ─────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="${TMPDIR:-$HF_HOME}/hf-datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
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

export PYTHONPATH="$PWD/vendor:$PWD/experiments:${PYTHONPATH:-}"

# Helpers call `uv run python ...`; shim to plain `python` so the .venv
# we just activated is used (no uv binary on the cluster).
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
INPUT="${INPUT:-results/curated/g2_theta0.3_temp1_iter30/generations.jsonl}"
OUT_ROOT="${OUT_ROOT:-results/paper_main}"
LAMBDA="${LAMBDA:-6-10}"
SEED="${SEED:-42}"
N_GPUS="${N_GPUS:-4}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SIGMA_RANDOM="${SIGMA_RANDOM:-4.77}"

if [[ ! -f "$INPUT" ]]; then
  echo "FATAL: input not found: $INPUT" >&2
  echo "Hint: produce G2 seed generations first (>=30 per record)" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" logs

# ── cell list ───────────────────────────────────────────────────────────
# Each line: "subdir | extra args to run_augmentation.sh"
BUDGETS=(10 15 20 25 30)
CELLS=()
for k in "${BUDGETS[@]}"; do
  CELLS+=("k${k}/interp_g2|--mode interp --target-n ${k} --seed-ratio 0.3")
  CELLS+=("k${k}/random_g2|--mode random --sigma ${SIGMA_RANDOM} --target-n ${k} --seed-ratio 0.3")
  CELLS+=("k${k}/random_no_g2|--mode random --sigma ${SIGMA_RANDOM} --target-n ${k} --seed-ratio 0")
done

run_cell() {
  local gpu_id="$1" spec="$2"
  local subdir="${spec%%|*}" extra="${spec##*|}"
  local out="$OUT_ROOT/$subdir"
  local log="logs/paper_${subdir//\//_}_${SLURM_JOB_ID:-local}.log"
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
       --lambda "$LAMBDA" \
       --seed   "$SEED" \
       --gpu    "$gpu_id" \
       $extra \
       > "$log" 2>&1
  local rc=$?
  echo "[$(date +%T)] [GPU $gpu_id] DONE   $subdir   rc=$rc   log=$log"
  return $rc
}
export -f run_cell
export INPUT OUT_ROOT LAMBDA SEED

# ── 4-GPU worker pool via FIFO ──────────────────────────────────────────
FIFO=$(mktemp -u "${TMPDIR:-/tmp}/gok_paper_XXXXXX")
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm "$FIFO"
for ((g=0; g<N_GPUS; g++)); do echo "$g" >&9; done

failed=0
for spec in "${CELLS[@]}"; do
  read -u 9 gpu
  (
    if run_cell "$gpu" "$spec"; then :; else
      echo "FAILED: $spec" >> logs/paper_failed_${SLURM_JOB_ID:-local}.txt
    fi
    echo "$gpu" >&9
  ) &
done
wait
exec 9>&-

if [[ -f logs/paper_failed_${SLURM_JOB_ID:-local}.txt ]]; then
  echo "Some cells failed:" >&2
  cat logs/paper_failed_${SLURM_JOB_ID:-local}.txt >&2
  failed=1
fi

# ── evaluation pass ─────────────────────────────────────────────────────
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
        > "logs/paper_eval_${subdir//\//_}_${SLURM_JOB_ID:-local}.log" 2>&1 || \
      echo "EVAL FAILED: $subdir" >&2
  done
fi

echo "[$(date +%T)] all done. results in: $OUT_ROOT"
exit $failed
