#!/usr/bin/env bash
# Probe the geometry of the xRAG projector output.
# See experiments/probe_projector_distribution.py for details.
#
# Submit from the repo root with $WCSS_ACCOUNT set, e.g.
#
#   sbatch \
#     --extra=FORCE_RM_TMPDIR \
#     --account="$WCSS_ACCOUNT" \
#     --partition=lem-gpu-short \
#     --exclude=r10-12,r16-4 \
#     --chdir="$PWD" \
#     --gres=gpu:hopper:1 \
#     --cpus-per-task=8 \
#     --mem=64G \
#     --time=00:15:00 \
#     --job-name=proj-probe \
#     --output=logs/proj-probe_%j.out \
#     --error=logs/proj-probe_%j.err \
#     hpc/wcss_probe_projector.slurm

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

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

if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[setup] python: $(which python)  $(python --version)"

uv pip install --quiet -r requirements.txt
uv pip install --quiet hf_transfer

export PYTHONPATH="$PWD/vendor:$PWD/experiments:${PYTHONPATH:-}"

INPUT="${INPUT:-results/curated/g2_theta0.3_temp1_iter15/generations.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
SEED_RATIO="${SEED_RATIO:-0.3}"

ARGS=(--num-prompts "$NUM_PROMPTS" --seed-ratio "$SEED_RATIO")
if [[ -f "$INPUT" ]]; then
  ARGS+=(--input "$INPUT")
fi

python experiments/probe_projector_distribution.py "${ARGS[@]}"
