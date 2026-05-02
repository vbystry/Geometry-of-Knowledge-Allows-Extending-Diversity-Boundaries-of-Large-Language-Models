#!/usr/bin/env bash
# NoveltyBench ablation grid on WCSS (lem-gpu, 4× H100).
#
# Submit from the repo root. Set $WCSS_ACCOUNT to your allocation, e.g.
#   export WCSS_ACCOUNT=hpc-<group>-<id>
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
#   OUT_ROOT=results/ablations LAMBDA=6-10 sbatch ... hpc/wcss_ablations.slurm

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── env ─────────────────────────────────────────────────────────────────
# Models are ~14 GB each (xrag-7b + SFR). Cache on lustre, NOT $HOME / /tmp.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
# Datasets cache lives on TMPDIR (local node scratch) so its disk-space
# self-check doesn't trip on the near-quota lustre PD.
export HF_DATASETS_CACHE="${TMPDIR:-$HF_HOME}/hf-datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
# Other tools also default to ~/.cache, which would blow $HOME quota.
export XDG_CACHE_HOME="$PWD/.cache"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Plain venv (.venv on lustre — $HOME has only 1 GB quota).
# uv cache also on lustre — uv ignores PIP_CACHE_DIR and defaults to
# ~/.cache/uv, which would blow the $HOME quota.
export UV_CACHE_DIR="$PWD/.uv_cache"
export PIP_CACHE_DIR="$PWD/.pip_cache"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

# Optional: module loads (uncomment / adjust if your shell doesn't already see CUDA)
# module purge
# module load CUDA/12.4

if [[ ! -d .venv ]]; then
  echo "[setup] creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[setup] python: $(which python)  $(python --version)"

echo "[setup] uv pip install -r requirements.txt"
uv pip install --quiet -r requirements.txt
# requirements.txt doesn't list hf_transfer, but HF_HUB_ENABLE_HF_TRANSFER=1
# above requires it.
uv pip install --quiet hf_transfer

# augment_responses.py imports `xRAG.src.model` directly — make vendor importable.
# experiments/src/*.py uses `from src.common ...` — needs experiments/ on path.
export PYTHONPATH="$PWD/vendor:$PWD/experiments:${PYTHONPATH:-}"

# Helper scripts call `uv run python ...`; shim it to plain `python ...` so the
# .venv we just activated is what gets used (no uv installed on the cluster).
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

# ── ablation cell list   (subdir | extra args to run_augmentation.sh) ───
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
  # (D) calibrated noise levels: sigma chosen relative to the empirical
  # anchor scale (see experiments/probe_embedding_scale.py).
  # NoveltyBench / SFR-Embedding-Mistral reference: anchor-anchor L2 ~176,
  # cluster radius ~108, per-coord std ~4.77.
  # gauss sigma in {1,2}: perturbation ~36% / ~73% of anchor-anchor L2.
  "D_calibrated/gauss_s1|--mode gauss --sigma 1.0"
  "D_calibrated/gauss_s2|--mode gauss --sigma 2.0"
  # anchor-noise in {1,2,5,20,50}: cos(orig, perturbed) drops to
  # 0.98/0.92/0.69/~0.18/~0.07 -- s20/s50 effectively turn anchors into
  # random vectors of larger-than-natural norm.
  "D_calibrated/an_s1|--mode interp --anchor-noise 1.0"
  "D_calibrated/an_s2|--mode interp --anchor-noise 2.0"
  "D_calibrated/an_s5|--mode interp --anchor-noise 5.0"
  "D_calibrated/an_s20|--mode interp --anchor-noise 20.0"
  "D_calibrated/an_s50|--mode interp --anchor-noise 50.0"
  # extreme: latent norm 21x / 42x / 105x the natural anchor norm
  "D_calibrated/an_s100|--mode interp --anchor-noise 100.0"
  "D_calibrated/an_s200|--mode interp --anchor-noise 200.0"
  "D_calibrated/an_s500|--mode interp --anchor-noise 500.0"
  # random conditioning: ignore anchors entirely; sigma=4.77 = per-coord std
  # so ||z|| matches the natural anchor norm (~305).
  "D_calibrated/random|--mode random --sigma 4.77"
  # random at larger-than-natural scale: same comparison axis as the s20/s50
  # anchor-noise rows, so we can see whether the gain depends on the
  # natural-scale prior or just on a stochastic latent.
  "D_calibrated/random_s20|--mode random --sigma 20.0"
  "D_calibrated/random_s50|--mode random --sigma 50.0"
  "D_calibrated/random_s100|--mode random --sigma 100.0"
  "D_calibrated/random_s200|--mode random --sigma 200.0"
  "D_calibrated/random_s500|--mode random --sigma 500.0"
  # (E) bypass_random: feed z directly into the LLM input-embedding sequence
  # at the xRAG token position, skipping the projector. Three sigma scales
  # (probe values from probe_projector_distribution.py):
  #   - 0.0027: matches the per-coord std of model.embed_tokens.weight
  #             (i.e. ||z|| ~ 0.17 == natural Mistral token-embedding norm).
  #   - 1.0   : intermediate (||z|| ~ 64).
  #   - 200   : matches the projector-image per-coord std (||z|| ~ 12.8k,
  #             i.e. what the LLM is fine-tuned to receive when the projector
  #             is in use).
  "E_bypass/bypass_nat|--mode bypass_random --sigma 0.0027"
  "E_bypass/bypass_mid|--mode bypass_random --sigma 1.0"
  "E_bypass/bypass_proj|--mode bypass_random --sigma 200.0"
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

# ── 4-GPU worker pool via FIFO  ─────────────────────────────────────────
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
    CUDA_VISIBLE_DEVICES=0 bash experiments/scripts/run_evaluation.sh \
        --eval-dir "$out" \
        > "logs/eval_${subdir//\//_}_${SLURM_JOB_ID:-local}.log" 2>&1 || \
      echo "EVAL FAILED: $subdir" >&2
  done
fi

echo "[$(date +%T)] all done. results in: $OUT_ROOT"
exit $failed
