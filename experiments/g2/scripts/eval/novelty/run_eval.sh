export CUDA_VISIBLE_DEVICES=3
theta=0.3
iter_num=30
temperature=1
python_file=run_eval.py
outputfile=results/novelty/g2_theta${theta}_temp${temperature}_iter${iter_num}
model="mistralai/Mistral-7B-Instruct-v0.2"

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Export cache variables explicitly to ensure they're used
export HF_HOME=${HF_HOME:-/mnt/bystry/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/bystry/hf/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bystry/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/bystry/datasets}

# Use the local .venv for generation
.venv/bin/python eval/novelty-bench/src/${python_file} --model_name_or_path $model --data curated --eval-dir $outputfile --iter_num $iter_num --temperature $temperature --theta $theta

# Use the local .venv for scoring
.venv/bin/python eval/novelty-bench/src/partition.py --eval-dir $outputfile --alg classifier
.venv/bin/python eval/novelty-bench/src/score.py --eval-dir $outputfile --patience 0.8
.venv/bin/python eval/novelty-bench/src/summarize.py --eval-dir $outputfile