# this scripts use vllm to run_sample
# if you want use transformers, please change "eval/novelty-bench/src/inference.py" to "eval/novelty-bench/src/run_sample.py"
export CUDA_VISIBLE_DEVICES=0
top_p=1.0
top_k=50
min_p=0
iter_num=10
temperature=1.5

outputfile=results/novelty/sample_temp${temperature}
model="mistralai/Mistral-7B-Instruct-v0.2"

# Activate the inference environment for generation
source .venv/bin/activate
python eval/novelty-bench/src/inference.py --model $model --data curated --eval-dir $outputfile --num-generations 10 --temperature $temperature --top_p $top_p --top_k $top_k --min_p $min_p

# Activate the evaluation environment for scoring
source .venv/bin/activate
python eval/novelty-bench/src/partition.py --eval-dir $outputfile --alg classifier
python eval/novelty-bench/src/score.py --eval-dir $outputfile --patience 0.8
python eval/novelty-bench/src/summarize.py --eval-dir $outputfile