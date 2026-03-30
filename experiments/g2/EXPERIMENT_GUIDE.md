# Running Experiments on Novelty-Bench with Mistral-7B-Instruct-v0.2

This guide explains how to run experiments on Novelty-Bench using the Mistral-7B-Instruct-v0.2 model.

## Prerequisites

1. **Virtual Environment**: The `.venv` environment has been set up with all required dependencies from `g2_eval_requirements.txt`.

2. **Model Access**: Ensure you have access to the `mistralai/Mistral-7B-Instruct-v0.2` model (either through HuggingFace or local path).

## Running G2 Method (Recommended)

The G2 method uses guided generation to enhance output diversity. To run it:

```bash
cd /mnt/bystry/research/emnlp25-g2
bash scripts/eval/novelty/run_eval.sh
```

This script will:
1. Generate outputs using G2 method with `theta=0.3` and `temperature=1.0`
2. Partition the results using a classifier
3. Score the outputs
4. Generate a summary

**Configuration** (in `scripts/eval/novelty/run_eval.sh`):
- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Theta: `0.3` (G2 guidance parameter)
- Temperature: `1.0`
- Iterations: `10` (number of generations per prompt)
- Dataset: `curated` (can be changed to `wildchat`)
- Output directory: `results/novelty/g2_theta0.3_temp1`

## Running Sampling-Based Methods (Baseline)

To compare with sampling-based methods (temperature, top-k, top-p, min-p):

```bash
cd /mnt/bystry/research/emnlp25-g2
bash scripts/eval/novelty/run_sample.sh
```

This script uses vLLM for faster inference and generates outputs with:
- Temperature: `1.5`
- Top-p: `1.0`
- Top-k: `50`
- Min-p: `0`

## Customizing Parameters

### For G2 Method

Edit `scripts/eval/novelty/run_eval.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0
theta=0.3          # G2 guidance parameter (adjust for different diversity levels)
iter_num=10        # Number of generations per prompt
temperature=1      # Generation temperature
model="mistralai/Mistral-7B-Instruct-v0.2"
```

### For Sampling Methods

Edit `scripts/eval/novelty/run_sample.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0
temperature=1.5    # Sampling temperature
top_p=1.0         # Top-p sampling
top_k=50          # Top-k sampling
min_p=0           # Min-p sampling
model="mistralai/Mistral-7B-Instruct-v0.2"
```

## Changing Dataset

To use the `wildchat` dataset instead of `curated`, modify the `--data` parameter in the scripts:

```bash
python eval/novelty-bench/src/run_eval.py --model_name_or_path $model --data wildchat --eval-dir $outputfile ...
```

## Output Structure

After running the evaluation, you'll find:

```
results/novelty/g2_theta0.3_temp1/
├── generations.jsonl    # Generated outputs
├── partitions.jsonl     # Partitioned results
├── scores.jsonl        # Quality scores
└── summary.json        # Summary statistics
```

## Troubleshooting

1. **Module Import Error**: The scripts now automatically detect the project root. If you still see import errors, ensure you're running from the project root directory.

2. **CUDA Out of Memory**: Reduce `iter_num` or use a smaller batch size by modifying the `--eval_batch_size` parameter.

3. **Missing Files**: Ensure the generation step completes successfully before running partition/score steps.

4. **Environment Issues**: Make sure the `.venv` is activated and contains all required packages from `g2_eval_requirements.txt`.

## Additional Diversity Metrics

To evaluate additional diversity metrics (sentence-BERT, self-BLEU, EAD):

```bash
cd /mnt/bystry/research/emnlp25-g2
source .venv/bin/activate
bash scripts/eval/novelty/diversity.sh
```

Make sure to update the paths in `diversity.sh` to point to your results directory.


