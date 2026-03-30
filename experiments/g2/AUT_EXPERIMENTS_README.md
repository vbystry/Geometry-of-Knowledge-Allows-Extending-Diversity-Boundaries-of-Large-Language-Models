# AUT Experiments (only the paper setup)

This repo contains the exact two AUT (Alternative Uses Task) experiment configurations used in the paper:

- **Seeded / anchored**: use the **first generations from LLM-Discussion** as contextual seeds/anchors.
- **From-scratch**: generate ideas **without any discussion-derived context**.

We evaluated **G2** in both configurations. **Our proposed method** was likewise evaluated in the **seeded / anchored** configuration using the same LLM-Discussion outputs as anchors.

## Common setup

- **Generations per approach**: **500** (`iter_num=500`).

## (1) From-scratch AUT (no discussion context)

Runner: `scripts/eval/novelty/run_aut_eval.sh`

```bash
cd /mnt/bystry/research/emnlp25-g2
bash scripts/eval/novelty/run_aut_eval.sh
```

Key settings in the script:

- `iter_num=500`
- output: `results/aut/g2_theta{theta}_temp{temperature}_iter500_task{task_idx}/generations.jsonl`

## (2) Seeded / anchored AUT (LLM-Discussion first generation as seed)

Runner: `scripts/eval/novelty/run_aut_eval_seeded.sh`

To match “use the **first generations** as contextual seeds”, set:

- `max_seeds=100` (use all seeds from LLM Discussion)
- `iter_num=500`
- `seed_file=...` pointing to the LLM-Discussion AUT outputs you want to use as anchors

```bash
cd /mnt/bystry/research/emnlp25-g2
bash scripts/eval/novelty/run_aut_eval_seeded.sh
```

Outputs:

- `results/aut/g2_seeded_theta{theta}_temp{temperature}_iter500_maxseeds1/generations.jsonl`
- `results/aut/g2_seeded_theta{theta}_temp{temperature}_iter500_maxseeds1/ideas.csv` (auto-generated)

