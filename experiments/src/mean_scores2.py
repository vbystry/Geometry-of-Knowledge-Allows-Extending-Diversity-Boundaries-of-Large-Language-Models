import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    scores_file = os.path.join(eval_dir, "scores.jsonl")
    
    if not os.path.exists(scores_file):
        print(f"Error: {scores_file} not found!")
        return
    
    df = pd.read_json(scores_file, lines=True)
    
    # Flatten all generation scores and calculate mean
    all_generation_scores = []
    for scores in df["generation_scores"]:
        all_generation_scores.extend(scores)
    
    # Flatten all partition scores and calculate mean
    all_partition_scores = []
    for scores in df["partition_scores"]:
        all_partition_scores.extend(scores)
    
    # Calculate means
    mean_generation_score = np.mean(all_generation_scores) if all_generation_scores else 0.0
    mean_partition_score = np.mean(all_partition_scores) if all_partition_scores else 0.0
    
    summary = {
        "mean_generation_score": float(mean_generation_score),
        "mean_partition_score": float(mean_partition_score),
    }
    
    output_file = os.path.join(eval_dir, "mean_scores2.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Mean generation score: {mean_generation_score:.4f}")
    print(f"Mean partition score: {mean_partition_score:.4f}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()

