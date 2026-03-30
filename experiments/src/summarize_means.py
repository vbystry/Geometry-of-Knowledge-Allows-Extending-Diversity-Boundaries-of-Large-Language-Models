import argparse
import json
import os

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    summary = {}

    summary["mean_distinct"] = np.mean(df["partition_scores"].map(len))
    # Calculate mean of all generation scores across all instances
    all_scores = []
    for scores in df["generation_scores"]:
        all_scores.extend(scores)
    summary["mean_scores"] = np.mean(all_scores)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    df = pd.read_json(os.path.join(eval_dir, "scores_mean.jsonl"), lines=True)
    summary = summarize(df)
    with open(os.path.join(eval_dir, "summary2.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()


