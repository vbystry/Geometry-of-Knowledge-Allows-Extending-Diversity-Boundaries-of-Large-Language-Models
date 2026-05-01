"""Calibrate the anchor-noise scale used in the weak-seed ablation.

The ablation in Section 5 (Robustness to Weak Anchor Seeds) injects
isotropic Gaussian noise into each anchor embedding before
exploration::

    arr = arr + torch.randn_like(arr) * float(anchor_noise)   # (1)

For the reported noise levels (sigma in {0.10, 0.25}) to constitute a
meaningful perturbation, sigma has to be interpreted relative to the
natural scale of the SFR-Embedding-Mistral retriever output, which is
NOT L2-normalised inside our pipeline. This script measures that
scale empirically and reports the resulting noise-to-signal statistics
that contextualise the ablation table.

Two views are produced:

(a) Toy probe: a handful of hand-written sentences are embedded and
    summary statistics (per-vector L2 norm, per-coordinate magnitude,
    pairwise cosine) are printed; this anchors the order of magnitude
    in a small, easily-replicable setting.

(b) Real-anchor probe: when a NoveltyBench generations.jsonl path is
    passed via --input, the script replicates the actual anchor-set
    construction performed by experiments/augment_responses.py
    (seed-ratio sub-selection, optional max-anchors truncation) on the
    first --num-prompts prompts, embeds those anchors, and reports
    the same statistics on the in-distribution embeddings used during
    the ablation.

For each sigma in --sigmas the script prints:
    - mean L2 norm of the noise vector (sqrt(d) * sigma in expectation),
    - noise-to-signal ratio ||eta|| / ||e||,
    - mean cosine similarity between original and perturbed anchor.

Run on a single GPU node, e.g. via the supplied SLURM submission script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# vendor/xRAG must be on PYTHONPATH (set by hpc/wcss_ablations.slurm or by
# wcss_probe_embedding.slurm).
from xRAG.src.model import SFR
from xRAG.src.language_modeling.utils import get_retrieval_embeds


TOY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A creative use for a paperclip is to bookmark a page.",
    "Use a brick as a doorstop to keep doors open in a windy hallway.",
    "Repurpose the brick as a weight for pressing flowers between book pages.",
    "Stack bricks to build a small outdoor herb garden border.",
]


def _embed_texts(retriever, tokenizer, texts, batch_size: int = 4):
    """Mirrors the tokenize+embed call in experiments/augment_responses.py."""
    device = next(retriever.parameters()).device
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = tokenizer(
                batch,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            e = get_retrieval_embeds(
                retriever,
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            ).float()
            embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def _summarise(name, embs):
    norms = embs.norm(dim=-1)
    print(f"--- {name} ---")
    print(f"  N={embs.shape[0]}  d={embs.shape[1]}")
    print(f"  per-vector ||e||: mean={norms.mean().item():.3f}"
          f"  std={norms.std().item():.3f}"
          f"  min={norms.min().item():.3f}"
          f"  max={norms.max().item():.3f}")
    print(f"  per-coord |e_i|: mean={embs.abs().mean().item():.4f}"
          f"  std={embs.std().item():.4f}"
          f"  max={embs.abs().max().item():.4f}")
    if embs.shape[0] >= 2:
        cos = torch.nn.functional.cosine_similarity(
            embs.unsqueeze(0), embs.unsqueeze(1), dim=-1
        )
        # off-diagonal mean
        mask = ~torch.eye(embs.shape[0], dtype=torch.bool)
        print(f"  pairwise cos sim (off-diag): mean={cos[mask].mean().item():.3f}"
              f"  min={cos[mask].min().item():.3f}"
              f"  max={cos[mask].max().item():.3f}")


def _noise_table(embs, sigmas, seed=0):
    g = torch.Generator().manual_seed(seed)
    norms = embs.norm(dim=-1)
    print("sigma   |eta|     |eta|/|e|   cos(e, e+eta)")
    for sigma in sigmas:
        eta = torch.randn(embs.shape, generator=g) * float(sigma)
        nn = eta.norm(dim=-1)
        ratio = (nn / norms).mean().item()
        perturbed = embs + eta
        cos = torch.nn.functional.cosine_similarity(embs, perturbed, dim=-1).mean().item()
        print(f"{sigma:>5.2f}  {nn.mean().item():>7.3f}   {ratio:>9.3f}   {cos:>+7.4f}")


def _load_real_anchors(path: Path, num_prompts: int, seed_ratio: float, max_anchors: int | None):
    """Replicate the anchor selection from augment_responses.py:select_seeds.

    Each line of generations.jsonl has fields including 'prompt' and a list of
    candidate generations under 'generations' (or 'responses'/'outputs'); the
    augmentation pipeline keeps the top floor(seed_ratio * len) generations as
    seeds. For the embedding-scale probe we only need the seed *texts* — the
    pipeline embeds them with SFR and treats the result as the anchor set.
    """
    seed_texts = []
    candidate_keys = ("generations", "responses", "outputs", "completions")
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= num_prompts:
                break
            rec = json.loads(line)
            cand = None
            for k in candidate_keys:
                if k in rec:
                    cand = rec[k]
                    break
            if cand is None:
                # fall back: maybe the record is just {prompt, response}
                if "response" in rec:
                    cand = [rec["response"]]
                else:
                    continue
            if isinstance(cand, list) and cand and isinstance(cand[0], dict):
                # records of the form [{"text": "...", ...}, ...]
                cand = [c.get("text") or c.get("response") or c.get("output") for c in cand]
            cand = [c for c in cand if isinstance(c, str) and c.strip()]
            if not cand:
                continue
            n_seed = max(1, int(seed_ratio * len(cand)))
            seeds = cand[:n_seed]
            if max_anchors is not None and len(seeds) > max_anchors:
                seeds = seeds[:max_anchors]
            seed_texts.extend(seeds)
    return seed_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="Salesforce/SFR-Embedding-Mistral")
    ap.add_argument("--input", type=Path, default=None,
                    help="Optional generations.jsonl to draw real anchors from.")
    ap.add_argument("--num-prompts", type=int, default=20,
                    help="If --input is given, embed anchors from this many prompts.")
    ap.add_argument("--seed-ratio", type=float, default=0.3,
                    help="Same as augment_responses.py --seed-ratio.")
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.10, 0.25, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading {args.retriever} ...")
    ret = SFR.from_pretrained(
        args.retriever, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    ret.eval()
    tok = AutoTokenizer.from_pretrained(args.retriever)

    print()
    print("=" * 64)
    print("(a) Toy probe — 5 hand-written sentences")
    print("=" * 64)
    toy = _embed_texts(ret, tok, TOY_TEXTS)
    _summarise("toy", toy)
    print()
    _noise_table(toy, args.sigmas, seed=args.seed)

    if args.input is not None:
        if not args.input.is_file():
            print(f"!! --input not found: {args.input}", file=sys.stderr)
            sys.exit(2)
        print()
        print("=" * 64)
        print(f"(b) Real-anchor probe — {args.input}")
        print("=" * 64)
        seeds = _load_real_anchors(
            args.input, args.num_prompts, args.seed_ratio, args.max_anchors
        )
        print(f"Loaded {len(seeds)} seed strings from {args.num_prompts} prompts.")
        if not seeds:
            print("No seeds extracted; aborting probe (b).")
            return
        real = _embed_texts(ret, tok, seeds)
        _summarise("real anchors", real)
        print()
        _noise_table(real, args.sigmas, seed=args.seed)


if __name__ == "__main__":
    main()
