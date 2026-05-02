"""Characterise the geometry of the xRAG projector output.

xMistral injects the conditioning vector into the LLM by replacing the
embedding at the position of the special XRAG_TOKEN with the output of
its projector ``model.projector(retrieval_embed)``. This script reports
the statistics of that output for three input regimes:

(R) real anchors -- SFR-Embedding-Mistral encodings of NoveltyBench
    seed generations (the operating regime of the proposed method);
(N) random natural-scale -- z ~ N(0, sigma_nat^2 I) with
    sigma_nat = 4.77 so ||z|| matches the natural anchor norm
    (the regime that already nearly matches `interp` on the curated
    NoveltyBench subset, see results/ablations);
(L) large-sigma -- z ~ N(0, 50^2 I); the latent norm is ~10x the
    natural one but the projector still maps it into the LLM
    embedding space. Included to see whether the projector squashes
    out-of-distribution latents back onto the same image manifold.

For each regime, we also compare the projector output to a sample of
the natural Mistral input-embedding distribution
``model.model.embed_tokens.weight``, since that is the distribution
the LLM is conditioned to expect at every position.

Usage::

    python experiments/probe_projector_distribution.py \\
        --input results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \\
        --num-prompts 20

Run on a single GPU node via hpc/wcss_probe_projector.slurm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# vendor/xRAG and experiments must be on PYTHONPATH (set by the slurm wrapper).
from xRAG.src.model import SFR, XMistralForCausalLM
from xRAG.src.language_modeling.utils import get_retrieval_embeds


def _embed_texts(retriever, tokenizer, texts, batch_size: int = 4):
    device = next(retriever.parameters()).device
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = tokenizer(
                batch, max_length=256, padding=True, truncation=True,
                return_tensors="pt",
            )
            e = get_retrieval_embeds(
                retriever,
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            ).float()
            embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def _summarise(name, X):
    norms = X.norm(dim=-1)
    print(f"--- {name} ---")
    print(f"  N={X.shape[0]}  d={X.shape[1]}")
    print(f"  per-vector ||x||: mean={norms.mean().item():.3f}"
          f"  std={norms.std().item():.3f}"
          f"  min={norms.min().item():.3f}"
          f"  max={norms.max().item():.3f}")
    print(f"  per-coord  |x_i|: mean={X.abs().mean().item():.4f}"
          f"  std={X.std().item():.4f}"
          f"  max={X.abs().max().item():.4f}")
    if X.shape[0] >= 2:
        n = min(X.shape[0], 256)
        S = X[:n]
        diffs = S.unsqueeze(0) - S.unsqueeze(1)
        d = diffs.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(
            S.unsqueeze(0), S.unsqueeze(1), dim=-1
        )
        mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
        print(f"  pairwise ||x_i - x_j||_2: mean={d[mask].mean().item():.3f}"
              f"  median={d[mask].median().item():.3f}"
              f"  min={d[mask].min().item():.3f}"
              f"  max={d[mask].max().item():.3f}")
        print(f"  pairwise cos(x_i, x_j):  mean={cos[mask].mean().item():.3f}"
              f"  min={cos[mask].min().item():.3f}"
              f"  max={cos[mask].max().item():.3f}")
        # diameter (max pairwise) and centroid spread
        c = X.mean(dim=0)
        radii = (X - c).norm(dim=-1)
        print(f"  cluster radius ||x - c||: mean={radii.mean().item():.3f}"
              f"  median={radii.median().item():.3f}"
              f"  max={radii.max().item():.3f}")


def _load_real_anchors(path: Path, num_prompts: int, seed_ratio: float):
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
                    cand = rec[k]; break
            if cand is None:
                if "response" in rec:
                    cand = [rec["response"]]
                else:
                    continue
            if isinstance(cand, list) and cand and isinstance(cand[0], dict):
                cand = [c.get("text") or c.get("response") or c.get("output") for c in cand]
            cand = [c for c in cand if isinstance(c, str) and c.strip()]
            if not cand:
                continue
            n_seed = max(1, int(seed_ratio * len(cand)))
            seed_texts.extend(cand[:n_seed])
    return seed_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Hannibal046/xrag-7b",
                    help="xMistral checkpoint with the trained xRAG projector.")
    ap.add_argument("--retriever", default="Salesforce/SFR-Embedding-Mistral")
    ap.add_argument("--input", type=Path, default=None,
                    help="generations.jsonl to draw real anchors from.")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--seed-ratio", type=float, default=0.3)
    ap.add_argument("--n-natural", type=int, default=512,
                    help="N random latent vectors at natural anchor scale.")
    ap.add_argument("--n-large", type=int, default=512,
                    help="N random latent vectors at large sigma.")
    ap.add_argument("--sigma-natural", type=float, default=4.77)
    ap.add_argument("--sigma-large", type=float, default=50.0)
    ap.add_argument("--n-token-emb", type=int, default=4096,
                    help="N rows sampled from the LLM input-embedding matrix "
                         "for the reference distribution.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading retriever {args.retriever} ...")
    ret = SFR.from_pretrained(
        args.retriever, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    ret.eval()
    ret_tok = AutoTokenizer.from_pretrained(args.retriever)

    print(f"Loading LLM {args.llm} (xRAG projector lives inside) ...")
    llm = XMistralForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    llm.eval()
    proj = llm.projector  # nn.Sequential (mlp2x_gelu) on cuda:0
    d = next(proj.parameters()).shape[-1]  # in-features of first Linear
    print(f"projector dtype: {next(proj.parameters()).dtype}, d_in={d}")

    # Reference: natural Mistral input-embedding distribution.
    embed_tokens = llm.model.embed_tokens
    V, dE = embed_tokens.weight.shape
    n_ref = min(args.n_token_emb, V)
    idx = torch.randperm(V, generator=g)[:n_ref]
    ref = embed_tokens.weight[idx].detach().to(torch.float32).cpu()

    # Real-anchor regime.
    real = None
    if args.input is not None and args.input.is_file():
        seeds = _load_real_anchors(args.input, args.num_prompts, args.seed_ratio)
        if seeds:
            print(f"\nEmbedding {len(seeds)} real anchors from {args.input} ...")
            real_in = _embed_texts(ret, ret_tok, seeds)  # cpu, float32
            with torch.no_grad():
                real = proj(real_in.to(torch.bfloat16).cuda()).float().cpu()
    else:
        if args.input is not None:
            print(f"!! --input not found: {args.input}", file=sys.stderr)

    # Random natural-scale.
    nat_in = torch.randn(args.n_natural, d, generator=g) * float(args.sigma_natural)
    with torch.no_grad():
        nat = proj(nat_in.to(torch.bfloat16).cuda()).float().cpu()

    # Random large-sigma.
    large_in = torch.randn(args.n_large, d, generator=g) * float(args.sigma_large)
    with torch.no_grad():
        large = proj(large_in.to(torch.bfloat16).cuda()).float().cpu()

    # ---- report ----
    print()
    print("=" * 72)
    print("INPUT side  (z fed into the projector)")
    print("=" * 72)
    if real is not None:
        _summarise("real anchors (SFR(seed))", real_in)
        print()
    _summarise(f"random natural-scale (sigma={args.sigma_natural})", nat_in)
    print()
    _summarise(f"random large-sigma (sigma={args.sigma_large})", large_in)

    print()
    print("=" * 72)
    print("OUTPUT side (projector(z) -- the vector LLM sees at the xRAG slot)")
    print("=" * 72)
    if real is not None:
        _summarise("projector(real anchors)", real)
        print()
    _summarise(f"projector(random natural-scale)", nat)
    print()
    _summarise(f"projector(random large-sigma)", large)

    print()
    print("=" * 72)
    print("REFERENCE: Mistral input-embedding matrix (one row per token)")
    print("=" * 72)
    _summarise(f"sampled rows of model.embed_tokens.weight (n={n_ref})", ref)


if __name__ == "__main__":
    main()
