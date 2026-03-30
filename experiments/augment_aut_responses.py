#!/usr/bin/env python
"""
Expand AUT (Alternative Uses Task) generations:

- Input:  JSON file with structure like:
    [
        {"item": "Umbrella", "uses": ["use1", "use2", ...], "Agent": "..."},
        ...
    ]

- For each UNIQUE item:
    * AGGREGATE all uses from all agents for that item
    * Use xRAG latent exploration to GENERATE new uses
      until the total length of `uses` is exactly target_n.
    * Output record has the structure:
        {"item", "uses", "agents"}  (len(uses) == target_n)

NOTE (kept 1:1 with your current behavior):
- Output "uses" contains ONLY newly generated uses (len == target_n).
- Original aggregated uses are stored in "seeds" for CSV/inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import csv
import random
import argparse

import torch
from transformers import AutoTokenizer

# --- xRAG / SFR imports (your code) ---
from xRAG.src.model import SFR, XMistralForCausalLM
from xRAG.src.language_modeling.utils import get_retrieval_embeds, XRAG_TOKEN


# ---------------------------
# Model context (no globals)
# ---------------------------
@dataclass(frozen=True)
class ModelContext:
    device: torch.device
    retriever_device: torch.device
    dtype_llm: torch.dtype
    dtype_retr: torch.dtype
    llm: XMistralForCausalLM
    llm_tokenizer: Any
    retriever: SFR
    retriever_tokenizer: Any


def initialize_models() -> ModelContext:
    """Initialize dtypes and models with automatic device mapping (same behavior as before)."""
    # Determine dtypes based on available hardware
    if torch.cuda.is_available():
        dtype_llm = torch.bfloat16
        dtype_retr = torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dtype_llm = torch.float16
        dtype_retr = torch.float16
    else:  # cpu
        dtype_llm = torch.float32
        dtype_retr = torch.float32

    print(f"LLM dtype: {dtype_llm}, retriever dtype: {dtype_retr}")

    llm_name_or_path = "Hannibal046/xrag-7b"
    retriever_name_or_path = "Salesforce/SFR-Embedding-Mistral"

    print("Loading LLM with device_map='auto'...")
    llm = XMistralForCausalLM.from_pretrained(
        llm_name_or_path,
        torch_dtype=dtype_llm,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name_or_path,
        add_eos_token=False,
        use_fast=False,
        padding_side="left",
    )

    # XRAG placeholder token id
    llm.set_xrag_token_id(llm_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

    # Monkey-patch prepare_inputs_embeds to ensure device alignment (same as original)
    def patched_prepare_inputs_embeds(input_ids, retrieval_embeds):
        inputs_embeds = llm.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, llm.retriever_hidden_size)

        num_xrag_tokens = torch.sum(input_ids == llm.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)

        retrieval_embeds = llm.projector(retrieval_embeds.to(inputs_embeds.dtype))
        retrieval_embeds = retrieval_embeds.to(inputs_embeds.device)
        inputs_embeds[input_ids == llm.xrag_token_id] = retrieval_embeds
        return inputs_embeds

    llm.prepare_inputs_embeds = patched_prepare_inputs_embeds

    # Determine device for input tensors based on model's device map (same logic as original)
    if hasattr(llm, "hf_device_map") and llm.hf_device_map:
        embed_layer_names = ["model.embed_tokens", "embed_tokens"]
        device = None
        for name in embed_layer_names:
            if name in llm.hf_device_map:
                device = torch.device(llm.hf_device_map[name])
                break
        if device is None:
            try:
                device = next(llm.model.embed_tokens.parameters()).device
            except Exception:
                device = torch.device(next(iter(llm.hf_device_map.values())))
    else:
        try:
            device = next(llm.model.embed_tokens.parameters()).device
        except Exception:
            device = llm.device if hasattr(llm, "device") else torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

    # Optional warning about projector device (kept behavior)
    if hasattr(llm, "projector") and llm.projector is not None:
        try:
            projector_device = next(llm.projector.parameters()).device
            if projector_device != device:
                print(f"Warning: Projector is on {projector_device} but embedding layer is on {device}")
        except Exception:
            pass

    print(f"Using device for input tensors (embedding layer): {device}")

    print("Loading retriever with device_map='auto'...")
    retriever = SFR.from_pretrained(
        retriever_name_or_path,
        torch_dtype=dtype_retr,
        device_map="auto",
    ).eval()

    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

    # Determine device for retriever input tensors (same logic as original)
    if hasattr(retriever, "hf_device_map") and retriever.hf_device_map:
        retriever_device = torch.device(next(iter(retriever.hf_device_map.values())))
    else:
        retriever_device = retriever.device if hasattr(retriever, "device") else device

    print(f"Using device for retriever input tensors: {retriever_device}")
    print("LLM, retriever and tokenizers loaded.")

    return ModelContext(
        device=device,
        retriever_device=retriever_device,
        dtype_llm=dtype_llm,
        dtype_retr=dtype_retr,
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        retriever=retriever,
        retriever_tokenizer=retriever_tokenizer,
    )


# ---------------------------
# Core helpers (no HTTP)
# ---------------------------
rag_template = """[INST] Background: {document}

Question: {prompt} [/INST] The answer is:"""


def get_aut_prompt(item: str) -> str:
    return (
        f"Generate exactly ONE creative and unusual use for a {item}. "
        f"Write it as a single idea in the following format:\n\n"
        f"Short, descriptive title for the use, then a colon, then 1–3 full sentences "
        f"explaining the idea in detail.\n\n"
    )


def embed_text(ctx: ModelContext, documents: List[str]) -> torch.Tensor:
    """Return retrieval embeddings tensor shape [B, D] for the given documents."""
    with torch.no_grad():
        toks = ctx.retriever_tokenizer(
            documents,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        embs = get_retrieval_embeds(
            ctx.retriever,
            input_ids=toks["input_ids"].to(ctx.retriever_device),
            attention_mask=toks["attention_mask"].to(ctx.retriever_device),
        )
    return embs


def generate_from_embedding(ctx: ModelContext, embedding: torch.Tensor, prompt: str) -> str:
    """Use xRAG-7B with a single retrieval embedding to generate text."""
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)  # [1, D]

    embedding = embedding.to(ctx.device)

    formatted_prompt = rag_template.format_map(dict(document=XRAG_TOKEN, prompt=prompt))
    encoded = ctx.llm_tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(ctx.device)
    attention_mask = encoded.attention_mask.to(ctx.device)

    with torch.no_grad():
        out = ctx.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=200,
            pad_token_id=ctx.llm_tokenizer.pad_token_id,
            retrieval_embeds=embedding,  # [1, D]
        )
    text = ctx.llm_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return text.split("The answer is:", 1)[-1].strip()


def style_with_llm(ctx: ModelContext, fmt_example: str, idea: str, prompt: str, item: str) -> str:
    """Rewrite `idea` to match the strict AUT response format (same as original)."""
    sys_msg = """
You are a strict editing assistant that rewrites the Response so it fully obeys the Prompt.

Priority:
1. Obey the Prompt exactly (format, length, style).
2. Be clear and concise.
3. Reuse good ideas from the original Response only if they fit the Prompt.

Rules:
- Output must be in format: "Title: Description" (1-3 sentences explaining the idea)
- The idea must be a creative/unusual use for the specified item
- Do NOT add extra commentary. Output only the final answer.
- Keep the response focused on a SINGLE creative use
"""
    user_msg = f"""
Your goal is to produce the best possible answer to the Prompt.
You may treat the Original Response as a noisy draft: reuse only what helps.

Prompt:
{prompt}

Item: {item}

Original Response:
{idea}

Example format:
{fmt_example}

Refined Response:
"""
    formatted_prompt = f"[INST] {sys_msg}\n\n{user_msg} [/INST]"

    encoded = ctx.llm_tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(ctx.device)
    attention_mask = encoded.attention_mask.to(ctx.device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        out = ctx.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=200,
            pad_token_id=ctx.llm_tokenizer.pad_token_id,
        )

    generated_ids = out[0][input_length:]
    return ctx.llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ==========================
# Latent exploration helpers
# ==========================
def _interpolate(v1: Sequence[float], v2: Sequence[float], lam: float) -> List[float]:
    return [lam * a + (1 - lam) * b for a, b in zip(v1, v2)]


def explore(
    seeds: List[List[float]],
    k: int,
    sigma: float = 0.05,
    lam_value: float | None = None,
) -> List[List[float]]:
    """
    Simple midpoint-ish interpolation among seed embeddings to produce k new latent vectors.

    NOTE: sigma is intentionally unused to preserve *exact* behavior of the original script.
    (It was passed around but never applied.)
    """
    _ = sigma  # keep 1:1 behavior; do not affect RNG / outputs

    if not seeds or k <= 0:
        return []

    indices = list(range(len(seeds)))
    if len(indices) == 1:
        pairs = [(0, 0)]
    else:
        pairs = list(combinations(indices, 2))

    out: List[List[float]] = []
    while len(out) < k:
        i, j = random.choice(pairs)
        lam = lam_value if lam_value is not None else random.uniform(6, 10) * random.choice([-1, 1])
        out.append(_interpolate(seeds[i], seeds[j], lam))
    return out


# ==============================================
# Aggregate records by item
# ==============================================
def aggregate_by_item(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate all uses from different agents for each unique item.

    Input:  {"item": ..., "uses": [...], "Agent": ...}
    Output: {"item": ..., "uses": [...], "agents": [...]}
    """
    item_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"uses": [], "agents": set()})

    for rec in records:
        item = rec.get("item", "unknown")
        uses = rec.get("uses", [])
        agent = rec.get("Agent", "unknown-agent")

        item_data[item]["uses"].extend(uses)
        item_data[item]["agents"].add(agent)

    aggregated = []
    for item, data in item_data.items():
        aggregated.append(
            {
                "item": item,
                "uses": data["uses"],
                "agents": sorted(list(data["agents"])),
            }
        )
    return aggregated


# ==============================================
# Expand one record: use all seeds, generate target_n new uses
# ==============================================
def expand_record_uses(
    ctx: ModelContext,
    rec: Dict[str, Any],
    target_n: int = 10,
    sigma: float = 0.05,
    use_style_normalization: bool = True,
    lam_value: float | None = None,
) -> Dict[str, Any]:
    """
    Take a single aggregated input record:
      {"item": ..., "uses": [...], "agents": [...]}

    - Use ALL existing uses as seeds for embedding exploration.
    - Generate exactly target_n NEW uses (via xRAG latent interpolation).
    - Output "uses" contains ONLY the newly generated uses (seeds not included).
    """
    item = rec["item"]
    agents = rec.get("agents", ["unknown-agent"])
    uses_in: List[str] = rec.get("uses", [])

    prompt = get_aut_prompt(item)

    if not uses_in:
        uses_in = [f"Creative use for {item}"]

    seeds_text = uses_in

    with torch.no_grad():
        seed_vecs_t = embed_text(ctx, seeds_text)
    seed_vecs = seed_vecs_t.detach().to("cpu").tolist()

    fmt_example = seeds_text[0] if seeds_text else f"Example Title: A creative description of using {item} in an unusual way."
    generated: List[str] = []

    gen_num = 0
    while len(generated) < target_n:
        gen_num += 1
        new_vecs = explore(seed_vecs, k=1, sigma=sigma, lam_value=lam_value)
        if not new_vecs:
            break

        vec = new_vecs[0]
        emb = torch.tensor(vec, device=ctx.device, dtype=ctx.dtype_retr)

        raw_idea = generate_from_embedding(ctx, emb, prompt)
        final_idea = (
            style_with_llm(ctx, fmt_example, raw_idea, prompt, item)
            if (use_style_normalization and fmt_example)
            else raw_idea
        )
        final_idea = final_idea.strip()
        generated.append(final_idea)

        # Re-encode the styled output and use it as a new seed embedding (same behavior)
        with torch.no_grad():
            new_embedding = embed_text(ctx, [final_idea])
            seed_vecs.append(new_embedding.detach().to("cpu").tolist()[0])

        print(f"  Generated {gen_num}/{target_n} for item '{item}'")

    uses_out = generated[:target_n]
    return {
        "item": item,
        "uses": uses_out,
        "seeds": seeds_text,
        "agents": agents,
        "num_seeds_used": len(seeds_text),
    }


# ==============================================
# CSV output helper
# ==============================================
def write_csv(output_records: List[Dict[str, Any]], csv_path: Path) -> None:
    """Write output records to CSV with seeds (gen_no=0) and generated items (gen_no=1..)."""
    rows: List[Dict[str, Any]] = []
    ts = datetime.utcnow().isoformat(timespec="seconds")

    for rec in output_records:
        item = rec["item"]
        seeds = rec.get("seeds", [])
        uses = rec.get("uses", [])

        for seed in seeds:
            rows.append({"timestamp": ts, "item": item, "idea": seed, "gen_no": 0, "source": "seed"})

        for idx, use in enumerate(uses, start=1):
            rows.append({"timestamp": ts, "item": item, "idea": use, "gen_no": idx, "source": "generated"})

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "item", "idea", "gen_no", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Wrote {len(rows)} rows to CSV: {csv_path}")


# ==============================================
# End-to-end processing: JSON → JSON
# ==============================================
def process_json(
    ctx: ModelContext,
    input_path: Path,
    output_path: Path,
    target_n: int = 10,
    seed: int | None = 42,
    sigma: float = 0.05,
    use_style_normalization: bool = True,
    lam_value: float | None = None,
    csv_output: Path | None = None,
) -> None:
    """
    Read input JSON, aggregate by item, expand each aggregated record, write output JSON.
    All existing uses are used as seeds, and exactly target_n NEW uses are generated per item.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    with input_path.open("r", encoding="utf-8") as fin:
        records = json.load(fin)

    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of records")

    n_in = len(records)
    print(f"Read {n_in} records from input file")

    aggregated_records = aggregate_by_item(records)
    n_items = len(aggregated_records)

    print(f"Aggregated into {n_items} unique items")
    for rec in aggregated_records:
        print(f"  - {rec['item']}: {len(rec['uses'])} uses from {len(rec['agents'])} agents")
    print()

    output_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(aggregated_records):
        item = rec.get("item", "unknown")
        n_uses = len(rec.get("uses", []))
        n_agents = len(rec.get("agents", []))
        print(f"\nProcessing item {idx + 1}/{n_items}: '{item}' ({n_uses} uses from {n_agents} agents)")

        expanded = expand_record_uses(
            ctx,
            rec,
            target_n=target_n,
            sigma=sigma,
            use_style_normalization=use_style_normalization,
            lam_value=lam_value,
        )
        output_records.append(expanded)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{n_items} items...", flush=True)

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(output_records, fout, ensure_ascii=False, indent=4)

    print(f"\nDone. Input records: {n_in}, unique items: {n_items}, output records: {len(output_records)}")

    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_records, csv_output)


# ==============================================
# Main
# ==============================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand AUT uses using xRAG latent exploration. "
        "Aggregates all uses from different agents for each unique item before expanding."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument(
        "--target-n",
        type=int,
        required=True,
        help="Target number of NEW uses to generate per item (seeds not included in output)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Sigma parameter for exploration (kept for compatibility; has no effect in original logic)",
    )
    parser.add_argument(
        "--use-style-normalization",
        action="store_true",
        help="Enable style normalization (default: False)",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=None,
        help="Optional fixed lambda for latent interpolation (default: random)",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional CSV output path (writes seeds and generated ideas).",
    )

    args = parser.parse_args()

    # IMPORTANT: same call order as original (init models first, then seed inside process_json)
    ctx = initialize_models()

    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_output_path = Path(args.csv_output) if args.csv_output else None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    if csv_output_path:
        print(f"  CSV Output: {csv_output_path}")
    print(f"  Target NEW uses per item: {args.target_n}")
    print(f"  Random seed: {args.seed}")
    print(f"  Style normalization: {args.use_style_normalization}")
    print(f"  Lambda value: {args.lambda_value}")
    print("  NOTE: All uses from all agents will be used as seeds (but NOT included in output)")
    print()

    process_json(
        ctx=ctx,
        input_path=input_path,
        output_path=output_path,
        target_n=args.target_n,
        seed=args.seed,
        sigma=args.sigma,
        use_style_normalization=args.use_style_normalization,
        lam_value=args.lambda_value,
        csv_output=csv_output_path,
    )


if __name__ == "__main__":
    main()
