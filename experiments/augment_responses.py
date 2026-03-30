#!/usr/bin/env python
"""
Expand wildchat generations (xRAG latent exploration).

- Input: JSONL with lines like
    {"id": "...", "prompt": "...", "model": "...", "generations": ["...", ...]}

- For each record:
    * Keep the first ~seed_ratio of original generations as "seeds" (exactly like the original script).
    * Use xRAG latent exploration to generate new generations
      until the total length of `generations` is exactly target_n.
    * Output record has the same structure:
        {"id", "prompt", "model", "generations"}  (len(generations) == target_n)

Notes on refactor constraints:
- Logic/output is preserved 1:1 with the provided script.
- Only truly unused parts were removed (e.g., used_pairs, pandas/datetime, unused diverse-seed selection code path).
- Seeding behavior is identical (random.seed + torch.manual_seed only).
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence
from itertools import combinations
import json
import random
import argparse

import torch
from transformers import AutoTokenizer

# --- xRAG / SFR imports (your code) ---
from xRAG.src.model import SFR, XMistralForCausalLM
from xRAG.src.language_modeling.utils import get_retrieval_embeds, XRAG_TOKEN


# ---------------------------
# Globals (initialized in initialize_models)
# ---------------------------
device = None
retriever_device = None
_dtype_llm = None
_dtype_retr = None

llm = None
llm_tokenizer = None
retriever = None
retriever_tokenizer = None


def initialize_models() -> None:
    """Initialize device, dtypes, and models with automatic device mapping."""
    global device, retriever_device, _dtype_llm, _dtype_retr
    global llm, llm_tokenizer, retriever, retriever_tokenizer

    # Determine dtypes based on available hardware (unchanged)
    if torch.cuda.is_available():
        _dtype_llm = torch.bfloat16
        _dtype_retr = torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _dtype_llm = torch.float16
        _dtype_retr = torch.float16
    else:  # cpu
        _dtype_llm = torch.float32
        _dtype_retr = torch.float32

    print(f"LLM dtype: {_dtype_llm}, retriever dtype: {_dtype_retr}")

    llm_name_or_path = "Hannibal046/xrag-7b"
    retriever_name_or_path = "Salesforce/SFR-Embedding-Mistral"

    # ---------------------------
    # Load LLM
    # ---------------------------
    print("Loading LLM with device_map='auto'...")
    llm = XMistralForCausalLM.from_pretrained(
        llm_name_or_path,
        torch_dtype=_dtype_llm,
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

    # Monkey-patch prepare_inputs_embeds to ensure device alignment (unchanged)
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

    # Determine device for input tensors based on model's device map (unchanged)
    if hasattr(llm, "hf_device_map") and llm.hf_device_map:
        embed_layer_names = ["model.embed_tokens", "embed_tokens"]
        device_candidate = None
        for name in embed_layer_names:
            if name in llm.hf_device_map:
                device_candidate = torch.device(llm.hf_device_map[name])
                break
        if device_candidate is None:
            try:
                device_candidate = next(llm.model.embed_tokens.parameters()).device
            except Exception:
                device_candidate = torch.device(next(iter(llm.hf_device_map.values())))
        device = device_candidate
    else:
        try:
            device = next(llm.model.embed_tokens.parameters()).device
        except Exception:
            device = llm.device if hasattr(llm, "device") else torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

    # Inform if projector differs (unchanged, informational only)
    if hasattr(llm, "projector") and llm.projector is not None:
        try:
            projector_device = next(llm.projector.parameters()).device
            if projector_device != device:
                print(f"Warning: Projector is on {projector_device} but embedding layer is on {device}")
        except Exception:
            pass

    print(f"Using device for input tensors (embedding layer): {device}")

    # ---------------------------
    # Load retriever
    # ---------------------------
    print("Loading retriever with device_map='auto'...")
    retriever = SFR.from_pretrained(
        retriever_name_or_path,
        torch_dtype=_dtype_retr,
        device_map="auto",
    ).eval()

    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

    # Determine device for retriever input tensors (unchanged)
    if hasattr(retriever, "hf_device_map") and retriever.hf_device_map:
        retriever_device = torch.device(next(iter(retriever.hf_device_map.values())))
    else:
        retriever_device = retriever.device if hasattr(retriever, "device") else device

    print(f"Using device for retriever input tensors: {retriever_device}")
    print("LLM, retriever and tokenizers loaded.")


# ---------------------------
# Core helpers (no HTTP)
# ---------------------------
rag_template = """[INST] Background: {document}

Question: {prompt} [/INST] The answer is:"""


def embed_text(documents: List[str]) -> torch.Tensor:
    """Return retrieval embeddings tensor shape [B, D] for the given documents."""
    with torch.no_grad():
        toks = retriever_tokenizer(
            documents,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        embs = get_retrieval_embeds(
            retriever,
            input_ids=toks["input_ids"].to(retriever_device),
            attention_mask=toks["attention_mask"].to(retriever_device),
        )  # [B, D]
    return embs


def generate_from_embedding(embedding: torch.Tensor, prompt: str) -> str:
    """
    Use xRAG-7B with a single retrieval embedding to generate text.
    `embedding`: [D] or [1, D] (torch.Tensor on any device).
    """
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)  # [1, D]

    # Ensure embedding is on the correct device for the model (unchanged)
    embedding = embedding.to(device)

    formatted_prompt = rag_template.format_map(dict(document=XRAG_TOKEN, prompt=prompt))
    encoded = llm_tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=200,
            pad_token_id=llm_tokenizer.pad_token_id,
            retrieval_embeds=embedding,  # [1, D]
        )
    text = llm_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return text.split("The answer is:", 1)[-1].strip()


def style_with_llm(idea: str, prompt: str) -> str:
    """
    Rewrite `idea` to obey `prompt` strictly (no retrieval).
    (This preserves the original behavior; fmt_example was never used.)
    """
    sys_msg = """
You are a strict editing assistant that rewrites the Response so it fully obeys the Prompt.

Priority:
1. Obey the Prompt exactly (format, length, “one X”, “exactly N” etc.).
2. Be clear and concise.
3. Reuse good ideas from the original Response only if they fit the Prompt.

Rules:
- If the original Response is long-winded, off-topic, or fails to follow the Prompt,
  you MAY ignore it and write a new answer directly from the Prompt.
- If the Prompt asks for ONE item (one person, one digit, one job, one book etc.),
  output ONLY that item, with no explanation, no list, no extra text.
- If the Prompt specifies a length/format (e.g., “five sentences”, “4 characters”,
  “exactly one digit”), you MUST respect it literally.
- Do NOT add extra commentary. Output only the final answer.
"""

    user_msg = f"""
Your goal is to produce the best possible answer to the Prompt.
You may treat the Original Response as a noisy draft: reuse only what helps.

Prompt:
{prompt}

Original Response:
{idea}

Refined Response:
"""
    formatted_prompt = f"[INST] {sys_msg}\n\n{user_msg} [/INST]"

    encoded = llm_tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=200,
            pad_token_id=llm_tokenizer.pad_token_id,
        )

    generated_ids = out[0][input_length:]
    return llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ==========================
# Latent exploration helpers
# ==========================
def _interpolate(v1: Sequence[float], v2: Sequence[float], lam: float) -> List[float]:
    return [lam * a + (1 - lam) * b for a, b in zip(v1, v2)]


def explore(
    seeds: List[List[float]],
    k: int,
    sigma: float = 0.05,          # kept for API compatibility; still unused exactly like original
    lam_value: float | None = None,
) -> List[List[float]]:
    """
    Simple interpolation among seed embeddings to produce k new latent vectors.
    (Preserves original behavior exactly.)
    """
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
# Expand one record
# ==============================================
def expand_record_generations(
    rec: Dict[str, Any],
    target_n: int = 10,
    seed_ratio: float = 0.3,
    sigma: float = 0.05,
    use_style_normalization: bool = True,
    lam_value: float | None = None,
) -> Dict[str, Any]:
    """
    Preserve original behavior:
    - If generations empty: use [prompt].
    - n_seed = max(1, int(len(gens_in) * seed_ratio)) (or 1 if empty input).
    - Seeds are the FIRST n_seed generations (not diversity-selected).
    - Generate until total == target_n.
    - Optionally style-normalize, but only if use_style_normalization and first seed is truthy.
    - Re-encode each final_idea and append its embedding to seed_vecs (for subsequent exploration).
    """
    prompt = rec["prompt"]
    model = rec.get("model", "unknown-model")
    gens_in: List[str] = rec.get("generations", [])

    n_seed = max(1, int(len(gens_in) * seed_ratio)) if gens_in else 1

    if not gens_in:
        gens_in = [prompt]

    # Seeds: first n_seed generations (matches original script’s effective behavior)
    seeds_text = gens_in[:n_seed]
    if not seeds_text:
        seeds_text = [prompt]

    with torch.no_grad():
        seed_vecs_t = embed_text(seeds_text)  # [S, D]
    seed_vecs = seed_vecs_t.detach().to("cpu").tolist()

    fmt_example = seeds_text[0]  # gating only (matches original)
    generated: List[str] = []

    gen_num = 0
    while len(seeds_text) + len(generated) < target_n:
        gen_num += 1

        new_vecs = explore(seed_vecs, k=1, sigma=sigma, lam_value=lam_value)
        if not new_vecs:
            break

        vec = new_vecs[0]
        emb = torch.tensor(vec, device=device, dtype=_dtype_retr)
        raw_idea = generate_from_embedding(emb, prompt)

        if use_style_normalization and fmt_example:
            final_idea = style_with_llm(raw_idea, prompt)
        else:
            final_idea = raw_idea

        final_idea = final_idea.strip()
        generated.append(final_idea)

        # Re-encode styled output and append as new seed embedding (matches original)
        with torch.no_grad():
            new_embedding = embed_text([final_idea])  # [1, D]
            new_embedding_list = new_embedding.detach().to("cpu").tolist()[0]
            seed_vecs.append(new_embedding_list)

    generations_out = (seeds_text + generated)[:target_n]

    return {
        "id": rec.get("id"),
        "prompt": prompt,
        "model": model,
        "generations": generations_out,
    }


# ==============================================
# Helper: determine target_n from source file
# ==============================================
def get_target_n_from_source(input_path: Path) -> int:
    """Read the first non-empty record from input file to determine target_n."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            return len(rec.get("generations", []))

    raise ValueError(f"No valid records found in {input_path}")


# ==============================================
# End-to-end processing: JSONL → JSONL
# ==============================================
def process_jsonl(
    input_path: Path,
    output_path: Path,
    target_n: int = 10,
    seed_ratio: float = 0.3,
    seed: int | None = 42,
    sigma: float = 0.05,
    use_style_normalization: bool = True,
    lam_value: float | None = None,
) -> None:
    """
    Read input JSONL, expand each record, write output JSONL.
    Seeding behavior preserved exactly.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    n_in = 0
    n_out = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            rec = json.loads(line)

            expanded = expand_record_generations(
                rec,
                target_n=target_n,
                seed_ratio=seed_ratio,
                sigma=sigma,
                use_style_normalization=use_style_normalization,
                lam_value=lam_value,
            )

            fout.write(json.dumps(expanded, ensure_ascii=False) + "\n")
            n_out += 1

            if n_in % 10 == 0:
                print(f"Processed {n_in} records...", flush=True)

    print(f"Done. Input records: {n_in}, output records: {n_out}")


# ==============================================
# Main
# ==============================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Expand generations using xRAG latent exploration")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed-ratio",
        type=float,
        default=0.3,
        help="Fraction of original generations to keep as seeds (default: 0.3)",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Target number of generations (default: auto-detect from source file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Sigma parameter for exploration (default: 0.05)",
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

    args = parser.parse_args()

    initialize_models()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.target_n is None:
        target_n = get_target_n_from_source(input_path)
        print(f"Auto-detected target_n={target_n} from source file")
    else:
        target_n = args.target_n

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target generations: {target_n}")
    print(f"  Seed ratio: {args.seed_ratio} (will keep ~{int(target_n * args.seed_ratio)} seeds)")
    print(f"  Random seed: {args.seed}")
    print(f"  Style normalization: {args.use_style_normalization}")
    print()

    process_jsonl(
        input_path=input_path,
        output_path=output_path,
        target_n=target_n,
        seed_ratio=args.seed_ratio,
        seed=args.seed,
        sigma=args.sigma,
        use_style_normalization=args.use_style_normalization,
        lam_value=args.lambda_value,
    )


if __name__ == "__main__":
    main()
