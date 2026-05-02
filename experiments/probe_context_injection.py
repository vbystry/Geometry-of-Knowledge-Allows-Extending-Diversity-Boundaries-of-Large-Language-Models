"""Qualitative case study: arbitrary text context shifts content but preserves task.

Motivates the post-hoc reading of the random-latent ablation: feeding the LLM
an arbitrary, unrelated chunk of text alongside the user prompt produces
outputs that still satisfy the task but absorb traces of the injected
context. The proposed continuous-conditioning channel achieves the same
effect via the xRAG projector slot, so isotropic random latents already
break diversity collapse without any structured ``manifold'' interpretation
-- they are simply the embedding-space analogue of injecting an arbitrary
context. The geometric anchor manifold provides directional control over
which arbitrary context is injected and is therefore useful for steered
exploration tasks (optimisation, iterative refinement) rather than for raw
diversity coverage on NoveltyBench.

Usage::

    python experiments/probe_context_injection.py \\
        --input results/curated/g2_theta0.3_temp1_iter15/generations.jsonl \\
        --num-prompts 5 --num-samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PANCAKE_RECIPE = """\
Classic pancake recipe.

Ingredients (makes about 8 pancakes):
- 1 1/2 cups all-purpose flour
- 3 tablespoons sugar
- 1 tablespoon baking powder
- 1/2 teaspoon salt
- 1 1/4 cups whole milk
- 1 large egg, lightly beaten
- 3 tablespoons unsalted butter, melted, plus more for the pan
- 1 teaspoon vanilla extract

Method. Whisk the flour, sugar, baking powder, and salt in a large bowl.
In a separate bowl, whisk the milk, egg, melted butter, and vanilla.
Pour the wet ingredients into the dry ingredients and stir gently with a
spatula until the batter is just combined; a few small lumps are fine.
Rest the batter for 5 minutes while you heat a non-stick pan over
medium-low heat with a thin film of butter. Pour 1/4 cup of batter per
pancake. Cook for about 2 minutes, until bubbles form across the surface
and the edges look set, then flip and cook for another minute until
golden brown. Serve immediately with butter and maple syrup."""


SECOND_INJECTION = """\
Below is a chunk of dense technical content unrelated to the user's
question, included only to demonstrate how the language model's output is
shaped by additional context: ''A ductile cast-iron pipe with a nominal
diameter of 200 mm and a wall thickness of 6.4 mm is laid in a trench
backfilled with compacted granular bedding to a depth of 1.5 m. The
allowable hoop stress is 240 MPa and the safety factor against burst is
2.0. Soil pressure is computed using Marston's formula assuming a
trench-condition coefficient C_d of 1.4 and a unit weight gamma of
18 kN/m^3.''"""


def _format_prompt(question: str, system_context: str | None) -> str:
    if system_context:
        return (
            f"[INST] Background context (use only if relevant):\n"
            f"{system_context.strip()}\n\n"
            f"Question: {question.strip()} [/INST]"
        )
    return f"[INST] {question.strip()} [/INST]"


def _generate(model, tokenizer, full_prompt: str, max_new_tokens: int = 400,
              temperature: float = 0.8, top_p: float = 0.95) -> str:
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def _load_prompts(path: Path, n: int) -> list[str]:
    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            p = rec.get("prompt") or rec.get("question") or rec.get("input")
            if isinstance(p, str) and p.strip():
                out.append(p.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mistralai/Mistral-7B-Instruct-v0.2",
                    help="Base Mistral-7B-Instruct (not the xRAG fine-tune).")
    ap.add_argument("--input", type=Path, required=True,
                    help="NoveltyBench generations.jsonl to draw prompts from.")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=3,
                    help="Generations per (prompt, condition) cell.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"!! --input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    torch.manual_seed(args.seed)

    print(f"Loading {args.llm} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    prompts = _load_prompts(args.input, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.\n")

    contexts = [
        ("vanilla",        None),
        ("pancake_recipe", PANCAKE_RECIPE),
        ("pipe_engineering", SECOND_INJECTION),
    ]

    for pi, q in enumerate(prompts):
        print("=" * 72)
        print(f"PROMPT {pi+1}: {q}")
        print("=" * 72)

        for cname, cctx in contexts:
            full = _format_prompt(q, cctx)
            for s in range(args.num_samples):
                print(f"\n--- [{cname}] sample {s+1}/{args.num_samples} ---")
                gen = _generate(
                    model, tokenizer, full,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(gen)
        print()


if __name__ == "__main__":
    main()
