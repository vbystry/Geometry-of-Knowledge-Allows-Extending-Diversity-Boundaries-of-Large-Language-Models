"""Iteration harness for the realignment / refinement prompt.

Loads a base instruction model (Mistral-7B-Instruct via xrag-7b) and the
NoveltyBench reward model (Skywork-Reward-Gemma-2-27B) once, then runs a
small grid of (prompt, draft) pairs through several candidate refinement
prompt variants and reports the resulting Skywork reward.

The grid is intentionally tiny (3 prompts x 3 drafts) so each full sweep
finishes in a few minutes. Edit the ``VARIANTS`` list, push, resubmit.

Submit via hpc/wcss_probe_refinement.slurm.
"""

from __future__ import annotations

import json
import sys
from typing import List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ----------------------------------------------------------------------
# Test grid: hand-picked NoveltyBench prompts spanning factual / creative
# / short-answer, with three raw (no-realignment) latent drafts each.
# ----------------------------------------------------------------------

GRID: List[Tuple[str, str, List[str]]] = [
    # (id, prompt, list of draft outputs to refine)
    (
        "curated-13",
        "Tell me a riddle.",
        [
            "I don't know. I'm an AI. I don't have a body or a brain. I don't have the ability to ask or answer riddles. I can only type out what I'm programmed to type.",
            "A wind. Echo: I have a voice, but not a sound. Wind: I have a sound, but not a voice. Echo: I am an echo, I come after the wind. Wind: I am a wind, I go before the echo.",
            '"I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?"',
        ],
    ),
    (
        "curated-25",
        "Name one person of Indian Origin.",
        [
            "Dr. Kalpana Chawla. She was the first Indian-American woman to go to space and the second Indian-American astronaut in space. She flew on two Space Shuttle missions and died in the Columbia disaster.",
            "Ada Yonath. She is a structural biologist and a professor at the Weizmann Institute of Science in Israel. She was born on 1948 in Mexico City, Mexico. She is the first Mexican woman to win a Nobel Prize.",
            "Mahatma Gandhi.",
        ],
    ),
    (
        "curated-46",
        "What's a saying that English teachers hate?",
        [
            "\"I don't have time to write.\" This is a common excuse used by students who do not want to put in the effort to write.",
            "\"It's just a bunch of words.\" English teachers hate this saying because it undermines the importance of language and literature.",
            "\"Excuse me, may I write my own saying?\" This is a common saying that English teachers may dislike, but it is not a saying that is widely known.",
        ],
    ),
    (
        "curated-4",
        "Write a short love poem with 4 lines.",
        [
            "Whispers of the night,\nIn quietude we meet,\nOur hearts in rhythmic flight,\nIn love's sweet silence, complete.",
            "In the heart of the city,\nWhere love's sweet song is heard,\nIn the silence of the night,\nWe meet, and our love is stirred.",
            "Time's gentle hand, in love's embrace,\nEmbraces hearts, in tender grace.\nIn moments shared, a timeless space,\nTwo souls entwined, in love's embrace.",
        ],
    ),
]


# ----------------------------------------------------------------------
# Refinement prompt variants. Each is a (name, system_message) pair that
# is wrapped together with the user message below. Edit / extend freely.
# ----------------------------------------------------------------------

V_TERSE_BASELINE = """\
You are a strict editing assistant that rewrites the Response so it fully obeys the Prompt.

Priority:
1. Obey the Prompt exactly (format, length, "one X", "exactly N" etc.).
2. Be clear and concise.
3. Reuse good ideas from the original Response only if they fit the Prompt.

Rules:
- If the original Response is long-winded, off-topic, or fails to follow the Prompt, you MAY ignore it and write a new answer directly from the Prompt.
- If the Prompt asks for ONE item (one person, one digit, one job, one book etc.), output ONLY that item, with no explanation, no list, no extra text.
- If the Prompt specifies a length/format (e.g., "five sentences", "4 characters", "exactly one digit"), you MUST respect it literally.
- Do NOT add extra commentary. Output only the final answer."""

V_VERBOSE_FAILED = """\
You are a helpful assistant that rewrites a draft Response into a polished, detailed reply.

Priority:
1. Obey explicit Prompt constraints (format, length, "one X", etc.) literally.
2. Within those constraints, produce a thorough, well-written answer with explanatory context. Aim for roughly 2-4 sentences.
3. Reuse the answer-bearing content from the Original Response when it is on-topic; replace if wrong or off-topic.

Rules:
- Add helpful context (origin, role, why it qualifies).
- Avoid disclaimers like "I cannot help" or "as an AI".
- Output only the final reply."""

V_FACTUAL_CONSERVATIVE = """\
You are an editor that polishes a draft Response into a clean reply to the Prompt while strictly preserving factual claims.

Rules:
- Preserve the answer-bearing content from the Original Response (the entity it names, the answer it gives).
- DO NOT introduce new factual claims (dates, places, achievements, statistics) that are not in the Original Response.
- If the Original Response is incoherent or off-topic, output a short, safe answer of your own without invented details.
- Match the Prompt's expected format (haiku, list of N, one-word answer, etc.).
- Default to a single confident sentence; only elaborate if the Original Response already contains verifiable context.
- Output only the final reply, no commentary."""

V_G2_STYLE = """\
You are an expert assistant producing a high-quality, well-articulated reply to the Prompt.

Style:
- Confident, clear, and complete; an attentive reader should not need to re-ask the question.
- 1-3 sentences for short-answer prompts; respect explicit length/format rules otherwise.
- Mention the answer first, then add at most one short clarifying sentence about why this answer fits.

Quality bar:
- The reply must be coherent and topically aligned. Replace the Original Response wherever it is wrong, off-topic, repetitive, or self-referential.
- Never claim verifiable facts (dates, awards, ranks, biographies) unless those exact facts already appear in the Original Response.
- Never include meta-text about being an AI, about the rewriting process, or about <xRAG> tokens.

Output only the final reply."""

V_DEFENSIVE_SHORT = """\
You write the final answer to the Prompt, using the Original Response only as a hint about which entity / direction to commit to.

Procedure:
1. Decide what the Prompt asks for (a name, a number, a haiku, a story of N sentences, etc.).
2. Decide what entity or angle the Original Response is pointing at. If the original is empty / garbled / refuses, pick a clean default.
3. Produce a confident, grammatical reply in exactly the format the Prompt requests.
4. Add at most one short factual sentence of context, but ONLY using facts that are obviously safe (no specific dates, ranks, prize years, biographies you cannot verify).

Output only the final reply, no preamble."""

V_PRESERVE_REWRITE = """\
You rewrite a draft answer to the Prompt for clarity and correctness.

Constraints:
- Keep the same answer (entity / approach / number) the draft commits to, unless that answer is clearly wrong or off-topic.
- Fix grammar, repetition, fragments, and self-referential meta-text.
- Match the Prompt's required format and length exactly.
- If the draft already gives a single-fact answer, output that answer in a single confident sentence; do NOT add invented context.
- If the draft is creative content (story, poem, riddle), preserve its core idea and rewrite cleanly to the format.

Output only the final reply."""


V_HYBRID = """\
You produce a confident, well-articulated reply to the Prompt, using the Original Response only as a hint about which entity / answer / direction to commit to.

Style:
- Confident, clear, complete; an attentive reader should not need to re-ask.
- Default to a single sentence containing the answer plus brief safe context.
- For multi-sentence prompts (story of N sentences, list of N items, haiku, poem with N lines, JSON), match the requested format EXACTLY and do not add commentary outside it.

Factual safety (critical):
- NEVER add specific facts (dates, places, awards, ranks, statistics, years, biographies) that are not already in the Original Response.
- If the Original Response contains a wrong fact, OMIT it rather than repeat it.
- If the Original Response is empty, off-topic, or merely refuses, supply a clean short answer of your own without invented details.

Forbidden:
- "As an AI ...", "I cannot ...", meta-commentary about the rewriting.
- References to <xRAG>, the Original Response, or the rewriting process.
- Padding with filler (e.g. "This is interesting because ..." without substantive content).

Output only the final reply."""


V_ROBUST = """\
You produce the best confident reply to the Prompt, using the Original Response only as a hint about which entity / answer / direction to commit to.

Procedure:
1. Read the Prompt and decide its required form (a single name, a number, a haiku, a 4-line poem, a story of N sentences, JSON, etc.).
2. Read the Original Response and decide which entity / angle / answer it is pointing at.
   - If the Original is empty, refuses ("I'm an AI..."), is meta-commentary, or is so garbled that no clear answer is recoverable, IGNORE it and produce a clean default answer of your own.
   - Otherwise, commit to the same answer the Original is pointing at.
3. Write the final reply.

Style of the reply:
- Confident, well-articulated, single coherent piece of text.
- For short-answer prompts, default to one sentence: the answer, plus at most one short clarifying clause (role, category, why it qualifies). For multi-line creative prompts, follow the requested format EXACTLY (4 lines means exactly 4 lines, 5 sentences means exactly 5 sentences, a haiku means a haiku).
- The reply must be coherent and topically aligned. Replace the Original wherever it is wrong, off-topic, repetitive, or self-referential.

Factual safety (most important rule):
- NEVER add specific factual claims (dates, places, awards, ranks, birth years, statistics, biographies, prize years) that are not already present in the Original Response.
- If the Original Response itself contains a clearly wrong specific fact (e.g. a misattribution, a wrong nationality, a fabricated rank), OMIT that fact rather than repeat it; keep the reply on the safe entity-level claim.
- A short, fact-light reply is always better than an elaborated reply that risks confabulation.

Forbidden:
- "As an AI...", "I cannot help...", apologies, meta-commentary about the rewriting.
- References to <xRAG>, the Original Response, or the rewriting process.
- Filler such as "This is a great question because..." that adds no content.

Output only the final reply, with no preamble."""


V_MINIMAL_INVASIVE = """\
Fix only grammar, formatting, length, and obvious noise so the Original Response satisfies the Prompt. Otherwise KEEP THE ORIGINAL'S CONTENT UNCHANGED.

What to do:
- If the Original answers the Prompt correctly and is on-topic, output it almost verbatim. Fix only typos, mid-sentence repetitions, broken sentences, "as an AI" disclaimers, and stray meta-text.
- If the Prompt asks for a specific FORMAT (riddle, haiku, 4-line poem, 5-sentence story, JSON, exactly N items), reshape the Original to that format without inventing new content.
- If the Original is empty, a refusal, or completely off-topic, output a short safe default answer of one sentence using only obvious low-risk content.

What NOT to do (these all hurt the score):
- Do NOT add new factual claims (dates, places, awards, ranks, prize years, biographies, etymologies, statistics, organizations) that are not already in the Original.
- Do NOT add explanatory context, "this is interesting because...", or any elaboration the Original did not contain.
- Do NOT reference the Original Response, the Prompt, the rewriting process, or "<xRAG>" — and never use phrases like "off-topic", "as per", "in the Original".
- Do NOT turn a creative-format prompt into a description of that format. If the Prompt asks for a riddle, output the riddle itself; do not write "the answer to the riddle is X because...". Same for haiku, joke, story.
- Do NOT homogenize — if the Original carries a particular angle / entity / style, keep that angle even if it differs from a generic reply.

Output only the final reply."""


VARIANTS: List[Tuple[str, str]] = [
    ("v0_terse_baseline",     V_TERSE_BASELINE),
    ("v1_verbose_failed",     V_VERBOSE_FAILED),
    ("v2_factual_conservative", V_FACTUAL_CONSERVATIVE),
    ("v3_g2_style",           V_G2_STYLE),
    ("v4_defensive_short",    V_DEFENSIVE_SHORT),
    ("v5_preserve_rewrite",   V_PRESERVE_REWRITE),
    ("v6_hybrid",             V_HYBRID),
    ("v7_robust",             V_ROBUST),
    ("v8_minimal_invasive",   V_MINIMAL_INVASIVE),
]


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

LLM_NAME = "Hannibal046/xrag-7b"  # base Mistral-7B-Instruct + projector
REWARD_NAME = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"


def build_user_msg(prompt: str, draft: str) -> str:
    return (
        "Rewrite the Original Response into the best reply to the Prompt.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Original Response:\n{draft}\n\n"
        "Refined Response:\n"
    )


def main():
    print(f"Loading LLM: {LLM_NAME} ...")
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    llm_tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=False, padding_side="left")
    if llm_tok.pad_token_id is None:
        llm_tok.pad_token_id = llm_tok.eos_token_id

    print(f"Loading reward: {REWARD_NAME} ...")
    reward = AutoModelForSequenceClassification.from_pretrained(
        REWARD_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0",
        num_labels=1,
    ).eval()
    reward_tok = AutoTokenizer.from_pretrained(REWARD_NAME)

    @torch.no_grad()
    def refine(system_msg: str, prompt: str, draft: str, max_new_tokens: int = 400) -> str:
        user = build_user_msg(prompt, draft)
        full = f"[INST] {system_msg}\n\n{user} [/INST]"
        enc = llm_tok(full, return_tensors="pt").to(llm.device)
        out = llm.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=llm_tok.pad_token_id,
        )
        gen = llm_tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return gen

    @torch.no_grad()
    def score(prompt: str, response: str) -> float:
        chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        ids = reward_tok.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(reward.device)
        return reward(ids).logits[0, 0].item()

    # G2 baseline reference: score the original G2 outputs from the input
    # file on the same prompts. Provides an upper-bound reference for
    # what we'd get if we just used G2 directly without any latent step.
    print()
    print("=" * 80)
    print("G2 BASELINE REFERENCE (raw G2 outputs from input, same prompts)")
    print("=" * 80)
    g2_input = "results/curated/g2_theta0.3_temp1_iter15/generations.jsonl"
    g2_scores: dict[str, list[float]] = {}
    target_ids = {pid for pid, _, _ in GRID}
    with open(g2_input) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in target_ids:
                gens = r["generations"]
                ss = []
                for gi, g in enumerate(gens):
                    s = score(r["prompt"], g)
                    ss.append(s)
                g2_scores[r["id"]] = ss
                print(f"\n=== {r['id']}: {r['prompt']}")
                for gi, (g, s) in enumerate(zip(gens, ss)):
                    short = g[:120].strip().replace("\n", " ")
                    if len(g) > 120: short += "..."
                    print(f"  [G2-{gi+1:>2}] u={s:>+6.3f}  ({len(g)} chars)  {short}")

    # Run grid
    print()
    print("=" * 80)
    print("REFINEMENT VARIANT COMPARISON")
    print("=" * 80)

    summary: dict[str, list[float]] = {v: [] for v, _ in VARIANTS}
    full_log = []

    for pid, prompt, drafts in GRID:
        for di, draft in enumerate(drafts):
            print(f"\n=== {pid} draft#{di+1}: {prompt}")
            print(f"  DRAFT ({len(draft)} chars): {draft[:140].strip()}{'...' if len(draft) > 140 else ''}")
            for vname, vsys in VARIANTS:
                refined = refine(vsys, prompt, draft)
                u = score(prompt, refined)
                summary[vname].append(u)
                short = refined[:140].strip().replace("\n", " ")
                if len(refined) > 140: short += "..."
                print(f"  [{vname:<26}] u={u:>+6.3f}  ({len(refined)} chars)  {short}")
                full_log.append({
                    "id": pid, "prompt": prompt, "draft_idx": di, "draft": draft,
                    "variant": vname, "refined": refined, "score": u,
                })

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY (mean Skywork reward across all (prompt, draft) pairs)")
    print("=" * 80)
    print(f"{'variant':<28} {'n':>3} {'mean_u':>8} {'min_u':>8} {'max_u':>8}")
    # G2 baseline first
    all_g2 = [s for ss in g2_scores.values() for s in ss]
    if all_g2:
        n = len(all_g2)
        m = sum(all_g2) / n
        print(f"{'G2_baseline':<28} {n:>3} {m:>+8.3f} {min(all_g2):>+8.3f} {max(all_g2):>+8.3f}")
        # Mean across just first 4 (the G2 anchors used as seeds)
        first4 = []
        for ss in g2_scores.values():
            first4.extend(ss[:4])
        n4 = len(first4)
        m4 = sum(first4) / n4
        print(f"{'G2_first4_anchors':<28} {n4:>3} {m4:>+8.3f} {min(first4):>+8.3f} {max(first4):>+8.3f}")
        print(f"{'-'*60}")
    for vname, scores_v in summary.items():
        if not scores_v:
            continue
        n = len(scores_v)
        m = sum(scores_v) / n
        print(f"{vname:<28} {n:>3} {m:>+8.3f} {min(scores_v):>+8.3f} {max(scores_v):>+8.3f}")

    # Persist full log next to the python file for later inspection.
    out_path = "logs/refinement_probe.jsonl"
    with open(out_path, "w") as f:
        for r in full_log:
            f.write(json.dumps(r) + "\n")
    print(f"\nFull log written to {out_path}")


if __name__ == "__main__":
    main()
