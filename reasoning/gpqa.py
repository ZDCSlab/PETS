#!/usr/bin/env python3
"""GPQA Diamond inference and evaluation."""

import argparse
import random
import re

import common


PROMPT_TEMPLATE = r"""Answer the following multiple choice question.
The last line of your response MUST be exactly in the format: Answer: \boxed{{LETTER}}
where LETTER is one of A, B, C, D.

Example (last line only): Answer: \boxed{{A}}

Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def build_prompt(item, seed=42):
    """Build GPQA prompt with shuffled answer choices.

    Returns (prompt_text, correct_letter).
    """
    rng = random.Random(seed + hash(item.get("Question", "")))

    question = item.get("Question") or item.get("question", "")
    correct = item.get("Correct Answer") or item.get("correct_answer", "")
    choices = [
        correct,
        item.get("Incorrect Answer 1") or item.get("incorrect_answer_1", ""),
        item.get("Incorrect Answer 2") or item.get("incorrect_answer_2", ""),
        item.get("Incorrect Answer 3") or item.get("incorrect_answer_3", ""),
    ]

    indices = list(range(4))
    rng.shuffle(indices)
    shuffled = [choices[i] for i in indices]
    correct_letter = chr(ord("A") + indices.index(0))

    prompt = PROMPT_TEMPLATE.format(
        question=question, A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3],
    )
    return prompt, correct_letter


def extract_answer(text, default=""):
    """Extract single-letter answer (A-D) from model output."""
    # Answer: \boxed{X}
    matches = re.findall(r"(?i)answer:\s*\\boxed\s*\{\s*([ABCD])\s*\}", text)
    if matches:
        return matches[-1].upper()
    # Answer: X
    matches = re.findall(r"(?i)answer:\s*\$?\s*([ABCD])\s*\$?", text)
    if matches:
        return matches[-1].upper()
    # Standalone letter at end
    m = re.search(r"\b([ABCD])\s*$", text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # (X) pattern
    matches = re.findall(r"\(([ABCD])\)", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return default


def evaluate(results_path):
    results = common.load_results(results_path)
    total = correct = 0
    for res in results:
        if not res.get("success"):
            continue
        total += 1
        if res.get("pred", "").strip().upper() == res.get("answer", "").strip().upper():
            correct += 1
    acc = f"{100 * correct / total:.1f}%" if total else "N/A"
    print(f"GPQA Diamond: {correct}/{total} = {acc}")


def main():
    ap = argparse.ArgumentParser(description="GPQA Diamond Inference & Evaluation")
    common.add_common_args(ap)
    ap.add_argument(
        "--data_path",
        default="Idavidrein/gpqa",
        help="HuggingFace dataset name or local dataset path (default: Idavidrein/gpqa)",
    )
    ap.add_argument("--subset", default="gpqa_diamond", help="Dataset subset")
    ap.add_argument("--out", default="gpqa_preds.jsonl", help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for choice shuffling")
    args = ap.parse_args()

    if args.eval_only:
        evaluate(args.out)
        return

    client, model = common.create_client(args.host, args.port, args.model_name)

    print(f"Loading GPQA dataset ({args.data_path}, {args.subset})...")
    dataset = common.load_hf_split(args.data_path, split="train", subset=args.subset)
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"Loaded {len(dataset)} problems\n")

    def process(item):
        prompt_text, correct_letter = build_prompt(item, seed=args.seed)
        return common.process_question(
            client, model, item,
            n_samples=args.n, temperature=args.temperature,
            top_p=args.top_p, top_logprobs=args.top_logprobs,
            prompt_text=prompt_text,
            extract_fn=extract_answer,
            gold_answer=correct_letter,
            problem_text=prompt_text,
            system_prompt="You are a helpful assistant that solves problems.",
        )

    common.run_inference(dataset, process, args.out, args.max_workers, desc="GPQA Diamond")


if __name__ == "__main__":
    main()
