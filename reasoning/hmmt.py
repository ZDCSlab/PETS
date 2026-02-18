#!/usr/bin/env python3
"""HMMT February 2025 inference and evaluation."""

import argparse
import re

import common


PROMPT_TEMPLATE = """\
Solve the following math problem step by step. Put your final answer
in a \\boxed{{}} command.

{problem}

Remember to put your final answer in \\boxed{{}}.

Reasoning:"""


def build_prompt(item):
    problem = item.get("problem") or item.get("question") or item.get("prompt", "")
    return PROMPT_TEMPLATE.format(problem=problem)


def extract_answer(text, default=""):
    """Extract answer from \\boxed{} (nested-brace safe), with fallbacks."""
    boxed = common.extract_boxed(text)
    if boxed:
        return boxed
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    if lines and lines[-1].strip() and not lines[-1].strip().endswith(("?", ".")):
        return lines[-1].strip()
    return default


def evaluate(results_path):
    from math_verify import verify

    results = common.load_results(results_path)
    total = correct = 0
    for res in results:
        if not res.get("success"):
            continue
        total += 1
        gold = common._parse_math(str(res.get("answer", "")).strip())
        pred = common._parse_math(str(res.get("pred", "")).strip())
        if gold and pred and verify(gold, pred):
            correct += 1
    acc = f"{100 * correct / total:.1f}%" if total else "N/A"
    print(f"HMMT Feb 2025: {correct}/{total} = {acc}")


def main():
    ap = argparse.ArgumentParser(description="HMMT Feb 2025 Inference & Evaluation")
    common.add_common_args(ap)
    ap.add_argument(
        "--data_path",
        default="MathArena/hmmt_feb_2025",
        help="Local parquet directory or HuggingFace dataset name (default: MathArena/hmmt_feb_2025)",
    )
    ap.add_argument("--out", default="hmmt_preds.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    if args.eval_only:
        evaluate(args.out)
        return

    client, model = common.create_client(args.host, args.port, args.model_name)
    dataset = common.load_parquet_or_hf(args.data_path, split="train")
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"Loaded {len(dataset)} problems\n")

    def process(item):
        return common.process_question(
            client, model, item,
            n_samples=args.n, temperature=args.temperature,
            top_p=args.top_p, top_logprobs=args.top_logprobs,
            prompt_text=build_prompt(item),
            extract_fn=extract_answer,
            vote_fn=common.vote_majority_equiv,
            system_prompt="You are a helpful assistant that solves math problems.",
        )

    common.run_inference(dataset, process, args.out, args.max_workers, desc="HMMT Feb 2025")


if __name__ == "__main__":
    main()
