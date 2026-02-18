#!/usr/bin/env python3
"""MMLU-Pro inference and evaluation with few-shot chain-of-thought."""

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

import common


CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]

DEFAULT_INITIAL_PROMPT = (
    'The following are multiple choice questions (with answers) about {$}. '
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n\n"
)


def _preprocess(split) -> List[Dict]:
    """Remove N/A options and convert HF rows to plain dicts."""
    out: List[Dict] = []
    for item in split:
        row = dict(item)
        row["options"] = [opt for opt in row.get("options", []) if opt != "N/A"]
        out.append(row)
    return out


def load_mmlu_pro(data_path: str):
    """Load MMLU-Pro dataset, returning (test_rows, validation_rows)."""
    ds = load_dataset(data_path)
    if "test" not in ds or "validation" not in ds:
        raise ValueError(
            f"Dataset '{data_path}' must contain 'test' and 'validation' splits; got {list(ds.keys())}."
        )
    return _preprocess(ds["test"]), _preprocess(ds["validation"])


def select_by_category(rows: List[Dict], category: str) -> List[Dict]:
    return [item for item in rows if item.get("category") == category]


def load_initial_prompt(path: Optional[str]) -> str:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return DEFAULT_INITIAL_PROMPT


def _format_example(example: Dict, include_answer: bool) -> str:
    prompt = "Question:\n" + example.get("question", "") + "\nOptions:\n"
    for i, opt in enumerate(example.get("options", [])):
        if i >= len(CHOICE_LABELS):
            break
        prompt += f"{CHOICE_LABELS[i]}. {opt}\n"

    if include_answer:
        cot = example.get("cot_content", "")
        cot = cot.replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_prompt(item: Dict, val_rows: List[Dict], k: int, initial_prompt_text: str) -> str:
    """Build the few-shot CoT prompt for one test question."""
    subject = item.get("category", "")
    examples = select_by_category(val_rows, subject)[:k]
    prompt = initial_prompt_text.replace("{$}", subject) + "\n"
    for ex in examples:
        prompt += _format_example(ex, include_answer=True)
    prompt += _format_example(item, include_answer=False)
    return prompt


def extract_answer(text: str, default: str = "") -> str:
    """Extract letter answer (A-P) from model output."""
    m = re.search(r"answer is \(?([A-P])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"[aA]nswer:\s*([A-P])", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-P])\b(?!.*\b[A-P]\b)", text, re.DOTALL)
    if m:
        return m.group(1).upper()
    return default


def normalize_answer(answer) -> str:
    if answer is None:
        return ""
    m = re.search(r"\b([A-P])\b", str(answer).strip().upper())
    return m.group(1) if m else str(answer).strip().upper()


def get_gold_letter(item: Dict) -> str:
    """Resolve gold answer to a canonical letter whenever possible."""
    gold = normalize_answer(item.get("answer", ""))
    if gold:
        return gold
    idx = item.get("answer_index")
    if isinstance(idx, int) and 0 <= idx < len(CHOICE_LABELS):
        return CHOICE_LABELS[idx]
    return ""


def evaluate(results_path: str):
    results = common.load_results(results_path)
    total = correct = 0
    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for res in results:
        if not res.get("success"):
            continue

        total += 1
        subject = res.get("subject") or res.get("category", "unknown")
        subject_stats[subject]["total"] += 1

        pred = normalize_answer(res.get("pred", ""))
        gold = normalize_answer(res.get("answer", ""))
        idx = res.get("answer_index")
        if not gold and isinstance(idx, int) and idx < len(CHOICE_LABELS):
            gold = CHOICE_LABELS[idx]

        if pred == gold:
            correct += 1
            subject_stats[subject]["correct"] += 1

    acc = f"{100 * correct / total:.1f}%" if total else "N/A"
    print(f"MMLU-Pro Overall: {correct}/{total} = {acc}\n")
    for subject in sorted(subject_stats):
        stats = subject_stats[subject]
        subject_acc = (
            f"{100 * stats['correct'] / stats['total']:.1f}%" if stats["total"] else "N/A"
        )
        print(f"  {subject:40s}  {stats['correct']:4d}/{stats['total']:4d} = {subject_acc}")


def _select_subjects(test_rows: List[Dict], selected_subjects: str) -> List[str]:
    all_subjects = sorted({item.get("category", "") for item in test_rows if item.get("category")})
    if selected_subjects == "all":
        return all_subjects

    selected = [s.strip() for s in selected_subjects.split(",") if s.strip()]
    normalized_selected = [s.replace(" ", "_") for s in selected]
    return [
        subject for subject in all_subjects
        if any(sel in subject.replace(" ", "_") for sel in normalized_selected)
    ]


def main():
    ap = argparse.ArgumentParser(description="MMLU-Pro Inference & Evaluation")
    common.add_common_args(ap)
    ap.add_argument(
        "--data_path",
        default="TIGER-Lab/MMLU-Pro",
        help="HuggingFace dataset name or local dataset path (default: TIGER-Lab/MMLU-Pro)",
    )
    ap.add_argument("--out", default="mmlu_pro_preds.jsonl", help="Output JSONL path")
    ap.add_argument("--selected_subjects", "-sub", default="all", help="Comma-separated subjects or 'all'")
    ap.add_argument("--ntrain", "-k", type=int, default=5, help="Few-shot examples per subject")
    ap.add_argument("--initial_prompt", default=None, help="Path to initial prompt template file")
    args = ap.parse_args()

    if args.eval_only:
        evaluate(args.out)
        return

    client, model = common.create_client(args.host, args.port, args.model_name)
    initial_prompt_text = load_initial_prompt(args.initial_prompt)

    print(f"Loading MMLU-Pro dataset ({args.data_path})...")
    test_rows, val_rows = load_mmlu_pro(args.data_path)
    if args.limit:
        test_rows = test_rows[:args.limit]
    print(f"Loaded {len(test_rows)} test / {len(val_rows)} validation examples\n")

    subjects = _select_subjects(test_rows, args.selected_subjects)
    if not subjects:
        raise ValueError(f"No subjects selected from --selected_subjects={args.selected_subjects!r}")
    print(f"Subjects ({len(subjects)}): {', '.join(subjects)}\n")

    workers = args.max_workers or min(32, (os.cpu_count() or 8) * 5)

    with open(args.out, "w", encoding="utf-8") as outf:
        for subject in subjects:
            sub_test = select_by_category(test_rows, subject)
            sub_val = select_by_category(val_rows, subject)
            print(f"--- {subject} ({len(sub_test)} questions) ---")

            def process(item, _sub_val=sub_val, _subject=subject):
                prompt_text = build_prompt(item, _sub_val, args.ntrain, initial_prompt_text)
                res = common.process_question(
                    client,
                    model,
                    item,
                    n_samples=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_logprobs=args.top_logprobs,
                    prompt_text=prompt_text,
                    extract_fn=extract_answer,
                    gold_answer=get_gold_letter(item),
                    problem_text=item.get("question", ""),
                    system_prompt="You are a helpful assistant.",
                )
                res["subject"] = _subject
                res["category"] = _subject
                if "answer_index" in item:
                    res["answer_index"] = item["answer_index"]
                return res

            with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(
                total=len(sub_test), desc=subject, unit="q"
            ) as pbar:
                for res in pool.map(process, sub_test):
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                    pbar.update(1)

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
