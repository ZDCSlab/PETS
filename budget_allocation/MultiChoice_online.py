#!/usr/bin/env python3
"""
Oracle difficulty via KMeans (k=5) for (a,b) fits (GPQA).

This file is a variant of `gpqa_streaming.py`:
- Predictor training / streaming policy stays the same (reused from gpqa_streaming).
- Oracle setting changes from (a,b) quantile grid to kmeans buckets.

Workflow (oracle):
1) Fit per-question (a_q,b_q) on training questions (full-answer oracle fits)
2) Fit KMeans on (a,b) -> k cluster centers
3) Allocate budgets across the k centers (greedy marginal gains under average budget)
4) For each test question, compute its (a,b), assign to nearest center, and use that bucket's budget
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import gpqa_streaming as base
from oracle_kmeans_common import (
    OracleDifficultyModelKMeans,
    build_oracle_difficulty_model_from_params,
    greedy_budget_allocation_oracle_common,
    locate_param_bin_oracle as locate_param_bin_oracle_common,
)


def weighted_majority_vote_min(samples: Sequence[object], weights: Sequence[object]) -> Optional[object]:
    """Confidence-weighted majority vote.

    Aggregates weights per label, then tie-breaks by the minimal label to mirror
    `base.majority_vote_with_tie_break`.
    """
    if not samples:
        return None
    from collections import defaultdict

    weighted = defaultdict(float)
    for i, s in enumerate(samples):
        w_obj = weights[i] if i < len(weights) else 1.0
        try:
            w = float(w_obj)
            if not math.isfinite(w):
                w = 1.0
        except Exception:
            w = 1.0
        weighted[s] += w
    if not weighted:
        return None
    max_w = max(weighted.values())
    winners = [k for k, v in weighted.items() if v == max_w]
    return min(winners)


def weighted_vote_variant(
    samples: Sequence[object],
    weights: Sequence[object],
    *,
    variant: str = "weighted",
) -> Optional[object]:
    """Weighted voting variants (mirrors gpqa_offline.py patterns).

    variant:
      - "weighted": use all samples
      - "top10"/"top30"/"top50"/"top70"/"top90": keep only the top-X% samples by weight

    Tie-break is deterministic via minimal label, matching base.majority_vote_with_tie_break.
    """
    if not samples:
        return None
    pct_map = {
        "weighted": 1.0,
        "top10": 0.10,
        "top30": 0.30,
        "top50": 0.50,
        "top70": 0.70,
        "top90": 0.90,
    }
    p = pct_map.get(str(variant), None)
    if p is None:
        raise ValueError(f"Unknown conf voting variant: {variant}")

    indexed: List[Tuple[int, object, float]] = []
    for idx, s in enumerate(samples):
        w_obj = weights[idx] if idx < len(weights) else 1.0
        try:
            w = float(w_obj)
            if not math.isfinite(w):
                w = 1.0
        except Exception:
            w = 1.0
        indexed.append((idx, s, w))

    if not indexed:
        return None

    keep_k = max(1, int(round(len(indexed) * p)))
    keep_k = min(keep_k, len(indexed))
    kept = sorted(indexed, key=lambda t: t[2], reverse=True)[:keep_k]

    from collections import defaultdict

    weighted_cnt = defaultdict(float)
    for _idx, ans, w in kept:
        weighted_cnt[ans] += float(w)
    if not weighted_cnt:
        return None
    max_w = max(weighted_cnt.values())
    winners = [k for k, v in weighted_cnt.items() if v == max_w]
    return min(winners)


def _pseudo_label_conf_from_full_pool(q: base.QuestionRecord, *, conf_variant: str) -> Optional[object]:
    """Run-specific pseudo label for confidence-weighted consistency.

    Uses the full pool (typically 64 answers + confidences) from the current run.
    Falls back to q.final when the weighted vote cannot be computed.
    """
    answers = list(q.answers) if q.answers is not None else []
    if not answers:
        return getattr(q, "final", None)
    weights = list(getattr(q, "confs", None) or [])
    try:
        voted = weighted_vote_variant(answers, weights, variant=str(conf_variant))
    except Exception:
        voted = None
    return voted if voted is not None else getattr(q, "final", None)


def evaluate_streaming_conf(
    test_questions: Sequence[base.QuestionRecord],
    budget_plan: base.BudgetPlan,
    *,
    conf_variant: str = "weighted",
) -> Dict[str, float]:
    """Predictor evaluation using the same deterministic prefix sampling, but weighted voting."""
    correct_acc = 0
    evaluated_acc = 0
    correct_cons = 0
    evaluated_cons = 0
    skipped = 0
    total_budget_used = 0.0

    for q in test_questions:
        answers = list(q.answers) if q.answers is not None else []
        if not answers:
            skipped += 1
            continue

        # Deterministic predictor: bucket from first K0, then take B_target prefix.
        if len(answers) < int(base.K0):
            skipped += 1
            continue
        t = base.bucket_from_samples4(answers[: int(base.K0)])
        B_target = int(budget_plan.B_t[int(t) - 1])
        k = min(int(B_target), len(answers))
        samples = answers[:k]
        confs = list(getattr(q, "confs", None) or [])
        weights = confs[:k]
        total_budget_used += float(len(samples))

        pred = weighted_vote_variant(samples, weights, variant=str(conf_variant))
        if pred is None:
            skipped += 1
            continue

        if q.correct is not None:
            evaluated_acc += 1
            if pred == q.correct:
                correct_acc += 1
        if getattr(q, "final", None) is not None:
            evaluated_cons += 1
            gold_conf = _pseudo_label_conf_from_full_pool(q, conf_variant=str(conf_variant))
            if pred == gold_conf:
                correct_cons += 1
        if q.correct is None and getattr(q, "final", None) is None:
            skipped += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    return {
        "accuracy_conf": float(accuracy),
        "consistency_conf": float(consistency),
        "skipped": float(skipped),
        "total_budget_used": float(total_budget_used),
    }


def evaluate_fixed_budget_majority_conf(
    test_questions: Sequence[base.QuestionRecord],
    per_question_budget: int,
    *,
    conf_variant: str = "weighted",
) -> Dict[str, float]:
    """Baseline evaluation with fixed prefix budget, but weighted voting."""
    budget = max(int(base.K0), int(per_question_budget))

    evaluated_acc = 0
    evaluated_cons = 0
    skipped = 0
    correct_acc = 0
    correct_cons = 0
    total_budget_used = 0.0

    for q in test_questions:
        answers = list(q.answers) if q.answers is not None else []
        if not answers:
            skipped += 1
            continue

        k = min(int(budget), len(answers))
        samples = answers[:k]
        confs = list(getattr(q, "confs", None) or [])
        weights = confs[:k]
        total_budget_used += float(len(samples))

        pred = weighted_vote_variant(samples, weights, variant=str(conf_variant))
        if q.correct is not None:
            evaluated_acc += 1
            correct_acc += int(pred == q.correct)
        if getattr(q, "final", None) is not None:
            evaluated_cons += 1
            gold_conf = _pseudo_label_conf_from_full_pool(q, conf_variant=str(conf_variant))
            correct_cons += int(pred == gold_conf)
        if q.correct is None and getattr(q, "final", None) is None:
            skipped += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    return {
        "accuracy_conf": float(accuracy),
        "consistency_conf": float(consistency),
        "skipped": float(skipped),
        "total_budget_used": float(total_budget_used),
    }


# -----------------------------
# Oracle difficulty via KMeans
# -----------------------------

def build_oracle_difficulty_model(
    train_params: Dict[str, Tuple[float, float]],
    *,
    k: int = 5,
    random_seed: int = 0,
) -> Optional[OracleDifficultyModelKMeans]:
    """Fit KMeans(k=5) model on training (a,b) and return bucket centers + probs."""
    return build_oracle_difficulty_model_from_params(
        train_params,
        score_fn=lambda a, b: float(base.A_probit(1, float(a), float(b))),
        k=int(k),
        random_seed=int(random_seed),
    )


def greedy_budget_allocation_oracle(
    model: OracleDifficultyModelKMeans,
    *,
    average_budget: float,
    B_max: int = 64,
    min_budget: int = 4,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Greedy budget allocation over KMeans oracle buckets.

    Properties:
      - Hard constraint: expected budget never exceeds `average_budget` (up to `eps`).
      - Near-tightness: if gain becomes numerically ~0 (e.g., probit saturates), we still
        try to consume remaining slack via a second "fill-to-tight" phase.
    """
    def marginal_gain(t: int, cur: int, centers: np.ndarray) -> float:
        """Stable Δ(k)=A(k+1)-A(k) for probit-sqrt(k).

        Naively computing `norm.cdf(x2) - norm.cdf(x1)` can underflow to 0 when
        cdf() saturates to 1.0 at float precision. Use the tail probability:
          cdf(x2)-cdf(x1) = (1-sf(x2))-(1-sf(x1)) = sf(x1)-sf(x2)
        and compute the difference in log-space to preserve tiny gains.
        """
        a, b = float(centers[t, 0]), float(centers[t, 1])

        # Local import: scipy is already required by `gpqa_streaming.py`.
        from scipy.stats import norm  # type: ignore

        x1 = float(a * math.sqrt(float(cur)) + b)
        x2 = float(a * math.sqrt(float(cur + 1)) + b)

        # sf(x1) >= sf(x2) when x2 > x1; guard anyway.
        logsf1 = float(norm.logsf(x1))
        logsf2 = float(norm.logsf(x2))
        if not (math.isfinite(logsf1) and math.isfinite(logsf2)):
            # Fallback: best-effort numeric (may be 0 when saturated).
            return float(base.A_probit(cur + 1, a, b) - base.A_probit(cur, a, b))

        # sf_diff = exp(logsf_small) - exp(logsf_large) computed stably.
        # Ensure logsf1 >= logsf2 (expected), otherwise swap.
        if logsf1 < logsf2:
            logsf1, logsf2 = logsf2, logsf1

        sf1 = math.exp(logsf1)
        # sf1 - sf2 = sf1 * (1 - exp(logsf2-logsf1))
        return float(sf1 * (-math.expm1(logsf2 - logsf1)))

    return greedy_budget_allocation_oracle_common(
        model,
        average_budget=float(average_budget),
        B_max=int(B_max),
        min_budget=int(min_budget),
        marginal_gain_fn=marginal_gain,
        eps=float(eps),
    )


def locate_param_bin_oracle(
    a_value: float,
    b_value: float,
    model: OracleDifficultyModelKMeans,
) -> int:
    """Assign (a,b) to nearest KMeans center and return ordered bucket index [0..k-1]."""
    return int(locate_param_bin_oracle_common(float(a_value), float(b_value), model))


def evaluate_oracle_setting(
    train_questions: Sequence[base.QuestionRecord],
    test_questions: Sequence[base.QuestionRecord],
    *,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    average_budget: float,
    max_per_question: int,
    k_max_curve: int = 40,
    curve_mc_trials: int = 2000,
    oracle_model_override: Optional[OracleDifficultyModelKMeans] = None,
    oracle_test_params_override: Optional[Dict[str, Tuple[float, float]]] = None,
    kmeans_k: int = 5,
    kmeans_seed: int = 0,
) -> Tuple[
    Optional[Dict[str, float]],
    Optional[float],
    Optional[np.ndarray],
    Optional[OracleDifficultyModelKMeans],
    Dict[str, Tuple[float, float]],
]:
    """End-to-end oracle evaluation using KMeans buckets + greedy allocation."""
    oracle_model = oracle_model_override
    test_params = oracle_test_params_override

    if oracle_model is None or test_params is None:
        train_params = base.compute_question_param_map(
            train_questions,
            k_max_curve=k_max_curve,
            curve_mc_trials=curve_mc_trials,
        )
        test_params = base.compute_question_param_map(
            test_questions,
            k_max_curve=k_max_curve,
            curve_mc_trials=curve_mc_trials,
        )
        oracle_model = build_oracle_difficulty_model(train_params, k=kmeans_k, random_seed=kmeans_seed)

    if oracle_model is None or not test_params:
        return None, None, None, oracle_model, test_params or {}

    budget_by_bucket, _ = greedy_budget_allocation_oracle(
        oracle_model,
        average_budget=float(average_budget),
        B_max=int(max_per_question),
        min_budget=4,
    )
    expected_budget = float(np.sum(oracle_model.probs * budget_by_bucket))

    evaluated_acc = 0
    correct_acc = 0
    correct_acc_conf = 0
    evaluated_cons = 0
    correct_cons = 0
    correct_cons_conf = 0
    skipped = 0
    total_budget_used = 0.0
    per_bucket_budget = np.zeros_like(budget_by_bucket, dtype=float)

    for q in test_questions:
        params = test_params.get(q.qid)
        if params is None or not q.answers:
            skipped += 1
            continue
        a_val, b_val = params
        bucket = locate_param_bin_oracle(a_val, b_val, oracle_model)
        if bucket < 0 or bucket >= int(budget_by_bucket.size):
            skipped += 1
            continue

        budget = int(budget_by_bucket[bucket])
        if budget <= 0:
            skipped += 1
            continue

        samples = list(q.answers)[: min(budget, len(q.answers))]
        if not samples:
            skipped += 1
            continue

        total_budget_used += float(len(samples))
        per_bucket_budget[bucket] += float(len(samples))

        pred = base.majority_vote_with_tie_break(samples)
        pred_conf = None
        if add_conf:
            pred_conf = weighted_vote_variant(
                samples,
                list(getattr(q, "confs", None) or [])[: len(samples)],
                variant=str(conf_variant),
            )
        if pred is None:
            skipped += 1
            continue

        if q.correct is not None:
            evaluated_acc += 1
            if pred == q.correct:
                correct_acc += 1
            if add_conf and pred_conf is not None and pred_conf == q.correct:
                correct_acc_conf += 1
        if getattr(q, "final", None) is not None:
            evaluated_cons += 1
            if pred == getattr(q, "final"):
                correct_cons += 1
            if add_conf and pred_conf is not None:
                gold_conf = _pseudo_label_conf_from_full_pool(q, conf_variant=str(conf_variant))
                if pred_conf == gold_conf:
                    correct_cons_conf += 1
        if q.correct is None and getattr(q, "final", None) is None:
            skipped += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy),
        "consistency": float(consistency),
        "skipped": float(skipped),
        "total_budget_used": float(total_budget_used),
    }
    if add_conf:
        accuracy_conf = correct_acc_conf / evaluated_acc if evaluated_acc else float("nan")
        consistency_conf = correct_cons_conf / evaluated_cons if evaluated_cons else float("nan")
        metrics["accuracy_conf"] = float(accuracy_conf)
        metrics["consistency_conf"] = float(consistency_conf)
    return metrics, expected_budget, budget_by_bucket, oracle_model, test_params


def sweep_average_budgets(
    stats: base.BucketStats,
    test_questions: Sequence[base.QuestionRecord],
    *,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    sweep_max: int,
    B_max: int,
    rng_seed: int = 0,
    oracle_model: Optional[OracleDifficultyModelKMeans] = None,
    oracle_test_params: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, object]]:
    """Sweep average budgets and collect predictor/baseline/oracle metrics (oracle=kmeans)."""
    rows: List[Dict[str, object]] = []

    start_budget = max(base.K0, 1)
    for avg_budget in range(start_budget, sweep_max + 1):
        plan = base.solve_budget_plan_greedy_marginal(stats, B_bar=float(avg_budget), B_max=B_max, k0=base.K0)
        predictor_metrics, _ = base.evaluate_streaming(test_questions, plan, rng_seed=rng_seed)
        predictor_conf_metrics: Dict[str, float] = {}
        if add_conf:
            predictor_conf_metrics = evaluate_streaming_conf(test_questions, plan, conf_variant=str(conf_variant))
        baseline_metrics = base.evaluate_fixed_budget_majority(
            test_questions, per_question_budget=avg_budget, rng_seed=rng_seed
        )
        baseline_conf_metrics: Dict[str, float] = {}
        if add_conf:
            baseline_conf_metrics = evaluate_fixed_budget_majority_conf(
                test_questions,
                per_question_budget=avg_budget,
                conf_variant=str(conf_variant),
            )
        expected_budget = float(np.sum(stats.pi_t * plan.B_t))

        oracle_metrics: Optional[Dict[str, object]] = None
        oracle_budget_by_bucket: Optional[np.ndarray] = None
        oracle_expected = None
        if oracle_model is not None and oracle_test_params is not None:
            oracle_metrics, oracle_expected, oracle_budget_by_bucket, _, _ = evaluate_oracle_setting(
                train_questions=[],
                test_questions=test_questions,
                add_conf=bool(add_conf),
                conf_variant=str(conf_variant),
                average_budget=float(avg_budget),
                max_per_question=B_max,
                oracle_model_override=oracle_model,
                oracle_test_params_override=oracle_test_params,
                k_max_curve=0,
                curve_mc_trials=0,
            )

        rows.append(
            {
                "average_budget": avg_budget,
                "predictor_total": predictor_metrics["total_budget_used"],
                "predictor_accuracy": predictor_metrics["accuracy"],
                "predictor_accuracy_conf": (predictor_conf_metrics.get("accuracy_conf") if add_conf else None),
                "predictor_consistency": predictor_metrics.get("consistency"),
                "predictor_consistency_conf": (predictor_conf_metrics.get("consistency_conf") if add_conf else None),
                "predictor_expected": expected_budget,
                "predictor_skipped": predictor_metrics.get("skipped"),
                "baseline_total": baseline_metrics["total_budget_used"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_accuracy_conf": (baseline_conf_metrics.get("accuracy_conf") if add_conf else None),
                "baseline_consistency": baseline_metrics.get("consistency"),
                "baseline_consistency_conf": (baseline_conf_metrics.get("consistency_conf") if add_conf else None),
                "baseline_skipped": baseline_metrics.get("skipped"),
                "budget_plan": plan.B_t.tolist(),
                "oracle_total": oracle_metrics["total_budget_used"] if oracle_metrics else None,
                "oracle_accuracy": oracle_metrics["accuracy"] if oracle_metrics else None,
                "oracle_accuracy_conf": (oracle_metrics.get("accuracy_conf") if (add_conf and oracle_metrics) else None),
                "oracle_consistency": oracle_metrics.get("consistency") if oracle_metrics else None,
                "oracle_consistency_conf": (
                    oracle_metrics.get("consistency_conf") if (add_conf and oracle_metrics) else None
                ),
                "oracle_expected": oracle_expected,
                "oracle_budget_grid": oracle_budget_by_bucket.tolist() if oracle_budget_by_bucket is not None else None,
            }
        )

    return rows


# -----------------------------
# CLI entrypoint (mostly reused)
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming GPQA bucketed allocation with sweep (oracle=kmeans).")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(__file__).parent / "gpqa_conf_qwen3_64.jsonl"),
        help="Path to GPQA JSONL file.",
    )
    parser.add_argument(
        "--conf-metric",
        type=str,
        default="mean",
        help=(
            "Which scalar confidence to extract from trace_confidence entries for *_conf methods. "
            "Examples: mean/Conf/tail/bottom, or a raw key name present in each entry."
        ),
    )
    parser.add_argument(
        "--conf-variant",
        type=str,
        default="weighted",
        choices=["weighted", "top10", "top30", "top50", "top70", "top90"],
        help="Weighted voting variant for *_conf methods (mirrors gpqa_offline.py).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=30,
        help="Number of questions used for training (rest for testing).",
    )
    parser.add_argument(
        "--oracle-fit-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Oracle upper bound: fit KMeans buckets (and bucket probs) on the full dataset (train+test) "
            "instead of train-only. This leaks test distribution info by design."
        ),
    )
    parser.add_argument(
        "--average-budget",
        type=float,
        default=16.0,
        help="Target average budget used for the greedy allocator.",
    )
    parser.add_argument(
        "--max-per-question",
        type=int,
        default=64,
        help="Maximum budget allowed for a single question.",
    )
    parser.add_argument(
        "--k-max-curve",
        type=int,
        default=40,
        help="Maximum k for per-question accuracy curve fitting.",
    )
    parser.add_argument(
        "--subsample4-draws",
        type=int,
        default=2000,
        help="Number of subsample draws for estimating pi_q(t).",
    )
    parser.add_argument(
        "--curve-mc-trials",
        type=int,
        default=2000,
        help="Monte Carlo trials for accuracy curve estimation.",
    )
    parser.add_argument(
        "--add_conf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute/plot confidence-weighted (_conf) methods in online/sweep outputs.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable sweep mode over average budgets and plot predictor vs baseline.",
    )
    parser.add_argument(
        "--sweep-max",
        type=int,
        default=64,
        help="Maximum average budget (inclusive) when sweeping.",
    )
    parser.add_argument(
        "--accuracy-plot",
        type=str,
        default="gpqa_model_online_accuracy.png",
        help="(Reference-style) accuracy curve plot path (PNG).",
    )
    parser.add_argument(
        "--consistency-plot",
        type=str,
        default="gpqa_model_online_consistency.png",
        help="(Reference-style) consistency curve plot path (PNG).",
    )
    parser.add_argument(
        "--accuracy-csv",
        type=str,
        default=None,
        help="Accuracy curve summary CSV path (default: same as --accuracy-plot with .csv).",
    )
    parser.add_argument(
        "--consistency-csv",
        type=str,
        default=None,
        help="Consistency curve summary CSV path (default: same as --consistency-plot with .csv).",
    )
    parser.add_argument(
        "--multi_run_jsonl",
        type=str,
        default="gpqa_streaming_sweep_kmeans_multi64.jsonl",
        help="(Like gpqa_offline) Export multi-run curves+stats to JSONL (optional; requires --sweep).",
    )
    parser.add_argument(
        "--multi-runs",
        type=int,
        default=3,
        help="Number of repeated sweeps with shuffled answer pools (requires --sweep).",
    )
    parser.add_argument(
        "--multi-pool-size",
        type=int,
        default=64,
        help="multi-run 模式：每题先 shuffle answers 顺序，再取前 K 条作为本次 run 的 answers pool（默认 64）",
    )
    parser.add_argument(
        "--multi-relabel-mv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="multi-run 模式：是否用 subsample 后的 answers pool 做 MV 并作为本次 run 的 gold label（等价旧方案的 final）",
    )
    parser.add_argument(
        "--oracle-kmeans-k",
        type=int,
        default=5,
        help="Number of kmeans difficulty buckets for oracle setting.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    all_questions = base.load_gpqa_jsonl(str(data_path), conf_metric=str(args.conf_metric))
    if len(all_questions) <= args.train_size:
        raise ValueError(f"Dataset needs more than {args.train_size} questions to create a test split.")

    train_questions = all_questions[: args.train_size]
    test_questions = all_questions[args.train_size :]

    # predictor training and evaluation setup
    stats, plan = base.train_and_build_budget_plan(
        train_questions,
        B_bar=args.average_budget,
        k_max_curve=args.k_max_curve,
        subsample4_draws=args.subsample4_draws,
        curve_mc_trials=args.curve_mc_trials,
        B_max=args.max_per_question,
        rng_seed=args.rng_seed,
    )
    expected_budget = float(np.sum(stats.pi_t * plan.B_t))

    print("Training complete.")
    print("pi_t:", stats.pi_t)
    print("B_t:", plan.B_t)
    print(f"Expected average budget (predictor): {expected_budget:.2f}")

    # predictor/baseline/oracle evaluation
    predictor_metrics, _results = base.evaluate_streaming(test_questions, plan, rng_seed=args.rng_seed)
    baseline_metrics = base.evaluate_fixed_budget_majority(
        test_questions, per_question_budget=args.average_budget, rng_seed=args.rng_seed
    )

    # Oracle preparation (train/test (a,b) fits). Compute once and reuse.
    # Optionally fit oracle buckets on full data for a stronger (leaky) upper bound.
    oracle_fit_questions = all_questions if bool(getattr(args, "oracle_fit_all", False)) else train_questions
    train_params_oracle = base.compute_question_param_map(
        oracle_fit_questions,
        k_max_curve=args.k_max_curve,
        curve_mc_trials=args.curve_mc_trials,
    )
    test_params_oracle = base.compute_question_param_map(
        test_questions,
        k_max_curve=args.k_max_curve,
        curve_mc_trials=args.curve_mc_trials,
    )
    oracle_model_pre = build_oracle_difficulty_model(
        train_params_oracle,
        k=int(args.oracle_kmeans_k),
        random_seed=int(args.rng_seed),
    )

    (
        oracle_metrics,
        oracle_expected,
        oracle_budget_by_bucket,
        oracle_model,
        oracle_test_params,
    ) = evaluate_oracle_setting(
        train_questions=train_questions,
        test_questions=test_questions,
        add_conf=bool(args.add_conf),
        conf_variant=str(args.conf_variant),
        average_budget=args.average_budget,
        max_per_question=args.max_per_question,
        k_max_curve=args.k_max_curve,
        curve_mc_trials=args.curve_mc_trials,
        oracle_model_override=oracle_model_pre,
        oracle_test_params_override=test_params_oracle,
        kmeans_k=int(args.oracle_kmeans_k),
        kmeans_seed=int(args.rng_seed),
    )
    if oracle_metrics and oracle_model is not None and oracle_budget_by_bucket is not None and oracle_expected is not None:
        print("Oracle kmeans centers (a,b) and weights/budgets:")
        for idx, ((a_c, b_c), p_c, B_c) in enumerate(
            zip(oracle_model.centers_ab.tolist(), oracle_model.probs.tolist(), oracle_budget_by_bucket.tolist())
        ):
            print(f"  bucket {idx+1}: center_a={a_c:.6g}, center_b={b_c:.6g}, prob={p_c:.4f}, budget={int(B_c)}")
        print("Oracle expected budget (per question):", f"{oracle_expected:.2f}")
        print("Oracle budget by bucket:", [int(x) for x in oracle_budget_by_bucket.tolist()])

    print(f"\nPredictor accuracy: {predictor_metrics['accuracy']:.4f}")
    print(f"Predictor consistency: {predictor_metrics.get('consistency', float('nan')):.4f}")
    print(f"Predictor total budget: {predictor_metrics['total_budget_used']:.1f}")
    print(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline consistency: {baseline_metrics.get('consistency', float('nan')):.4f}")
    print(f"Baseline total budget: {baseline_metrics['total_budget_used']:.1f}")
    if oracle_metrics:
        print(
            f"Oracle accuracy: {oracle_metrics['accuracy']:.4f} "
            f"(skipped {int(oracle_metrics.get('skipped', 0))})"
        )
        print(f"Oracle consistency: {oracle_metrics.get('consistency', float('nan')):.4f}")
        print(f"Oracle total budget: {oracle_metrics['total_budget_used']:.1f}")

    # multi-run sweep (reuse base utilities; oracle uses this file's sweep)
    if args.sweep:
        if args.multi_runs > 1:
            # IMPORTANT: when multi-runs is enabled, run0 should also use the same
            # "shuffle 128 -> take first K -> MV as gold" protocol; otherwise the
            # printed sweep rows won't match the multi-run definition.
            sweep_runs: List[Tuple[int, Sequence[Dict[str, object]]]] = []
            for run_idx in range(args.multi_runs):
                rng = random.Random(args.rng_seed + run_idx)
                train_q_run = base.shuffle_subsample_and_relabel_question_records(
                    train_questions,
                    rng,
                    pool_size=int(args.multi_pool_size),
                    relabel_with_pool_mv=bool(args.multi_relabel_mv),
                )
                test_q_run = base.shuffle_subsample_and_relabel_question_records(
                    test_questions,
                    rng,
                    pool_size=int(args.multi_pool_size),
                    relabel_with_pool_mv=bool(args.multi_relabel_mv),
                )

                oracle_fit_q_run = (
                    (list(train_q_run) + list(test_q_run))
                    if bool(getattr(args, "oracle_fit_all", False))
                    else train_q_run
                )

                # oracle preparation
                train_params_run = base.compute_question_param_map(
                    oracle_fit_q_run,
                    k_max_curve=args.k_max_curve,
                    curve_mc_trials=args.curve_mc_trials,
                )
                test_params_run = base.compute_question_param_map(
                    test_q_run,
                    k_max_curve=args.k_max_curve,
                    curve_mc_trials=args.curve_mc_trials,
                )
                oracle_model_run = build_oracle_difficulty_model(
                    train_params_run, k=int(args.oracle_kmeans_k), random_seed=int(args.rng_seed + run_idx)
                )

                # predictor preparation
                stats_run, _plan_run = base.train_and_build_budget_plan(
                    train_q_run,
                    B_bar=args.average_budget,
                    k_max_curve=args.k_max_curve,
                    subsample4_draws=args.subsample4_draws,
                    curve_mc_trials=args.curve_mc_trials,
                    B_max=args.max_per_question,
                    rng_seed=args.rng_seed + run_idx,
                )

                sweep_rows_run = sweep_average_budgets(
                    stats_run,
                    test_q_run,
                    add_conf=bool(args.add_conf),
                    conf_variant=str(args.conf_variant),
                    sweep_max=args.sweep_max,
                    B_max=args.max_per_question,
                    rng_seed=args.rng_seed + run_idx,
                    oracle_model=oracle_model_run,
                    oracle_test_params=test_params_run if test_params_run else None,
                )
                sweep_runs.append((run_idx, sweep_rows_run))

            # Use run0 rows for printing / single-run plot
            sweep_rows = sweep_runs[0][1] if sweep_runs else []
            if sweep_rows:
                title_suffix = (
                    f"train={len(train_questions)}, test={len(test_questions)}, avg≤{args.sweep_max}, "
                    f"pool={int(args.multi_pool_size)}, relabel_mv={bool(args.multi_relabel_mv)}"
                )
                print("\nSweep results (avg_budget: pred_acc, base_acc, expected, B_t):")
                for row in sweep_rows:
                    print(
                        f"  {int(row['average_budget']):2d}: "
                        f"pred={row['predictor_accuracy']:.4f}, "
                        f"base={row['baseline_accuracy']:.4f}, "
                        f"expected={row['predictor_expected']:.2f}, "
                        f"B_t={row['budget_plan']}"
                    )
                    if row.get("oracle_accuracy") is not None:
                        print(
                            "     oracle_acc="
                            f"{row['oracle_accuracy']:.4f}, oracle_total={row['oracle_total']:.1f}, "
                            f"oracle_expected={row.get('oracle_expected', float('nan')):.2f}"
                        )

            if sweep_runs:
                summaries = base.aggregate_multi_run_accuracy_stats(sweep_runs)
                print("\nMulti-run predictor summary (avg_budget, total_mean±std, acc_mean±std, runs):")
                for entry in summaries.get("predictor", []):
                    print(
                        f"  avg~{entry['avg_budget_rounded']:.0f}, "
                        f"total={entry['total_mean']:.1f}±{entry['total_std']:.1f}, "
                        f"acc={entry['accuracy_mean']:.4f}±{entry['accuracy_std']:.4f} "
                        f"(n={entry['num_runs']})"
                    )
                print("Multi-run baseline summary (avg_budget, total_mean±std, acc_mean±std, runs):")
                for entry in summaries.get("baseline", []):
                    print(
                        f"  avg~{entry['avg_budget_rounded']:.0f}, "
                        f"total={entry['total_mean']:.1f}±{entry['total_std']:.1f}, "
                        f"acc={entry['accuracy_mean']:.4f}±{entry['accuracy_std']:.4f} "
                        f"(n={entry['num_runs']})"
                    )

                # Reference-style plots + CSVs for BOTH metrics (accuracy + consistency)
                curve_runs_acc = base._sweep_runs_to_curve_runs_total(sweep_runs, metric="accuracy")
                curve_runs_cons = base._sweep_runs_to_curve_runs_total(sweep_runs, metric="consistency")
                base.plot_accuracy_multi_run_curves(
                    sweep_runs,
                    args.accuracy_plot,
                    csv_path=(args.accuracy_csv if args.accuracy_csv else None),
                )
                base.plot_consistency_multi_run_curves(
                    sweep_runs,
                    args.consistency_plot,
                    csv_path=(args.consistency_csv if args.consistency_csv else None),
                )
                if args.multi_run_jsonl:
                    base.export_multi_run_curves_jsonl(
                        curve_runs_cons,
                        curve_runs_acc,
                        str(args.multi_run_jsonl),
                        sweep_runs=sweep_runs,
                    )
        else:
            sweep_rows = sweep_average_budgets(
                stats,
                test_questions,
                add_conf=bool(args.add_conf),
                conf_variant=str(args.conf_variant),
                sweep_max=args.sweep_max,
                B_max=args.max_per_question,
                rng_seed=args.rng_seed,
                oracle_model=oracle_model,
                oracle_test_params=oracle_test_params if oracle_test_params else None,
            )
            if sweep_rows:
                title_suffix = f"train={len(train_questions)}, test={len(test_questions)}, avg≤{args.sweep_max}"
                print("\nSweep results (avg_budget: pred_acc, base_acc, expected, B_t):")
                for row in sweep_rows:
                    print(
                        f"  {int(row['average_budget']):2d}: "
                        f"pred={row['predictor_accuracy']:.4f}, "
                        f"base={row['baseline_accuracy']:.4f}, "
                        f"expected={row['predictor_expected']:.2f}, "
                        f"B_t={row['budget_plan']}"
                    )
                    if row.get("oracle_accuracy") is not None:
                        print(
                            "     oracle_acc="
                            f"{row['oracle_accuracy']:.4f}, oracle_total={row['oracle_total']:.1f}, "
                            f"oracle_expected={row.get('oracle_expected', float('nan')):.2f}"
                        )

                curve_runs_acc = base._sweep_runs_to_curve_runs_total([(0, sweep_rows)], metric="accuracy")
                curve_runs_cons = base._sweep_runs_to_curve_runs_total([(0, sweep_rows)], metric="consistency")
                base.plot_accuracy_multi_run_curves(
                    [(0, sweep_rows)],
                    args.accuracy_plot,
                    csv_path=(args.accuracy_csv if args.accuracy_csv else None),
                )
                base.plot_consistency_multi_run_curves(
                    [(0, sweep_rows)],
                    args.consistency_plot,
                    csv_path=(args.consistency_csv if args.consistency_csv else None),
                )
                if args.multi_run_jsonl:
                    base.export_multi_run_curves_jsonl(
                        curve_runs_cons,
                        curve_runs_acc,
                        str(args.multi_run_jsonl),
                        sweep_runs=[(0, sweep_rows)],
                    )


if __name__ == "__main__":
    main()
