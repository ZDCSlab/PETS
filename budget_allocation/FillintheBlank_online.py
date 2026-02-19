#!/usr/bin/env python3
"""
AIME streaming (bucketed allocation) adapted from `mmlu_streaming.py`.

Key difference:
  - AIME is fill-in (free-form) answers with potentially many unique strings.
  - We "treat it as multiple-choice" by mapping per-question answer strings into
    at most `max_options` distinct options. Once a question has already seen
    `max_options` distinct answers, any new unseen answer is **discarded** (not
    recorded for voting / stats). This mirrors the logic in `aime_offline.py`
    (see the per-question `option_maps` mapping used during consumption).

Notes:
  - Budget accounting: each model response costs 1 budget even if it is discarded.
  - Voting tie-break for AIME uses *earliest occurrence* among tied answers (same
    as `aime_offline.py`), not lexicographic min.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm
from oracle_kmeans_common import (
    OracleDifficultyModelKMeans,
    build_oracle_difficulty_model_from_params,
    greedy_budget_allocation_oracle_common,
    locate_param_bin_oracle as locate_param_bin_oracle_common,
)
from plots.online_sweep import (
    plot_accuracy_multi_run_curves,
    plot_consistency_multi_run_curves,
)

K0 = 4  # warm-up samples


@dataclass
class PerQuestionFit:
    """Per-question outputs in training."""

    qid: str
    a_q: float
    b_q: float
    pi_q: np.ndarray


@dataclass
class BucketStats:
    """Global bucket statistics after training aggregation."""

    pi_t: np.ndarray
    a_t: np.ndarray
    b_t: np.ndarray


@dataclass
class BudgetPlan:
    """Budget plan B_t for each bucket."""

    B_t: np.ndarray


def exact_prob_pick_argmax_multinom4(theta4, k, log_fact=None, *, mc_samples=20000, rng=None):
    """Probability majority vote picks argmax(theta) with tie uniform.

    For >4 options the exact multinomial enumeration explodes combinatorially;
    this implementation uses a Monte Carlo estimate in that case.
    """
    theta = np.asarray(theta4, dtype=float)
    theta = theta / theta.sum()
    if np.any(theta <= 0):
        eps = 1e-12
        theta = np.clip(theta, eps, None)
        theta = theta / theta.sum()
    n_opt = theta.size
    if n_opt < 2:
        raise ValueError("theta must have length >= 2.")

    argmax_idx = int(np.argmax(theta))
    if argmax_idx != 0:
        theta = np.concatenate([[theta[argmax_idx]], theta[:argmax_idx], theta[argmax_idx + 1 :]])

    if n_opt == 4:
        if log_fact is None or len(log_fact) <= k:
            log_fact = [math.lgamma(i + 1) for i in range(k + 1)]
        log_theta = np.log(theta)

        log_total = -math.inf
        for c0 in range(k + 1):
            for c1 in range(k - c0 + 1):
                for c2 in range(k - c0 - c1 + 1):
                    c3 = k - c0 - c1 - c2
                    counts = (c0, c1, c2, c3)
                    m = max(counts)
                    winners = [i for i, c in enumerate(counts) if c == m]
                    if 0 not in winners:
                        continue
                    log_coef = log_fact[k] - (log_fact[c0] + log_fact[c1] + log_fact[c2] + log_fact[c3])
                    log_p = log_coef + (
                        c0 * log_theta[0] + c1 * log_theta[1] + c2 * log_theta[2] + c3 * log_theta[3]
                    )
                    log_term = log_p - math.log(len(winners))
                    log_total = np.logaddexp(log_total, log_term)
        return float(math.exp(log_total))

    rng = np.random.default_rng(rng)
    mc_samples = int(mc_samples)
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive.")
    counts = rng.multinomial(k, theta, size=mc_samples)
    top = counts.max(axis=1)
    winners = counts == top[:, None]
    tie_sizes = winners.sum(axis=1)
    wins = winners[:, 0]
    contrib = np.where(wins, 1.0 / tie_sizes, 0.0)
    return float(contrib.mean())


def fit_2param_probit_sqrtk(
    A: np.ndarray,
    k_min: int = 3,
    k_max: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit A(k)=Phi(a*sqrt(k)+b) using least squares in probit space."""
    K = len(A) - 1
    if k_max is None:
        k_max = K
    k_max = min(k_max, K)

    ks = np.arange(1, K + 1)
    mask = (ks >= k_min) & (ks <= k_max)
    x = np.sqrt(ks[mask].astype(float))
    y = norm.ppf(np.clip(A[ks[mask]], 1e-12, 1 - 1e-12))

    X = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def A_probit(k: int, a: float, b: float) -> float:
    """Bucket-level / question-level A(k)=Phi(a*sqrt(k)+b)."""
    if k <= 0:
        return 0.5
    return float(norm.cdf(a * math.sqrt(k) + b))


def estimate_accuracy_curve_from_pool_oracle(
    answers: Sequence,
    correct: object,
    k_max: int,
    *,
    num_trials: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Estimate A_q(k) for k=1..k_max using multinomial majority probability."""
    ctr = Counter(answers)
    if not ctr:
        raise ValueError("empty answer pool")
    if correct not in ctr:
        ctr[correct] = 0

    others = [opt for opt in ctr.keys() if opt != correct]
    others = sorted(others, key=lambda x: ctr[x], reverse=True)[:9]

    while len(others) < 9:
        placeholder = f"__dummy{len(others)}"
        if placeholder not in ctr:
            ctr[placeholder] = 0
        others.append(placeholder)

    counts = [ctr[correct]] + [ctr[o] for o in others]
    theta = np.asarray(counts, dtype=float)

    A = np.zeros(k_max + 1, dtype=float)
    A[0] = 0.5
    log_fact_cache = [math.lgamma(i + 1) for i in range(k_max + 1)]
    for k in range(1, k_max + 1):
        A[k] = exact_prob_pick_argmax_multinom4(theta, k, log_fact_cache)
    return A


def plot_sweep_results(
    rows: Sequence[Dict[str, object]],
    plot_path: Optional[Path],
    *,
    title_suffix: str = "",
) -> None:
    """Plot predictor and baseline accuracy vs total budget and save figure."""
    if not rows or not plot_path:
        return

    predictor_x = [row["predictor_total"] for row in rows]
    predictor_y = [row["predictor_accuracy"] for row in rows]
    baseline_x = [row["baseline_total"] for row in rows]
    baseline_y = [row["baseline_accuracy"] for row in rows]
    oracle_points = [
        (row.get("oracle_total"), row.get("oracle_accuracy"))
        for row in rows
        if row.get("oracle_total") is not None and row.get("oracle_accuracy") is not None
    ]
    oracle_x = [pt[0] for pt in oracle_points]
    oracle_y = [pt[1] for pt in oracle_points]

    plt.figure(figsize=(8, 5))
    predictor_color = "#d55e00"
    baseline_color = "#0072b2"
    oracle_color = "#009e73"
    plt.plot(
        predictor_x,
        predictor_y,
        marker="o",
        color=predictor_color,
        label="Bucketed predictor",
    )
    plt.plot(
        baseline_x,
        baseline_y,
        marker="o",
        color=baseline_color,
        label="Fixed baseline",
    )
    if oracle_x and oracle_y:
        plt.plot(
            oracle_x,
            oracle_y,
            marker="o",
            color=oracle_color,
            label="Oracle difficulty",
        )
    plt.xlabel("Total budget consumed")
    plt.ylabel("Consistency Rate")
    title_line = "Consistency Rate vs Total Budget (streaming)"
    if title_suffix:
        title_line += f"\n{title_suffix}"
    plt.title(title_line)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved sweep plot to {plot_path}")


class _BaseNamespace:
    """Local compatibility namespace replacing `import mmlu_streaming as base`."""


base = _BaseNamespace()
base.K0 = K0
base.PerQuestionFit = PerQuestionFit
base.BucketStats = BucketStats
base.BudgetPlan = BudgetPlan
base.A_probit = A_probit
base.estimate_accuracy_curve_from_pool_oracle = estimate_accuracy_curve_from_pool_oracle
base.fit_2param_probit_sqrtk = fit_2param_probit_sqrtk
base.plot_sweep_results = plot_sweep_results
def stable_marginal_gain_probit_sqrtk(
    cur_k: int,
    a: float,
    b: float,
    *,
    step: int = 1,
) -> float:
    """Stable Δ = A(cur_k+step) - A(cur_k) for probit-sqrt(k).

    Avoids losing tiny gains when `norm.cdf` saturates by using:
      cdf(x2) - cdf(x1) = sf(x1) - sf(x2)
    and computing the difference in log-space.

    Falls back to `base.A_probit` difference if scipy/logsf is unavailable.
    """
    cur_k = int(cur_k)
    step = int(step)
    if step <= 0:
        return 0.0

    next_k = cur_k + step
    if cur_k < 0 or next_k < 0:
        return 0.0

    a = float(a)
    b = float(b)
    x1 = float(a * math.sqrt(float(cur_k)) + b)
    x2 = float(a * math.sqrt(float(next_k)) + b)

    try:
        from scipy.stats import norm  # type: ignore

        logsf1 = float(norm.logsf(x1))
        logsf2 = float(norm.logsf(x2))
        if not (math.isfinite(logsf1) and math.isfinite(logsf2)):
            raise ValueError("non-finite logsf")

        if logsf1 < logsf2:
            logsf1, logsf2 = logsf2, logsf1

        sf1 = math.exp(logsf1)
        return float(sf1 * (-math.expm1(logsf2 - logsf1)))
    except Exception:
        try:
            return float(base.A_probit(next_k, a, b) - base.A_probit(cur_k, a, b))
        except Exception:
            return 0.0


NUM_BUCKETS_SIMPLIFIED = 2
ORACLE_KMEANS_K = 5


@dataclass(frozen=True)
class QuestionRecord:
    """AIME question record with two labels:

    - correct: dataset gold label (for accuracy), e.g. `correct_answer`
    - final: pseudo label (for self-consistency), e.g. `final` (fallbacks allowed)
    """

    qid: str
    answers: Sequence[str]
    correct: Optional[str] = None
    final: Optional[str] = None
    confs: Optional[Sequence[object]] = None


def _extract_mean_confidence(entry: object) -> float:
    """Backward-compatible mean_confidence extractor (kept for older callers)."""
    return _extract_confidence(entry, metric="mean")


# Mirror gpqa_offline/gpqa_streaming naming; keep minimal + backward compatible.
CONF_METRIC_KEYS: Dict[str, str] = {
    "Conf": "mean_confidence",
    "mean": "mean_confidence",
    "tail": "tail_2048_mean_conf",
    "bottom": "bottom_0.1_sliding_2048_mean_conf",
}


def _extract_confidence(entry: object, *, metric: str = "mean") -> float:
    """Best-effort extract a scalar confidence value from a trace_confidence entry.

    - If `metric` is in CONF_METRIC_KEYS (e.g. mean/tail/bottom), use the mapped key.
    - Otherwise treat `metric` as a raw dict key to read.
    - Supports nested schema: {"conf_summary": {...}} by flattening.
    - Falls back to mean_confidence / Conf / 1.0 when missing.
    """
    if entry is None:
        return 1.0
    if isinstance(entry, (int, float)):
        try:
            v = float(entry)
            return v if math.isfinite(v) else 1.0
        except Exception:
            return 1.0
    if not isinstance(entry, dict):
        return 1.0

    src = dict(entry)
    nested = src.get("conf_summary")
    if isinstance(nested, dict):
        src = {**src, **nested}

    key = CONF_METRIC_KEYS.get(str(metric), str(metric))
    v = src.get(key)
    if v is None:
        v = src.get("mean_confidence")
        if v is None:
            v = src.get("Conf")
    try:
        vf = float(v) if v is not None else 1.0
        return vf if math.isfinite(vf) else 1.0
    except Exception:
        return 1.0


def weighted_vote_majority_earliest(answers: Sequence[str], weights: Sequence[object]) -> str:
    """Weighted majority vote; tie-break by earliest occurrence among tied winners."""
    if not answers:
        return ""
    weighted_cnt: Dict[str, float] = defaultdict(float)
    for i, a in enumerate(answers):
        w_obj = weights[i] if i < len(weights) else 1.0
        try:
            w = float(w_obj)
            if not math.isfinite(w):
                w = 1.0
        except Exception:
            w = 1.0
        weighted_cnt[_norm_str(a)] += w
    if not weighted_cnt:
        return ""
    max_w = max(weighted_cnt.values())
    tied = {a for a, w in weighted_cnt.items() if w == max_w}
    first_pos: Dict[str, int] = {}
    for idx, ans in enumerate(answers):
        a = _norm_str(ans)
        if a in tied and a not in first_pos:
            first_pos[a] = idx
    return min(tied, key=lambda a: first_pos.get(a, 10**18)) if tied else ""


def weighted_vote_variant_majority_earliest(
    answers: Sequence[str],
    weights: Sequence[object],
    *,
    variant: str = "weighted",
) -> str:
    """Weighted voting variants (mirrors gpqa_offline.py patterns).

    variant:
      - "weighted": use all samples
      - "top10"/"top30"/"top50"/"top70"/"top90": keep only the top-X% samples by weight

    Tie-break among weight-tied winners uses earliest occurrence (AIME convention).
    """
    if not answers:
        return ""
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

    indexed: List[Tuple[int, str, float]] = []
    for idx, ans in enumerate(answers):
        w_obj = weights[idx] if idx < len(weights) else 1.0
        try:
            w = float(w_obj)
            if not math.isfinite(w):
                w = 1.0
        except Exception:
            w = 1.0
        indexed.append((idx, _norm_str(ans), w))

    if not indexed:
        return ""

    keep_k = max(1, int(round(len(indexed) * p)))
    keep_k = min(keep_k, len(indexed))
    kept = sorted(indexed, key=lambda t: t[2], reverse=True)[:keep_k]

    weighted_cnt: Dict[str, float] = defaultdict(float)
    for _idx, ans, w in kept:
        weighted_cnt[_norm_str(ans)] += float(w)
    if not weighted_cnt:
        return ""
    max_w = max(weighted_cnt.values())
    tied = {a for a, w in weighted_cnt.items() if w == max_w}
    first_pos: Dict[str, int] = {}
    for idx, ans in enumerate(answers):
        a = _norm_str(ans)
        if a in tied and a not in first_pos:
            first_pos[a] = idx
    return min(tied, key=lambda a: first_pos.get(a, 10**18)) if tied else ""


def _training_label(q: QuestionRecord) -> Optional[str]:
    """Label used for training fits / bucket calibration.

    Mirror gpqa_streaming behavior: prefer pseudo label (`final`) if present, else fall back to gold (`correct`).
    """

    return q.final if q.final is not None else q.correct


# -----------------------------
# Oracle difficulty via KMeans (k=2)
# -----------------------------

def fit_question_difficulty_params(
    question: QuestionRecord,
    *,
    max_options: int,
    k_max_curve: int = 32,
    curve_mc_trials: int = 4000,
) -> Optional[Tuple[float, float]]:
    """Fit (a,b) parameters from the full answer pool for oracle evaluation (AIME, capped by max_options)."""
    train_label = _training_label(question)
    if train_label is None or not question.answers:
        return None
    capped_answers = filter_pool_by_max_unique(question.answers, max_options=int(max_options))
    if not capped_answers:
        return None
    try:
        A = base.estimate_accuracy_curve_from_pool_oracle(
            capped_answers,
            train_label,
            k_max=int(k_max_curve),
            num_trials=int(curve_mc_trials),
        )
        a_q, b_q = base.fit_2param_probit_sqrtk(A, k_min=3, k_max=min(int(k_max_curve), len(A) - 1))
    except Exception:
        return None
    return float(a_q), float(b_q)


def compute_question_param_map(
    questions: Sequence[QuestionRecord],
    *,
    max_options: int,
    k_max_curve: int = 32,
    curve_mc_trials: int = 4000,
) -> Dict[str, Tuple[float, float]]:
    """Compute (a,b) fits for a list of questions (oracle fits from full pools)."""
    params: Dict[str, Tuple[float, float]] = {}
    for q in questions:
        fitted = fit_question_difficulty_params(
            q,
            max_options=int(max_options),
            k_max_curve=int(k_max_curve),
            curve_mc_trials=int(curve_mc_trials),
        )
        if fitted is not None:
            params[q.qid] = fitted
    return params


def build_oracle_difficulty_model(
    train_params: Dict[str, Tuple[float, float]],
    *,
    k: int = ORACLE_KMEANS_K,
    random_seed: int = 0,
) -> Optional[OracleDifficultyModelKMeans]:
    """Fit KMeans(k=2) model on training (a,b) and return ordered bucket centers + probs."""
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
    min_budget: int = 1,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Greedy budget allocation over KMeans oracle buckets.

    Properties:
      - Hard constraint: expected budget never exceeds `average_budget` (up to `eps`).
      - Near-tightness: if marginal gains become numerically ~0 (e.g., probit saturates),
        we still try to consume remaining slack via a second "fill-to-tight" phase.
    """
    def marginal_gain(t: int, cur: int, centers: np.ndarray) -> float:
        a, b = float(centers[t, 0]), float(centers[t, 1])
        return float(stable_marginal_gain_probit_sqrtk(cur, a, b, step=1))

    return greedy_budget_allocation_oracle_common(
        model,
        average_budget=float(average_budget),
        B_max=int(B_max),
        min_budget=int(min_budget),
        marginal_gain_fn=marginal_gain,
        eps=float(eps),
    )


def locate_param_bin_oracle(a_value: float, b_value: float, model: OracleDifficultyModelKMeans) -> int:
    """Assign (a,b) to nearest KMeans center and return ordered bucket index [0..k-1]."""
    return int(locate_param_bin_oracle_common(float(a_value), float(b_value), model))


def evaluate_oracle_setting(
    train_questions: Sequence[QuestionRecord],
    test_questions: Sequence[QuestionRecord],
    *,
    average_budget: float,
    max_per_question: int,
    max_options: int,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    k_max_curve: int = 32,
    curve_mc_trials: int = 4000,
    oracle_model_override: Optional[OracleDifficultyModelKMeans] = None,
    oracle_test_params_override: Optional[Dict[str, Tuple[float, float]]] = None,
    kmeans_k: int = ORACLE_KMEANS_K,
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
        train_params = compute_question_param_map(
            train_questions,
            max_options=int(max_options),
            k_max_curve=int(k_max_curve),
            curve_mc_trials=int(curve_mc_trials),
        )
        test_params = compute_question_param_map(
            test_questions,
            max_options=int(max_options),
            k_max_curve=int(k_max_curve),
            curve_mc_trials=int(curve_mc_trials),
        )
        oracle_model = build_oracle_difficulty_model(train_params, k=int(kmeans_k), random_seed=int(kmeans_seed))

    if oracle_model is None or not test_params:
        return None, None, None, oracle_model, test_params or {}

    budget_by_bucket, _ = greedy_budget_allocation_oracle(
        oracle_model,
        average_budget=float(average_budget),
        B_max=int(max_per_question),
        # Align budget accounting with OKG/Base: enforce at least K0 attempts.
        min_budget=int(base.K0),
    )
    expected_budget = float(np.sum(oracle_model.probs * budget_by_bucket))

    evaluated_acc = 0
    correct_acc = 0
    evaluated_cons = 0
    correct_cons = 0
    correct_acc_conf = 0
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

        attempted = min(int(budget), len(q.answers))
        total_budget_used += float(attempted)
        per_bucket_budget[bucket] += float(attempted)

        option_map: Dict[str, int] = {}
        accepted: List[str] = []
        accepted_w: List[float] = []
        conf_pool: List[object] = list(q.confs) if q.confs is not None else []
        for i in range(attempted):
            a = _norm_str(q.answers[i])
            status, _opt = map_fillin_answer_to_option(a, option_map, max_options=int(max_options))
            if status == "accepted":
                accepted.append(a)
                accepted_w.append(_extract_mean_confidence(conf_pool[i] if i < len(conf_pool) else None))

        if not accepted:
            skipped += 1
            continue
        pred = vote_majority_earliest(accepted)
        pred_conf = (
            weighted_vote_variant_majority_earliest(accepted, accepted_w, variant=str(conf_variant)) if add_conf else ""
        )

        if q.correct is not None:
            evaluated_acc += 1
            if _norm_str(pred) == _norm_str(q.correct):
                correct_acc += 1
            if add_conf and _norm_str(pred_conf) == _norm_str(q.correct):
                correct_acc_conf += 1
        if q.final is not None:
            evaluated_cons += 1
            if _norm_str(pred) == _norm_str(q.final):
                correct_cons += 1
            if add_conf:
                gold_conf = _pseudo_label_conf_from_full_pool(
                    q,
                    max_options=int(max_options),
                    conf_variant=str(conf_variant),
                )
                if _norm_str(pred_conf) == _norm_str(gold_conf):
                    correct_cons_conf += 1
        if q.correct is None and q.final is None:
            skipped += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    metrics = {
        "accuracy": float(accuracy),
        "consistency": float(consistency),
        "skipped": float(skipped),
        "total_budget_used": float(total_budget_used),
        "per_bucket_budget": per_bucket_budget,
    }
    if add_conf:
        accuracy_conf = correct_acc_conf / evaluated_acc if evaluated_acc else float("nan")
        consistency_conf = correct_cons_conf / evaluated_cons if evaluated_cons else float("nan")
        metrics.update({"accuracy_conf": float(accuracy_conf), "consistency_conf": float(consistency_conf)})
    return metrics, expected_budget, budget_by_bucket, oracle_model, test_params

def _norm_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _pseudo_label_conf_from_full_pool(
    q: QuestionRecord,
    *,
    max_options: int,
    conf_variant: str,
) -> str:
    """Run-specific pseudo label for confidence-weighted consistency.

    Uses the full pool (e.g. 64 answers + confidences) and applies the same
    fill-in->option mapping as evaluation to respect the max_options cap.
    """
    if not q.answers:
        return _norm_str(q.final) if q.final is not None else ""

    option_map: Dict[str, int] = {}
    accepted: List[str] = []
    accepted_w: List[float] = []
    conf_pool: List[object] = list(q.confs) if q.confs is not None else []
    for i, a_raw in enumerate(q.answers):
        a = _norm_str(a_raw)
        status, _opt = map_fillin_answer_to_option(a, option_map, max_options=int(max_options))
        if status == "accepted":
            accepted.append(a)
            accepted_w.append(_extract_mean_confidence(conf_pool[i] if i < len(conf_pool) else None))

    if not accepted:
        return _norm_str(q.final) if q.final is not None else ""

    try:
        pseudo = weighted_vote_variant_majority_earliest(accepted, accepted_w, variant=str(conf_variant))
    except Exception:
        pseudo = ""

    pseudo = _norm_str(pseudo)
    if pseudo != "":
        return pseudo
    return _norm_str(q.final) if q.final is not None else ""


def bucket2_from_samples4(samples4: Sequence[str]) -> int:
    """2-bucket difficulty:

    - bucket 1: 4 warm-up accepted answers are all identical
    - bucket 2: otherwise (there exists at least one disagreement)
    """
    if len(samples4) != int(base.K0):
        raise ValueError(f"samples4 must have length {int(base.K0)}")
    s = [_norm_str(x) for x in samples4]
    # empty strings should not appear in accepted, but treat them as normal tokens anyway
    return 1 if len(set(s)) == 1 else 2


def estimate_pi_q2_via_subsample4(
    answers: Sequence[str],
    *,
    num_draws: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Estimate pi_q over 2 buckets via repeated subsampling of 4 answers (without replacement)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(answers)
    if n < int(base.K0):
        raise ValueError(f"need at least {int(base.K0)} answers to subsample {int(base.K0)}")

    counts = np.zeros(NUM_BUCKETS_SIMPLIFIED, dtype=float)
    for _ in range(int(num_draws)):
        idx = rng.choice(n, size=int(base.K0), replace=False)
        s4 = [_norm_str(answers[i]) for i in idx]
        t = bucket2_from_samples4(s4)  # 1..2
        counts[t - 1] += 1.0
    pi_q = counts / (counts.sum() + 1e-12)
    return pi_q


def aggregate_bucket2_stats(fits: Sequence[base.PerQuestionFit]) -> base.BucketStats:
    """Aggregate per-question fits into 2-bucket stats (pi_t, a_t, b_t)."""
    if not fits:
        raise ValueError("empty fits")
    pi_stack = np.stack([np.asarray(f.pi_q, dtype=float) for f in fits], axis=0)  # (Q,2)
    if pi_stack.ndim != 2 or pi_stack.shape[1] != NUM_BUCKETS_SIMPLIFIED:
        raise ValueError(f"expected pi_q shape (2,), got {pi_stack.shape}")

    pi_t = pi_stack.mean(axis=0)
    pi_t = pi_t / (pi_t.sum() + 1e-12)

    a_qs = np.array([float(f.a_q) for f in fits], dtype=float)  # (Q,)
    b_qs = np.array([float(f.b_q) for f in fits], dtype=float)  # (Q,)

    weights = pi_stack  # (Q,2)
    denom = weights.sum(axis=0) + 1e-12
    a_t = (weights.T @ a_qs) / denom
    b_t = (weights.T @ b_qs) / denom
    return base.BucketStats(pi_t=np.asarray(pi_t, dtype=float), a_t=np.asarray(a_t, dtype=float), b_t=np.asarray(b_t, dtype=float))


def solve_budget_plan_greedy_marginal_anyk(
    stats: base.BucketStats,
    *,
    B_bar: float,
    B_max: int,
    k0: int,
) -> base.BudgetPlan:
    """Greedy allocation under average budget, generalized to any number of buckets."""
    pi = np.asarray(stats.pi_t, dtype=float)
    a_t = np.asarray(stats.a_t, dtype=float)
    b_t = np.asarray(stats.b_t, dtype=float)
    k = int(pi.size)
    if k == 0:
        return base.BudgetPlan(B_t=np.zeros((0,), dtype=int))
    if a_t.size != k or b_t.size != k:
        raise ValueError("stats arrays must have matching sizes")

    B = np.full(k, int(k0), dtype=int)
    used_budget = float(np.sum(pi * B))
    if used_budget >= float(B_bar):
        return base.BudgetPlan(B_t=B)

    import heapq

    def marginal_gain(idx: int, step: int = 1) -> float:
        cur = int(B[idx])
        if cur + int(step) > int(B_max):
            return -math.inf
        return float(stable_marginal_gain_probit_sqrtk(cur, float(a_t[idx]), float(b_t[idx]), step=int(step)))

    heap: List[Tuple[float, int]] = []
    for t in range(k):
        gain = marginal_gain(t, step=1)
        if gain > 0:
            heapq.heappush(heap, (-gain, t))

    eps = 1e-12
    while heap and used_budget + eps < float(B_bar):
        neg_gain, t = heapq.heappop(heap)
        gain = -float(neg_gain)
        cost = float(pi[t])
        if used_budget + cost > float(B_bar) + eps:
            break
        if B[t] >= int(B_max):
            continue
        if gain <= 0:
            continue

        B[t] += 1
        used_budget += cost

        next_gain = marginal_gain(t, step=1)
        if next_gain > 0 and B[t] < int(B_max):
            heapq.heappush(heap, (-next_gain, t))

    return base.BudgetPlan(B_t=B)


def vote_majority_earliest(answers: Sequence[str]) -> str:
    """Majority vote with deterministic tie-break by earliest occurrence."""
    if not answers:
        return ""
    cnt = Counter(answers)
    max_cnt = max(cnt.values())
    tied = {a for a, c in cnt.items() if c == max_cnt}
    first_pos: Dict[str, int] = {}
    for idx, ans in enumerate(answers):
        if ans in tied and ans not in first_pos:
            first_pos[ans] = idx
    return min(tied, key=lambda a: first_pos[a])


def map_fillin_answer_to_option(
    ans_str: str,
    option_map: Dict[str, int],
    *,
    max_options: int,
) -> Tuple[str, Optional[int]]:
    """Map a fill-in answer string into a bounded option index.

    Returns:
      - ("accepted", opt_idx) if the answer is within the first `max_options`
        distinct answers for this question (option_map grows as answers arrive).
      - ("discarded", None) if it's a new unseen answer and option_map is full.

    This mirrors `aime_offline.py` logic (721-738) but scoped to one question.
    """
    ans_str = _norm_str(ans_str)
    if ans_str == "":
        # Treat empty as discard (doesn't create new option).
        return "discarded", None

    if ans_str not in option_map:
        if len(option_map) >= int(max_options):
            return "discarded", None
        option_map[ans_str] = len(option_map)

    opt_idx = option_map[ans_str]
    if opt_idx >= int(max_options):
        opt_idx = int(max_options) - 1
    return "accepted", int(opt_idx)


def filter_pool_by_max_unique(
    answers: Sequence[str],
    *,
    max_options: int,
) -> List[str]:
    """Apply the same mapping rule to an entire pool and keep only accepted answers."""
    option_map: Dict[str, int] = {}
    kept: List[str] = []
    for a in answers:
        status, _ = map_fillin_answer_to_option(a, option_map, max_options=max_options)
        if status == "accepted":
            kept.append(_norm_str(a))
    return kept


def load_aime_jsonl(path: str, *, conf_metric: str = "mean") -> List[QuestionRecord]:
    """Load AIME-style jsonl into `QuestionRecord`.

    Labels:
      - accuracy (gold): prefer `correct_answer`, fallback: `answer`, `label`
      - consistency (pseudo): prefer `final`, fallback: gold (to avoid empty consistency curves)
    """
    records: List[QuestionRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = _norm_str(obj.get("id"))
            if not qid:
                qid = f"q{len(records)}"

            raw_answers = obj.get("answers", []) or []
            answers = [_norm_str(a) for a in raw_answers if _norm_str(a) != ""]

            # accuracy (gold)
            gold = obj.get("correct_answer")
            if gold is None:
                gold = obj.get("answer")
            if gold is None:
                gold = obj.get("label")
            correct = _norm_str(gold)
            correct_label = correct if correct != "" else None

            # consistency (pseudo)
            pseudo = obj.get("final")
            if pseudo is None:
                pseudo = correct_label
            final = _norm_str(pseudo)
            final_label = final if final != "" else None

            raw_confs = obj.get("trace_confidence") or []
            if not isinstance(raw_confs, list):
                raw_confs = []

            answers: List[str] = []
            confs: List[float] = []
            for i, a in enumerate(raw_answers):
                aa = _norm_str(a)
                if aa == "":
                    continue
                answers.append(aa)
                confs.append(_extract_confidence(raw_confs[i] if i < len(raw_confs) else None, metric=str(conf_metric)))
            records.append(
                QuestionRecord(
                    qid=qid,
                    answers=answers,
                    correct=correct_label,
                    final=final_label,
                    confs=confs if confs else None,
                )
            )
    return records


def training_fit_all_questions(
    train_questions: Sequence[QuestionRecord],
    *,
    max_options: int,
    k_max_curve: int = 32,
    subsample4_draws: int = 2000,
    curve_mc_trials: int = 4000,
    rng_seed: int = 0,
) -> List[base.PerQuestionFit]:
    """AIME training: 10-choice curve estimation + 2-bucket pi_q via subsample4."""
    rng = np.random.default_rng(rng_seed)
    outputs: List[base.PerQuestionFit] = []

    for q in train_questions:
        train_label = _training_label(q)
        if train_label is None:
            raise ValueError(f"Training question {q.qid} missing training label (need `final` or `correct_answer`/fallbacks)")

        # Apply the fill-in->choice mapping to the pool so curves reflect the 10-option cap.
        capped_answers = filter_pool_by_max_unique(q.answers, max_options=max_options)
        if len(capped_answers) < 1:
            raise ValueError(f"Question {q.qid} has empty answer pool after max_options filtering")

        # Estimate A_q(k) with a 10-choice multinomial (MC inside base for n_opt=10).
        A = base.estimate_accuracy_curve_from_pool_oracle(
            capped_answers,
            train_label,
            k_max=k_max_curve,
            num_trials=curve_mc_trials,
            rng=rng,
        )
        a_q, b_q = base.fit_2param_probit_sqrtk(A, k_min=3, k_max=min(k_max_curve, len(A) - 1))

        # pi_q(t) depends only on the warm-up 4 pattern; max_options cap doesn't bind at 4.
        # Here we simplify to 2 buckets: all-4-same vs any disagreement.
        pi_q = estimate_pi_q2_via_subsample4(q.answers, num_draws=subsample4_draws, rng=rng)

        outputs.append(base.PerQuestionFit(qid=q.qid, a_q=a_q, b_q=b_q, pi_q=pi_q))

    return outputs


def train_and_build_budget_plan(
    train_questions: Sequence[QuestionRecord],
    *,
    average_budget: float,
    max_options: int,
    k_max_curve: int = 32,
    subsample4_draws: int = 2000,
    curve_mc_trials: int = 4000,
    max_per_question: int = 64,
    rng_seed: int = 0,
) -> Tuple[base.BucketStats, base.BudgetPlan]:
    fits = training_fit_all_questions(
        train_questions,
        max_options=max_options,
        k_max_curve=k_max_curve,
        subsample4_draws=subsample4_draws,
        curve_mc_trials=curve_mc_trials,
        rng_seed=rng_seed,
    )
    stats = aggregate_bucket2_stats(fits)
    plan = solve_budget_plan_greedy_marginal_anyk(
        stats,
        B_bar=float(average_budget),
        B_max=int(max_per_question),
        k0=int(base.K0),
    )
    return stats, plan


def streaming_allocate_for_question(
    answers: Sequence[str],
    *,
    budget_plan: base.BudgetPlan,
    max_options: int,
    confs: Optional[Sequence[object]] = None,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Streaming policy for a single AIME question with discard-aware budget."""
    if rng is None:
        rng = np.random.default_rng()

    pool = list(answers)
    conf_pool: List[object] = list(confs) if confs is not None else []
    if not pool:
        return {
            "bucket": None,
            "B_target": 0,
            "attempted": 0,
            "discarded": 0,
            "samples": [],
            "final_pred": "",
            "final_pred_conf": "",
            "status": "empty",
        }

    option_map: Dict[str, int] = {}
    attempted = 0
    discarded = 0
    accepted: List[str] = []
    accepted_w: List[float] = []

    idx = 0

    def _next_raw() -> Tuple[str, float]:
        nonlocal idx
        if idx >= len(pool):
            raise RuntimeError("sampler exhausted: no more answers in pool")
        a = _norm_str(pool[idx])
        # `q.confs` is expected to already contain scalar confidences; but stay defensive.
        w = _extract_mean_confidence(conf_pool[idx] if idx < len(conf_pool) else None)
        idx += 1
        return a, w

    # 1) Warm up until we get 4 accepted samples (each attempt costs budget).
    while len(accepted) < base.K0 and idx < len(pool):
        a, w = _next_raw()
        attempted += 1
        status, _opt = map_fillin_answer_to_option(a, option_map, max_options=max_options)
        if status == "accepted":
            accepted.append(a)
            accepted_w.append(w)
        else:
            discarded += 1

    if len(accepted) < base.K0:
        final_pred_conf = (
            weighted_vote_variant_majority_earliest(accepted, accepted_w, variant=str(conf_variant)) if add_conf else ""
        )
        return {
            "bucket": None,
            "B_target": 0,
            "attempted": attempted,
            "discarded": discarded,
            "samples": accepted,
            "final_pred": vote_majority_earliest(accepted),
            "final_pred_conf": final_pred_conf,
            "status": "exhausted_warmup",
        }

    t = bucket2_from_samples4(accepted[: base.K0])  # 1..2
    B_target = int(budget_plan.B_t[t - 1])

    # 2) Continue attempting samples until reaching attempted budget B_target.
    while attempted < B_target and idx < len(pool):
        a, w = _next_raw()
        attempted += 1
        status, _opt = map_fillin_answer_to_option(a, option_map, max_options=max_options)
        if status == "accepted":
            accepted.append(a)
            accepted_w.append(w)
        else:
            discarded += 1

    final_pred = vote_majority_earliest(accepted)
    final_pred_conf = (
        weighted_vote_variant_majority_earliest(accepted, accepted_w, variant=str(conf_variant)) if add_conf else ""
    )
    return {
        "bucket": t,
        "B_target": B_target,
        "attempted": attempted,
        "discarded": discarded,
        "samples": accepted,
        "final_pred": final_pred,
        "final_pred_conf": final_pred_conf,
        "status": "ok",
    }


def evaluate_streaming(
    test_questions: Sequence[QuestionRecord],
    budget_plan: base.BudgetPlan,
    *,
    max_options: int,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    rng_seed: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    rng = np.random.default_rng(rng_seed)
    results: List[Dict[str, object]] = []

    correct_acc = 0
    evaluated_acc = 0
    correct_cons = 0
    evaluated_cons = 0
    skipped = 0
    total_budget_used = 0.0

    for q in test_questions:
        if not q.answers:
            skipped += 1
            continue

        out = streaming_allocate_for_question(
            q.answers,
            budget_plan=budget_plan,
            max_options=max_options,
            confs=q.confs,
            add_conf=bool(add_conf),
            conf_variant=str(conf_variant),
            rng=rng,
        )
        total_budget_used += float(out.get("attempted", 0.0))

        pred = _norm_str(out.get("final_pred", ""))
        pred_conf = _norm_str(out.get("final_pred_conf", ""))

        is_correct = None
        is_consistent = None
        is_correct_conf = None
        is_consistent_conf = None
        if q.correct is not None:
            evaluated_acc += 1
            is_correct = bool(pred == _norm_str(q.correct))
            correct_acc += int(bool(is_correct))
            if add_conf:
                is_correct_conf = bool(pred_conf == _norm_str(q.correct))
        if q.final is not None:
            evaluated_cons += 1
            is_consistent = bool(pred == _norm_str(q.final))
            correct_cons += int(bool(is_consistent))
            if add_conf:
                gold_conf = _pseudo_label_conf_from_full_pool(
                    q,
                    max_options=int(max_options),
                    conf_variant=str(conf_variant),
                )
                is_consistent_conf = bool(pred_conf == _norm_str(gold_conf))
        if q.correct is None and q.final is None:
            skipped += 1

        out.update(
            {
                "qid": q.qid,
                "correct": q.correct,
                "final": q.final,
                "is_correct": is_correct,
                "is_consistent": is_consistent,
                "is_correct_conf": is_correct_conf,
                "is_consistent_conf": is_consistent_conf,
            }
        )
        results.append(out)

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    metrics = {
        "accuracy": float(accuracy),
        "consistency": float(consistency),
        "evaluated_acc": float(evaluated_acc),
        "evaluated_cons": float(evaluated_cons),
        "skipped": float(skipped),
        "correct_acc": float(correct_acc),
        "correct_cons": float(correct_cons),
        "total_budget_used": float(total_budget_used),
    }
    if add_conf:
        correct_acc_conf = sum(1 for r in results if r.get("is_correct_conf") is True)
        correct_cons_conf = sum(1 for r in results if r.get("is_consistent_conf") is True)
        accuracy_conf = correct_acc_conf / evaluated_acc if evaluated_acc else float("nan")
        consistency_conf = correct_cons_conf / evaluated_cons if evaluated_cons else float("nan")
        metrics.update(
            {
                "accuracy_conf": float(accuracy_conf),
                "consistency_conf": float(consistency_conf),
                "correct_acc_conf": float(correct_acc_conf),
                "correct_cons_conf": float(correct_cons_conf),
            }
        )
    return metrics, results


def evaluate_fixed_budget_majority(
    test_questions: Sequence[QuestionRecord],
    per_question_budget: int,
    *,
    max_options: int,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    rng_seed: int = 0,
) -> Dict[str, float]:
    _ = np.random.default_rng(rng_seed)  # kept for API symmetry / future use
    budget = max(int(base.K0), int(per_question_budget))

    evaluated_acc = 0
    evaluated_cons = 0
    skipped = 0
    correct_acc = 0
    correct_cons = 0
    correct_acc_conf = 0
    correct_cons_conf = 0
    total_budget_used = 0.0

    for q in test_questions:
        pool = list(q.answers)
        if not pool:
            skipped += 1
            continue

        k = min(budget, len(pool))
        option_map: Dict[str, int] = {}
        accepted: List[str] = []
        accepted_w: List[float] = []
        _discarded = 0
        conf_pool: List[object] = list(q.confs) if q.confs is not None else []
        for i in range(k):
            a = _norm_str(pool[i])
            status, _opt = map_fillin_answer_to_option(a, option_map, max_options=max_options)
            if status == "accepted":
                accepted.append(a)
                accepted_w.append(_extract_mean_confidence(conf_pool[i] if i < len(conf_pool) else None))
            else:
                _discarded += 1

        total_budget_used += float(k)
        pred = vote_majority_earliest(accepted)
        pred_conf = (
            weighted_vote_variant_majority_earliest(accepted, accepted_w, variant=str(conf_variant)) if add_conf else ""
        )
        if q.correct is not None:
            evaluated_acc += 1
            if _norm_str(pred) == _norm_str(q.correct):
                correct_acc += 1
            if add_conf and _norm_str(pred_conf) == _norm_str(q.correct):
                correct_acc_conf += 1
        if q.final is not None:
            evaluated_cons += 1
            if _norm_str(pred) == _norm_str(q.final):
                correct_cons += 1
            if add_conf:
                gold_conf = _pseudo_label_conf_from_full_pool(
                    q,
                    max_options=int(max_options),
                    conf_variant=str(conf_variant),
                )
                if _norm_str(pred_conf) == _norm_str(gold_conf):
                    correct_cons_conf += 1
        if q.correct is None and q.final is None:
            skipped += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    out = {
        "accuracy": float(accuracy),
        "consistency": float(consistency),
        "evaluated_acc": float(evaluated_acc),
        "evaluated_cons": float(evaluated_cons),
        "skipped": float(skipped),
        "correct_acc": float(correct_acc),
        "correct_cons": float(correct_cons),
        "total_budget_used": float(total_budget_used),
    }
    if add_conf:
        accuracy_conf = correct_acc_conf / evaluated_acc if evaluated_acc else float("nan")
        consistency_conf = correct_cons_conf / evaluated_cons if evaluated_cons else float("nan")
        out.update(
            {
                "accuracy_conf": float(accuracy_conf),
                "consistency_conf": float(consistency_conf),
                "correct_acc_conf": float(correct_acc_conf),
                "correct_cons_conf": float(correct_cons_conf),
            }
        )
    return out


def sweep_average_budgets(
    stats: base.BucketStats,
    test_questions: Sequence[QuestionRecord],
    *,
    max_options: int,
    add_conf: bool = True,
    conf_variant: str = "weighted",
    sweep_max: int,
    max_per_question: int,
    rng_seed: int = 0,
    oracle_model: Optional[OracleDifficultyModelKMeans] = None,
    oracle_test_params: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    start_budget = max(int(base.K0), 1)
    for avg_budget in range(start_budget, int(sweep_max) + 1):
        plan = solve_budget_plan_greedy_marginal_anyk(
            stats,
            B_bar=float(avg_budget),
            B_max=int(max_per_question),
            k0=int(base.K0),
        )
        predictor_metrics, _ = evaluate_streaming(
            test_questions,
            plan,
            max_options=max_options,
            add_conf=bool(add_conf),
            conf_variant=str(conf_variant),
            rng_seed=rng_seed,
        )
        baseline_metrics = evaluate_fixed_budget_majority(
            test_questions,
            per_question_budget=avg_budget,
            max_options=max_options,
            add_conf=bool(add_conf),
            conf_variant=str(conf_variant),
            rng_seed=rng_seed,
        )
        expected_budget = float(np.sum(stats.pi_t * plan.B_t))

        oracle_metrics: Optional[Dict[str, object]] = None
        oracle_budget_by_bucket: Optional[np.ndarray] = None
        oracle_expected = None
        if oracle_model is not None and oracle_test_params is not None:
            oracle_metrics, oracle_expected, oracle_budget_by_bucket, _, _ = evaluate_oracle_setting(
                train_questions=[],
                test_questions=test_questions,
                average_budget=float(avg_budget),
                max_per_question=int(max_per_question),
                max_options=int(max_options),
                add_conf=bool(add_conf),
                conf_variant=str(conf_variant),
                oracle_model_override=oracle_model,
                oracle_test_params_override=oracle_test_params,
                k_max_curve=0,
                curve_mc_trials=0,
                kmeans_k=int(ORACLE_KMEANS_K),
                kmeans_seed=int(rng_seed),
            )

        rows.append(
            {
                "average_budget": avg_budget,
                "predictor_total": predictor_metrics["total_budget_used"],
                "predictor_accuracy": predictor_metrics["accuracy"],
                "predictor_accuracy_conf": (predictor_metrics.get("accuracy_conf") if add_conf else None),
                "predictor_consistency": predictor_metrics.get("consistency"),
                "predictor_consistency_conf": (predictor_metrics.get("consistency_conf") if add_conf else None),
                "predictor_expected": expected_budget,
                "predictor_evaluated": predictor_metrics.get("evaluated_acc"),
                "predictor_skipped": predictor_metrics["skipped"],
                "baseline_total": baseline_metrics["total_budget_used"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_accuracy_conf": (baseline_metrics.get("accuracy_conf") if add_conf else None),
                "baseline_consistency": baseline_metrics.get("consistency"),
                "baseline_consistency_conf": (baseline_metrics.get("consistency_conf") if add_conf else None),
                "baseline_evaluated": baseline_metrics.get("evaluated_acc"),
                "baseline_skipped": baseline_metrics["skipped"],
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


def shuffle_subsample_and_relabel_question_records(
    records: Sequence[QuestionRecord],
    rng: random.Random,
    *,
    pool_size: Optional[int] = None,
    relabel_with_pool_mv: bool = False,
    max_options: int = 10,
) -> List[QuestionRecord]:
    """Shuffle per-question answer order, optionally take a prefix as the pool, and optionally relabel pseudo-gold via MV.

    - Gold label (`correct`) is preserved.
    - Only pseudo label (`final`) is optionally relabeled.
    """
    k = None if pool_size is None else max(0, int(pool_size))
    out: List[QuestionRecord] = []
    for record in records:
        answers_obj = record.answers
        if isinstance(answers_obj, Sequence) and not isinstance(answers_obj, (str, bytes)):
            answers_list = list(answers_obj)
            indices = list(range(len(answers_list)))
            rng.shuffle(indices)
            shuffled_answers: List[str] = [_norm_str(answers_list[idx]) for idx in indices]
            conf_list: List[object] = list(record.confs) if record.confs is not None else []
            shuffled_confs: Optional[List[object]] = None
            if conf_list:
                shuffled_confs = [conf_list[idx] if idx < len(conf_list) else None for idx in indices]
            if k is not None and k > 0:
                shuffled_answers = shuffled_answers[: min(k, len(shuffled_answers))]
                if shuffled_confs is not None:
                    shuffled_confs = shuffled_confs[: min(k, len(shuffled_confs))]
        else:
            shuffled_answers = list(record.answers)
            shuffled_confs = list(record.confs) if record.confs is not None else None

        new_final = record.final
        if relabel_with_pool_mv:
            capped = filter_pool_by_max_unique(shuffled_answers, max_options=int(max_options))
            if capped:
                new_final = vote_majority_earliest(capped)

        out.append(
            QuestionRecord(
                qid=record.qid,
                answers=shuffled_answers,
                correct=record.correct,
                final=new_final,
                confs=shuffled_confs,
            )
        )
    return out



def _sweep_rows_to_curve_dict_total(
    rows: Sequence[Dict[str, object]],
    *,
    metric: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """Convert sweep rows to curves keyed by label, using realized TOTAL budget on x-axis."""
    out: Dict[str, Dict[int, float]] = {}
    label_to_keys = {
        "Pred": ("predictor_total", f"predictor_{metric}"),
        "Pred_Conf": ("predictor_total", f"predictor_{metric}_conf"),
        "Base": ("baseline_total", f"baseline_{metric}"),
        "Base_Conf": ("baseline_total", f"baseline_{metric}_conf"),
        "Oracle": ("oracle_total", f"oracle_{metric}"),
        "Oracle_Conf": ("oracle_total", f"oracle_{metric}_conf"),
    }
    for label, (x_key, y_key) in label_to_keys.items():
        x_to_y: Dict[int, float] = {}
        for row in rows or []:
            x_raw = row.get(x_key)
            y_raw = row.get(y_key)
            if x_raw is None or y_raw is None:
                continue
            try:
                xx = float(x_raw)
                yy = float(y_raw)
            except Exception:
                continue
            if not (math.isfinite(xx) and math.isfinite(yy)):
                continue
            x_to_y[int(round(xx))] = yy
        if x_to_y:
            out[label] = x_to_y
    return {lab: sorted([(x, y) for x, y in mp.items()], key=lambda t: t[0]) for lab, mp in out.items()}


def _sweep_runs_to_curve_runs_total(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    *,
    metric: str,
) -> List[Tuple[int, Dict[str, List[Tuple[int, float]]]]]:
    curve_runs: List[Tuple[int, Dict[str, List[Tuple[int, float]]]]] = []
    for run_idx, rows in sweep_runs:
        curve_runs.append((int(run_idx), _sweep_rows_to_curve_dict_total(rows, metric=metric)))
    return curve_runs


def _first_budget_at_max(points: Sequence[Tuple[int, float]]) -> Optional[int]:
    """Return the smallest budget that achieves the maximum value in points."""
    if not points:
        return None
    best_val: Optional[float] = None
    best_budget: Optional[int] = None
    for b, v in points:
        try:
            bb = int(b)
            vv = float(v)
        except Exception:
            continue
        if not math.isfinite(vv):
            continue
        if best_val is None or vv > best_val:
            best_val = vv
            best_budget = bb
        elif best_val is not None and vv == best_val:
            if best_budget is None or bb < best_budget:
                best_budget = bb
    return best_budget


def _points_to_budget_map(points: Sequence[Tuple[int, float]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for b, v in points or []:
        try:
            out[int(b)] = float(v)
        except Exception:
            continue
    return out


def _scalar_mean_std(values: Sequence[Optional[Union[int, float]]]) -> Tuple[float, float, int]:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std, int(arr.size)


def export_multi_run_curves_jsonl(
    curve_runs_consistency: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    curve_runs_accuracy: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    output_path: str,
    *,
    methods: Optional[Sequence[str]] = None,
    total_runs: Optional[Sequence[Tuple[int, Dict[str, Dict[int, float]]]]] = None,
    sweep_runs: Optional[Sequence[Tuple[int, Sequence[Dict[str, object]]]]] = None,
) -> None:
    """Export multi-run curves + derived stats into a JSONL file (gpqa_offline style)."""
    wanted = list(methods or ["Base", "Pred", "Oracle"])

    cons_by_run: Dict[int, Dict[str, List[Tuple[int, float]]]] = {i: d for i, d in (curve_runs_consistency or [])}
    acc_by_run: Dict[int, Dict[str, List[Tuple[int, float]]]] = {i: d for i, d in (curve_runs_accuracy or [])}
    totals_by_run: Dict[int, Dict[str, Dict[int, float]]] = {i: d for i, d in (total_runs or [])}
    run_indices = sorted(set(cons_by_run.keys()) | set(acc_by_run.keys()))

    # If caller didn't request methods explicitly, auto-include *_Conf when available.
    if methods is None:
        maybe_conf = ["Base_Conf", "Pred_Conf", "Oracle_Conf"]
        has_conf = False
        # 1) prefer curve presence
        for run_idx in run_indices:
            cc = cons_by_run.get(run_idx, {}) or {}
            ac = acc_by_run.get(run_idx, {}) or {}
            if any(m in cc or m in ac for m in maybe_conf):
                has_conf = True
                break
        # 2) fallback: detect conf metrics in sweep rows
        if not has_conf:
            for _run_idx, rows in (sweep_runs or []):
                for r in rows or []:
                    if (
                        r.get("predictor_accuracy_conf") is not None
                        or r.get("predictor_consistency_conf") is not None
                        or r.get("baseline_accuracy_conf") is not None
                        or r.get("baseline_consistency_conf") is not None
                        or r.get("oracle_accuracy_conf") is not None
                        or r.get("oracle_consistency_conf") is not None
                    ):
                        has_conf = True
                        break
                if has_conf:
                    break
        if has_conf:
            for m in maybe_conf:
                if m not in wanted:
                    wanted.append(m)

    def _find_key(curves: Dict[str, List[Tuple[int, float]]], canonical: str) -> Optional[str]:
        if not curves:
            return None
        if canonical in curves:
            return canonical
        aliases: List[str] = [canonical]
        low = canonical.lower()
        if low == "base":
            aliases.extend(["mv", "base"])
        elif low == "pred":
            aliases.extend(["okg", "predictor", "pred"])
        elif low == "oracle":
            aliases.extend(["oracle"])
        elif low in {"base_conf", "pred_conf", "oracle_conf"}:
            aliases.extend([canonical, low, low.replace("_", ""), low.replace("_", "-")])
        aliases_low = {a.lower() for a in aliases if a}
        for k in curves.keys():
            if str(k).lower() in aliases_low:
                return k
        return None

    avg_budgets_at_cons1: Dict[str, List[Optional[int]]] = {m: [] for m in wanted}
    total_budgets_at_cons1: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}
    at_pred_con1_acc: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}
    at_pred_con1_cons: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}

    sweep_by_run: Dict[int, Sequence[Dict[str, object]]] = {i: rows for i, rows in (sweep_runs or [])}

    def _row_map_by_avg(rows: Sequence[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
        mp: Dict[int, Dict[str, object]] = {}
        for r in rows or []:
            b = r.get("average_budget")
            if b is None:
                continue
            try:
                bb = int(b)
            except Exception:
                continue
            mp[bb] = dict(r)
        return mp

    def _first_avg_budget_at_max(rows: Sequence[Dict[str, object]], metric_key: str) -> Optional[int]:
        best_val: Optional[float] = None
        best_budget: Optional[int] = None
        for r in rows or []:
            b = r.get("average_budget")
            v = r.get(metric_key)
            if b is None or v is None:
                continue
            try:
                bb = int(b)
                vv = float(v)
            except Exception:
                continue
            if not math.isfinite(vv):
                continue
            if best_val is None or vv > best_val:
                best_val = vv
                best_budget = bb
            elif best_val is not None and vv == best_val:
                if best_budget is None or bb < best_budget:
                    best_budget = bb
        return best_budget

    out_path_obj = Path(output_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_path_obj.open("w", encoding="utf-8") as f:
        for run_idx in run_indices:
            cons_curves = cons_by_run.get(run_idx, {}) or {}
            acc_curves = acc_by_run.get(run_idx, {}) or {}
            totals_maps = totals_by_run.get(run_idx, {}) or {}
            sweep_rows = sweep_by_run.get(run_idx, None)
            rows_by_avg = _row_map_by_avg(sweep_rows) if sweep_rows is not None else {}

            peak_avg_for: Dict[str, Optional[int]] = {}
            if sweep_rows is not None:
                key_by_method = {
                    "Base": "baseline_consistency",
                    "Base_Conf": "baseline_consistency_conf",
                    "Pred": "predictor_consistency",
                    "Pred_Conf": "predictor_consistency_conf",
                    "Oracle": "oracle_consistency",
                    "Oracle_Conf": "oracle_consistency_conf",
                }
                for m in wanted:
                    mk = key_by_method.get(m)
                    peak_avg_for[m] = _first_avg_budget_at_max(sweep_rows, mk) if mk else None
            else:
                for m in wanted:
                    k = _find_key(cons_curves, m)
                    peak_avg_for[m] = _first_budget_at_max(cons_curves.get(k, []) if k else [])

            for m in wanted:
                bpk = peak_avg_for.get(m)
                avg_budgets_at_cons1[m].append(bpk)
                if sweep_rows is not None and bpk is not None and bpk in rows_by_avg:
                    row = rows_by_avg[bpk]
                    mm = str(m).lower()
                    if mm.startswith("base"):
                        total_key = "baseline_total"
                    elif mm.startswith("pred") or mm.startswith("okg"):
                        total_key = "predictor_total"
                    else:
                        total_key = "oracle_total"
                    tb = row.get(total_key)
                    try:
                        total_budgets_at_cons1[m].append(float(tb) if tb is not None else None)
                    except Exception:
                        total_budgets_at_cons1[m].append(None)
                else:
                    total_map = totals_maps.get(m, {}) or {}
                    tb = total_map.get(int(bpk)) if (bpk is not None and total_map) else (float(bpk) if bpk is not None else None)
                    total_budgets_at_cons1[m].append(tb)

            pred_peak_avg = peak_avg_for.get("Pred")
            if sweep_rows is not None and pred_peak_avg is not None and pred_peak_avg in rows_by_avg:
                row = rows_by_avg[pred_peak_avg]
                for m in wanted:
                    mm = str(m).lower()
                    is_conf = "conf" in mm
                    if mm.startswith("base"):
                        acc_key = "baseline_accuracy_conf" if is_conf else "baseline_accuracy"
                        cons_key = "baseline_consistency_conf" if is_conf else "baseline_consistency"
                    elif mm.startswith("pred") or mm.startswith("okg"):
                        acc_key = "predictor_accuracy_conf" if is_conf else "predictor_accuracy"
                        cons_key = "predictor_consistency_conf" if is_conf else "predictor_consistency"
                    else:
                        acc_key = "oracle_accuracy_conf" if is_conf else "oracle_accuracy"
                        cons_key = "oracle_consistency_conf" if is_conf else "oracle_consistency"
                    a = row.get(acc_key)
                    c = row.get(cons_key)
                    try:
                        at_pred_con1_acc[m].append(float(a) if a is not None else None)
                    except Exception:
                        at_pred_con1_acc[m].append(None)
                    try:
                        at_pred_con1_cons[m].append(float(c) if c is not None else None)
                    except Exception:
                        at_pred_con1_cons[m].append(None)
            else:
                pred_budget = peak_avg_for.get("Pred") or peak_avg_for.get("OKG")
                for m in wanted:
                    cons_key = _find_key(cons_curves, m)
                    acc_key = _find_key(acc_curves, m)
                    cons_map = _points_to_budget_map(cons_curves.get(cons_key, []) if cons_key else [])
                    acc_map = _points_to_budget_map(acc_curves.get(acc_key, []) if acc_key else [])
                    at_pred_con1_cons[m].append(cons_map.get(int(pred_budget)) if pred_budget is not None else None)
                    at_pred_con1_acc[m].append(acc_map.get(int(pred_budget)) if pred_budget is not None else None)

            record: Dict[str, object] = {
                "type": "run",
                "run_index": int(run_idx),
                "curves": {
                    m: {
                        "consistency": cons_curves.get(_find_key(cons_curves, m) or "", []),
                        "accuracy": acc_curves.get(_find_key(acc_curves, m) or "", []),
                    }
                    for m in wanted
                },
                "Pred_avg_budget_at_cons1": peak_avg_for.get("Pred"),
                "Base_avg_budget_at_cons1": peak_avg_for.get("Base"),
                "Oracle_avg_budget_at_cons1": peak_avg_for.get("Oracle"),
                "Pred_total_budget_at_cons1": total_budgets_at_cons1["Pred"][-1] if total_budgets_at_cons1.get("Pred") else None,
                "Base_total_budget_at_cons1": total_budgets_at_cons1["Base"][-1] if total_budgets_at_cons1.get("Base") else None,
                "Oracle_total_budget_at_cons1": total_budgets_at_cons1["Oracle"][-1] if total_budgets_at_cons1.get("Oracle") else None,
                "Base_consistency_at_con1": at_pred_con1_cons.get("Base", [None])[-1],
                "Base_accuracy_at_con1": at_pred_con1_acc.get("Base", [None])[-1],
                "Pred_consistency_at_con1": at_pred_con1_cons.get("Pred", [None])[-1],
                "Pred_accuracy_at_con1": at_pred_con1_acc.get("Pred", [None])[-1],
                "Oracle_consistency_at_con1": at_pred_con1_cons.get("Oracle", [None])[-1],
                "Oracle_accuracy_at_con1": at_pred_con1_acc.get("Oracle", [None])[-1],
            }

            for m in ("Base_Conf", "Pred_Conf", "Oracle_Conf"):
                if m in wanted:
                    record[f"{m}_avg_budget_at_cons1"] = peak_avg_for.get(m)
                    record[f"{m}_total_budget_at_cons1"] = total_budgets_at_cons1.get(m, [None])[-1]
                    record[f"{m}_consistency_at_con1"] = at_pred_con1_cons.get(m, [None])[-1]
                    record[f"{m}_accuracy_at_con1"] = at_pred_con1_acc.get(m, [None])[-1]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary: Dict[str, object] = {"type": "summary", "num_runs": int(len(run_indices))}
        total_budget_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        for m in wanted:
            mean, std, n = _scalar_mean_std(total_budgets_at_cons1[m])
            total_budget_stats[m] = {"mean": mean, "std": std, "num_runs": n}
        summary["budget_at_cons1_stats"] = total_budget_stats

        metric_stats: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {}
        for m in wanted:
            a_mean, a_std, a_n = _scalar_mean_std(at_pred_con1_acc[m])
            c_mean, c_std, c_n = _scalar_mean_std(at_pred_con1_cons[m])
            metric_stats[m] = {
                "accuracy_at_con1": {"mean": a_mean, "std": a_std, "num_runs": a_n},
                "consistency_at_con1": {"mean": c_mean, "std": c_std, "num_runs": c_n},
            }
        summary["metrics_at_pred_cons1_budget_stats"] = metric_stats

        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"[multi-run] curves+stats jsonl saved: {out_path_obj}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming AIME (fill-in) with 10-option mapping + bucketed allocation.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(__file__).parent / "aime25_conf_64.jsonl"),
        help="Path to AIME JSONL file (must contain answers + correct_answer/label/answer/final).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10,
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
        "--add_conf",
        "--add-conf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute/plot confidence-weighted (_conf) methods in sweep outputs.",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=10,
        help="Max distinct answers (options) to keep per question; unseen answers beyond are discarded.",
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
        help="Maximum budget allowed for a single question (attempted samples).",
    )
    parser.add_argument(
        "--k-max-curve",
        type=int,
        default=32,
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
        default=4000,
        help="Monte Carlo trials for 10-choice accuracy curve estimation.",
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
        "--plot-path",
        type=str,
        default="aime_streaming_sweep.png",
        help="Where to save the sweep line chart (PNG).",
    )
    parser.add_argument(
        "--accuracy-plot",
        type=str,
        default="aime_model_online_accuracy.png",
        help="(Reference-style) accuracy curve plot path (PNG).",
    )
    parser.add_argument(
        "--consistency-plot",
        type=str,
        default="aime_model_online_consistency.png",
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
        default="aime_streaming_multi_run.jsonl",
        help="(Like gpqa_offline) Export multi-run curves+stats to JSONL (optional; requires --sweep).",
    )
    parser.add_argument(
        "--multi-runs",
        type=int,
        default=10,
        help="Number of repeated sweeps with shuffled answer pools (requires --sweep).",
    )
    parser.add_argument(
        "--multi-pool-size",
        type=int,
        default=64,
        help="multi-run: shuffle answers then take first K as this run's pool (default 64).",
    )
    parser.add_argument(
        "--multi-relabel-mv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="multi-run: relabel pseudo label (`final`) as MV(subsampled pool) for this run.",
    )
    parser.add_argument(
        "--multi-run-plot",
        type=str,
        default="aime_streaming_sweep_multi.png",
        help="Where to save the multi-run swepep plot (PNG).",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    all_questions = load_aime_jsonl(str(data_path), conf_metric=str(args.conf_metric))
    if len(all_questions) <= int(args.train_size):
        raise ValueError(f"Dataset needs more than {args.train_size} questions to create a test split.")

    train_questions = all_questions[: int(args.train_size)]
    test_questions = all_questions[int(args.train_size) :]

    stats, plan = train_and_build_budget_plan(
        train_questions,
        average_budget=float(args.average_budget),
        max_options=int(args.max_options),
        k_max_curve=int(args.k_max_curve),
        subsample4_draws=int(args.subsample4_draws),
        curve_mc_trials=int(args.curve_mc_trials),
        max_per_question=int(args.max_per_question),
        rng_seed=int(args.rng_seed),
    )
    expected_budget = float(np.sum(stats.pi_t * plan.B_t))

    print("Training complete.")
    print("pi_t:", stats.pi_t)
    print("B_t:", plan.B_t)
    print(f"Expected average budget (predictor): {expected_budget:.2f}")

    predictor_metrics, _ = evaluate_streaming(
        test_questions,
        plan,
        max_options=int(args.max_options),
        add_conf=bool(args.add_conf),
        conf_variant=str(args.conf_variant),
        rng_seed=int(args.rng_seed),
    )
    baseline_metrics = evaluate_fixed_budget_majority(
        test_questions,
        per_question_budget=int(round(float(args.average_budget))),
        max_options=int(args.max_options),
        add_conf=bool(args.add_conf),
        conf_variant=str(args.conf_variant),
        rng_seed=int(args.rng_seed),
    )

    # Oracle difficulty (kmeans k=2) setup: compute (a,b) fits once and reuse in sweeps.
    # Optionally fit oracle buckets on full data for a stronger (leaky) upper bound.
    oracle_fit_questions = all_questions if bool(getattr(args, "oracle_fit_all", False)) else train_questions
    train_params_oracle = compute_question_param_map(
        oracle_fit_questions,
        max_options=int(args.max_options),
        k_max_curve=int(args.k_max_curve),
        curve_mc_trials=int(args.curve_mc_trials),
    )
    test_params_oracle = compute_question_param_map(
        test_questions,
        max_options=int(args.max_options),
        k_max_curve=int(args.k_max_curve),
        curve_mc_trials=int(args.curve_mc_trials),
    )
    oracle_model = build_oracle_difficulty_model(
        train_params_oracle,
        k=int(ORACLE_KMEANS_K),
        random_seed=int(args.rng_seed),
    )
    (
        oracle_metrics,
        oracle_expected,
        oracle_budget_by_bucket,
        _oracle_model_ret,
        oracle_test_params,
    ) = evaluate_oracle_setting(
        train_questions=train_questions,
        test_questions=test_questions,
        average_budget=float(args.average_budget),
        max_per_question=int(args.max_per_question),
        max_options=int(args.max_options),
        add_conf=bool(args.add_conf),
        conf_variant=str(args.conf_variant),
        k_max_curve=int(args.k_max_curve),
        curve_mc_trials=int(args.curve_mc_trials),
        oracle_model_override=oracle_model,
        oracle_test_params_override=test_params_oracle,
        kmeans_k=int(ORACLE_KMEANS_K),
        kmeans_seed=int(args.rng_seed),
    )

    print(
        f"\nPredictor accuracy: {predictor_metrics['accuracy']:.4f} "
        f"(consistency={predictor_metrics.get('consistency', float('nan')):.4f}) "
        f"on {int(predictor_metrics.get('evaluated_acc', 0))} questions"
    )
    print(f"Predictor total budget (attempted): {predictor_metrics['total_budget_used']:.1f}")
    print(
        f"Baseline accuracy: {baseline_metrics['accuracy']:.4f} "
        f"(consistency={baseline_metrics.get('consistency', float('nan')):.4f}) "
        f"on {int(baseline_metrics.get('evaluated_acc', 0))} questions"
    )
    print(f"Baseline total budget (attempted): {baseline_metrics['total_budget_used']:.1f}")
    if oracle_metrics:
        print(
            f"Oracle accuracy: {float(oracle_metrics['accuracy']):.4f} "
            f"(consistency={float(oracle_metrics.get('consistency', float('nan'))):.4f}) "
            f"(skipped {int(float(oracle_metrics.get('skipped', 0)))} )"
        )
        print(f"Oracle total budget (attempted): {float(oracle_metrics['total_budget_used']):.1f}")
        if oracle_model is not None and oracle_budget_by_bucket is not None and oracle_expected is not None:
            print("Oracle kmeans centers (a,b) (easy->hard):")
            for i, (a, b) in enumerate(np.asarray(oracle_model.centers_ab, dtype=float).tolist()):
                print(f"  bucket {i}: a={a:.4f}, b={b:.4f}, prob={float(oracle_model.probs[i]):.3f}, B={int(oracle_budget_by_bucket[i])}")
            print("Oracle expected budget (per question):", f"{float(oracle_expected):.2f}")

    if args.sweep:
        sweep_rows = sweep_average_budgets(
            stats,
            test_questions,
            max_options=int(args.max_options),
            add_conf=bool(args.add_conf),
            conf_variant=str(args.conf_variant),
            sweep_max=int(args.sweep_max),
            max_per_question=int(args.max_per_question),
            rng_seed=int(args.rng_seed),
            oracle_model=oracle_model,
            oracle_test_params=oracle_test_params if oracle_test_params else None,
        )
        if sweep_rows:
            title_suffix = (
                f"train={len(train_questions)}, test={len(test_questions)}, "
                f"max_options={int(args.max_options)}, avg≤{int(args.sweep_max)}"
            )
            base.plot_sweep_results(sweep_rows, Path(args.plot_path), title_suffix=title_suffix)
            print("\nSweep results (avg_budget: pred_acc, base_acc, expected, B_t):")
            for row in sweep_rows:
                print(
                    f"  {int(row['average_budget']):2d}: "
                    f"pred={float(row['predictor_accuracy']):.4f}, "
                    f"base={float(row['baseline_accuracy']):.4f}, "
                    f"expected={float(row['predictor_expected']):.2f}, "
                    f"B_t={row['budget_plan']}"
                )

            # Multi-run: shuffle/subsample per run, and export reference-style plots+CSVs+JSONL.
            if int(args.multi_runs) > 1:
                sweep_runs: List[Tuple[int, Sequence[Dict[str, object]]]] = []
                for run_idx in range(int(args.multi_runs)):
                    rng = random.Random(int(args.rng_seed) + run_idx)
                    train_q_run = shuffle_subsample_and_relabel_question_records(
                        train_questions,
                        rng,
                        pool_size=int(getattr(args, "multi_pool_size", 64)),
                        relabel_with_pool_mv=bool(getattr(args, "multi_relabel_mv", True)),
                        max_options=int(args.max_options),
                    )
                    test_q_run = shuffle_subsample_and_relabel_question_records(
                        test_questions,
                        rng,
                        pool_size=int(getattr(args, "multi_pool_size", 64)),
                        relabel_with_pool_mv=bool(getattr(args, "multi_relabel_mv", True)),
                        max_options=int(args.max_options),
                    )

                    oracle_fit_q_run = (
                        (list(train_q_run) + list(test_q_run))
                        if bool(getattr(args, "oracle_fit_all", False))
                        else train_q_run
                    )

                    stats_run, _plan_run = train_and_build_budget_plan(
                        train_q_run,
                        average_budget=float(args.average_budget),
                        max_options=int(args.max_options),
                        k_max_curve=int(args.k_max_curve),
                        subsample4_draws=int(args.subsample4_draws),
                        curve_mc_trials=int(args.curve_mc_trials),
                        max_per_question=int(args.max_per_question),
                        rng_seed=int(args.rng_seed) + run_idx,
                    )

                    train_params_run = compute_question_param_map(
                        oracle_fit_q_run,
                        max_options=int(args.max_options),
                        k_max_curve=int(args.k_max_curve),
                        curve_mc_trials=int(args.curve_mc_trials),
                    )
                    test_params_run = compute_question_param_map(
                        test_q_run,
                        max_options=int(args.max_options),
                        k_max_curve=int(args.k_max_curve),
                        curve_mc_trials=int(args.curve_mc_trials),
                    )
                    oracle_model_run = build_oracle_difficulty_model(
                        train_params_run,
                        k=int(ORACLE_KMEANS_K),
                        random_seed=int(args.rng_seed) + run_idx,
                    )

                    sweep_rows_run = sweep_average_budgets(
                        stats_run,
                        test_q_run,
                        max_options=int(args.max_options),
                        add_conf=bool(args.add_conf),
                        conf_variant=str(args.conf_variant),
                        sweep_max=int(args.sweep_max),
                        max_per_question=int(args.max_per_question),
                        rng_seed=int(args.rng_seed) + run_idx,
                        oracle_model=oracle_model_run,
                        oracle_test_params=test_params_run if test_params_run else None,
                    )
                    sweep_runs.append((run_idx, sweep_rows_run))

                sweep_runs_for_export = sweep_runs
            else:
                sweep_runs_for_export = [(0, sweep_rows)]

            plot_accuracy_multi_run_curves(
                sweep_runs_for_export,
                args.accuracy_plot,
                csv_path=(args.accuracy_csv if args.accuracy_csv else None),
            )
            plot_consistency_multi_run_curves(
                sweep_runs_for_export,
                args.consistency_plot,
                csv_path=(args.consistency_csv if args.consistency_csv else None),
            )
            if args.multi_run_jsonl:
                curve_runs_acc = _sweep_runs_to_curve_runs_total(sweep_runs_for_export, metric="accuracy")
                curve_runs_cons = _sweep_runs_to_curve_runs_total(sweep_runs_for_export, metric="consistency")
                export_multi_run_curves_jsonl(
                    curve_runs_cons,
                    curve_runs_acc,
                    str(args.multi_run_jsonl),
                    sweep_runs=sweep_runs_for_export,
                )


if __name__ == "__main__":
    main()
