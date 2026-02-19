#!/usr/bin/env python3
"""
Oracle difficulty via KMeans (k=5) for (a,b) fits (GPQA).

This file contains an inlined streaming core (historically from `gpqa_streaming.py`):
- Predictor training / streaming policy stays the same (now explicit in this file).
- Oracle setting changes from (a,b) quantile grid to kmeans buckets.

Workflow (oracle):
1) Fit per-question (a_q,b_q) on training questions (full-answer oracle fits)
2) Fit KMeans on (a,b) -> k cluster centers
3) Allocate budgets across the k centers (greedy marginal gains under average budget)
4) For each test question, compute its (a,b), assign to nearest center, and use that bucket's budget
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from oracle_kmeans_common import (
    OracleDifficultyModelKMeans,
    build_oracle_difficulty_model_from_params,
    greedy_budget_allocation_oracle_common,
    locate_param_bin_oracle as locate_param_bin_oracle_common,
)


# -----------------------------
# Inlined GPQA streaming core
# -----------------------------
K0 = 4  # warm-up samples
NUM_BUCKETS = 5
ORACLE_QUANTILES = (0.33, 0.69)

# Five possible sorted count patterns for 4 samples (descending)
PATTERNS: List[Tuple[int, int, int, int]] = [
    (4, 0, 0, 0),
    (3, 1, 0, 0),
    (2, 2, 0, 0),
    (2, 1, 1, 0),
    (1, 1, 1, 1),
]


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class QuestionRecord:
    """A question with an answer pool plus (optional) labels.

    answers: each element is an option label (e.g., "A","B","C","D") or int {0,1,2,3}
    correct: ground-truth answer label/index (for accuracy). Often `correct_letter`.
    final: model's pooled decision label/index (for self-consistency). Often `final`.
    """
    qid: str
    answers: Sequence  # length e.g. 64 in training
    correct: Optional[object] = None
    final: Optional[object] = None
    # Optional per-answer confidence weights aligned with `answers`.
    # When present, each element should be a float-like confidence (e.g. mean_confidence).
    confs: Optional[Sequence[object]] = None


def _extract_mean_confidence(entry: object) -> float:
    """Backward-compatible mean_confidence extractor (kept for older callers)."""
    return _extract_confidence(entry, metric="mean")


# Mirror gpqa_offline.py's metric naming, but keep it minimal and backward-compatible.
CONF_METRIC_KEYS: Dict[str, str] = {
    "Conf": "mean_confidence",
    "mean": "mean_confidence",
    "tail": "tail_2048_mean_conf",
    "bottom": "bottom_0.1_sliding_2048_mean_conf",
}


def _extract_confidence(entry: object, *, metric: str = "mean") -> float:
    """Best-effort extract a scalar confidence value from a trace_confidence entry.

    - If `metric` is one of CONF_METRIC_KEYS (e.g. "mean", "tail"), use the mapped key.
    - Otherwise, treat `metric` as the raw dict key to read.
    - Supports nested schema: {"conf_summary": {...}} by flattening it.
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
        # Allow a few legacy aliases + safe fallback.
        v = src.get("mean_confidence")
        if v is None:
            v = src.get("Conf")
    try:
        vf = float(v) if v is not None else 1.0
        return vf if math.isfinite(vf) else 1.0
    except Exception:
        return 1.0


def _training_label(q: QuestionRecord) -> Optional[object]:
    """Label used for training fits (A_q(k)) / bucket calibration.

    By default we follow the previous behavior of this script, which trained on `final`
    (a pool-based pseudo-gold). If `final` is missing, fall back to `correct`.
    """
    return q.final if q.final is not None else q.correct


@dataclass
class PerQuestionFit:
    """Per-question outputs in training."""
    qid: str
    a_q: float
    b_q: float
    pi_q: np.ndarray  # shape (5,) bucket probabilities


@dataclass
class BucketStats:
    """Global bucket statistics after training aggregation."""
    pi_t: np.ndarray            # shape (5,)
    a_t: np.ndarray             # shape (5,)
    b_t: np.ndarray             # shape (5,)


@dataclass
class BudgetPlan:
    """Budget plan B_t for each bucket."""
    B_t: np.ndarray  # shape (5,), integer budgets >= K0


@dataclass
class OracleDifficultyModel:
    """Difficulty buckets derived from full-answer fits (oracle setting)."""

    thresholds_a: np.ndarray
    thresholds_b: np.ndarray
    probs_grid: np.ndarray
    mean_a_grid: np.ndarray
    mean_b_grid: np.ndarray


# -----------------------------
# Helper: bucket mapping
# -----------------------------
def count_pattern_4(samples4: Sequence, num_options: int = 4) -> Tuple[int, int, int, int]:
    """Compute sorted count pattern from 4 samples.

    Input:
      samples4: length-4 list of answers (labels or indices)
      num_options: 4 for multiple-choice
    Output:
      one of PATTERNS, as a tuple sorted descending, e.g. (3,1,0,0)
    """
    if len(samples4) != 4:
        raise ValueError("samples4 must have length 4")

    # Count frequencies
    # NOTE: If answers are strings, we just count distinct; if you want strict 4-option set,
    # map to {0,1,2,3} beforehand.
    from collections import Counter
    ctr = Counter(samples4)
    counts = sorted(ctr.values(), reverse=True)
    # pad with zeros to length 4
    counts = (counts + [0, 0, 0, 0])[:4]
    return tuple(counts)  # type: ignore


def bucket_from_pattern(pattern: Tuple[int, int, int, int]) -> int:
    """Deterministic mapping g(C^(4)) -> bucket index in {1..5}.

    Here we map patterns in the order listed in PATTERNS:
      (4,0,0,0)->1, (3,1,0,0)->2, ..., (1,1,1,1)->5
    """
    try:
        return PATTERNS.index(pattern) + 1
    except ValueError:
        raise ValueError(f"Unknown pattern {pattern}; expected one of {PATTERNS}")


def bucket_from_samples4(samples4: Sequence) -> int:
    """Convenience: samples4 -> pattern -> bucket."""
    pat = count_pattern_4(samples4)
    return bucket_from_pattern(pat)


# -----------------------------
# Accuracy curve + probit fit
# -----------------------------
def exact_prob_pick_argmax_multinom4(theta4, k, log_fact=None):
    """Probability majority vote picks argmax(theta) with tie uniform."""
    theta = np.asarray(theta4, dtype=float)
    theta = theta / theta.sum()
    if np.any(theta <= 0):
        eps = 1e-12
        theta = np.clip(theta, eps, None)
        theta = theta / theta.sum()
    if theta.size != 4:
        raise ValueError("theta must have length 4.")

    # move argmax to front
    argmax_idx = int(np.argmax(theta))
    if argmax_idx != 0:
        theta = np.concatenate([[theta[argmax_idx]], theta[:argmax_idx], theta[argmax_idx + 1 :]])

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


def estimate_accuracy_curve_from_pool(
    answers: Sequence,
    correct: object,
    k_max: int,
    *,
    num_trials: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Estimate A_q(k) for k=1..k_max using exact multinomial majority probability.

    The empirical label distribution is converted to a Multinomial(theta, k)
    assumption (with replacement). Accuracy is the probability the correct
    label wins majority vote with uniform tie-breaking, computed exactly
    (no Monte Carlo).
    """

    # Build empirical theta over four options, placing the correct label first
    ctr = Counter(answers)
    if not ctr:
        raise ValueError("empty answer pool")
    if correct not in ctr:
        ctr[correct] = 0

    others = [opt for opt in ctr.keys() if opt != correct]
    # If more than 3 other labels exist, keep the three most common to fit length-4 theta
    others = sorted(others, key=lambda x: ctr[x], reverse=True)[:3]
    while len(others) < 3:
        # pad with dummy labels of zero count
        placeholder = f"__dummy{len(others)}"
        if placeholder not in ctr:
            ctr[placeholder] = 0
        others.append(placeholder)

    counts = [ctr[correct]] + [ctr[o] for o in others]
    theta = np.asarray(counts, dtype=float)

    A = np.zeros(k_max + 1, dtype=float)
    A[0] = 0.5  # convention; not used for fit typically
    log_fact_cache = [math.lgamma(i + 1) for i in range(k_max + 1)]
    for k in range(1, k_max + 1):
        A[k] = exact_prob_pick_argmax_multinom4(theta, k, log_fact_cache)
    return A


def fit_2param_probit_sqrtk(
    A: np.ndarray,
    k_min: int = 3,
    k_max: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit A(k)=Phi(a*sqrt(k)+b) using least squares in probit space.

    Input:
      A: array indexed by k, shape (K+1,)
      k_min/k_max: fitting range
    Output:
      (a, b)
    """
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


def delta_probit(k: int, a: float, b: float) -> float:
    """Marginal gain Δ(k)=A(k+1)-A(k), computed stably.

    Naively computing `norm.cdf(x2) - norm.cdf(x1)` can underflow to 0 when `cdf`
    saturates to 1.0 at float precision. For k>=1, use tail probabilities:
      cdf(x2)-cdf(x1) = sf(x1) - sf(x2)
    and compute the difference in log-space.

    For k<=0, note this codebase defines A(0)=0.5 (convention), so:
      Δ(0) = A(1) - 0.5
    """
    k = int(k)
    a = float(a)
    b = float(b)

    # k<=0: A(0) is defined as 0.5 here (not Phi(b)).
    if k <= 0:
        return float(norm.cdf(a * math.sqrt(1.0) + b) - 0.5)

    try:
        x1 = float(a * math.sqrt(float(k)) + b)
        x2 = float(a * math.sqrt(float(k + 1)) + b)

        logsf1 = float(norm.logsf(x1))
        logsf2 = float(norm.logsf(x2))
        if not (math.isfinite(logsf1) and math.isfinite(logsf2)):
            raise ValueError("non-finite logsf")

        # Ensure logsf1 >= logsf2 (expected when x2 > x1); swap defensively.
        if logsf1 < logsf2:
            logsf1, logsf2 = logsf2, logsf1

        sf1 = math.exp(logsf1)
        # sf1 - sf2 = sf1 * (1 - exp(logsf2 - logsf1))
        return float(sf1 * (-math.expm1(logsf2 - logsf1)))
    except Exception:
        return float(A_probit(k + 1, a, b) - A_probit(k, a, b))


# -----------------------------
# Oracle difficulty helpers (full 64-answer fits)
# -----------------------------
def fit_question_difficulty_params(
    question: QuestionRecord,
    *,
    k_max_curve: int = 40,
    curve_mc_trials: int = 2000,
) -> Optional[Tuple[float, float]]:
    """Fit (a, b) parameters from the full answer pool for oracle evaluation."""
    train_label = _training_label(question)
    if train_label is None or not question.answers:
        return None

    try:
        A = estimate_accuracy_curve_from_pool(
            question.answers,
            train_label,
            k_max=k_max_curve,
            num_trials=curve_mc_trials,
        )
        a_q, b_q = fit_2param_probit_sqrtk(A, k_min=3, k_max=min(k_max_curve, len(A) - 1))
    except Exception:
        return None
    return a_q, b_q


def compute_question_param_map(
    questions: Sequence[QuestionRecord],
    *,
    k_max_curve: int = 40,
    curve_mc_trials: int = 2000,
) -> Dict[str, Tuple[float, float]]:
    """Compute (a, b) fits for a list of questions."""
    params: Dict[str, Tuple[float, float]] = {}
    for q in questions:
        fitted = fit_question_difficulty_params(q, k_max_curve=k_max_curve, curve_mc_trials=curve_mc_trials)
        if fitted is not None:
            params[q.qid] = fitted
    return params


def build_oracle_difficulty_model(
    train_params: Dict[str, Tuple[float, float]],
    *,
    quantiles: Sequence[float] = ORACLE_QUANTILES,
) -> Optional[OracleDifficultyModel]:
    """Estimate oracle difficulty buckets using per-question (a, b) fits."""
    if not train_params:
        return None

    a_values = np.asarray([v[0] for v in train_params.values()], dtype=float)
    b_values = np.asarray([v[1] for v in train_params.values()], dtype=float)
    if a_values.size == 0 or b_values.size == 0:
        return None

    thresholds_a = np.quantile(a_values, quantiles).astype(float) if quantiles else np.array([], dtype=float)
    thresholds_b = np.quantile(b_values, quantiles).astype(float) if quantiles else np.array([], dtype=float)

    num_bins_a = len(thresholds_a) + 1
    num_bins_b = len(thresholds_b) + 1

    counts = np.zeros((num_bins_a, num_bins_b), dtype=float)
    sum_a = np.zeros_like(counts)
    sum_b = np.zeros_like(counts)

    for a_val, b_val in zip(a_values, b_values):
        idx_a = int(np.searchsorted(thresholds_a, a_val, side="right"))
        idx_b = int(np.searchsorted(thresholds_b, b_val, side="right"))
        counts[idx_a, idx_b] += 1.0
        sum_a[idx_a, idx_b] += a_val
        sum_b[idx_a, idx_b] += b_val

    total = counts.sum()
    if total <= 0:
        return None

    probs_grid = counts / total
    default_a = float(np.mean(a_values))
    default_b = float(np.mean(b_values))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a_grid = np.divide(sum_a, counts, where=counts > 0)
        mean_b_grid = np.divide(sum_b, counts, where=counts > 0)
    mean_a_grid = np.where(counts > 0, mean_a_grid, default_a)
    mean_b_grid = np.where(counts > 0, mean_b_grid, default_b)

    return OracleDifficultyModel(
        thresholds_a=np.asarray(thresholds_a, dtype=float),
        thresholds_b=np.asarray(thresholds_b, dtype=float),
        probs_grid=probs_grid,
        mean_a_grid=mean_a_grid,
        mean_b_grid=mean_b_grid,
    )


def greedy_budget_allocation_oracle(
    model: OracleDifficultyModel,
    *,
    average_budget: float,
    B_max: int = 64,
) -> Tuple[np.ndarray, float]:
    """Greedy budget allocation over oracle difficulty bins."""
    a_flat = model.mean_a_grid.reshape(-1, order="C")
    b_flat = model.mean_b_grid.reshape(-1, order="C")
    probs_flat = model.probs_grid.reshape(-1, order="C")

    num_bins = a_flat.size
    budgets_flat = np.zeros(num_bins, dtype=int)
    used_budget = 0.0

    heap: List[Tuple[float, int]] = []
    for idx in range(num_bins):
        if probs_flat[idx] <= 0:
            continue
        gain = delta_probit(0, a_flat[idx], b_flat[idx])
        if gain > 0:
            heapq.heappush(heap, (-gain, idx))

    eps = 1e-12
    while heap and used_budget + eps < average_budget:
        neg_gain, idx = heapq.heappop(heap)
        gain = -neg_gain
        if gain <= 0:
            continue
        if budgets_flat[idx] >= B_max:
            continue

        cost = probs_flat[idx]
        if used_budget + cost > average_budget + eps:
            break

        budgets_flat[idx] += 1
        used_budget += cost

        next_gain = delta_probit(budgets_flat[idx], a_flat[idx], b_flat[idx])
        if budgets_flat[idx] < B_max and next_gain > 0:
            heapq.heappush(heap, (-next_gain, idx))

    # Ensure every bin receives at least one sample to avoid degenerate skips
    for idx in range(num_bins):
        if budgets_flat[idx] == 0 and budgets_flat[idx] < B_max:
            budgets_flat[idx] = 1
            used_budget += probs_flat[idx]

    budget_grid = budgets_flat.reshape(model.mean_a_grid.shape, order="C")
    return budget_grid, float(used_budget)


def locate_param_bin_oracle(
    a_value: float,
    b_value: float,
    thresholds_a: Sequence[float],
    thresholds_b: Sequence[float],
) -> Tuple[int, int]:
    """Locate oracle bin index for (a, b) parameters."""
    idx_a = int(np.searchsorted(thresholds_a, a_value, side="right"))
    idx_b = int(np.searchsorted(thresholds_b, b_value, side="right"))
    return idx_a, idx_b


def evaluate_oracle_bucketed(
    test_questions: Sequence[QuestionRecord],
    question_params: Dict[str, Tuple[float, float]],
    *,
    thresholds_a: Sequence[float],
    thresholds_b: Sequence[float],
    budget_grid: np.ndarray,
) -> Dict[str, float]:
    """Evaluate oracle allocation using difficulty labels computed from full answers."""
    evaluated_acc = 0
    correct_acc = 0
    evaluated_cons = 0
    correct_cons = 0
    skipped = 0
    total_budget_used = 0.0
    per_bucket_budget = np.zeros_like(budget_grid, dtype=float)

    for q in test_questions:
        params = question_params.get(q.qid)
        if params is None or not q.answers:
            skipped += 1
            continue
        a_val, b_val = params
        idx_a, idx_b = locate_param_bin_oracle(a_val, b_val, thresholds_a, thresholds_b)
        if idx_a >= budget_grid.shape[0] or idx_b >= budget_grid.shape[1]:
            skipped += 1
            continue

        budget = int(budget_grid[idx_a, idx_b])
        if budget <= 0:
            skipped += 1
            continue

        samples = list(q.answers)[:budget]
        if not samples:
            skipped += 1
            continue

        total_budget_used += float(len(samples))
        per_bucket_budget[idx_a, idx_b] += len(samples)

        pred = majority_vote_with_tie_break(samples)
        if pred is None:
            skipped += 1
            continue

        if q.correct is not None:
            evaluated_acc += 1
        if pred == q.correct:
                correct_acc += 1
        if q.final is not None:
            evaluated_cons += 1
            if pred == q.final:
                correct_cons += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    return {
        "accuracy": accuracy,
        "consistency": consistency,
        "skipped": float(skipped),
        "total_budget_used": total_budget_used,
        "per_bucket_budget": per_bucket_budget,
    }


# -----------------------------
# Voting
# -----------------------------
def majority_vote_with_tie_break(
    samples: Sequence,
    *,
    rng: Optional[np.random.Generator] = None,
):
    """Majority vote with deterministic tie-break (alphabetical/min)."""
    from collections import Counter

    ctr = Counter(samples)
    maxc = max(ctr.values())
    winners = [x for x, c in ctr.items() if c == maxc]
    # Tie-break to match predictor_param2: pick the minimal (alphabetical/ordered) entry
    return min(winners)


# -----------------------------
# Training: estimate pi_q(t)
# -----------------------------
def estimate_pi_q_via_subsample4(
    answers: Sequence,
    *,
    num_draws: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Estimate pi_q(t)=P(T_q=t | question q) by repeatedly drawing 4 samples.

    Output:
      pi_q: np.ndarray shape (5,), sums to 1
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(answers)
    if n < 4:
        raise ValueError("need at least 4 answers to subsample 4")

    counts = np.zeros(NUM_BUCKETS, dtype=float)
    for _ in range(num_draws):
        idx = rng.choice(n, size=4, replace=False)
        s4 = [answers[i] for i in idx]
        t = bucket_from_samples4(s4)  # 1..5
        counts[t - 1] += 1
    pi_q = counts / counts.sum()
    return pi_q


def training_fit_all_questions(
    train_questions: Sequence[QuestionRecord],
    *,
    k_max_curve: int = 40,
    subsample4_draws: int = 2000,
    curve_mc_trials: int = 2000,
    rng_seed: int = 0,
) -> List[PerQuestionFit]:
    """Fit (a_q,b_q) and pi_q for each training question."""
    rng = np.random.default_rng(rng_seed)
    outputs: List[PerQuestionFit] = []

    for q in train_questions:
        train_label = _training_label(q)
        if train_label is None:
            raise ValueError(
                f"Training question {q.qid} missing training label (need `final` or `correct` for A_q(k))"
            )

        # 1) Estimate A_q(k) (placeholder MC; replace with your method)
        A = estimate_accuracy_curve_from_pool(
            q.answers, train_label, k_max=k_max_curve, num_trials=curve_mc_trials, rng=rng
        )

        # 2) Fit (a_q,b_q)
        a_q, b_q = fit_2param_probit_sqrtk(A, k_min=3, k_max=min(k_max_curve, len(A) - 1))

        # 3) Estimate pi_q(t)
        pi_q = estimate_pi_q_via_subsample4(q.answers, num_draws=subsample4_draws, rng=rng)

        outputs.append(PerQuestionFit(qid=q.qid, a_q=a_q, b_q=b_q, pi_q=pi_q))

    return outputs


def aggregate_bucket_stats(fits: Sequence[PerQuestionFit]) -> BucketStats:
    """Compute pi_t and (a_t,b_t) via soft assignment."""
    if not fits:
        raise ValueError("empty fits")

    # pi_t = E_q[pi_q(t)]
    pi_stack = np.stack([f.pi_q for f in fits], axis=0)  # (Q,5)
    pi_t = pi_stack.mean(axis=0)
    pi_t = pi_t / pi_t.sum()

    # (a_t,b_t) = weighted average of per-question params with weights pi_q(t)
    a_qs = np.array([f.a_q for f in fits], dtype=float)  # (Q,)
    b_qs = np.array([f.b_q for f in fits], dtype=float)  # (Q,)

    weights = pi_stack  # (Q,5)
    denom = weights.sum(axis=0) + 1e-12

    a_t = (weights.T @ a_qs) / denom
    b_t = (weights.T @ b_qs) / denom

    return BucketStats(pi_t=pi_t, a_t=a_t, b_t=b_t)


# -----------------------------
# Offline budget calibration
# -----------------------------
def solve_budget_plan_greedy_marginal(
    stats: BucketStats,
    *,
    B_bar: float,
    B_max: int = 64,
    k0: int = K0,
) -> BudgetPlan:
    """Greedy allocation under average budget, mirroring predictor_conf Algorithm 1."""
    pi = stats.pi_t
    a_t = stats.a_t
    b_t = stats.b_t

    # Initialize budgets at k0
    B = np.full(NUM_BUCKETS, k0, dtype=int)
    used_budget = float(np.sum(pi * B))

    if used_budget >= B_bar:
        # Already at or above budget; return baseline warm-up
        return BudgetPlan(B_t=B)

    # Max-heap via negative gain
    heap: List[Tuple[float, int, int]] = []
    for t in range(NUM_BUCKETS):
        if B[t] >= B_max:
            continue
        gain = delta_probit(int(B[t]), float(a_t[t]), float(b_t[t]))
        if gain > 0:
            heapq.heappush(heap, (-gain, 1, t))

    eps = 1e-12
    while heap and used_budget + eps < B_bar:
        neg_gain, step, t = heapq.heappop(heap)
        gain = -neg_gain
        cost = step * pi[t]

        # If this step would exceed budget, skip it (other buckets may still fit).
        if used_budget + cost > B_bar + eps:
            continue

        # Apply allocation
        B[t] += step
        used_budget += cost

        # Push next marginal gain if possible
        if B[t] >= B_max:
            continue
        next_gain = delta_probit(int(B[t]), float(a_t[t]), float(b_t[t]))
        if next_gain > 0:
            heapq.heappush(heap, (-next_gain, 1, t))

    return BudgetPlan(B_t=B)


# -----------------------------
# Testing / streaming policy
# -----------------------------
def streaming_allocate_for_question(
    sampler_fn,
    *,
    budget_plan: BudgetPlan,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Run streaming policy for one incoming question.

    Input:
      sampler_fn: callable that returns ONE new model response each time you call it.
                  e.g., sampler_fn() -> "A"/"B"/"C"/"D"
      budget_plan: B_t array
    Output:
      dict with:
        - 'bucket': assigned bucket t (1..5)
        - 'B_target': target total budget
        - 'samples': list of collected responses (length B_target)
        - 'final_pred': majority vote prediction after B_target samples
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Warm up 4 samples
    # K0 是全局常量，通常等于4，用于初始采样判断 bucket；见文件顶部或相关常量定义
    samples = [sampler_fn() for _ in range(K0)]
    t = bucket_from_samples4(samples)  # 1..5
    B_target = int(budget_plan.B_t[t - 1])

    # 2) Collect until reaching B_target
    while len(samples) < B_target:
        samples.append(sampler_fn())

    final_pred = majority_vote_with_tie_break(samples, rng=rng)
    return {
        "bucket": t,
        "B_target": B_target,
        "samples": samples,
        "final_pred": final_pred,
    }


# -----------------------------
# High-level orchestration
# -----------------------------
def load_gpqa_jsonl(path: str, *, conf_metric: str = "mean") -> List[QuestionRecord]:
    """Load GPQA JSONL file into QuestionRecord objects.

    conf_metric:
      Which scalar confidence to extract from each trace_confidence entry.
      Examples: "mean"/"Conf" (mean_confidence), "tail", "bottom", or a raw key name.
    """
    records: List[QuestionRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("id") or f"q{len(records)}"
            answers = obj.get("answers", [])
            correct = obj.get("correct_letter") or obj.get("correct") or obj.get("answer")
            final = obj.get("final")

            raw_confs = obj.get("trace_confidence")
            confs: Optional[List[float]] = None
            if isinstance(raw_confs, list) and raw_confs:
                confs = []
                for i in range(len(answers) if isinstance(answers, list) else len(raw_confs)):
                    entry = raw_confs[i] if i < len(raw_confs) else None
                    confs.append(_extract_confidence(entry, metric=str(conf_metric)))

            records.append(QuestionRecord(qid=qid, answers=answers, correct=correct, final=final, confs=confs))
    return records


def make_sampler_from_answers(
    answers: Sequence,
    *,
    rng: Optional[np.random.Generator] = None,
):
    """Return a callable that emits answers sequentially (prefix-first, deterministic).

    This mirrors predictor_param2's evaluation sampling: allocating B samples
    for a question simply takes the first B answers from the pool. `rng` is
    kept for API compatibility but not used here.
    """
    # rng kept to maintain the call signature; sampling is deterministic
    pool = list(answers)
    if not pool:
        raise ValueError("empty answers for sampler")

    def _sampler():
        if not pool:
            raise RuntimeError("sampler exhausted: no more answers in pool")
        return pool.pop(0)

    return _sampler


def evaluate_streaming(
    test_questions: Sequence[QuestionRecord],
    budget_plan: BudgetPlan,
    *,
    rng_seed: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Run streaming policy on a test split and compute (accuracy, consistency) + budget.

    - accuracy: compare prediction vs `QuestionRecord.correct` (e.g. correct_letter), if present
    - consistency: compare prediction vs `QuestionRecord.final`, if present
    """
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

        sampler_fn = make_sampler_from_answers(q.answers, rng=rng)
        out = streaming_allocate_for_question(sampler_fn, budget_plan=budget_plan, rng=rng)
        total_budget_used += float(len(out["samples"]))

        pred = out.get("final_pred")
        is_correct = None
        is_consistent = None
        if q.correct is not None:
            evaluated_acc += 1
            is_correct = bool(pred == q.correct)
            correct_acc += int(is_correct)
        if q.final is not None:
            evaluated_cons += 1
            is_consistent = bool(pred == q.final)
            correct_cons += int(is_consistent)
        if q.correct is None and q.final is None:
            skipped += 1

        out.update(
            {
                "qid": q.qid,
                "correct": q.correct,
                "final": q.final,
                "is_correct": is_correct,
                "is_consistent": is_consistent,
            }
        )
        results.append(out)

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    metrics = {
        "accuracy": accuracy,
        "consistency": consistency,
        "skipped": float(skipped),
        "total_budget_used": total_budget_used,
    }
    return metrics, results


def evaluate_fixed_budget_majority(
    test_questions: Sequence[QuestionRecord],
    per_question_budget: int,
    *,
    rng_seed: int = 0,
) -> Dict[str, float]:
    """Baseline: uniform fixed budget per question with majority vote.

    - accuracy: compare prediction vs `QuestionRecord.correct` (if present)
    - consistency: compare prediction vs `QuestionRecord.final` (if present)
    """
    rng = np.random.default_rng(rng_seed)
    budget = max(K0, int(per_question_budget))

    evaluated_acc = 0
    evaluated_cons = 0
    skipped = 0
    correct_acc = 0
    correct_cons = 0
    total_budget_used = 0.0

    for q in test_questions:
        answers = list(q.answers)
        if not answers:
            skipped += 1
            continue

        k = min(budget, len(answers))
        samples = answers[:k]
        total_budget_used += float(len(samples))

        pred = majority_vote_with_tie_break(samples, rng=rng)
        if q.correct is not None:
            evaluated_acc += 1
            correct_acc += int(pred == q.correct)
        if q.final is not None:
            evaluated_cons += 1
            correct_cons += int(pred == q.final)
        if q.correct is None and q.final is None:
            skipped += 1

    accuracy = correct_acc / evaluated_acc if evaluated_acc else float("nan")
    consistency = correct_cons / evaluated_cons if evaluated_cons else float("nan")
    return {
        "accuracy": accuracy,
        "consistency": consistency,
        "skipped": float(skipped),
        "total_budget_used": total_budget_used,
    }


def shuffle_question_records(
    records: Sequence[QuestionRecord],
    rng: random.Random,
) -> List[QuestionRecord]:
    """Return a copy of QuestionRecords with answers shuffled per question."""
    shuffled: List[QuestionRecord] = []
    for record in records:
        answers_obj = record.answers
        confs_obj = record.confs
        if isinstance(answers_obj, Sequence) and not isinstance(answers_obj, (str, bytes)):
            answers_list = list(answers_obj)
            indices = list(range(len(answers_list)))
            rng.shuffle(indices)
            shuffled_answers = [answers_list[idx] for idx in indices]
            if isinstance(answers_obj, tuple):
                shuffled_answers = tuple(shuffled_answers)

            shuffled_confs = None
            if isinstance(confs_obj, Sequence) and not isinstance(confs_obj, (str, bytes)):
                confs_list = list(confs_obj)
                shuffled_confs = [confs_list[idx] if idx < len(confs_list) else 1.0 for idx in indices]
        else:
            shuffled_answers = answers_obj
            shuffled_confs = confs_obj
        shuffled.append(
            QuestionRecord(
                qid=record.qid,
                answers=shuffled_answers,
                correct=record.correct,
                final=record.final,
                confs=shuffled_confs,
            )
        )
    return shuffled


def shuffle_subsample_and_relabel_question_records(
    records: Sequence[QuestionRecord],
    rng: random.Random,
    *,
    pool_size: Optional[int] = None,
    relabel_with_pool_mv: bool = False,
) -> List[QuestionRecord]:
    """Shuffle per-question answer order, optionally take a prefix as the pool, and optionally relabel gold via MV.

    This supports multi-run experiments where each run:
    - shuffles the original pool (e.g. 128)
    - takes the first K answers as this run's pool (e.g. 64)
    - defines this run's gold label as MV over that K answers (equivalent to old 'final')
    """
    k = None if pool_size is None else max(0, int(pool_size))
    out: List[QuestionRecord] = []
    for record in records:
        answers_obj = record.answers
        confs_obj = record.confs
        if isinstance(answers_obj, Sequence) and not isinstance(answers_obj, (str, bytes)):
            answers_list = list(answers_obj)
            indices = list(range(len(answers_list)))
            rng.shuffle(indices)
            shuffled_answers = [answers_list[idx] for idx in indices]
            shuffled_confs = None
            if isinstance(confs_obj, Sequence) and not isinstance(confs_obj, (str, bytes)):
                confs_list = list(confs_obj)
                shuffled_confs = [confs_list[idx] if idx < len(confs_list) else 1.0 for idx in indices]
            if k is not None and k > 0:
                shuffled_answers = shuffled_answers[: min(k, len(shuffled_answers))]
                if shuffled_confs is not None:
                    shuffled_confs = shuffled_confs[: min(k, len(shuffled_confs))]
        else:
            shuffled_answers = answers_obj
            shuffled_confs = confs_obj

        # Preserve ground-truth label; only relabel the pseudo-gold (`final`) when requested.
        new_final = record.final
        if relabel_with_pool_mv and isinstance(shuffled_answers, Sequence) and not isinstance(shuffled_answers, (str, bytes)):
            if shuffled_answers:
                new_final = majority_vote_with_tie_break(shuffled_answers)

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


def aggregate_multi_run_accuracy_stats(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]]
) -> Dict[str, List[Dict[str, Union[float, int]]]]:
    """Compute accuracy statistics grouped by total budget for predictor/baseline/oracle."""
    if not sweep_runs:
        return {"predictor": [], "baseline": [], "oracle": []}

    def _default_bucket() -> Dict[str, List[float]]:
        return {
            "totals": [],
            "accuracies": [],
            "avg_budgets": [],
        }

    predictor_data: Dict[int, Dict[str, List[float]]] = defaultdict(_default_bucket)
    baseline_data: Dict[int, Dict[str, List[float]]] = defaultdict(_default_bucket)
    oracle_data: Dict[int, Dict[str, List[float]]] = defaultdict(_default_bucket)

    for _, rows in sweep_runs:
        for row in rows:
            avg_budget = float(row.get("average_budget", 0.0))
            avg_key = int(round(avg_budget))

            pred_total_raw = float(row.get("predictor_total", 0.0))
            base_total_raw = float(row.get("baseline_total", 0.0))

            predictor_bucket = predictor_data[avg_key]
            predictor_bucket["totals"].append(pred_total_raw)
            predictor_bucket["accuracies"].append(float(row.get("predictor_accuracy", 0.0)))
            predictor_bucket["avg_budgets"].append(avg_budget)

            baseline_bucket = baseline_data[avg_key]
            baseline_bucket["totals"].append(base_total_raw)
            baseline_bucket["accuracies"].append(float(row.get("baseline_accuracy", 0.0)))
            baseline_bucket["avg_budgets"].append(avg_budget)

            oracle_total_raw = row.get("oracle_total")
            oracle_acc_raw = row.get("oracle_accuracy")
            if oracle_total_raw is not None and oracle_acc_raw is not None:
                oracle_bucket = oracle_data[avg_key]
                oracle_bucket["totals"].append(float(oracle_total_raw))
                oracle_bucket["accuracies"].append(float(oracle_acc_raw))
                oracle_bucket["avg_budgets"].append(avg_budget)

    def _summarize(buckets: Dict[int, Dict[str, List[float]]]) -> List[Dict[str, Union[float, int]]]:
        summaries: List[Dict[str, Union[float, int]]] = []
        for key in sorted(buckets.keys()):
            totals = np.asarray(buckets[key]["totals"], dtype=float)
            accuracies = np.asarray(buckets[key]["accuracies"], dtype=float)
            avg_budgets = np.asarray(buckets[key]["avg_budgets"], dtype=float)

            acc_mean = float(np.mean(accuracies)) if accuracies.size else float("nan")
            if accuracies.size > 1:
                acc_std = float(np.std(accuracies, ddof=1))
            else:
                acc_std = 0.0 if accuracies.size else float("nan")

            total_mean = float(np.mean(totals)) if totals.size else float("nan")
            if totals.size > 1:
                total_std = float(np.std(totals, ddof=1))
            else:
                total_std = 0.0 if totals.size else float("nan")

            avg_budget_mean = float(np.mean(avg_budgets)) if avg_budgets.size else float("nan")
            if avg_budgets.size > 1:
                avg_budget_std = float(np.std(avg_budgets, ddof=1))
            else:
                avg_budget_std = 0.0 if avg_budgets.size else float("nan")

            summaries.append(
                {
                    "avg_budget_rounded": float(key),
                    "total_budget": float(np.mean(totals)) if totals.size else float("nan"),
                    "total_mean": total_mean,
                    "total_std": total_std,
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "avg_budget_mean": avg_budget_mean,
                    "avg_budget_std": avg_budget_std,
                    "num_runs": int(accuracies.size),
                }
            )
        return summaries

    return {
        "predictor": _summarize(predictor_data),
        "baseline": _summarize(baseline_data),
        "oracle": _summarize(oracle_data),
    }


def sweep_average_budgets(
    stats: BucketStats,
    test_questions: Sequence[QuestionRecord],
    *,
    sweep_max: int,
    B_max: int,
    rng_seed: int = 0,
    oracle_model: Optional[OracleDifficultyModel] = None,
    oracle_test_params: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, object]]:
    """Sweep average budgets and collect predictor/baseline metrics."""
    rows: List[Dict[str, object]] = []

    start_budget = max(K0, 1)
    for avg_budget in range(start_budget, sweep_max + 1):
        plan = solve_budget_plan_greedy_marginal(stats, B_bar=float(avg_budget), B_max=B_max, k0=K0)
        predictor_metrics, _ = evaluate_streaming(test_questions, plan, rng_seed=rng_seed)
        baseline_metrics = evaluate_fixed_budget_majority(
            test_questions, per_question_budget=avg_budget, rng_seed=rng_seed
        )
        expected_budget = float(np.sum(stats.pi_t * plan.B_t))

        oracle_metrics: Optional[Dict[str, object]] = None
        oracle_budget_grid: Optional[np.ndarray] = None
        oracle_expected = None
        if oracle_model is not None and oracle_test_params is not None:
            oracle_budget_grid, oracle_expected_val = greedy_budget_allocation_oracle(
                oracle_model,
                average_budget=float(avg_budget),
                B_max=B_max,
            )
            oracle_metrics = evaluate_oracle_bucketed(
                test_questions,
                oracle_test_params,
                thresholds_a=oracle_model.thresholds_a,
                thresholds_b=oracle_model.thresholds_b,
                budget_grid=oracle_budget_grid,
            )
            oracle_expected = float(np.sum(oracle_model.probs_grid * oracle_budget_grid))

        rows.append(
            {
                "average_budget": avg_budget,
                "predictor_total": predictor_metrics["total_budget_used"],
                "predictor_accuracy": predictor_metrics["accuracy"],
                "predictor_consistency": predictor_metrics.get("consistency"),
                "predictor_expected": expected_budget,
                "predictor_skipped": predictor_metrics.get("skipped"),
                "baseline_total": baseline_metrics["total_budget_used"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_consistency": baseline_metrics.get("consistency"),
                "baseline_skipped": baseline_metrics.get("skipped"),
                "budget_plan": plan.B_t.tolist(),
                "oracle_total": oracle_metrics["total_budget_used"] if oracle_metrics else None,
                "oracle_accuracy": oracle_metrics["accuracy"] if oracle_metrics else None,
                "oracle_consistency": oracle_metrics.get("consistency") if oracle_metrics else None,
                "oracle_expected": oracle_expected,
                "oracle_budget_grid": oracle_budget_grid.tolist() if oracle_budget_grid is not None else None,
            }
        )

    return rows


# -----------------------------
# Plotting helpers (reference style from gpqa_offline.py)
# -----------------------------
def aggregate_multi_run_curve_stats(
    curve_runs: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]]
) -> Dict[str, List[Dict[str, Union[int, float]]]]:
    """Aggregate curves across repeated runs keyed by label."""
    if not curve_runs:
        return {}

    buckets: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _, curves in curve_runs:
        if not curves:
            continue
        for label, points in curves.items():
            if not points:
                continue
            for budget, val in points:
                try:
                    b_val = int(budget)
                    v_val = float(val)
                except Exception:
                    continue
                if not math.isfinite(v_val):
                    continue
                buckets[label][b_val].append(v_val)

    summaries: Dict[str, List[Dict[str, Union[int, float]]]] = {}
    for label, budget_map in buckets.items():
        entries: List[Dict[str, Union[int, float]]] = []
        for b in sorted(budget_map):
            vals = np.asarray(budget_map[b], dtype=float)
            mean = float(np.mean(vals)) if vals.size else float("nan")
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else (0.0 if vals.size else float("nan"))
            entries.append(
                {
                    "budget": int(b),
                    "mean": mean,
                    "std": std,
                    "num_runs": int(vals.size),
                }
            )
        summaries[label] = entries
    return summaries


def _apply_reference_plot_style(ax, *, grid: bool = True) -> None:
    """Apply a clean, publication-style plot look (mirrors gpqa_offline.py)."""
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    if grid:
        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=1.0,
            color="#D9D9D9",
            alpha=0.8,
        )
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFCFCF")
        spine.set_linewidth(1.0)

    ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=1.0,
        colors="#333333",
        labelsize=11,
        pad=3,
        top=False,
        right=False,
    )


def plot_multi_run_curves(
    curve_runs: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    output_path: Optional[Union[str, Path]],
    *,
    csv_path: Optional[Union[str, Path]] = None,
    overlay_runs: bool = False,
    y_margin: float = 0.03,
    y_max_cap: float = 1.01,
) -> None:
    """Plot mean ± std curves over repeated runs and optionally save a summary CSV.

    Notes:
    - Uses paper-friendly style (no title/legend/axis labels).
    - CSV schema matches gpqa_offline: label,budget,mean,std,num_runs
    """
    if not curve_runs or not output_path:
        return
    output_path = str(output_path)
    if csv_path is None:
        csv_path = str(Path(output_path).with_suffix(".csv"))

    summaries = aggregate_multi_run_curve_stats(curve_runs)
    if not summaries:
        return

    num_runs = len(curve_runs)
    warm_colors = plt.cm.OrRd(np.linspace(0.4, 0.95, max(num_runs, 1)))
    cool_colors = plt.cm.Blues(np.linspace(0.4, 0.95, max(num_runs, 1)))
    oracle_colors = plt.cm.Greys(np.linspace(0.4, 0.85, max(num_runs, 1)))

    PRED_COLOR = "#9c6ed9"
    BASE_COLOR = "#F0B851"
    PRED_CONF_COLOR = "#bd9ee6"
    BASE_CONF_COLOR = "#F5D000"
    ORACLE_COLOR = "#6c757d"
    ORACLE_CONF_COLOR = "#8c9094"
    OTHER_COLOR = "#6c757d"

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    _apply_reference_plot_style(ax, grid=True)

    def _label_kind_for(name: str) -> str:
        low = name.lower()
        if "oracle" in low:
            return "oracle"
        if low.startswith("okg") or low.startswith("pred"):
            return "okg"
        if low.startswith("base") or low.startswith("mv"):
            return "base"
        return "other"

    def _run_color_for(label: str, idx: int) -> str:
        kind = _label_kind_for(label)
        palette = warm_colors if kind == "okg" else cool_colors if kind == "base" else oracle_colors
        return palette[min(idx, len(palette) - 1)]

    def _is_main_label(kind: str, label: str) -> bool:
        low = label.lower()
        if kind == "okg":
            return low in {"okg", "pred"}
        if kind == "base":
            return low == "base" or low == "mv"
        return False

    def _is_conf_label(label: str) -> bool:
        return "conf" in label.lower()

    labels_by_kind: Dict[str, List[str]] = defaultdict(list)
    for lab, entries in summaries.items():
        if entries:
            labels_by_kind[_label_kind_for(lab)].append(lab)

    mean_color_for_label: Dict[str, str] = {}
    for kind, labs in labels_by_kind.items():
        for lab in labs:
            if kind == "okg":
                mean_color_for_label[lab] = PRED_CONF_COLOR if _is_conf_label(lab) else PRED_COLOR
            elif kind == "base":
                mean_color_for_label[lab] = BASE_CONF_COLOR if _is_conf_label(lab) else BASE_COLOR
            elif kind == "oracle":
                mean_color_for_label[lab] = ORACLE_CONF_COLOR if _is_conf_label(lab) else ORACLE_COLOR
            else:
                mean_color_for_label[lab] = OTHER_COLOR

    if overlay_runs:
        run_alpha = 0.15
        for run_idx, curves in curve_runs:
            for label, points in curves.items():
                if not points:
                    continue
                try:
                    budgets, ys = zip(*points)
                except ValueError:
                    continue
                kind = _label_kind_for(label)
                color = _run_color_for(label, run_idx)
                ls = "-" if kind == "okg" else "--" if kind == "base" else ":" if kind == "oracle" else "-."
                ax.plot(
                    budgets,
                    ys,
                    color=color,
                    alpha=run_alpha,
                    linewidth=1.0,
                    linestyle=ls,
                    marker="",
                    label="_nolegend_",
                )

    all_means: List[float] = []
    for label, entries in sorted(summaries.items()):
        if not entries:
            continue
        budgets = [int(e["budget"]) for e in entries]
        means = np.asarray([float(e["mean"]) for e in entries], dtype=float)
        stds = np.asarray([float(e["std"]) for e in entries], dtype=float)
        all_means.extend(means.tolist())

        kind = _label_kind_for(label)
        color = mean_color_for_label.get(
            label,
            PRED_COLOR if kind == "okg" else BASE_COLOR if kind == "base" else ORACLE_COLOR,
        )
        if _is_conf_label(label):
            ls = "--"
            lw = 2.6
        else:
            ls = "-"
            lw = 3.3
        ax.plot(
            budgets,
            means,
            color=color,
            linewidth=lw,
            marker="",
            linestyle=ls,
            label="_nolegend_",
        )
        if _is_main_label(kind, label) and (not _is_conf_label(label)):
            ax.fill_between(
                budgets,
                means - stds,
                means + stds,
                color=color,
                alpha=0.12,
                label="_nolegend_",
                edgecolor="none",
                linewidth=0,
            )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    if all_means:
        ymin = max(0.0, min(all_means) - float(y_margin))
        ymax = min(float(y_max_cap), max(all_means) + float(y_margin))
        if ymax <= ymin:
            ymax = min(float(y_max_cap), ymin + 0.05)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[multi-run] plot saved: {output_path}")

    if csv_path:
        csv_rows: List[Dict[str, Union[int, float, str]]] = []
        for label, entries in summaries.items():
            for e in entries:
                csv_rows.append(
                    {
                        "label": label,
                        "budget": e.get("budget", float("nan")),
                        "mean": e.get("mean", float("nan")),
                        "std": e.get("std", float("nan")),
                        "num_runs": e.get("num_runs", 0),
                    }
                )
        if csv_rows:
            csv_path_obj = Path(str(csv_path))
            csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with csv_path_obj.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["label", "budget", "mean", "std", "num_runs"])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[multi-run] summary CSV saved: {csv_path_obj}")


def plot_accuracy_multi_run_curves(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    output_path: Optional[Union[str, Path]],
    *,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot aggregated accuracy curve over sweep runs.

    When per-run realized total budgets differ for the same sweep avg_budget, we align by
    avg_budget and use:
      x = mean(total_budget_used across runs)
      y = mean(accuracy across runs)
    """
    plot_multi_run_sweep_runs(
        sweep_runs,
        output_path,
        metric="accuracy",
        csv_path=csv_path,
        y_margin=0.007,
        y_max_cap=1.0,
    )


def plot_consistency_multi_run_curves(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    output_path: Optional[Union[str, Path]],
    *,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    plot_multi_run_sweep_runs(
        sweep_runs,
        output_path,
        metric="consistency",
        csv_path=csv_path,
        y_margin=0.03,
        y_max_cap=1.01,
    )


def aggregate_multi_run_sweep_xy_stats(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    *,
    metric: str,
) -> Dict[str, List[Dict[str, Union[int, float]]]]:
    """Aggregate multi-run sweep results by avg_budget, but report x as mean(total_budget).

    Returns per label entries:
      - avg_budget (int)
      - total_mean, total_std
      - metric_mean, metric_std
      - num_runs
    """
    if not sweep_runs:
        return {}

    label_to_keys = {
        "Pred": ("predictor_total", f"predictor_{metric}"),
        "Pred_Conf": ("predictor_total", f"predictor_{metric}_conf"),
        "Base": ("baseline_total", f"baseline_{metric}"),
        "Base_Conf": ("baseline_total", f"baseline_{metric}_conf"),
        "Oracle": ("oracle_total", f"oracle_{metric}"),
        "Oracle_Conf": ("oracle_total", f"oracle_{metric}_conf"),
    }

    # buckets[label][avg_budget] -> {"totals": [...], "metrics": [...]}
    buckets: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: {"totals": [], "metrics": []}))

    for _run_idx, rows in sweep_runs:
        for row in rows or []:
            avg_b = row.get("average_budget")
            if avg_b is None:
                continue
            try:
                avg_key = int(avg_b)
            except Exception:
                continue

            for label, (x_key, y_key) in label_to_keys.items():
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
                buckets[label][avg_key]["totals"].append(xx)
                buckets[label][avg_key]["metrics"].append(yy)

    summaries: Dict[str, List[Dict[str, Union[int, float]]]] = {}
    for label, by_budget in buckets.items():
        entries: List[Dict[str, Union[int, float]]] = []
        for avg_key in sorted(by_budget.keys()):
            totals = np.asarray(by_budget[avg_key]["totals"], dtype=float)
            metrics = np.asarray(by_budget[avg_key]["metrics"], dtype=float)
            if totals.size == 0 or metrics.size == 0:
                continue
            total_mean = float(np.mean(totals))
            total_std = float(np.std(totals, ddof=1)) if totals.size > 1 else 0.0
            metric_mean = float(np.mean(metrics))
            metric_std = float(np.std(metrics, ddof=1)) if metrics.size > 1 else 0.0
            entries.append(
                {
                    "avg_budget": int(avg_key),
                    "total_mean": total_mean,
                    "total_std": total_std,
                    "metric_mean": metric_mean,
                    "metric_std": metric_std,
                    "num_runs": int(min(totals.size, metrics.size)),
                }
            )
        if entries:
            summaries[label] = entries
    return summaries


def plot_multi_run_sweep_runs(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    output_path: Optional[Union[str, Path]],
    *,
    metric: str,
    csv_path: Optional[Union[str, Path]] = None,
    y_margin: float = 0.03,
    y_max_cap: float = 1.01,
) -> None:
    """Reference-style plot for sweep runs with x=mean(total_budget), y=mean(metric)."""
    if not sweep_runs or not output_path:
        return
    output_path = str(output_path)
    if csv_path is None:
        csv_path = str(Path(output_path).with_suffix(".csv"))

    summaries = aggregate_multi_run_sweep_xy_stats(sweep_runs, metric=metric)
    if not summaries:
        return

    PRED_COLOR = "#9c6ed9"
    Pred_Conf_COLOR = "#bd9ee6"
    BASE_COLOR = "#F0B851"
    BASE_Conf_COLOR = "#F5D000"
    ORACLE_COLOR = "#6c757d"
    ORACLE_Conf_COLOR = "#8c9094"

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    _apply_reference_plot_style(ax, grid=True)

    def _kind(label: str) -> str:
        low = str(label).lower()
        if "oracle" in low:
            return "oracle"
        if low.startswith("okg") or low.startswith("pred"):
            return "okg"
        if low.startswith("base") or low.startswith("mv"):
            return "base"
        return "other"

    def _is_conf_label(label: str) -> bool:
        return "conf" in str(label).lower()

    def _color(label: str) -> str:
        k = _kind(label)
        is_conf = _is_conf_label(label)
        if k == "okg":
            return Pred_Conf_COLOR if is_conf else PRED_COLOR
        if k == "base":
            return BASE_Conf_COLOR if is_conf else BASE_COLOR
        if k == "oracle":
            return ORACLE_Conf_COLOR if is_conf else ORACLE_COLOR
        return ORACLE_Conf_COLOR if is_conf else ORACLE_COLOR

    all_means: List[float] = []
    for label, entries in sorted(summaries.items()):
        xs = np.asarray([float(e["total_mean"]) for e in entries], dtype=float)
        ys = np.asarray([float(e["metric_mean"]) for e in entries], dtype=float)
        ystd = np.asarray([float(e["metric_std"]) for e in entries], dtype=float)
        all_means.extend(ys.tolist())

        c = _color(label)
        if _is_conf_label(label):
            ls = "--"
            lw = 2.6
        else:
            ls = "-"
            lw = 3.3

        ax.plot(xs, ys, color=c, linewidth=lw, marker="", linestyle=ls, label="_nolegend_")
        # std band only for main (non-conf) curves
        if not _is_conf_label(label):
            ax.fill_between(xs, ys - ystd, ys + ystd, color=c, alpha=0.12, edgecolor="none", linewidth=0)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    if all_means:
        ymin = max(0.0, min(all_means) - float(y_margin))
        ymax = min(float(y_max_cap), max(all_means) + float(y_margin))
        if ymax <= ymin:
            ymax = min(float(y_max_cap), ymin + 0.05)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[multi-run] plot saved: {output_path}")

    if csv_path:
        csv_rows: List[Dict[str, Union[int, float, str]]] = []
        for label, entries in summaries.items():
            for e in entries:
                csv_rows.append(
                    {
                        "label": label,
                        "avg_budget": e.get("avg_budget", float("nan")),
                        "total_mean": e.get("total_mean", float("nan")),
                        "total_std": e.get("total_std", float("nan")),
                        "mean": e.get("metric_mean", float("nan")),
                        "std": e.get("metric_std", float("nan")),
                        "num_runs": e.get("num_runs", 0),
                    }
                )
        if csv_rows:
            csv_path_obj = Path(str(csv_path))
            csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with csv_path_obj.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["label", "avg_budget", "total_mean", "total_std", "mean", "std", "num_runs"],
                )
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[multi-run] summary CSV saved: {csv_path_obj}")


def _sweep_rows_to_curve_dict(
    rows: Sequence[Dict[str, object]],
    *,
    metric: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """Convert sweep rows to offline-style curves dict keyed by label."""
    out: Dict[str, List[Tuple[int, float]]] = {}
    label_to_key = {
        "Pred": f"predictor_{metric}",
        "Pred_Conf": f"predictor_{metric}_conf",
        "Base": f"baseline_{metric}",
        "Base_Conf": f"baseline_{metric}_conf",
        "Oracle": f"oracle_{metric}",
        "Oracle_Conf": f"oracle_{metric}_conf",
    }
    for label, key in label_to_key.items():
        pts: List[Tuple[int, float]] = []
        for row in rows:
            b = row.get("average_budget")
            v = row.get(key)
            if b is None or v is None:
                continue
            try:
                bb = int(b)
                vv = float(v)
            except Exception:
                continue
            if not math.isfinite(vv):
                continue
            pts.append((bb, vv))
        if pts:
            out[label] = pts
    return out


def _sweep_rows_to_curve_dict_total(
    rows: Sequence[Dict[str, object]],
    *,
    metric: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """Convert sweep rows to curves keyed by label, using realized TOTAL budget on x-axis.

    - x: total budget consumed (rounded to int for stable multi-run aggregation)
    - y: metric value (accuracy/consistency)
    """
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
            # round total budget to nearest int to make aggregation consistent
            x_to_y[int(round(xx))] = yy
        if x_to_y:
            out[label] = x_to_y
    # convert to sorted lists
    return {lab: sorted([(x, y) for x, y in mp.items()], key=lambda t: t[0]) for lab, mp in out.items()}


def _sweep_runs_to_curve_runs(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    *,
    metric: str,
) -> List[Tuple[int, Dict[str, List[Tuple[int, float]]]]]:
    curve_runs: List[Tuple[int, Dict[str, List[Tuple[int, float]]]]] = []
    for run_idx, rows in sweep_runs:
        curve_runs.append((int(run_idx), _sweep_rows_to_curve_dict(rows, metric=metric)))
    return curve_runs


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


def _sweep_runs_to_total_runs(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]]
) -> List[Tuple[int, Dict[str, Dict[int, float]]]]:
    """Build per-run mapping from avg_budget -> realized total budgets for each method.

    Returns: [(run_idx, {"Base": {...}, "Pred": {...}, "Oracle": {...}, "Base_Conf": {...}, ...}), ...]
    """
    out: List[Tuple[int, Dict[str, Dict[int, float]]]] = []
    for run_idx, rows in sweep_runs:
        base_map: Dict[int, float] = {}
        pred_map: Dict[int, float] = {}
        oracle_map: Dict[int, float] = {}
        # Conf variants reuse the same realized total budgets (they vote differently, not sample differently).
        base_conf_map: Dict[int, float] = {}
        pred_conf_map: Dict[int, float] = {}
        oracle_conf_map: Dict[int, float] = {}
        for row in rows or []:
            b = row.get("average_budget")
            if b is None:
                continue
            try:
                bb = int(b)
            except Exception:
                continue
            # predictor/baseline totals are always available
            try:
                pred_map[bb] = float(row.get("predictor_total", float("nan")))
            except Exception:
                pass
            try:
                base_map[bb] = float(row.get("baseline_total", float("nan")))
            except Exception:
                pass
            # oracle may be missing
            ot = row.get("oracle_total")
            if ot is not None:
                try:
                    oracle_map[bb] = float(ot)
                except Exception:
                    pass

            # mirror for *_Conf
            if bb in pred_map:
                pred_conf_map[bb] = pred_map[bb]
            if bb in base_map:
                base_conf_map[bb] = base_map[bb]
            if bb in oracle_map:
                oracle_conf_map[bb] = oracle_map[bb]
        out.append(
            (
                int(run_idx),
                {
                    "Base": base_map,
                    "Pred": pred_map,
                    "Oracle": oracle_map,
                    "Base_Conf": base_conf_map,
                    "Pred_Conf": pred_conf_map,
                    "Oracle_Conf": oracle_conf_map,
                },
            )
        )
    return out


def export_multi_run_curves_jsonl(
    curve_runs_consistency: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    curve_runs_accuracy: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    output_path: str,
    *,
    methods: Optional[Sequence[str]] = None,
    total_runs: Optional[Sequence[Tuple[int, Dict[str, Dict[int, float]]]]] = None,
    sweep_runs: Optional[Sequence[Tuple[int, Sequence[Dict[str, object]]]]] = None,
) -> None:
    """Export multi-run curves + requested stats into a JSONL file.

    This is adapted from `gpqa_offline.py::export_multi_run_curves_jsonl`, but the
    canonical method names for streaming are:
      - Base   (fixed baseline)
      - Pred   (our predictor / bucketed policy; label alias: OKG)
      - Oracle (oracle difficulty)
      - Base_Conf / Pred_Conf / Oracle_Conf (confidence-weighted voting variants, if present)
    """
    wanted = list(methods or ["Base", "Pred", "Oracle"])

    cons_by_run: Dict[int, Dict[str, List[Tuple[int, float]]]] = {i: d for i, d in (curve_runs_consistency or [])}
    acc_by_run: Dict[int, Dict[str, List[Tuple[int, float]]]] = {i: d for i, d in (curve_runs_accuracy or [])}
    totals_by_run: Dict[int, Dict[str, Dict[int, float]]] = {i: d for i, d in (total_runs or [])}
    run_indices = sorted(set(cons_by_run.keys()) | set(acc_by_run.keys()))

    # If caller didn't request methods explicitly, auto-include *_Conf when available.
    if methods is None:
        maybe_conf = ["Base_Conf", "Pred_Conf", "Oracle_Conf"]
        has_conf = False
        for run_idx in run_indices:
            cc = cons_by_run.get(run_idx, {}) or {}
            ac = acc_by_run.get(run_idx, {}) or {}
            if any(m in cc or m in ac for m in maybe_conf):
                has_conf = True
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
        # label aliases (case-insensitive)
        aliases: List[str] = [canonical]
        low = canonical.lower()
        if low == "base":
            aliases.extend(["mv", "base"])
        elif low == "pred":
            # historical label used in earlier codepaths
            aliases.extend(["okg", "predictor", "pred"])
        elif low == "oracle":
            aliases.extend(["oracle"])
        elif low in {"base_conf", "pred_conf", "oracle_conf"}:
            # accept common spellings
            aliases.extend([canonical, low, low.replace("_", ""), low.replace("_", "-")])
        aliases_low = {a.lower() for a in aliases if a}
        for k in curves.keys():
            if str(k).lower() in aliases_low:
                return k
        return None

    # Per-method: (1) peak-consistency avg_budget, (2) realized total budget at that avg_budget
    avg_budgets_at_cons1: Dict[str, List[Optional[int]]] = {m: [] for m in wanted}
    total_budgets_at_cons1: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}
    # Per-method metrics at Pred's peak-consistency avg_budget
    at_pred_con1_acc: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}
    at_pred_con1_cons: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}

    # If sweep_runs is provided, we can compute all derived stats in the *avg_budget* domain
    # (shared across methods), while still keeping curves in total-budget x domain.
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

            # Peak-consistency avg_budget for each method (preferred, stable across methods)
            peak_avg_for: Dict[str, Optional[int]] = {}
            if sweep_rows is not None:
                # Determine per-method peak based on sweep table values
                key_by_method = {
                    "Base": "baseline_consistency",
                    "Pred": "predictor_consistency",
                    "Oracle": "oracle_consistency",
                    "Base_Conf": "baseline_consistency_conf",
                    "Pred_Conf": "predictor_consistency_conf",
                    "Oracle_Conf": "oracle_consistency_conf",
                }
                for m in wanted:
                    mk = key_by_method.get(m)
                    peak_avg_for[m] = _first_avg_budget_at_max(sweep_rows, mk) if mk else None
            else:
                # Fallback: infer peak from curve x-domain (less ideal when x differs per method)
                for m in wanted:
                    k = _find_key(cons_curves, m)
                    peak_avg_for[m] = _first_budget_at_max(cons_curves.get(k, []) if k else [])

            # Store per-method peak avg_budget + total budget at that avg_budget
            for m in wanted:
                bpk = peak_avg_for.get(m)
                avg_budgets_at_cons1[m].append(bpk)
                if sweep_rows is not None and bpk is not None and bpk in rows_by_avg:
                    row = rows_by_avg[bpk]
                    if m.startswith("Base"):
                        total_key = "baseline_total"
                    elif m.startswith("Pred"):
                        total_key = "predictor_total"
                    else:
                        total_key = "oracle_total"
                    tb = row.get(total_key)
                    try:
                        total_budgets_at_cons1[m].append(float(tb) if tb is not None else None)
                    except Exception:
                        total_budgets_at_cons1[m].append(None)
                else:
                    # Backward compatibility
                    total_map = totals_maps.get(m, {}) or {}
                    tb = total_map.get(int(bpk)) if (bpk is not None and total_map) else (float(bpk) if bpk is not None else None)
                    total_budgets_at_cons1[m].append(tb)

            # Pred's peak-consistency avg_budget (used as the shared evaluation point)
            pred_peak_avg = peak_avg_for.get("Pred")
            if sweep_rows is not None and pred_peak_avg is not None and pred_peak_avg in rows_by_avg:
                row = rows_by_avg[pred_peak_avg]
                for m in wanted:
                    if m == "Base":
                        acc_key, cons_key = "baseline_accuracy", "baseline_consistency"
                    elif m == "Pred":
                        acc_key, cons_key = "predictor_accuracy", "predictor_consistency"
                    elif m == "Oracle":
                        acc_key, cons_key = "oracle_accuracy", "oracle_consistency"
                    elif m == "Base_Conf":
                        acc_key, cons_key = "baseline_accuracy_conf", "baseline_consistency_conf"
                    elif m == "Pred_Conf":
                        acc_key, cons_key = "predictor_accuracy_conf", "predictor_consistency_conf"
                    elif m == "Oracle_Conf":
                        acc_key, cons_key = "oracle_accuracy_conf", "oracle_consistency_conf"
                    else:
                        acc_key, cons_key = None, None
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
                # Fallback: old behavior based on curve x-domain (may yield None when x differs)
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
                # (1) per-method peak-consistency budget index (avg_budget) + realized total budget
                "Pred_avg_budget_at_cons1": peak_avg_for.get("Pred"),
                "Base_avg_budget_at_cons1": peak_avg_for.get("Base"),
                "Oracle_avg_budget_at_cons1": peak_avg_for.get("Oracle"),
                "Pred_total_budget_at_cons1": total_budgets_at_cons1["Pred"][-1] if total_budgets_at_cons1.get("Pred") else None,
                "Base_total_budget_at_cons1": total_budgets_at_cons1["Base"][-1] if total_budgets_at_cons1.get("Base") else None,
                "Oracle_total_budget_at_cons1": total_budgets_at_cons1["Oracle"][-1] if total_budgets_at_cons1.get("Oracle") else None,
                # (2) metrics at Pred's peak-consistency avg_budget (per run)
                "Base_consistency_at_con1": at_pred_con1_cons.get("Base", [None])[-1],
                "Base_accuracy_at_con1": at_pred_con1_acc.get("Base", [None])[-1],
                "Pred_consistency_at_con1": at_pred_con1_cons.get("Pred", [None])[-1],
                "Pred_accuracy_at_con1": at_pred_con1_acc.get("Pred", [None])[-1],
                "Oracle_consistency_at_con1": at_pred_con1_cons.get("Oracle", [None])[-1],
                "Oracle_accuracy_at_con1": at_pred_con1_acc.get("Oracle", [None])[-1],
            }

            # Also export *_Conf per-run stats when present.
            for m in ("Base_Conf", "Pred_Conf", "Oracle_Conf"):
                if m in wanted:
                    record[f"{m}_avg_budget_at_cons1"] = peak_avg_for.get(m)
                    record[f"{m}_total_budget_at_cons1"] = total_budgets_at_cons1.get(m, [None])[-1]
                    record[f"{m}_consistency_at_con1"] = at_pred_con1_cons.get(m, [None])[-1]
                    record[f"{m}_accuracy_at_con1"] = at_pred_con1_acc.get(m, [None])[-1]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary: Dict[str, object] = {"type": "summary", "num_runs": int(len(run_indices))}

        # NOTE: in streaming we care about realized TOTAL budget (not the avg_budget index).
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


def train_and_build_budget_plan(
    train_questions: Sequence[QuestionRecord],
    *,
    B_bar: float,
    k_max_curve: int = 40,
    subsample4_draws: int = 2000,
    curve_mc_trials: int = 2000,
    B_max: int = 64,
    rng_seed: int = 0,
) -> Tuple[BucketStats, BudgetPlan]:
    """End-to-end training: fits -> bucket stats -> budget plan."""
    fits = training_fit_all_questions(
        train_questions,
        k_max_curve=k_max_curve,
        subsample4_draws=subsample4_draws,
        curve_mc_trials=curve_mc_trials,
        rng_seed=rng_seed,
    )
    stats = aggregate_bucket_stats(fits)
    plan = solve_budget_plan_greedy_marginal(stats, B_bar=B_bar, B_max=B_max, k0=K0)
    return stats, plan


# -----------------------------
# CLI entrypoint
# -----------------------------

def weighted_majority_vote_min(samples: Sequence[object], weights: Sequence[object]) -> Optional[object]:
    """Confidence-weighted majority vote.

    Aggregates weights per label, then tie-breaks by the minimal label to mirror
    `majority_vote_with_tie_break`.
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

    Tie-break is deterministic via minimal label, matching majority_vote_with_tie_break.
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


def _pseudo_label_conf_from_full_pool(q: QuestionRecord, *, conf_variant: str) -> Optional[object]:
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
    test_questions: Sequence[QuestionRecord],
    budget_plan: BudgetPlan,
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
        if len(answers) < int(K0):
            skipped += 1
            continue
        t = bucket_from_samples4(answers[: int(K0)])
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
    test_questions: Sequence[QuestionRecord],
    per_question_budget: int,
    *,
    conf_variant: str = "weighted",
) -> Dict[str, float]:
    """Baseline evaluation with fixed prefix budget, but weighted voting."""
    budget = max(int(K0), int(per_question_budget))

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
        score_fn=lambda a, b: float(A_probit(1, float(a), float(b))),
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

        # Local import; scipy is required by the inlined streaming core.
        from scipy.stats import norm  # type: ignore

        x1 = float(a * math.sqrt(float(cur)) + b)
        x2 = float(a * math.sqrt(float(cur + 1)) + b)

        # sf(x1) >= sf(x2) when x2 > x1; guard anyway.
        logsf1 = float(norm.logsf(x1))
        logsf2 = float(norm.logsf(x2))
        if not (math.isfinite(logsf1) and math.isfinite(logsf2)):
            # Fallback: best-effort numeric (may be 0 when saturated).
            return float(A_probit(cur + 1, a, b) - A_probit(cur, a, b))

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
    train_questions: Sequence[QuestionRecord],
    test_questions: Sequence[QuestionRecord],
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
        train_params = compute_question_param_map(
            train_questions,
            k_max_curve=k_max_curve,
            curve_mc_trials=curve_mc_trials,
        )
        test_params = compute_question_param_map(
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

        pred = majority_vote_with_tie_break(samples)
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
    stats: BucketStats,
    test_questions: Sequence[QuestionRecord],
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

    start_budget = max(K0, 1)
    for avg_budget in range(start_budget, sweep_max + 1):
        plan = solve_budget_plan_greedy_marginal(stats, B_bar=float(avg_budget), B_max=B_max, k0=K0)
        predictor_metrics, _ = evaluate_streaming(test_questions, plan, rng_seed=rng_seed)
        predictor_conf_metrics: Dict[str, float] = {}
        if add_conf:
            predictor_conf_metrics = evaluate_streaming_conf(test_questions, plan, conf_variant=str(conf_variant))
        baseline_metrics = evaluate_fixed_budget_majority(
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
    all_questions = load_gpqa_jsonl(str(data_path), conf_metric=str(args.conf_metric))
    if len(all_questions) <= args.train_size:
        raise ValueError(f"Dataset needs more than {args.train_size} questions to create a test split.")

    train_questions = all_questions[: args.train_size]
    test_questions = all_questions[args.train_size :]

    # predictor training and evaluation setup
    stats, plan = train_and_build_budget_plan(
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
    predictor_metrics, _results = evaluate_streaming(test_questions, plan, rng_seed=args.rng_seed)
    baseline_metrics = evaluate_fixed_budget_majority(
        test_questions, per_question_budget=args.average_budget, rng_seed=args.rng_seed
    )

    # Oracle preparation (train/test (a,b) fits). Compute once and reuse.
    # Optionally fit oracle buckets on full data for a stronger (leaky) upper bound.
    oracle_fit_questions = all_questions if bool(getattr(args, "oracle_fit_all", False)) else train_questions
    train_params_oracle = compute_question_param_map(
        oracle_fit_questions,
        k_max_curve=args.k_max_curve,
        curve_mc_trials=args.curve_mc_trials,
    )
    test_params_oracle = compute_question_param_map(
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
                train_q_run = shuffle_subsample_and_relabel_question_records(
                    train_questions,
                    rng,
                    pool_size=int(args.multi_pool_size),
                    relabel_with_pool_mv=bool(args.multi_relabel_mv),
                )
                test_q_run = shuffle_subsample_and_relabel_question_records(
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
                train_params_run = compute_question_param_map(
                    oracle_fit_q_run,
                    k_max_curve=args.k_max_curve,
                    curve_mc_trials=args.curve_mc_trials,
                )
                test_params_run = compute_question_param_map(
                    test_q_run,
                    k_max_curve=args.k_max_curve,
                    curve_mc_trials=args.curve_mc_trials,
                )
                oracle_model_run = build_oracle_difficulty_model(
                    train_params_run, k=int(args.oracle_kmeans_k), random_seed=int(args.rng_seed + run_idx)
                )

                # predictor preparation
                stats_run, _plan_run = train_and_build_budget_plan(
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
                summaries = aggregate_multi_run_accuracy_stats(sweep_runs)
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
                curve_runs_acc = _sweep_runs_to_curve_runs_total(sweep_runs, metric="accuracy")
                curve_runs_cons = _sweep_runs_to_curve_runs_total(sweep_runs, metric="consistency")
                plot_accuracy_multi_run_curves(
                    sweep_runs,
                    args.accuracy_plot,
                    csv_path=(args.accuracy_csv if args.accuracy_csv else None),
                )
                plot_consistency_multi_run_curves(
                    sweep_runs,
                    args.consistency_plot,
                    csv_path=(args.consistency_csv if args.consistency_csv else None),
                )
                if args.multi_run_jsonl:
                    export_multi_run_curves_jsonl(
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

                curve_runs_acc = _sweep_runs_to_curve_runs_total([(0, sweep_rows)], metric="accuracy")
                curve_runs_cons = _sweep_runs_to_curve_runs_total([(0, sweep_rows)], metric="consistency")
                plot_accuracy_multi_run_curves(
                    [(0, sweep_rows)],
                    args.accuracy_plot,
                    csv_path=(args.accuracy_csv if args.accuracy_csv else None),
                )
                plot_consistency_multi_run_curves(
                    [(0, sweep_rows)],
                    args.consistency_plot,
                    csv_path=(args.consistency_csv if args.consistency_csv else None),
                )
                if args.multi_run_jsonl:
                    export_multi_run_curves_jsonl(
                        curve_runs_cons,
                        curve_runs_acc,
                        str(args.multi_run_jsonl),
                        sweep_runs=[(0, sweep_rows)],
                    )


if __name__ == "__main__":
    main()
