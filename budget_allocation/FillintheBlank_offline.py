#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIME offline OKG evaluation (fill-in numeric answers).

Compared to MMLU/GPQA (multiple-choice), AIME has **no predefined options**.
So we need to map arbitrary numeric answer strings to a fixed-size option space
for the Dirichlet OKG allocator. This is done via per-question `option_maps`.

Input jsonl (aime_conf_64.jsonl / aime25_conf_64.jsonl) expected fields:
  - id: str/int
  - question: str (optional)
  - answers: list[str|int]   # sampled predictions
  - correct_answer: str|int  # gold label (preferred)
    (fallback: label / answer / final)

"""

from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from plots.offline_curves import (
    aggregate_multi_run_curve_stats,
    plot_accuracy_multi_run_curves,
    plot_multi_run_curves,
)


def _norm_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


# -----------------------------
# Optional math equivalence via math_verify
# -----------------------------
try:
    from math_verify import parse as _mv_parse_raw, verify as _mv_verify  # type: ignore
    from math_verify.parser import ExprExtractionConfig  # type: ignore

    _MATH_VERIFY_AVAILABLE = True
except Exception:  # pragma: no cover
    _MATH_VERIFY_AVAILABLE = False
    _mv_parse_raw = None  # type: ignore
    _mv_verify = None  # type: ignore


SPACING = [r"\\,", r"\\;", r"\\:", r"\\!", r"\\quad", r"\\qquad"]


def normalize_for_math_verify(s: str) -> str:
    """Normalize answer strings before math_verify parsing.

    - Remove common LaTeX spacing commands (\\,, \\;, \\:, \\!, \\quad, \\qquad)
    - If there is '=', keep only the RHS of the last '=' (e.g. 'AB=...' -> '...')

    This is applied at data load time to the answer pool.
    """
    s = _norm_str(s)
    if s in {"\\]", "\\[", "\\)", "\\(", "$$", "$"}:
        return ""

    s = re.sub(r"\\displaystyle\s*", "", s)
    s = re.sub(r"\\textstyle\s*", "", s)
    s = re.sub(r"\\scriptstyle\s*", "", s)
    s = re.sub(r"\\scriptscriptstyle\s*", "", s)

    # \dfrac is display-style fraction; treat it as \frac for equivalence.
    s = re.sub(r"\\dfrac(?![A-Za-z])", r"\\frac", s)

    # Normalize common LaTeX forms that math parsers may be picky about.
    s = re.sub(r"\\sqrt\s*(?!\{)\s*([0-9A-Za-z])", r"\\sqrt{\1}", s)

    for pat in SPACING:
        s = re.sub(pat, "", s)

    s = re.sub(r"^\\\[", "", s).strip()
    s = re.sub(r"\\\]$", "", s).strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()

    if "=" in s:
        s = s.split("=")[-1].strip()

    if re.fullmatch(r"[\\\[\\\]\\\(\\\)\$\s]+", s or ""):
        return ""
    return s


def _latexish_to_expr(s: str) -> str:
    """Best-effort conversion from common LaTeX-ish math to a plain expression."""
    s = _norm_str(s)
    if s == "":
        return ""

    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)

    s = s.replace(r"\cdot", "*")
    s = s.replace(r"\times", "*")
    s = s.replace(r"\pi", "pi")

    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([0-9A-Za-z]+)", r"**\1", s)

    s = s.replace("\\", "")
    return s


@lru_cache(maxsize=200_000)
def _mv_parse_cached(s: str):
    if not _MATH_VERIFY_AVAILABLE:
        return []
    ss = normalize_for_math_verify(s)
    if ss == "":
        return []
    try:
        ss_expr = _latexish_to_expr(ss)
        out_expr = _mv_parse_raw(
            ss_expr,
            extraction_config=[ExprExtractionConfig()],
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            parsing_timeout=2,
            raise_on_error=False,
        )
        return out_expr or []
    except Exception:
        return []


def _math_verify_equal(gold_str: str, pred_str: str) -> bool:
    """Return True iff pred is mathematically equivalent to gold.

    Falls back to normalized string equality if parsing fails or math_verify isn't available.
    Note: `math_verify.verify` is non-symmetric; gold must be passed first.
    """
    g0 = _norm_str(gold_str)
    p0 = _norm_str(pred_str)
    if g0 == "" or p0 == "":
        return g0 == p0
    if g0 == p0:
        return True

    g = normalize_for_math_verify(g0)
    p = normalize_for_math_verify(p0)
    if g == "" or p == "":
        return g == p
    if g == p:
        return True
    if not _MATH_VERIFY_AVAILABLE:
        return False

    g_parsed = _mv_parse_cached(g)
    p_parsed = _mv_parse_cached(p)
    if not g_parsed or not p_parsed:
        return False

    try:
        import sympy  # type: ignore

        g0_obj = g_parsed[0] if len(g_parsed) == 1 else None
        p0_obj = p_parsed[0] if len(p_parsed) == 1 else None
        if isinstance(g0_obj, sympy.Basic) and isinstance(p0_obj, sympy.Basic):
            if g0_obj == p0_obj:
                return True
            try:
                if sympy.simplify(g0_obj - p0_obj) == 0:
                    return True
            except Exception:
                pass
    except Exception:
        pass

    try:
        return bool(
            _mv_verify(
                g_parsed,
                p_parsed,
                strict=True,
                timeout_seconds=5,
                raise_on_error=False,
            )
        )
    except Exception:
        return False


def _find_equivalent_key(ans_str: str, option_map: Dict[str, int]) -> Optional[str]:
    """Find an existing representative key in option_map math-equivalent to ans_str."""
    a = _norm_str(ans_str)
    if a == "":
        return None
    if a in option_map:
        return a
    for rep in option_map.keys():
        # math_verify.verify is non-symmetric; treat as equivalent if either direction matches.
        if _math_verify_equal(rep, a) or _math_verify_equal(a, rep):
            return rep
    return None


# ----------------- Helpers -----------------

def vote_majority(answers: List[str]) -> str:
    """Majority vote with deterministic tie-breaking (earliest wins)."""
    if not answers:
        return ""
    # Count answers after collapsing math-equivalent forms.
    groups: List[Dict[str, object]] = []
    for idx, raw in enumerate(answers):
        ans = _norm_str(raw)
        if ans == "":
            continue

        matched = False
        for g in groups:
            rep = str(g["rep"])
            if _math_verify_equal(rep, ans) or _math_verify_equal(ans, rep):
                g["count"] = int(g["count"]) + 1
                matched = True
                break
        if not matched:
            groups.append({"rep": ans, "first_idx": idx, "count": 1})

    if not groups:
        return ""
    max_cnt = max(int(g["count"]) for g in groups)
    tied = [g for g in groups if int(g["count"]) == max_cnt]
    tied.sort(key=lambda g: int(g["first_idx"]))
    return str(tied[0]["rep"])


def weighted_vote_majority(answers: List[str], weights: List[float]) -> str:
    """Weighted majority vote. Tie-break by earliest occurrence in the (filtered) list."""
    if not answers or not weights:
        return ""
    groups: List[Dict[str, object]] = []
    for idx, (a_raw, w_raw) in enumerate(zip(answers, weights)):
        a = _norm_str(a_raw)
        if a == "":
            continue
        try:
            w = float(w_raw)
        except Exception:
            w = 1.0

        matched = False
        for g in groups:
            rep = str(g["rep"])
            if _math_verify_equal(rep, a) or _math_verify_equal(a, rep):
                g["weight"] = float(g["weight"]) + w
                matched = True
                break
        if not matched:
            groups.append({"rep": a, "first_idx": idx, "weight": float(w)})

    if not groups:
        return ""
    max_weight = max(float(g["weight"]) for g in groups)
    tied = [g for g in groups if float(g["weight"]) == max_weight]
    tied.sort(key=lambda g: int(g["first_idx"]))
    return str(tied[0]["rep"])


def weighted_top_percent_vote_majority(answers: List[str], weights: List[float], frac: float) -> str:
    """Weighted vote over top `frac` samples by weight (e.g. 0.7 keeps top 70%)."""
    if not answers or not weights:
        return ""
    indexed: List[Tuple[int, str, float]] = []
    for idx, (ans, w) in enumerate(zip(answers, weights)):
        try:
            wf = float(w)
        except Exception:
            wf = 1.0
        indexed.append((idx, ans, wf))
    if not indexed:
        return ""
    frac = float(frac)
    frac = max(0.0, min(1.0, frac))
    top_k = max(1, int(len(indexed) * frac))
    top_k = min(top_k, len(indexed))
    top_samples = sorted(indexed, key=lambda x: x[2], reverse=True)[:top_k]
    top_indices = {idx for idx, _, _ in top_samples}
    groups: List[Dict[str, object]] = []
    for idx, ans, conf in top_samples:
        a = _norm_str(ans)
        if a == "":
            continue
        matched = False
        for g in groups:
            rep = str(g["rep"])
            if _math_verify_equal(rep, a) or _math_verify_equal(a, rep):
                g["weight"] = float(g["weight"]) + float(conf)
                matched = True
                break
        if not matched:
            groups.append({"rep": a, "first_idx": int(idx), "weight": float(conf)})

    if not groups:
        return ""
    max_weight = max(float(g["weight"]) for g in groups)
    tied = [g for g in groups if float(g["weight"]) == max_weight]
    tied.sort(key=lambda g: int(g["first_idx"]))
    return str(tied[0]["rep"])


def weighted_top10percent_vote_majority(answers: List[str], weights: List[float]) -> str:
    return weighted_top_percent_vote_majority(answers, weights, 0.1)


def weighted_top30percent_vote_majority(answers: List[str], weights: List[float]) -> str:
    return weighted_top_percent_vote_majority(answers, weights, 0.3)


def weighted_top50percent_vote_majority(answers: List[str], weights: List[float]) -> str:
    return weighted_top_percent_vote_majority(answers, weights, 0.5)


def weighted_top70percent_vote_majority(answers: List[str], weights: List[float]) -> str:
    return weighted_top_percent_vote_majority(answers, weights, 0.7)


def weighted_top90percent_vote_majority(answers: List[str], weights: List[float]) -> str:
    return weighted_top_percent_vote_majority(answers, weights, 0.9)


VARIANT_FUNC_MAP: Dict[str, Callable[[List[str], List[float]], str]] = {
    "weighted": weighted_vote_majority,
    "top10": weighted_top10percent_vote_majority,
    "top30": weighted_top30percent_vote_majority,
    "top50": weighted_top50percent_vote_majority,
    "top70": weighted_top70percent_vote_majority,
    "top90": weighted_top90percent_vote_majority,
}


# Confidence metrics available inside trace_confidence entries
CONF_METRIC_KEYS = {
    "Conf": "mean_confidence",
    "mean": "mean_confidence",  # alias
    "tail": "tail_2048_mean_conf",
    "bottom": "bottom_0.1_sliding_2048_mean_conf",
}

DEFAULT_CONF_VALS = {
    "mean_confidence": 1.0,
    "tail_2048_mean_conf": 1.0,
    "bottom_0.1_sliding_2048_mean_conf": 1.0,
}


def _default_conf_entry() -> dict:
    return dict(DEFAULT_CONF_VALS)


def _normalize_conf_entry(raw: object) -> dict:
    """Normalize a raw trace_confidence entry to gpqa-style dict keys."""
    if not isinstance(raw, dict):
        # allow numeric formats; treat as mean_confidence when possible
        out = _default_conf_entry()
        try:
            out["mean_confidence"] = float(raw)  # type: ignore[arg-type]
        except Exception:
            pass
        return out

    entry = dict(raw)
    # Support nested schema: {"conf_summary": {...}} (e.g. server_stats)
    conf_src = entry.get("conf_summary")
    if isinstance(conf_src, dict):
        entry = {**entry, **conf_src}

    out = _default_conf_entry()
    try:
        out["mean_confidence"] = float(entry.get("mean_confidence", 1.0))
    except Exception:
        out["mean_confidence"] = 1.0
    try:
        out["tail_2048_mean_conf"] = float(entry.get("tail_2048_mean_conf", entry.get("tail_mean_conf", 1.0)))
    except Exception:
        out["tail_2048_mean_conf"] = 1.0
    try:
        out["bottom_0.1_sliding_2048_mean_conf"] = float(
            entry.get("bottom_0.1_sliding_2048_mean_conf", entry.get("bottom_mean_conf", 1.0))
        )
    except Exception:
        out["bottom_0.1_sliding_2048_mean_conf"] = 1.0
    return out


DEFAULT_OKG_METHODS = [
    "Conf",
    "tail_top90",
    "bottom_top90",
]

DEFAULT_BASELINE_METHODS = ["mv"] + DEFAULT_OKG_METHODS


def _get_metric_weights(answers: List[str], conf_dicts: Optional[List[dict]], metric: str) -> List[float]:
    """Return weights aligned with answers for the given metric."""
    n = len(answers)
    if n == 0:
        return []
    key = CONF_METRIC_KEYS.get(metric, "mean_confidence")
    if not conf_dicts:
        return [1.0] * n
    out: List[float] = []
    for i in range(n):
        entry = conf_dicts[i] if i < len(conf_dicts) else None
        if not isinstance(entry, dict):
            try:
                out.append(float(entry))
            except Exception:
                out.append(1.0)
            continue
        value = entry.get(key)
        if value is None:
            value = entry.get("mean_confidence", 1.0)
        try:
            out.append(float(value))
        except Exception:
            out.append(1.0)
    return out


def build_method_functions(method_names: Sequence[str]) -> List[Tuple[str, Callable]]:
    """Convert method name strings into (name, callable) pairs.

    Supported naming patterns (same idea as `mmlu_offline.py`):
      - "mv": plain majority voting
      - Variant names: "weighted", "top70" (assume metric=mean)
    - Metric-prefixed names: "Conf", "Conf_top90", "tail_top70", "tail_top90", "bottom_top10"
    """
    result: List[Tuple[str, Callable]] = []
    seen = set()
    for raw_name in method_names:
        raw_name = str(raw_name).strip()
        if not raw_name or raw_name in seen:
            continue
        if raw_name == "mv":
            result.append((raw_name, lambda answers, confs=None: vote_majority(answers)))
            seen.add(raw_name)
            continue

        # Variant-only name (assume mean_confidence)
        if raw_name in VARIANT_FUNC_MAP:
            base_func = VARIANT_FUNC_MAP[raw_name]

            def make_wrapper(func=base_func):
                return lambda answers, confs: func(answers, _get_metric_weights(answers, confs, "mean"))

            result.append((raw_name, make_wrapper()))
            seen.add(raw_name)
            continue

        if "_" in raw_name:
            metric, variant = raw_name.split("_", 1)
        else:
            metric, variant = raw_name, "weighted"

        if metric not in CONF_METRIC_KEYS:
            continue

        variant_key = "weighted" if variant in ("weighted", "mean") else variant
        base_variant = VARIANT_FUNC_MAP.get(variant_key)
        if base_variant is None:
            continue

        def make_metric_wrapper(func=base_variant, metric_name=metric):
            return lambda answers, confs: func(answers, _get_metric_weights(answers, confs, metric_name))

        result.append((raw_name, make_metric_wrapper()))
        seen.add(raw_name)
    return result


LabelsType = Union[Dict[str, str], Dict[str, Dict[str, str]]]


def _default_labels(labels: LabelsType) -> Dict[str, str]:
    if not labels:
        return {}
    try:
        if all(isinstance(v, dict) for v in labels.values()):
            default = labels.get("__default__")  # type: ignore[assignment]
            if isinstance(default, dict):
                return default
            return next(iter(labels.values()))  # type: ignore[return-value]
    except Exception:
        pass
    return labels  # type: ignore[return-value]


def _label_for_method(labels: LabelsType, *, method_name: str, qid: str) -> str:
    base = _default_labels(labels)
    if not base:
        return ""
    try:
        if labels and all(isinstance(v, dict) for v in labels.values()):
            m = labels.get(method_name)  # type: ignore[assignment]
            if isinstance(m, dict):
                v = m.get(qid)
                if v is not None:
                    return str(v)
    except Exception:
        pass
    return str(base.get(qid, ""))


def _evaluate_methods_on_qs(
    qids: List[str],
    labels: LabelsType,
    answers_map: Dict[str, List[str]],
    confs_map: Optional[Dict[str, List[dict]]],
    methods: List[Tuple[str, Callable]],
) -> Dict[str, float]:
    base_labels = _default_labels(labels)
    valid_qids = [q for q in qids if q in base_labels]
    if not valid_qids:
        return {name: 0.0 for name, _ in methods}
    correct_counts = {name: 0 for name, _ in methods}
    total = 0
    for q in valid_qids:
        answers = answers_map.get(q, [])
        confs = (confs_map.get(q, []) if confs_map is not None else [])
        for name, func in methods:
            try:
                try:
                    pred = func(answers, confs)
                except TypeError:
                    pred = func(answers)
            except Exception:
                pred = ""

            gold = _label_for_method(labels, method_name=name, qid=q)
            if _math_verify_equal(gold, pred) or _math_verify_equal(pred, gold):
                correct_counts[name] += 1
        total += 1
    return {name: (correct_counts[name] / total) for name, _ in methods}


def compute_baseline_curve(
    qids: List[str],
    pools: Dict[str, List[str]],
    labels: LabelsType,
    budgets: List[int],
    warm_up: int,
    warmup_answers: Dict[str, List[str]],
    warmup_indices: Dict[str, List[int]],
    confs_pools: Optional[Dict[str, List[dict]]],
    methods: List[Tuple[str, Callable]],
) -> Dict[str, Dict[int, float]]:
    """Compute uniform baseline curves on the same budgets as OKG.

    This mirrors `mmlu_offline.py:compute_baseline_curve`:
    - Use warm-up samples first (if provided), then fill remaining needs from the pool
      while skipping already-used warm-up indices.
    """
    K = len(qids)
    base_labels = _default_labels(labels)
    if K == 0 or not base_labels:
        return {}
    valid_qids = [q for q in qids if q in base_labels]
    if not valid_qids:
        return {}

    warm_up = max(0, int(warm_up))
    curves: Dict[str, Dict[int, float]] = {name: {} for name, _ in methods}

    for budget in budgets:
        try:
            budget = int(budget)
        except Exception:
            continue
        b = budget // K
        if b <= 0:
            continue

        selected_map: Dict[str, List[str]] = {}
        selected_confs_map: Dict[str, List[dict]] = {} if confs_pools is not None else {}

        for q in valid_qids:
            pool_answers = pools.get(q, [])
            if not pool_answers:
                continue

            warmup_obs = warmup_answers.get(q, [])
            warmup_take = min(warm_up, len(warmup_obs), b)
            answers: List[str] = list(warmup_obs[:warmup_take]) if warmup_take > 0 else []
            confs: List[dict] = []

            if confs_pools is not None and warmup_take > 0:
                used = warmup_indices.get(q, [])[:warmup_take]
                cpool = confs_pools.get(q, [])
                for idx in used:
                    entry = cpool[idx] if idx < len(cpool) else None
                    confs.append(_normalize_conf_entry(entry))

            remaining_need = max(0, b - warmup_take)
            if remaining_need > 0:
                used = warmup_indices.get(q, [])[:warmup_take]
                used_set = set(used)
                added = 0
                for idx, ans in enumerate(pool_answers):
                    if idx in used_set:
                        continue
                    answers.append(ans)
                    if confs_pools is not None:
                        cpool = confs_pools.get(q, [])
                        entry = cpool[idx] if idx < len(cpool) else None
                        confs.append(_normalize_conf_entry(entry))
                    added += 1
                    if added >= remaining_need:
                        break

            if not answers:
                continue
            selected_map[q] = answers
            if confs_pools is not None:
                selected_confs_map[q] = confs

        selected_confs_arg = selected_confs_map if confs_pools is not None else None
        accs = _evaluate_methods_on_qs(valid_qids, labels, selected_map, selected_confs_arg, methods)
        for name in accs:
            curves[name][budget] = float(accs[name])

    return curves


def maybe_record_curve(
    current_budget: int,
    curve_records: List[Tuple[int, Dict[str, float]]],
    ordered_qids: List[str],
    labels: LabelsType,
    collected_answers: Dict[str, List[str]],
    collected_confs: Optional[Dict[str, List[dict]]],
    record_budgets: Optional[set],
    force: bool = False,
    method_names: Optional[Sequence[str]] = None,
) -> None:
    """Record OKG accuracy (majority + extra variants) at current_budget."""
    if not _default_labels(labels):
        return
    if not force and record_budgets is not None and current_budget not in record_budgets:
        return

    names = list(dict.fromkeys(method_names or DEFAULT_OKG_METHODS))
    names = [n for n in names if n and n != "mv"]
    methods: List[Tuple[str, Callable]] = [("OKG", lambda a, c=None: vote_majority(a))]
    methods.extend(build_method_functions(names))

    acc_dict = _evaluate_methods_on_qs(ordered_qids, labels, collected_answers, collected_confs, methods)
    if curve_records and curve_records[-1][0] == current_budget:
        curve_records[-1] = (current_budget, acc_dict)
    else:
        curve_records.append((current_budget, acc_dict))


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
) -> None:
    """Export multi-run curves + requested stats into a JSONL file.

    The file contains per-run entries plus a final "summary" entry.
    By default we focus on 4 methods: Base, Base_Conf, OKG, OKG_Conf.
    """
    wanted = list(methods or ["Base", "Base_Conf", "OKG", "OKG_Conf"])

    cons_by_run: Dict[int, Dict[str, List[Tuple[int, float]]]] = {i: d for i, d in (curve_runs_consistency or [])}
    acc_by_run: Dict[int, Dict[str, List[Tuple[int, float]]]] = {i: d for i, d in (curve_runs_accuracy or [])}
    run_indices = sorted(set(cons_by_run.keys()) | set(acc_by_run.keys()))

    def _find_key(curves: Dict[str, List[Tuple[int, float]]], canonical: str) -> Optional[str]:
        if not curves:
            return None
        if canonical in curves:
            return canonical
        low = canonical.lower()
        for k in curves.keys():
            if str(k).lower() == low:
                return k
        if low == "base":
            for k in curves.keys():
                if str(k).lower() in ("mv", "base"):
                    return k
        if low in ("base_conf", "okg_conf"):
            keys = [str(k) for k in curves.keys()]

            def _base_candidates() -> List[str]:
                return sorted([k for k in keys if k.startswith("Base_") and k != "Base"])

            def _okg_candidates() -> List[str]:
                return sorted([k for k in keys if k.startswith("OKG_") and k != "OKG"])

            if low == "base_conf":
                cand = _base_candidates()
                if not cand:
                    return None
                if "Base_Conf" in cand:
                    return "Base_Conf"
                if len(cand) == 1:
                    return cand[0]
                for k in cand:
                    if "conf" in k.lower():
                        return k
                return cand[0]

            if low == "okg_conf":
                okg_cand = _okg_candidates()
                if not okg_cand:
                    return None
                if "OKG_Conf" in okg_cand:
                    return "OKG_Conf"
                base_cand = _base_candidates()
                if base_cand:
                    base_pick = "Base_Conf" if "Base_Conf" in base_cand else base_cand[0]
                    suffix = base_pick[len("Base_") :]
                    aligned = f"OKG_{suffix}"
                    if aligned in okg_cand:
                        return aligned
                if len(okg_cand) == 1:
                    return okg_cand[0]
                for k in okg_cand:
                    if "conf" in k.lower():
                        return k
                return okg_cand[0]
        return None

    budgets_at_cons1: Dict[str, List[Optional[int]]] = {m: [] for m in wanted}
    at_okg_budget_acc: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}
    at_okg_budget_cons: Dict[str, List[Optional[float]]] = {m: [] for m in wanted}

    out_path_obj = Path(output_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_path_obj.open("w", encoding="utf-8") as f:
        for run_idx in run_indices:
            cons_curves = cons_by_run.get(run_idx, {}) or {}
            acc_curves = acc_by_run.get(run_idx, {}) or {}

            peak_budget_for: Dict[str, Optional[int]] = {}
            for m in wanted:
                k = _find_key(cons_curves, m)
                bpk = _first_budget_at_max(cons_curves.get(k, []) if k else [])
                peak_budget_for[m] = bpk
                budgets_at_cons1[m].append(bpk)

            okg_budget = peak_budget_for.get("OKG")

            at_budget_for_run: Dict[str, Dict[str, Optional[float]]] = {}
            for m in wanted:
                cons_key = _find_key(cons_curves, m)
                acc_key = _find_key(acc_curves, m)
                cons_map = _points_to_budget_map(cons_curves.get(cons_key, []) if cons_key else [])
                acc_map = _points_to_budget_map(acc_curves.get(acc_key, []) if acc_key else [])

                c_val = cons_map.get(int(okg_budget)) if okg_budget is not None else None
                a_val = acc_map.get(int(okg_budget)) if okg_budget is not None else None
                at_budget_for_run[m] = {"consistency": c_val, "accuracy": a_val}
                at_okg_budget_cons[m].append(c_val)
                at_okg_budget_acc[m].append(a_val)

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
                "OKG_budget_at_cons1": okg_budget,
                "base_budget_at_cons1": peak_budget_for.get("Base"),
                "base_conf_budget_at_cons1": peak_budget_for.get("Base_Conf"),
                "OKG_conf_budget_at_cons1": peak_budget_for.get("OKG_Conf"),
                "base_consistency_at_con1": at_budget_for_run.get("Base", {}).get("consistency"),
                "base_accuracy_at_con1": at_budget_for_run.get("Base", {}).get("accuracy"),
                "base_conf_consistency_at_con1": at_budget_for_run.get("Base_Conf", {}).get("consistency"),
                "base_conf_accuracy_at_con1": at_budget_for_run.get("Base_Conf", {}).get("accuracy"),
                "OKG_consistency_at_con1": at_budget_for_run.get("OKG", {}).get("consistency"),
                "OKG_accuracy_at_con1": at_budget_for_run.get("OKG", {}).get("accuracy"),
                "OKG_conf_consistency_at_con1": at_budget_for_run.get("OKG_Conf", {}).get("consistency"),
                "OKG_conf_accuracy_at_con1": at_budget_for_run.get("OKG_Conf", {}).get("accuracy"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary: Dict[str, object] = {"type": "summary", "num_runs": int(len(run_indices))}
        budget_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        for m in wanted:
            mean, std, n = _scalar_mean_std(budgets_at_cons1[m])
            budget_stats[m] = {"mean": mean, "std": std, "num_runs": n}
        summary["budget_at_cons1_stats"] = budget_stats

        metric_stats: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {}
        for m in wanted:
            a_mean, a_std, a_n = _scalar_mean_std(at_okg_budget_acc[m])
            c_mean, c_std, c_n = _scalar_mean_std(at_okg_budget_cons[m])
            metric_stats[m] = {
                "accuracy_at_con1": {"mean": a_mean, "std": a_std, "num_runs": a_n},
                "consistency_at_con1": {"mean": c_mean, "std": c_std, "num_runs": c_n},
            }
        summary["metrics_at_okg_cons1_budget_stats"] = metric_stats

        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"[multi-run] curves+stats jsonl saved: {out_path_obj}")



# ----------------- OKG Allocator -----------------

class OKGAllocator:
    """Standard OKG allocator operating on Dirichlet counts."""

    def __init__(self, M: int, nsamples: int = 500, seed: Optional[int] = None) -> None:
        self.M = int(M)
        self.nsamples = int(nsamples)
        self.rng = np.random.default_rng(seed)

    def _estimate_I_vector(self, alpha_row: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha_row, dtype=float)
        if alpha.ndim != 1 or alpha.size != self.M:
            raise ValueError("alpha must be 1-D with length M")
        gamma_samples = self.rng.gamma(shape=alpha, scale=1.0, size=(self.nsamples, self.M))
        argmax = np.argmax(gamma_samples, axis=1)
        counts = np.bincount(argmax, minlength=self.M)
        return counts / self.nsamples

    @staticmethod
    def _compute_h(I_vector: np.ndarray) -> float:
        return float(np.max(I_vector))

    def select_next(self, alpha_matrix: np.ndarray, c: float = 1.0) -> int:
        if alpha_matrix.ndim != 2 or alpha_matrix.shape[1] != self.M:
            raise ValueError("alpha_matrix must be shape (K_available, M)")
        k_available = alpha_matrix.shape[0]
        rewards = np.empty(k_available, dtype=float)
        for i in range(k_available):
            alpha_i = alpha_matrix[i]
            base = self._estimate_I_vector(alpha_i)
            h_base = self._compute_h(base)
            deltas = np.empty(self.M, dtype=float)
            for m in range(self.M):
                alpha_plus = alpha_i.copy()
                alpha_plus[m] += c
                plus = self._estimate_I_vector(alpha_plus)
                deltas[m] = self._compute_h(plus) - h_base
            rewards[i] = np.max(deltas)
        return int(np.argmax(rewards))

    @staticmethod
    def update(alpha_matrix: np.ndarray, q_idx: int, option_idx: int, c: float = 1.0) -> None:
        alpha_matrix[q_idx, option_idx] += c


# ----------------- Loading -----------------

def load_aime_conf(
    path: str,
    *,
    include_fallbacks_for_pseudo: bool = True,
) -> Tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, List[dict]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
]:
    """Load AIME-style numeric pools and labels from a jsonl file.

    Returns:
      - order: qids in file order
      - pools: qid -> list of answer strings
      - confs: qid -> list of per-sample confidence dicts (aligned with answers; may be empty)
      - pseudo_labels: qid -> pseudo label string (for self-consistency; prefer "final")
      - true_labels: qid -> true gold label string (for accuracy; prefer "correct_answer")
      - questions: qid -> question text (optional)
    """
    order: List[str] = []
    pools: Dict[str, List[str]] = {}
    confs: Dict[str, List[dict]] = {}
    pseudo_labels: Dict[str, str] = {}
    true_labels: Dict[str, str] = {}
    questions: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = _norm_str(obj.get("id"))
            if not qid:
                qid = str(len(order))
            order.append(qid)

            questions[qid] = _norm_str(obj.get("question", ""))

            # answers pool (normalize formatting for math equivalence) + keep confs aligned
            raw_answers = obj.get("answers", []) or []
            raw_confs = obj.get("trace_confidence", []) or []
            raw_confs_list = raw_confs if isinstance(raw_confs, list) else []

            answers_norm: List[str] = []
            confs_norm: List[dict] = []
            for i, a in enumerate(raw_answers):
                aa = normalize_for_math_verify(a)
                if aa == "":
                    continue
                answers_norm.append(aa)
                entry = raw_confs_list[i] if i < len(raw_confs_list) else None
                confs_norm.append(_normalize_conf_entry(entry))

            pools[qid] = answers_norm
            confs[qid] = confs_norm  # aligned 1:1 with pools[qid]

            # pseudo label (self-consistency): prefer "final", optionally fall back
            pseudo = obj.get("final")
            if pseudo is None and include_fallbacks_for_pseudo:
                pseudo = obj.get("correct_answer")
            if pseudo is None and include_fallbacks_for_pseudo:
                pseudo = obj.get("label")
            if pseudo is None and include_fallbacks_for_pseudo:
                pseudo = obj.get("answer")
            pseudo = _norm_str(pseudo)
            if pseudo != "":
                pseudo_labels[qid] = pseudo

            # true label (accuracy): prefer dataset gold
            gold = obj.get("correct_answer")
            if gold is None:
                gold = obj.get("answer")
            if gold is None:
                gold = obj.get("label")
            gold = _norm_str(gold)
            if gold != "":
                true_labels[qid] = gold

    return order, pools, confs, pseudo_labels, true_labels, questions


# ----------------- OKG (AIME mapping) -----------------
# NOTE: We intentionally do NOT "sweep" B by re-running multiple times.
# We run ONCE to total budget T=K*B, and record curve points at budgets K*b (b=1..B),
# matching gpqa_offline.py's single-trajectory behavior.
def run(
    preds_path: str,
    B: int = 32,
    max_options: int = 32,
    nsamples: int = 500,
    seed: int = 2025,
    out_path: Optional[str] = None,
    with_baseline: bool = False,
    baseline_methods: Optional[Sequence[str]] = None,
    warm_up: int = 1,
    smoke_n: Optional[int] = None,
    subsample_pool_size: Optional[int] = None,
    relabel_from_subsample_mv: bool = False,
    return_curve_data: bool = False,
    save_outputs: bool = True,
) -> Optional[dict]:
    # rng_pool: dedicated RNG for per-run answer-pool shuffling/subsampling so toggling
    # subsampling doesn't inadvertently change OKG randomness.
    rng_pool = np.random.default_rng(int(seed or 0) + 1000003)

    # Load inputs: answer pools, confidence traces, labels, and optional question text.
    order, pools, confs_pools, labels_consistency, labels_accuracy, questions = load_aime_conf(preds_path)
    if not order:
        raise RuntimeError("No questions found in preds file")

    K = len(order)
    labeled_cons = sum(1 for qid in order if qid in labels_consistency)
    labeled_acc = sum(1 for qid in order if qid in labels_accuracy)
    print(
        f"[load] preds={preds_path} K={K} labeled_consistency={labeled_cons} labeled_accuracy={labeled_acc} max_options={max_options}"
    )
    if not labels_consistency:
        print('[warning] No pseudo labels found; consistency curve will be empty (need "final" or fallbacks)')
    if not labels_accuracy:
        print('[warning] No gold labels found; accuracy curve will be empty (need "correct_answer"/"answer")')

    B = max(1, int(B))
    max_options = max(1, int(max_options))
    warm_up = max(0, int(warm_up))
    T = K * B

    # Smoke-test mode: optionally restrict to first N questions to run faster
    if smoke_n is not None and int(smoke_n) > 0 and int(smoke_n) < K:
        smoke_n = int(smoke_n)
        print(f"[smoke] restricting to first {smoke_n} questions for a quick run")
        keep = set(order[:smoke_n])
        order = [q for q in order if q in keep]
        pools = {q: pools.get(q, []) for q in order}
        confs_pools = {q: confs_pools.get(q, []) for q in order}
        labels_consistency = {q: labels_consistency[q] for q in order if q in labels_consistency}
        labels_accuracy = {q: labels_accuracy[q] for q in order if q in labels_accuracy}
        questions = {q: questions.get(q, "") for q in order}
        K = len(order)
        T = K * B

    # multi-run helper: per-run shuffle the full pool, take the first K as the pool,
    # and (optionally) treat MV(pool[:K]) as this run's gold label (equivalent to "final").
    if subsample_pool_size is not None and int(subsample_pool_size) > 0:
        pool_k = int(subsample_pool_size)
        for qid in order:
            pool = list(pools.get(qid, []) or [])
            if not pool:
                continue
            n = len(pool)
            k_eff = min(pool_k, n)
            idx = np.arange(n, dtype=int)
            rng_pool.shuffle(idx)
            idx = idx[:k_eff]
            pools[qid] = [pool[i] for i in idx]
            if confs_pools is not None:
                cpool = list(confs_pools.get(qid, []) or [])
                if cpool:
                    confs_pools[qid] = [cpool[i] if i < len(cpool) else {} for i in idx]
            # Optional: MV relabeling ONLY affects pseudo-labels (consistency),
            # never overrides dataset gold labels (accuracy).
            if relabel_from_subsample_mv:
                labels_consistency[qid] = vote_majority(pools[qid])

    # Parse methods (baseline weighted variants) and derive OKG variants from them.
    if baseline_methods:
        requested_base = [s.strip() for s in baseline_methods if str(s).strip()]
    else:
        requested_base = list(DEFAULT_BASELINE_METHODS)
    # Ensure mv baseline included when with_baseline
    if with_baseline and "mv" not in requested_base:
        requested_base = ["mv"] + requested_base

    baseline_method_names = requested_base
    # OKG uses the same variant list as baseline (excluding mv).
    okg_method_names = [m for m in baseline_method_names if m and m != "mv"]

    # Consistency labels: for confidence-weighted methods, use a run-specific pseudo label
    # computed from the full (current-run) pool answers + confidence weights.
    labels_consistency_by_method: Dict[str, Dict[str, str]] = {
        "__default__": dict(labels_consistency),
        "OKG": dict(labels_consistency),
        "mv": dict(labels_consistency),
    }
    weighted_method_names: List[str] = [
        m
        for m in list(dict.fromkeys((baseline_method_names + okg_method_names)))
        if m and m not in ("mv", "OKG")
    ]
    method_funcs = {name: fn for name, fn in build_method_functions(weighted_method_names)}
    if weighted_method_names and method_funcs and labels_consistency:
        for m in weighted_method_names:
            fn = method_funcs.get(m)
            if fn is None:
                continue
            per_method: Dict[str, str] = {}
            for qid in order:
                if qid not in labels_consistency:
                    continue
                pool_answers = pools.get(qid, []) or []
                pool_confs = confs_pools.get(qid, []) if confs_pools is not None else []
                try:
                    pseudo_m = fn(pool_answers, pool_confs)
                except Exception:
                    pseudo_m = ""
                per_method[qid] = _norm_str(pseudo_m) if _norm_str(pseudo_m) != "" else labels_consistency.get(qid, "")
            labels_consistency_by_method[m] = per_method

    # Single-run OKG allocation: record at budgets K*b (b=1..B).
    record_budgets = {K * b for b in range(1, B + 1)}
    curve_records_consistency: List[Tuple[int, Dict[str, float]]] = []
    curve_records_accuracy: List[Tuple[int, Dict[str, float]]] = []

    # Allocator state and per-question option buckets.
    allocator = OKGAllocator(M=max_options, nsamples=nsamples, seed=int(seed))
    alphas = np.ones((K, max_options), dtype=float)
    pointers: Dict[str, int] = {qid: 0 for qid in order}
    option_maps: Dict[str, Dict[str, int]] = {qid: {} for qid in order}
    collected_answers: Dict[str, List[str]] = {qid: [] for qid in order}
    collected_confs: Dict[str, List[dict]] = {qid: [] for qid in order}

    warmup_answers: Dict[str, List[str]] = {qid: [] for qid in order}
    warmup_indices: Dict[str, List[int]] = {qid: [] for qid in order}

    def _consume_one(global_idx: int) -> str:
        """Consume the next sample for question `global_idx`.

        Returns:
          - "accepted": mapped into option buckets and used to update alpha + voting
          - "discarded": unseen answer but option buckets are full; sample is ignored (still costs budget)
          - "exhausted": no more samples available for this question
        """
        qid = order[global_idx]
        # Allow consuming beyond max_options; max_options only caps option buckets, not per-question samples.
        limit = len(pools.get(qid, []))
        if pointers[qid] >= limit:
            return "exhausted"
        ans_pos = pointers[qid]
        pointers[qid] += 1
        ans_str = pools[qid][ans_pos]

        # map answer to option bucket (math-equivalence aware)
        mapping = option_maps[qid]
        rep_key = _find_equivalent_key(ans_str, mapping)
        if rep_key is None:
            if len(mapping) >= max_options:
                return "discarded"
            mapping[ans_str] = len(mapping)
            rep_key = ans_str

        opt_idx = mapping[rep_key]
        if opt_idx >= max_options:
            opt_idx = max_options - 1

        # accepted: record answer/conf and update alpha
        # Store representative key so voting collapses equivalent formats.
        collected_answers[qid].append(rep_key)
        if confs_pools is not None:
            conf_list = confs_pools.get(qid, [])
            raw_entry = conf_list[ans_pos] if ans_pos < len(conf_list) else None
            collected_confs[qid].append(_normalize_conf_entry(raw_entry))
        OKGAllocator.update(alphas, global_idx, opt_idx, c=1.0)
        return "accepted"

    def _peek_next_sample(qid: str) -> Tuple[Optional[int], Optional[str]]:
        """Peek next (pos, answer_str) without consuming."""
        try:
            pos = int(pointers.get(qid, 0))
        except Exception:
            pos = 0
        pool = pools.get(qid, []) or []
        if 0 <= pos < len(pool):
            return pos, pool[pos]
        return None, None

    total_budget = 0
    # Warm-up: uniform per question.
    if warm_up > 0 and total_budget < T:
        print(f"[warmup] allocating up to {warm_up} answers per question before OKG")
        for idx, qid in enumerate(order):
            if total_budget >= T:
                break
            limit = len(pools.get(qid, []))
            num_samples = min(warm_up, limit, T - total_budget)
            for _ in range(num_samples):
                ans_pos_peek, ans_str_peek = _peek_next_sample(qid)
                status = _consume_one(idx)
                if status == "exhausted":
                    break
                total_budget += 1  # budget step consumed even if discarded
                if status == "accepted":
                    opt_map = option_maps.get(qid, {})
                    rep = _find_equivalent_key(str(ans_str_peek), opt_map) or str(ans_str_peek)
                    opt_idx = opt_map.get(rep, None)
                    print(
                        f"[t={total_budget}] selected idx={idx} qid={qid} sampled_ans={ans_str_peek} (ans_pos={ans_pos_peek}, opt_idx={opt_idx})",
                        flush=True,
                    )
                elif status == "discarded":
                    print(
                        f"[t={total_budget}] selected idx={idx} qid={qid} discarded_unseen_ans={ans_str_peek} (ans_pos={ans_pos_peek})",
                        flush=True,
                    )
                if status == "accepted":
                    warmup_answers[qid].append(collected_answers[qid][-1])
                    warmup_indices[qid].append(pointers[qid] - 1)
                maybe_record_curve(
                    total_budget,
                    curve_records_consistency,
                    order,
                    labels_consistency_by_method,
                    collected_answers,
                    collected_confs if confs_pools is not None else None,
                    record_budgets,
                    force=False,
                    method_names=okg_method_names,
                )
                maybe_record_curve(
                    total_budget,
                    curve_records_accuracy,
                    order,
                    labels_accuracy,
                    collected_answers,
                    collected_confs if confs_pools is not None else None,
                    record_budgets,
                    force=False,
                    method_names=okg_method_names,
                )
        # ensure we record warmup end
        maybe_record_curve(
            total_budget,
            curve_records_consistency,
            order,
            labels_consistency_by_method,
            collected_answers,
            collected_confs if confs_pools is not None else None,
            record_budgets,
            force=True,
            method_names=okg_method_names,
        )
        maybe_record_curve(
            total_budget,
            curve_records_accuracy,
            order,
            labels_accuracy,
            collected_answers,
            collected_confs if confs_pools is not None else None,
            record_budgets,
            force=True,
            method_names=okg_method_names,
        )
    else:
        # allow recording at 0 when warmup disabled (mainly for debugging)
        maybe_record_curve(
            0,
            curve_records_consistency,
            order,
            labels_consistency_by_method,
            collected_answers,
            collected_confs if confs_pools is not None else None,
            record_budgets,
            force=True,
            method_names=okg_method_names,
        )
        maybe_record_curve(
            0,
            curve_records_accuracy,
            order,
            labels_accuracy,
            collected_answers,
            collected_confs if confs_pools is not None else None,
            record_budgets,
            force=True,
            method_names=okg_method_names,
        )

    warmup_budget_end = total_budget
    warmup_answers_snapshot = {qid: list(warmup_answers[qid]) for qid in order}
    warmup_indices_snapshot = {qid: list(warmup_indices[qid]) for qid in order}

    # OKG allocation loop (single run).
    while total_budget < T:
        available_indices: List[int] = []
        for idx, qid in enumerate(order):
            limit = len(pools.get(qid, []))
            if pointers[qid] < limit:
                available_indices.append(idx)
        if not available_indices:
            break
        chosen_local = allocator.select_next(alphas[available_indices], c=1.0)
        global_idx = available_indices[int(chosen_local)]
        qid = order[global_idx]
        ans_pos_peek, ans_str_peek = _peek_next_sample(qid)
        status = _consume_one(global_idx)
        if status == "exhausted":
            continue
        total_budget += 1  # budget step consumed even if discarded
        if status == "accepted":
            opt_map = option_maps.get(qid, {})
            rep = _find_equivalent_key(str(ans_str_peek), opt_map) or str(ans_str_peek)
            opt_idx = opt_map.get(rep, None)
            print(
                f"[t={total_budget}] selected idx={global_idx} qid={qid} sampled_ans={ans_str_peek} (ans_pos={ans_pos_peek}, opt_idx={opt_idx})",
                flush=True,
            )
        elif status == "discarded":
            print(
                f"[t={total_budget}] selected idx={global_idx} qid={qid} discarded_unseen_ans={ans_str_peek} (ans_pos={ans_pos_peek})",
                flush=True,
            )
        maybe_record_curve(
            total_budget,
            curve_records_consistency,
            order,
            labels_consistency_by_method,
            collected_answers,
            collected_confs if confs_pools is not None else None,
            record_budgets,
            force=False,
            method_names=okg_method_names,
        )
        maybe_record_curve(
            total_budget,
            curve_records_accuracy,
            order,
            labels_accuracy,
            collected_answers,
            collected_confs if confs_pools is not None else None,
            record_budgets,
            force=False,
            method_names=okg_method_names,
        )

    # Build curve payloads for both metrics (start after warm-up ends).
    start_budget = warmup_budget_end
    curve_consistency: List[Tuple[int, Dict[str, float]]] = [
        entry for entry in curve_records_consistency if entry[0] >= start_budget
    ]
    curve_accuracy: List[Tuple[int, Dict[str, float]]] = [entry for entry in curve_records_accuracy if entry[0] >= start_budget]

    if curve_consistency or curve_accuracy:
        budgets_print = sorted({t for t, _ in (curve_consistency or [])} | {t for t, _ in (curve_accuracy or [])})
        cons_map = {t: d for t, d in curve_consistency}
        acc_map = {t: d for t, d in curve_accuracy}
        for t_val in budgets_print:
            b_val = t_val // K if K else 0
            c_okg = float(cons_map.get(t_val, {}).get("OKG", 0.0))
            a_okg = float(acc_map.get(t_val, {}).get("OKG", 0.0))
            print(f"[OKG-AIME] b={b_val} budget={t_val} cons={c_okg:.4f} acc={a_okg:.4f}")

    def _build_curves_dict_for_metric(
        curve: List[Tuple[int, Dict[str, float]]],
        metric_labels: LabelsType,
        *,
        with_baseline_flag: bool,
    ) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[int, float], Dict[str, Dict[int, float]], List[str]]:
        mv_curve_local: Dict[int, float] = {}
        weighted_baselines_local: Dict[str, Dict[int, float]] = {}
        selected_okg_variants_local: List[str] = [n for n in okg_method_names]

        if with_baseline_flag and metric_labels and curve:
            eval_budgets = [t for t, _ in curve]
            base_method_funcs = build_method_functions(baseline_method_names)
            if not base_method_funcs:
                base_method_funcs = [("mv", lambda answers, confs=None: vote_majority(answers))]
            baseline_results = compute_baseline_curve(
                qids=order,
                pools=pools,
                labels=metric_labels,
                budgets=eval_budgets,
                warm_up=warm_up,
                warmup_answers=warmup_answers_snapshot,
                warmup_indices=warmup_indices_snapshot,
                confs_pools=confs_pools,
                methods=base_method_funcs,
            )
            mv_curve_local = baseline_results.get("mv", {})
            weighted_baselines_local = {name: table for name, table in baseline_results.items() if name != "mv"}

        curve_variant_order: List[str] = []
        for _, accd in curve:
            for key in accd.keys():
                if key == "OKG" or not key:
                    continue
                if key not in curve_variant_order:
                    curve_variant_order.append(key)
        if selected_okg_variants_local:
            selected_okg_variants_local = [n for n in selected_okg_variants_local if n in curve_variant_order]
            for n in curve_variant_order:
                if n not in selected_okg_variants_local:
                    selected_okg_variants_local.append(n)
        else:
            selected_okg_variants_local = curve_variant_order

        curves_dict_local: Dict[str, List[Tuple[int, float]]] = {}
        if curve:
            curves_dict_local["OKG"] = [(t, accd.get("OKG", 0.0)) for t, accd in curve]
            if with_baseline_flag and mv_curve_local:
                curves_dict_local["Base"] = [(t, mv_curve_local.get(t, 0.0)) for t, _ in curve]
            for name, table in (weighted_baselines_local or {}).items():
                if name == "mv":
                    continue
                curves_dict_local[f"Base_{name}"] = [(t, table.get(t, 0.0)) for t, _ in curve]
            for name in selected_okg_variants_local:
                curves_dict_local[f"OKG_{name}"] = [(t, accd.get(name, 0.0)) for t, accd in curve]

        return curves_dict_local, mv_curve_local, weighted_baselines_local, selected_okg_variants_local

    curves_dict_consistency, mv_curve_consistency, weighted_baselines_consistency, selected_okg_variants_consistency = _build_curves_dict_for_metric(
        curve_consistency, labels_consistency_by_method, with_baseline_flag=with_baseline
    )
    curves_dict_accuracy, mv_curve_accuracy, weighted_baselines_accuracy, selected_okg_variants_accuracy = _build_curves_dict_for_metric(
        curve_accuracy, labels_accuracy, with_baseline_flag=with_baseline
    )

    if with_baseline:
        if curves_dict_consistency:
            print("[Baseline] Computed baseline variants (consistency)")
        if curves_dict_accuracy:
            print("[Baseline] Computed baseline variants (accuracy)")

    # Write last-run outputs (optional).
    if save_outputs and out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for qid in order:
                answers_list = collected_answers.get(qid, [])
                final = vote_majority(answers_list)
                obj = {
                    "id": qid,
                    "question": questions.get(qid, ""),
                    "answers": answers_list,
                    "final": final,
                    "correct_answer": labels_accuracy.get(qid, labels_consistency.get(qid, "")),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OKG-AIME] saved last-run outputs: {out_path}")

    curve_payload: Optional[dict] = None
    if return_curve_data:
        curve_payload = {
            "curve_consistency": curve_consistency,
            "curve_accuracy": curve_accuracy,
            "curves_dict_consistency": curves_dict_consistency,
            "curves_dict_accuracy": curves_dict_accuracy,
            "mv_curve_consistency": mv_curve_consistency,
            "mv_curve_accuracy": mv_curve_accuracy,
            "weighted_baselines_consistency": weighted_baselines_consistency,
            "weighted_baselines_accuracy": weighted_baselines_accuracy,
            "selected_okg_variants_consistency": selected_okg_variants_consistency,
            "selected_okg_variants_accuracy": selected_okg_variants_accuracy,
            "start_budget": int(start_budget),
            # backward-compatible aliases (old single-metric output)
            "curves_dict": curves_dict_consistency,
            "mv_curve": mv_curve_consistency,
            "weighted_baselines": weighted_baselines_consistency,
            "selected_okg_variants": selected_okg_variants_consistency,
        }
        return curve_payload
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline OKG evaluation for AIME numeric answers (multi-run).")
    ap.add_argument(
        "--preds",
        default="aime25_conf_64.jsonl",
        help="input jsonl",
    )
    ap.add_argument("--B", type=int, default=64, help="per-question budget (total budget = K*B)")
    ap.add_argument(
        "--max_options",
        type=int,
        default=16,
        help="Dirichlet option buckets M per question (caps unique answer->bucket mapping)",
    )
    ap.add_argument("--nsamples", type=int, default=500, help="MC samples for OKG I(alpha)")
    ap.add_argument("--seed", type=int, default=2025, help="random seed for OKG sampling")
    ap.add_argument("--warm_up", type=int, default=1, help="uniform warm-up samples per question before OKG (0 disables)")
    ap.add_argument("--smoke_n", type=int, default=0, help="if >0, only use the first N questions for a quick run")

    ap.add_argument("--with_baseline", action="store_true", help="also compute baseline curves")
    ap.add_argument(
        "--vote_method",
        default="mv,Conf",
        help="comma-separated vote method list. Supports mv,weighted,top10,top30,top50,top70,top90,Conf,Conf_top90,tail,tail_top70,tail_top90,bottom,bottom_top90",
    )
    ap.add_argument("--multi_runs", type=int, default=10, help="number of repeated runs (>=1)")
    ap.add_argument("--accuracy_plot", default="aime_offline_multi_accuracy.png", help="multi-run accuracy summary plot path")
    ap.add_argument("--consistency_plot", default="aime_offline_multi_consistency.png", help="multi-run consistency summary plot path")
    ap.add_argument("--accuracy_csv", default=None, help="accuracy summary CSV path (default: same as plot with .csv)")
    ap.add_argument("--consistency_csv", default=None, help="consistency summary CSV path (default: same as plot with .csv)")
    ap.add_argument("--multi_run_jsonl", default=None, help="export multi-run curves+stats to JSONL (optional)")
    ap.add_argument(
        "--multi_pool_size",
        type=int,
        default=64,
        help="multi-run: per question shuffle answers, then take first K as the pool (default 64)",
    )
    ap.add_argument(
        "--multi_relabel_mv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="multi-run: when NOT evaluating accuracy, relabel gold as MV(subsampled pool) (equiv. to using final)",
    )

    args = ap.parse_args()

    bm = None
    if args.vote_method:
        bm = [s.strip() for s in str(args.vote_method).split(",") if s.strip()]
    multi_runs = max(1, int(args.multi_runs))
    multi_curves_consistency: List[Tuple[int, Dict[str, List[Tuple[int, float]]]]] = []
    multi_curves_accuracy: List[Tuple[int, Dict[str, List[Tuple[int, float]]]]] = []
    for run_idx in range(multi_runs):
        run_seed = int(args.seed) + run_idx
        payload = run(
            preds_path=str(args.preds),
            B=int(args.B),
            max_options=int(args.max_options),
            nsamples=int(args.nsamples),
            seed=run_seed,
            with_baseline=bool(args.with_baseline),
            baseline_methods=bm,
            warm_up=int(args.warm_up),
            smoke_n=int(args.smoke_n) if int(args.smoke_n) > 0 else None,
            subsample_pool_size=int(args.multi_pool_size),
            relabel_from_subsample_mv=bool(args.multi_relabel_mv),
            return_curve_data=True,
            save_outputs=False,
        )
        if isinstance(payload, dict):
            if payload.get("curves_dict_consistency"):
                multi_curves_consistency.append((run_idx, payload["curves_dict_consistency"]))  # type: ignore[arg-type]
            if payload.get("curves_dict_accuracy"):
                multi_curves_accuracy.append((run_idx, payload["curves_dict_accuracy"]))  # type: ignore[arg-type]

    if not multi_curves_consistency and not multi_curves_accuracy:
        print("[multi-run] no valid curve data collected; skip summary")
        raise SystemExit(2)

    # Plot both metrics (if available)
    if multi_curves_consistency and args.consistency_plot:
        plot_multi_run_curves(
            multi_curves_consistency,
            str(args.consistency_plot),
            title="Consistency Rate vs Total Budget (multi-run)",
            csv_path=(str(args.consistency_csv) if args.consistency_csv else None),
            overlay_runs=False,
            y_label="Consistency Rate",
            y_margin=0.03,
            y_max_cap=1.01,
        )
    else:
        print("[multi-run] no consistency curves or empty --consistency_plot; skip consistency plotting")

    if multi_curves_accuracy and args.accuracy_plot:
        plot_accuracy_multi_run_curves(
            multi_curves_accuracy,
            str(args.accuracy_plot),
            csv_path=(str(args.accuracy_csv) if args.accuracy_csv else None),
        )
    else:
        print("[multi-run] no accuracy curves or empty --accuracy_plot; skip accuracy plotting")

    if args.multi_run_jsonl:
        export_multi_run_curves_jsonl(
            multi_curves_consistency,
            multi_curves_accuracy,
            str(args.multi_run_jsonl),
        )

    # Print quick summaries for main curves
    if multi_curves_consistency:
        summaries = aggregate_multi_run_curve_stats(multi_curves_consistency)
        okg_summary = summaries.get("OKG", []) or summaries.get("okg", [])
        base_summary = summaries.get("Base", []) or summaries.get("base", [])
        if okg_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in okg_summary
            )
            print(f"[multi-run][consistency] OKG summary: {msg}")
        if base_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in base_summary
            )
            print(f"[multi-run][consistency] Base summary: {msg}")

    if multi_curves_accuracy:
        summaries = aggregate_multi_run_curve_stats(multi_curves_accuracy)
        okg_summary = summaries.get("OKG", []) or summaries.get("okg", [])
        base_summary = summaries.get("Base", []) or summaries.get("base", [])
        if okg_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in okg_summary
            )
            print(f"[multi-run][accuracy] OKG summary: {msg}")
        if base_summary:
            msg = ", ".join(
                f"t~{entry['budget']}: {entry['mean']:.4f}±{entry['std']:.4f} (n={entry['num_runs']})"
                for entry in base_summary
            )
            print(f"[multi-run][accuracy] Base summary: {msg}")
