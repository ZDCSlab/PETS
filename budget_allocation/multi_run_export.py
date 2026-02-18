from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


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
