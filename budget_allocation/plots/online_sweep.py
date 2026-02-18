from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .common import apply_reference_plot_style, setup_matplotlib


def aggregate_multi_run_sweep_xy_stats(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    *,
    metric: str,
) -> Dict[str, List[Dict[str, Union[int, float]]]]:
    """Aggregate multi-run sweep rows by average budget and realized total budget."""
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

    buckets: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"totals": [], "metrics": []})
    )

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
    """Plot multi-run sweep curves with x=mean(total budget), y=mean(metric)."""
    if not sweep_runs or not output_path:
        return

    plt = setup_matplotlib()
    if plt is None:
        return

    output_path = str(output_path)
    if csv_path is None:
        csv_path = str(Path(output_path).with_suffix(".csv"))

    summaries = aggregate_multi_run_sweep_xy_stats(sweep_runs, metric=metric)
    if not summaries:
        return

    pred_color = "#9c6ed9"
    base_color = "#F0B851"
    oracle_color = "#6c757d"
    pred_conf_color = "#bd9ee6"
    base_conf_color = "#F5D000"
    oracle_conf_color = "#8c9094"

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    apply_reference_plot_style(ax, grid=True)

    def _is_conf_label(label: str) -> bool:
        return "conf" in str(label).lower()

    def _kind(label: str) -> str:
        low = str(label).lower()
        if "oracle" in low:
            return "oracle"
        if low.startswith("okg") or low.startswith("pred"):
            return "okg"
        if low.startswith("base") or low.startswith("mv"):
            return "base"
        return "other"

    def _color(label: str) -> str:
        kind = _kind(label)
        is_conf = _is_conf_label(label)
        if kind == "okg":
            return pred_conf_color if is_conf else pred_color
        if kind == "base":
            return base_conf_color if is_conf else base_color
        return oracle_conf_color if is_conf else oracle_color

    all_means: List[float] = []
    for label, entries in sorted(summaries.items()):
        xs = np.asarray([float(e["total_mean"]) for e in entries], dtype=float)
        ys = np.asarray([float(e["metric_mean"]) for e in entries], dtype=float)
        ystd = np.asarray([float(e["metric_std"]) for e in entries], dtype=float)
        all_means.extend(ys.tolist())

        color = _color(label)
        linestyle = "--" if _is_conf_label(label) else "-"
        ax.plot(xs, ys, color=color, linewidth=3.3, marker="", linestyle=linestyle, label="_nolegend_")
        if not _is_conf_label(label):
            ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.12, edgecolor="none", linewidth=0)

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


def plot_accuracy_multi_run_curves(
    sweep_runs: Sequence[Tuple[int, Sequence[Dict[str, object]]]],
    output_path: Optional[Union[str, Path]],
    *,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
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

