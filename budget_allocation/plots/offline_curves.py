from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .common import apply_reference_plot_style, setup_matplotlib


def aggregate_multi_run_curve_stats(
    curve_runs: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]]
) -> Dict[str, List[Dict[str, Union[int, float]]]]:
    """Aggregate multi-run curves by label into mean and std."""
    if not curve_runs:
        return {}

    buckets: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _, curves in curve_runs:
        if not curves:
            continue
        for label, points in curves.items():
            if not points:
                continue
            for budget, acc in points:
                try:
                    b_val = int(budget)
                    acc_val = float(acc)
                except Exception:
                    continue
                if not math.isfinite(acc_val):
                    continue
                buckets[label][b_val].append(acc_val)

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


def plot_multi_run_curves(
    curve_runs: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    output_path: Optional[str],
    *,
    title: str = "Consistency Rate vs Total Budget (multi-run)",
    csv_path: Optional[str] = None,
    overlay_runs: bool = False,
    y_label: str = "Consistency Rate",
    y_margin: float = 0.03,
    y_max_cap: float = 1.01,
) -> None:
    """Plot multi-run mean curves with optional run overlays and summary CSV."""
    if not curve_runs or not output_path:
        return

    plt = setup_matplotlib()
    if plt is None:
        return

    if csv_path is None:
        csv_path = str(Path(output_path).with_suffix(".csv"))

    summaries = aggregate_multi_run_curve_stats(curve_runs)
    if not summaries:
        return

    num_runs = len(curve_runs)
    warm_colors = plt.cm.OrRd(np.linspace(0.4, 0.95, max(num_runs, 1)))
    cool_colors = plt.cm.Blues(np.linspace(0.4, 0.95, max(num_runs, 1)))
    oracle_colors = plt.cm.Greys(np.linspace(0.4, 0.85, max(num_runs, 1)))

    pred_color = "#9c6ed9"
    base_color = "#F0B851"
    pred_conf_color = "#bd9ee6"
    base_conf_color = "#F5D000"
    oracle_color = "#6c757d"
    other_color = "#6c757d"

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    apply_reference_plot_style(ax, grid=True)

    def _label_kind_for(name: str) -> str:
        low = name.lower()
        if "oracle" in low:
            return "oracle"
        if low.startswith("okg"):
            return "okg"
        if low.startswith("base") or low.startswith("mv"):
            return "base"
        return "other"

    def _run_color_for(label: str, idx: int):
        kind = _label_kind_for(label)
        palette = warm_colors if kind == "okg" else cool_colors if kind == "base" else oracle_colors
        return palette[min(idx, len(palette) - 1)]

    def _is_main_label(kind: str, label: str) -> bool:
        low = label.lower()
        if kind == "okg":
            return low == "okg"
        if kind == "base":
            return low == "base" or low == "mv"
        return False

    def _is_conf_label(label: str) -> bool:
        return "tail_top70" in label.lower()

    labels_by_kind: Dict[str, List[str]] = defaultdict(list)
    for lab, entries in summaries.items():
        if entries:
            labels_by_kind[_label_kind_for(lab)].append(lab)

    mean_color_for_label: Dict[str, str] = {}
    for kind, labs in labels_by_kind.items():
        for lab in labs:
            if kind == "okg":
                mean_color_for_label[lab] = pred_conf_color if _is_conf_label(lab) else pred_color
            elif kind == "base":
                mean_color_for_label[lab] = base_conf_color if _is_conf_label(lab) else base_color
            elif kind == "oracle":
                mean_color_for_label[lab] = oracle_color
            else:
                mean_color_for_label[lab] = other_color

    if overlay_runs:
        run_alpha = 0.15
        for run_idx, curves in curve_runs:
            for label, points in curves.items():
                if not points:
                    continue
                try:
                    budgets, accs = zip(*points)
                except ValueError:
                    continue
                kind = _label_kind_for(label)
                color = _run_color_for(label, run_idx)
                ls = "-" if kind == "okg" else "--" if kind == "base" else ":" if kind == "oracle" else "-."
                ax.plot(
                    budgets,
                    accs,
                    color=color,
                    alpha=run_alpha,
                    linewidth=1.0,
                    linestyle=ls,
                    marker="",
                    label="_nolegend_",
                )

    all_accs: List[float] = []
    for label, entries in sorted(summaries.items()):
        if not entries:
            continue
        budgets = [int(e["budget"]) for e in entries]
        means = np.asarray([float(e["mean"]) for e in entries], dtype=float)
        stds = np.asarray([float(e["std"]) for e in entries], dtype=float)
        all_accs.extend(means.tolist())

        kind = _label_kind_for(label)
        color = mean_color_for_label.get(
            label,
            pred_color if kind == "okg" else base_color if kind == "base" else oracle_color,
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
            lower = means - stds
            upper = means + stds
            ax.fill_between(
                budgets,
                lower,
                upper,
                color=color,
                alpha=0.12,
                label="_nolegend_",
                edgecolor="none",
                linewidth=0,
            )

    _ = y_label
    _ = title
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    if all_accs:
        ymin = max(0.0, min(all_accs) - float(y_margin))
        ymax = min(float(y_max_cap), max(all_accs) + float(y_margin))
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
            csv_path_obj = Path(csv_path)
            csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with csv_path_obj.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["label", "budget", "mean", "std", "num_runs"])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[multi-run] summary CSV saved: {csv_path_obj}")


def plot_accuracy_multi_run_curves(
    curve_runs: Sequence[Tuple[int, Dict[str, List[Tuple[int, float]]]]],
    output_path: Optional[str],
    *,
    title: str = "",
    csv_path: Optional[str] = None,
    overlay_runs: bool = True,
) -> None:
    """Accuracy wrapper with tighter y-axis margin."""
    _ = overlay_runs
    plot_multi_run_curves(
        curve_runs,
        output_path,
        title=title,
        csv_path=csv_path,
        overlay_runs=False,
        y_label="Accuracy",
        y_margin=0.007,
        y_max_cap=1.0,
    )

