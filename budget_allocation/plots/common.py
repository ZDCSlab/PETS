from __future__ import annotations


def setup_matplotlib():
    """Return a headless matplotlib.pyplot module, or None when unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("[warning] matplotlib not installed. pip install matplotlib")
        return None


def apply_reference_plot_style(ax, *, grid: bool = True) -> None:
    """Apply the publication-style visual theme used across budget plots."""
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

