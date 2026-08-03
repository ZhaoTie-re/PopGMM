"""The per-PC case/control density grid, shared by two stages.

``major_cluster_all_pcs_kde`` and ``subcluster_all_pcs_kde`` drew this grid from
blocks that differed in exactly one expression -- the panel title, which carries
variance-explained in the first and is a bare ``PC{n}`` in the second. Passing
the titles in collapses the duplication without deciding that question here.

Only the *plotter* is shared. The two step modules stay separate: they differ in
how they select rows (numeric component membership vs string group equality),
in their output TSV schema, and in their log text, and merging those would
couple a figure title to a row filter.
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from scripts.plotting.style import CASE_COLOR, CONTROL_COLOR


def plot_pc_kde_grid(
    *,
    case_values: Sequence[np.ndarray],
    control_values: Sequence[np.ndarray],
    titles: Sequence[str],
    suptitle: str,
    case_label: str,
    control_label: str,
    n_cols: int = 5,
    alpha: float = 0.65,
    case_color: str = CASE_COLOR,
    control_color: str = CONTROL_COLOR,
) -> Figure:
    """One filled KDE panel per PC, case over control.

    ``case_values[i]`` and ``control_values[i]`` are the finite values for the
    PC named by ``titles[i]``; the caller does the column resolution and the
    filtering, so this function never touches a DataFrame.
    """
    if not (len(case_values) == len(control_values) == len(titles)):
        raise ValueError(
            f"case_values ({len(case_values)}), control_values ({len(control_values)}) "
            f"and titles ({len(titles)}) must be the same length"
        )

    n_panels = len(titles)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.2 * n_cols, 4.8 * n_rows), squeeze=False
    )
    fig.subplots_adjust(wspace=0.30, hspace=0.45, left=0.06, right=0.98, top=0.82, bottom=0.07)

    for i in range(n_panels):
        ax = axes[i // n_cols][i % n_cols]
        x_ctrl = np.asarray(control_values[i])
        x_case = np.asarray(case_values[i])
        # A KDE needs at least two points to have a bandwidth.
        if x_ctrl.size > 1:
            sns.kdeplot(x=x_ctrl, color=control_color, fill=True, alpha=alpha, linewidth=2.5, ax=ax)
        if x_case.size > 1:
            sns.kdeplot(x=x_case, color=case_color, fill=True, alpha=alpha, linewidth=2.5, ax=ax)
        ax.set_title(titles[i], pad=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for j in range(n_panels, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(suptitle, fontweight="bold", y=0.995)
    fig.legend(
        handles=[
            mpatches.Patch(color=control_color, alpha=alpha, label=control_label),
            mpatches.Patch(color=case_color, alpha=alpha, label=case_label),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=22,
    )
    return fig
