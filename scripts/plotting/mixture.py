"""The Gaussian-mixture overview: BIC selection, structure, cluster sizes, confidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from scripts.common import STORE_DTYPE
from scripts.common import build_distinct_palette as _build_distinct_palette
from scripts.common import format_pc_axis_label as _format_pc_axis_label


def plot_mixture_overview(
    x: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    bic_table: pd.DataFrame,
    best_k: int,
    selected_pc_cols: list[str],
    eigenval: pd.DataFrame | None,
) -> Figure | None:
    """Returns ``None`` when fewer than two PCs were fitted."""
    if x.shape[1] < 2:
        return None

    # Match the multi-panel publication layout used in demo.ipynb cell 5.
    # Styling is the caller's business: it wraps this in
    # ``figure_context(...)``, which is what makes the figure look the same
    # regardless of what was drawn before it.

    pc1_label = _format_pc_axis_label(selected_pc_cols[0], eigenval)
    pc2_label = _format_pc_axis_label(selected_pc_cols[1], eigenval)

    fig, axes = plt.subplots(2, 2, figsize=(25, 18), constrained_layout=False)
    fig.subplots_adjust(wspace=0.22, hspace=0.18, right=0.92)
    fig.patch.set_facecolor("white")

    for ax in axes.ravel():
        ax.set_facecolor("white")

    line_color = "#1F4E79"
    highlight_color = "#B22222"
    best_bic = float(bic_table.loc[bic_table["n_components"] == best_k, "bic"].iloc[0])

    # A) BIC line plot.
    ax0 = axes[0, 0]
    ks = bic_table["n_components"].to_numpy(dtype=np.int32)
    bics = bic_table["bic"].to_numpy(dtype=np.float64)

    # Keep absolute BIC as requested, but optimize readability for wide k ranges.
    ax0.plot(
        ks,
        bics,
        color=line_color,
        linewidth=3.2,
        zorder=3,
    )
    ax0.scatter(
        ks,
        bics,
        s=28,
        color="white",
        edgecolor=line_color,
        linewidth=1.0,
        alpha=0.95,
        zorder=4,
    )
    ax0.scatter(
        [best_k],
        [best_bic],
        s=280,
        color=highlight_color,
        edgecolor="white",
        linewidth=2.2,
        zorder=5,
        label="Selected model",
    )
    # Robust tick generation for large contiguous k-ranges.
    if ks.shape[0] > 12:
        max_tick_count = 11
        sampled = np.rint(np.linspace(int(ks[0]), int(ks[-1]), num=max_tick_count)).astype(np.int32)
        tick_candidates = np.unique(sampled).tolist()
        if len(tick_candidates) >= 2:
            median_step = float(np.median(np.diff(np.array(tick_candidates, dtype=np.float64))))
        else:
            median_step = 1.0
        if all(abs(int(best_k) - int(t)) > max(1.0, median_step * 0.35) for t in tick_candidates):
            tick_candidates.append(int(best_k))
        sparse_ticks = sorted(set(tick_candidates + [int(ks[0]), int(ks[-1])]))
        ax0.set_xticks(sparse_ticks)
    else:
        ax0.set_xticks(ks.tolist())

    y_min = float(np.min(bics))
    y_max = float(np.max(bics))
    y_pad = max(1.0, (y_max - y_min) * 0.06)
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:,.0f}"))

    ax0.set_xlim(float(ks[0]) - 0.5, float(ks[-1]) + 0.5)
    ax0.set_ylim(y_min - y_pad, y_max + y_pad)
    ax0.grid(True, linestyle="--", alpha=0.35, color="#BFBFBF")
    _ = ax0.legend(loc="upper right", frameon=True, framealpha=0.95)

    ax0.text(
        0.62,
        0.76,
        (
            f"Selected PCs = {x.shape[1]}\n"
            f"Optimal k = {best_k}\n"
            f"BIC = {best_bic:,.0f}"
        ),
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.40", "fc": "#F8F9FB", "ec": "#4D4D4D", "lw": 1.2, "alpha": 0.98},
        zorder=10,
    )

    ax0.set_title("A. Model Selection (BIC Curve)", loc="left", fontweight="bold", pad=10)
    ax0.set_xlabel("Number of Clusters")
    ax0.set_ylabel("BIC")
    ax0.tick_params(axis="x", labelrotation=0)
    ax0.tick_params(axis="both", which="major", length=5.0, width=1.2)
    for spine in ax0.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.1)

    # Shared view range for panels B/D.
    pc1 = x[:, 0]
    pc2 = x[:, 1]
    x_min, x_max = float(pc1.min()), float(pc1.max())
    y_min, y_max = float(pc2.min()), float(pc2.max())
    max_span = max(x_max - x_min, y_max - y_min)
    if max_span == 0:
        max_span = 1.0
    view_span = max_span * 1.15
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    # B) Cluster scatter.
    ax1 = axes[0, 1]
    unique_labels = np.unique(labels)
    palette = _build_distinct_palette(len(unique_labels))
    color_map = {int(k): palette[i] for i, k in enumerate(unique_labels.tolist())}
    ax1.scatter(
        pc1,
        pc2,
        s=11,
        c=[color_map[int(v)] for v in labels],
        alpha=0.90,
        edgecolors="white",
        linewidths=0.22,
        rasterized=True,
    )

    # Panel B intentionally has no legend; cluster-color mapping is shown in panel C bars.
    ax1.set_title("B. Population Substructure (BBJ Samples)", loc="left", fontweight="bold", pad=10)
    ax1.set_xlabel(pc1_label)
    ax1.set_ylabel(pc2_label)
    ax1.set_xlim(x_center - view_span / 2.0, x_center + view_span / 2.0)
    ax1.set_ylim(y_center - view_span / 2.0, y_center + view_span / 2.0)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_box_aspect(1)
    ax1.set_anchor("C")
    ax1.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
    ax1.tick_params(axis="both", which="major", length=4.8, width=1.1)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.1)

    # C) Cluster size summary (bar chart, publication-friendly alternative to dense tables).
    ax2 = axes[1, 0]
    ax2.set_title("C. Cluster Size Summary", loc="left", fontweight="bold", pad=10)

    cluster_counts = (
        pd.Series(labels, name="Cluster ID")
        .value_counts()
        .reset_index(name="Samples")
        .rename(columns={"index": "Cluster ID"})
    )
    # Sort by cluster size (Samples) as requested; tie-break by Cluster ID.
    cluster_counts = cluster_counts.sort_values(["Samples", "Cluster ID"], ascending=[False, True]).reset_index(
        drop=True
    )
    cluster_counts["Share (%)"] = cluster_counts["Samples"] / float(cluster_counts["Samples"].sum()) * 100.0
    cluster_ids = cluster_counts["Cluster ID"].astype(int).to_numpy()
    sample_counts = cluster_counts["Samples"].astype(float).to_numpy()
    bar_colors = [color_map[int(cid)] for cid in cluster_ids.tolist()]

    # Use index positions to force rendering every cluster ID tick label.
    y_pos = np.arange(cluster_ids.shape[0], dtype=np.int32)
    ax2.barh(y_pos, sample_counts, color=bar_colors, edgecolor="white", linewidth=0.6, alpha=0.95)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([str(int(v)) for v in cluster_ids.tolist()])
    ax2.invert_yaxis()
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Cluster ID")
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v):,}"))
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35, color="#C3C3C3")
    ax2.grid(False, axis="y")

    # Make panel-C typography larger while staying readable for many clusters.
    y_tick_size = 12 if cluster_ids.shape[0] >= 22 else 14
    x_tick_size = 14
    ax2.tick_params(axis="y", labelsize=y_tick_size)
    ax2.tick_params(axis="x", labelsize=x_tick_size)
    ax2.tick_params(axis="both", which="major", length=4.8, width=1.1)
    ax2.set_xlim(0.0, float(sample_counts.max()) * 1.12)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.1)

    # D) Assignment confidence.
    ax3 = axes[1, 1]
    confidence = np.max(probabilities, axis=1).astype(STORE_DTYPE, copy=False)
    sort_idx = np.argsort(-confidence)
    sc = ax3.scatter(
        pc1[sort_idx],
        pc2[sort_idx],
        c=confidence[sort_idx],
        cmap="cividis",
        vmin=float(np.percentile(confidence, 1.0)),
        vmax=float(np.percentile(confidence, 99.0)),
        alpha=0.9,
        s=9,
        edgecolor="none",
        rasterized=True,
    )
    ax3.set_title("D. Assignment Confidence", loc="left", fontweight="bold", pad=10)
    ax3.set_xlabel(pc1_label)
    ax3.set_ylabel(pc2_label)
    ax3.set_xlim(x_center - view_span / 2.0, x_center + view_span / 2.0)
    ax3.set_ylim(y_center - view_span / 2.0, y_center + view_span / 2.0)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_box_aspect(1)
    ax3.set_anchor("C")
    ax3.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
    ax3.tick_params(axis="both", which="major", length=4.8, width=1.1)
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.1)

    # Use a dedicated colorbar axis so panel D keeps the same plotting size as panel B.
    pos = ax3.get_position()
    cax = fig.add_axes(
        (
            float(pos.x1 + 0.015),
            float(pos.y0 + pos.height * 0.04),
            0.010,
            float(pos.height * 0.92),
        )
    )
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Max Posterior Probability (Confidence)", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle("BBJ Population Structure Analysis via Gaussian Mixture Models", fontsize=30, fontweight="bold")
    return fig
