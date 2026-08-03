"""The HDBSCAN denoising overview: PCA, the noise call, and the retained clusters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from scripts.common import format_pc_axis_label as _format_pc_axis_label


def plot_denoising_overview(
    reference_samples_hdbscan: pd.DataFrame,
    selected_pc_cols: list[str],
    eigenval: pd.DataFrame | None = None,
) -> Figure | None:
    """Generate a compact 3-panel QC plot for HDBSCAN results.

    Visualization is intentionally concise: all points, noise map,
    and non-noise cluster view.

    Returns ``None`` when there are fewer than two PC columns to plot against.
    """

    if len(selected_pc_cols) < 2:
        return None

    plot_df = reference_samples_hdbscan

    pc1_col, pc2_col = selected_pc_cols[0], selected_pc_cols[1]
    pc1_label = _format_pc_axis_label(pc1_col, eigenval)
    pc2_label = _format_pc_axis_label(pc2_col, eigenval)
    x = plot_df[pc1_col].to_numpy(dtype=np.float32, copy=False)
    y = plot_df[pc2_col].to_numpy(dtype=np.float32, copy=False)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    span = max(x_max - x_min, y_max - y_min) * 1.08 if len(plot_df) > 1 else 1.0
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    # Styling is the caller's business: it wraps this in
    # ``figure_context(THEME_DENOISING)``, which is what makes the figure look
    # the same regardless of what was drawn before it.
    fig, axes = plt.subplots(1, 3, figsize=(34, 10), constrained_layout=False)

    base_color = "#4C78A8"
    non_noise_color = "#2B6EA6"
    noise_color = "#D94B4B"

    ax1 = axes[0]
    ax1.scatter(
        x,
        y,
        s=10,
        c=base_color,
        alpha=0.64,
        edgecolors="white",
        linewidths=0.18,
        rasterized=True,
    )
    ax1.set_title("A. PCA of All BBJ Samples", loc="left", pad=12, fontweight="bold")
    ax1.set_xlabel(pc1_label)
    ax1.set_ylabel(pc2_label)

    ax2 = axes[1]
    is_noise = plot_df["HDBSCAN_IsNoise"].to_numpy(dtype=bool, copy=False)
    noise_palette = {False: non_noise_color, True: noise_color}
    sns.scatterplot(
        data=plot_df,
        x=pc1_col,
        y=pc2_col,
        hue="HDBSCAN_IsNoise",
        hue_order=[False, True],
        palette=noise_palette,
        s=11,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.15,
        legend=False,
        ax=ax2,
    )
    ax2.set_title("B. HDBSCAN Noise Identification", loc="left", pad=12, fontweight="bold")
    ax2.set_xlabel(pc1_label)
    ax2.set_ylabel(pc2_label)

    noise_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Non-noise",
            markerfacecolor=noise_palette[False],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Noise",
            markerfacecolor=noise_palette[True],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8.5,
        ),
    ]
    ax2.legend(
        handles=noise_legend_handles,
        title="Label",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    ax3 = axes[2]
    non_noise_df = plot_df.loc[~plot_df["HDBSCAN_IsNoise"]]
    if len(non_noise_df) > 0:
        cluster_labels = non_noise_df["HDBSCAN_Label"].to_numpy(dtype=np.int32, copy=False)
        unique_labels = np.unique(cluster_labels)
        cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
        label_to_idx = {int(label): idx for idx, label in enumerate(unique_labels.tolist())}
        colors = [cmap(label_to_idx[int(label)]) for label in cluster_labels]
        ax3.scatter(
            non_noise_df[pc1_col].to_numpy(dtype=np.float32, copy=False),
            non_noise_df[pc2_col].to_numpy(dtype=np.float32, copy=False),
            s=12,
            c=colors,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
        )
        if len(unique_labels) <= 20:
            cluster_legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Cluster {int(label)}",
                    markerfacecolor=cmap(label_to_idx[int(label)]),
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    markersize=8.5,
                )
                for label in unique_labels.tolist()
            ]
            ax3.legend(
                handles=cluster_legend_handles,
                title="Non-noise clusters",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
    else:
        ax3.text(0.5, 0.5, "No non-noise points", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("C. HDBSCAN Clusters (Noise Excluded)", loc="left", pad=12, fontweight="bold")
    ax3.set_xlabel(pc1_label)
    ax3.set_ylabel(pc2_label)

    for ax in axes:
        ax.set_xlim(x_center - span / 2.0, x_center + span / 2.0)
        ax.set_ylim(y_center - span / 2.0, y_center + span / 2.0)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=15)

    fig.suptitle(
        "HDBSCAN-Based Noise Filtering on BBJ Samples (PCA Space)",
        y=0.985,
        fontsize=24,
        fontweight="bold",
        color="#1E1E1E",
    )
    plt.tight_layout(rect=(0.01, 0.01, 0.9, 0.935))
    return fig
