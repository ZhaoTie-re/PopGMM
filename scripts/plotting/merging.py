"""The component-merging overview: distance matrix, dendrogram, merged clusters, confidence.

Carries two helpers that exist only for this figure and its step module:
``build_merged_cluster_palette`` (also used when the merge summary is written)
and ``compute_conf_vmin_vmax``. The config type is imported only for typing, so
the runtime import direction stays one-way: step module -> plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, PowerNorm, to_hex, to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, to_tree

from scripts.common import build_distinct_palette as _build_distinct_palette
from scripts.common import format_pc_axis_label as _shared_format_pc_axis_label
from scripts.common import format_sigfig as _format_sigfig

if TYPE_CHECKING:
    from scripts.gmm_component_merging import GMMComponentMergingConfig


def _format_pc_axis_label(col: str, eigenval: pd.DataFrame | None) -> str:
    """``PC1 (39.2%)`` -- one decimal, unlike the two used elsewhere.

    Kept deliberately: see the note in ``scripts/common.py``.
    """
    return _shared_format_pc_axis_label(col, eigenval, decimals=1)


def build_merged_cluster_palette(
    new_k: int, config: GMMComponentMergingConfig
) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    """Build a deterministic palette for merged clusters.

    Returns
    -------
    palette_rgba : list
        Matplotlib-compatible colors for plotting.
    palette_hex : list[str]
        Hex strings (e.g. "#1f77b4") corresponding to palette_rgba.
    """

    n = int(new_k)
    if n <= 0:
        return [], []

    if int(n) <= int(config.palette_small_k_max):
        # seaborn returns RGB tuples; normalize to RGBA for consistent typing/plotting.
        palette_rgb = sns.color_palette(str(config.palette_name_small_k), n_colors=int(n))
        palette_rgba = [to_rgba(c) for c in palette_rgb]
    else:
        palette_rgba = _build_distinct_palette(int(n))

    palette_hex = [to_hex(c) for c in palette_rgba]
    return list(palette_rgba), list(palette_hex)


def compute_conf_vmin_vmax(conf: np.ndarray, config: GMMComponentMergingConfig) -> tuple[float, float]:
    mode = str(getattr(config, "conf_scale_mode", "fixed")).lower()
    if mode == "fixed":
        min_allowed, max_allowed = config.conf_scale_bounds
        vmin = float(getattr(config, "conf_scale_fixed_vmin", float(min_allowed)))
        vmax = float(getattr(config, "conf_scale_fixed_vmax", float(max_allowed)))
        vmin = max(float(min_allowed), min(float(max_allowed), vmin))
        vmax = max(float(min_allowed), min(float(max_allowed), vmax))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmax == vmin:
            vmax = min(float(max_allowed), vmin + float(config.conf_scale_min_range))
        return float(vmin), float(vmax)

    low_pct = float(config.conf_scale_low_pct)
    high_pct = float(config.conf_scale_high_pct)

    vmin = float(np.percentile(conf, low_pct))
    vmax = float(np.percentile(conf, high_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.min(conf))
        vmax = float(np.max(conf))
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    min_allowed, max_allowed = config.conf_scale_bounds
    vmin = max(float(min_allowed), vmin)
    vmax = min(float(max_allowed), vmax)

    hard_floor = getattr(config, "conf_scale_hard_floor", None)
    if hard_floor is not None:
        hard_floor_f = float(hard_floor)
        hard_floor_f = max(float(min_allowed), min(float(max_allowed), hard_floor_f))
        vmin = max(vmin, hard_floor_f)
    else:
        # If confidence is highly concentrated near 1.0, percentile-based vmin can be
        # too high (e.g., >0.99), making the uppermost range dominate the colormap.
        soft_floor = float(getattr(config, "conf_scale_soft_floor", float(min_allowed)))
        soft_floor = max(float(min_allowed), min(float(max_allowed), soft_floor))
        if vmin > soft_floor:
            vmin = soft_floor

    if vmax < vmin:
        vmax = vmin

    if vmax - vmin < float(config.conf_scale_min_range):
        mid = (vmin + vmax) / 2.0
        half = float(config.conf_scale_min_range) / 2.0
        vmin = max(float(min_allowed), mid - half)
        vmax = min(float(max_allowed), mid + half)
        if vmax <= vmin:
            vmin = max(float(min_allowed), float(max_allowed) - float(config.conf_scale_min_range))
            vmax = float(max_allowed)

    return float(vmin), float(vmax)


def plot_component_merging(
    *,
    dist_df: pd.DataFrame,
    component_labels: list[str],
    linkage_matrix: np.ndarray,
    merge_threshold: float,
    old_k: int,
    new_label_map: np.ndarray,
    labels_merged: np.ndarray,
    confidence_merged: np.ndarray,
    reference_samples_gmm_merged: pd.DataFrame,
    pc_cols_used: list[str],
    eigenval: pd.DataFrame | None,
    config: GMMComponentMergingConfig,
) -> Figure:
    pc1_col, pc2_col = pc_cols_used[0], pc_cols_used[1]

    pc1 = reference_samples_gmm_merged[pc1_col].to_numpy(dtype=np.float64, copy=False)
    pc2 = reference_samples_gmm_merged[pc2_col].to_numpy(dtype=np.float64, copy=False)

    xlab = _format_pc_axis_label(pc1_col, eigenval)
    ylab = _format_pc_axis_label(pc2_col, eigenval)

    # Match the multi-panel publication layout used in scripts/gmm_clustering.py.
    # Styling is the caller's business: it wraps this in
    # ``figure_context(...)``, which is what makes the figure look the same
    # regardless of what was drawn before it.

    new_k = int(np.unique(new_label_map).size)
    cluster_palette, cluster_palette_hex = build_merged_cluster_palette(int(new_k), config)

    cluster_color = {int(k): cluster_palette[int(k)] for k in range(int(new_k))}
    cluster_color_hex = {int(k): cluster_palette_hex[int(k)] for k in range(int(new_k))}

    fig = plt.figure(figsize=(17.0, 16.0), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 1.00],
        height_ratios=[1.0, 1.0],
        left=0.055,
        right=0.985,
        bottom=0.06,
        top=0.90,
        wspace=0.28,
        hspace=0.26,
    )

    # Use dendrogram leaf order to align heatmap rows/cols with panel B.
    dendro_preview = dendrogram(
        linkage_matrix,
        labels=component_labels,
        no_plot=True,
        color_threshold=float(merge_threshold),
    )
    leaf_labels = [str(x) for x in dendro_preview["ivl"]]
    dist_df_plot: pd.DataFrame = dist_df.loc[leaf_labels, leaf_labels]

    x_tick_rotation = 90
    title_fontsize = 24
    label_fontsize = 20
    tick_fontsize = 15
    dense_tick_fontsize = 13
    cbar_label_fontsize = 16

    # A) heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3.8%", pad=0.08)
    _ = sns.heatmap(
        dist_df_plot,
        cmap="YlGnBu",
        square=True,
        cbar_ax=cax1,
        cbar_kws={"label": "Mahalanobis Distance"},
        ax=ax1,
        linewidths=0.25,
        linecolor="white",
    )
    ax1.set_title(
        "A. Mahalanobis Distance Matrix",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
        pad=8,
    )
    ax1.set_xlabel("Cluster ID", fontsize=label_fontsize)
    ax1.set_ylabel("Cluster ID", fontsize=label_fontsize)
    plt.setp(ax1.get_xticklabels(), rotation=x_tick_rotation, ha="center", fontsize=dense_tick_fontsize)
    plt.setp(ax1.get_yticklabels(), rotation=0, ha="right", fontsize=dense_tick_fontsize)
    for t in ax1.get_xticklabels():
        t.set_fontweight("bold")
    for t in ax1.get_yticklabels():
        t.set_fontweight("bold")
    cax1.tick_params(labelsize=max(10, tick_fontsize - 1))
    cax1.set_ylabel("Mahalanobis Distance", fontsize=cbar_label_fontsize)

    # B) dendrogram
    ax2 = fig.add_subplot(gs[0, 1])
    tree_result = to_tree(linkage_matrix, rd=True)
    if isinstance(tree_result, tuple):
        root, nodes = tree_result
    else:
        root, nodes = tree_result, []
    node_dist = {int(n.id): float(getattr(n, "dist", 0.0)) for n in nodes}

    node_cluster: dict[int, int | None] = {}

    def _compute_node_cluster(node) -> int | None:
        nid = int(node.id)
        if nid < int(old_k):
            node_cluster[nid] = int(new_label_map[nid])
            return node_cluster[nid]
        left_c = _compute_node_cluster(node.get_left())
        right_c = _compute_node_cluster(node.get_right())
        node_cluster[nid] = left_c if left_c == right_c else None
        return node_cluster[nid]

    _ = _compute_node_cluster(root)

    above_color = "#333333"

    def _link_color_func(k):
        k = int(k)
        if node_dist.get(k, 0.0) >= float(merge_threshold):
            return above_color
        cid = node_cluster.get(k, None)
        if cid is None:
            return above_color
        return cluster_color_hex[int(cid)]

    _ = dendrogram(
        linkage_matrix,
        ax=ax2,
        labels=component_labels,
        color_threshold=float(merge_threshold),
        above_threshold_color=above_color,
        link_color_func=_link_color_func,
    )

    for tick in ax2.get_xmajorticklabels():
        txt = tick.get_text()
        try:
            comp_id = int(txt)
        except ValueError:
            continue
        cid = int(new_label_map[comp_id])
        tick.set_color(cluster_color_hex.get(cid, "#000000"))
        tick.set_fontweight("bold")

    ax2.axhline(
        float(merge_threshold),
        color="#ff4d4d",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold={_format_sigfig(float(merge_threshold), 2)}",
    )
    ax2.legend(loc="upper right", frameon=True, fontsize=12)
    ax2.set_title(
        "B. Hierarchical Clustering Dendrogram",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
        pad=8,
    )
    linkage_method = str(getattr(config, "linkage_method", "average"))
    method_label_map = {
        "average": "Average-linkage",
        "complete": "Complete-linkage",
        "single": "Single-linkage",
        # Ward is not a linkage *distance* in the same sense; use a more generic height label.
        "ward": "Ward linkage height",
    }
    if linkage_method in method_label_map:
        base_label = method_label_map[linkage_method]
        if linkage_method == "ward":
            ax2.set_ylabel(base_label, fontsize=label_fontsize)
        else:
            ax2.set_ylabel(f"{base_label} distance (Mahalanobis)", fontsize=label_fontsize)
    else:
        ax2.set_ylabel(f"{linkage_method}-linkage distance (Mahalanobis)", fontsize=label_fontsize)
    ax2.set_xlabel("Cluster ID", fontsize=label_fontsize)
    plt.setp(ax2.get_xticklabels(), rotation=x_tick_rotation, ha="center", fontsize=dense_tick_fontsize)
    ax2.tick_params(axis="y", labelsize=tick_fontsize)

    # shared view range for C/D
    x_min, x_max = float(pc1.min()), float(pc1.max())
    y_min, y_max = float(pc2.min()), float(pc2.max())
    max_span = max(x_max - x_min, y_max - y_min)
    if max_span == 0:
        max_span = 1.0
    view_span = max_span * 1.15
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    # C) merged cluster scatter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        pc1,
        pc2,
        s=11,
        c=[cluster_color[int(v)] for v in labels_merged],
        alpha=0.90,
        edgecolors="white",
        linewidths=0.22,
        rasterized=True,
    )
    ax3.set_title(
        f"C. Final Merged Clusters (N={new_k})",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
        pad=8,
    )
    ax3.set_xlabel(xlab, fontsize=label_fontsize)
    ax3.set_ylabel(ylab, fontsize=label_fontsize)
    ax3.set_xlim(x_center - view_span / 2.0, x_center + view_span / 2.0)
    ax3.set_ylim(y_center - view_span / 2.0, y_center + view_span / 2.0)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_box_aspect(1)
    ax3.set_anchor("C")
    ax3.grid(True, linestyle="--", alpha=0.30, color="#C3C3C3")
    ax3.tick_params(axis="both", which="major", length=4.8, width=1.1, labelsize=tick_fontsize)
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.1)

    # D) confidence scatter
    ax4 = fig.add_subplot(gs[1, 1])
    conf = confidence_merged.astype(np.float32, copy=False)
    data_min = float(np.nanmin(conf))
    data_max = float(np.nanmax(conf))
    vmin, vmax = compute_conf_vmin_vmax(conf, config)

    conf_norm_mode = str(getattr(config, "conf_norm", "power")).lower()
    if conf_norm_mode == "linear":
        conf_norm: Normalize = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    else:
        gamma = float(getattr(config, "conf_power_gamma", 0.40))
        if not np.isfinite(gamma) or gamma <= 0:
            gamma = 0.40
        conf_norm = PowerNorm(gamma=float(gamma), vmin=float(vmin), vmax=float(vmax), clip=True)

    sort_idx = np.argsort(-conf)
    sc = ax4.scatter(
        pc1[sort_idx],
        pc2[sort_idx],
        c=conf[sort_idx],
        cmap="cividis",
        norm=conf_norm,
        alpha=0.9,
        s=9,
        edgecolor="none",
        rasterized=True,
    )
    ax4.set_title(
        "D. Assignment Confidence (Merged)",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
        pad=8,
    )
    ax4.set_xlabel(xlab, fontsize=label_fontsize)
    ax4.set_ylabel(ylab, fontsize=label_fontsize)
    ax4.set_xlim(x_center - view_span / 2.0, x_center + view_span / 2.0)
    ax4.set_ylim(y_center - view_span / 2.0, y_center + view_span / 2.0)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_box_aspect(1)
    ax4.set_anchor("C")
    ax4.grid(True, linestyle="--", alpha=0.30, color="#C3C3C3")
    ax4.tick_params(axis="both", which="major", length=4.8, width=1.1, labelsize=tick_fontsize)
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.1)

    # Colorbar aligned to Panel D without affecting overall grid geometry.
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="3.8%", pad=0.08)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Confidence (Merged)", fontsize=cbar_label_fontsize)
    # Ensure the scale endpoints are visible on the legend.
    ticks = np.linspace(float(vmin), float(vmax), 6).tolist()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=max(10, tick_fontsize - 1))

    thresh_txt = _format_sigfig(float(merge_threshold), 2)
    fig.suptitle(
        f"Hierarchical Cluster Merging (Mahalanobis Distance, Threshold={thresh_txt})",
        fontsize=30,
        fontweight="bold",
        y=0.965,
    )

    return fig
