"""Merge adjacent mixture components and identify the major cluster.

Distances are Mahalanobis between component means under the pooled covariance
``S_ij = (Sigma_i + Sigma_j) / 2``, so component shape is accounted for rather
than centroid distance alone. Hierarchical clustering on that matrix is cut at a
configured height.

The **major cluster** is the merged cluster holding the most pre-merge
components, ties broken by the smallest id. It is derived, never configured --
the count is a property of the fitted model. Its population-genetic
interpretation is an assumption this module does not verify;
``summarize_threshold_robustness`` exists so the identification can at least be
shown stable across cut heights.

Inputs
------
The fitted mixture, the clustered reference panel, eigenvalues, run summary.

Outputs
-------
Merge map, pairwise distance table, merged sample table, merged posteriors,
major-cluster reference, merge summary JSON, and a 2x2 overview figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, PowerNorm, to_hex, to_rgba
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, to_tree
from scipy.spatial.distance import squareform
from sklearn.mixture import GaussianMixture

from scripts.common import (
    build_distinct_palette as _build_distinct_palette,
    format_pc_axis_label as _shared_format_pc_axis_label,
    pc_index_from_col as _pc_index_from_col,
    pc_sort_key as _pc_sort_key,
    resolve_pc_columns as _resolve_pc_columns,
)


def _format_pc_axis_label(col: str, eigenval: pd.DataFrame | None) -> str:
    """One decimal here, unlike the two used elsewhere -- keeping the
    published axis text of gmm_component_merging_overview.png unchanged."""
    return _shared_format_pc_axis_label(col, eigenval, decimals=1)


class GMMComponentMergingOutput(NamedTuple):
    """Container for GMM component merging outputs."""

    dist_df: pd.DataFrame
    merge_map: pd.DataFrame
    merged_counts: pd.DataFrame
    reference_samples_gmm_merged: pd.DataFrame
    probs_merged: np.ndarray
    confidence_merged: np.ndarray
    labels_merged: np.ndarray
    label_map: dict[int, int]
    linkage_matrix: np.ndarray
    summary: dict[str, Any]
    major_cluster_id: int
    major_cluster_component_id: int
    major_cluster_component_ids: list[int]
    output_dir: Path
    figure_path: Path | None


@dataclass(frozen=True)
class GMMComponentMergingConfig:
    """Configuration for merging GMM components.

    Parameters
    ----------
    merge_threshold : float
        Dendrogram cut threshold under distance criterion.
    linkage_method : str
        Linkage method for hierarchical clustering (default: "average").
    output_dir : str
        Directory for outputs.
    save_plot : bool
        Save the 2x2 overview figure.
    show_plot : bool
        Show the figure (useful in notebooks).
    save_tables : bool
        Save TSV/NPY/JSON outputs.
    palette_name_small_k : str
        Seaborn categorical palette used when new_k <= palette_small_k_max.
    palette_small_k_max : int
        Upper bound for using palette_name_small_k.
    conf_scale_mode : str
        How to choose the color legend range for confidence.
        - "fixed": use conf_scale_fixed_vmin/vmax (recommended for stable,
          interpretable legends across runs).
        - "percentile": use percentile bounds (conf_scale_low_pct/high_pct)
          with optional floor constraints.
    conf_scale_fixed_vmin, conf_scale_fixed_vmax : float
        Fixed confidence legend endpoints used when conf_scale_mode="fixed".
    conf_scale_low_pct, conf_scale_high_pct : float
        Percentile bounds for confidence scaling when conf_scale_mode="percentile".
    conf_scale_soft_floor : float
        Soft lower bound for percentile vmin (only used in percentile mode).
        If percentile-based vmin is higher than this value, vmin will be lowered
        to this floor to avoid allocating most of the colormap to 0.99–1.00.
    conf_scale_hard_floor : float | None
        Hard lower bound for percentile vmin (only used in percentile mode).
        If set, vmin will be raised to at least this value (and values below
        vmin will be clipped to the lowest color). If None, soft-floor is used.
    conf_scale_min_range : float
        Minimum (vmax-vmin) allowed for confidence scaling.
    conf_scale_bounds : tuple
        Hard bounds for confidence vmin/vmax.
    conf_norm : str
        Color normalization for confidence.
        - "power": uses a PowerNorm with gamma=conf_power_gamma.
          gamma<1 compresses differences near vmax (high-confidence end), which
          reduces over-contrast when values concentrate close to 1.0.
          gamma>1 compresses differences near vmin (low-confidence end).
        - "linear": standard linear scale.
    conf_power_gamma : float
        Gamma for power normalization (see conf_norm).
    verbose : bool
        Print structured logs.
    """

    merge_threshold: float = 5.0
    linkage_method: Literal["average", "complete", "single", "ward"] = "average"
    output_dir: str = "results/01_reference_model/component_merging"
    save_plot: bool = True
    show_plot: bool = False
    save_tables: bool = True

    palette_name_small_k: str = "husl"
    palette_small_k_max: int = 20

    conf_scale_mode: Literal["fixed", "percentile"] = "fixed"
    conf_scale_fixed_vmin: float = 0.95
    conf_scale_fixed_vmax: float = 1.0

    conf_scale_low_pct: float = 1.0
    conf_scale_high_pct: float = 99.0
    conf_scale_soft_floor: float = 0.95
    conf_scale_hard_floor: float | None = None
    conf_scale_min_range: float = 0.005
    conf_scale_bounds: tuple[float, float] = (0.0, 1.0)

    conf_norm: Literal["power", "linear"] = "power"
    conf_power_gamma: float = 0.40

    verbose: bool = True


def _build_merged_cluster_palette(
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


def _format_sigfig(value: float, sig: int = 2) -> str:
    """Format a number with a fixed number of significant digits.

    Uses fixed-point formatting where appropriate to preserve trailing zeros
    (e.g., 5.0 for 2 sig figs), and rounds to tens/hundreds for large values.
    """

    v = float(value)
    if not np.isfinite(v):
        return str(v)
    if v == 0.0:
        return "0"

    sig_i = int(sig)
    if sig_i <= 0:
        sig_i = 1

    abs_v = abs(v)
    exp = int(np.floor(np.log10(abs_v)))
    decimals = sig_i - 1 - exp

    if decimals >= 0:
        return f"{v:.{decimals}f}"

    # decimals < 0 -> round to nearest 10/100/... and print as integer.
    rounded = round(v, -decimals)
    return f"{rounded:.0f}"


def _extract_component_covariances(model: GaussianMixture) -> np.ndarray:
    means_all = np.asarray(model.means_, dtype=np.float64)
    old_k, d = means_all.shape
    cov_type = str(getattr(model, "covariance_type", "full"))

    if cov_type == "full":
        cov_raw = np.asarray(model.covariances_, dtype=np.float64)
    elif cov_type == "tied":
        tied = np.asarray(model.covariances_, dtype=np.float64)
        cov_raw = np.repeat(tied[None, :, :], old_k, axis=0)
    elif cov_type == "diag":
        diag_cov = np.asarray(model.covariances_, dtype=np.float64)
        cov_raw = np.array([np.diag(diag_cov[i]) for i in range(old_k)], dtype=np.float64)
    elif cov_type == "spherical":
        sph = np.asarray(model.covariances_, dtype=np.float64)
        cov_raw = np.array([np.eye(d, dtype=np.float64) * float(sph[i]) for i in range(old_k)], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported covariance_type: {cov_type}")

    return cov_raw


def _pairwise_pooled_mahalanobis(means: np.ndarray, cov_raw: np.ndarray) -> np.ndarray:
    old_k, d = means.shape
    dist_matrix = np.zeros((old_k, old_k), dtype=np.float64)

    for i in range(old_k):
        for j in range(i + 1, old_k):
            diff = means[i] - means[j]
            s_ij = 0.5 * (cov_raw[i] + cov_raw[j])
            s_ij = (s_ij + s_ij.T) / 2.0
            s_ij += np.eye(d, dtype=np.float64) * 1e-9
            s_inv = np.linalg.pinv(s_ij, hermitian=True)
            maha_dist = float(np.sqrt(np.clip(diff @ s_inv @ diff.T, 0.0, None)))
            dist_matrix[i, j] = maha_dist
            dist_matrix[j, i] = maha_dist

    return dist_matrix


def _compute_conf_vmin_vmax(conf: np.ndarray, config: GMMComponentMergingConfig) -> tuple[float, float]:
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


def _plot_overview_2x2(
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
    output_path: Path,
) -> None:
    pc1_col, pc2_col = pc_cols_used[0], pc_cols_used[1]

    pc1 = reference_samples_gmm_merged[pc1_col].to_numpy(dtype=np.float64, copy=False)
    pc2 = reference_samples_gmm_merged[pc2_col].to_numpy(dtype=np.float64, copy=False)

    xlab = _format_pc_axis_label(pc1_col, eigenval)
    ylab = _format_pc_axis_label(pc2_col, eigenval)

    # Match the multi-panel publication layout used in scripts/gmm_clustering.py.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.title_fontsize": 14,
            "legend.fontsize": 12,
            "figure.titlesize": 30,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
        }
    )

    new_k = int(np.unique(new_label_map).size)
    cluster_palette, cluster_palette_hex = _build_merged_cluster_palette(int(new_k), config)

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
    vmin, vmax = _compute_conf_vmin_vmax(conf, config)

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

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    if bool(config.show_plot):
        plt.show()
    else:
        plt.close(fig)


def run_gmm_component_merging(
    *,
    gmm_model: GaussianMixture,
    reference_samples_gmm: pd.DataFrame,
    eigenval: pd.DataFrame | None = None,
    gmm_summary: dict[str, Any] | None = None,
    config: GMMComponentMergingConfig | None = None,
) -> GMMComponentMergingOutput:
    config = config or GMMComponentMergingConfig()

    if reference_samples_gmm.empty:
        raise ValueError("bbj_samples_gmm is empty; cannot run component merging.")

    if float(config.merge_threshold) <= 0:
        raise ValueError("merge_threshold must be > 0.")

    if "GMM_Cluster" not in reference_samples_gmm.columns:
        raise ValueError("bbj_samples_gmm must contain 'GMM_Cluster' column.")

    # Resolve PCs used by the fitted GMM (prefer recorded summary, fallback to table columns).
    pc_cols_used: list[str] = []
    if isinstance(gmm_summary, dict):
        pc_cols_used = [c for c in gmm_summary.get("pc_columns_used", []) if c in reference_samples_gmm.columns]

    if not pc_cols_used:
        pc_cols_used = _resolve_pc_columns(reference_samples_gmm)
        n_features = int(getattr(gmm_model, "n_features_in_", 2))
        pc_cols_used = pc_cols_used[: max(2, n_features)]

    if len(pc_cols_used) < 2:
        raise RuntimeError("At least two PC columns are required for visualization.")

    x_model = reference_samples_gmm[pc_cols_used].to_numpy(dtype=np.float64, copy=False)
    labels_raw = reference_samples_gmm["GMM_Cluster"].to_numpy(dtype=np.int32, copy=False)

    means_all = np.asarray(gmm_model.means_, dtype=np.float64)
    old_k, _d = means_all.shape
    cov_type = str(getattr(gmm_model, "covariance_type", "full"))

    cov_raw = _extract_component_covariances(gmm_model)
    dist_matrix = _pairwise_pooled_mahalanobis(means_all, cov_raw)

    component_labels = [str(i) for i in range(int(old_k))]
    dist_df = pd.DataFrame(dist_matrix, index=component_labels, columns=component_labels)

    dists_condensed = squareform(dist_matrix, checks=False)
    linkage_method = str(config.linkage_method)
    Z = linkage(dists_condensed, method=linkage_method)

    labels_compact = fcluster(Z, t=float(config.merge_threshold), criterion="distance")
    new_label_map = labels_compact.astype(np.int32) - 1
    new_k = int(np.unique(new_label_map).size)

    label_map = {int(old): int(new_label_map[old]) for old in range(int(old_k))}
    labels_merged = np.array([label_map[int(x)] for x in labels_raw], dtype=np.int32)

    reference_samples_gmm_merged = reference_samples_gmm.copy()
    reference_samples_gmm_merged["GMM_Cluster_Merged"] = labels_merged

    probs_all = gmm_model.predict_proba(x_model).astype(np.float64, copy=False)
    probs_merged = np.zeros((probs_all.shape[0], int(new_k)), dtype=np.float64)
    for c_old in range(int(old_k)):
        probs_merged[:, label_map[c_old]] += probs_all[:, c_old]

    row_sums = probs_merged.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_merged = probs_merged / row_sums
    confidence_merged = probs_merged.max(axis=1)

    merged_counts = (
        reference_samples_gmm_merged["GMM_Cluster_Merged"]
        .value_counts()
        .sort_index()
        .rename_axis("GMM_Cluster_Merged")
        .reset_index(name="Samples")
    )
    merged_counts["Share(%)"] = merged_counts["Samples"] / float(merged_counts["Samples"].sum()) * 100.0

    merge_map = (
        pd.DataFrame(
            {
                "GMM_Component": np.arange(int(old_k), dtype=np.int32),
                "Merged_Cluster": np.array([label_map[i] for i in range(int(old_k))], dtype=np.int32),
            }
        )
        .sort_values(["Merged_Cluster", "GMM_Component"])
        .reset_index(drop=True)
    )

    # Define a reproducible "mainland" merged cluster and its representative
    # pre-merge cluster ID for downstream tracking.
    merged_component_stats = (
        merge_map.groupby("Merged_Cluster", as_index=False)
        .agg(
            Merged_Component_Count=("GMM_Component", "count"),
        )
        .sort_values(["Merged_Component_Count", "Merged_Cluster"], ascending=[False, True])
        .reset_index(drop=True)
    )

    top_merged_cluster = merged_component_stats.iloc[0]
    major_cluster_component_count = int(np.asarray(top_merged_cluster["Merged_Component_Count"]).item())
    major_cluster_id = int(np.asarray(top_merged_cluster["Merged_Cluster"]).item())

    major_cluster_components_df = (
        merge_map.loc[merge_map["Merged_Cluster"] == major_cluster_id, ["GMM_Component"]]
        .sort_values(["GMM_Component"], ascending=[True])
        .reset_index(drop=True)
    )
    major_cluster_component_ids = major_cluster_components_df["GMM_Component"].astype(int).tolist()
    major_cluster_component_id = int(np.asarray(major_cluster_components_df.iloc[0]["GMM_Component"]).item())

    merge_map["Is_Mainland_Merged_Cluster"] = merge_map["Merged_Cluster"].eq(major_cluster_id)

    # Store merged-cluster colors (Panel C palette) for reproducibility.
    _palette_rgba, palette_hex = _build_merged_cluster_palette(int(new_k), config)
    merged_cluster_to_hex = {int(k): str(palette_hex[int(k)]) for k in range(int(new_k))}
    merge_map["Merged_Cluster_Color"] = merge_map["Merged_Cluster"].map(merged_cluster_to_hex)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "old_k": int(old_k),
        "new_k": int(new_k),
        "covariance_type": str(cov_type),
        "distance_metric": "mahalanobis_between_component_means",
        "covariance_for_distance": "pairwise_average_component_covariance",
        "pairwise_covariance_formula": "S_ij=0.5*(Sigma_i+Sigma_j)",
        "linkage_method": str(linkage_method),
        "merge_threshold": float(config.merge_threshold),
        "pc_columns_used": list(pc_cols_used),
        "mainland_merged_cluster_id": int(major_cluster_id),
        "mainland_premerge_cluster_id": int(major_cluster_component_id),
        "mainland_premerge_cluster_ids": [int(x) for x in major_cluster_component_ids],
        "mainland_selection_rule": "merged_cluster_with_max_premerge_components_then_smallest_id",
        "mainland_premerge_selection_rule": "within_mainland_merged_cluster_smallest_premerge_component_id",
        "output_rows": int(reference_samples_gmm_merged.shape[0]),
    }

    major_cluster_reference = {
        "mainland_merged_cluster_id": int(major_cluster_id),
        "mainland_premerge_cluster_id": int(major_cluster_component_id),
        "mainland_premerge_cluster_ids": [int(x) for x in major_cluster_component_ids],
        "mainland_merged_component_count": int(major_cluster_component_count),
        "mainland_selection_rule": "merged_cluster_with_max_premerge_components_then_smallest_id",
        "mainland_premerge_selection_rule": "within_mainland_merged_cluster_smallest_premerge_component_id",
    }

    figure_path: Path | None = None
    if bool(config.save_tables):
        dist_df.to_csv(out_dir / "component_mahalanobis_distance.tsv", sep="\t")
        merge_map.to_csv(out_dir / "component_merge_map.tsv", sep="\t", index=False)
        reference_samples_gmm_merged.to_csv(out_dir / "reference_samples_merged.tsv", sep="\t", index=False)
        merged_counts.to_csv(out_dir / "merged_cluster_summary.tsv", sep="\t", index=False)
        np.save(out_dir / "merged_posterior_probabilities.npy", probs_merged)

        pd.DataFrame(
            [
                {
                    "Mainland_Merged_Cluster": int(major_cluster_id),
                    "Mainland_Premerge_Cluster_Default": int(major_cluster_component_id),
                    "Mainland_Premerge_Clusters": ",".join(str(x) for x in major_cluster_component_ids),
                    "Mainland_Merged_Component_Count": int(major_cluster_component_count),
                }
            ]
        ).to_csv(out_dir / "major_cluster_reference.tsv", sep="\t", index=False)

        with (out_dir / "merge_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with (out_dir / "major_cluster_reference.json").open("w", encoding="utf-8") as f:
            json.dump(major_cluster_reference, f, indent=2)

    if bool(config.save_plot):
        figure_path = out_dir / "component_merging_overview.png"
        _plot_overview_2x2(
            dist_df=dist_df,
            component_labels=component_labels,
            linkage_matrix=Z,
            merge_threshold=float(config.merge_threshold),
            old_k=int(old_k),
            new_label_map=new_label_map,
            labels_merged=labels_merged,
            confidence_merged=confidence_merged,
            reference_samples_gmm_merged=reference_samples_gmm_merged,
            pc_cols_used=pc_cols_used,
            eigenval=eigenval,
            config=config,
            output_path=figure_path,
        )

    if bool(config.verbose):
        print("\n" + "=" * 80)
        print("COMPONENT MERGING (MAHALANOBIS + HIERARCHICAL CLUSTERING)".center(80))
        print("=" * 80)
        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  linkage_method        : {linkage_method}")
        print(f"  merge_threshold       : {_format_sigfig(float(config.merge_threshold), 2)}")
        print(f"  covariance_type       : {cov_type}")
        print(f"  conf_scale_mode       : {str(getattr(config, 'conf_scale_mode', 'fixed'))}")
        print(
            "  conf_scale_fixed      : "
            f"{float(getattr(config, 'conf_scale_fixed_vmin', 0.95)):.3f}–"
            f"{float(getattr(config, 'conf_scale_fixed_vmax', 1.00)):.3f}"
        )
        print(f"  conf_norm             : {str(getattr(config, 'conf_norm', 'power'))}")
        print(f"  conf_power_gamma      : {float(getattr(config, 'conf_power_gamma', 0.40)):.3f}")
        print(f"  conf_scale_hard_floor : {getattr(config, 'conf_scale_hard_floor', None)}")
        print(f"  save_plot             : {bool(config.save_plot)}")
        print(f"  save_tables           : {bool(config.save_tables)}")
        print(f"  mainland_merged_id    : {int(major_cluster_id)}")
        print(f"  mainland_premerge_id  : {int(major_cluster_component_id)}")
        print(f"  output_dir            : {out_dir}")
        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  input_rows            : {reference_samples_gmm_merged.shape[0]:,}")
        print(f"  original_components   : {old_k}")
        print(f"  merged_components     : {new_k}")
        print("=" * 80)

    return GMMComponentMergingOutput(
        dist_df=dist_df,
        merge_map=merge_map,
        merged_counts=merged_counts,
        reference_samples_gmm_merged=reference_samples_gmm_merged,
        probs_merged=probs_merged,
        confidence_merged=confidence_merged,
        labels_merged=labels_merged,
        label_map=label_map,
        linkage_matrix=Z,
        summary=summary,
        major_cluster_id=int(major_cluster_id),
        major_cluster_component_id=int(major_cluster_component_id),
        major_cluster_component_ids=[int(x) for x in major_cluster_component_ids],
        output_dir=out_dir,
        figure_path=figure_path,
    )


def summarize_threshold_robustness(
    *,
    results_by_threshold: dict[float, GMMComponentMergingOutput],
    main_threshold: float,
    output_path: Path | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare the major-cluster identification across dendrogram cut heights.

    The major cluster is defined as the merged cluster holding the most
    pre-merge components. That rule is unambiguous at any single threshold, but
    a reader is entitled to ask whether it picks out the same region of PC space
    when the cut moves. This tabulates that: for each threshold, how many
    components and samples the major cluster holds, and how its component set
    relates to the main analysis -- a strict subset means tightening the cut
    only carves the same region more finely, whereas a low Jaccard index would
    mean the identification jumps elsewhere and is not robust.

    Inputs
    ------
    One merging result per threshold, including the main one.

    Outputs
    -------
    The comparison table, written to ``output_path`` when given.
    """
    main = results_by_threshold.get(main_threshold)
    if main is None:
        raise KeyError(f"results_by_threshold must include the main threshold {main_threshold!r}")
    main_ids = set(int(x) for x in main.major_cluster_component_ids)

    rows: list[dict[str, Any]] = []
    for threshold in sorted(results_by_threshold):
        out = results_by_threshold[threshold]
        ids = set(int(x) for x in out.major_cluster_component_ids)
        merged_col = "GMM_Cluster_Merged" if "GMM_Cluster_Merged" in out.reference_samples_gmm_merged.columns else None
        if merged_col is not None:
            counts = out.reference_samples_gmm_merged[merged_col]
            n_samples = int((counts == int(out.major_cluster_id)).sum())
            total = int(len(counts))
        else:
            n_samples, total = -1, -1
        union = ids | main_ids
        rows.append(
            {
                "merge_threshold": float(threshold),
                "is_main_analysis": bool(threshold == main_threshold),
                "n_merged_clusters": int(out.summary.get("new_k", -1)),
                "major_cluster_id": int(out.major_cluster_id),
                "n_components": int(len(ids)),
                "n_samples": n_samples,
                "sample_share": (n_samples / total) if total > 0 else float("nan"),
                "is_subset_of_main": bool(ids <= main_ids),
                "jaccard_vs_main": (len(ids & main_ids) / len(union)) if union else float("nan"),
                "components": ",".join(str(c) for c in sorted(ids)),
            }
        )

    table = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(str(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False)

    if verbose:
        print("\n" + "=" * 78)
        print("MAJOR-CLUSTER ROBUSTNESS ACROSS MERGE THRESHOLDS")
        print("=" * 78)
        for row in rows:
            mark = "*" if row["is_main_analysis"] else " "
            print(
                f" {mark} threshold {row['merge_threshold']:>4.1f}: "
                f"{row['n_merged_clusters']:>3} clusters, major holds "
                f"{row['n_components']:>3} components / {row['n_samples']:>7,} samples "
                f"({row['sample_share']:.2%}), subset={row['is_subset_of_main']}, "
                f"jaccard={row['jaccard_vs_main']:.3f}"
            )
    return table
