"""Fit the reference-panel mixture and select the component count by BIC.

Searches a range of component counts, keeping the minimum-BIC model that leaves
no component empty, then refits on the full denoised panel. Search fits run
across a process pool; the per-fit audit trail is written by ``gmm_search_audit``.

Computation is float64. Under float32 the log-likelihood sum over ~180k samples
was sensitive to BLAS reduction order, so the search log differed between runs
even with the seed fixed, and with several EM restarts a last-bit difference
could flip which restart won.

Inputs
------
Denoised reference panel; eigenvalues for axis labels.

Outputs
-------
Clustered sample table, BIC search table, per-component summary, run summary
JSON, an overview figure, and the audit logs under ``tmp/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter
from sklearn.mixture import GaussianMixture
import time

from scripts.gmm_search_audit import GMMAuditLogger

from scripts.common import (
    COMPUTE_DTYPE,
    STORE_DTYPE,
    build_distinct_palette as _build_distinct_palette,
    format_pc_axis_label as _format_pc_axis_label,
    pc_index_from_col as _pc_index_from_col,
    pc_sort_key as _pc_sort_key,
    resolve_pc_columns as _resolve_pc_columns,
)


class GMMClusteringOutput(NamedTuple):
    """Container for fixed-PC GMM clustering results."""

    labels: np.ndarray
    probabilities: np.ndarray
    bic_table: pd.DataFrame
    reference_samples_with_cluster: pd.DataFrame
    cluster_summary: pd.DataFrame
    selected_pc_cols: list[str]
    summary: dict[str, Any]
    model: GaussianMixture


@dataclass(frozen=True)
class GMMConfig:
    """Configuration for fixed-PC GMM clustering.

    Parameters
    ----------
    fixed_n_pcs : int
        Number of leading PCs used for both search and final model.
    k_min : int
        Minimum number of mixture components to evaluate.
    k_max : int
        Maximum number of mixture components to evaluate (inclusive).
    use_zscale : bool
        Whether to z-score selected PCs before fitting.
    covariance_type : str
        Covariance type for GaussianMixture.
    n_init : int
        Number of initializations for each GMM fit.
    init_params : str
        Initialization method for GaussianMixture.
    reg_covar : float
        Non-negative regularization added to covariance diagonal.
    max_iter : int
        Maximum number of EM iterations for each fit.
    random_state : int
        Random seed for reproducibility.
    search_max_samples : int
        Maximum samples used during BIC search. Final fit still uses all rows.
    search_workers : int
        Number of parallel workers used during BIC search (futures-based).
        Set to 1 to disable parallelism.
    require_non_empty_clusters : bool
        If True, select best k from models without empty clusters only.
    output_dir : str
        Directory for plot/table/json outputs.
    save_plot : bool
        Save BIC curve + clustering overview plot.
    save_tables : bool
        Save tables and JSON summary.
    verbose : bool
        Print structured logs.
    """

    fixed_n_pcs: int = 5
    k_min: int = 2
    k_max: int = 15
    use_zscale: bool = False
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    n_init: int = 10
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "k-means++"
    reg_covar: float = 1e-7
    max_iter: int = 500
    random_state: int = 42
    search_max_samples: int = 60000
    search_workers: int = 4
    require_non_empty_clusters: bool = True
    output_dir: str = "results/01_reference_model/mixture_model"
    save_plot: bool = True
    save_tables: bool = True
    verbose: bool = True


_GMM_SEARCH_X: np.ndarray | None = None
_GMM_SEARCH_PARAMS: dict[str, Any] | None = None


def _init_gmm_search_worker(x_search: np.ndarray, params: dict[str, Any]) -> None:
    """Initializer for ProcessPool workers to avoid re-pickling x_search per task."""

    global _GMM_SEARCH_X, _GMM_SEARCH_PARAMS
    _GMM_SEARCH_X = x_search
    _GMM_SEARCH_PARAMS = params


def _fit_gmm_search_for_k(k: int) -> dict[str, Any]:
    """Fit one GMM for a given k using globals set by _init_gmm_search_worker."""

    if _GMM_SEARCH_X is None or _GMM_SEARCH_PARAMS is None:
        raise RuntimeError("GMM search worker not initialized. This is a bug.")

    x_search = _GMM_SEARCH_X
    params = _GMM_SEARCH_PARAMS

    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=int(k),
        covariance_type=cast(Literal["full", "tied", "diag", "spherical"], params["covariance_type"]),
        random_state=int(params["random_state"]),
        n_init=int(params["n_init"]),
        init_params=cast(
            Literal["kmeans", "k-means++", "random", "random_from_data"],
            params["init_params"],
        ),
        reg_covar=float(params["reg_covar"]),
        max_iter=int(params["max_iter"]),
    )
    model.fit(x_search)
    labels_search = model.predict(x_search).astype(np.int32, copy=False)
    n_non_empty = int(np.unique(labels_search).shape[0])
    has_empty = n_non_empty < int(k)
    bic = float(model.bic(x_search))
    aic = float(model.aic(x_search))
    elapsed = float(time.perf_counter() - t0)

    return {
        "n_components": int(k),
        "bic": float(bic),
        "aic": float(aic),
        "has_empty_clusters": bool(has_empty),
        "n_non_empty_clusters": int(n_non_empty),
        "elapsed_seconds": float(elapsed),
        "model": model,
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    means = x.mean(axis=0, dtype=np.float64)
    stds = x.std(axis=0, dtype=np.float64)
    stds[stds == 0] = 1.0
    return ((x - means) / stds).astype(COMPUTE_DTYPE)


def _build_gmm(n_components: int, config: GMMConfig) -> GaussianMixture:
    """Create GMM model from one shared parameter set.

    This builder is used by both model search and final model extraction,
    ensuring strict parameter consistency.
    """

    return GaussianMixture(
        n_components=n_components,
        covariance_type=config.covariance_type,
        random_state=config.random_state,
        n_init=config.n_init,
        init_params=config.init_params,
        reg_covar=config.reg_covar,
        max_iter=config.max_iter,
    )


def _plot_gmm_overview(
    x: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    bic_table: pd.DataFrame,
    best_k: int,
    selected_pc_cols: list[str],
    output_path: Path,
    eigenval: pd.DataFrame | None,
) -> None:
    if x.shape[1] < 2:
        return

    # Match the multi-panel publication layout used in demo.ipynb cell 5.
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
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def run_gmm_fixed_pcs(
    reference_samples_filtered: pd.DataFrame,
    eigenval: pd.DataFrame | None = None,
    config: GMMConfig | None = None,
) -> GMMClusteringOutput:
    """Run fixed-PC GMM search and return clustering outputs.

    Workflow:
    1) Select fixed number of leading PCs.
    2) Optionally z-scale selected PCs.
    3) Search k in [k_min, k_max] using BIC.
    4) Choose model with minimum BIC (single supported strategy).
    5) Export summary/plot/tables.
    """

    config = config or GMMConfig()

    if reference_samples_filtered.empty:
        raise ValueError("bbj_samples_filtered is empty; cannot run GMM clustering.")

    pc_cols = _resolve_pc_columns(reference_samples_filtered)
    if not pc_cols:
        raise ValueError("No PC columns found in bbj_samples_filtered.")

    n_pcs = min(max(2, int(config.fixed_n_pcs)), len(pc_cols))
    selected_pc_cols = pc_cols[:n_pcs]

    x = reference_samples_filtered[selected_pc_cols].to_numpy(dtype=COMPUTE_DTYPE, copy=True)
    if config.use_zscale:
        x = _standardize(x)

    if config.search_max_samples < 1000:
        raise ValueError("search_max_samples should be >= 1000 for stable BIC search.")

    x_search = x
    if x.shape[0] > config.search_max_samples:
        rng = np.random.default_rng(config.random_state)
        idx = rng.choice(x.shape[0], size=int(config.search_max_samples), replace=False)
        x_search = x[idx]

    if config.k_min < 2 or config.k_max < config.k_min:
        raise ValueError("Invalid k range. Require 2 <= k_min <= k_max.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = GMMAuditLogger(
        output_dir=output_dir,
        config_dict={
            "fixed_n_pcs": int(config.fixed_n_pcs),
            "k_min": int(config.k_min),
            "k_max": int(config.k_max),
            "use_zscale": bool(config.use_zscale),
            "covariance_type": str(config.covariance_type),
            "n_init": int(config.n_init),
            "init_params": str(config.init_params),
            "reg_covar": float(config.reg_covar),
            "max_iter": int(config.max_iter),
            "random_state": int(config.random_state),
            "search_max_samples": int(config.search_max_samples),
            "search_workers": int(config.search_workers),
            "require_non_empty_clusters": bool(config.require_non_empty_clusters),
        },
        selected_pc_cols=selected_pc_cols,
        n_rows_full=int(x.shape[0]),
        n_rows_search=int(x_search.shape[0]),
        random_state=int(config.random_state),
        use_zscale=bool(config.use_zscale),
    )

    k_values = list(range(int(config.k_min), int(config.k_max) + 1))
    search_rows: list[dict[str, Any]] = []

    use_parallel = int(getattr(config, "search_workers", 1)) > 1 and len(k_values) > 1
    if use_parallel:
        search_params = {
            "covariance_type": str(config.covariance_type),
            "random_state": int(config.random_state),
            "n_init": int(config.n_init),
            "init_params": str(config.init_params),
            "reg_covar": float(config.reg_covar),
            "max_iter": int(config.max_iter),
        }

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=int(config.search_workers),
            initializer=_init_gmm_search_worker,
            initargs=(x_search, search_params),
        ) as ex:
            fut_map = {ex.submit(_fit_gmm_search_for_k, int(k)): int(k) for k in k_values}
            for fut in as_completed(fut_map):
                results.append(fut.result())

        # Keep deterministic output/log order.
        results.sort(key=lambda r: int(r["n_components"]))
        for row in results:
            k = int(row["n_components"])
            bic = float(row["bic"])
            aic = float(row["aic"])
            has_empty = bool(row["has_empty_clusters"])
            n_non_empty = int(row["n_non_empty_clusters"])
            elapsed = float(row["elapsed_seconds"])
            model = row["model"]

            search_rows.append(
                {
                    "n_components": int(k),
                    "n_pcs": int(n_pcs),
                    "bic": bic,
                    "aic": aic,
                    "has_empty_clusters": bool(has_empty),
                    "n_non_empty_clusters": int(n_non_empty),
                }
            )
            audit.log_search_iteration(
                k=int(k),
                model=model,
                x_used=x_search,
                bic=float(bic),
                aic=float(aic),
                elapsed_seconds=float(elapsed),
                has_empty_clusters=bool(has_empty),
                n_non_empty_clusters=int(n_non_empty),
            )
    else:
        for k in k_values:
            t0 = time.perf_counter()
            model = _build_gmm(n_components=k, config=config)
            model.fit(x_search)
            labels_search = model.predict(x_search).astype(np.int32, copy=False)
            n_non_empty = int(np.unique(labels_search).shape[0])
            has_empty = n_non_empty < int(k)
            bic = float(model.bic(x_search))
            aic = float(model.aic(x_search))
            elapsed = float(time.perf_counter() - t0)
            search_rows.append(
                {
                    "n_components": int(k),
                    "n_pcs": int(n_pcs),
                    "bic": bic,
                    "aic": aic,
                    "has_empty_clusters": bool(has_empty),
                    "n_non_empty_clusters": int(n_non_empty),
                }
            )
            audit.log_search_iteration(
                k=int(k),
                model=model,
                x_used=x_search,
                bic=float(bic),
                aic=float(aic),
                elapsed_seconds=elapsed,
                has_empty_clusters=bool(has_empty),
                n_non_empty_clusters=int(n_non_empty),
            )

    bic_table = pd.DataFrame(search_rows).sort_values("n_components").reset_index(drop=True)
    if config.require_non_empty_clusters:
        eligible_table = bic_table.loc[~bic_table["has_empty_clusters"]].copy()
        if eligible_table.empty:
            raise RuntimeError(
                "No non-empty-cluster model found in search range. "
                "Try reducing k_max or increasing search_max_samples."
            )
    else:
        eligible_table = bic_table

    eligible_bic = eligible_table["bic"].to_numpy(dtype=np.float64)
    eligible_k = eligible_table["n_components"].to_numpy(dtype=np.int32)
    best_pos = int(np.argmin(eligible_bic))
    best_k = int(eligible_k[best_pos])
    best_bic = float(eligible_bic[best_pos])
    audit.log_best_selection(
        bic_table=bic_table,
        best_k=int(best_k),
        best_bic=float(best_bic),
        require_non_empty_clusters=bool(config.require_non_empty_clusters),
    )

    # Final model uses full data with the exact same parameter set as search.
    t_final = time.perf_counter()
    final_model = _build_gmm(n_components=best_k, config=config)
    final_model.fit(x)
    labels = final_model.predict(x).astype(np.int32, copy=False)
    final_elapsed = float(time.perf_counter() - t_final)
    audit.log_final_model(
        model=final_model,
        x_full=x,
        labels=labels,
        best_k=int(best_k),
        elapsed_seconds=final_elapsed,
    )
    audit.finalize()

    probabilities_raw = final_model.predict_proba(x)
    probabilities = np.asarray(probabilities_raw, dtype=STORE_DTYPE)

    reference_samples_with_cluster = reference_samples_filtered.copy()
    reference_samples_with_cluster["GMM_Cluster"] = labels

    cluster_summary = (
        pd.Series(labels, name="GMM_Cluster")
        .value_counts()
        .sort_index()
        .rename_axis("GMM_Cluster")
        .reset_index(name="Samples")
    )
    cluster_summary["Share(%)"] = cluster_summary["Samples"] / float(cluster_summary["Samples"].sum()) * 100.0

    summary: dict[str, Any] = {
        "input_rows": int(x.shape[0]),
        "pc_columns_used": selected_pc_cols,
        "fixed_n_pcs": int(n_pcs),
        "k_min": int(config.k_min),
        "k_max": int(config.k_max),
        "best_k": int(best_k),
        "best_bic": float(best_bic),
        "covariance_type": config.covariance_type,
        "n_init": int(config.n_init),
        "init_params": config.init_params,
        "reg_covar": float(config.reg_covar),
        "max_iter": int(config.max_iter),
        "random_state": int(config.random_state),
        "use_zscale": bool(config.use_zscale),
        "search_rows": int(x_search.shape[0]),
        "full_rows": int(x.shape[0]),
        "search_max_samples": int(config.search_max_samples),
        "require_non_empty_clusters": bool(config.require_non_empty_clusters),
        "search_non_empty_model_count": int((~bic_table["has_empty_clusters"]).sum()),
    }

    if config.save_plot:
        _plot_gmm_overview(
            x=x,
            labels=labels,
            probabilities=probabilities,
            bic_table=bic_table,
            best_k=best_k,
            selected_pc_cols=selected_pc_cols,
            output_path=output_dir / "mixture_model_overview.png",
            eigenval=eigenval,
        )

    if config.save_tables:
        bic_table.to_csv(output_dir / "bic_search.tsv", sep="\t", index=False)
        reference_samples_with_cluster.to_csv(output_dir / "reference_samples_clustered.tsv", sep="\t", index=False)
        cluster_summary.to_csv(output_dir / "component_summary.tsv", sep="\t", index=False)
        with (output_dir / "mixture_model_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if config.verbose:
        print("\n" + "=" * 80)
        print("MIXTURE MODEL (FIXED PCs, MIN-BIC SELECTION)".center(80))
        print("=" * 80)
        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  fixed_n_pcs           : {n_pcs}")
        print(f"  k_range               : {config.k_min}..{config.k_max}")
        print(f"  covariance_type       : {config.covariance_type}")
        print(f"  n_init                : {config.n_init}")
        print(f"  init_params           : {config.init_params}")
        print(f"  reg_covar             : {config.reg_covar}")
        print(f"  max_iter              : {config.max_iter}")
        print(f"  random_state          : {config.random_state}")
        print(f"  search_rows           : {x_search.shape[0]:,}")
        print(f"  full_rows             : {x.shape[0]:,}")
        print(f"  search_workers        : {config.search_workers}")
        print(f"  require_non_empty     : {config.require_non_empty_clusters}")
        print(f"  use_zscale            : {config.use_zscale}")
        print(f"  output_dir            : {output_dir}")
        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  input_rows            : {summary['input_rows']:,}")
        print(f"  best_k                : {summary['best_k']}")
        print(f"  best_bic              : {summary['best_bic']:,.2f}")
        print(f"  non_empty_models      : {summary['search_non_empty_model_count']}")
        print(f"  clusters_found        : {cluster_summary.shape[0]}")
        print("=" * 80)

    return GMMClusteringOutput(
        labels=labels,
        probabilities=probabilities,
        bic_table=bic_table,
        reference_samples_with_cluster=reference_samples_with_cluster,
        cluster_summary=cluster_summary,
        selected_pc_cols=selected_pc_cols,
        summary=summary,
        model=final_model,
    )
