"""Denoise the reference panel in PC space before mixture modelling.

HDBSCAN separates dense population structure from sparse outliers, and only the
non-noise samples go downstream. This matters because a Gaussian mixture will
otherwise spend components describing scatter.

Pinned to python-hdbscan and errors if it is unavailable: the alternative
implementation disagrees on the noise set, and silently substituting it would
change the cohort while the summary file still reported the requested
parameters. Every configured parameter is verified against the constructed
estimator for the same reason.

Inputs
------
Reference panel samples with PC columns; eigenvalues for axis labels.

Outputs
-------
Denoised sample table, run summary JSON, and a three-panel QC figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from scripts.common import (
    COMPUTE_DTYPE,
    STORE_DTYPE,
    format_pc_axis_label as _format_pc_axis_label,
    resolve_pc_columns as _resolve_pc_columns,
)


class HDBSCANDenoiseOutput(NamedTuple):
    """Container for HDBSCAN denoising outputs."""

    # Lightweight table for plotting/auditing only (PCs + labels + noise flag).
    reference_samples_hdbscan: pd.DataFrame
    # Main output used downstream: non-noise BBJ samples with cluster label.
    reference_samples_filtered: pd.DataFrame
    labels: np.ndarray
    probabilities: np.ndarray
    selected_pc_cols: list[str]
    summary: dict[str, Any]


@dataclass(frozen=True)
class HDBSCANConfig:
    """
    Configuration for HDBSCAN denoising in PCA space.

    Parameters
    ----------
    n_pcs_hdbscan : int
        Number of leading PCs used as clustering features.
    use_zscale_hdbscan : bool
        Whether to z-score PCs before clustering.
    min_cluster_size : int
        Minimum cluster size. Larger values suppress small islands.
    min_samples : int
        Local density strictness. Larger values increase noise assignment.
    cluster_selection_epsilon : float
        Merge tolerance in cluster selection; larger values can reduce
        over-fragmentation by merging nearby dense regions.
    cluster_selection_method : str
        "eom" tends to produce more stable/coarse clusters;
        "leaf" tends to produce finer clusters.
    metric : str
        Distance metric used by HDBSCAN.
    alpha : float
        Scaling factor for robust single-linkage distances.
    allow_single_cluster : bool
        If True, allows one dominant cluster when structure is very compact.
    leaf_size : int
        Tree leaf size for neighbor search structures.
    algorithm : str
        Neighbor-search algorithm.
    approx_min_span_tree : bool
        Use approximate minimum spanning tree for speed.
    gen_min_span_tree : bool
        Generate and keep minimum spanning tree data.
    output_dir : str
        Directory where outputs are saved.
    save_plot : bool
        Save HDBSCAN overview figure.
    save_tables : bool
        Save non-noise sample table and summary JSON.
    verbose : bool
        Print structured runtime logs.
    """

    n_pcs_hdbscan: int = 2
    use_zscale_hdbscan: bool = True
    min_cluster_size: int = 25
    min_samples: int = 7
    cluster_selection_epsilon: float = 0.002
    cluster_selection_method: str = "eom"
    metric: str = "euclidean"
    alpha: float = 1.0
    allow_single_cluster: bool = False
    leaf_size: int = 40
    algorithm: str = "best"
    approx_min_span_tree: bool = True
    gen_min_span_tree: bool = False
    output_dir: str | Path = "results/01_reference_model/denoising"
    save_plot: bool = True
    save_tables: bool = True
    verbose: bool = True


def _standardize_inplace(x: np.ndarray) -> None:
    means = x.mean(axis=0, dtype=np.float64)
    stds = x.std(axis=0, dtype=np.float64)
    stds[stds == 0] = 1.0
    x -= means
    x /= stds


#: The one supported HDBSCAN backend. This pipeline is pinned to
#: python-hdbscan (see environment.yml) rather than accepting whichever
#: implementation happens to be importable.
#:
#: There used to be a fallback to sklearn.cluster.HDBSCAN behind a bare
#: `except Exception`. That was a trap: the two backends do not agree on the
#: noise set, sklearn silently ignores approx_min_span_tree/gen_min_span_tree
#: and needs algorithm="auto" where python-hdbscan wants "best" -- and any
#: exception at all, not just a missing package, would switch implementations
#: without a word. Worse, hdbscan_summary.json reports config values rather
#: than what the estimator received, so the audit trail would have asserted
#: parameters that were never applied. A reproducer would silently get a
#: different cohort. Now it is an error.
HDBSCAN_BACKEND = "hdbscan (python-hdbscan)"


def _make_hdbscan_estimator(config: HDBSCANConfig, metric_kwargs: dict[str, Any] | None = None):
    """Build the pinned python-hdbscan estimator.

    Raises ImportError if python-hdbscan is unavailable, and ValueError if it
    would not accept one of the configured parameters -- silently dropping a
    parameter is the same class of bug as silently swapping the backend.
    """

    model_kwargs: dict[str, Any] = {
        "min_cluster_size": config.min_cluster_size,
        "min_samples": config.min_samples,
        "cluster_selection_epsilon": config.cluster_selection_epsilon,
        "cluster_selection_method": config.cluster_selection_method,
        "metric": config.metric,
        "alpha": config.alpha,
        "allow_single_cluster": config.allow_single_cluster,
        "leaf_size": config.leaf_size,
        "algorithm": config.algorithm,
        "approx_min_span_tree": config.approx_min_span_tree,
        "gen_min_span_tree": config.gen_min_span_tree,
    }

    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-hdbscan is required and is not installed. This pipeline is "
            "pinned to it (environment.yml: hdbscan==0.8.42); "
            "sklearn.cluster.HDBSCAN is NOT an acceptable substitute because it "
            "produces a different noise set. Install it with "
            "`pip install hdbscan==0.8.42`."
        ) from exc

    kwargs = dict(model_kwargs)
    if metric_kwargs:
        kwargs.update(metric_kwargs)
    if kwargs.get("algorithm") == "auto":
        # python-hdbscan expects values like "best", not "auto".
        kwargs["algorithm"] = "best"
    kwargs["prediction_data"] = False
    kwargs["core_dist_n_jobs"] = 1

    estimator = hdbscan.HDBSCAN(**kwargs)

    # hdbscan.HDBSCAN takes **kwargs, so inspecting its signature can never
    # reveal a dropped parameter -- an unrecognised keyword is swallowed
    # silently. Verify against the constructed object instead: every configured
    # value must actually have landed on it.
    missing = [
        f"{key}={value!r} (estimator has {getattr(estimator, key, '<absent>')!r})"
        for key, value in kwargs.items()
        if getattr(estimator, key, object()) != value
    ]
    if missing:
        raise ValueError(
            f"Installed hdbscan "
            f"{getattr(hdbscan, '__version__', '(version not exposed)')} did not "
            f"apply these configured parameters: {missing}. Silently ignoring them "
            f"would change the denoising result, so this is an error. Pin "
            f"hdbscan==0.8.42 or update HDBSCANConfig to match the installed API."
        )
    return estimator


def _plot_hdbscan_overview(
    reference_samples_hdbscan: pd.DataFrame,
    selected_pc_cols: list[str],
    output_path: Path,
    eigenval: pd.DataFrame | None = None,
) -> None:
    """Generate a compact 3-panel QC plot for HDBSCAN results.

    Visualization is intentionally concise: all points, noise map,
    and non-noise cluster view.
    """

    if len(selected_pc_cols) < 2:
        return

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

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.edgecolor": "#3A3A3A",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#C9CED6",
            "grid.linestyle": "--",
            "grid.alpha": 0.40,
            "legend.frameon": True,
            "legend.framealpha": 0.97,
            "legend.edgecolor": "#C3C7CE",
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
        }
    )
    sns.set_context("talk", font_scale=1.08)
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
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_hdbscan_denoise(
    reference_samples: pd.DataFrame,
    eigenval: pd.DataFrame | None = None,
    config: HDBSCANConfig | None = None,
) -> HDBSCANDenoiseOutput:
    """Run HDBSCAN on BBJ samples and return denoised outputs.

    Processing steps:
    1) Resolve available PC columns and select leading PCs.
    2) Optional z-score normalization.
    3) Fit HDBSCAN and obtain labels/noise mask.
    4) Keep only non-noise BBJ samples for downstream usage.
    5) Persist summary/table/plot according to config switches.
    """

    config = config or HDBSCANConfig()

    if reference_samples.empty:
        raise ValueError("bbj_samples is empty; cannot run HDBSCAN denoising.")

    pc_cols = _resolve_pc_columns(reference_samples)
    if not pc_cols:
        raise ValueError("No PC columns were found in bbj_samples.")

    n_pcs = min(config.n_pcs_hdbscan, len(pc_cols))
    selected_pc_cols = pc_cols[:n_pcs]

    x = reference_samples[selected_pc_cols].to_numpy(dtype=COMPUTE_DTYPE, copy=True)
    if config.use_zscale_hdbscan:
        _standardize_inplace(x)

    metric_kwargs: dict[str, Any] | None = None
    if str(config.metric).lower() == "mahalanobis":
        cov = np.cov(x, rowvar=False).astype(np.float64, copy=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        # Small diagonal jitter improves numerical stability for near-singular covariance.
        cov = cov + (1e-8 * np.eye(cov.shape[0], dtype=np.float64))
        metric_kwargs = {"V": cov, "VI": np.linalg.pinv(cov)}

    estimator = _make_hdbscan_estimator(config, metric_kwargs=metric_kwargs)
    labels = estimator.fit_predict(x).astype(np.int32, copy=False)

    probabilities_raw = getattr(estimator, "probabilities_", None)
    if probabilities_raw is None:
        probabilities = np.full(labels.shape[0], np.nan, dtype=STORE_DTYPE)
    else:
        probabilities = np.asarray(probabilities_raw, dtype=STORE_DTYPE)

    noise_mask = labels == -1
    keep_mask = ~noise_mask

    # Keep only plotting-required columns to avoid copying the full table in memory.
    viz_cols = list(selected_pc_cols)
    reference_samples_hdbscan = reference_samples.loc[:, viz_cols].copy()
    reference_samples_hdbscan["HDBSCAN_Label"] = labels
    reference_samples_hdbscan["HDBSCAN_IsNoise"] = noise_mask

    # Main output: retain full original columns for non-noise BBJ samples.
    reference_samples_filtered = reference_samples.loc[keep_mask].copy()
    reference_samples_filtered["HDBSCAN_Label"] = labels[keep_mask]

    n_total = int(labels.shape[0])
    n_noise = int(noise_mask.sum())
    unique_clusters = np.unique(labels[keep_mask]) if keep_mask.any() else np.array([], dtype=np.int32)

    summary: dict[str, Any] = {
        "input_rows": n_total,
        "output_rows": int(reference_samples_filtered.shape[0]),
        "noise_rows": n_noise,
        "noise_ratio": (n_noise / n_total) if n_total > 0 else 0.0,
        "clusters_found": int(unique_clusters.shape[0]),
        "pc_columns_used": selected_pc_cols,
        "n_pcs_hdbscan": n_pcs,
        "min_cluster_size": config.min_cluster_size,
        "min_samples": config.min_samples,
        "cluster_selection_epsilon": config.cluster_selection_epsilon,
        "cluster_selection_method": config.cluster_selection_method,
        "metric": config.metric,
        "alpha": config.alpha,
        "allow_single_cluster": config.allow_single_cluster,
        "leaf_size": config.leaf_size,
        "algorithm": config.algorithm,
        "approx_min_span_tree": config.approx_min_span_tree,
        "gen_min_span_tree": config.gen_min_span_tree,
        "use_zscale_hdbscan": config.use_zscale_hdbscan,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.save_plot:
        _plot_hdbscan_overview(
            reference_samples_hdbscan=reference_samples_hdbscan,
            selected_pc_cols=selected_pc_cols,
            output_path=output_dir / "denoising_overview.png",
            eigenval=eigenval,
        )

    if config.save_tables:
        # Only write the non-noise table with cluster label.
        reference_samples_filtered.to_csv(output_dir / "reference_samples_denoised.tsv", sep="\t", index=False)
        with (output_dir / "denoising_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if config.verbose:
        print("\n" + "=" * 80)
        print("HDBSCAN DENOISING (REFERENCE PANEL)".center(80))
        print("=" * 80)
        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  n_pcs_hdbscan         : {n_pcs}")
        print(f"  min_cluster_size      : {config.min_cluster_size}")
        print(f"  min_samples           : {config.min_samples}")
        print(f"  cluster_epsilon       : {config.cluster_selection_epsilon}")
        print(f"  cluster_method        : {config.cluster_selection_method}")
        print(f"  metric                : {config.metric}")
        print(f"  alpha                 : {config.alpha}")
        print(f"  allow_single_cluster  : {config.allow_single_cluster}")
        print(f"  leaf_size             : {config.leaf_size}")
        print(f"  algorithm             : {config.algorithm}")
        print(f"  approx_min_span_tree  : {config.approx_min_span_tree}")
        print(f"  gen_min_span_tree     : {config.gen_min_span_tree}")
        print(f"  use_zscale            : {config.use_zscale_hdbscan}")
        print(f"  save_plot             : {config.save_plot}")
        print(f"  output_dir            : {output_dir}")
        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  input_rows            : {summary['input_rows']:,}")
        print(f"  output_rows           : {summary['output_rows']:,}")
        print(f"  noise_rows            : {summary['noise_rows']:,}")
        print(f"  noise_ratio           : {summary['noise_ratio']:.2%}")
        print(f"  clusters_found        : {summary['clusters_found']}")
        print("=" * 80)

    return HDBSCANDenoiseOutput(
        reference_samples_hdbscan=reference_samples_hdbscan,
        reference_samples_filtered=reference_samples_filtered,
        labels=labels,
        probabilities=probabilities,
        selected_pc_cols=selected_pc_cols,
        summary=summary,
    )
