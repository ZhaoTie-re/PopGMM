from __future__ import annotations

"""Assign study cohort samples to merged GMM clusters.

Author: ZHAO TIE

Purpose
-------
This module projects study cohort samples into a trained BBJ GMM, aggregates
component posterior probabilities into merged clusters (from STEP3), saves a
posterior TSV, and generates a 2x2 composite figure:
A) Study cohort overlay (case/control)
B) Assigned merged cluster (colors sourced from STEP3 merge_map)
C) Assignment confidence
D) Case/control/total table by cluster

Design goals
------------
- Single-entry function with a dataclass config (mirrors other STEP scripts).
- Plot colors for merged clusters are reproducible by reading
  merge_map['Merged_Cluster_Color'].
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from sklearn.mixture import GaussianMixture


class OURAssignmentOutput(NamedTuple):
    """Container for OUR assignment outputs."""

    df_results: pd.DataFrame
    probs_merged_our: np.ndarray
    assigned_merged: np.ndarray
    assignment_confidence: np.ndarray
    cluster_stats: pd.DataFrame
    output_dir: Path
    assignment_tsv: Path | None
    figure_path: Path | None


class OURAssignmentConfidenceQCOutput(NamedTuple):
    """Container for STEP4 assignment confidence QC outputs."""

    confidence_scores: pd.Series
    summary: pd.Series
    high_conf_counts: dict[float, int]
    output_dir: Path
    figure_path: Path | None


@dataclass(frozen=True)
class OURAssignmentConfig:
    """Configuration for OUR cohort assignment to merged GMM clusters.

    Parameters
    ----------
    output_dir : str
        Output directory for STEP4 results.
    save_plot : bool
        Save 2x2 composite figure.
    show_plot : bool
        Show figure (useful in interactive sessions).
    save_tables : bool
        Save TSV outputs.
    output_file : str
        Filename for posterior TSV.
    figure_file : str
        Filename for composite figure.
    bbj_color : str
        Background color for BBJ points.
    bbj_alpha : float
        Alpha for BBJ background points.
    our_point_size : float
        Point size for OUR points.
    case_label : str
        Display label for case samples in the figure/table.
    control_label : str
        Display label for control samples in the figure/table.
    verbose : bool
        Print structured logs.
    """

    output_dir: str = "results/04_our_assignment"
    save_plot: bool = True
    show_plot: bool = False
    save_tables: bool = True

    output_file: str = "our_posterior_probabilities_merged.tsv"
    figure_file: str = "our_assignment_2x2.png"

    bbj_color: str = "#B0B0B0"
    bbj_alpha: float = 0.20
    our_point_size: float = 60.0

    case_label: str = "Case"
    control_label: str = "Control"

    verbose: bool = True


@dataclass(frozen=True)
class OURAssignmentConfidenceQCConfig:
    """Configuration for STEP4 assignment confidence QC.

    Parameters
    ----------
    output_dir : str | None
        Output directory for QC figure. If None, uses STEP4 output_dir.
    figure_file : str
        Filename for QC figure.
    thresholds : tuple[float, ...]
        Confidence thresholds to annotate on the CDF plot.
    save_plot : bool
        Save QC figure.
    show_plot : bool
        Show QC figure.
    verbose : bool
        Print summary stats and counts.
    """

    output_dir: str | None = None
    figure_file: str = "step4_assignment_confidence_distribution.png"
    thresholds: tuple[float, ...] = (0.80, 0.90, 0.95)
    save_plot: bool = True
    show_plot: bool = False
    verbose: bool = True


_PLOT_STYLE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.title_fontsize": 24,
    "legend.fontsize": 22,
    "figure.titlesize": 40,
    "figure.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
}


def run_step4_assignment_confidence_qc(
    *,
    df_results: pd.DataFrame | None = None,
    step4_config: OURAssignmentConfig | None = None,
    qc_config: OURAssignmentConfidenceQCConfig | None = None,
) -> OURAssignmentConfidenceQCOutput:
    """Generate a STEP4 QC figure for assignment confidence (KDE + CDF) and print summary stats.

    This mirrors the demo.ipynb confidence-distribution analysis, but is adapted
    to PopGMM STEP4 outputs.
    """

    step4_config = step4_config or OURAssignmentConfig()
    qc_config = qc_config or OURAssignmentConfidenceQCConfig()

    # Resolve df_results.
    if df_results is None:
        out_dir_step4 = Path(str(step4_config.output_dir))
        input_path = out_dir_step4 / str(step4_config.output_file)
        if not input_path.exists():
            raise FileNotFoundError(
                "df_results is None and STEP4 output TSV does not exist:\n"
                f"{input_path}\n"
                "Please run STEP4 first or pass df_results."
            )
        df_results = pd.read_csv(input_path, sep="\t")

    if "Assignment_Confidence" not in df_results.columns:
        raise KeyError("df_results must contain 'Assignment_Confidence'.")

    confidence_scores = pd.to_numeric(df_results["Assignment_Confidence"], errors="coerce").dropna()
    if confidence_scores.empty:
        raise ValueError("No valid Assignment_Confidence values found.")

    out_dir = Path(str(qc_config.output_dir or step4_config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_path = out_dir / str(qc_config.figure_file) if bool(qc_config.save_plot) else None

    if bool(qc_config.save_plot) or bool(qc_config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("poster", font_scale=1.2)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": 22,
                "axes.titlesize": 28,
                "axes.labelsize": 26,
                "xtick.labelsize": 22,
                "ytick.labelsize": 22,
                "legend.title_fontsize": 22,
                "legend.fontsize": 20,
                "figure.titlesize": 34,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(28, 12))
        fig.subplots_adjust(wspace=0.25, left=0.08, right=0.95, top=0.84, bottom=0.15)

        # A: KDE
        ax1 = axes[0]
        sns.kdeplot(x=confidence_scores.to_numpy(), color="#4C72B0", fill=True, alpha=0.5, linewidth=4, ax=ax1)
        ax1.set_title("A. Distribution of Confidence", loc="left", pad=30, fontweight="bold")
        ax1.set_xlabel("Max Posterior Probability (Confidence)", labelpad=20)
        ax1.set_ylabel("Density", labelpad=20)
        ax1.grid(True, linestyle="--", alpha=0.4, color="#bbbbbb", linewidth=1.5)

        # B: CDF
        ax2 = axes[1]
        sns.ecdfplot(x=confidence_scores.to_numpy(), color="#8172B2", linewidth=6, ax=ax2)
        ax2.set_title("B. Cumulative Distribution (CDF)", loc="left", pad=30, fontweight="bold")
        ax2.set_xlabel("Max Posterior Probability (Confidence)", labelpad=20)
        ax2.set_ylabel("Cumulative Proportion", labelpad=20)
        ax2.grid(True, linestyle="--", alpha=0.4, color="#bbbbbb", linewidth=1.5)

        thresholds = tuple(float(t) for t in qc_config.thresholds)
        if len(thresholds) <= 3:
            colors = ["#CCB974", "#64B5CD", "#C44E52"][: len(thresholds)]
        else:
            colors = sns.color_palette("tab10", n_colors=len(thresholds)).as_hex()

        for i, thresh in enumerate(thresholds):
            prop_below = float((confidence_scores < thresh).mean())
            prop_above = 1.0 - prop_below
            ax2.axvline(thresh, color=colors[i], linestyle=":", linewidth=3, alpha=0.8)
            ax2.axhline(prop_below, color=colors[i], linestyle=":", linewidth=3, alpha=0.8)
            ax2.annotate(
                f"≥{thresh:.2f}: {prop_above:.1%}",
                xy=(thresh, prop_below),
                xytext=(thresh - 0.15, min(1.0, prop_below + (0.07 * (i + 1)))),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=3),
                fontsize=22,
                color=colors[i],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors[i], alpha=0.95, lw=2),
            )

        fig.suptitle(
            "Assignment Quality Metrics",
            fontsize=34,
            fontweight="bold",
            y=0.96,
            ha="center",
        )

        if figure_path is not None:
            fig.savefig(figure_path, bbox_inches="tight", dpi=400)

        if bool(qc_config.show_plot):
            plt.show()
        else:
            plt.close(fig)

    summary = confidence_scores.describe()

    report_thresholds = tuple(sorted(set([*qc_config.thresholds, 0.99])))
    high_conf_counts = {float(t): int((confidence_scores >= float(t)).sum()) for t in report_thresholds}

    if bool(qc_config.verbose):
        print("\n" + "=" * 80)
        print("STEP4 QC: CONFIDENCE STATISTICS SUMMARY".center(80))
        print("=" * 80)
        n_total = int(len(confidence_scores))
        keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for k in keys:
            if k == "count":
                print(f"{k:<6}{n_total:>12d}")
            else:
                print(f"{k:<6}{float(summary[k]):>12.6f}")
        print("-" * 80)
        print("High Confidence Sample Counts:")
        for t in report_thresholds:
            count = int(high_conf_counts[float(t)])
            pct = (count / n_total) * 100.0 if n_total else 0.0
            print(f"  • >= {float(t):.2f} : {count:5d} samples ({pct:.1f}%)")
        print("=" * 80)

    return OURAssignmentConfidenceQCOutput(
        confidence_scores=confidence_scores,
        summary=summary,
        high_conf_counts=high_conf_counts,
        output_dir=out_dir,
        figure_path=figure_path,
    )


def _resolve_pc_columns_for_projection(
    *,
    our_samples: pd.DataFrame,
    gmm_model: GaussianMixture,
    gmm_summary: dict[str, Any] | None,
) -> list[str]:
    pc_cols_used: list[str] = []

    if isinstance(gmm_summary, dict):
        pc_cols_used = [c for c in gmm_summary.get("pc_columns_used", []) if c in our_samples.columns]

    if not pc_cols_used:
        n_features = int(getattr(gmm_model, "n_features_in_", 2))
        pc_candidates = [c for c in our_samples.columns if str(c).startswith("PC") and str(c).endswith("_AVG")]

        def _pc_num(col: str) -> int:
            try:
                return int(str(col).split("_AVG")[0].replace("PC", ""))
            except Exception:
                return 10**9

        pc_cols_used = sorted(pc_candidates, key=_pc_num)[: max(2, n_features)]

    if len(pc_cols_used) < 2:
        raise RuntimeError("At least two PC columns are required for assignment visualization.")

    return list(pc_cols_used)


def _build_merged_cluster_palette(
    *,
    merge_map: pd.DataFrame | None,
    new_k: int,
    fallback_palette: str = "tab20",
) -> dict[int, str]:
    merged_cluster_palette: dict[int, str] = {}

    if isinstance(merge_map, pd.DataFrame) and ("Merged_Cluster" in merge_map.columns):
        if "Merged_Cluster_Color" in merge_map.columns:
            tmp = (
                merge_map[["Merged_Cluster", "Merged_Cluster_Color"]]
                .dropna()
                .drop_duplicates(subset=["Merged_Cluster"])
                .sort_values("Merged_Cluster")
            )
            merged_cluster_palette = {
                int(k): str(v) for k, v in zip(tmp["Merged_Cluster"].to_numpy(), tmp["Merged_Cluster_Color"].to_numpy())
            }

    # Ensure all clusters have a color.
    fallback_hex = sns.color_palette(str(fallback_palette), n_colors=int(new_k)).as_hex()
    for i in range(int(new_k)):
        merged_cluster_palette.setdefault(int(i), str(fallback_hex[int(i)]))

    return merged_cluster_palette


def _build_distinct_palette(n_colors: int) -> list[tuple[float, float, float, float]]:
    """Match STEP2 categorical palette construction for many clusters."""
    if n_colors <= 0:
        return []

    palette: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        for i in range(cmap.N):
            palette.append(cmap(i))

    if n_colors > len(palette):
        extra = n_colors - len(palette)
        hsv = plt.get_cmap("hsv")
        for i in range(extra):
            palette.append(hsv((i / max(1, extra)) % 1.0))

    return palette[:n_colors]


def _build_premerge_component_palette(
    *,
    bbj_samples_gmm: pd.DataFrame,
    n_clusters: int,
) -> dict[int, str]:
    """Build STEP2-consistent component colors for pre-merge assignment mode."""
    if "GMM_Cluster" in bbj_samples_gmm.columns:
        labels = bbj_samples_gmm["GMM_Cluster"].to_numpy(dtype=np.int32, copy=False)
        unique_labels = np.unique(labels).astype(int)
    else:
        unique_labels = np.arange(int(n_clusters), dtype=np.int32)

    palette = _build_distinct_palette(int(len(unique_labels)))
    color_map = {int(k): str(to_hex(palette[i])) for i, k in enumerate(unique_labels.tolist())}

    # Ensure every expected component has a color.
    if int(n_clusters) > len(unique_labels):
        fallback_hex = sns.color_palette("tab20", n_colors=int(n_clusters)).as_hex()
        for i in range(int(n_clusters)):
            color_map.setdefault(int(i), str(fallback_hex[int(i)]))

    return color_map


def _extract_mainland_premerge_cluster_ids(merge_map: pd.DataFrame | None) -> list[int]:
    """Extract mainland pre-merge component IDs when STEP3 metadata is available."""
    if not isinstance(merge_map, pd.DataFrame) or merge_map.empty:
        return []

    if "GMM_Component" not in merge_map.columns:
        return []

    if "Is_Mainland_Merged_Cluster" not in merge_map.columns:
        return []

    try:
        mask = merge_map["Is_Mainland_Merged_Cluster"].astype(bool)
    except Exception:
        return []

    vals = pd.to_numeric(merge_map.loc[mask, "GMM_Component"], errors="coerce").dropna().astype(int)
    if vals.empty:
        return []

    return sorted(set(int(v) for v in vals.tolist()))


def run_our_assignment_to_merged_gmm(
    *,
    gmm_model: GaussianMixture,
    bbj_samples_gmm: pd.DataFrame,
    our_samples: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    label_map: dict[int, int],
    merge_map: pd.DataFrame | None = None,
    eigenval: pd.DataFrame | None = None,
    gmm_summary: dict[str, Any] | None = None,
    training_use_zscale: bool = False,
    config: OURAssignmentConfig | None = None,
) -> OURAssignmentOutput:
    config = config or OURAssignmentConfig()

    if bool(training_use_zscale):
        raise RuntimeError(
            "training_use_zscale=True is not supported for projection without storing training scaler stats."
        )

    if our_samples.empty:
        raise ValueError("our_samples is empty; cannot run STEP4 assignment.")

    if "IID" not in our_samples.columns:
        raise ValueError("our_samples must contain an 'IID' column.")

    if not isinstance(label_map, dict) or len(label_map) == 0:
        raise ValueError("label_map is missing/empty; run STEP3 to obtain component->merged mapping.")

    pc_cols_used = _resolve_pc_columns_for_projection(our_samples=our_samples, gmm_model=gmm_model, gmm_summary=gmm_summary)

    x_our = our_samples[pc_cols_used].to_numpy(dtype=np.float64, copy=False)
    probs_original = gmm_model.predict_proba(x_our).astype(np.float64, copy=False)

    old_k = int(probs_original.shape[1])
    new_k = int(max(label_map.values())) + 1
    is_premerge_identity_mode = bool(
        int(new_k) == int(old_k)
        and all(int(label_map.get(int(i), -1)) == int(i) for i in range(int(old_k)))
    )

    probs_merged_our = np.zeros((probs_original.shape[0], int(new_k)), dtype=np.float64)
    for c_old in range(int(old_k)):
        probs_merged_our[:, int(label_map[int(c_old)])] += probs_original[:, c_old]

    row_sums = probs_merged_our.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_merged_our = probs_merged_our / row_sums

    assigned_merged = np.argmax(probs_merged_our, axis=1).astype(int)
    assignment_conf = np.max(probs_merged_our, axis=1).astype(np.float32)

    fid_col = "#FID" if "#FID" in our_samples.columns else ("FID" if "FID" in our_samples.columns else None)
    meta_cols = [c for c in [fid_col, "IID"] if c is not None and c in our_samples.columns]

    df_results = our_samples[meta_cols + pc_cols_used[:2]].copy() if meta_cols else our_samples[pc_cols_used[:2]].copy()
    if fid_col == "#FID":
        df_results = df_results.rename(columns={"#FID": "FID"})

    df_results["Assigned_Merged_Cluster"] = assigned_merged
    df_results["Assignment_Confidence"] = assignment_conf
    for m in range(int(new_k)):
        df_results[f"Prob_Merge_Cluster_{m}"] = probs_merged_our[:, m].astype(np.float32)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_tsv: Path | None = None
    if bool(config.save_tables):
        assignment_tsv = out_dir / str(config.output_file)
        df_results.to_csv(assignment_tsv, sep="\t", index=False)

    # ===== Plot =====
    figure_path: Path | None = None
    cluster_stats: pd.DataFrame = pd.DataFrame()

    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=2.5)
        plt.rcParams.update(dict(_PLOT_STYLE_RC))

        bbj_pc1 = bbj_samples_gmm[pc_cols_used[0]].to_numpy(dtype=np.float64, copy=False)
        bbj_pc2 = bbj_samples_gmm[pc_cols_used[1]].to_numpy(dtype=np.float64, copy=False)
        our_pc1 = our_samples[pc_cols_used[0]].to_numpy(dtype=np.float64, copy=False)
        our_pc2 = our_samples[pc_cols_used[1]].to_numpy(dtype=np.float64, copy=False)

        our_iid = our_samples["IID"].astype(str)
        case_set = set(str(x) for x in our_case_iids)
        ctrl_set = set(str(x) for x in our_ctrl_iids)
        is_case = our_iid.isin(case_set).to_numpy()
        is_ctrl = our_iid.isin(ctrl_set).to_numpy()
        is_other = ~(is_case | is_ctrl)

        try:
            var1 = float(eigenval.loc[eigenval["PC"] == 1, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
            var2 = float(eigenval.loc[eigenval["PC"] == 2, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
        except Exception:
            var1, var2 = 0.0, 0.0

        if is_premerge_identity_mode:
            merged_cluster_palette = _build_premerge_component_palette(
                bbj_samples_gmm=bbj_samples_gmm,
                n_clusters=int(new_k),
            )
        else:
            merged_cluster_palette = _build_merged_cluster_palette(merge_map=merge_map, new_k=int(new_k))
        hue_order = list(range(int(new_k)))

        all_x = np.concatenate([bbj_pc1, our_pc1])
        all_y = np.concatenate([bbj_pc2, our_pc2])
        x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
        y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        max_span = max(x_max - x_min, y_max - y_min)
        if max_span == 0:
            max_span = 1.0
        view_span = max_span * 1.05
        x_center = (x_max + x_min) / 2.0
        y_center = (y_max + y_min) / 2.0

        def _apply_equal_centered_limits(ax) -> None:
            ax.set_xlim(x_center - view_span / 2.0, x_center + view_span / 2.0)
            ax.set_ylim(y_center - view_span / 2.0, y_center + view_span / 2.0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
            ax.set_anchor("C")

        def _add_bbj_background(ax, label: str = "BBJ") -> None:
            ax.scatter(
                bbj_pc1,
                bbj_pc2,
                c=str(config.bbj_color),
                s=20,
                alpha=float(config.bbj_alpha),
                label=label,
                rasterized=True,
                zorder=0,
            )

        fig = plt.figure(figsize=(26, 24))
        fig.patch.set_facecolor("white")
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
        # Reserve figure margin for outside legends and the colorbar.
        fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.88, wspace=0.28, hspace=0.55)

        # A
        ax1 = fig.add_subplot(gs[0, 0])
        _add_bbj_background(ax1, label="BBJ")

        ctrl_color = "#1F78B4"
        case_color = "#E31A1C"
        other_color = "#33A02C"
        s_our = float(config.our_point_size)

        if bool(is_ctrl.any()):
            ax1.scatter(
                our_pc1[is_ctrl],
                our_pc2[is_ctrl],
                c=ctrl_color,
                s=s_our,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
                label=str(config.control_label),
                rasterized=True,
                zorder=2,
            )
        if bool(is_case.any()):
            ax1.scatter(
                our_pc1[is_case],
                our_pc2[is_case],
                c=case_color,
                s=s_our,
                alpha=0.90,
                edgecolor="white",
                linewidth=0.3,
                label=str(config.case_label),
                rasterized=True,
                zorder=3,
            )
        if bool(is_other.any()):
            ax1.scatter(
                our_pc1[is_other],
                our_pc2[is_other],
                c=other_color,
                s=s_our,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
                label="Other",
                rasterized=True,
                zorder=1,
            )

        _apply_equal_centered_limits(ax1)
        ax1.set_title("A. Study Cohort", loc="left", pad=25, fontweight="bold")
        ax1.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax1.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
        ax1.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
        ax1.tick_params(axis="both", which="major", length=4.8, width=1.1)
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color("#4A4A4A")
            spine.set_linewidth(1.1)
        handles_a, labels_a = ax1.get_legend_handles_labels()
        if "BBJ" in labels_a:
            bbj_handle_a = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=str(config.bbj_color),
                markeredgecolor="#4A4A4A",
                markeredgewidth=0.8,
                alpha=1.0,
                markersize=float(np.sqrt(s_our)),
            )
            handles_a = [bbj_handle_a if lab == "BBJ" else h for h, lab in zip(handles_a, labels_a)]

        ax1.legend(
            handles_a,
            labels_a,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            fontsize=18,
            markerscale=1.4,
        )

        # B
        ax2 = fig.add_subplot(gs[0, 1])
        _add_bbj_background(ax2, label="BBJ")

        df_plot_b = pd.DataFrame({"PC1": our_pc1, "PC2": our_pc2, "Cluster": assigned_merged.astype(int)})
        sns.scatterplot(
            data=df_plot_b,
            x="PC1",
            y="PC2",
            hue="Cluster",
            hue_order=hue_order,
            palette=merged_cluster_palette,
            s=s_our,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.3,
            ax=ax2,
            zorder=2,
        )

        _apply_equal_centered_limits(ax2)
        panel_b_title = "B. Assigned Pre-merge Cluster" if is_premerge_identity_mode else "B. Assigned Merged Cluster"
        ax2.set_title(panel_b_title, loc="left", pad=25, fontweight="bold")
        ax2.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax2.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
        ax2.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
        ax2.tick_params(axis="both", which="major", length=4.8, width=1.1)
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color("#4A4A4A")
            spine.set_linewidth(1.1)

        handles_b, labels_b = ax2.get_legend_handles_labels()
        bbj_handle_b = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=str(config.bbj_color),
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.8,
            alpha=1.0,
            markersize=float(np.sqrt(s_our)),
        )

        new_handles_b: list[Any] = []
        new_labels_b: list[str] = []
        for h, lab in zip(handles_b, labels_b):
            if lab == "BBJ":
                new_handles_b.append(bbj_handle_b)
                new_labels_b.append("BBJ")
            else:
                new_handles_b.append(h)
                try:
                    new_labels_b.append(f"Cluster {int(lab)}")
                except Exception:
                    new_labels_b.append(str(lab))

        ax2.legend(
            new_handles_b,
            new_labels_b,
            title=None,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fontsize=18,
            markerscale=1.4,
        )

        # C
        ax3 = fig.add_subplot(gs[1, 0])
        _add_bbj_background(ax3, label="BBJ")

        sort_idx = np.argsort(assignment_conf)[::-1]
        # Use the same colormap family as run_gmm_fixed_pcs (cividis), but
        # do not use the same percentile-based scaling.
        vmin = float(np.nanmin(assignment_conf))
        vmax = float(np.nanmax(assignment_conf))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = min(1.0, vmin + 1e-6)
        sc = ax3.scatter(
            our_pc1[sort_idx],
            our_pc2[sort_idx],
            c=assignment_conf[sort_idx],
            cmap="cividis",
            vmin=vmin,
            vmax=vmax,
            s=s_our,
            alpha=0.90,
            edgecolors="none",
            zorder=2,
        )

        _apply_equal_centered_limits(ax3)
        ax3.set_title("C. Assignment Confidence", loc="left", pad=25, fontweight="bold")
        ax3.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax3.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
        ax3.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
        ax3.tick_params(axis="both", which="major", length=4.8, width=1.1)
        for spine in ax3.spines.values():
            spine.set_visible(True)
            spine.set_color("#4A4A4A")
            spine.set_linewidth(1.1)

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
        cbar.set_label("Max Posterior Probability (Confidence)", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # D
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        ax4.set_title("D. Statistics by Cluster", loc="center", pad=40, fontweight="bold")

        df_cc = pd.DataFrame({"Cluster": assigned_merged.astype(int), "Case": is_case.astype(int), "Control": is_ctrl.astype(int)})
        stats = pd.DataFrame(index=range(int(new_k)))
        stats["Case"] = df_cc.groupby("Cluster")["Case"].sum().reindex(range(int(new_k))).fillna(0).astype(int)
        stats["Control"] = df_cc.groupby("Cluster")["Control"].sum().reindex(range(int(new_k))).fillna(0).astype(int)
        stats["Total"] = df_cc.groupby("Cluster").size().reindex(range(int(new_k))).fillna(0).astype(int)

        case_counts_arr = stats["Case"].to_numpy(dtype=np.int64, copy=False)
        control_counts_arr = stats["Control"].to_numpy(dtype=np.int64, copy=False)
        total_counts_arr = stats["Total"].to_numpy(dtype=np.int64, copy=False)

        case_counts_by_cluster = {cid: case_counts_arr[cid] for cid in range(int(new_k))}
        control_counts_by_cluster = {cid: control_counts_arr[cid] for cid in range(int(new_k))}
        total_counts_by_cluster = {cid: total_counts_arr[cid] for cid in range(int(new_k))}
        all_cluster_ids = list(range(int(new_k)))

        mainland_premerge_cluster_ids = _extract_mainland_premerge_cluster_ids(merge_map)
        priority_ids = [cid for cid in mainland_premerge_cluster_ids if cid in case_counts_by_cluster]
        priority_ids = sorted(priority_ids, key=lambda cid: (-case_counts_by_cluster.get(cid, 0), cid))
        priority_set = set(priority_ids)
        ordered_cluster_ids = priority_ids + [cid for cid in all_cluster_ids if cid not in priority_set]

        columns = ["Cluster", str(config.case_label), str(config.control_label), "Total"]
        cell_text: list[list[str]] = []
        for cid in ordered_cluster_ids:
            cell_text.append([
                f"Cluster {cid}",
                f"{case_counts_by_cluster[cid]:,}",
                f"{control_counts_by_cluster[cid]:,}",
                f"{total_counts_by_cluster[cid]:,}",
            ])

        cell_text.append([
            "Grand Total",
            f"{cast(int, stats['Case'].sum()):,}",
            f"{cast(int, stats['Control'].sum()):,}",
            f"{cast(int, stats['Total'].sum()):,}",
        ])

        # Adaptive typography/geometry to keep dense cluster tables readable.
        n_cluster_rows = int(len(ordered_cluster_ids))
        if n_cluster_rows >= 30:
            table_fontsize = 15
            header_fontsize = 17
            row_label_fontsize = 15
        elif n_cluster_rows >= 22:
            table_fontsize = 17
            header_fontsize = 19
            row_label_fontsize = 17
        else:
            table_fontsize = 20
            header_fontsize = 22
            row_label_fontsize = 20

        the_table = ax4.table(
            cellText=cell_text,
            colLabels=columns,
            colWidths=[0.34, 0.22, 0.22, 0.22],
            loc="center",
            cellLoc="center",
            bbox=Bbox.from_bounds(0.05, 0.05, 0.90, 0.86),
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(table_fontsize)

        cells = the_table.get_celld()
        n_rows_table = len(cell_text) + 1
        row_height = 0.84 / n_rows_table
        for (row, col), cell in cells.items():
            if row == 0:
                cell.set_height(row_height * 1.20)
                cell.set_text_props(weight="bold", color="white", size=header_fontsize)
                cell.set_facecolor("#4C72B0")
                cell.set_linewidth(1.5)
                cell.set_edgecolor("white")
            else:
                cell.set_height(row_height)
                cell.set_text_props(size=table_fontsize)
                cell.set_linewidth(0.5)
                cell.set_edgecolor("#dddddd")
                if row == len(cell_text):
                    cell.set_facecolor("#e6f3ff")
                    cell.set_text_props(weight="bold")
                elif col == 0:
                    cid = ordered_cluster_ids[row - 1]
                    base_color = merged_cluster_palette.get(int(cid), "#f2f2f2")
                    cell.set_facecolor(to_rgba(base_color, alpha=0.38))
                    cell.set_text_props(weight="bold", size=row_label_fontsize)
                elif row % 2 == 0:
                    cell.set_facecolor("#fbfbfb")
                else:
                    cell.set_facecolor("white")

        # Mainland annotation bracket for the prioritized pre-merge rows.
        mainland_rows = [i + 1 for i, cid in enumerate(ordered_cluster_ids) if cid in priority_set]
        if mainland_rows:
            # Ensure table cell geometry is finalized before querying coordinates.
            fig.canvas.draw()

            first_row = min(mainland_rows)
            last_row = max(mainland_rows)

            x_last, y_last = cells[(last_row, 0)].get_xy()
            _x_first, y_first = cells[(first_row, 0)].get_xy()
            h_first = cells[(first_row, 0)].get_height()
            y_top = float(y_first + h_first)
            y_bot = float(y_last)

            brace_x = float(max(0.01, x_last - 0.020))
            tick_x = float(x_last - 0.003)
            ax4.plot([brace_x, brace_x], [y_bot, y_top], color="#4A4A4A", lw=2.0, transform=ax4.transAxes, clip_on=False)
            ax4.plot([brace_x, tick_x], [y_top, y_top], color="#4A4A4A", lw=2.0, transform=ax4.transAxes, clip_on=False)
            ax4.plot([brace_x, tick_x], [y_bot, y_bot], color="#4A4A4A", lw=2.0, transform=ax4.transAxes, clip_on=False)
            ax4.text(
                float(brace_x - 0.010),
                float((y_top + y_bot) / 2.0),
                "Mainland",
                rotation=90,
                va="center",
                ha="right",
                fontsize=max(12, table_fontsize - 1),
                fontweight="bold",
                color="#4A4A4A",
                transform=ax4.transAxes,
                clip_on=False,
            )

        cohort_title = f"{str(config.case_label)} & {str(config.control_label)}"
        fig.suptitle(
            f"Bayesian Ancestry Inference Results ({cohort_title})",
            fontsize=34,
            fontweight="bold",
            y=0.965,
        )

        if bool(config.save_plot):
            figure_path = out_dir / str(config.figure_file)
            fig.savefig(figure_path, bbox_inches="tight", dpi=400)

        if bool(config.show_plot):
            plt.show()
        else:
            plt.close(fig)

        cluster_stats = stats.loc[ordered_cluster_ids].reset_index().rename(columns={"index": "Cluster"})

    if bool(getattr(config, "verbose", True)):
        print("\n" + "=" * 80)
        mode_name = "PRE-MERGE GMM COMPONENTS" if is_premerge_identity_mode else "MERGED GMM CLUSTERS"
        print(f"STEP4: COHORT ASSIGNMENT TO {mode_name}".center(80))
        print("=" * 80)
        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  output_dir            : {out_dir}")
        print(f"  save_tables           : {bool(config.save_tables)}")
        print(f"  save_plot             : {bool(config.save_plot)}")
        print(f"  show_plot             : {bool(config.show_plot)}")
        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  cohort rows           : {our_samples.shape[0]:,}")
        print(f"  assigned_clusters (K) : {int(new_k)}")
        if assignment_tsv is not None:
            print(f"  assignment_tsv        : {assignment_tsv}")
        print("=" * 80)

    return OURAssignmentOutput(
        df_results=df_results,
        probs_merged_our=probs_merged_our,
        assigned_merged=assigned_merged,
        assignment_confidence=assignment_conf,
        cluster_stats=cluster_stats,
        output_dir=out_dir,
        assignment_tsv=assignment_tsv,
        figure_path=figure_path,
    )
