"""Plot the selected subcluster on PC1-PC2 against the reference panel.

A focused two-panel view: where the retained samples sit in the panel's PC space,
and how many of them there are per group. Deliberately narrow -- the all-PC
comparison lives in its own module.

Inputs
------
A subcluster assignment frame, case/control lists, the clustered reference panel.

Outputs
-------
A two-panel figure and the per-group counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import re

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.common import pc_sort_key as _pc_sort_key
from scripts.common import resolve_pc_columns as _shared_resolve_pc_columns


def _resolve_pc_columns(df: pd.DataFrame) -> list[str]:
    """First two PC columns only; this module plots a 2-D scatter."""
    pc_cols = _shared_resolve_pc_columns(df)
    if len(pc_cols) < 2:
        raise ValueError("Need at least 2 PC columns in df_assigned (e.g., PC1_AVG, PC2_AVG).")
    return pc_cols[:2]


class SubclusterViewOutput(NamedTuple):
    df_subcluster: pd.DataFrame
    figure_path: Path | None
    counts_by_group: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class SubclusterViewConfig:
    output_dir: str = "results/04_subcluster_variants/refined"
    figure_file: str = "subcluster_view.png"

    group_label: str = "Mainland Subcluster"
    assigned_group_col: str = "Assigned_Mainland_Subcluster_Group"
    confidence_col: str = "Assignment_Confidence"

    case_label: str = "Case"
    control_label: str = "Control"

    group_color: str = "#E67E22"
    reference_color: str = "#B0B0B0"
    reference_alpha: float = 0.20
    study_point_size: float = 60.0

    save_plot: bool = True
    show_plot: bool = False
    verbose: bool = True


def _resolve_assigned_group_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for candidate in ("Assigned_Mainland_Subcluster_Group", "Assigned_Customize_Group", "Assigned_Composite_Group"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No assigned group column found in df_assigned.")


def run_subcluster_view(
    *,
    df_assigned: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    config: SubclusterViewConfig | None = None,
    reference_samples_gmm: pd.DataFrame | None = None,
    eigenval: pd.DataFrame | None = None,
) -> SubclusterViewOutput:
    config = config or SubclusterViewConfig()

    if "IID" not in df_assigned.columns:
        raise KeyError("df_assigned must contain IID column.")

    assigned_group_col = _resolve_assigned_group_col(df_assigned, config.assigned_group_col)
    pc_cols = _resolve_pc_columns(df_assigned)
    confidence_col = str(config.confidence_col)
    if confidence_col not in df_assigned.columns:
        raise KeyError(f"df_assigned must contain {confidence_col}.")

    mainland_label = str(config.group_label)
    mask = df_assigned[assigned_group_col].astype(str) == mainland_label
    df_subcluster = df_assigned.loc[mask].copy()

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)


    case_set = set(str(x) for x in case_iids)
    ctrl_set = set(str(x) for x in control_iids)

    iid_vals = df_subcluster["IID"].astype(str)
    group = np.where(iid_vals.isin(case_set), str(config.case_label), np.where(iid_vals.isin(ctrl_set), str(config.control_label), ""))
    df_subcluster["Group"] = group

    counts = (
        df_subcluster["Group"]
        .astype(str)
        .value_counts()
        .reindex([str(config.case_label), str(config.control_label)], fill_value=0)
        .rename_axis("Group")
        .reset_index(name="Count")
    )
    n_unlabeled = int((df_subcluster["Group"].astype(str) == "").sum())

    confidence = pd.to_numeric(df_subcluster[confidence_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(confidence).all():
        raise ValueError("Selected mainland_subcluster subset contains invalid assignment confidence values.")

    study_pc1 = pd.to_numeric(df_subcluster[pc_cols[0]], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    study_pc2 = pd.to_numeric(df_subcluster[pc_cols[1]], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(study_pc1).all() or not np.isfinite(study_pc2).all():
        raise ValueError("Selected mainland_subcluster subset contains invalid PC coordinates.")

    try:
        var1 = float(eigenval.loc[eigenval["PC"] == 1, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
        var2 = float(eigenval.loc[eigenval["PC"] == 2, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
    except Exception:
        var1, var2 = 0.0, 0.0

    figure_path: Path | None = None
    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=2.5)

        reference_pc1 = np.array([], dtype=np.float64)
        reference_pc2 = np.array([], dtype=np.float64)
        if reference_samples_gmm is not None and pc_cols[0] in reference_samples_gmm.columns and pc_cols[1] in reference_samples_gmm.columns:
            reference_pc1 = reference_samples_gmm[pc_cols[0]].to_numpy(dtype=np.float64, copy=False)
            reference_pc2 = reference_samples_gmm[pc_cols[1]].to_numpy(dtype=np.float64, copy=False)

        all_x = np.concatenate([reference_pc1, study_pc1]) if reference_pc1.size > 0 else study_pc1
        all_y = np.concatenate([reference_pc2, study_pc2]) if reference_pc2.size > 0 else study_pc2
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

        def _add_reference_background(ax, label: str = "BBJ") -> None:
            if reference_pc1.size == 0:
                return
            ax.scatter(
                reference_pc1,
                reference_pc2,
                c=str(config.reference_color),
                s=20,
                alpha=float(config.reference_alpha),
                label=label,
                rasterized=True,
                zorder=0,
            )

        is_case = (df_subcluster["Group"].astype(str).to_numpy() == str(config.case_label))
        is_ctrl = (df_subcluster["Group"].astype(str).to_numpy() == str(config.control_label))
        fig = plt.figure(figsize=(26, 24))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
        fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.88, wspace=0.28, hspace=0.55)

        ax1 = fig.add_subplot(gs[0, 0])
        _add_reference_background(ax1, label="BBJ")
        s_study = float(config.study_point_size)
        if bool(is_ctrl.any()):
            ax1.scatter(study_pc1[is_ctrl], study_pc2[is_ctrl], c="#1F78B4", s=s_study, alpha=0.85, edgecolor="white", linewidth=0.3, label=str(config.control_label), rasterized=True, zorder=2)
        if bool(is_case.any()):
            ax1.scatter(study_pc1[is_case], study_pc2[is_case], c="#E31A1C", s=s_study, alpha=0.90, edgecolor="white", linewidth=0.3, label=str(config.case_label), rasterized=True, zorder=3)
        _apply_equal_centered_limits(ax1)
        ax1.set_title("A. Study Cohort", loc="left", pad=25, fontweight="bold")
        ax1.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax1.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=18, markerscale=1.3)

        ax2 = fig.add_subplot(gs[0, 1])
        group_order_b = [str(config.case_label), str(config.control_label)]
        palette_b = {
            str(config.case_label): "#E31A1C",
            str(config.control_label): "#1F78B4",
        }
        counts_b = (
            df_subcluster["Group"]
            .astype(str)
            .value_counts()
            .reindex(group_order_b, fill_value=0)
            .rename_axis("Group")
            .reset_index(name="Count")
        )
        sns.barplot(
            data=counts_b,
            x="Group",
            y="Count",
            hue="Group",
            order=group_order_b,
            hue_order=group_order_b,
            palette=palette_b,
            legend=False,
            ax=ax2,
        )
        for p in ax2.patches:
            p.set_linewidth(1.2)
            p.set_edgecolor("white")

        total_b = int(counts_b["Count"].sum())
        ymax = float(max(1, counts_b["Count"].max()))
        ax2.set_ylim(0, ymax * 1.20)
        for i, grp in enumerate(group_order_b):
            val = int(counts_b.loc[counts_b["Group"] == grp, "Count"].iloc[0])
            ax2.text(
                i,
                float(val) + ymax * 0.04,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                color="#1f1f1f",
                linespacing=1.05,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "none"},
            )
        ax2.set_title("B. Group Sample Size in Mainland Subcluster", loc="left", pad=25, fontweight="bold")
        ax2.set_xlabel("Group", labelpad=15, fontsize=26)
        ax2.set_ylabel("Sample Count", labelpad=15, fontsize=26)
        ax2.tick_params(axis="x", labelsize=22)
        ax2.tick_params(axis="y", labelsize=22)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        if n_unlabeled > 0:
            ax2.text(
                0.98,
                0.98,
                f"Unlabeled (not case/control): {n_unlabeled:,}",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                color="#555555",
            )

        ax3 = fig.add_subplot(gs[1, 0])
        _add_reference_background(ax3, label="BBJ")
        sort_idx = np.argsort(confidence)[::-1]
        vmin = float(np.nanmin(confidence))
        vmax = float(np.nanmax(confidence))
        if vmax <= vmin:
            vmax = min(1.0, vmin + 1e-6)
        sc = ax3.scatter(study_pc1[sort_idx], study_pc2[sort_idx], c=confidence[sort_idx], cmap="cividis", vmin=vmin, vmax=vmax, s=s_study, alpha=0.90, edgecolors="none", zorder=2)
        _apply_equal_centered_limits(ax3)
        ax3.set_title("C. Assignment Confidence", loc="left", pad=25, fontweight="bold")
        ax3.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax3.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)

        pos = ax3.get_position()
        cax = fig.add_axes((float(pos.x1 + 0.015), float(pos.y0 + pos.height * 0.04), 0.010, float(pos.height * 0.92)))
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("Max Posterior Probability (Confidence)", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        fig.suptitle(
            "Subcluster View Under the Recomputed Composite Posterior",
            fontsize=34,
            fontweight="bold",
            y=0.965,
        )

        if bool(config.save_plot):
            figure_path = out_dir / str(config.figure_file)
            fig.savefig(figure_path, dpi=400, bbox_inches="tight")

        if bool(config.show_plot):
            plt.show()
        else:
            plt.close(fig)

    if bool(config.verbose):
        print("\n" + "=" * 80)
        print("SUBCLUSTER VIEW".center(80))
        print("=" * 80)
        print(f"  assigned_group_col : {assigned_group_col}")
        print(f"  mainland_label     : {mainland_label}")
        print(f"  rows (unfiltered)  : {int(df_subcluster.shape[0])}")
        print(f"  unlabeled rows     : {n_unlabeled}")
        if figure_path is not None:
            print(f"  figure_file        : {figure_path}")
        print("=" * 80)

    return SubclusterViewOutput(
        df_subcluster=df_subcluster,
        figure_path=figure_path,
        counts_by_group=counts,
        output_dir=out_dir,
    )
