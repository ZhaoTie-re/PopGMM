"""Figures for the cohort-assignment and subcluster stages.

Every figure here draws the study cohort over the reference panel in PC1-PC2, so
they share the window (:class:`PCWindow`) and the grey background cloud
(``panels.add_reference_background``). Keeping the window in one place is what
makes the panels of a figure -- and the figures of two variants -- comparable:
a cluster must not appear to move because an axis auto-scaled differently.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from scripts.plotting.panels import add_reference_background, apply_equal_centered_limits
from scripts.plotting.style import (
    CASE_COLOR,
    CONFIDENCE_CMAP,
    CONTROL_COLOR,
    REFERENCE_ALPHA,
    REFERENCE_COLOR,
)


@dataclass(frozen=True)
class PCWindow:
    """The square PC1-PC2 window shared by every panel of a figure.

    Built from the reference panel plus the study cohort, never from one panel's
    own points, so all panels frame the same region.
    """

    x_center: float
    y_center: float
    span: float
    var1: float = 0.0
    var2: float = 0.0

    @classmethod
    def from_points(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pad: float = 1.05,
        var1: float = 0.0,
        var2: float = 0.0,
    ) -> "PCWindow":
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        max_span = max(x_max - x_min, y_max - y_min)
        if max_span == 0:
            max_span = 1.0
        return cls(
            x_center=(x_max + x_min) / 2.0,
            y_center=(y_max + y_min) / 2.0,
            span=max_span * pad,
            var1=var1,
            var2=var2,
        )

    def apply(self, ax: Axes) -> None:
        apply_equal_centered_limits(
            ax, x_center=self.x_center, y_center=self.y_center, span=self.span
        )

    @property
    def xlabel(self) -> str:
        return f"PC1 ({self.var1:.1%})"

    @property
    def ylabel(self) -> str:
        return f"PC2 ({self.var2:.1%})"


@dataclass(frozen=True)
class ReferenceCloud:
    """The grey reference-panel background, drawn beneath every scatter panel."""

    pc1: np.ndarray
    pc2: np.ndarray
    color: str = REFERENCE_COLOR
    alpha: float = REFERENCE_ALPHA

    def draw(self, ax: Axes, *, label: str = "BBJ") -> None:
        add_reference_background(
            ax, self.pc1, self.pc2, label=label, color=self.color, alpha=self.alpha
        )


def _confidence_colorbar(fig: Figure, ax: Axes, mappable) -> None:
    """Slim vertical colourbar pinned to the right edge of ``ax``.

    Positioned from ``ax.get_position()``, which is only correct once the layout
    is final -- hence no layout engine on these figures.
    """
    pos = ax.get_position()
    cax = fig.add_axes(
        (
            float(pos.x1 + 0.015),
            float(pos.y0 + pos.height * 0.04),
            0.010,
            float(pos.height * 0.92),
        )
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Max Posterior Probability (Confidence)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot_subcluster_view(
    *,
    study_pc1: np.ndarray,
    study_pc2: np.ndarray,
    is_case: np.ndarray,
    is_ctrl: np.ndarray,
    confidence: np.ndarray,
    reference: ReferenceCloud,
    window: PCWindow,
    counts: pd.DataFrame,
    n_unlabeled: int,
    case_label: str,
    control_label: str,
    group_label: str,
    basis_label: str,
    study_point_size: float,
) -> Figure:
    """PC1-PC2 view of one subcluster variant under the recomputed posterior.

    ``counts`` is the per-group table the caller already wrote to disk, so panel
    B and ``subcluster_view_counts.tsv`` cannot disagree.
    """
    fig = plt.figure(figsize=(26, 24))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.06, top=0.885, wspace=0.28, hspace=0.30)

    # --- A. study cohort by group ------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    reference.draw(ax1, label="BBJ")
    if bool(is_ctrl.any()):
        ax1.scatter(
            study_pc1[is_ctrl], study_pc2[is_ctrl], c=CONTROL_COLOR, s=study_point_size,
            alpha=0.85, edgecolor="white", linewidth=0.3, label=control_label,
            rasterized=True, zorder=2,
        )
    if bool(is_case.any()):
        ax1.scatter(
            study_pc1[is_case], study_pc2[is_case], c=CASE_COLOR, s=study_point_size,
            alpha=0.90, edgecolor="white", linewidth=0.3, label=case_label,
            rasterized=True, zorder=3,
        )
    window.apply(ax1)
    ax1.set_title("A. Study Cohort", loc="left", pad=25, fontweight="bold")
    ax1.set_xlabel(window.xlabel, labelpad=15)
    ax1.set_ylabel(window.ylabel, labelpad=15)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=18, markerscale=1.3)

    # --- B. group sizes -----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    group_order = [case_label, control_label]
    palette = {case_label: CASE_COLOR, control_label: CONTROL_COLOR}
    counts_b = counts.set_index("Group").reindex(group_order, fill_value=0).reset_index()
    sns.barplot(
        data=counts_b, x="Group", y="Count", hue="Group",
        order=group_order, hue_order=group_order, palette=palette, legend=False, ax=ax2,
    )
    for patch in ax2.patches:
        patch.set_linewidth(1.2)
        patch.set_edgecolor("white")

    ymax = float(max(1, counts_b["Count"].max()))
    ax2.set_ylim(0, ymax * 1.20)
    for i, grp in enumerate(group_order):
        val = int(counts_b.loc[counts_b["Group"] == grp, "Count"].iloc[0])
        ax2.text(
            i, float(val) + ymax * 0.04, f"{val:,}", ha="center", va="bottom",
            fontsize=20, fontweight="bold", color="#1f1f1f", linespacing=1.05,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "none"},
        )
    ax2.set_title(f"B. Group Sample Size in {group_label}", loc="left", pad=25, fontweight="bold")
    ax2.set_xlabel("Group", labelpad=15, fontsize=26)
    ax2.set_ylabel("Sample Count", labelpad=15, fontsize=26)
    ax2.tick_params(axis="x", labelsize=22)
    ax2.tick_params(axis="y", labelsize=22)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    if n_unlabeled > 0:
        ax2.text(
            0.98, 0.98, f"Unlabeled (not case/control): {n_unlabeled:,}",
            transform=ax2.transAxes, ha="right", va="top", fontsize=12, color="#555555",
        )

    # --- C. assignment confidence ------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    reference.draw(ax3, label="BBJ")
    # Highest confidence drawn last so the certain core is not buried under the
    # uncertain rim.
    sort_idx = np.argsort(confidence)[::-1]
    vmin = float(np.nanmin(confidence))
    vmax = float(np.nanmax(confidence))
    if vmax <= vmin:
        vmax = min(1.0, vmin + 1e-6)
    scatter = ax3.scatter(
        study_pc1[sort_idx], study_pc2[sort_idx], c=confidence[sort_idx],
        cmap=CONFIDENCE_CMAP, vmin=vmin, vmax=vmax, s=study_point_size,
        alpha=0.90, edgecolors="none", zorder=2,
    )
    window.apply(ax3)
    ax3.set_title("C. Assignment Confidence", loc="left", pad=25, fontweight="bold")
    ax3.set_xlabel(window.xlabel, labelpad=15)
    ax3.set_ylabel(window.ylabel, labelpad=15)
    _confidence_colorbar(fig, ax3, scatter)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    fig.suptitle(
        f"Subcluster View Under the Recomputed Composite Posterior  [{basis_label}]",
        fontsize=34, fontweight="bold", y=0.965,
    )
    return fig


def plot_subcluster_assignment(
    *,
    study_pc1: np.ndarray,
    study_pc2: np.ndarray,
    is_case: np.ndarray,
    is_ctrl: np.ndarray,
    is_other: np.ndarray,
    assigned_group: np.ndarray,
    assignment_conf: np.ndarray,
    stats: pd.DataFrame,
    group_order: list[str],
    group_palette: dict[str, str],
    group_label: str,
    subcluster_component_ids: list[int],
    var1: float,
    var2: float,
    window: PCWindow,
    reference: ReferenceCloud,
    config: Any,
) -> Figure:
    """Study cohort, recomputed-posterior assignment, confidence, and a summary table.

    ``stats`` is computed by the caller and passed in, so the table panel and
    the statistics TSV are guaranteed to be the same numbers.
    """
    fig = plt.figure(figsize=(26, 24))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.06, top=0.885, wspace=0.28, hspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    reference.draw(ax1, label="BBJ")

    s_study = float(config.study_point_size)
    if bool(is_ctrl.any()):
        ax1.scatter(study_pc1[is_ctrl], study_pc2[is_ctrl], c="#1F78B4", s=s_study, alpha=0.85, edgecolor="white", linewidth=0.3, label=str(config.control_label), rasterized=True, zorder=2)
    if bool(is_case.any()):
        ax1.scatter(study_pc1[is_case], study_pc2[is_case], c="#E31A1C", s=s_study, alpha=0.90, edgecolor="white", linewidth=0.3, label=str(config.case_label), rasterized=True, zorder=3)
    if bool(is_other.any()):
        ax1.scatter(study_pc1[is_other], study_pc2[is_other], c="#33A02C", s=s_study, alpha=0.85, edgecolor="white", linewidth=0.3, label="Other", rasterized=True, zorder=1)

    window.apply(ax1)
    ax1.set_title("A. Study Cohort", loc="left", pad=25, fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
    ax1.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=18, markerscale=1.3)

    ax2 = fig.add_subplot(gs[0, 1])
    reference.draw(ax2, label="BBJ")
    plot_b = pd.DataFrame({"PC1": study_pc1, "PC2": study_pc2, "Group": assigned_group})
    sns.scatterplot(
        data=plot_b,
        x="PC1",
        y="PC2",
        hue="Group",
        hue_order=group_order,
        palette=group_palette,
        s=s_study,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.3,
        ax=ax2,
        zorder=2,
    )
    window.apply(ax2)
    ax2.set_title("B. Assignment Under Recomputed Posterior", loc="left", pad=25, fontweight="bold")
    ax2.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
    ax2.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
    legend_b = ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=18, markerscale=1.3)
    for t in legend_b.get_texts():
        t.set_fontweight("bold")

    ax3 = fig.add_subplot(gs[1, 0])
    reference.draw(ax3, label="BBJ")
    sort_idx = np.argsort(assignment_conf)[::-1]
    vmin = float(np.nanmin(assignment_conf))
    vmax = float(np.nanmax(assignment_conf))
    if vmax <= vmin:
        vmax = min(1.0, vmin + 1e-6)
    sc = ax3.scatter(study_pc1[sort_idx], study_pc2[sort_idx], c=assignment_conf[sort_idx], cmap="cividis", vmin=vmin, vmax=vmax, s=s_study, alpha=0.90, edgecolors="none", zorder=2)
    window.apply(ax3)
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
    ax4.set_title("D. Assignment Statistics (Recomputed Posterior)", loc="center", pad=40, fontweight="bold")

    df_cc = pd.DataFrame({"Group": assigned_group, "Case": is_case.astype(int), "Control": is_ctrl.astype(int)})
    stats = pd.DataFrame(index=group_order)  # pyright: ignore[reportArgumentType]
    stats["Case"] = df_cc.groupby("Group")["Case"].sum().reindex(group_order).fillna(0).astype(int)
    stats["Control"] = df_cc.groupby("Group")["Control"].sum().reindex(group_order).fillna(0).astype(int)
    stats["Total"] = df_cc.groupby("Group").size().reindex(group_order).fillna(0).astype(int)

    case_counts = stats["Case"].to_numpy(dtype=np.int64, copy=False)
    control_counts = stats["Control"].to_numpy(dtype=np.int64, copy=False)
    total_counts = stats["Total"].to_numpy(dtype=np.int64, copy=False)
    case_count_map = {group_order[i]: case_counts[i] for i in range(len(group_order))}
    control_count_map = {group_order[i]: control_counts[i] for i in range(len(group_order))}
    total_count_map = {group_order[i]: total_counts[i] for i in range(len(group_order))}

    rows = [[group, f"{case_count_map[group]:,}", f"{control_count_map[group]:,}", f"{total_count_map[group]:,}"] for group in group_order]
    rows.append(["Grand Total", f"{int(case_counts.sum()):,}", f"{int(control_counts.sum()):,}", f"{int(total_counts.sum()):,}"])

    table = ax4.table(cellText=rows, colLabels=["Cluster", str(config.case_label), str(config.control_label), "Total"], loc="center", cellLoc="center", bbox=Bbox.from_bounds(0.05, 0.05, 0.90, 0.86))
    table.auto_set_font_size(False)
    table.set_fontsize(16)

    cells = table.get_celld()
    n_rows_table = len(rows) + 1
    row_h = 0.84 / n_rows_table
    for (row, col), cell in cells.items():
        if row == 0:
            cell.set_height(row_h * 1.2)
            cell.set_text_props(weight="bold", color="white", size=17)
            cell.set_facecolor("#4C72B0")
            cell.set_edgecolor("white")
            cell.set_linewidth(1.2)
        else:
            cell.set_height(row_h)
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(0.5)
            if row == len(rows):
                cell.set_facecolor("#e6f3ff")
                cell.set_text_props(weight="bold")
            elif col == 0:
                group_name = str(rows[row - 1][0])
                cell.set_text_props(weight="bold")
                if group_name == group_label:
                    cell.set_facecolor(to_rgba(str(config.group_color), alpha=0.35))
                elif row % 2 == 0:
                    cell.set_facecolor("#f7f7f7")
                else:
                    cell.set_facecolor("white")
            elif row % 2 == 0:
                cell.set_facecolor("#fbfbfb")
            else:
                cell.set_facecolor("white")

    included_txt = ", ".join(str(int(x)) for x in sorted(subcluster_component_ids))
    # Wrapped, because the component list grows with the variant and this figure
    # is saved with bbox_inches="tight": an unwrapped title overflows the 26 in
    # canvas and the crop then *widens* the output to fit it. Measured, the three
    # variants came out 14738 / 15578 / 16942 px wide -- the same figure at three
    # different scales, purely because their titles were different lengths.
    # The wrap keeps every word; it only stops the text from setting the width.
    fig.suptitle(
        "\n".join(
            textwrap.wrap(
                "Global Posterior Reassignment with Mainland Subcluster as a "
                f"Composite Group (Subcluster definition includes mainland "
                f"components: {included_txt})",
                # 100 gives two lines for every variant (88 gave three for `full`,
                # which then overlapped panel B's title).
                width=100,
            )
        ),
        fontsize=34,
        fontweight="bold",
        y=0.975,
        va="top",
    )
    return fig


def plot_cohort_assignment(
    *,
    study_pc1: np.ndarray,
    study_pc2: np.ndarray,
    is_case: np.ndarray,
    is_ctrl: np.ndarray,
    is_other: np.ndarray,
    assigned_merged: np.ndarray,
    assignment_conf: np.ndarray,
    merged_cluster_palette: dict,
    hue_order: list,
    is_premerge_identity_mode: bool,
    stats: pd.DataFrame,
    ratio_map: dict,
    rank_map: dict,
    priority_set: set,
    ordered_cluster_ids: list,
    case_counts_by_cluster: dict,
    control_counts_by_cluster: dict,
    total_counts_by_cluster: dict,
    var1: float,
    var2: float,
    window: PCWindow,
    reference: ReferenceCloud,
    config: Any,
) -> Figure:
    """Study cohort, assigned cluster, confidence, and the per-cluster statistics table.

    The ranking arguments come from ``cohort_assignment.compute_cluster_ranking``
    and are computed whether or not this runs, so the statistics tables do not
    depend on plotting being enabled.
    """
    fig = plt.figure(figsize=(26, 24))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
    # Reserve figure margin for outside legends and the colorbar.
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.06, top=0.885, wspace=0.28, hspace=0.30)

    # A
    ax1 = fig.add_subplot(gs[0, 0])
    reference.draw(ax1, label="BBJ")

    ctrl_color = "#1F78B4"
    case_color = "#E31A1C"
    other_color = "#33A02C"
    s_study = float(config.study_point_size)

    if bool(is_ctrl.any()):
        ax1.scatter(
            study_pc1[is_ctrl],
            study_pc2[is_ctrl],
            c=ctrl_color,
            s=s_study,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
            label=str(config.control_label),
            rasterized=True,
            zorder=2,
        )
    if bool(is_case.any()):
        ax1.scatter(
            study_pc1[is_case],
            study_pc2[is_case],
            c=case_color,
            s=s_study,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.3,
            label=str(config.case_label),
            rasterized=True,
            zorder=3,
        )
    if bool(is_other.any()):
        ax1.scatter(
            study_pc1[is_other],
            study_pc2[is_other],
            c=other_color,
            s=s_study,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
            label="Other",
            rasterized=True,
            zorder=1,
        )

    window.apply(ax1)
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
        reference_handle_a = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=str(config.reference_color),
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.8,
            alpha=1.0,
            markersize=float(np.sqrt(s_study)),
        )
        handles_a = [reference_handle_a if lab == "BBJ" else h for h, lab in zip(handles_a, labels_a)]

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
    reference.draw(ax2, label="BBJ")

    df_plot_b = pd.DataFrame({"PC1": study_pc1, "PC2": study_pc2, "Cluster": assigned_merged.astype(int)})
    sns.scatterplot(
        data=df_plot_b,
        x="PC1",
        y="PC2",
        hue="Cluster",
        hue_order=hue_order,
        palette=merged_cluster_palette,
        s=s_study,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.3,
        ax=ax2,
        zorder=2,
    )

    window.apply(ax2)
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
    reference_handle_b = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=str(config.reference_color),
        markeredgecolor="#4A4A4A",
        markeredgewidth=0.8,
        alpha=1.0,
        markersize=float(np.sqrt(s_study)),
    )

    new_handles_b: list[Any] = []
    new_labels_b: list[str] = []
    for h, lab in zip(handles_b, labels_b):
        if lab == "BBJ":
            new_handles_b.append(reference_handle_b)
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
    reference.draw(ax3, label="BBJ")

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
        study_pc1[sort_idx],
        study_pc2[sort_idx],
        c=assignment_conf[sort_idx],
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        s=s_study,
        alpha=0.90,
        edgecolors="none",
        zorder=2,
    )

    window.apply(ax3)
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

    columns = ["Cluster", f"{config.case_label}*", f"{config.control_label}*", "Total*", "Case/Ctrl", "Rank"]
    cell_text: list[list[str]] = []
    for cid in ordered_cluster_ids:
        ratio = ratio_map[cid]
        rank_str = str(rank_map[cid]) if cid in priority_set else "-"
        if cid in priority_set:
            ratio_str = "inf" if np.isinf(ratio) else f"{ratio:.3f}"
        else:
            ratio_str = "-"
        cell_text.append([
            f"Cluster {cid}",
            f"{case_counts_by_cluster[cid]:,}",
            f"{control_counts_by_cluster[cid]:,}",
            f"{total_counts_by_cluster[cid]:,}",
            ratio_str,
            rank_str,
        ])

    cell_text.append([
        "Grand Total",
        f"{cast(int, stats['Case'].sum()):,}",
        f"{cast(int, stats['Control'].sum()):,}",
        f"{cast(int, stats['Total'].sum()):,}",
        "-",
        "-",
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
        colWidths=[0.24, 0.18, 0.18, 0.18, 0.12, 0.10],
        loc="center",
        cellLoc="center",
        bbox=Bbox.from_bounds(0.02, 0.05, 0.96, 0.86),
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
            elif col == 4:  # Case/Ctrl column
                cid = ordered_cluster_ids[row - 1]
                if cid in priority_set:
                    cell.set_text_props(weight="bold", color="#D32F2F")
                    cell.set_facecolor("#FFF3E0")
                else:
                    if row % 2 == 0:
                        cell.set_facecolor("#fbfbfb")
                    else:
                        cell.set_facecolor("white")
            elif col == 5:  # Rank column
                cid = ordered_cluster_ids[row - 1]
                if cid in priority_set:
                    cell.set_text_props(weight="bold", color="#1565C0")
                    cell.set_facecolor("#E3F2FD")
                else:
                    if row % 2 == 0:
                        cell.set_facecolor("#fbfbfb")
                    else:
                        cell.set_facecolor("white")
            elif row % 2 == 0:
                cell.set_facecolor("#fbfbfb")
            else:
                cell.set_facecolor("white")

    # Counting method footnote below the table.
    ax4.text(
        0.02, 0.038,
        f"* Per-component argmax (MAP) assignment. "
        f"Cumulative rank analysis uses composite posterior recomputation.",
        transform=ax4.transAxes,
        fontsize=max(9, table_fontsize - 9),
        color="#757575",
        fontstyle="italic",
        va="top", ha="left",
        clip_on=False,
    )

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
    return fig
