from __future__ import annotations

"""STEP5: High-confidence subset visualization for STEP4 assignment results.

This script filters STEP4 results by an assignment confidence threshold and
renders a 2x2 composite figure following the same visual style/layout as
`scripts.our_assignment.run_our_assignment_to_merged_gmm`.

Panels
------
A) Study cohort (case/control) overlay on BBJ background
B) Assigned merged cluster (colors sourced from STEP3 merge_map)
C) Assignment confidence (continuous colormap)
D) Case/control/total table by cluster (computed on high-confidence subset)
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
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from scripts.our_assignment import OURAssignmentConfig, _PLOT_STYLE_RC, _build_merged_cluster_palette


class HighConfidenceVisualizationOutput(NamedTuple):
    df_results_highconf: pd.DataFrame
    threshold: float
    n_total: int
    n_kept: int
    output_dir: Path
    output_tsv: Path | None
    figure_path: Path | None
    cluster_stats: pd.DataFrame


@dataclass(frozen=True)
class HighConfidenceVizConfig:
    """Configuration for high-confidence subset visualization."""

    output_dir: str = "results/05_high_confidence_visualization"
    save_tables: bool = True
    # Filename templates support placeholders:
    # - {threshold} (e.g., 0.95)
    # - {threshold_str} (e.g., 0.95)
    # - {threshold_tag} (e.g., 0p95)
    # - {threshold_pct} (e.g., 95)
    output_file: str = "our_posterior_probabilities_conf_ge_{threshold_tag}.tsv"
    figure_file: str = "our_assignment_conf_ge_{threshold_tag}.png"

    threshold: float = 0.95

    save_plot: bool = True
    show_plot: bool = False

    bbj_color: str = "#B0B0B0"
    bbj_alpha: float = 0.20
    our_point_size: float = 60.0

    case_label: str = "Case"
    control_label: str = "Control"

    verbose: bool = True


def _pc_sort_key(col: str) -> int:
    import re

    match = re.match(r"^PC(\d+)(?:_AVG)?$", str(col))
    return int(match.group(1)) if match else 10**9


def _resolve_pc_columns_from_results(df_results: pd.DataFrame) -> list[str]:
    """Resolve the first two PC columns from a STEP4 df_results table."""

    import re

    pc_cols = [c for c in df_results.columns if re.match(r"^PC\d+(?:_AVG)?$", str(c))]
    pc_cols = sorted(pc_cols, key=_pc_sort_key)
    if len(pc_cols) < 2:
        raise RuntimeError("df_results must contain at least two PC columns (e.g., PC1_AVG, PC2_AVG).")
    return pc_cols[:2]


def _format_filename_template(template: str, *, threshold: float) -> str:
    """Format filename template with threshold placeholders.

    If template does not contain any placeholders, it is returned as-is.
    """

    if "{" not in str(template):
        return str(template)

    threshold_val = float(threshold)
    threshold_str = f"{threshold_val:.2f}"
    threshold_tag = threshold_str.replace(".", "p")
    threshold_pct = int(round(threshold_val * 100.0))

    try:
        return str(template).format(
            threshold=threshold_val,
            threshold_str=threshold_str,
            threshold_tag=threshold_tag,
            threshold_pct=threshold_pct,
        )
    except Exception:
        # Fail safe: keep original template if formatting fails.
        return str(template)


def run_high_confidence_assignment_visualization(
    *,
    bbj_samples_gmm: Any,
    df_results: pd.DataFrame | None = None,
    our_case_iids: list[Any] | None = None,
    our_ctrl_iids: list[Any] | None = None,
    merge_map: pd.DataFrame | None = None,
    eigenval: pd.DataFrame | None = None,
    step4_config: OURAssignmentConfig | None = None,
    config: HighConfidenceVizConfig | None = None,
) -> HighConfidenceVisualizationOutput:
    """STEP5: Filter STEP4 results by confidence threshold and generate a 2x2 composite figure."""

    # NOTE: In notebooks, Pylance sometimes fails to infer the type of variables
    # defined in earlier cells and can report false-positive type errors.
    # Accepting Any here keeps the notebook call site clean, while we still
    # rely on DataFrame semantics at runtime.
    bbj_samples_gmm = cast(pd.DataFrame, bbj_samples_gmm)

    step4_config = step4_config or OURAssignmentConfig()
    config = config or HighConfidenceVizConfig(
        output_dir=str(step4_config.output_dir),
        bbj_alpha=float(step4_config.bbj_alpha),
        case_label=str(step4_config.case_label),
        control_label=str(step4_config.control_label),
    )

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
    if "Assigned_Merged_Cluster" not in df_results.columns:
        raise KeyError("df_results must contain 'Assigned_Merged_Cluster'.")

    pc_cols = _resolve_pc_columns_from_results(df_results)
    for c in pc_cols:
        if c not in bbj_samples_gmm.columns:
            raise KeyError(f"bbj_samples_gmm is missing required PC column: {c}")

    # Full-data confidence values (used for Panel C colorbar scaling).
    # Even though we plot a filtered subset, we keep the same legend scale
    # as STEP4 (run_our_assignment_to_merged_gmm), which uses full-data min/max.
    conf = pd.to_numeric(df_results["Assignment_Confidence"], errors="coerce")
    vmin_full = float(np.nanmin(conf.to_numpy(dtype=np.float32, copy=False)))
    vmax_full = float(np.nanmax(conf.to_numpy(dtype=np.float32, copy=False)))
    if not np.isfinite(vmin_full) or not np.isfinite(vmax_full):
        vmin_full, vmax_full = 0.0, 1.0
    if vmax_full <= vmin_full:
        vmax_full = min(1.0, vmin_full + 1e-6)
    mask = conf >= float(config.threshold)

    n_total = int(df_results.shape[0])
    df_high = df_results.loc[mask].copy()
    n_kept = int(df_high.shape[0])
    if n_kept == 0:
        raise ValueError(f"No samples have Assignment_Confidence >= {float(config.threshold):.3f}.")

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    output_tsv: Path | None = None
    if bool(config.save_tables):
        output_file = _format_filename_template(str(config.output_file), threshold=float(config.threshold))
        output_tsv = out_dir / str(output_file)
        df_high.to_csv(output_tsv, sep="\t", index=False)

    figure_file = _format_filename_template(str(config.figure_file), threshold=float(config.threshold))
    figure_path = out_dir / str(figure_file) if bool(config.save_plot) else None

    our_case_iids = our_case_iids or []
    our_ctrl_iids = our_ctrl_iids or []
    case_set = set(str(x) for x in our_case_iids)
    ctrl_set = set(str(x) for x in our_ctrl_iids)

    iid_col = "IID" if "IID" in df_high.columns else None
    if iid_col is None:
        raise KeyError("df_results must contain 'IID' for case/control labeling.")

    our_iid = df_high[iid_col].astype(str)
    is_case = our_iid.isin(case_set).to_numpy()
    is_ctrl = our_iid.isin(ctrl_set).to_numpy()
    is_other = ~(is_case | is_ctrl)

    bbj_pc1 = bbj_samples_gmm[pc_cols[0]].to_numpy(dtype=np.float64, copy=False)
    bbj_pc2 = bbj_samples_gmm[pc_cols[1]].to_numpy(dtype=np.float64, copy=False)

    our_pc1 = pd.to_numeric(df_high[pc_cols[0]], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    our_pc2 = pd.to_numeric(df_high[pc_cols[1]], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    assigned = pd.to_numeric(df_high["Assigned_Merged_Cluster"], errors="coerce").fillna(-1).astype(int).to_numpy()
    confidence = pd.to_numeric(df_high["Assignment_Confidence"], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    if not np.isfinite(our_pc1).all() or not np.isfinite(our_pc2).all():
        raise ValueError("High-confidence subset contains non-finite PC coordinates.")

    try:
        var1 = float(eigenval.loc[eigenval["PC"] == 1, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
        var2 = float(eigenval.loc[eigenval["PC"] == 2, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
    except Exception:
        var1, var2 = 0.0, 0.0

    new_k = int(np.max(assigned)) + 1 if int(np.max(assigned)) >= 0 else 0
    merged_cluster_palette = _build_merged_cluster_palette(merge_map=merge_map, new_k=int(max(new_k, 1)))
    hue_order = list(range(int(max(new_k, 1))))

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

    cluster_stats = pd.DataFrame()

    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=2.5)
        plt.rcParams.update(dict(_PLOT_STYLE_RC))

        fig = plt.figure(figsize=(26, 24))
        fig.patch.set_facecolor("white")
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
        fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.88, wspace=0.28, hspace=0.55)

        ctrl_color = "#1F78B4"
        case_color = "#E31A1C"
        other_color = "#33A02C"
        s_our = float(config.our_point_size)

        # A
        ax1 = fig.add_subplot(gs[0, 0])
        _add_bbj_background(ax1, label="BBJ")

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
        ax1.set_title("A. Study Cohort (High Confidence)", loc="left", pad=25, fontweight="bold")
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

        df_plot_b = pd.DataFrame({"PC1": our_pc1, "PC2": our_pc2, "Cluster": assigned.astype(int)})
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
        ax2.set_title("B. Assigned Merged Cluster (High Confidence)", loc="left", pad=25, fontweight="bold")
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

        sort_idx = np.argsort(confidence)[::-1]
        sc = ax3.scatter(
            our_pc1[sort_idx],
            our_pc2[sort_idx],
            c=confidence[sort_idx],
            cmap="cividis",
            vmin=vmin_full,
            vmax=vmax_full,
            s=s_our,
            alpha=0.90,
            edgecolors="none",
            zorder=2,
        )

        _apply_equal_centered_limits(ax3)
        ax3.set_title("C. Assignment Confidence (High Confidence)", loc="left", pad=25, fontweight="bold")
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
        ax4.set_title("D. Statistics by Cluster (High Confidence)", loc="center", pad=40, fontweight="bold")

        df_cc = pd.DataFrame({"Cluster": assigned.astype(int), "Case": is_case.astype(int), "Control": is_ctrl.astype(int)})

        if new_k <= 0:
            stats = pd.DataFrame({"Cluster": [], "Case": [], "Control": [], "Total": []})
        else:
            stats = pd.DataFrame(index=range(int(new_k)))
            stats["Case"] = df_cc.groupby("Cluster")["Case"].sum().reindex(range(int(new_k))).fillna(0).astype(int)
            stats["Control"] = df_cc.groupby("Cluster")["Control"].sum().reindex(range(int(new_k))).fillna(0).astype(int)
            stats["Total"] = df_cc.groupby("Cluster").size().reindex(range(int(new_k))).fillna(0).astype(int)

        columns = [str(config.case_label), str(config.control_label), "Total"]
        row_labels = [f"Cluster {i}" for i in range(int(new_k))]
        cell_text: list[list[str]] = []
        for i in range(int(new_k)):
            cell_text.append([
                f"{cast(int, stats.loc[i, 'Case']):,}",
                f"{cast(int, stats.loc[i, 'Control']):,}",
                f"{cast(int, stats.loc[i, 'Total']):,}",
            ])

        if new_k > 0:
            cell_text.append([
                f"{cast(int, stats['Case'].sum()):,}",
                f"{cast(int, stats['Control'].sum()):,}",
                f"{cast(int, stats['Total'].sum()):,}",
            ])
            row_labels.append("Grand Total")

        the_table = ax4.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            bbox=Bbox.from_bounds(0.15, 0.1, 0.8, 0.8),
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(24)

        cells = the_table.get_celld()
        n_rows_table = len(cell_text) + 1
        row_height = 0.9 / max(1, n_rows_table)
        for (row, col), cell in cells.items():
            cell.set_height(row_height)
            if row == 0:
                cell.set_text_props(weight="bold", color="white", size=26)
                cell.set_facecolor("#4C72B0")
                cell.set_linewidth(1.5)
                cell.set_edgecolor("white")
            elif col == -1:
                cell.set_text_props(weight="bold", size=24)
                cell.set_facecolor("#f2f2f2")
                cell.set_linewidth(1)
                cell.set_edgecolor("white")
                if row == len(cell_text):
                    cell.set_facecolor("#d1dceb")
            else:
                cell.set_text_props(size=24)
                cell.set_linewidth(0.5)
                cell.set_edgecolor("#dddddd")
                if row == len(cell_text):
                    cell.set_facecolor("#e6f3ff")
                    cell.set_text_props(weight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#fbfbfb")
                else:
                    cell.set_facecolor("white")

        cohort_title = f"{str(config.case_label)} & {str(config.control_label)}"
        fig.suptitle(
            f"Bayesian Ancestry Inference Results ({cohort_title})\n"
            f"High-Confidence Subset: confidence ≥ {float(config.threshold):.2f}  (n={n_kept:,}/{n_total:,})",
            fontsize=34,
            fontweight="bold",
            y=0.965,
        )

        if figure_path is not None:
            fig.savefig(figure_path, bbox_inches="tight", dpi=400)

        if bool(config.show_plot):
            plt.show()
        else:
            plt.close(fig)

        if new_k > 0:
            cluster_stats = stats.reset_index().rename(columns={"index": "Cluster"})

    if bool(getattr(config, "verbose", True)):
        print("\n" + "=" * 80)
        print("STEP5: HIGH-CONFIDENCE SUBSET VISUALIZATION".center(80))
        print("=" * 80)
        print(f"  threshold             : {float(config.threshold):.4f}")
        print(f"  n_total               : {n_total:,}")
        print(f"  n_kept                : {n_kept:,}")
        print(f"  kept_fraction         : {n_kept / max(1, n_total):.3%}")
        if output_tsv is not None:
            print(f"  output_tsv            : {output_tsv}")
        print("=" * 80)

    return HighConfidenceVisualizationOutput(
        df_results_highconf=df_high,
        threshold=float(config.threshold),
        n_total=n_total,
        n_kept=n_kept,
        output_dir=out_dir,
        output_tsv=output_tsv,
        figure_path=figure_path,
        cluster_stats=cluster_stats,
    )
