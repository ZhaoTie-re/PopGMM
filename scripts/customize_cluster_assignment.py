from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import json

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


class CustomizeClusterAssignmentOutput(NamedTuple):
    df_results: pd.DataFrame
    customize_cluster: list[int]
    probs_customize: np.ndarray
    assigned_customize: np.ndarray
    assignment_confidence: np.ndarray
    cluster_stats: pd.DataFrame
    output_dir: Path
    assignment_tsv: Path | None
    figure_path: Path | None


# Backward-compatible alias.
MainlandSubclusterReassignmentOutput = CustomizeClusterAssignmentOutput


@dataclass(frozen=True)
class CustomizeClusterAssignmentConfig:
    output_dir: str = "results/05_customize_cluster_assignment"
    save_plot: bool = True
    show_plot: bool = False
    save_tables: bool = True

    output_file: str = "our_posterior_probabilities_customize_merged.tsv"
    figure_file: str = "our_assignment_customize_merged.png"

    # Display name/color for the merged customize cluster.
    custom_group_label: str = "Mainland Subcluster"
    custom_group_color: str = "#E67E22"

    # If customize_cluster is provided, it has priority.
    customize_cluster: tuple[int, ...] | None = None
    # Remove these clusters from mainland_premerge_cluster_ids by default.
    exclude_cluster_ids: int | tuple[int, ...] | list[int] = (13, 15)

    bbj_color: str = "#B0B0B0"
    bbj_alpha: float = 0.20
    our_point_size: float = 60.0

    case_label: str = "Case"
    control_label: str = "Control"

    verbose: bool = True


# Backward-compatible alias with professional naming.
MainlandSubclusterReassignmentConfig = CustomizeClusterAssignmentConfig


def _normalize_cluster_ids(ids: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if ids is None:
        return tuple()
    if isinstance(ids, (int, np.integer)):
        return (int(ids),)
    return tuple(int(x) for x in ids)


def _slugify_label(label: str) -> str:
    return "_".join(str(label).strip().split())


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


def _build_distinct_palette(n_colors: int) -> list[tuple[float, float, float, float]]:
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


def _build_premerge_component_palette(bbj_samples_gmm: pd.DataFrame, n_clusters: int) -> dict[int, str]:
    if "GMM_Cluster" in bbj_samples_gmm.columns:
        labels = bbj_samples_gmm["GMM_Cluster"].to_numpy(dtype=np.int32, copy=False)
        unique_labels = np.unique(labels).astype(int)
    else:
        unique_labels = np.arange(int(n_clusters), dtype=np.int32)

    palette = _build_distinct_palette(int(len(unique_labels)))
    color_map = {int(k): str(to_hex(palette[i])) for i, k in enumerate(unique_labels.tolist())}

    if int(n_clusters) > len(unique_labels):
        fallback_hex = sns.color_palette("tab20", n_colors=int(n_clusters)).as_hex()
        for i in range(int(n_clusters)):
            color_map.setdefault(int(i), str(fallback_hex[int(i)]))

    return color_map


def _build_customize_cluster(
    *,
    mainland_premerge_cluster_ids: list[int],
    n_components: int,
    config: CustomizeClusterAssignmentConfig,
) -> list[int]:
    if config.customize_cluster is not None:
        candidate = [int(x) for x in config.customize_cluster]
    else:
        exclude_ids = _normalize_cluster_ids(config.exclude_cluster_ids)
        exclude = set(int(x) for x in exclude_ids)
        candidate = [int(x) for x in mainland_premerge_cluster_ids if int(x) not in exclude]

    candidate = sorted(set(candidate))
    candidate = [x for x in candidate if 0 <= int(x) < int(n_components)]
    if not candidate:
        raise ValueError("customize_cluster is empty after filtering; please adjust config.")

    return candidate


def run_customize_cluster_assignment(
    *,
    gmm_model: GaussianMixture,
    bbj_samples_gmm: pd.DataFrame,
    our_samples: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    mainland_premerge_cluster_ids: list[int],
    eigenval: pd.DataFrame | None = None,
    gmm_summary: dict[str, Any] | None = None,
    config: CustomizeClusterAssignmentConfig | None = None,
) -> CustomizeClusterAssignmentOutput:
    config = config or CustomizeClusterAssignmentConfig()
    exclude_cluster_ids = _normalize_cluster_ids(config.exclude_cluster_ids)

    if our_samples.empty:
        raise ValueError("our_samples is empty; cannot run customize-cluster assignment.")
    if "IID" not in our_samples.columns:
        raise ValueError("our_samples must contain an 'IID' column.")

    pc_cols_used = _resolve_pc_columns_for_projection(our_samples=our_samples, gmm_model=gmm_model, gmm_summary=gmm_summary)

    x_our = our_samples[pc_cols_used].to_numpy(dtype=np.float64, copy=False)
    probs_original = gmm_model.predict_proba(x_our).astype(np.float64, copy=False)

    old_k = int(probs_original.shape[1])
    mainland_subcluster_component_ids = _build_customize_cluster(
        mainland_premerge_cluster_ids=mainland_premerge_cluster_ids,
        n_components=int(old_k),
        config=config,
    )

    mainland_subcluster_set = set(int(x) for x in mainland_subcluster_component_ids)
    remaining_component_ids = [int(cid) for cid in range(int(old_k)) if int(cid) not in mainland_subcluster_set]

    # Recompute posterior by treating customize_cluster as ONE merged cluster,
    # while keeping all remaining pre-merge components as separate clusters.
    n_groups = 1 + int(len(remaining_component_ids))
    probs_customize = np.zeros((probs_original.shape[0], n_groups), dtype=np.float64)
    probs_customize[:, 0] = probs_original[:, mainland_subcluster_component_ids].sum(axis=1)
    for i, cid in enumerate(remaining_component_ids, start=1):
        probs_customize[:, i] = probs_original[:, int(cid)]

    row_sums = probs_customize.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_customize = probs_customize / row_sums

    assigned_idx = np.argmax(probs_customize, axis=1).astype(np.int32)
    custom_group_label = str(config.custom_group_label)
    custom_group_slug = _slugify_label(custom_group_label)
    assigned_customize = np.asarray(
        [(-1 if int(i) == 0 else int(remaining_component_ids[int(i) - 1])) for i in assigned_idx],
        dtype=np.int32,
    )
    assigned_group = np.asarray(
        [(custom_group_label if int(i) == 0 else f"Cluster {remaining_component_ids[int(i) - 1]}") for i in assigned_idx],
        dtype=object,
    )
    assignment_conf = np.max(probs_customize, axis=1).astype(np.float32)

    fid_col = "#FID" if "#FID" in our_samples.columns else ("FID" if "FID" in our_samples.columns else None)
    meta_cols = [c for c in [fid_col, "IID"] if c is not None and c in our_samples.columns]

    df_results = our_samples[meta_cols + pc_cols_used[:2]].copy() if meta_cols else our_samples[pc_cols_used[:2]].copy()
    if fid_col == "#FID":
        df_results = df_results.rename(columns={"#FID": "FID"})

    # Professional primary fields.
    df_results["Assigned_Mainland_Subcluster_Group"] = assigned_group
    df_results["Assigned_Component_ID_If_Not_Subcluster"] = assigned_customize
    df_results["Prob_Mainland_Subcluster"] = probs_customize[:, 0].astype(np.float32)

    # Backward-compatible fields.
    df_results["Assigned_Customize_Group"] = assigned_group
    df_results["Assigned_Composite_Group"] = assigned_group
    df_results["Assigned_Customize_ClusterID"] = assigned_customize
    df_results["Assignment_Confidence"] = assignment_conf
    df_results["Prob_Customize_Cluster"] = probs_customize[:, 0].astype(np.float32)
    df_results[f"Prob_{custom_group_slug}"] = probs_customize[:, 0].astype(np.float32)
    for i, cid in enumerate(remaining_component_ids, start=1):
        df_results[f"Prob_Cluster_{cid}"] = probs_customize[:, i].astype(np.float32)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_tsv: Path | None = None
    if bool(config.save_tables):
        assignment_tsv = out_dir / str(config.output_file)
        df_results.to_csv(assignment_tsv, sep="\t", index=False)

        summary = {
            "customize_cluster": [int(x) for x in mainland_subcluster_component_ids],
            "mainland_subcluster_component_ids": [int(x) for x in mainland_subcluster_component_ids],
            "exclude_cluster_ids": [int(x) for x in exclude_cluster_ids],
            "remaining_clusters": [int(x) for x in remaining_component_ids],
            "remaining_component_ids": [int(x) for x in remaining_component_ids],
            "custom_group_label": str(config.custom_group_label),
            "custom_group_color": str(config.custom_group_color),
            "n_groups": int(n_groups),
            "rows": int(df_results.shape[0]),
            "workflow_name": "mainland_subcluster_reassignment",
        }
        with (out_dir / "customize_cluster_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    figure_path: Path | None = None
    cluster_stats = pd.DataFrame()

    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=2.5)

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

        full_palette = _build_premerge_component_palette(bbj_samples_gmm=bbj_samples_gmm, n_clusters=int(old_k))
        group_order = [custom_group_label] + [f"Cluster {cid}" for cid in remaining_component_ids]
        group_palette: dict[str, str] = {custom_group_label: str(config.custom_group_color)}
        for cid in remaining_component_ids:
            group_palette[f"Cluster {cid}"] = full_palette.get(int(cid), "#999999")

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
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
        fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.88, wspace=0.28, hspace=0.55)

        ax1 = fig.add_subplot(gs[0, 0])
        _add_bbj_background(ax1, label="BBJ")

        s_our = float(config.our_point_size)
        if bool(is_ctrl.any()):
            ax1.scatter(our_pc1[is_ctrl], our_pc2[is_ctrl], c="#1F78B4", s=s_our, alpha=0.85, edgecolor="white", linewidth=0.3, label=str(config.control_label), rasterized=True, zorder=2)
        if bool(is_case.any()):
            ax1.scatter(our_pc1[is_case], our_pc2[is_case], c="#E31A1C", s=s_our, alpha=0.90, edgecolor="white", linewidth=0.3, label=str(config.case_label), rasterized=True, zorder=3)
        if bool(is_other.any()):
            ax1.scatter(our_pc1[is_other], our_pc2[is_other], c="#33A02C", s=s_our, alpha=0.85, edgecolor="white", linewidth=0.3, label="Other", rasterized=True, zorder=1)

        _apply_equal_centered_limits(ax1)
        ax1.set_title("A. Study Cohort", loc="left", pad=25, fontweight="bold")
        ax1.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax1.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=18, markerscale=1.3)

        ax2 = fig.add_subplot(gs[0, 1])
        _add_bbj_background(ax2, label="BBJ")
        plot_b = pd.DataFrame({"PC1": our_pc1, "PC2": our_pc2, "Group": assigned_group})
        sns.scatterplot(
            data=plot_b,
            x="PC1",
            y="PC2",
            hue="Group",
            hue_order=group_order,
            palette=group_palette,
            s=s_our,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.3,
            ax=ax2,
            zorder=2,
        )
        _apply_equal_centered_limits(ax2)
        ax2.set_title("B. Assignment Under Recomputed Posterior", loc="left", pad=25, fontweight="bold")
        ax2.set_xlabel(f"PC1 ({var1:.1%})", labelpad=15)
        ax2.set_ylabel(f"PC2 ({var2:.1%})", labelpad=15)
        legend_b = ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=18, markerscale=1.3)
        for t in legend_b.get_texts():
            t.set_fontweight("bold")

        ax3 = fig.add_subplot(gs[1, 0])
        _add_bbj_background(ax3, label="BBJ")
        sort_idx = np.argsort(assignment_conf)[::-1]
        vmin = float(np.nanmin(assignment_conf))
        vmax = float(np.nanmax(assignment_conf))
        if vmax <= vmin:
            vmax = min(1.0, vmin + 1e-6)
        sc = ax3.scatter(our_pc1[sort_idx], our_pc2[sort_idx], c=assignment_conf[sort_idx], cmap="cividis", vmin=vmin, vmax=vmax, s=s_our, alpha=0.90, edgecolors="none", zorder=2)
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
        ax4.set_title("D. Assignment Statistics (Recomputed Posterior)", loc="center", pad=40, fontweight="bold")

        df_cc = pd.DataFrame({"Group": assigned_group, "Case": is_case.astype(int), "Control": is_ctrl.astype(int)})
        stats = pd.DataFrame(index=group_order)
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
                    if group_name == custom_group_label:
                        cell.set_facecolor(to_rgba(str(config.custom_group_color), alpha=0.35))
                    elif row % 2 == 0:
                        cell.set_facecolor("#f7f7f7")
                    else:
                        cell.set_facecolor("white")
                elif row % 2 == 0:
                    cell.set_facecolor("#fbfbfb")
                else:
                    cell.set_facecolor("white")

        excluded_txt = ", ".join(str(int(x)) for x in exclude_cluster_ids)
        fig.suptitle(
            (
                "Global Posterior Reassignment with Mainland Subcluster as a Composite Group "
                f"(Subcluster definition excludes mainland components: {excluded_txt})"
            ),
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

        cluster_stats = stats.reset_index().rename(columns={"index": "Group"})

    if bool(config.verbose):
        print("\n" + "=" * 80)
        print("STEP5: MAINLAND SUBCLUSTER REASSIGNMENT".center(80))
        print("=" * 80)
        print(f"  mainland_subcluster_ids: {mainland_subcluster_component_ids}")
        print(f"  remaining_component_ids: {remaining_component_ids}")
        print(f"  excluded_cluster_ids   : {list(exclude_cluster_ids)}")
        print(f"  output_dir             : {out_dir}")
        if assignment_tsv is not None:
            print(f"  assignment_tsv         : {assignment_tsv}")
        print("=" * 80)

    return CustomizeClusterAssignmentOutput(
        df_results=df_results,
        customize_cluster=mainland_subcluster_component_ids,
        probs_customize=probs_customize,
        assigned_customize=assigned_customize,
        assignment_confidence=assignment_conf,
        cluster_stats=cluster_stats,
        output_dir=out_dir,
        assignment_tsv=assignment_tsv,
        figure_path=figure_path,
    )


def run_mainland_subcluster_reassignment(
    *,
    gmm_model: GaussianMixture,
    bbj_samples_gmm: pd.DataFrame,
    our_samples: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    mainland_premerge_cluster_ids: list[int],
    eigenval: pd.DataFrame | None = None,
    gmm_summary: dict[str, Any] | None = None,
    config: MainlandSubclusterReassignmentConfig | None = None,
) -> MainlandSubclusterReassignmentOutput:
    """Professional alias for Step5 reassignment API."""
    return run_customize_cluster_assignment(
        gmm_model=gmm_model,
        bbj_samples_gmm=bbj_samples_gmm,
        our_samples=our_samples,
        our_case_iids=our_case_iids,
        our_ctrl_iids=our_ctrl_iids,
        mainland_premerge_cluster_ids=mainland_premerge_cluster_ids,
        eigenval=eigenval,
        gmm_summary=gmm_summary,
        config=config,
    )
