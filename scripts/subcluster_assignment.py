"""Reassign the study cohort with one subcluster treated as a single group.

Takes the selected major-cluster components, sums their posteriors into one
composite group, keeps every other component separate, renormalizes and
reassigns by argmax. A borderline sample therefore joins the subcluster only
when its joint posterior over those components beats every single outside
component -- which is what makes this different from thresholding a per-component
posterior.

Inputs
------
The fitted mixture, the clustered reference panel, study samples, case/control
lists, the major-cluster component ids, and the components to exclude.

Outputs
-------
Per-sample posterior table, group summary JSON, and an assignment figure.
"""

# NOTE on the `pyright: ignore` comments below:
# pandas-stubs types these column-name arguments as SequenceNotStr, whose
# index() must accept keyword arguments -- list.index is positional-only, so
# no list literal can ever satisfy it. Passing a list of column names is the
# documented pandas usage; the stub is wrong, not the call.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from sklearn.mixture import GaussianMixture

from scripts.common import STORE_DTYPE
from scripts.common import build_distinct_palette as _build_distinct_palette
from scripts.plotting.assignment import (
    PCWindow,
    ReferenceCloud,
    plot_subcluster_assignment,
)
from scripts.plotting.style import THEME_SUBCLUSTER, figure_context, save_figure


class SubclusterAssignmentOutput(NamedTuple):
    df_results: pd.DataFrame
    subcluster_components: list[int]
    probs_subcluster: np.ndarray
    assigned_component: np.ndarray
    assignment_confidence: np.ndarray
    cluster_stats: pd.DataFrame
    output_dir: Path
    assignment_tsv: Path | None
    figure_path: Path | None


# Backward-compatible alias.
@dataclass(frozen=True)
class SubclusterAssignmentConfig:
    output_dir: str | Path = "results/04_subcluster_variants/narrow"
    save_plot: bool = True
    show_plot: bool = False
    save_tables: bool = True

    output_file: str = "subcluster_posterior_probabilities.tsv"
    figure_file: str = "subcluster_assignment_overview.png"
    #: Per-group case/control counts. Previously these existed only as a
    #: rendered table inside the figure, at roughly 3 pt on a printed page.
    statistics_file: str = "subcluster_group_statistics.tsv"

    # Display name/color for the merged customize cluster.
    group_label: str = "Mainland Subcluster"
    group_color: str = "#E67E22"

    # If customize_cluster is provided, it has priority.
    subcluster_components: tuple[int, ...] | None = None
    # Remove these clusters from mainland_premerge_cluster_ids by default.
    # No default membership: (13, 15) was a leftover from one specific old run
    # and would silently mis-run any caller that forgot to pass this. The
    # notebook always supplies it explicitly, derived from the rank cut.
    exclude_cluster_ids: int | tuple[int, ...] | list[int] = ()

    reference_color: str = "#B0B0B0"
    reference_alpha: float = 0.20
    study_point_size: float = 60.0

    case_label: str = "Case"
    control_label: str = "Control"

    verbose: bool = True


# Backward-compatible alias with professional naming.
def _normalize_cluster_ids(ids: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if ids is None:
        return tuple()
    if isinstance(ids, (int, np.integer)):
        return (int(ids),)
    return tuple(int(x) for x in ids)


def _resolve_pc_columns_for_projection(
    *,
    study_samples: pd.DataFrame,
    gmm_model: GaussianMixture,
    gmm_summary: dict[str, Any] | None,
) -> list[str]:
    pc_cols_used: list[str] = []

    if isinstance(gmm_summary, dict):
        pc_cols_used = [c for c in gmm_summary.get("pc_columns_used", []) if c in study_samples.columns]

    if not pc_cols_used:
        n_features = int(getattr(gmm_model, "n_features_in_", 2))
        pc_candidates = [c for c in study_samples.columns if str(c).startswith("PC") and str(c).endswith("_AVG")]

        def _pc_num(col: str) -> int:
            try:
                return int(str(col).split("_AVG")[0].replace("PC", ""))
            except Exception:
                return 10**9

        pc_cols_used = sorted(pc_candidates, key=_pc_num)[: max(2, n_features)]

    if len(pc_cols_used) < 2:
        raise RuntimeError("At least two PC columns are required for assignment visualization.")

    return list(pc_cols_used)


def _build_premerge_component_palette(reference_samples_gmm: pd.DataFrame, n_clusters: int) -> dict[int, str]:
    if "GMM_Cluster" in reference_samples_gmm.columns:
        labels = reference_samples_gmm["GMM_Cluster"].to_numpy(dtype=np.int32, copy=False)
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


def _build_subcluster_components(
    *,
    major_cluster_component_ids: list[int],
    n_components: int,
    config: SubclusterAssignmentConfig,
) -> list[int]:
    if config.subcluster_components is not None:
        candidate = [int(x) for x in config.subcluster_components]
    else:
        exclude_ids = _normalize_cluster_ids(config.exclude_cluster_ids)
        exclude = set(int(x) for x in exclude_ids)
        candidate = [int(x) for x in major_cluster_component_ids if int(x) not in exclude]

    candidate = sorted(set(candidate))
    candidate = [x for x in candidate if 0 <= int(x) < int(n_components)]
    if not candidate:
        raise ValueError("customize_cluster is empty after filtering; please adjust config.")

    return candidate


def run_subcluster_assignment(
    *,
    gmm_model: GaussianMixture,
    reference_samples_gmm: pd.DataFrame,
    study_samples: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    major_cluster_component_ids: list[int],
    reference_samples_background: pd.DataFrame | None = None,
    eigenval: pd.DataFrame | None = None,
    gmm_summary: dict[str, Any] | None = None,
    config: SubclusterAssignmentConfig | None = None,
) -> SubclusterAssignmentOutput:
    config = config or SubclusterAssignmentConfig()
    exclude_cluster_ids = _normalize_cluster_ids(config.exclude_cluster_ids)

    if study_samples.empty:
        raise ValueError("study_samples is empty; cannot run subcluster assignment.")
    if "IID" not in study_samples.columns:
        raise ValueError("study_samples must contain an 'IID' column.")

    pc_cols_used = _resolve_pc_columns_for_projection(study_samples=study_samples, gmm_model=gmm_model, gmm_summary=gmm_summary)

    x_study = study_samples[pc_cols_used].to_numpy(dtype=np.float64, copy=False)
    probs_original = gmm_model.predict_proba(x_study).astype(np.float64, copy=False)

    old_k = int(probs_original.shape[1])
    subcluster_component_ids = _build_subcluster_components(
        major_cluster_component_ids=major_cluster_component_ids,
        n_components=int(old_k),
        config=config,
    )

    subcluster_component_set = set(int(x) for x in subcluster_component_ids)
    remaining_component_ids = [int(cid) for cid in range(int(old_k)) if int(cid) not in subcluster_component_set]

    # Recompute posterior by treating customize_cluster as ONE merged cluster,
    # while keeping all remaining pre-merge components as separate clusters.
    n_groups = 1 + int(len(remaining_component_ids))
    probs_subcluster = np.zeros((probs_original.shape[0], n_groups), dtype=np.float64)
    probs_subcluster[:, 0] = probs_original[:, subcluster_component_ids].sum(axis=1)
    for i, cid in enumerate(remaining_component_ids, start=1):
        probs_subcluster[:, i] = probs_original[:, int(cid)]

    row_sums = probs_subcluster.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_subcluster = probs_subcluster / row_sums

    # Assignment is based on the recomputed composite posterior: a sample belongs to
    # Mainland Subcluster iff the merged group posterior is highest.  This naturally
    # absorbs borderline samples whose joint subcluster probability exceeds any single
    # non-subcluster component.
    assigned_idx = np.argmax(probs_subcluster, axis=1).astype(np.int32)
    group_label = str(config.group_label)
    assigned_component = np.asarray(
        [(-1 if int(i) == 0 else int(remaining_component_ids[int(i) - 1])) for i in assigned_idx],
        dtype=np.int32,
    )
    assigned_group = np.asarray(
        [(group_label if int(i) == 0 else f"Cluster {remaining_component_ids[int(i) - 1]}") for i in assigned_idx],
        dtype=object,
    )
    assignment_conf = np.max(probs_subcluster, axis=1).astype(STORE_DTYPE)

    fid_col = "#FID" if "#FID" in study_samples.columns else ("FID" if "FID" in study_samples.columns else None)
    meta_cols = [c for c in [fid_col, "IID"] if c is not None and c in study_samples.columns]

    # meta_cols may be empty, in which case `[] + pc_cols` is just `pc_cols`.
    df_results = cast(pd.DataFrame, study_samples[meta_cols + pc_cols_used[:2]]).copy()
    if fid_col == "#FID":
        df_results = df_results.rename(columns={"#FID": "FID"})

    df_results["Assigned_Mainland_Subcluster_Group"] = assigned_group
    df_results["Assigned_Component_ID_If_Not_Subcluster"] = assigned_component
    # Posterior that the sample belongs to the subcluster, under the recomputed model.
    df_results["Prob_Mainland_Subcluster"] = probs_subcluster[:, 0].astype(STORE_DTYPE)
    # Confidence = max posterior under the recomputed (merged) model.
    df_results["Assignment_Confidence"] = assignment_conf
    for i, cid in enumerate(remaining_component_ids, start=1):
        df_results[f"Prob_Cluster_{cid}"] = probs_subcluster[:, i].astype(STORE_DTYPE)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_tsv: Path | None = None
    if bool(config.save_tables):
        assignment_tsv = out_dir / str(config.output_file)
        df_results.to_csv(assignment_tsv, sep="\t", index=False)

        summary = {
            "customize_cluster": [int(x) for x in subcluster_component_ids],
            "mainland_subcluster_component_ids": [int(x) for x in subcluster_component_ids],
            "exclude_cluster_ids": [int(x) for x in exclude_cluster_ids],
            "remaining_clusters": [int(x) for x in remaining_component_ids],
            "remaining_component_ids": [int(x) for x in remaining_component_ids],
            "custom_group_label": str(config.group_label),
            "custom_group_color": str(config.group_color),
            "n_groups": int(n_groups),
            "rows": int(df_results.shape[0]),
            "workflow_name": "mainland_subcluster_reassignment",
        }
        with (out_dir / "subcluster_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Computed unconditionally: the group statistics are a deliverable, not a
    # drawing detail. They used to live inside `if save_plot or show_plot:`,
    # so turning plotting off silently returned an empty cluster_stats.
    reference_pc1 = reference_samples_gmm[pc_cols_used[0]].to_numpy(dtype=np.float64, copy=False)
    reference_pc2 = reference_samples_gmm[pc_cols_used[1]].to_numpy(dtype=np.float64, copy=False)

    # The background shows every reference sample, including the ones HDBSCAN
    # removed. The axes are still framed on the modelled panel above, so the
    # few far outliers fall outside the view instead of shrinking the
    # structure the figure exists to show.
    _background = reference_samples_gmm if reference_samples_background is None else reference_samples_background
    background_pc1 = _background[pc_cols_used[0]].to_numpy(dtype=np.float64, copy=False)
    background_pc2 = _background[pc_cols_used[1]].to_numpy(dtype=np.float64, copy=False)
    study_pc1 = study_samples[pc_cols_used[0]].to_numpy(dtype=np.float64, copy=False)
    study_pc2 = study_samples[pc_cols_used[1]].to_numpy(dtype=np.float64, copy=False)

    study_iid = study_samples["IID"].astype(str)
    case_set = set(str(x) for x in case_iids)
    ctrl_set = set(str(x) for x in control_iids)
    is_case = study_iid.isin(list(case_set)).to_numpy()
    is_ctrl = study_iid.isin(list(ctrl_set)).to_numpy()
    is_other = ~(is_case | is_ctrl)

    try:
        var1 = float(eigenval.loc[eigenval["PC"] == 1, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
        var2 = float(eigenval.loc[eigenval["PC"] == 2, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
    except Exception:
        var1, var2 = 0.0, 0.0

    full_palette = _build_premerge_component_palette(reference_samples_gmm=reference_samples_gmm, n_clusters=int(old_k))
    group_order = [group_label] + [f"Cluster {cid}" for cid in remaining_component_ids]
    group_palette: dict[str, str] = {group_label: str(config.group_color)}
    for cid in remaining_component_ids:
        group_palette[f"Cluster {cid}"] = full_palette.get(int(cid), "#999999")

    all_x = np.concatenate([reference_pc1, study_pc1])
    all_y = np.concatenate([reference_pc2, study_pc2])
    x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    max_span = max(x_max - x_min, y_max - y_min)
    if max_span == 0:
        max_span = 1.0
    view_span = max_span * 1.05
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    window = PCWindow.from_points(all_x, all_y, pad=1.05, var1=var1, var2=var2)
    reference = ReferenceCloud(
        pc1=background_pc1,
        pc2=background_pc2,
        color=str(config.reference_color),
        alpha=float(config.reference_alpha),
    )

    df_cc = pd.DataFrame({"Group": assigned_group, "Case": is_case.astype(int), "Control": is_ctrl.astype(int)})
    stats = pd.DataFrame(index=group_order)  # pyright: ignore[reportArgumentType]
    stats["Case"] = df_cc.groupby("Group")["Case"].sum().reindex(group_order).fillna(0).astype(int)
    stats["Control"] = df_cc.groupby("Group")["Control"].sum().reindex(group_order).fillna(0).astype(int)
    stats["Total"] = df_cc.groupby("Group").size().reindex(group_order).fillna(0).astype(int)
    cluster_stats = stats.reset_index().rename(columns={"index": "Group"})
    if bool(config.save_tables):
        cluster_stats.to_csv(out_dir / str(config.statistics_file), sep="\t", index=False)

    figure_path: Path | None = None
    if bool(config.save_plot) or bool(config.show_plot):
        with figure_context(THEME_SUBCLUSTER):
            fig = plot_subcluster_assignment(
                study_pc1=study_pc1,
                study_pc2=study_pc2,
                is_case=is_case,
                is_ctrl=is_ctrl,
                is_other=is_other,
                assigned_group=assigned_group,
                assignment_conf=assignment_conf,
                stats=stats,
                group_order=group_order,
                group_palette=group_palette,
                group_label=group_label,
                subcluster_component_ids=list(subcluster_component_ids),
                var1=var1,
                var2=var2,
                window=window,
                reference=reference,
                config=config,
            )
            if bool(config.save_plot):
                figure_path = out_dir / str(config.figure_file)
                save_figure(fig, figure_path, dpi=400)
            if bool(config.show_plot):
                plt.show()
            else:
                plt.close(fig)

    if bool(config.verbose):
        print("\n" + "=" * 80)
        print("SUBCLUSTER REASSIGNMENT".center(80))
        print("=" * 80)
        print(f"  mainland_subcluster_ids: {subcluster_component_ids}")
        print(f"  remaining_component_ids: {remaining_component_ids}")
        print(f"  excluded_cluster_ids   : {list(exclude_cluster_ids)}")
        print(f"  output_dir             : {out_dir}")
        if assignment_tsv is not None:
            print(f"  assignment_tsv         : {assignment_tsv}")
        print("=" * 80)

    return SubclusterAssignmentOutput(
        df_results=df_results,
        subcluster_components=subcluster_component_ids,
        probs_subcluster=probs_subcluster,
        assigned_component=assigned_component,
        assignment_confidence=assignment_conf,
        cluster_stats=cluster_stats,
        output_dir=out_dir,
        assignment_tsv=assignment_tsv,
        figure_path=figure_path,
    )


