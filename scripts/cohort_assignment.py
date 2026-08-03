"""Assign the study cohort to reference-panel components by posterior.

Projects the study samples into the fitted mixture, takes ``predict_proba``,
optionally group-sums the posteriors under a label map, renormalizes, and
assigns by argmax. ``Assignment_Confidence`` is the retained maximum posterior,
so a sample sitting between components is identifiable rather than silently
placed.

Z-scaled training is rejected rather than approximated: the training scaler is
not persisted, so projecting new samples through it is not possible.

Inputs
------
The fitted mixture, the clustered reference panel, study samples, case/control
IID lists, a label map, and the merge map.

Outputs
-------
Per-sample posterior table, component rank table, and a four-panel figure.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from sklearn.mixture import GaussianMixture

from scripts.common import (
    to_numeric_series,
    STORE_DTYPE,
    build_distinct_palette as _build_distinct_palette,
)
from scripts.plotting.assignment import (
    PCWindow,
    ReferenceCloud,
    plot_cohort_assignment,
)
from scripts.plotting.style import THEME_ASSIGNMENT, figure_context, save_figure


class CohortAssignmentOutput(NamedTuple):
    """Container for OUR assignment outputs."""

    df_results: pd.DataFrame
    probs_merged_our: np.ndarray
    assigned_merged: np.ndarray
    assignment_confidence: np.ndarray
    cluster_stats: pd.DataFrame
    output_dir: Path
    assignment_tsv: Path | None
    figure_path: Path | None


@dataclass(frozen=True)
class CohortAssignmentConfig:
    """Configuration for OUR cohort assignment to merged GMM clusters.

    Parameters
    ----------
    output_dir : str
        Output directory for assignment results.
    save_plot : bool
        Save 2x2 composite figure.
    show_plot : bool
        Show figure (useful in interactive sessions).
    save_tables : bool
        Save TSV outputs.
    output_file : str
        Filename for posterior TSV.
    mainland_cluster_rank_file : str
        Filename for Mainland sample-level cluster/rank TSV aligned to panel D.
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

    output_dir: str | Path = "results/02_cohort_assignment"
    save_plot: bool = True
    show_plot: bool = False
    save_tables: bool = True

    output_file: str = "cohort_posterior_probabilities.tsv"
    component_rank_file: str = "major_cluster_component_ranks.tsv"
    #: Per-cluster case/control counts and ranks. Previously these existed
    #: only as a rendered table panel inside the figure.
    statistics_file: str = "cohort_cluster_statistics.tsv"
    figure_file: str = "cohort_assignment_overview.png"

    reference_color: str = "#B0B0B0"
    reference_alpha: float = 0.20
    study_point_size: float = 60.0

    case_label: str = "Case"
    control_label: str = "Control"

    verbose: bool = True


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
                merge_map[["Merged_Cluster", "Merged_Cluster_Color"]]  # pyright: ignore[reportCallIssue]
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


def _build_premerge_component_palette(
    *,
    reference_samples_gmm: pd.DataFrame,
    n_clusters: int,
) -> dict[int, str]:
    """Component colors matching the mixture stage, for pre-merge assignment mode."""
    if "GMM_Cluster" in reference_samples_gmm.columns:
        labels = reference_samples_gmm["GMM_Cluster"].to_numpy(dtype=np.int32, copy=False)
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


def _extract_major_cluster_component_ids(merge_map: pd.DataFrame | None) -> list[int]:
    """Major-cluster component ids, when the merge map carries that metadata."""
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

    vals = to_numeric_series(merge_map.loc[mask, "GMM_Component"]).dropna().astype(int)
    if vals.empty:
        return []

    return sorted(set(int(v) for v in vals.tolist()))


@dataclass(frozen=True)
class ClusterRanking:
    """Per-cluster case/control counts and the case-ratio ranking derived from them.

    Computed from the assignment alone -- no figure involved. It used to live
    inside the plotting branch, which meant ``save_plot=False`` returned empty
    statistics and skipped writing ``major_cluster_component_ranks.tsv``
    entirely, even with ``save_tables=True``.
    """

    stats: pd.DataFrame
    ratio_map: dict[int, float]
    rank_map: dict[int, int]
    priority_ids: list[int]
    priority_set: set[int]
    ordered_cluster_ids: list[int]
    case_counts_by_cluster: dict[int, Any]
    control_counts_by_cluster: dict[int, Any]
    total_counts_by_cluster: dict[int, Any]
    all_cluster_ids: list[int]
    major_cluster_component_ids: list[int]


def compute_cluster_ranking(
    *,
    assigned_merged: np.ndarray,
    is_case: np.ndarray,
    is_ctrl: np.ndarray,
    new_k: int,
    merge_map: pd.DataFrame,
) -> ClusterRanking:
    """Rank the major cluster's components by case/control ratio, descending."""
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

    major_cluster_component_ids = _extract_major_cluster_component_ids(merge_map)
    priority_ids = [cid for cid in major_cluster_component_ids if cid in case_counts_by_cluster]
    priority_set = set(priority_ids)

    # Compute Case/Control ratio for mainland ordering and ranking.
    ratio_map: dict[int, float] = {}
    for cid in range(int(new_k)):
        case_n = int(case_counts_by_cluster[cid])
        ctrl_n = int(control_counts_by_cluster[cid])
        if ctrl_n > 0:
            ratio = float(case_n) / float(ctrl_n)
        elif case_n > 0:
            ratio = float("inf")
        else:
            ratio = 0.0
        ratio_map[cid] = ratio

    # Mainland clusters are sorted by Case/Ctrl ratio (descending);
    # Rank follows this order as 1, 2, 3, ... (ascending).
    mainland_ratio_sorted = sorted(
        priority_ids,
        key=lambda cid: (-float(ratio_map[cid]), int(cid)),
    )
    rank_map: dict[int, int] = {}
    for rank, cid in enumerate(mainland_ratio_sorted, start=1):
        rank_map[cid] = rank

    # Sort mainland clusters by Rank (ascending: 1, 2, 3, ...).
    priority_ids_sorted_by_rank = sorted(priority_ids, key=lambda cid: (rank_map[cid], int(cid)))
    all_cluster_ids = list(range(int(new_k)))
    ordered_cluster_ids = priority_ids_sorted_by_rank + [cid for cid in all_cluster_ids if cid not in priority_set]

    return ClusterRanking(
        stats=stats,
        ratio_map=ratio_map,
        rank_map=rank_map,
        priority_ids=priority_ids,
        priority_set=priority_set,
        ordered_cluster_ids=ordered_cluster_ids,
        case_counts_by_cluster=case_counts_by_cluster,
        control_counts_by_cluster=control_counts_by_cluster,
        total_counts_by_cluster=total_counts_by_cluster,
        all_cluster_ids=all_cluster_ids,
        major_cluster_component_ids=major_cluster_component_ids,
    )


def run_cohort_assignment(
    *,
    gmm_model: GaussianMixture,
    reference_samples_gmm: pd.DataFrame,
    study_samples: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    label_map: dict[int, int],
    reference_samples_background: pd.DataFrame | None = None,
    merge_map: pd.DataFrame | None = None,
    eigenval: pd.DataFrame | None = None,
    gmm_summary: dict[str, Any] | None = None,
    training_use_zscale: bool = False,
    config: CohortAssignmentConfig | None = None,
) -> CohortAssignmentOutput:
    config = config or CohortAssignmentConfig()

    if bool(training_use_zscale):
        raise RuntimeError(
            "training_use_zscale=True is not supported for projection without storing training scaler stats."
        )

    if study_samples.empty:
        raise ValueError("study_samples is empty; cannot assign the cohort.")

    if "IID" not in study_samples.columns:
        raise ValueError("study_samples must contain an 'IID' column.")

    if not isinstance(label_map, dict) or len(label_map) == 0:
        raise ValueError("label_map is missing/empty; run component merging to obtain the component->merged mapping.")

    pc_cols_used = _resolve_pc_columns_for_projection(study_samples=study_samples, gmm_model=gmm_model, gmm_summary=gmm_summary)

    x_study = study_samples[pc_cols_used].to_numpy(dtype=np.float64, copy=False)
    probs_original = gmm_model.predict_proba(x_study).astype(np.float64, copy=False)

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
    assignment_conf = np.max(probs_merged_our, axis=1).astype(STORE_DTYPE)

    fid_col = "#FID" if "#FID" in study_samples.columns else ("FID" if "FID" in study_samples.columns else None)
    meta_cols = [c for c in [fid_col, "IID"] if c is not None and c in study_samples.columns]

    # meta_cols may be empty, in which case `[] + pc_cols` is just `pc_cols`.
    df_results = cast(pd.DataFrame, study_samples[meta_cols + pc_cols_used[:2]]).copy()
    if fid_col == "#FID":
        df_results = df_results.rename(columns={"#FID": "FID"})

    df_results["Assigned_Merged_Cluster"] = assigned_merged
    df_results["Assignment_Confidence"] = assignment_conf
    for m in range(int(new_k)):
        df_results[f"Prob_Merge_Cluster_{m}"] = probs_merged_our[:, m].astype(STORE_DTYPE)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_tsv: Path | None = None
    mainland_samples_tsv: Path | None = None
    mainland_cluster_rank_tsv: Path | None = None
    if bool(config.save_tables):
        assignment_tsv = out_dir / str(config.output_file)
        df_results.to_csv(assignment_tsv, sep="\t", index=False)

    # Setup and ranking run unconditionally. Both feed data deliverables, and
    # both used to sit inside `if save_plot or show_plot:`.
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

    if is_premerge_identity_mode:
        merged_cluster_palette = _build_premerge_component_palette(
            reference_samples_gmm=reference_samples_gmm,
            n_clusters=int(new_k),
        )
    else:
        merged_cluster_palette = _build_merged_cluster_palette(merge_map=merge_map, new_k=int(new_k))
    hue_order = list(range(int(new_k)))

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

    ranking = compute_cluster_ranking(
        assigned_merged=assigned_merged,
        is_case=is_case,
        is_ctrl=is_ctrl,
        new_k=int(new_k),
        merge_map=merge_map,
    )
    stats = ranking.stats
    priority_set = ranking.priority_set
    rank_map = ranking.rank_map
    ordered_cluster_ids = ranking.ordered_cluster_ids
    cluster_stats = stats.loc[ordered_cluster_ids].reset_index().rename(columns={"index": "Cluster"})

    figure_path: Path | None = None
    mainland_cluster_rank_tsv: Path | None = None
    if bool(config.save_plot) or bool(config.show_plot):
        with figure_context(THEME_ASSIGNMENT):
            fig = plot_cohort_assignment(
                study_pc1=study_pc1,
                study_pc2=study_pc2,
                is_case=is_case,
                is_ctrl=is_ctrl,
                is_other=is_other,
                assigned_merged=assigned_merged,
                assignment_conf=assignment_conf,
                merged_cluster_palette=merged_cluster_palette,
                hue_order=hue_order,
                is_premerge_identity_mode=is_premerge_identity_mode,
                stats=stats,
                ratio_map=ranking.ratio_map,
                rank_map=rank_map,
                priority_set=priority_set,
                ordered_cluster_ids=ordered_cluster_ids,
                case_counts_by_cluster=ranking.case_counts_by_cluster,
                control_counts_by_cluster=ranking.control_counts_by_cluster,
                total_counts_by_cluster=ranking.total_counts_by_cluster,
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

    if bool(config.save_tables):
        # Per-sample component ranking, restricted to the major cluster and
        # ordered as in the assignment figure's ranking panel.
        #
        # This used to be nested inside `if save_plot or show_plot:`, so a run
        # with save_tables=True but save_plot=False wrote no ranking table at
        # all and returned an empty cluster_stats. It depends on the ranking,
        # not on the figure.
        major_cluster_mask = (
            np.isin(assigned_merged, np.array(sorted(priority_set), dtype=int))
            if priority_set
            else np.zeros_like(assigned_merged, dtype=bool)
        )
        mainland_cluster_rank_tsv = out_dir / str(config.component_rank_file)
        sample_id_col = "IID" if "IID" in df_results.columns else ("FID" if "FID" in df_results.columns else None)
        if sample_id_col is not None:
            mainland_cluster_rank_df = pd.DataFrame(
                {
                    "Sample_ID": df_results.loc[major_cluster_mask, sample_id_col].astype(str).to_numpy(),
                    "Original_Cluster": [f"Cluster {int(cid)}" for cid in assigned_merged[major_cluster_mask]],
                    "Rank": [int(rank_map[int(cid)]) for cid in assigned_merged[major_cluster_mask]],
                }
            )
            mainland_cluster_rank_df["_cluster_id"] = [int(cid) for cid in assigned_merged[major_cluster_mask]]
            mainland_cluster_rank_df = mainland_cluster_rank_df.sort_values(
                by=["Rank", "_cluster_id", "Sample_ID"],
                ascending=[True, True, True],
                kind="stable",
            ).drop(columns=["_cluster_id"])
            mainland_cluster_rank_df.to_csv(mainland_cluster_rank_tsv, sep="\t", index=False)

        # The per-cluster statistics themselves, previously only ever rendered
        # as a table panel inside the figure at roughly 3 pt on a printed page.
        cluster_stats.to_csv(out_dir / str(config.statistics_file), sep="\t", index=False)

    if bool(getattr(config, "verbose", True)):
        print("\n" + "=" * 80)
        mode_name = "PRE-MERGE GMM COMPONENTS" if is_premerge_identity_mode else "MERGED GMM CLUSTERS"
        print(f"COHORT ASSIGNMENT TO {mode_name}".center(80))
        print("=" * 80)
        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  output_dir            : {out_dir}")
        print(f"  save_tables           : {bool(config.save_tables)}")
        print(f"  save_plot             : {bool(config.save_plot)}")
        print(f"  show_plot             : {bool(config.show_plot)}")
        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  cohort rows           : {study_samples.shape[0]:,}")
        print(f"  assigned_clusters (K) : {int(new_k)}")
        if assignment_tsv is not None:
            print(f"  assignment_tsv        : {assignment_tsv}")
        if mainland_samples_tsv is not None:
            print(f"  mainland_samples_tsv  : {mainland_samples_tsv}")
        if mainland_cluster_rank_tsv is not None:
            print(f"  mainland_cluster_rank : {mainland_cluster_rank_tsv}")
        print("=" * 80)

    return CohortAssignmentOutput(
        df_results=df_results,
        probs_merged_our=probs_merged_our,
        assigned_merged=assigned_merged,
        assignment_confidence=assignment_conf,
        cluster_stats=cluster_stats,
        output_dir=out_dir,
        assignment_tsv=assignment_tsv,
        figure_path=figure_path,
    )
