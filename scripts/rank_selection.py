"""Quantify the trade-off that defines the rank cut.

Ranks the major cluster's components by case/control ratio, then walks the
cumulative sets: for each k, the top-k components are merged into one group, the
posteriors are recomputed and renormalized, and samples are reassigned by argmax
-- the same operation the subcluster stage performs, so the reported metrics
describe the sets that would actually be produced.

Two quantities are traded off. ``GWAS_Neff`` is the effective sample size, which
rises as more components are included. Residual genetic spread rises too, and is
reported in two bases that are never compared with each other: ``RGV_Global`` on
the global PCA's PC1-PC2, and ``RGV_Mainland`` on a PCA fitted to the major
cluster, over as many axes as the caller asks for. ``rgv_basis`` picks which one
the Pareto front and the distance-to-ideal score are computed from.

The global PCA's leading axes separate the major cluster from the other regions,
so within it they are nearly constant; a major-cluster PCA spends its axes on the
structure that actually remains. That is the reason for the second basis.

This module produces evidence, not a decision. A cut may be forced by the
caller; otherwise the Pareto optimum is reported.

Inputs
------
Cohort assignment results, the merge map, the fitted mixture, case/control lists.

Outputs
-------
Component rank table, cumulative metrics, decision table, and a trade-off figure
(PNG + PDF).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.rank import plot_rank_tradeoff
from scripts.plotting.style import THEME_RANK, figure_context, save_figure
from scripts.common import gwas_neff as _gwas_neff
from scripts.common import to_numeric_array, to_numeric_series
from scripts.common import pc12_case_ctrl_separation as _case_ctrl_separation_pc12
from scripts.common import pc12_rgv as _global_rgv_pc12
from scripts.common import resolve_pc_columns, rgv as _rgv


class RankSelectionOutput(NamedTuple):
    """Rank-progression tables, the recommended cut, and the figure paths."""

    rank_table: pd.DataFrame
    cumulative_metrics: pd.DataFrame
    decision_table: pd.DataFrame
    recommended_rank: int | None
    output_dir: Path
    rank_table_path: Path
    cumulative_table_path: Path
    decision_table_path: Path
    figure_path: Path | None


@dataclass(frozen=True)
class RankSelectionConfig:
    """Configuration for the rank-selection trade-off analysis."""

    output_dir: str | Path = "results/03_rank_selection"
    rank_table_file: str = "component_rank_table.tsv"
    cumulative_table_file: str = "rank_cumulative_metrics.tsv"
    decision_table_file: str = "rank_decision_table.tsv"
    figure_file: str = "rank_selection_tradeoff.png"

    # How many of the ranked mainland components to walk. None means "all of
    # them", discovered from the mainland component list rather than stated as a
    # literal -- the count is a property of the fitted model (17 under the
    # previous float32 run, 16 under float64), so hard-coding it goes stale
    # silently every time the model changes.
    max_rank: int | None = None

    case_label: str = "Case"
    control_label: str = "Control"

    # Residual spread is reported in two bases. RGV_Global is fixed at the global
    # PCA's PC1-PC2 so it stays comparable with earlier runs; RGV_Mainland is
    # computed over this many leading axes of the major-cluster PCA, and only
    # exists when `mainland_coordinates` is supplied.
    mainland_rgv_n_pcs: int = 2

    # Which basis the Pareto front, the normalised utility and the recommended
    # rank are computed from. Defaults to "global" so the stage behaves exactly
    # as before unless a caller opts in.
    rgv_basis: Literal["global", "mainland"] = "global"

    # Override the Pareto-detected recommended rank.  When set, the figure
    # annotation and decision-table Is_Recommended column will use this value
    # instead of the automatically computed optimum. None => use the Pareto
    # optimum, i.e. let the data choose the cut.
    forced_recommended_rank: int | None = None

    save_plot: bool = True
    show_plot: bool = False
    figure_dpi: int = 600
    verbose: bool = True


def _extract_major_cluster_components(merge_map: pd.DataFrame) -> list[int]:
    if "GMM_Component" not in merge_map.columns:
        raise KeyError("merge_map must contain GMM_Component column.")
    if "Is_Mainland_Merged_Cluster" not in merge_map.columns:
        raise KeyError("merge_map must contain Is_Mainland_Merged_Cluster column.")

    mask = merge_map["Is_Mainland_Merged_Cluster"].astype(bool)
    vals = to_numeric_series(merge_map.loc[mask, "GMM_Component"]).dropna().astype(int)
    return sorted(set(int(v) for v in vals.tolist()))


def _resolve_pc12_columns(df: pd.DataFrame) -> tuple[str, str]:
    preferred_pairs = [
        ("PC1_AVG", "PC2_AVG"),
        ("PC1", "PC2"),
    ]
    for c1, c2 in preferred_pairs:
        if c1 in df.columns and c2 in df.columns:
            return c1, c2

    pc1_candidates = [c for c in df.columns if str(c).upper().startswith("PC1")]
    pc2_candidates = [c for c in df.columns if str(c).upper().startswith("PC2")]
    if pc1_candidates and pc2_candidates:
        return str(pc1_candidates[0]), str(pc2_candidates[0])

    raise KeyError("df_results must contain PC1/PC2 columns (e.g., PC1_AVG and PC2_AVG).")


def _resolve_model_pc_columns(
    df: pd.DataFrame,
    gmm_model: Any,
    gmm_summary: dict[str, Any] | None,
) -> list[str]:
    """Return the PC column names that match the GMM model's feature dimensionality."""
    if isinstance(gmm_summary, dict):
        cols = [c for c in gmm_summary.get("pc_columns_used", []) if c in df.columns]
        if len(cols) >= 2:
            return cols
    n_features = int(getattr(gmm_model, "n_features_in_", 2))
    candidates = sorted(
        [c for c in df.columns if str(c).upper().startswith("PC") and str(c).upper().endswith("_AVG")],
        key=lambda c: int("".join(ch for ch in str(c).split("_AVG")[0] if ch.isdigit()) or "0"),
    )
    if len(candidates) >= n_features:
        return candidates[:n_features]
    any_pc = [c for c in df.columns if str(c).upper().startswith("PC")][:n_features]
    if len(any_pc) < n_features:
        raise KeyError(f"Cannot find {n_features} PC columns in df_results for GMM projection.")
    return any_pc


def _mainland_axes(
    mainland_coordinates: pd.DataFrame,
    n_pcs: int,
) -> list[str]:
    """The leading `n_pcs` PC columns of the major-cluster projection."""
    if "IID" not in mainland_coordinates.columns:
        raise KeyError("mainland_coordinates must contain an IID column.")
    axes = resolve_pc_columns(mainland_coordinates)
    if len(axes) < n_pcs:
        raise ValueError(
            f"mainland_coordinates carries {len(axes)} PC columns but "
            f"mainland_rgv_n_pcs is {n_pcs}."
        )
    return axes[:n_pcs]


def _align_mainland(
    df: pd.DataFrame,
    mainland_coordinates: pd.DataFrame,
    axes: list[str],
) -> np.ndarray:
    """Mainland coordinates as an array row-aligned with `df`, NaN where absent.

    A left merge on IID rather than a reindex: the projection covers only the
    major cluster, so it is expected to be missing rows, and the caller checks
    that the missing ones are never retained.
    """
    right = mainland_coordinates.loc[:, ["IID", *axes]].copy()
    right["IID"] = right["IID"].astype(str)
    if right["IID"].duplicated().any():
        n_dup = int(right["IID"].duplicated().sum())
        raise ValueError(f"mainland_coordinates has {n_dup} duplicated IIDs.")

    merged = df.loc[:, ["IID"]].merge(right, on="IID", how="left", validate="one_to_one")
    return merged[axes].to_numpy(dtype=np.float64, copy=False)


def _pareto_front_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-optimal points for minimizing x and maximizing y."""
    n = int(len(x))
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i]))
        if np.any(dominated):
            keep[i] = False
    return np.where(keep)[0]


def _safe_minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    span = a_max - a_min
    if (not np.isfinite(span)) or span <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - a_min) / span


def run_rank_selection(
    *,
    df_results: pd.DataFrame,
    merge_map: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    gmm_model: Any,
    gmm_summary: dict[str, Any] | None = None,
    mainland_coordinates: pd.DataFrame | None = None,
    config: RankSelectionConfig | None = None,
) -> RankSelectionOutput:
    """Walk the cumulative sets of major-cluster components ranked by case/control ratio.

    Per-cluster ranking uses direct pre-merge MAP assignment from df_results
    (Assigned_Merged_Cluster) to compute case/ctrl ratios for sorting.

    `mainland_coordinates` is an optional frame carrying IID plus the PC columns
    of a major-cluster PCA; supplying it adds the RGV_Mainland column. It is a
    frame rather than a path because every other input to this stage is one.

    Cumulative trade-off metrics (case/ctrl counts, Neff, residual spread) use
    the same composite posterior recomputation the subcluster stage performs: for each k,
    the top-k clusters are merged into a composite group, posteriors are
    re-normalized, and assignment is via argmax.  This ensures consistency
    with the subcluster stage's sample counts.
    """

    config = config or RankSelectionConfig()

    if "IID" not in df_results.columns:
        raise KeyError("df_results must contain IID column.")
    if "Assigned_Merged_Cluster" not in df_results.columns:
        raise KeyError("df_results must contain Assigned_Merged_Cluster column.")

    pc1_col, pc2_col = _resolve_pc12_columns(df_results)
    major_cluster_component_ids = _extract_major_cluster_components(merge_map)
    if len(major_cluster_component_ids) == 0:
        raise ValueError("No mainland clusters identified in merge_map.")

    df = df_results.copy()
    df["IID"] = df["IID"].astype(str)
    df["Assigned_Merged_Cluster"] = to_numeric_series(df["Assigned_Merged_Cluster"]).astype("Int64")
    df[pc1_col] = to_numeric_series(df[pc1_col])
    df[pc2_col] = to_numeric_series(df[pc2_col])

    case_set = set(str(x) for x in case_iids)
    ctrl_set = set(str(x) for x in control_iids)

    df["__is_case"] = df["IID"].isin(list(case_set))
    df["__is_ctrl"] = df["IID"].isin(list(ctrl_set))

    all_cluster_ids = sorted(
        set(int(x) for x in df["Assigned_Merged_Cluster"].dropna().astype(int).tolist())
        | set(int(x) for x in major_cluster_component_ids)
    )

    case_counts: dict[int, int] = {}
    ctrl_counts: dict[int, int] = {}
    total_counts: dict[int, int] = {}
    ratio_map: dict[int, float] = {}

    for cid in all_cluster_ids:
        mask = df["Assigned_Merged_Cluster"] == int(cid)
        case_n = int((mask & df["__is_case"]).sum())
        ctrl_n = int((mask & df["__is_ctrl"]).sum())
        total_n = int(mask.sum())

        case_counts[int(cid)] = case_n
        ctrl_counts[int(cid)] = ctrl_n
        total_counts[int(cid)] = total_n

        if ctrl_n > 0:
            ratio = float(case_n) / float(ctrl_n)
        elif case_n > 0:
            ratio = float("inf")
        else:
            ratio = 0.0
        ratio_map[int(cid)] = ratio

    ranked_components = sorted(
        [int(cid) for cid in major_cluster_component_ids if int(cid) in ratio_map],
        key=lambda cid: (-float(ratio_map[cid]), int(cid)),
    )

    n_mainland = len(ranked_components)
    max_rank = n_mainland if config.max_rank is None else min(int(config.max_rank), n_mainland)
    if max_rank <= 0:
        raise ValueError("No mainland clusters available for ranking.")

    rank_table_records: list[dict[str, Any]] = []
    for rank, cid in enumerate(ranked_components[:max_rank], start=1):
        ratio = ratio_map[int(cid)]
        rank_table_records.append(
            {
                "Rank": int(rank),
                "Cluster": int(cid),
                f"{config.case_label}_Count_MAP": int(case_counts[int(cid)]),
                f"{config.control_label}_Count_MAP": int(ctrl_counts[int(cid)]),
                "Total_Count_MAP": int(total_counts[int(cid)]),
                "Case_Ctrl_Ratio": float(ratio),
            }
        )
    rank_table = pd.DataFrame.from_records(rank_table_records)

    cum_records: list[dict[str, Any]] = []
    selected_clusters = ranked_components[:max_rank]

    # Pre-compute raw posteriors once; the subcluster stage does the same.
    pc_cols_model = _resolve_model_pc_columns(df, gmm_model, gmm_summary)
    x_study = df[pc_cols_model].to_numpy(dtype=np.float64, copy=False)
    probs_original = gmm_model.predict_proba(x_study).astype(np.float64, copy=False)
    old_k = int(probs_original.shape[1])
    is_case_arr = df["__is_case"].to_numpy(dtype=bool, copy=False)
    is_ctrl_arr = df["__is_ctrl"].to_numpy(dtype=bool, copy=False)
    pc1_arr = to_numeric_array(df[pc1_col])
    pc2_arr = to_numeric_array(df[pc2_col])

    mainland_axes: list[str] = []
    mainland_xy: np.ndarray | None = None
    if mainland_coordinates is not None:
        mainland_axes = _mainland_axes(mainland_coordinates, int(config.mainland_rgv_n_pcs))
        mainland_xy = _align_mainland(df, mainland_coordinates, mainland_axes)
    elif config.rgv_basis == "mainland":
        raise ValueError(
            'rgv_basis="mainland" needs mainland_coordinates; none was supplied.'
        )

    for k in range(1, max_rank + 1):
        included_clusters = selected_clusters[:k]
        included_set = set(int(c) for c in included_clusters)
        remaining = [c for c in range(old_k) if c not in included_set]

        # Build composite posterior: col 0 = mainland subcluster (sum of included),
        # cols 1..m = individual remaining components, then re-normalize row-wise.
        n_groups = 1 + len(remaining)
        probs_k = np.zeros((probs_original.shape[0], n_groups), dtype=np.float64)
        probs_k[:, 0] = probs_original[:, sorted(included_set)].sum(axis=1)
        for i, cid in enumerate(remaining, start=1):
            probs_k[:, i] = probs_original[:, cid]
        row_sums = probs_k.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        probs_k /= row_sums

        in_sub = np.argmax(probs_k, axis=1) == 0

        case_n = int((in_sub & is_case_arr).sum())
        ctrl_n = int((in_sub & is_ctrl_arr).sum())
        total_n = int(in_sub.sum())

        neff = _gwas_neff(case_n, ctrl_n)

        finite_mask = in_sub & np.isfinite(pc1_arr) & np.isfinite(pc2_arr)
        all_xy = np.stack([pc1_arr[finite_mask], pc2_arr[finite_mask]], axis=1)
        rgv_global = _global_rgv_pc12(all_xy)

        rgv_mainland = float("nan")
        if mainland_xy is not None:
            covered = np.isfinite(mainland_xy).all(axis=1)
            # The projection covers the major cluster, and the retained set is
            # nested inside it, so a retained sample without coordinates means the
            # two files disagree. Dropping it would compute the spread and Neff on
            # different sample sets, which is exactly the kind of silent mismatch
            # this stage exists to avoid.
            missing = int((in_sub & ~covered).sum())
            if missing:
                raise ValueError(
                    f"rank {k}: {missing} of {int(in_sub.sum())} retained samples have no "
                    f"mainland coordinates. The projection must cover every sample the "
                    f"walk can retain."
                )
            rgv_mainland = _rgv(mainland_xy[in_sub])

        # Diagnostic only -- reported next to the two selection metrics, never
        # optimised against. See scripts.common.pc12_case_ctrl_separation for
        # why minimising it would be the wrong objective.
        separation = _case_ctrl_separation_pc12(
            np.stack([pc1_arr[finite_mask & is_case_arr], pc2_arr[finite_mask & is_case_arr]], axis=1),
            np.stack([pc1_arr[finite_mask & is_ctrl_arr], pc2_arr[finite_mask & is_ctrl_arr]], axis=1),
        )

        cum_records.append(
            {
                "Included_Max_Rank": int(k),
                "Included_Cluster_Count": int(len(included_clusters)),
                "Included_Clusters": ",".join(str(int(x)) for x in included_clusters),
                f"{config.case_label}_Count": int(case_n),
                f"{config.control_label}_Count": int(ctrl_n),
                "Case_Control_Ratio": (float(case_n) / float(ctrl_n)) if ctrl_n > 0 else np.nan,
                "Total_Count": int(total_n),
                "GWAS_Neff": float(neff),
                "RGV_Global": float(rgv_global),
                "RGV_Mainland": float(rgv_mainland),
                "PC12_CaseCtrl_Mahalanobis": float(separation.mahalanobis),
                "PC12_CaseCtrl_HotellingT2": float(separation.hotelling_t2),
                "PC12_CaseCtrl_P": float(separation.p_value),
            }
        )

    cumulative_metrics = pd.DataFrame.from_records(cum_records)

    # ===== Decision table =====
    decision_table = cumulative_metrics.copy()
    decision_table = decision_table.sort_values("Included_Max_Rank").reset_index(drop=True)

    # The Pareto front, the normalisation and the recommendation all read one
    # basis; the other is carried alongside for reference only. The two are on
    # different scales and must never be mixed within a comparison.
    rgv_column = "RGV_Mainland" if config.rgv_basis == "mainland" else "RGV_Global"
    neff_arr = decision_table["GWAS_Neff"].to_numpy(dtype=np.float64, copy=False)
    het_arr = decision_table[rgv_column].to_numpy(dtype=np.float64, copy=False)

    delta_neff = np.diff(neff_arr, prepend=np.nan)
    delta_het = np.diff(het_arr, prepend=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        neff_gain_per_het = delta_neff / delta_het

    neff_norm = _safe_minmax_norm(neff_arr)
    het_norm = _safe_minmax_norm(het_arr)
    utility_score = neff_norm - het_norm

    valid_mask = np.isfinite(neff_arr) & np.isfinite(het_arr)
    pareto_mask = np.zeros_like(valid_mask, dtype=bool)
    dist_to_ideal = np.full_like(neff_arr, np.nan, dtype=np.float64)
    recommended_rank: int | None = None

    if bool(np.any(valid_mask)):
        valid_idx = np.where(valid_mask)[0]
        x_valid = het_arr[valid_mask]
        y_valid = neff_arr[valid_mask]

        pf_local = _pareto_front_indices(x_valid, y_valid)
        pf_global = valid_idx[pf_local]
        pareto_mask[pf_global] = True

        x_norm = _safe_minmax_norm(x_valid)
        y_norm = _safe_minmax_norm(y_valid)
        # Ideal corner is (min heterogeneity, max Neff) -> (0, 1)
        d_valid = np.sqrt((x_norm - 0.0) ** 2 + (1.0 - y_norm) ** 2)
        dist_to_ideal[valid_idx] = d_valid

        if pf_global.size > 0:
            d_pf = dist_to_ideal[pf_global]
            k_pf = decision_table.loc[pf_global, "Included_Max_Rank"].to_numpy(dtype=int, copy=False)
            order = np.lexsort((k_pf, d_pf))
            best_idx = int(pf_global[int(order[0])])
            recommended_rank = int(decision_table.loc[best_idx, "Included_Max_Rank"])

    # Override with user-specified rank when provided.
    if config.forced_recommended_rank is not None:
        forced = int(config.forced_recommended_rank)
        valid_ranks = decision_table["Included_Max_Rank"].tolist()
        if forced not in valid_ranks:
            raise ValueError(
                f"forced_recommended_rank={forced} is not in the valid rank range "
                f"{valid_ranks[0]}..{valid_ranks[-1]}."
            )
        recommended_rank = forced

    decision_table["Delta_Neff"] = delta_neff
    decision_table["Delta_RGV"] = delta_het
    decision_table["Neff_Gain_per_RGV"] = neff_gain_per_het
    decision_table["Neff_Norm"] = neff_norm
    decision_table["RGV_Norm"] = het_norm
    decision_table["Utility_Neff_minus_RGV"] = utility_score
    decision_table["Is_Pareto"] = pareto_mask
    decision_table["Distance_To_Ideal"] = dist_to_ideal
    decision_table["Is_Recommended"] = decision_table["Included_Max_Rank"].eq(recommended_rank)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_table_path = out_dir / str(config.rank_table_file)
    cumulative_table_path = out_dir / str(config.cumulative_table_file)
    decision_table_path = out_dir / str(config.decision_table_file)
    rank_table.to_csv(rank_table_path, sep="\t", index=False)
    cumulative_metrics.to_csv(cumulative_table_path, sep="\t", index=False)
    decision_table.to_csv(decision_table_path, sep="\t", index=False)

    figure_path: Path | None = None
    if bool(config.save_plot) or bool(config.show_plot):
        with figure_context(THEME_RANK):
            fig = plot_rank_tradeoff(
                decision_table=decision_table,
                rgv_column=rgv_column,
                mainland_axes=mainland_axes,
                config=config,
            )
            if bool(config.save_plot):
                figure_path = out_dir / str(config.figure_file)
                save_figure(fig, figure_path, dpi=int(config.figure_dpi))
            if bool(config.show_plot):
                plt.show()
            else:
                plt.close(fig)


    if bool(config.verbose):
        print("\n" + "=" * 92)
        print("RANK SELECTION: EFFECTIVE SAMPLE SIZE vs RESIDUAL SPREAD".center(92))
        print("=" * 92)
        print(f"Mainland clusters ranked (top {max_rank}): {selected_clusters}")
        _rec_source = "(forced)" if config.forced_recommended_rank is not None else "(Pareto-auto)"
        print(f"Recommended rank k   : {recommended_rank}  {_rec_source}")
        print(f"Rank table saved      : {rank_table_path}")
        print(f"Cumulative table saved: {cumulative_table_path}")
        print(f"Decision table saved  : {decision_table_path}")
        if figure_path is not None:
            print(f"Figure saved          : {figure_path}")
        print("-" * 92)
        print(decision_table.to_string(index=False))
        print("=" * 92 + "\n")

    return RankSelectionOutput(
        rank_table=rank_table,
        cumulative_metrics=cumulative_metrics,
        decision_table=decision_table,
        recommended_rank=recommended_rank,
        output_dir=out_dir,
        rank_table_path=rank_table_path,
        cumulative_table_path=cumulative_table_path,
        decision_table_path=decision_table_path,
        figure_path=figure_path,
    )
