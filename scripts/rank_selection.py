"""Quantify the trade-off that defines the rank cut.

Ranks the major cluster's components by case/control ratio, then walks the
cumulative sets: for each k, the top-k components are merged into one group, the
posteriors are recomputed and renormalized, and samples are reassigned by argmax
-- the same operation the subcluster stage performs, so the reported metrics
describe the sets that would actually be produced.

Two quantities are traded off. ``GWAS_Neff`` is the effective sample size,
which rises as more components are included. ``PC12_RGV`` is the residual
genetic spread of the retained samples, which rises too. The Pareto front and a
distance-to-ideal score summarise the frontier.

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
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import gwas_neff as _gwas_neff
from scripts.common import pc12_rgv as _overall_heterogeneity_pc12


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

    output_dir: str = "results/03_rank_selection"
    rank_table_file: str = "component_rank_table.tsv"
    cumulative_table_file: str = "rank_cumulative_metrics.tsv"
    decision_table_file: str = "rank_decision_table.tsv"
    figure_file: str = "rank_selection_tradeoff.png"
    figure_pdf_file: str = "rank_selection_tradeoff.pdf"

    # How many of the ranked mainland components to walk. None means "all of
    # them", discovered from the mainland component list rather than stated as a
    # literal -- the count is a property of the fitted model (17 under the
    # previous float32 run, 16 under float64), so hard-coding it goes stale
    # silently every time the model changes.
    max_rank: int | None = None

    case_label: str = "Case"
    control_label: str = "Control"

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
    vals = pd.to_numeric(merge_map.loc[mask, "GMM_Component"], errors="coerce").dropna().astype(int)
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


def _case_ctrl_mahalanobis_pc12(case_xy: np.ndarray, ctrl_xy: np.ndarray) -> float:
    if case_xy.ndim != 2 or ctrl_xy.ndim != 2 or case_xy.shape[1] != 2 or ctrl_xy.shape[1] != 2:
        return float("nan")

    n_case = int(case_xy.shape[0])
    n_ctrl = int(ctrl_xy.shape[0])
    if n_case < 2 or n_ctrl < 2:
        return float("nan")

    case_mean = case_xy.mean(axis=0)
    ctrl_mean = ctrl_xy.mean(axis=0)
    delta = case_mean - ctrl_mean

    cov_case = np.cov(case_xy, rowvar=False)
    cov_ctrl = np.cov(ctrl_xy, rowvar=False)

    dof = n_case + n_ctrl - 2
    if dof <= 0:
        return float("nan")

    pooled_cov = (((n_case - 1) * cov_case) + ((n_ctrl - 1) * cov_ctrl)) / float(dof)
    pooled_cov = np.asarray(pooled_cov, dtype=np.float64)

    # Small ridge for numerical stability under near-collinearity.
    pooled_cov = pooled_cov + np.eye(2, dtype=np.float64) * 1e-8

    try:
        inv_cov = np.linalg.pinv(pooled_cov)
        d2 = float(delta.T @ inv_cov @ delta)
    except Exception:
        return float("nan")

    if not np.isfinite(d2):
        return float("nan")
    return float(np.sqrt(max(0.0, d2)))


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
    config: RankSelectionConfig | None = None,
) -> RankSelectionOutput:
    """Walk the cumulative sets of major-cluster components ranked by case/control ratio.

    Per-cluster ranking uses direct pre-merge MAP assignment from df_results
    (Assigned_Merged_Cluster) to compute case/ctrl ratios for sorting.

    Cumulative trade-off metrics (case/ctrl counts, Neff, heterogeneity) use
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
    df["Assigned_Merged_Cluster"] = pd.to_numeric(df["Assigned_Merged_Cluster"], errors="coerce").astype("Int64")
    df[pc1_col] = pd.to_numeric(df[pc1_col], errors="coerce")
    df[pc2_col] = pd.to_numeric(df[pc2_col], errors="coerce")

    case_set = set(str(x) for x in case_iids)
    ctrl_set = set(str(x) for x in control_iids)

    df["__is_case"] = df["IID"].isin(case_set)
    df["__is_ctrl"] = df["IID"].isin(ctrl_set)

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
    pc1_arr = pd.to_numeric(df[pc1_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    pc2_arr = pd.to_numeric(df[pc2_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

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
        heterogeneity = _overall_heterogeneity_pc12(all_xy)

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
                "PC12_AllSample_Heterogeneity": float(heterogeneity),
            }
        )

    cumulative_metrics = pd.DataFrame.from_records(cum_records)

    # ===== Decision table =====
    decision_table = cumulative_metrics.copy()
    decision_table = decision_table.sort_values("Included_Max_Rank").reset_index(drop=True)

    neff_arr = decision_table["GWAS_Neff"].to_numpy(dtype=np.float64, copy=False)
    het_arr = decision_table["PC12_AllSample_Heterogeneity"].to_numpy(dtype=np.float64, copy=False)

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
    decision_table["Delta_Heterogeneity"] = delta_het
    decision_table["Neff_Gain_per_Heterogeneity"] = neff_gain_per_het
    decision_table["Neff_Norm"] = neff_norm
    decision_table["Heterogeneity_Norm"] = het_norm
    decision_table["Utility_NeffMinusHet"] = utility_score
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
        # ── Nature-style rcParams ─────────────────────────────────────────
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "axes.linewidth": 0.9,
                "axes.labelsize": 16,
                "axes.titlesize": 15,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 13,
                "figure.titlesize": 18,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 4.0,
                "ytick.major.size": 4.0,
                "xtick.major.width": 0.9,
                "ytick.major.width": 0.9,
            }
        )

        # ── Figure & axes layout ──────────────────────────────────────────
        fig = plt.figure(figsize=(18.0, 9.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.8], wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        ax_note = fig.add_subplot(gs[0, 1])
        fig.subplots_adjust(left=0.075, right=0.990, top=0.882, bottom=0.115)

        rank_vals = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
        neff_vals = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
        het_vals = decision_table["PC12_AllSample_Heterogeneity"].to_numpy(dtype=float, copy=False)
        pareto_vals = decision_table["Is_Pareto"].to_numpy(dtype=bool, copy=False)
        rec_vals = decision_table["Is_Recommended"].to_numpy(dtype=bool, copy=False)

        valid_mask = np.isfinite(neff_vals) & np.isfinite(het_vals)
        if bool(np.any(valid_mask)):
            x_valid = het_vals[valid_mask]
            y_valid = neff_vals[valid_mask]
            rank_valid = rank_vals[valid_mask]
            p_valid = pareto_vals[valid_mask]
            r_valid = rec_vals[valid_mask]

            order = np.argsort(rank_valid)
            x_plot = x_valid[order]
            y_plot = y_valid[order]
            rank_plot = rank_valid[order]
            p_plot = p_valid[order]
            r_plot = r_valid[order]

            x_range = float(x_plot.max() - x_plot.min()) if len(x_plot) > 1 else 1.0
            y_range = float(y_plot.max() - y_plot.min()) if len(y_plot) > 1 else 1.0

            # ── All-rank connecting line ──────────────────────────────────
            ax.plot(
                x_plot, y_plot,
                color="#BDBDBD", linewidth=1.2, alpha=0.85,
                zorder=1, solid_capstyle="round",
            )
            # ── All rank scatter points ───────────────────────────────────
            ax.scatter(
                x_plot, y_plot,
                color="#616161", s=60, alpha=0.85,
                edgecolors="white", linewidths=0.7, zorder=2,
            )
            # ── Pareto frontier ───────────────────────────────────────────
            if bool(np.any(p_plot)):
                pf_order = np.argsort(x_plot[p_plot])
                x_pf = x_plot[p_plot][pf_order]
                y_pf = y_plot[p_plot][pf_order]
                ax.plot(
                    x_pf, y_pf,
                    color="#D32F2F", linewidth=1.8, alpha=0.95,
                    zorder=3, solid_capstyle="round",
                )
                ax.scatter(
                    x_pf, y_pf,
                    s=130, facecolors="none",
                    edgecolors="#D32F2F", linewidths=1.6,
                    label="Pareto-optimal frontier", zorder=4,
                )
            # ── Recommended rank highlight ────────────────────────────────
            if bool(np.any(r_plot)):
                rec_idx = int(np.where(r_plot)[0][0])
                ax.scatter(
                    x_plot[r_plot], y_plot[r_plot],
                    marker="D", s=180,
                    facecolors="none", edgecolors="#1565C0", linewidths=2.2,
                    label="Recommended k", zorder=5,
                )
                # build sample-size annotation, placed in lower-right empty space
                rec_k = int(rank_plot[rec_idx])
                rec_row = decision_table[decision_table["Included_Max_Rank"] == rec_k]
                _case_col = f"{config.case_label}_Count"
                _ctrl_col = f"{config.control_label}_Count"
                _case_n = int(rec_row[_case_col].iloc[0]) if _case_col in rec_row.columns else None
                _ctrl_n = int(rec_row[_ctrl_col].iloc[0]) if _ctrl_col in rec_row.columns else None
                _total_n = int(rec_row["Total_Count"].iloc[0]) if "Total_Count" in rec_row.columns else None
                if _case_n is not None and _ctrl_n is not None and _total_n is not None:
                    _ann_lines = [
                        f"k = {rec_k}  (recommended)",
                        f"{config.case_label}: {_case_n:,}  |  {config.control_label}: {_ctrl_n:,}",
                        f"Total: {_total_n:,}  (composite posterior)",
                    ]
                else:
                    _ann_lines = [f"k = {rec_k}  (recommended)"]
                # place annotation box in the lower-right empty area of the plot
                ax.annotate(
                    "\n".join(_ann_lines),
                    xy=(float(x_plot[rec_idx]), float(y_plot[rec_idx])),
                    xycoords="data",
                    xytext=(0.68, 0.12),
                    textcoords="axes fraction",
                    fontsize=11.5, fontweight="normal",
                    color="#0D2B6E",
                    bbox={
                        "boxstyle": "round,pad=0.45",
                        "fc": "#EEF2FF", "ec": "#1565C0",
                        "alpha": 0.97, "lw": 1.1,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#1565C0",
                        "lw": 1.1,
                        "connectionstyle": "arc3,rad=-0.25",
                    },
                )
            # ── Rank number labels (skip recommended k — already annotated) ──
            for i, (xv, yv, kv) in enumerate(zip(x_plot, y_plot, rank_plot)):
                if r_plot[i]:   # recommended k has its own annotation box
                    continue
                ax.text(
                    float(xv) + x_range * 0.007,
                    float(yv) + y_range * 0.012,
                    str(int(kv)),
                    fontsize=11, ha="left", va="bottom",
                    color="#424242", zorder=6,
                )

            # ── Legend + label caption ────────────────────────────────────
            ax.legend(
                loc="lower right", frameon=False,
                handlelength=1.8, handleheight=1.2,
                fontsize=13, borderpad=0.6,
            )
            # Caption explaining the numeric label — italic, top-left, no box
            ax.text(
                0.018, 0.970,
                r"Label = cumulative rank $k$  (top-1 $\ldots$ top-$k$ mainland clusters included)",
                transform=ax.transAxes,
                fontsize=11, color="#757575",
                ha="left", va="top", style="italic",
            )

        ax.set_xlabel(r"Overall Heterogeneity  $H$  (RGV on PC1+PC2)", labelpad=10)
        ax.set_ylabel(r"GWAS  $N_{eff}$", labelpad=10)
        ax.set_title(
            r"Trade-off: Overall Heterogeneity vs. GWAS $N_{eff}$",
            loc="left", fontweight="bold", pad=11, fontsize=15,
        )
        ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.30, color="#9E9E9E")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(axis="both", which="major", length=4.0, width=0.9, pad=5)

        # ── Right panel: Methods / Formulas ──────────────────────────────
        ax_note.set_axis_off()
        ax_note.set_xlim(0.0, 1.0)
        ax_note.set_ylim(0.0, 1.0)

        _BK  = "#212121"   # near-black for headers
        _GR  = "#424242"   # body text
        _DIM = "#757575"   # secondary/dim text
        _X   = 0.05        # left indent

        def _rule(y: float) -> None:
            ax_note.plot([0.0, 1.0], [y, y], color="#E0E0E0", linewidth=0.9)

        # ══ Section 1: N_eff ════════════════════════════════════════════
        _rule(0.978)
        ax_note.text(
            _X, 0.958,
            r"Effective Sample Size  $N_{eff}$",
            fontsize=14.5, fontweight="bold", va="top", ha="left", color=_BK,
        )
        # Step 1
        ax_note.text(
            _X, 0.910,
            "Variance of allele-frequency difference (unbalanced design):",
            fontsize=11.0, fontstyle="italic", va="top", ha="left", color=_DIM,
        )
        ax_note.text(
            _X + 0.04, 0.870,
            r"$\mathrm{Var}=p(1-p)\!\left(\dfrac{1}{N_{\mathrm{case}}}+\dfrac{1}{N_{\mathrm{ctrl}}}\right)$",
            fontsize=13.0, va="top", ha="left", color=_GR,
        )
        # Step 2
        ax_note.text(
            _X, 0.798,
            r"Equate to balanced design  ($N_{eff}/2$ per arm):",
            fontsize=11.0, fontstyle="italic", va="top", ha="left", color=_DIM,
        )
        ax_note.text(
            _X + 0.04, 0.758,
            r"$=\dfrac{2\,p(1-p)}{N_{eff}}$",
            fontsize=13.0, va="top", ha="left", color=_GR,
        )
        # Main result
        ax_note.text(
            _X, 0.695,
            r"$\Rightarrow\;N_{eff}=\dfrac{4\,N_{\mathrm{case}}\,N_{\mathrm{ctrl}}}{"
            r"N_{\mathrm{case}}+N_{\mathrm{ctrl}}}=N_{tot}\cdot\dfrac{4r}{(1+r)^{2}}$",
            fontsize=14.5, va="top", ha="left", color=_BK,
        )
        # Meaning
        ax_note.text(
            _X, 0.612,
            r"Equivalent total $N$ of a balanced (1:1) case-control study",
            fontsize=12.0, va="top", ha="left", color=_GR,
        )
        ax_note.text(
            _X, 0.578,
            "with the same statistical test power.",
            fontsize=12.0, va="top", ha="left", color=_GR,
        )
        ax_note.text(
            _X, 0.542,
            r"($r=N_{\mathrm{case}}/N_{\mathrm{ctrl}},\;\;N_{tot}=N_{\mathrm{case}}+N_{\mathrm{ctrl}}$)",
            fontsize=11.0, va="top", ha="left", color=_DIM,
        )

        _rule(0.505)

        # ══ Section 2: Heterogeneity ════════════════════════════════════
        ax_note.text(
            _X, 0.482,
            r"Overall Heterogeneity  $H$",
            fontsize=14.5, fontweight="bold", va="top", ha="left", color=_BK,
        )
        # Main formula
        ax_note.text(
            _X + 0.04, 0.430,
            r"$H=\left|\,\Sigma_{PC1,PC2}\,\right|^{1/4}$",
            fontsize=15.5, va="top", ha="left", color=_BK,
        )
        # Determinant expansion
        ax_note.text(
            _X, 0.355,
            r"$|\Sigma|=\sigma^{2}_{PC1}\cdot\sigma^{2}_{PC2}-\sigma^{2}_{PC1,PC2}"
            r"=\sigma^{2}_{PC1}\cdot\sigma^{2}_{PC2}(1-\rho^{2})$",
            fontsize=11.5, va="top", ha="left", color=_GR,
        )
        # Closed form
        ax_note.text(
            _X, 0.292,
            r"$\Rightarrow H=(\sigma_{PC1}\cdot\sigma_{PC2})^{1/2}(1-\rho^{2})^{1/4}$",
            fontsize=13.0, va="top", ha="left", color=_GR,
        )
        # Meaning
        ax_note.text(
            _X, 0.220,
            r"Root Generalized Variance (2D): joint spread in PC1/PC2,",
            fontsize=12.0, va="top", ha="left", color=_GR,
        )
        ax_note.text(
            _X, 0.186,
            r"attenuated by inter-PC correlation $\rho$.",
            fontsize=12.0, va="top", ha="left", color=_GR,
        )

        _rule(0.072)

        # ══ Section 3: Counting methods ═══════════════════════════════════
        ax_note.text(
            _X, 0.057,
            "Counting methods:",
            fontsize=10.5, fontweight="bold", va="top", ha="left", color=_BK,
        )
        ax_note.text(
            _X, 0.035,
            r"Rank table: per-component argmax (MAP), as in the assignment figure.",
            fontsize=9.5, fontstyle="italic", va="top", ha="left", color=_DIM,
        )
        ax_note.text(
            _X, 0.011,
            r"Scatter: composite posterior recomputation (top-$k$ merged), as in the subcluster stage.",
            fontsize=9.5, fontstyle="italic", va="top", ha="left", color=_DIM,
        )

        fig.suptitle(
            "Mainland Rank-Cumulative Trade-off Analysis",
            fontsize=18, fontweight="bold", y=0.972,
        )

        if bool(config.save_plot):
            figure_path = out_dir / str(config.figure_file)
            fig.savefig(figure_path, dpi=int(config.figure_dpi), bbox_inches="tight")
            fig.savefig(out_dir / str(config.figure_pdf_file), dpi=300, bbox_inches="tight")

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
