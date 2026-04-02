from __future__ import annotations

"""STEP6: High-confidence cluster-specific all-PC KDE + statistics.

This module filters the high-confidence STEP5 subset to a selected merged cluster,
computes per-PC summary statistics and significance tests, and writes:
- a FID/IID TSV without a header row
- a log file
- a KDE figure across all detected PCs

The cluster selection and confidence threshold are parameterized through config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast
import logging
import math
import re
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind

from scripts.our_assignment import OURAssignmentConfig, _PLOT_STYLE_RC


class ClusterAllPCsKDEOutput(NamedTuple):
    output_dir: Path
    log_file: Path
    fid_iid_file: Path
    figure_png: Path
    df_cluster: pd.DataFrame
    tests_df: pd.DataFrame


@dataclass(frozen=True)
class ClusterAllPCsKDEConfig:
    cluster_id: int = 2
    threshold: float = 0.95
    output_dir: str = "results/06_cluster2_all_pcs_kde_allpcs"
    save_tables: bool = True
    save_plot: bool = True
    show_plot: bool = False
    output_file: str = "cluster{cluster_id}_highconf_fid_iid_conf_ge_{threshold_tag}.tsv"
    figure_file: str = "all_pcs_cluster{cluster_id}_case_control_kde_conf_ge_{threshold_tag}.png"
    log_file: str = "cluster{cluster_id}_all_pcs_conf_ge_{threshold_tag}.log"
    case_label: str = "Case"
    control_label: str = "Control"
    n_cols: int = 5
    bbj_color: str = "#1F78B4"
    case_color: str = "#E31A1C"
    alpha: float = 0.65
    verbose: bool = True


def _pc_sort_key(col: str) -> int:
    match = re.match(r"^PC(\d+)(?:_AVG)?$", str(col))
    return int(match.group(1)) if match else 10**9


def _resolve_all_pc_columns(df: pd.DataFrame) -> list[str]:
    pc_cols = sorted([c for c in df.columns if re.match(r"^PC\d+(?:_AVG)?$", str(c))], key=_pc_sort_key)
    if not pc_cols:
        raise RuntimeError("No PC columns detected (expected e.g., PC1_AVG ...).")
    return pc_cols


def _format_threshold_tag(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "p")


def _bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if not mask.any():
        return out
    p = pvals[mask]
    n = p.size
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[mask] = tmp
    return out


def _setup_file_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def _safe_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else float(np.std(x, ddof=0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _format_template(template: str, *, cluster_id: int, threshold: float) -> str:
    threshold_str = f"{float(threshold):.2f}"
    threshold_tag = threshold_str.replace(".", "p")
    return str(template).format(
        cluster_id=int(cluster_id),
        threshold=float(threshold),
        threshold_str=threshold_str,
        threshold_tag=threshold_tag,
        threshold_pct=int(round(float(threshold) * 100.0)),
    )


def run_cluster_all_pcs_kde(
    *,
    df_results_highconf: pd.DataFrame,
    our_samples: pd.DataFrame,
    our_case_iids: list[str],
    our_ctrl_iids: list[str],
    eigenval: pd.DataFrame | None = None,
    step4_config: OURAssignmentConfig | None = None,
    config: ClusterAllPCsKDEConfig | None = None,
) -> ClusterAllPCsKDEOutput:
    step4_config = step4_config or OURAssignmentConfig()
    config = config or ClusterAllPCsKDEConfig()

    df_high = pd.DataFrame(df_results_highconf).copy()
    our_df = pd.DataFrame(our_samples).copy()
    if "IID" not in df_high.columns or "IID" not in our_df.columns:
        raise KeyError("Both df_results_highconf and our_samples must contain 'IID'.")
    if "Assigned_Merged_Cluster" not in df_high.columns or "Assignment_Confidence" not in df_high.columns:
        raise KeyError("df_results_highconf must contain 'Assigned_Merged_Cluster' and 'Assignment_Confidence'.")

    df_high["IID"] = df_high["IID"].astype(str)
    our_df["IID"] = our_df["IID"].astype(str)

    df_join = our_df.merge(df_high[["IID", "Assigned_Merged_Cluster", "Assignment_Confidence"]], on="IID", how="inner")
    cluster_id = int(config.cluster_id)
    threshold = float(config.threshold)
    mask_cluster = pd.to_numeric(df_join["Assigned_Merged_Cluster"], errors="coerce").fillna(-1).astype(int) == cluster_id
    df_cluster = df_join.loc[mask_cluster].copy()
    df_cluster = df_cluster.loc[pd.to_numeric(df_cluster["Assignment_Confidence"], errors="coerce") >= threshold].copy()

    case_set = set(map(str, our_case_iids))
    ctrl_set = set(map(str, our_ctrl_iids))
    is_case = df_cluster["IID"].astype(str).isin(case_set).to_numpy()
    is_ctrl = df_cluster["IID"].astype(str).isin(ctrl_set).to_numpy()
    df_case = df_cluster.loc[is_case].copy()
    df_ctrl = df_cluster.loc[is_ctrl].copy()

    pc_cols = _resolve_all_pc_columns(df_join)
    case_label = str(getattr(step4_config, "case_label", config.case_label))
    ctrl_label = str(getattr(step4_config, "control_label", config.control_label))

    var_lookup: dict[int, float] = {}
    if eigenval is not None:
        try:
            ev = pd.DataFrame(eigenval)
            if {"PC", "variance_explained"}.issubset(ev.columns):
                for pc_num, var in zip(ev["PC"].to_numpy(), ev["variance_explained"].to_numpy()):
                    var_lookup[int(pc_num)] = float(var)
        except Exception:
            pass

    pc_stats: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for col in pc_cols:
        pc_num = int(_pc_sort_key(col))
        var = float(var_lookup.get(pc_num, np.nan))
        x_ctrl = pd.to_numeric(df_ctrl[col], errors="coerce").to_numpy(dtype=float, copy=False)
        x_case = pd.to_numeric(df_case[col], errors="coerce").to_numpy(dtype=float, copy=False)
        x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
        x_case = x_case[np.isfinite(x_case)]
        s_ctrl = _safe_stats(x_ctrl)
        s_case = _safe_stats(x_case)
        pc_stats.append(
            {
                "pc": f"PC{pc_num}",
                "var": var,
                "n_control": int(x_ctrl.size),
                "ctrl": s_ctrl,
                "n_case": int(x_case.size),
                "case": s_case,
                "diff": float(s_case["mean"] - s_ctrl["mean"]) if np.isfinite(s_case["mean"]) and np.isfinite(s_ctrl["mean"]) else np.nan,
            }
        )
        t_res = cast(Any, ttest_ind(x_case, x_ctrl, equal_var=False, nan_policy="omit"))
        u_res = cast(Any, mannwhitneyu(x_case, x_ctrl, alternative="two-sided"))
        test_rows.append(
            {
                "pc": f"PC{pc_num}",
                "var": var,
                "t_stat": float(t_res.statistic),
                "p_t": float(t_res.pvalue),
                "u_stat": float(u_res.statistic),
                "p_u": float(u_res.pvalue),
            }
        )

    tests_df = pd.DataFrame(test_rows).sort_values("pc", key=lambda s: s.str.replace("PC", "", regex=False).astype(int))
    tests_df["p_t_adj"] = _bh_fdr_adjust(tests_df["p_t"].to_numpy())
    tests_df["p_u_adj"] = _bh_fdr_adjust(tests_df["p_u"].to_numpy())
    tests_df["reject_t"] = tests_df["p_t_adj"] <= 0.05
    tests_df["reject_u"] = tests_df["p_u_adj"] <= 0.05

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _format_threshold_tag(threshold)
    log_path = out_dir / _format_template(config.log_file, cluster_id=cluster_id, threshold=threshold)
    logger = _setup_file_logger("cluster_all_pcs_kde", log_path)

    if bool(config.verbose):
        logger.info(f">>> ANALYZING ALL PC DISTRIBUTIONS FOR HIGH CONFIDENCE SAMPLES (CLUSTER {cluster_id})...")
        logger.info(f"   -> confidence_threshold = {threshold:.2f}")
        logger.info(f"   -> output_dir = {out_dir.as_posix()}")
        logger.info(f"   -> Cluster {cluster_id} samples: {len(df_cluster)} / {len(df_join)}")
        logger.info(f"   -> Case samples (Cluster {cluster_id}): {len(df_case)}")
        logger.info(f"   -> Control samples (Cluster {cluster_id}): {len(df_ctrl)}")
        logger.info(f"   -> Total PCs to analyze: {len(pc_cols)}")
        logger.info("   -> Running statistical tests...")
        logger.info("   -> Applying FDR correction (Benjamini-Hochberg method) to all tests...")
        logger.info("   -> FDR correction complete.")
        logger.info(f"      • Significant by t-test: {int(tests_df['reject_t'].sum())} PC(s)")
        logger.info(f"      • Significant by Mann-Whitney U: {int(tests_df['reject_u'].sum())} PC(s)")
        logger.info("=" * 80)
        logger.info(f"{'ALL PC DISTRIBUTION STATISTICS - CLUSTER ' + str(cluster_id):^80}")
        logger.info("=" * 80)
        for r in pc_stats:
            var_str = f" ({r['var']:.1%})" if np.isfinite(r["var"]) else ""
            logger.info(f"\n{r['pc']}{var_str} Statistics:")
            logger.info(f"  {ctrl_label} (n={r['n_control']}):")
            logger.info(f"    Mean: {r['ctrl']['mean']:8.4f}  |  Std: {r['ctrl']['std']:8.4f}")
            logger.info(f"    Min:  {r['ctrl']['min']:8.4f}  |  Max: {r['ctrl']['max']:8.4f}")
            logger.info(f"  {case_label} (n={r['n_case']}):")
            logger.info(f"    Mean: {r['case']['mean']:8.4f}  |  Std: {r['case']['std']:8.4f}")
            logger.info(f"    Min:  {r['case']['min']:8.4f}  |  Max: {r['case']['max']:8.4f}")
            logger.info(f"  Difference ({case_label} - {ctrl_label}): {r['diff']:8.4f}")
        logger.info("=" * 80)
        logger.info(f"\n>>> STATISTICAL TESTING ({case_label} vs {ctrl_label} - Cluster {cluster_id}) - ALL PCs")
        logger.info(">>> FDR Correction Applied: Benjamini-Hochberg Method (alpha=0.05)")
        for _, row in tests_df.iterrows():
            var_str = f" ({row['var']:.1%})" if np.isfinite(row['var']) else ""
            logger.info(f"\n--- {row['pc']}{var_str} Analysis ---")
            logger.info("  Welch's t-test (Mean):")
            logger.info(f"    t-statistic:        {row['t_stat']:8.4f}")
            logger.info(f"    p-value (raw):      {row['p_t']:.4e}")
            logger.info(f"    p-value (FDR-adj):  {row['p_t_adj']:.4e}")
            logger.info(f"    Significant:        {'Yes **' if bool(row['reject_t']) else 'No'}")
            logger.info("  Mann-Whitney U (Rank Sum):")
            logger.info(f"    U-statistic:        {row['u_stat']:8.4f}")
            logger.info(f"    p-value (raw):      {row['p_u']:.4e}")
            logger.info(f"    p-value (FDR-adj):  {row['p_u_adj']:.4e}")
            logger.info(f"    Significant:        {'Yes **' if bool(row['reject_u']) else 'No'}")
        logger.info("=" * 80)
        logger.info(f"\n>>> SUMMARY: SIGNIFICANT PCs (FDR-corrected, alpha = 0.05)")
        logger.info("=" * 80)
        sig_t = tests_df.loc[tests_df["reject_t"]]
        sig_u = tests_df.loc[tests_df["reject_u"]]
        logger.info("\nWelch's t-test (Mean Comparison):")
        logger.info(f"   -> {len(sig_t)} significant PC(s) found:" if len(sig_t) else "   -> No significant PCs found")
        if len(sig_t):
            for _, r in sig_t.iterrows():
                var_str = f" ({r['var']:.1%})" if np.isfinite(r["var"]) else ""
                logger.info(f"      • {r['pc']}{var_str}: p_raw={r['p_t']:.4e}, p_adj={r['p_t_adj']:.4e}")
        logger.info("\nMann-Whitney U Test (Rank/Median Comparison):")
        logger.info(f"   -> {len(sig_u)} significant PC(s) found:" if len(sig_u) else "   -> No significant PCs found")
        if len(sig_u):
            for _, r in sig_u.iterrows():
                var_str = f" ({r['var']:.1%})" if np.isfinite(r["var"]) else ""
                logger.info(f"      • {r['pc']}{var_str}: p_raw={r['p_u']:.4e}, p_adj={r['p_u_adj']:.4e}")
        logger.info("\nMULTIPLE TESTING CORRECTION INFO".center(80))
        logger.info(f"   -> Total tests per method: {len(pc_cols)}")
        logger.info("   -> FDR method: Benjamini-Hochberg")
        logger.info("   -> Significance level: alpha = 0.05")
        logger.info("=" * 80)
        logger.info(f">>> ALL PC ANALYSIS (CLUSTER {cluster_id}) COMPLETED SUCCESSFULLY.\n")

    fid_iid_file = out_dir / _format_template(config.output_file, cluster_id=cluster_id, threshold=threshold)
    if bool(config.save_tables):
        df_cluster.loc[:, ["FID", "IID"]].drop_duplicates().assign(
            FID=lambda d: d["FID"].astype(str),
            IID=lambda d: d["IID"].astype(str),
        ).to_csv(fid_iid_file, sep="\t", index=False, header=False)

    figure_png = out_dir / _format_template(config.figure_file, cluster_id=cluster_id, threshold=threshold)
    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=2.0)
        plt.rcParams.update(dict(_PLOT_STYLE_RC))
        n_cols = int(config.n_cols)
        n_rows = int(math.ceil(len(pc_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.8 * n_rows), squeeze=False)
        fig.subplots_adjust(wspace=0.30, hspace=0.45, left=0.06, right=0.98, top=0.82, bottom=0.07)
        for i, col in enumerate(pc_cols, start=1):
            ax = axes[(i - 1) // n_cols][(i - 1) % n_cols]
            x_ctrl = pd.to_numeric(df_ctrl[col], errors="coerce").to_numpy(dtype=float, copy=False)
            x_case = pd.to_numeric(df_case[col], errors="coerce").to_numpy(dtype=float, copy=False)
            x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
            x_case = x_case[np.isfinite(x_case)]
            if x_ctrl.size > 1:
                sns.kdeplot(x=x_ctrl, color=str(config.bbj_color), fill=True, alpha=float(config.alpha), linewidth=2.5, ax=ax)
            if x_case.size > 1:
                sns.kdeplot(x=x_case, color=str(config.case_color), fill=True, alpha=float(config.alpha), linewidth=2.5, ax=ax)
            pc_num = int(_pc_sort_key(col))
            var = float(var_lookup.get(pc_num, np.nan))
            ax.set_title(f"PC{pc_num}{f' ({var:.1%})' if np.isfinite(var) else ''}", pad=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        for j in range(len(pc_cols) + 1, n_rows * n_cols + 1):
            axes[(j - 1) // n_cols][(j - 1) % n_cols].axis("off")
        fig.suptitle(
            f"All PC Distribution Analysis: {case_label} vs {ctrl_label} - Cluster {cluster_id} Only (confidence $\\geq$ {threshold:.2f})",
            fontweight="bold",
            y=0.995,
        )
        fig.legend(
            handles=[
                mpatches.Patch(color=str(config.bbj_color), alpha=float(config.alpha), label=ctrl_label),
                mpatches.Patch(color=str(config.case_color), alpha=float(config.alpha), label=case_label),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=2,
            frameon=False,
            fontsize=22,
        )
        fig.savefig(figure_png, bbox_inches="tight", dpi=300)
        if bool(config.show_plot):
            plt.show()
        else:
            plt.close(fig)

    return ClusterAllPCsKDEOutput(
        output_dir=out_dir,
        log_file=log_path,
        fid_iid_file=fid_iid_file,
        figure_png=figure_png,
        df_cluster=df_cluster,
        tests_df=tests_df,
    )
