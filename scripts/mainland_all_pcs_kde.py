from __future__ import annotations

"""STEP4 mainland all-PC KDE visualization.

This module mirrors scripts/cluster_all_pcs_kde.py, but the sample selection is
hard-coded to the mainland clusters already defined in STEP4/STEP3 metadata.
It writes a headerless FID/IID file for mainland samples and produces an all-PC
KDE figure comparing case vs control distributions within mainland.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import logging
import math
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind

from scripts.our_assignment import OURAssignmentConfig

from scripts.common import (
    PLOT_STYLE_RC as _PLOT_STYLE_RC,
    bh_fdr_adjust as _bh_fdr_adjust,
    format_template as _format_template,
    pc_sort_key as _pc_sort_key,
    resolve_all_pc_columns as _resolve_all_pc_columns,
    safe_stats as _safe_stats,
    setup_file_logger as _setup_file_logger,
)


class MainlandAllPCsKDEOutput(NamedTuple):
    output_dir: Path
    log_file: Path
    fid_iid_file: Path
    figure_png: Path
    df_mainland: pd.DataFrame
    tests_df: pd.DataFrame


@dataclass(frozen=True)
class MainlandAllPCsKDEConfig:
    output_dir: str = "results/04_our_assignment"
    save_tables: bool = True
    save_plot: bool = True
    show_plot: bool = False
    output_file: str = "mainland_samples.fid_iid.txt"
    figure_file: str = "mainland_all_pcs_kde.png"
    log_file: str = "mainland_all_pcs_kde.log"
    case_label: str = "Case"
    control_label: str = "Control"
    n_cols: int = 5
    bbj_color: str = "#1F78B4"
    case_color: str = "#E31A1C"
    alpha: float = 0.65
    verbose: bool = True


def _format_title(pc_num: int, var: float) -> str:
    return f"PC{pc_num}{f' ({var:.1%})' if np.isfinite(var) else ''}"


def run_mainland_all_pcs_kde(
    *,
    df_results: pd.DataFrame,
    our_samples: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    mainland_cluster_ids: list[int],
    eigenval: pd.DataFrame | None = None,
    step4_config: OURAssignmentConfig | None = None,
    config: MainlandAllPCsKDEConfig | None = None,
) -> MainlandAllPCsKDEOutput:
    step4_config = step4_config or OURAssignmentConfig()
    config = config or MainlandAllPCsKDEConfig()

    df_results = pd.DataFrame(df_results).copy()
    our_df = pd.DataFrame(our_samples).copy()
    if "IID" not in df_results.columns or "IID" not in our_df.columns:
        raise KeyError("Both df_results and our_samples must contain 'IID'.")
    if "Assigned_Merged_Cluster" not in df_results.columns:
        raise KeyError("df_results must contain 'Assigned_Merged_Cluster'.")

    df_results["IID"] = df_results["IID"].astype(str)
    our_df["IID"] = our_df["IID"].astype(str)

    mainland_ids = sorted({int(x) for x in mainland_cluster_ids})
    if not mainland_ids:
        raise ValueError("mainland_cluster_ids is empty; please pass the mainland set from STEP3.")

    df_join = our_df.merge(df_results, on="IID", how="inner", suffixes=("", "_result"))
    df_join["Assigned_Merged_Cluster"] = pd.to_numeric(df_join["Assigned_Merged_Cluster"], errors="coerce").fillna(-1).astype(int)
    df_mainland = df_join.loc[df_join["Assigned_Merged_Cluster"].isin(mainland_ids)].copy()

    case_set = set(map(str, our_case_iids))
    ctrl_set = set(map(str, our_ctrl_iids))
    is_case = df_mainland["IID"].astype(str).isin(case_set).to_numpy()
    is_ctrl = df_mainland["IID"].astype(str).isin(ctrl_set).to_numpy()
    df_case = df_mainland.loc[is_case].copy()
    df_ctrl = df_mainland.loc[is_ctrl].copy()

    pc_cols = _resolve_all_pc_columns(df_join)

    var_lookup: dict[int, float] = {}
    if eigenval is not None:
        try:
            ev = pd.DataFrame(eigenval)
            if {"PC", "variance_explained"}.issubset(ev.columns):
                for pc_num, var in zip(ev["PC"].to_numpy(), ev["variance_explained"].to_numpy()):
                    var_lookup[int(pc_num)] = float(var)
        except Exception:
            pass

    test_rows: list[dict[str, Any]] = []
    for col in pc_cols:
        pc_num = int(_pc_sort_key(col))
        var = float(var_lookup.get(pc_num, np.nan))
        x_ctrl = pd.to_numeric(df_ctrl[col], errors="coerce").to_numpy(dtype=float, copy=False)
        x_case = pd.to_numeric(df_case[col], errors="coerce").to_numpy(dtype=float, copy=False)
        x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
        x_case = x_case[np.isfinite(x_case)]
        t_res = ttest_ind(x_case, x_ctrl, equal_var=False, nan_policy="omit")
        u_res = mannwhitneyu(x_case, x_ctrl, alternative="two-sided")
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
    log_path = out_dir / str(config.log_file)
    logger = _setup_file_logger("mainland_all_pcs_kde", log_path)

    if bool(config.verbose):
        logger.info(">>> ANALYZING ALL PC DISTRIBUTIONS FOR MAINLAND SAMPLES...")
        logger.info(f"   -> output_dir = {out_dir.as_posix()}")
        logger.info(f"   -> mainland clusters = {mainland_ids}")
        logger.info(f"   -> Mainland samples: {len(df_mainland)} / {len(df_join)}")
        logger.info(f"   -> Case samples (Mainland): {len(df_case)}")
        logger.info(f"   -> Control samples (Mainland): {len(df_ctrl)}")
        logger.info(f"   -> Total PCs to analyze: {len(pc_cols)}")
        logger.info("   -> Running statistical tests...")
        logger.info("   -> Applying FDR correction (Benjamini-Hochberg method) to all tests...")
        logger.info("   -> FDR correction complete.")
        logger.info(f"      • Significant by t-test: {int(tests_df['reject_t'].sum())} PC(s)")
        logger.info(f"      • Significant by Mann-Whitney U: {int(tests_df['reject_u'].sum())} PC(s)")
        logger.info("=" * 80)

    fid_iid_file = out_dir / str(config.output_file)
    if bool(config.save_tables):
        df_mainland.loc[:, [c for c in ["FID", "IID"] if c in df_mainland.columns]].drop_duplicates().to_csv(
            fid_iid_file, sep="\t", index=False, header=False
        )

    figure_png = out_dir / str(config.figure_file)
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
            ax.set_title(_format_title(pc_num, var), pad=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for j in range(len(pc_cols) + 1, n_rows * n_cols + 1):
            axes[(j - 1) // n_cols][(j - 1) % n_cols].axis("off")

        fig.suptitle(
            f"All PC Distribution Analysis: {str(config.case_label)} vs {str(config.control_label)} - Mainland Only",
            fontweight="bold",
            y=0.995,
        )
        fig.legend(
            handles=[
                mpatches.Patch(color=str(config.bbj_color), alpha=float(config.alpha), label=str(config.control_label)),
                mpatches.Patch(color=str(config.case_color), alpha=float(config.alpha), label=str(config.case_label)),
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

    return MainlandAllPCsKDEOutput(
        output_dir=out_dir,
        log_file=log_path,
        fid_iid_file=fid_iid_file,
        figure_png=figure_png,
        df_mainland=df_mainland,
        tests_df=tests_df,
    )