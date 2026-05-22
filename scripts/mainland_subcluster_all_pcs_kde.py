from __future__ import annotations

"""STEP6 mainland_subcluster all-PC KDE visualization.

This module mirrors scripts/cluster_all_pcs_kde.py, but the selected subset is
the mainland_subcluster rows produced by STEP5/STEP6.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind

from scripts.cluster_all_pcs_kde import (
    _PLOT_STYLE_RC,
    _bh_fdr_adjust,
    _format_template,
    _resolve_all_pc_columns,
    _safe_stats,
    _setup_file_logger,
)


class MainlandSubclusterAllPCsKDEOutput(NamedTuple):
    output_dir: Path
    log_file: Path
    fid_iid_file: Path
    figure_png: Path
    df_subcluster: pd.DataFrame
    tests_df: pd.DataFrame


@dataclass(frozen=True)
class MainlandSubclusterAllPCsKDEConfig:
    output_dir: str = "results/06_mainland_subcluster_only"
    save_tables: bool = True
    save_plot: bool = True
    show_plot: bool = False
    output_file: str = "mainland_subcluster_samples.fid_iid.txt"
    figure_file: str = "mainland_subcluster_all_pcs_kde.png"
    log_file: str = "mainland_subcluster_all_pcs_kde.log"
    mainland_group_label: str = "Mainland Subcluster"
    assigned_group_col: str = "Assigned_Mainland_Subcluster_Group"
    confidence_col: str = "Assignment_Confidence"
    case_label: str = "Case"
    control_label: str = "Control"
    n_cols: int = 5
    bbj_color: str = "#1F78B4"
    case_color: str = "#E31A1C"
    alpha: float = 0.65
    verbose: bool = True


def run_mainland_subcluster_all_pcs_kde(
    *,
    df_results_step5: pd.DataFrame,
    our_samples: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    config: MainlandSubclusterAllPCsKDEConfig | None = None,
) -> MainlandSubclusterAllPCsKDEOutput:
    config = config or MainlandSubclusterAllPCsKDEConfig()

    df = pd.DataFrame(df_results_step5).copy()
    our_df = pd.DataFrame(our_samples).copy()
    if "IID" not in df.columns or "IID" not in our_df.columns:
        raise KeyError("Both df_results_step5 and our_samples must contain 'IID'.")
    if str(config.assigned_group_col) not in df.columns:
        raise KeyError(f"df_results_step5 must contain '{config.assigned_group_col}'.")
    if str(config.confidence_col) not in df.columns:
        raise KeyError(f"df_results_step5 must contain '{config.confidence_col}'.")

    df["IID"] = df["IID"].astype(str)
    our_df["IID"] = our_df["IID"].astype(str)
    mainland_label = str(config.mainland_group_label)
    df_join = our_df.merge(df, on="IID", how="inner", suffixes=("", "_result"))
    mask = df_join[str(config.assigned_group_col)].astype(str) == mainland_label
    df_subcluster = df_join.loc[mask].copy()

    if df_subcluster.empty:
        raise ValueError("No mainland_subcluster rows found in df_results_step5.")

    pc_cols = _resolve_all_pc_columns(df_join)

    case_set = set(map(str, our_case_iids))
    ctrl_set = set(map(str, our_ctrl_iids))
    iid_vals = df_subcluster["IID"].astype(str)
    is_case = iid_vals.isin(case_set).to_numpy()
    is_ctrl = iid_vals.isin(ctrl_set).to_numpy()
    df_case = df_subcluster.loc[is_case].copy()
    df_ctrl = df_subcluster.loc[is_ctrl].copy()

    test_rows: list[dict[str, Any]] = []
    for col in pc_cols:
        pc_num = int(str(col).replace("PC", "").replace("_AVG", ""))
        x_ctrl = pd.to_numeric(df_ctrl[col], errors="coerce").to_numpy(dtype=float, copy=False)
        x_case = pd.to_numeric(df_case[col], errors="coerce").to_numpy(dtype=float, copy=False)
        x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
        x_case = x_case[np.isfinite(x_case)]
        s_ctrl = _safe_stats(x_ctrl)
        s_case = _safe_stats(x_case)
        t_res = cast(Any, ttest_ind(x_case, x_ctrl, equal_var=False, nan_policy="omit"))
        u_res = cast(Any, mannwhitneyu(x_case, x_ctrl, alternative="two-sided"))
        test_rows.append(
            {
                "pc": f"PC{pc_num}",
                "n_case": int(x_case.size),
                "n_control": int(x_ctrl.size),
                "case_mean": float(s_case["mean"]),
                "ctrl_mean": float(s_ctrl["mean"]),
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
    logger = _setup_file_logger("mainland_subcluster_all_pcs_kde", log_path)

    if bool(config.verbose):
        logger.info(">>> ANALYZING ALL PC DISTRIBUTIONS FOR MAINLAND_SUBCLUSTER SAMPLES...")
        logger.info(f"   -> output_dir = {out_dir.as_posix()}")
        logger.info(f"   -> mainland label = {mainland_label}")
        logger.info(f"   -> mainland_subcluster samples = {len(df_subcluster)} / {len(df)}")
        logger.info(f"   -> Case samples: {len(df_case)}")
        logger.info(f"   -> Control samples: {len(df_ctrl)}")
        logger.info(f"   -> Total PCs to analyze: {len(pc_cols)}")
        logger.info("   -> Running statistical tests...")
        logger.info("   -> Applying FDR correction (Benjamini-Hochberg method) to all tests...")
        logger.info("   -> FDR correction complete.")
        logger.info(f"      • Significant by t-test: {int(tests_df['reject_t'].sum())} PC(s)")
        logger.info(f"      • Significant by Mann-Whitney U: {int(tests_df['reject_u'].sum())} PC(s)")
        logger.info("=" * 80)

    fid_iid_file = out_dir / str(config.output_file)
    if bool(config.save_tables):
        df_subcluster.loc[:, [c for c in ["FID", "IID"] if c in df_subcluster.columns]].drop_duplicates().to_csv(
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
            pc_num = int(str(col).replace("PC", "").replace("_AVG", ""))
            ax.set_title(f"PC{pc_num}", pad=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.35, color="#C3C3C3")
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for j in range(len(pc_cols) + 1, n_rows * n_cols + 1):
            axes[(j - 1) // n_cols][(j - 1) % n_cols].axis("off")

        fig.suptitle(
            f"All PC Distribution Analysis: {str(config.case_label)} vs {str(config.control_label)} - Mainland Subcluster Only",
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

    return MainlandSubclusterAllPCsKDEOutput(
        output_dir=out_dir,
        log_file=log_path,
        fid_iid_file=fid_iid_file,
        figure_png=figure_png,
        df_subcluster=df_subcluster,
        tests_df=tests_df,
    )