"""Compare case and control distributions across every PC within a subcluster.

The same all-PC check as for the major cluster, applied to a subcluster variant:
Welch t-test and Mann-Whitney U per PC with Benjamini-Hochberg correction, plus
KDE panels. Samples are selected by the assigned-group column rather than by
component id, since the subcluster is a composite group.

Inputs
------
A subcluster assignment frame, study samples, case/control lists.

Outputs
-------
A KDE panel figure and a log of the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.common import (
    to_numeric_array,
    PLOT_STYLE_RC as _PLOT_STYLE_RC,
    bh_fdr_adjust as _bh_fdr_adjust,
    resolve_all_pc_columns as _resolve_all_pc_columns,
    safe_stats as _safe_stats,
    setup_file_logger as _setup_file_logger,
)
from scipy.stats import mannwhitneyu, ttest_ind

class SubclusterAllPCsKDEOutput(NamedTuple):
    output_dir: Path
    log_file: Path
    figure_png: Path
    df_subcluster: pd.DataFrame
    tests_df: pd.DataFrame


@dataclass(frozen=True)
class SubclusterAllPCsKDEConfig:
    output_dir: str | Path = "results/04_subcluster_variants/refined"
    save_plot: bool = True
    show_plot: bool = False
    figure_file: str = "subcluster_all_pcs_kde.png"
    log_file: str = "subcluster_all_pcs_kde.log"
    group_label: str = "Mainland Subcluster"
    assigned_group_col: str = "Assigned_Mainland_Subcluster_Group"
    confidence_col: str = "Assignment_Confidence"
    case_label: str = "Case"
    control_label: str = "Control"
    n_cols: int = 5
    reference_color: str = "#1F78B4"
    case_color: str = "#E31A1C"
    alpha: float = 0.65
    verbose: bool = True


def run_subcluster_all_pcs_kde(
    *,
    df_assigned: pd.DataFrame,
    study_samples: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    config: SubclusterAllPCsKDEConfig | None = None,
) -> SubclusterAllPCsKDEOutput:
    config = config or SubclusterAllPCsKDEConfig()

    df = pd.DataFrame(df_assigned).copy()
    study_df = pd.DataFrame(study_samples).copy()
    if "IID" not in df.columns or "IID" not in study_df.columns:
        raise KeyError("Both df_assigned and study_samples must contain 'IID'.")
    if str(config.assigned_group_col) not in df.columns:
        raise KeyError(f"df_assigned must contain '{config.assigned_group_col}'.")
    if str(config.confidence_col) not in df.columns:
        raise KeyError(f"df_assigned must contain '{config.confidence_col}'.")

    df["IID"] = df["IID"].astype(str)
    study_df["IID"] = study_df["IID"].astype(str)
    mainland_label = str(config.group_label)
    df_join = study_df.merge(df, on="IID", how="inner", suffixes=("", "_result"))
    mask = df_join[str(config.assigned_group_col)].astype(str) == mainland_label
    df_subcluster = df_join.loc[mask].copy()

    if df_subcluster.empty:
        raise ValueError("No subcluster rows found in df_assigned.")

    pc_cols = _resolve_all_pc_columns(df_join)

    case_set = set(map(str, case_iids))
    ctrl_set = set(map(str, control_iids))
    iid_vals = df_subcluster["IID"].astype(str)
    is_case = iid_vals.isin(case_set).to_numpy()
    is_ctrl = iid_vals.isin(ctrl_set).to_numpy()
    df_case = df_subcluster.loc[is_case].copy()
    df_ctrl = df_subcluster.loc[is_ctrl].copy()

    test_rows: list[dict[str, Any]] = []
    for col in pc_cols:
        pc_num = int(str(col).replace("PC", "").replace("_AVG", ""))
        x_ctrl = to_numeric_array(df_ctrl[col])
        x_case = to_numeric_array(df_case[col])
        x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
        x_case = x_case[np.isfinite(x_case)]
        s_ctrl = _safe_stats(x_ctrl)
        s_case = _safe_stats(x_case)
        # scipy builds these result classes with _make_tuple_bunch, so its stubs
        # expose neither .statistic/.pvalue nor usable element types -- unpacking
        # only moves the complaint to float(). Annotating as Any is the honest
        # remedy for a third-party stub gap; the attributes exist at runtime.
        t_res: Any = ttest_ind(x_case, x_ctrl, equal_var=False, nan_policy="omit")
        u_res: Any = mannwhitneyu(x_case, x_ctrl, alternative="two-sided")
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
            x_ctrl = to_numeric_array(df_ctrl[col])
            x_case = to_numeric_array(df_case[col])
            x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
            x_case = x_case[np.isfinite(x_case)]
            if x_ctrl.size > 1:
                sns.kdeplot(x=x_ctrl, color=str(config.reference_color), fill=True, alpha=float(config.alpha), linewidth=2.5, ax=ax)
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
                mpatches.Patch(color=str(config.reference_color), alpha=float(config.alpha), label=str(config.control_label)),
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

    return SubclusterAllPCsKDEOutput(
        output_dir=out_dir,
        log_file=log_path,
        figure_png=figure_png,
        df_subcluster=df_subcluster,
        tests_df=tests_df,
    )