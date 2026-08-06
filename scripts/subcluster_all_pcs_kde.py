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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import (
    to_numeric_array,
    bh_fdr_adjust as _bh_fdr_adjust,
    resolve_all_pc_columns as _resolve_all_pc_columns,
    safe_stats as _safe_stats,
    setup_file_logger as _setup_file_logger,
)
from scipy.stats import mannwhitneyu, ttest_ind
from scripts.plotting.kde import plot_pc_kde_grid
from scripts.plotting.style import THEME_KDE, figure_context, save_figure


class SubclusterAllPCsKDEOutput(NamedTuple):
    output_dir: Path
    log_file: Path
    figure_png: Path
    df_subcluster: pd.DataFrame
    tests_df: pd.DataFrame


@dataclass(frozen=True)
class SubclusterAllPCsKDEConfig:
    output_dir: str | Path = "results/04_subcluster_variants/narrow/pc_space_global"
    save_plot: bool = True
    show_plot: bool = False
    figure_file: str = "all_pcs_kde.png"
    log_file: str = "all_pcs_kde.log"
    tests_file: str = "all_pcs_kde_tests.tsv"

    #: Names the PCA the coordinates come from. Two runs in two bases produce
    #: otherwise identical-looking figures, so this goes in the title and the log.
    basis_label: str = "global PCA"
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


def _finite(values: pd.Series) -> np.ndarray:
    """Numeric values with non-finite entries dropped, ready for a KDE."""
    arr = to_numeric_array(values)
    return arr[np.isfinite(arr)]


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
        logger.info(f">>> ALL-PC DISTRIBUTIONS FOR {str(config.group_label).upper()} SAMPLES "
                    f"({config.basis_label})...")
        logger.info(f"   -> output_dir = {out_dir.as_posix()}")
        logger.info(f"   -> basis = {config.basis_label}")
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

    tests_path = out_dir / str(config.tests_file)
    tests_df.to_csv(tests_path, sep="\t", index=False)

    figure_png = out_dir / str(config.figure_file)
    if bool(config.save_plot) or bool(config.show_plot):
        control_values = [_finite(df_ctrl[col]) for col in pc_cols]
        case_values = [_finite(df_case[col]) for col in pc_cols]
        # Bare "PC7" here, against "PC7 (2.1%)" in the major-cluster twin: this
        # module is not given an eigenval, so it has no variance to report.
        titles = [f"PC{int(str(col).replace('PC', '').replace('_AVG', ''))}" for col in pc_cols]

        with figure_context(THEME_KDE):
            fig = plot_pc_kde_grid(
                case_values=case_values,
                control_values=control_values,
                titles=titles,
                suptitle=(
                    f"All PC Distribution Analysis: {str(config.case_label)} vs {str(config.control_label)}"
                    f" - {str(config.group_label)} Only  [{str(config.basis_label)}]"
                ),
                case_label=str(config.case_label),
                control_label=str(config.control_label),
                n_cols=int(config.n_cols),
                alpha=float(config.alpha),
                case_color=str(config.case_color),
                # `reference_color` is the CONTROL cohort in this module -- see
                # the note in major_cluster_all_pcs_kde.
                control_color=str(config.reference_color),
            )
            save_figure(fig, figure_png, dpi=300)
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