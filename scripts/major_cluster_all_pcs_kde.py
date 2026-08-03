"""Compare case and control distributions across every PC within the major cluster.

If the selection has removed ancestry structure, case and control densities
should agree on all PCs, not only the two the mixture was fitted on. Each PC gets
a Welch t-test and a Mann-Whitney U test, with Benjamini-Hochberg correction
across PCs, plus a KDE panel to show what any significant difference looks like.

Inputs
------
Cohort assignment results, study samples, case/control lists, the major-cluster
component ids, eigenvalues for the variance annotations.

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
from scipy.stats import mannwhitneyu, ttest_ind

from scripts.common import (
    to_numeric_series,
    to_numeric_array,
    bh_fdr_adjust as _bh_fdr_adjust,
    pc_sort_key as _pc_sort_key,
    resolve_all_pc_columns as _resolve_all_pc_columns,
    setup_file_logger as _setup_file_logger,
)
from scripts.plotting.kde import plot_pc_kde_grid
from scripts.plotting.style import THEME_KDE, figure_context, save_figure


class MajorClusterAllPCsKDEOutput(NamedTuple):
    output_dir: Path
    log_file: Path
    figure_png: Path
    df_major_cluster: pd.DataFrame
    tests_df: pd.DataFrame


@dataclass(frozen=True)
class MajorClusterAllPCsKDEConfig:
    output_dir: str | Path = "results/02_cohort_assignment/pc_space_global"
    save_plot: bool = True
    show_plot: bool = False
    figure_file: str = "all_pcs_kde.png"
    log_file: str = "all_pcs_kde.log"
    tests_file: str = "all_pcs_kde_tests.tsv"

    #: Names the PCA the coordinates come from. Two runs in two bases produce
    #: otherwise identical-looking figures, so this goes in the title and the log.
    basis_label: str = "global PCA"

    #: What the major cluster is called on figures; mirrors
    #: params.MAJOR_CLUSTER_DISPLAY_NAME rather than being hardcoded.
    group_label: str = "Mainland"
    case_label: str = "Case"
    control_label: str = "Control"
    n_cols: int = 5
    reference_color: str = "#1F78B4"
    case_color: str = "#E31A1C"
    alpha: float = 0.65
    verbose: bool = True


def _format_title(pc_num: int, var: float) -> str:
    return f"PC{pc_num}{f' ({var:.1%})' if np.isfinite(var) else ''}"


def _finite(values: pd.Series) -> np.ndarray:
    """Numeric values with non-finite entries dropped, ready for a KDE."""
    arr = to_numeric_array(values)
    return arr[np.isfinite(arr)]


def run_major_cluster_all_pcs_kde(
    *,
    df_results: pd.DataFrame,
    study_samples: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    major_cluster_component_ids: list[int],
    eigenval: pd.DataFrame | None = None,
    config: MajorClusterAllPCsKDEConfig | None = None,
) -> MajorClusterAllPCsKDEOutput:
    config = config or MajorClusterAllPCsKDEConfig()

    df_results = pd.DataFrame(df_results).copy()
    study_df = pd.DataFrame(study_samples).copy()
    if "IID" not in df_results.columns or "IID" not in study_df.columns:
        raise KeyError("Both df_results and study_samples must contain 'IID'.")
    if "Assigned_Merged_Cluster" not in df_results.columns:
        raise KeyError("df_results must contain 'Assigned_Merged_Cluster'.")

    df_results["IID"] = df_results["IID"].astype(str)
    study_df["IID"] = study_df["IID"].astype(str)

    mainland_ids = sorted({int(x) for x in major_cluster_component_ids})
    if not mainland_ids:
        raise ValueError("major_cluster_component_ids is empty; pass the set derived by component merging.")

    df_join = study_df.merge(df_results, on="IID", how="inner", suffixes=("", "_result"))
    df_join["Assigned_Merged_Cluster"] = to_numeric_series(df_join["Assigned_Merged_Cluster"]).fillna(-1).astype(int)
    df_major_cluster = df_join.loc[df_join["Assigned_Merged_Cluster"].isin(mainland_ids)].copy()

    case_set = set(map(str, case_iids))
    ctrl_set = set(map(str, control_iids))
    is_case = df_major_cluster["IID"].astype(str).isin(case_set).to_numpy()
    is_ctrl = df_major_cluster["IID"].astype(str).isin(ctrl_set).to_numpy()
    df_case = df_major_cluster.loc[is_case].copy()
    df_ctrl = df_major_cluster.loc[is_ctrl].copy()

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
        x_ctrl = to_numeric_array(df_ctrl[col])
        x_case = to_numeric_array(df_case[col])
        x_ctrl = x_ctrl[np.isfinite(x_ctrl)]
        x_case = x_case[np.isfinite(x_case)]
        # scipy builds these result classes with _make_tuple_bunch, so its stubs
        # expose neither .statistic/.pvalue nor usable element types -- unpacking
        # only moves the complaint to float(). Annotating as Any is the honest
        # remedy for a third-party stub gap; the attributes exist at runtime.
        t_res: Any = ttest_ind(x_case, x_ctrl, equal_var=False, nan_policy="omit")
        u_res: Any = mannwhitneyu(x_case, x_ctrl, alternative="two-sided")
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
        logger.info(f">>> ALL-PC DISTRIBUTIONS FOR {str(config.group_label).upper()} SAMPLES "
                    f"({config.basis_label})...")
        logger.info(f"   -> output_dir = {out_dir.as_posix()}")
        logger.info(f"   -> basis = {config.basis_label}")
        logger.info(f"   -> mainland clusters = {mainland_ids}")
        logger.info(f"   -> Mainland samples: {len(df_major_cluster)} / {len(df_join)}")
        logger.info(f"   -> Case samples (Mainland): {len(df_case)}")
        logger.info(f"   -> Control samples (Mainland): {len(df_ctrl)}")
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
        titles = [
            _format_title(int(_pc_sort_key(col)), float(var_lookup.get(int(_pc_sort_key(col)), np.nan)))
            for col in pc_cols
        ]

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
                # `reference_color` names the CONTROL cohort in this module, not
                # the BBJ background it names elsewhere. Mapped explicitly here so
                # the shared plotter can use the honest parameter name.
                control_color=str(config.reference_color),
            )
            save_figure(fig, figure_png, dpi=300)
            if bool(config.show_plot):
                plt.show()
            else:
                plt.close(fig)

    return MajorClusterAllPCsKDEOutput(
        output_dir=out_dir,
        log_file=log_path,
        figure_png=figure_png,
        df_major_cluster=df_major_cluster,
        tests_df=tests_df,
    )