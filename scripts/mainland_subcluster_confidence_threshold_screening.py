from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MainlandSubclusterConfidenceThresholdScreeningOutput(NamedTuple):
    """Output container for confidence threshold screening on mainland subcluster."""

    summary_table: pd.DataFrame
    thresholds: list[float]
    figure_path: Path | None
    table_path: Path
    output_dir: Path


@dataclass(frozen=True)
class MainlandSubclusterConfidenceThresholdScreeningConfig:
    """Configuration for Step8 confidence-threshold sensitivity screening."""

    output_dir: str = "results/08_mainland_subcluster_confidence_screening"
    figure_file: str = "mainland_subcluster_confidence_threshold_screening.png"
    table_file: str = "mainland_subcluster_confidence_threshold_screening.tsv"

    fixed_thresholds: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 0.99)
    include_case_min_threshold: bool = True

    case_label: str = "Case"
    control_label: str = "Control"

    save_plot: bool = True
    show_plot: bool = False
    verbose: bool = True


def run_mainland_subcluster_confidence_threshold_screening(
    *,
    df_mainland_subcluster: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    config: MainlandSubclusterConfidenceThresholdScreeningConfig | None = None,
) -> MainlandSubclusterConfidenceThresholdScreeningOutput:
    """Evaluate retain/remove impact under multiple confidence thresholds."""

    config = config or MainlandSubclusterConfidenceThresholdScreeningConfig()

    if "IID" not in df_mainland_subcluster.columns:
        raise KeyError("df_mainland_subcluster must contain IID column.")
    if "Assignment_Confidence" not in df_mainland_subcluster.columns:
        raise KeyError("df_mainland_subcluster must contain Assignment_Confidence column.")

    case_set = set(str(x) for x in our_case_iids)
    ctrl_set = set(str(x) for x in our_ctrl_iids)

    iid_vals = df_mainland_subcluster["IID"].astype(str)
    group_labels = np.where(
        iid_vals.isin(case_set),
        str(config.case_label),
        np.where(iid_vals.isin(ctrl_set), str(config.control_label), "Other"),
    )

    df_labeled = df_mainland_subcluster.copy()
    df_labeled["Group"] = group_labels
    df_labeled = df_labeled[df_labeled["Group"] != "Other"].copy()
    df_labeled["Assignment_Confidence"] = pd.to_numeric(
        df_labeled["Assignment_Confidence"], errors="coerce"
    )
    df_labeled = df_labeled.dropna(subset=["Assignment_Confidence"])

    if df_labeled.empty:
        raise ValueError("No valid labeled samples found for Step8 threshold screening.")

    case_conf = df_labeled.loc[
        df_labeled["Group"] == str(config.case_label), "Assignment_Confidence"
    ].to_numpy(dtype=np.float64, copy=False)
    if case_conf.size == 0:
        raise ValueError("No case samples found in mainland subcluster.")

    thresholds: list[float] = [float(x) for x in config.fixed_thresholds]
    if bool(config.include_case_min_threshold):
        thresholds.append(float(np.min(case_conf)))
    thresholds = sorted({round(float(t), 12) for t in thresholds})

    base_case_n = int((df_labeled["Group"] == str(config.case_label)).sum())
    base_ctrl_n = int((df_labeled["Group"] == str(config.control_label)).sum())

    records: list[dict[str, Any]] = []
    for thr in thresholds:
        mask_keep = df_labeled["Assignment_Confidence"] >= float(thr)

        case_keep = int(((df_labeled["Group"] == str(config.case_label)) & mask_keep).sum())
        ctrl_keep = int(((df_labeled["Group"] == str(config.control_label)) & mask_keep).sum())

        case_drop = base_case_n - case_keep
        ctrl_drop = base_ctrl_n - ctrl_keep

        case_keep_rate = case_keep / base_case_n if base_case_n > 0 else np.nan
        ctrl_keep_rate = ctrl_keep / base_ctrl_n if base_ctrl_n > 0 else np.nan
        case_drop_rate = case_drop / base_case_n if base_case_n > 0 else np.nan
        ctrl_drop_rate = ctrl_drop / base_ctrl_n if base_ctrl_n > 0 else np.nan

        records.append(
            {
                "threshold": float(thr),
                "case_total": base_case_n,
                "control_total": base_ctrl_n,
                "case_kept": case_keep,
                "control_kept": ctrl_keep,
                "case_removed": case_drop,
                "control_removed": ctrl_drop,
                "case_keep_rate": float(case_keep_rate),
                "control_keep_rate": float(ctrl_keep_rate),
                "case_removed_rate": float(case_drop_rate),
                "control_removed_rate": float(ctrl_drop_rate),
            }
        )

    summary_table = pd.DataFrame.from_records(records)

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / str(config.table_file)
    summary_table.to_csv(table_path, sep="\t", index=False)

    figure_path: Path | None = None
    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=1.9)

        x_labels = [f"{t:.4f}" if t < 0.8 else f"{t:.2f}" for t in summary_table["threshold"]]
        x = np.arange(len(summary_table))

        case_color = "#C1121F"
        ctrl_color = "#1D4E89"

        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        fig.subplots_adjust(wspace=0.22, hspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.09)

        # Panel A: kept counts
        ax_a = axes[0, 0]
        ax_a.bar(x, summary_table["case_kept"], color=case_color, alpha=0.90, label=str(config.case_label))
        ax_a.bar(
            x,
            summary_table["control_kept"],
            bottom=summary_table["case_kept"],
            color=ctrl_color,
            alpha=0.88,
            label=str(config.control_label),
        )
        ax_a.set_title("Retained Samples by Threshold", loc="left", fontweight="bold", pad=14)
        ax_a.set_xlabel("Confidence Threshold")
        ax_a.set_ylabel("Retained Sample Count")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(x_labels)
        ax_a.legend(loc="upper right", frameon=True)

        # Panel B: removed counts
        ax_b = axes[0, 1]
        ax_b.bar(x, summary_table["case_removed"], color=case_color, alpha=0.90, label=str(config.case_label))
        ax_b.bar(
            x,
            summary_table["control_removed"],
            bottom=summary_table["case_removed"],
            color=ctrl_color,
            alpha=0.88,
            label=str(config.control_label),
        )
        ax_b.set_title("Removed Samples by Threshold", loc="left", fontweight="bold", pad=14)
        ax_b.set_xlabel("Confidence Threshold")
        ax_b.set_ylabel("Removed Sample Count")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(x_labels)
        ax_b.legend(loc="upper left", frameon=True)

        # Panel C: retain rate trends
        ax_c = axes[1, 0]
        ax_c.plot(
            x,
            summary_table["case_keep_rate"] * 100.0,
            color=case_color,
            marker="o",
            markersize=8,
            linewidth=3,
            label=f"{config.case_label} Retained",
        )
        ax_c.plot(
            x,
            summary_table["control_keep_rate"] * 100.0,
            color=ctrl_color,
            marker="o",
            markersize=8,
            linewidth=3,
            label=f"{config.control_label} Retained",
        )
        ax_c.set_title("Retention Rate by Threshold", loc="left", fontweight="bold", pad=14)
        ax_c.set_xlabel("Confidence Threshold")
        ax_c.set_ylabel("Retention Rate (%)")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(x_labels)
        ax_c.set_ylim(0, 103)
        ax_c.legend(loc="lower left", frameon=True)

        # Panel D: heatmap (removed rate)
        ax_d = axes[1, 1]
        heat_df = pd.DataFrame(
            {
                str(config.case_label): summary_table["case_removed_rate"].to_numpy(dtype=float) * 100.0,
                str(config.control_label): summary_table["control_removed_rate"].to_numpy(dtype=float) * 100.0,
            },
            index=x_labels,
        ).T
        sns.heatmap(
            heat_df,
            cmap="YlOrRd",
            annot=True,
            fmt=".1f",
            linewidths=1.2,
            linecolor="white",
            cbar_kws={"label": "Removed Rate (%)"},
            ax=ax_d,
        )
        ax_d.set_title("Removal Pressure Heatmap", loc="left", fontweight="bold", pad=14)
        ax_d.set_xlabel("Confidence Threshold")
        ax_d.set_ylabel("Group")

        fig.suptitle(
            "Step8: Mainland Subcluster Confidence Threshold Sensitivity",
            fontsize=30,
            fontweight="bold",
            y=0.97,
        )

        if bool(config.save_plot):
            figure_path = out_dir / str(config.figure_file)
            fig.savefig(figure_path, dpi=400, bbox_inches="tight")

        if bool(config.show_plot):
            plt.show()
        else:
            plt.close(fig)

    if bool(config.verbose):
        print("\n" + "=" * 88)
        print("STEP8: CONFIDENCE THRESHOLD SCREENING".center(88))
        print("=" * 88)
        print(f"\n  Base sample sizes: {config.case_label}={base_case_n}, {config.control_label}={base_ctrl_n}")
        print("  Thresholds used  : " + ", ".join(f"{t:.6f}" for t in thresholds))
        print(f"  Table saved      : {table_path}")
        if figure_path is not None:
            print(f"  Figure saved     : {figure_path}")
        print("\n" + "-" * 88)
        print(summary_table.to_string(index=False))
        print("=" * 88 + "\n")

    return MainlandSubclusterConfidenceThresholdScreeningOutput(
        summary_table=summary_table,
        thresholds=thresholds,
        figure_path=figure_path,
        table_path=table_path,
        output_dir=out_dir,
    )
