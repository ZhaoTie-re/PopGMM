from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MainlandSubclusterConfidenceDistributionOutput(NamedTuple):
    """Output container for mainland subcluster confidence distribution analysis."""
    summary_stats: dict[str, Any]
    summary_by_group: pd.DataFrame
    figure_path: Path | None
    output_dir: Path


@dataclass(frozen=True)
class MainlandSubclusterConfidenceDistributionConfig:
    """Configuration for mainland subcluster confidence distribution visualization."""
    output_dir: str = "results/07_mainland_subcluster_confidence_distribution"
    figure_file: str = "mainland_subcluster_confidence_distribution.png"
    
    case_label: str = "Case"
    control_label: str = "Control"
    
    save_plot: bool = True
    show_plot: bool = False
    verbose: bool = True


def run_mainland_subcluster_confidence_distribution(
    *,
    df_mainland_subcluster: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    config: MainlandSubclusterConfidenceDistributionConfig | None = None,
) -> MainlandSubclusterConfidenceDistributionOutput:
    """Analyze confidence distribution for mainland subcluster samples, stratified by case/control.
    
    Generates KDE + histogram (left panel) and ECDF (right panel) with group stratification.
    """
    
    config = config or MainlandSubclusterConfidenceDistributionConfig()
    
    if "IID" not in df_mainland_subcluster.columns:
        raise KeyError("df_mainland_subcluster must contain IID column.")
    if "Assignment_Confidence" not in df_mainland_subcluster.columns:
        raise KeyError("df_mainland_subcluster must contain Assignment_Confidence column.")
    
    # Label samples by case/control
    case_set = set(str(x) for x in our_case_iids)
    ctrl_set = set(str(x) for x in our_ctrl_iids)
    
    iid_vals = df_mainland_subcluster["IID"].astype(str)
    group_labels = np.where(
        iid_vals.isin(case_set),
        str(config.case_label),
        np.where(iid_vals.isin(ctrl_set), str(config.control_label), "Other")
    )
    
    # Filter to case/control only (exclude "Other")
    mask_labeled = (group_labels != "Other")
    df_labeled = df_mainland_subcluster.loc[mask_labeled].copy()
    group_labeled = group_labels[mask_labeled]
    
    confidence_vals = pd.to_numeric(df_labeled["Assignment_Confidence"], errors="coerce")
    confidence_vals = confidence_vals.dropna().to_numpy(dtype=np.float64, copy=False)
    
    if len(confidence_vals) == 0:
        raise ValueError("No valid Assignment_Confidence values found in mainland subcluster.")
    
    # Compute summary statistics
    case_mask = group_labeled == str(config.case_label)
    ctrl_mask = group_labeled == str(config.control_label)
    
    case_conf = confidence_vals[case_mask]
    ctrl_conf = confidence_vals[ctrl_mask]
    
    summary_stats: dict[str, Any] = {
        "n_total": int(len(confidence_vals)),
        "n_case": int(np.sum(case_mask)),
        "n_control": int(np.sum(ctrl_mask)),
    }
    
    stats_case = {
        "mean": float(np.mean(case_conf)) if len(case_conf) > 0 else np.nan,
        "std": float(np.std(case_conf)) if len(case_conf) > 0 else np.nan,
        "min": float(np.min(case_conf)) if len(case_conf) > 0 else np.nan,
        "25%": float(np.percentile(case_conf, 25)) if len(case_conf) > 0 else np.nan,
        "50%": float(np.percentile(case_conf, 50)) if len(case_conf) > 0 else np.nan,
        "75%": float(np.percentile(case_conf, 75)) if len(case_conf) > 0 else np.nan,
        "max": float(np.max(case_conf)) if len(case_conf) > 0 else np.nan,
    }
    
    stats_ctrl = {
        "mean": float(np.mean(ctrl_conf)) if len(ctrl_conf) > 0 else np.nan,
        "std": float(np.std(ctrl_conf)) if len(ctrl_conf) > 0 else np.nan,
        "min": float(np.min(ctrl_conf)) if len(ctrl_conf) > 0 else np.nan,
        "25%": float(np.percentile(ctrl_conf, 25)) if len(ctrl_conf) > 0 else np.nan,
        "50%": float(np.percentile(ctrl_conf, 50)) if len(ctrl_conf) > 0 else np.nan,
        "75%": float(np.percentile(ctrl_conf, 75)) if len(ctrl_conf) > 0 else np.nan,
        "max": float(np.max(ctrl_conf)) if len(ctrl_conf) > 0 else np.nan,
    }
    
    summary_stats[str(config.case_label)] = stats_case
    summary_stats[str(config.control_label)] = stats_ctrl
    
    summary_by_group = pd.DataFrame({
        str(config.case_label): pd.Series(stats_case),
        str(config.control_label): pd.Series(stats_ctrl),
    })
    
    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    figure_path: Path | None = None
    
    if bool(config.save_plot) or bool(config.show_plot):
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=2.5)
        
        fig, axes = plt.subplots(1, 2, figsize=(28, 12))
        fig.subplots_adjust(wspace=0.25, left=0.08, right=0.95, top=0.84, bottom=0.15)
        
        # Color palette
        palette = {
            str(config.case_label): "#E31A1C",
            str(config.control_label): "#1F78B4",
        }
        
        # Panel A: KDE + histogram
        ax1 = axes[0]
        for group in [str(config.case_label), str(config.control_label)]:
            group_conf = confidence_vals[group_labeled == group]
            if len(group_conf) > 0:
                sns.kdeplot(
                    x=group_conf,
                    color=palette[group],
                    fill=True,
                    alpha=0.3,
                    linewidth=4,
                    label=f"{group} (n={len(group_conf)})",
                    ax=ax1,
                )
        
        ax1.set_title("Distribution of Confidence (Mainland Subcluster)", loc="left", pad=30, fontweight="bold")
        ax1.set_xlabel("Max Posterior Probability (Confidence)", labelpad=20)
        ax1.set_ylabel("Density", labelpad=20)
        ax1.grid(True, linestyle="--", alpha=0.4, color="#bbbbbb", linewidth=1.5)
        ax1.legend(loc="upper left", fontsize=22, frameon=True)
        
        # Panel B: ECDF
        ax2 = axes[1]
        for group in [str(config.case_label), str(config.control_label)]:
            group_conf = confidence_vals[group_labeled == group]
            if len(group_conf) > 0:
                sns.ecdfplot(
                    x=group_conf,
                    color=palette[group],
                    linewidth=6,
                    label=f"{group} (n={len(group_conf)})",
                    ax=ax2,
                )
        
        ax2.set_title("Cumulative Distribution (Mainland Subcluster)", loc="left", pad=30, fontweight="bold")
        ax2.set_xlabel("Max Posterior Probability (Confidence)", labelpad=20)
        ax2.set_ylabel("Cumulative Proportion", labelpad=20)
        ax2.grid(True, linestyle="--", alpha=0.4, color="#bbbbbb", linewidth=1.5)
        ax2.legend(loc="upper left", fontsize=22, frameon=True)
        
        fig.suptitle(
            "Mainland Subcluster Assignment Confidence Analysis",
            fontsize=34,
            fontweight="bold",
            y=0.96,
            ha="center",
        )
        
        if bool(config.save_plot):
            figure_path = out_dir / str(config.figure_file)
            fig.savefig(figure_path, bbox_inches="tight", dpi=400)
        
        if bool(config.show_plot):
            plt.show()
        else:
            plt.close(fig)
    
    if bool(config.verbose):
        print("\n" + "=" * 80)
        print("STEP7: MAINLAND SUBCLUSTER CONFIDENCE DISTRIBUTION".center(80))
        print("=" * 80)
        print(f"\n  Total samples (Case+Control): {summary_stats['n_total']}")
        print(f"  {config.case_label:12s}: {summary_stats['n_case']:5d}")
        print(f"  {config.control_label:12s}: {summary_stats['n_control']:5d}")
        print(f"\n  Output directory: {out_dir}")
        if figure_path is not None:
            print(f"  Figure saved   : {figure_path}")
        print("\n" + "-" * 80)
        print(f"\n{'Metric':<12} {config.case_label:>20} {config.control_label:>20}")
        print("-" * 80)
        for metric in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            case_val = summary_stats[str(config.case_label)].get(metric, np.nan)
            ctrl_val = summary_stats[str(config.control_label)].get(metric, np.nan)
            if pd.isna(case_val):
                print(f"{metric:<12} {'N/A':>20} {'N/A':>20}")
            else:
                print(f"{metric:<12} {float(case_val):>20.6f} {float(ctrl_val):>20.6f}")
        print("=" * 80 + "\n")
    
    return MainlandSubclusterConfidenceDistributionOutput(
        summary_stats=summary_stats,
        summary_by_group=summary_by_group,
        figure_path=figure_path,
        output_dir=out_dir,
    )
