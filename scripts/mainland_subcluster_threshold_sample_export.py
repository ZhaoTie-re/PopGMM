from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd


class MainlandSubclusterThresholdSampleExportOutput(NamedTuple):
    """Output container for Step9 threshold-based sample export."""

    summary_table: pd.DataFrame
    thresholds: list[float]
    default_threshold: float
    output_dir: Path
    summary_table_path: Path


@dataclass(frozen=True)
class MainlandSubclusterThresholdSampleExportConfig:
    """Configuration for exporting retained/removed FID-IID files by threshold."""

    output_dir: str = "results/09_threshold_sample_exports"
    summary_file: str = "threshold_retained_removed_summary.tsv"

    fixed_thresholds: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 0.99)
    include_case_min_threshold: bool = True

    case_label: str = "Case"
    control_label: str = "Control"

    fid_col: str = "FID"
    iid_col: str = "IID"
    confidence_col: str = "Assignment_Confidence"

    retained_case_prefix: str = "retained_case"
    retained_ctrl_prefix: str = "retained_ctrl"
    removed_case_prefix: str = "removed_case"
    removed_ctrl_prefix: str = "removed_ctrl"
    retained_all_prefix: str = "retained_all"

    export_case_ctrl_files: bool = False
    clean_legacy_group_files: bool = True
    write_default_alias_file: bool = False

    default_retained_all_file: str = "retained_all_default_case_min_thr.fid_iid.txt"
    verbose: bool = True


def _to_fid_iid(df_in: pd.DataFrame, iid_col: str, fid_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df_in.index)
    out[fid_col] = df_in[iid_col].astype(str)
    out[iid_col] = df_in[iid_col].astype(str)
    return out


def run_mainland_subcluster_threshold_sample_export(
    *,
    df_mainland_subcluster: pd.DataFrame,
    our_case_iids: list[Any],
    our_ctrl_iids: list[Any],
    config: MainlandSubclusterThresholdSampleExportConfig | None = None,
) -> MainlandSubclusterThresholdSampleExportOutput:
    """Export retained/removed sample FID-IID files under multiple confidence thresholds."""

    config = config or MainlandSubclusterThresholdSampleExportConfig()

    if config.iid_col not in df_mainland_subcluster.columns:
        raise KeyError(f"Missing IID column: {config.iid_col}")
    if config.confidence_col not in df_mainland_subcluster.columns:
        raise KeyError(f"Missing confidence column: {config.confidence_col}")

    case_set = set(str(x) for x in our_case_iids)
    ctrl_set = set(str(x) for x in our_ctrl_iids)

    df = df_mainland_subcluster.copy()
    df[config.iid_col] = df[config.iid_col].astype(str)
    df[config.confidence_col] = pd.to_numeric(df[config.confidence_col], errors="coerce")
    df = df.dropna(subset=[config.confidence_col]).copy()

    df["Group"] = np.where(
        df[config.iid_col].isin(case_set),
        str(config.case_label),
        np.where(df[config.iid_col].isin(ctrl_set), str(config.control_label), "Other"),
    )
    df = df[df["Group"] != "Other"].copy()

    case_conf = df.loc[df["Group"] == str(config.case_label), config.confidence_col]
    if case_conf.empty:
        raise ValueError("No case samples found after filtering.")

    case_min_thr = float(case_conf.min())
    default_threshold = case_min_thr

    thresholds: list[float] = [float(x) for x in config.fixed_thresholds]
    if bool(config.include_case_min_threshold):
        thresholds.append(case_min_thr)
    thresholds = sorted({round(float(t), 12) for t in thresholds})

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_case_n = int((df["Group"] == str(config.case_label)).sum())
    base_ctrl_n = int((df["Group"] == str(config.control_label)).sum())
    base_total_n = int(len(df))

    if bool(config.clean_legacy_group_files):
        for pattern in (
            f"{config.retained_case_prefix}_thr_*.fid_iid.txt",
            f"{config.retained_ctrl_prefix}_thr_*.fid_iid.txt",
            f"{config.removed_case_prefix}_thr_*.fid_iid.txt",
            f"{config.removed_ctrl_prefix}_thr_*.fid_iid.txt",
            "removed_all_thr_*.fid_iid.txt",
        ):
            for fp in out_dir.glob(pattern):
                fp.unlink(missing_ok=True)

    records: list[dict[str, Any]] = []

    for thr in thresholds:
        keep_mask = df[config.confidence_col] >= float(thr)
        kept = df[keep_mask].copy()

        kept_case = kept[kept["Group"] == str(config.case_label)].copy()
        kept_ctrl = kept[kept["Group"] == str(config.control_label)].copy()

        thr_tag = f"{thr:.4f}"

        if bool(config.export_case_ctrl_files):
            _to_fid_iid(kept_case, config.iid_col, config.fid_col).to_csv(
                out_dir / f"{config.retained_case_prefix}_thr_{thr_tag}.fid_iid.txt",
                sep="\t",
                index=False,
                header=False,
            )
            _to_fid_iid(kept_ctrl, config.iid_col, config.fid_col).to_csv(
                out_dir / f"{config.retained_ctrl_prefix}_thr_{thr_tag}.fid_iid.txt",
                sep="\t",
                index=False,
                header=False,
            )

        _to_fid_iid(kept, config.iid_col, config.fid_col).to_csv(
            out_dir / f"{config.retained_all_prefix}_thr_{thr_tag}.fid_iid.txt",
            sep="\t",
            index=False,
            header=False,
        )

        case_kept_n = int(len(kept_case))
        ctrl_kept_n = int(len(kept_ctrl))
        removed_n = int(base_total_n - len(kept))

        records.append(
            {
                "threshold": float(thr),
                "case_total": base_case_n,
                "ctrl_total": base_ctrl_n,
                "case_retained": case_kept_n,
                "ctrl_retained": ctrl_kept_n,
            }
        )

    if bool(config.write_default_alias_file):
        default_kept = df[df[config.confidence_col] >= default_threshold].copy()
        _to_fid_iid(default_kept, config.iid_col, config.fid_col).to_csv(
            out_dir / str(config.default_retained_all_file),
            sep="\t",
            index=False,
            header=False,
        )
    else:
        default_alias_path = out_dir / str(config.default_retained_all_file)
        default_alias_path.unlink(missing_ok=True)

    summary_table = pd.DataFrame.from_records(records)
    summary_path = out_dir / str(config.summary_file)
    summary_table.to_csv(summary_path, sep="\t", index=False)

    if bool(config.verbose):
        print("\n" + "=" * 92)
        print("STEP9: THRESHOLD-BASED FID/IID EXPORT".center(92))
        print("=" * 92)
        print(f"Default threshold (case min): {default_threshold:.6f}")
        print("Thresholds: " + ", ".join(f"{t:.4f}" for t in thresholds))
        print(f"Output directory: {out_dir}")
        print(f"Summary table   : {summary_path}")
        print("-" * 60)
        print(summary_table.to_string(index=False))
        print("=" * 60)

    return MainlandSubclusterThresholdSampleExportOutput(
        summary_table=summary_table,
        thresholds=thresholds,
        default_threshold=default_threshold,
        output_dir=out_dir,
        summary_table_path=summary_path,
    )
