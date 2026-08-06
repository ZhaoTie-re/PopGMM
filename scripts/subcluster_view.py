"""Plot the selected subcluster on PC1-PC2 against the reference panel.

A focused two-panel view: where the retained samples sit in the panel's PC space,
and how many of them there are per group. Deliberately narrow -- the all-PC
comparison lives in its own module.

Inputs
------
A subcluster assignment frame, case/control lists, the clustered reference panel.

Outputs
-------
A two-panel figure and the per-group counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import resolve_pc_columns as _shared_resolve_pc_columns
from scripts.common import to_numeric_array
from scripts.plotting.assignment import PCWindow, ReferenceCloud, plot_subcluster_view
from scripts.plotting.style import THEME_SUBCLUSTER, figure_context, save_figure


def _resolve_pc_columns(df: pd.DataFrame) -> list[str]:
    """First two PC columns only; this module plots a 2-D scatter."""
    pc_cols = _shared_resolve_pc_columns(df)
    if len(pc_cols) < 2:
        raise ValueError("Need at least 2 PC columns in df_assigned (e.g., PC1_AVG, PC2_AVG).")
    return pc_cols[:2]


class SubclusterViewOutput(NamedTuple):
    df_subcluster: pd.DataFrame
    figure_path: Path | None
    counts_by_group: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class SubclusterViewConfig:
    output_dir: str | Path = "results/04_subcluster_variants/narrow/pc_space_global"
    figure_file: str = "subcluster_view.png"
    counts_file: str = "subcluster_view_counts.tsv"

    #: Names the PCA the coordinates come from. Two runs in two bases produce
    #: otherwise identical-looking figures, so this goes in the title.
    basis_label: str = "global PCA"

    group_label: str = "Mainland Subcluster"
    assigned_group_col: str = "Assigned_Mainland_Subcluster_Group"
    confidence_col: str = "Assignment_Confidence"

    case_label: str = "Case"
    control_label: str = "Control"

    reference_color: str = "#B0B0B0"
    reference_alpha: float = 0.20
    study_point_size: float = 60.0

    save_plot: bool = True
    show_plot: bool = False
    verbose: bool = True


def _coordinate_lookup(
    pc_coordinates: pd.DataFrame,
    pc_cols: list[str],
) -> dict[str, tuple[float, float]]:
    """IID -> (PC1, PC2) in one basis.

    Both projections name their columns identically, so a frame cannot be asked
    which basis it is in. Routing every frame's coordinates through one lookup is
    what makes it impossible to draw the reference cloud in one basis and the
    study points in another -- a mix that renders without error and is nonsense.
    """
    if "IID" not in pc_coordinates.columns:
        raise KeyError("pc_coordinates must contain an IID column.")
    missing = [c for c in pc_cols if c not in pc_coordinates.columns]
    if missing:
        raise KeyError(f"pc_coordinates must contain {missing}.")
    iids = pc_coordinates["IID"].astype(str).to_numpy()
    if len(set(iids)) != len(iids):
        raise ValueError("pc_coordinates has duplicated IIDs.")
    xy = pc_coordinates[pc_cols].to_numpy(dtype=np.float64, copy=False)
    return dict(zip(iids, map(tuple, xy)))


def _resolve_assigned_group_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for candidate in ("Assigned_Mainland_Subcluster_Group", "Assigned_Customize_Group", "Assigned_Composite_Group"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No assigned group column found in df_assigned.")


def run_subcluster_view(
    *,
    df_assigned: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    config: SubclusterViewConfig | None = None,
    reference_samples_gmm: pd.DataFrame | None = None,
    reference_samples_background: pd.DataFrame | None = None,
    pc_coordinates: pd.DataFrame | None = None,
    eigenval: pd.DataFrame | None = None,
) -> SubclusterViewOutput:
    config = config or SubclusterViewConfig()

    if "IID" not in df_assigned.columns:
        raise KeyError("df_assigned must contain IID column.")

    assigned_group_col = _resolve_assigned_group_col(df_assigned, config.assigned_group_col)
    pc_cols = _resolve_pc_columns(df_assigned)
    confidence_col = str(config.confidence_col)
    if confidence_col not in df_assigned.columns:
        raise KeyError(f"df_assigned must contain {confidence_col}.")

    mainland_label = str(config.group_label)
    mask = df_assigned[assigned_group_col].astype(str) == mainland_label
    df_subcluster = df_assigned.loc[mask].copy()

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)


    case_set = set(str(x) for x in case_iids)
    ctrl_set = set(str(x) for x in control_iids)

    iid_vals = df_subcluster["IID"].astype(str)
    group = np.where(iid_vals.isin(case_set), str(config.case_label), np.where(iid_vals.isin(ctrl_set), str(config.control_label), ""))
    df_subcluster["Group"] = group

    counts = (
        df_subcluster["Group"]
        .astype(str)
        .value_counts()
        .reindex([str(config.case_label), str(config.control_label)], fill_value=0)
        .rename_axis("Group")
        .reset_index(name="Count")
    )
    n_unlabeled = int((df_subcluster["Group"].astype(str) == "").sum())

    confidence = to_numeric_array(df_subcluster[confidence_col], np.float32)
    if not np.isfinite(confidence).all():
        raise ValueError("Selected mainland_subcluster subset contains invalid assignment confidence values.")

    coord_lookup: dict[str, tuple[float, float]] | None = None
    if pc_coordinates is not None:
        coord_lookup = _coordinate_lookup(pc_coordinates, pc_cols)
        study_iids = df_subcluster["IID"].astype(str).to_numpy()
        absent = [i for i in study_iids if i not in coord_lookup]
        if absent:
            raise ValueError(
                f"{len(absent)} of {len(study_iids)} retained samples are absent from "
                f"pc_coordinates; the projection must cover every plotted sample."
            )
        study_xy = np.array([coord_lookup[i] for i in study_iids], dtype=np.float64)
        study_pc1, study_pc2 = study_xy[:, 0], study_xy[:, 1]
    else:
        study_pc1 = to_numeric_array(df_subcluster[pc_cols[0]])
        study_pc2 = to_numeric_array(df_subcluster[pc_cols[1]])
    if not np.isfinite(study_pc1).all() or not np.isfinite(study_pc2).all():
        raise ValueError("Selected mainland_subcluster subset contains invalid PC coordinates.")

    try:
        var1 = float(eigenval.loc[eigenval["PC"] == 1, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
        var2 = float(eigenval.loc[eigenval["PC"] == 2, "variance_explained"].iloc[0]) if eigenval is not None else 0.0
    except Exception:
        var1, var2 = 0.0, 0.0

    def _pc_arrays(frame: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
        empty = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        if frame is None:
            return empty
        if coord_lookup is not None:
            # Reference samples outside the projection are dropped rather than
            # read from their own columns: the alternative is silently mixing
            # two bases on one pair of axes.
            if "IID" not in frame.columns:
                raise KeyError("reference frames must contain IID when pc_coordinates is given.")
            hits = [coord_lookup[i] for i in frame["IID"].astype(str) if i in coord_lookup]
            if not hits:
                return empty
            arr = np.asarray(hits, dtype=np.float64)
            return arr[:, 0], arr[:, 1]
        if pc_cols[0] not in frame.columns or pc_cols[1] not in frame.columns:
            return empty
        return (frame[pc_cols[0]].to_numpy(dtype=np.float64, copy=False),
                frame[pc_cols[1]].to_numpy(dtype=np.float64, copy=False))

    # Resolved unconditionally, not inside the plotting branch: this is the
    # single-basis guarantee, not a drawing detail. Turning plotting off must not
    # turn off the check that every frame is in the same PC basis.
    #
    # reference_pc* frames the axes; background_pc* is only drawn. Passing the
    # pre-denoising panel as the background shows every reference sample while
    # leaving the view framed on the set the model was fitted to.
    reference_pc1, reference_pc2 = _pc_arrays(reference_samples_gmm)
    background_pc1, background_pc2 = _pc_arrays(
        reference_samples_gmm if reference_samples_background is None else reference_samples_background
    )

    all_x = np.concatenate([reference_pc1, study_pc1]) if reference_pc1.size > 0 else study_pc1
    all_y = np.concatenate([reference_pc2, study_pc2]) if reference_pc2.size > 0 else study_pc2
    window = PCWindow.from_points(all_x, all_y, pad=1.05, var1=var1, var2=var2)

    is_case = (df_subcluster["Group"].astype(str).to_numpy() == str(config.case_label))
    is_ctrl = (df_subcluster["Group"].astype(str).to_numpy() == str(config.control_label))

    figure_path: Path | None = None
    if bool(config.save_plot) or bool(config.show_plot):
        with figure_context(THEME_SUBCLUSTER):
            fig = plot_subcluster_view(
                study_pc1=study_pc1,
                study_pc2=study_pc2,
                is_case=is_case,
                is_ctrl=is_ctrl,
                confidence=confidence,
                reference=ReferenceCloud(
                    pc1=background_pc1,
                    pc2=background_pc2,
                    color=str(config.reference_color),
                    alpha=float(config.reference_alpha),
                ),
                window=window,
                counts=counts,
                n_unlabeled=n_unlabeled,
                case_label=str(config.case_label),
                control_label=str(config.control_label),
                group_label=mainland_label,
                basis_label=str(config.basis_label),
                study_point_size=float(config.study_point_size),
            )

            if bool(config.save_plot):
                figure_path = out_dir / str(config.figure_file)
                save_figure(fig, figure_path, dpi=400)

            if bool(config.show_plot):
                plt.show()
            else:
                plt.close(fig)

    if bool(config.verbose):
        print("\n" + "=" * 80)
        print("SUBCLUSTER VIEW".center(80))
        print("=" * 80)
        print(f"  assigned_group_col : {assigned_group_col}")
        print(f"  mainland_label     : {mainland_label}")
        print(f"  rows (unfiltered)  : {int(df_subcluster.shape[0])}")
        print(f"  unlabeled rows     : {n_unlabeled}")
        if figure_path is not None:
            print(f"  figure_file        : {figure_path}")
        print("=" * 80)

    # The per-group counts were only ever printed. Writing them gives each basis
    # an on-disk record, which is what makes two bases comparable after the fact.
    counts.to_csv(out_dir / str(config.counts_file), sep="\t", index=False)

    return SubclusterViewOutput(
        df_subcluster=df_subcluster,
        figure_path=figure_path,
        counts_by_group=counts,
        output_dir=out_dir,
    )
