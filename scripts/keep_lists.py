"""Emit the pipeline's deliverable: the ancestry-homogeneous sample lists.

This module owns the deliverable so that nothing else has to. Previously four
different stages each wrote a ``.fid_iid.txt`` somewhere in their own output
directory -- two of them to the *same* path, so one silently overwrote the
other, and one of those built the list with ``FID`` set to the ``IID`` value,
which would not have matched anything in a dataset where the two differ.

A list is a headerless, tab-separated ``FID IID`` file, which is what
``plink --keep`` expects. Alongside them, ``write_keep_lists`` emits a summary
table comparing the lists on the two quantities the rank cut trades off:
effective sample size and residual genetic spread.

Inputs
------
One assignment frame per variant, each carrying ``FID``/``IID``, the PC columns,
and a boolean-or-label column identifying the retained samples.

Outputs
-------
``<keep_list_dir>/<variant>_<display_name>.fid_iid.txt`` per variant, plus
``keep_list_summary.tsv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd

from scripts.common import gwas_neff, pc12_rgv, resolve_pc_columns


class KeepListOutput(NamedTuple):
    #: variant name -> written path
    paths: dict[str, Path]
    #: one row per variant, comparing size / balance / Neff / RGV
    summary: pd.DataFrame
    summary_path: Path | None


@dataclass(frozen=True)
class KeepListConfig:
    #: Directory the lists and the summary are written to.
    output_dir: str | Path = "results/keep_lists"

    #: Name of the comparison table.
    summary_file: str = "keep_list_summary.tsv"

    #: Suffix applied to every list name, e.g. "mainland" -> refined_mainland.
    name_suffix: str = "mainland"

    #: Column names in the incoming frames.
    fid_col: str = "FID"
    iid_col: str = "IID"

    case_label: str = "Case"
    control_label: str = "Control"

    verbose: bool = True


@dataclass(frozen=True)
class KeepListVariant:
    """One list to emit: a name, the retained samples, and how it was chosen."""

    #: Short name; becomes the filename stem together with ``name_suffix``.
    name: str

    #: Rows to retain. Must contain the FID/IID columns and the PC columns.
    frame: pd.DataFrame

    #: Rank cut that produced this variant; None for the uncut full set.
    rank_cut: int | None = None

    #: Pre-merge component ids included, for the summary table.
    component_ids: Sequence[int] = ()


def _fid_iid(frame: pd.DataFrame, fid_col: str, iid_col: str) -> pd.DataFrame:
    """Two-column FID/IID frame, tolerating PLINK's ``#FID`` header."""
    if iid_col not in frame.columns:
        raise KeyError(f"frame must contain {iid_col!r}; got {list(frame.columns)[:12]}")

    if fid_col in frame.columns:
        fid = frame[fid_col]
    elif f"#{fid_col}" in frame.columns:
        fid = frame[f"#{fid_col}"]
    else:
        raise KeyError(
            f"frame must contain {fid_col!r} or '#{fid_col}'. Refusing to fall back to "
            f"{iid_col!r}: a list whose FID column is really the IID will not match "
            f"anything in a dataset where the two differ."
        )

    return pd.DataFrame({fid_col: fid.astype(str).to_numpy(), iid_col: frame[iid_col].astype(str).to_numpy()})


def write_keep_lists(
    *,
    variants: Sequence[KeepListVariant],
    case_iids: Sequence[Any],
    control_iids: Sequence[Any],
    config: KeepListConfig | None = None,
) -> KeepListOutput:
    """Write one keep-list per variant plus the comparison summary."""
    config = config or KeepListConfig()
    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    case_set = {str(x) for x in case_iids}
    control_set = {str(x) for x in control_iids}

    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []

    for variant in variants:
        frame = variant.frame
        ids = _fid_iid(frame, config.fid_col, config.iid_col)
        path = out_dir / f"{variant.name}_{config.name_suffix}.fid_iid.txt"
        ids.to_csv(path, sep="\t", index=False, header=False)
        paths[variant.name] = path

        iid_str = frame[config.iid_col].astype(str)
        n_case = int(iid_str.isin(case_set).sum())
        n_control = int(iid_str.isin(control_set).sum())

        pc_cols = resolve_pc_columns(frame)[:2]
        xy = frame[pc_cols].to_numpy(dtype=np.float64, copy=False) if len(pc_cols) == 2 else np.empty((0, 2))

        rows.append(
            {
                "variant": variant.name,
                "rank_cut": "" if variant.rank_cut is None else int(variant.rank_cut),
                "n_components": len(variant.component_ids),
                "n_samples": int(len(frame)),
                f"{config.case_label}_Count": n_case,
                f"{config.control_label}_Count": n_control,
                "Case_Control_Ratio": (n_case / n_control) if n_control else float("nan"),
                "GWAS_Neff": gwas_neff(n_case, n_control),
                "PC12_RGV": pc12_rgv(xy),
                "components": ",".join(str(int(c)) for c in variant.component_ids),
                "file": path.name,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / str(config.summary_file)
    summary.to_csv(summary_path, sep="\t", index=False)

    if bool(config.verbose):
        print("\n" + "=" * 78)
        print("KEEP LISTS (deliverable)")
        print("=" * 78)
        for row in rows:
            print(
                f"  {row['variant']:<9s} n={row['n_samples']:>6,}  "
                f"{config.case_label}={row[f'{config.case_label}_Count']:>4,}  "
                f"{config.control_label}={row[f'{config.control_label}_Count']:>5,}  "
                f"Neff={row['GWAS_Neff']:>9.2f}  RGV={row['PC12_RGV']:.6f}"
            )
        print(f"  -> {out_dir.as_posix()}")

    return KeepListOutput(paths=paths, summary=summary, summary_path=summary_path)
