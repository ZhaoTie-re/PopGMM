"""Load the PCA scores and split them into reference panel and study cohort.

Reads the shared PLINK2 ``--score`` matrix in chunks so a large panel does not
have to fit in memory twice, and separates the two cohorts by IID prefix.
Case/control membership is derived from the phenotype column and applies to the
study cohort only -- the reference panel has no phenotype.

PC columns are auto-detected, accepting either ``PC<n>`` or ``PC<n>_AVG``, and
``#FID`` is normalised to ``FID``. The eigenvalue file is assumed to cover every
PC present in the score file, since variance-explained is computed as a share of
the total.

Inputs
------
Eigenvalue file (one value per line, no header) and the ``.sscore`` matrix.

Outputs
-------
In memory only: eigenvalues with variance-explained, the two sample frames, and
the case/control IID lists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from typing import Any, Literal

import pandas as pd

@dataclass(frozen=True)
class DataLoadingConfig:
    chunksize: int = 50000
    sep: str = "\t"
    reference_iid_prefix: str = "bbj_"
    usecols: tuple[str, ...] | None = None
    verbose: bool = True

    # Optional cohort filtering on the IID column.
    # - "include": keep rows whose IID starts with any prefix in iid_prefixes
    # - "exclude": drop rows whose IID starts with any prefix in iid_prefixes
    # - "all": keep all rows (no filtering)
    iid_prefix_mode: Literal["include", "exclude", "all"] = "include"
    iid_prefixes: tuple[str, ...] | None = None

    # Optional phenotype parsing (for case/control IID lists).
    # If phenotype_column is provided and exists in the sscore header, it will be loaded
    # (even when usecols is None and PCs are auto-detected).
    phenotype_column: str | None = None
    case_value: Any = None
    control_value: Any = None


# Backward-compatible alias.
def _col_candidates(col: str) -> list[str]:
    if col == "#FID" or col == "FID":
        return ["#FID", "FID"]
    if col.startswith("PC") and col.endswith("_AVG"):
        return [col, col.removesuffix("_AVG")]
    if col.startswith("PC") and not col.endswith("_AVG"):
        return [f"{col}_AVG", col]
    return [col]


def _resolve_usecols(requested: tuple[str, ...], available_cols: list[str]) -> tuple[list[str], dict[str, str]]:
    resolved: list[str] = []
    rename_map: dict[str, str] = {}

    for req in requested:
        candidates = _col_candidates(req)
        match = next((c for c in candidates if c in available_cols), None)
        if match is None:
            raise ValueError(
                f"Required column '{req}' was not found. Tried candidates: {candidates}"
            )

        resolved.append(match)

        if req in ("#FID", "FID"):
            rename_map[match] = "FID"
        elif req.startswith("PC") and req.endswith("_AVG"):
            rename_map[match] = req
        elif req.startswith("PC") and not req.endswith("_AVG"):
            rename_map[match] = f"{req}_AVG"
        else:
            rename_map[match] = req

    return resolved, rename_map


def _load_eigenval(eigenval_path: str) -> pd.DataFrame:
    eigenval = pd.read_csv(eigenval_path, header=None, names=["eigenvalue"])
    eigenval["eigenvalue"] = pd.to_numeric(eigenval["eigenvalue"], errors="coerce")
    eigenval = eigenval.dropna(subset=["eigenvalue"]).reset_index(drop=True)
    eigenval["variance_explained"] = eigenval["eigenvalue"] / eigenval["eigenvalue"].sum()
    eigenval["cumulative_variance"] = eigenval["variance_explained"].cumsum()
    eigenval.index.name = "PC"
    eigenval = eigenval.reset_index()
    eigenval["PC"] = eigenval["PC"] + 1
    return eigenval


def _prefix_mask(values: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    s = values.astype(str)
    mask = pd.Series(False, index=s.index)
    for p in prefixes:
        mask = mask | s.str.startswith(str(p))
    return mask


def load_cohort_samples(
    sscore_path: str,
    config: DataLoadingConfig | None = None,
) -> pd.DataFrame:
    """Load cohort samples only (chunked), without reading eigenval.

    This is useful when you only need the PC table for downstream mapping and
    want to avoid extra I/O.
    """

    config = config or DataLoadingConfig()

    header_cols = pd.read_csv(sscore_path, sep=config.sep, nrows=0).columns.tolist()

    # Auto-detect all PC columns if usecols is None
    if config.usecols is None:
        pc_cols = [col for col in header_cols if col.startswith("PC")]

        def extract_pc_number(col: str) -> int:
            col_base = col.replace("_AVG", "")
            return int(col_base[2:]) if col_base[2:].isdigit() else 0

        sorted_pcs = sorted(pc_cols, key=extract_pc_number)
        extra_cols: list[str] = []
        if config.phenotype_column and config.phenotype_column in header_cols:
            extra_cols.append(config.phenotype_column)
        requested_cols = ("#FID", "IID") + tuple(extra_cols) + tuple(sorted_pcs)
    else:
        requested_cols = config.usecols

    usecols_resolved, rename_map = _resolve_usecols(requested_cols, header_cols)
    if "IID" not in rename_map.values():
        raise ValueError("IID column must be included in usecols.")

    mode = str(getattr(config, "iid_prefix_mode", "all")).lower()
    prefixes = getattr(config, "iid_prefixes", None)
    if prefixes is None:
        prefixes = ()

    parts: list[pd.DataFrame] = []
    chunk_count = 0
    row_count_read = 0

    chunk = pd.DataFrame()
    for raw_chunk in pd.read_csv(
        sscore_path,
        sep=config.sep,
        usecols=usecols_resolved,
        chunksize=config.chunksize,
    ):
        chunk_count += 1
        chunk = raw_chunk
        row_count_read += len(chunk)
        chunk = chunk.rename(columns=rename_map)

        if mode == "all" or len(prefixes) == 0:
            keep_mask = pd.Series(True, index=chunk.index)
        else:
            prefix_hit = _prefix_mask(chunk["IID"], prefixes)
            if mode == "include":
                keep_mask = prefix_hit
            elif mode == "exclude":
                keep_mask = ~prefix_hit
            else:
                raise ValueError(f"Unsupported iid_prefix_mode: {mode}")

        parts.append(chunk.loc[keep_mask].copy())

    cohort_samples = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(rename_map.values()))

    del parts
    del chunk
    gc.collect()

    cohort_mem_mb = float(cohort_samples.memory_usage(deep=True).sum() / (1024 ** 2))

    if config.verbose:
        print("\n" + "=" * 80)
        print("COHORT SAMPLES LOADING".center(80))
        print("=" * 80)

        print("\n[INPUT CONFIGURATION]")
        print("-" * 80)
        print(f"  Score file path         : {sscore_path}")
        if config.usecols is None:
            print("  Columns requested       : [AUTO-DETECTED ALL PCs]")
        else:
            print(f"  Columns requested       : {', '.join(config.usecols)}")
        print(f"  Columns loaded          : {len(usecols_resolved)}")
        print(f"  Chunk size              : {config.chunksize:,}")
        print(f"  IID prefix mode         : {mode}")
        if len(prefixes) == 0:
            print("  IID prefixes            : [NONE]")
        else:
            print(f"  IID prefixes            : {', '.join(prefixes)}")

        print("\n[DATA PROCESSING SUMMARY]")
        print("-" * 80)
        print(f"  Total rows processed    : {row_count_read:,}")
        print(f"  Cohort samples extracted: {cohort_samples.shape[0]:,} × {cohort_samples.shape[1]} cols")
        print(f"  Memory footprint        : {cohort_mem_mb:.2f} MB")
        print(f"  Chunks read             : {chunk_count}")
        print("=" * 80 + "\n")

    return cohort_samples


def load_study_samples_with_case_control(
    sscore_path: str,
    phenotype_column: str | None = None,
    case_value: Any | None = None,
    control_value: Any | None = None,
    config: DataLoadingConfig | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load non-BBJ samples and return (samples_df, case_iids, control_iids).

    Notes
    -----
    - Uses the same reference-prefix exclusion as the study-cohort loader.
    - Ensures `phenotype_column` is loaded (when present in the sscore header).
    - If the phenotype column is missing, returns empty case/control lists.
    """

    base = config or DataLoadingConfig()

    phenotype_column = phenotype_column if phenotype_column is not None else base.phenotype_column
    case_value = case_value if case_value is not None else base.case_value
    control_value = control_value if control_value is not None else base.control_value

    # Backward-compatible defaults if nothing is provided.
    if phenotype_column is None:
        phenotype_column = "PHENO1"  # letter O; the previous default used a digit zero
    if case_value is None:
        case_value = 2
    if control_value is None:
        control_value = 1

    prefixes = base.iid_prefixes if base.iid_prefixes is not None else (base.reference_iid_prefix,)
    base = DataLoadingConfig(
        chunksize=base.chunksize,
        sep=base.sep,
        reference_iid_prefix=base.reference_iid_prefix,
        usecols=base.usecols,
        verbose=base.verbose,
        iid_prefix_mode="exclude",
        iid_prefixes=prefixes,
        phenotype_column=str(phenotype_column),
        case_value=case_value,
        control_value=control_value,
    )

    samples = load_cohort_samples(sscore_path=sscore_path, config=base)

    if phenotype_column not in samples.columns:
        if base.verbose:
            print(
                f"[WARN] Phenotype column '{phenotype_column}' not found in loaded samples; "
                "case/control lists will be empty."
            )
        return samples, [], []

    pheno_raw = samples[phenotype_column]
    pheno_num = pd.to_numeric(pheno_raw, errors="coerce")
    pheno_str = pheno_raw.astype(str).str.strip().str.lower()

    def _mask_for(target: Any) -> pd.Series:
        if target is None:
            return pd.Series(False, index=samples.index)

        target_str = str(target).strip().lower()
        target_num = pd.to_numeric(pd.Series([target]), errors="coerce").iloc[0]

        mask = pd.Series(False, index=samples.index)
        if target_str != "":
            mask = mask | (pheno_str == target_str)
        if not pd.isna(target_num):
            mask = mask | (pheno_num == float(target_num))
        return mask

    case_mask = _mask_for(case_value)
    ctrl_mask = _mask_for(control_value)

    case_iids = [str(x) for x in pd.Series(samples.loc[case_mask, "IID"]).to_list()]
    ctrl_iids = [str(x) for x in pd.Series(samples.loc[ctrl_mask, "IID"]).to_list()]

    # The phenotype column is only needed to derive case/control IID lists.
    # Keep the study-cohort frame lean for downstream stages.
    samples_out = samples.drop(columns=[phenotype_column], errors="ignore")
    return samples_out, case_iids, ctrl_iids


def load_reference_data(
    eigenval_path: str,
    sscore_path: str,
    config: DataLoadingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or DataLoadingConfig()

    header_cols = pd.read_csv(sscore_path, sep=config.sep, nrows=0).columns.tolist()
    
    # Auto-detect all PC columns if usecols is None
    if config.usecols is None:
        pc_cols = [col for col in header_cols if col.startswith("PC")]
        # Sort by PC number (handle both PC1_AVG and PC1 formats)
        def extract_pc_number(col: str) -> int:
            col_base = col.replace("_AVG", "")
            return int(col_base[2:]) if col_base[2:].isdigit() else 0
        
        sorted_pcs = sorted(pc_cols, key=extract_pc_number)
        requested_cols = ("#FID", "IID") + tuple(sorted_pcs)
    else:
        requested_cols = config.usecols
    
    usecols_resolved, rename_map = _resolve_usecols(requested_cols, header_cols)

    if "IID" not in rename_map.values():
        raise ValueError("IID column must be included in usecols.")



    eigenval = _load_eigenval(eigenval_path)

    reference_parts: list[pd.DataFrame] = []
    chunk_count = 0
    row_count_read = 0

    chunk = pd.DataFrame()
    for raw_chunk in pd.read_csv(
        sscore_path,
        sep=config.sep,
        usecols=usecols_resolved,
        chunksize=config.chunksize,
    ):
        chunk_count += 1
        chunk = raw_chunk
        row_count_read += len(chunk)
        chunk = chunk.rename(columns=rename_map)
        reference_mask = chunk["IID"].astype(str).str.startswith(config.reference_iid_prefix)
        reference_parts.append(chunk.loc[reference_mask].copy())

    reference_samples = pd.concat(reference_parts, ignore_index=True) if reference_parts else pd.DataFrame(columns=list(rename_map.values()))

    del reference_parts
    del chunk
    gc.collect()

    pc1_var = float(eigenval["variance_explained"].iloc[0]) if len(eigenval) >= 1 else 0.0
    pc12_cum = float(eigenval["cumulative_variance"].iloc[1]) if len(eigenval) >= 2 else pc1_var
    reference_mem_mb = float(reference_samples.memory_usage(deep=True).sum() / (1024 ** 2))

    summary = {
        "eigenvalues_shape": eigenval.shape,
        "rows_read": row_count_read,
        "bbj_samples_shape": reference_samples.shape,
        "pc1_explained": pc1_var,
        "pc1_2_cumulative": pc12_cum,
        "memory_bbj_mb": reference_mem_mb,
        "chunks_processed": chunk_count,
    }

    if config.verbose:
        print("\n" + "=" * 80)
        print("REFERENCE PANEL LOADING".center(80))
        print("=" * 80)
        
        print("\n[INPUT CONFIGURATION]")
        print("-" * 80)
        print(f"  Eigenvalue path         : {eigenval_path}")
        print(f"  Score file path         : {sscore_path}")
        if config.usecols is None:
            print(f"  Columns requested       : [AUTO-DETECTED ALL PCs]")
        else:
            print(f"  Columns requested       : {', '.join(config.usecols)}")
        print(f"  Columns loaded          : {len(usecols_resolved)}")
        print(f"  Chunk size              : {config.chunksize:,}")
        
        print("\n[DATA PROCESSING SUMMARY]")
        print("-" * 80)
        print(f"  Eigenvalues loaded      : {summary['eigenvalues_shape'][0]:,} PCs × {summary['eigenvalues_shape'][1]} metrics")
        print(f"  Total rows processed    : {summary['rows_read']:,}")
        print(f"  BBJ samples extracted   : {summary['bbj_samples_shape'][0]:,} × {summary['bbj_samples_shape'][1]} cols")
        print(f"  PC1 variance explained  : {summary['pc1_explained'] * 100:.2f}%")
        print(f"  PC1-2 cumulative var.   : {summary['pc1_2_cumulative'] * 100:.2f}%")
        print(f"  Memory footprint        : {summary['memory_bbj_mb']:.2f} MB")
        print(f"  Chunks read             : {summary['chunks_processed']}")
        
        print("=" * 80 + "\n")

    return eigenval, reference_samples


def load_reference_and_study(
    *,
    eigenval_path: str,
    sscore_path: str,
    config: DataLoadingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load both cohorts in one call from a single config.

    Loads, from the same `.sscore` file:
    - `eigenval` (from eigenval_path)
    - `bbj_samples` (IIDs starting with `config.bbj_prefix`)
    - `study_samples` (IIDs NOT starting with `config.reference_iid_prefix`)
    - `our_case_iids`, `our_ctrl_iids` parsed from `config.phenotype_column`

    Notes
    -----
    - Case/control lists are derived only for OUR samples.
    - Phenotype parsing uses `config.phenotype_column`, `config.case_value`,
      and `config.control_value` (with backward-compatible defaults inside
      `load_study_samples_with_case_control`).
    """

    base = config or DataLoadingConfig()
    silent = replace(base, verbose=False)

    eigenval, reference_samples = load_reference_data(
        eigenval_path=eigenval_path,
        sscore_path=sscore_path,
        config=silent,
    )

    study_samples, case_iids, control_iids = load_study_samples_with_case_control(
        sscore_path=sscore_path,
        config=silent,
    )

    if base.verbose:
        reference_mem_mb = float(reference_samples.memory_usage(deep=True).sum() / (1024 ** 2))
        study_mem_mb = float(study_samples.memory_usage(deep=True).sum() / (1024 ** 2))
        eigenval_shape = tuple(eigenval.shape)
        reference_shape = tuple(reference_samples.shape)
        study_shape = tuple(study_samples.shape)

        phenotype_column = base.phenotype_column
        case_value = base.case_value
        control_value = base.control_value

        pc1_var = float(eigenval["variance_explained"].iloc[0]) if len(eigenval) >= 1 else 0.0
        pc12_cum = float(eigenval["cumulative_variance"].iloc[1]) if len(eigenval) >= 2 else pc1_var

        print("\n" + "=" * 80)
        print("DATA LOADING (REFERENCE PANEL + STUDY COHORT)".center(80))
        print("=" * 80)

        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  Eigenvalue path      : {eigenval_path}")
        print(f"  Score file path      : {sscore_path}")
        print(f"  Chunk size           : {base.chunksize:,}")
        print(f"  BBJ prefix           : {base.reference_iid_prefix}")
        if phenotype_column is None:
            print("  Phenotype column     : [NONE]")
        else:
            print(f"  Phenotype column     : {phenotype_column}")
            print(f"  Case / Control value : {case_value!r} / {control_value!r}")

        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  Eigenvalues          : {eigenval_shape[0]:,} PCs × {eigenval_shape[1]} metrics")
        print(f"  PC1 variance explained: {pc1_var * 100:.2f}%")
        print(f"  PC1-2 cumulative var.: {pc12_cum * 100:.2f}%")
        print(f"  BBJ samples          : {reference_shape[0]:,} × {reference_shape[1]} cols ({reference_mem_mb:.2f} MB)")
        print(f"  OUR samples          : {study_shape[0]:,} × {study_shape[1]} cols ({study_mem_mb:.2f} MB)")
        if phenotype_column is not None:
            if len(case_iids) == 0 and len(control_iids) == 0:
                print("  OUR cases / ctrls    : [PHENOTYPE COLUMN NOT FOUND]")
            else:
                print(f"  OUR cases / ctrls    : {len(case_iids):,} / {len(control_iids):,}")
        print("=" * 80 + "\n")

    return eigenval, reference_samples, study_samples, case_iids, control_iids
