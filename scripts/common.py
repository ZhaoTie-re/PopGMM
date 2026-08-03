"""Shared helpers for the PopGMM step modules.

Import direction is strictly one-way: this module imports nothing from
``scripts.*``, and every other module imports from it. That dissolves the
previous tangle in which one KDE module had become an accidental utility
library for the others while itself importing the shared plot style back out of
the assignment module.

Figure styling is **not** here -- it lives in ``scripts.plotting.style``, which
also holds the frozen cohort palette. This module keeps only helpers that a
non-plotting caller could want.

Only helpers that were verified to be behaviourally identical across their
copies live here. Two deliberate exceptions are recorded so the next person does
not "finish the job" and change a figure:

* ``format_pc_axis_label`` takes ``decimals`` because
  ``gmm_component_merging`` formatted variance as ``.1f`` while
  ``gmm_clustering`` and ``hdbscan_filtering`` used ``.2f`` -- i.e. it drew
  ``PC1 (39.2%)`` where the others drew ``PC1 (39.24%)``. The call site in
  ``scripts/plotting/merging.py`` must pass ``decimals=1``.
* ``resolve_pc_columns`` returns *all* PC columns.
  ``subcluster_view`` needs a 2-PC variant that raises when fewer than two are
  present, so it keeps a thin local wrapper rather than changing this one.

``_build_merged_cluster_palette`` and ``_build_premerge_component_palette`` are
NOT here: they have two genuinely different implementations each, and unifying
them would recolour published figures for the sake of ~40 lines.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as _f_dist

_PC_RE = re.compile(r"^PC(\d+)(?:_AVG)?$")

# ---------------------------------------------------------------------------
# Numeric precision
# ---------------------------------------------------------------------------

#: dtype for the matrices HDBSCAN and the GMM are fitted on.
#:
#: This was float32 through the run that produced the original results, which
#: made the pipeline numerically fragile in a measurable way. float32 carries
#: ~7 decimal digits, so summing 181,817 per-sample log-likelihoods surfaces the
#: reduction order used by multi-threaded BLAS: independent runs of the k=2..100
#: search disagreed on 15-18 of the 99 candidates by up to 11.27 BIC units, and
#: with n_init=3 that was occasionally enough to flip which EM restart won.
#:
#: In float64 the same fits are bit-identical across independent processes with
#: BLAS threading left on (measured 5/5 at the two k values that wobble in
#: float32), so determinism no longer requires pinning thread counts.
#:
#: Changing this is NOT a refactor. float32 and float64 disagree strongly about
#: which model is best -- the BIC curve reshuffles rather than shifts (per-k
#: differences from -2222 to +2237, mean |d| 195) and the selected k moves from
#: 26 to 25, which renumbers every component and therefore changes the merged
#: partition, the mainland set, the rank cut and every keep-list. The search
#: also costs ~2.3x more compute.
COMPUTE_DTYPE = np.float64

#: dtype for posterior probabilities and confidences written to disk.
#:
#: Also float32 originally, which meant the confidence thresholds (0.80..0.99)
#: were applied to values already rounded to ~7 digits. Kept equal to
#: COMPUTE_DTYPE so a stored posterior is the one that was computed.
STORE_DTYPE = np.float64

# ---------------------------------------------------------------------------
# PC column handling
# ---------------------------------------------------------------------------


def pc_sort_key(col: str) -> int:
    """Sort key placing PC columns in numeric order and non-PC columns last."""
    match = _PC_RE.match(str(col))
    return int(match.group(1)) if match else 10**9


def pc_index_from_col(col: str) -> int | None:
    """``"PC3_AVG" -> 3``; ``None`` for anything that is not a PC column."""
    match = _PC_RE.match(str(col))
    return int(match.group(1)) if match else None


def resolve_pc_columns(df: pd.DataFrame) -> list[str]:
    """All PC columns of ``df`` in numeric order (possibly empty)."""
    return sorted((c for c in df.columns if _PC_RE.match(str(c))), key=pc_sort_key)


def resolve_all_pc_columns(df: pd.DataFrame) -> list[str]:
    """All PC columns of ``df`` in numeric order; raises when there are none."""
    pc_cols = resolve_pc_columns(df)
    if not pc_cols:
        raise RuntimeError("No PC columns detected (expected e.g., PC1_AVG ...).")
    return pc_cols


def format_pc_axis_label(
    col: str, eigenval: pd.DataFrame | None, *, decimals: int = 2
) -> str:
    """Axis label like ``PC1 (39.24%)``; never shows the ``_AVG`` suffix.

    ``decimals`` exists only to preserve the pre-existing difference between
    modules -- see the module docstring.
    """
    pc_idx = pc_index_from_col(col)
    if pc_idx is None:
        return str(col).replace("_AVG", "")

    base = f"PC{pc_idx}"
    if eigenval is None or eigenval.empty:
        return base
    if "PC" not in eigenval.columns or "variance_explained" not in eigenval.columns:
        return base

    row = eigenval.loc[eigenval["PC"] == pc_idx, "variance_explained"]
    if row.empty:
        return base

    return f"{base} ({float(row.iloc[0]) * 100.0:.{decimals}f}%)"


def format_sigfig(value: float, sig: int = 2) -> str:
    """Format a number with a fixed number of significant digits.

    Uses fixed-point formatting where appropriate to preserve trailing zeros
    (e.g., 5.0 for 2 sig figs), and rounds to tens/hundreds for large values.
    """

    v = float(value)
    if not np.isfinite(v):
        return str(v)
    if v == 0.0:
        return "0"

    sig_i = int(sig)
    if sig_i <= 0:
        sig_i = 1

    abs_v = abs(v)
    exp = int(np.floor(np.log10(abs_v)))
    decimals = sig_i - 1 - exp

    if decimals >= 0:
        return f"{v:.{decimals}f}"

    # decimals < 0 -> round to nearest 10/100/... and print as integer.
    rounded = round(v, -decimals)
    return f"{rounded:.0f}"


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


def build_distinct_palette(n_colors: int) -> list[tuple[float, float, float, float]]:
    """High-contrast categorical palette for many clusters.

    Uses tab20/tab20b/tab20c first (better categorical separability than a
    single colormap), then falls back to evenly spaced HSV colors if needed.
    """
    if n_colors <= 0:
        return []

    palette: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        for i in range(cmap.N):
            palette.append(cmap(i))

    if n_colors > len(palette):
        extra = n_colors - len(palette)
        hsv = plt.get_cmap("hsv")
        for i in range(extra):
            palette.append(hsv((i / max(1, extra)) % 1.0))

    return palette[:n_colors]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaNs pass through untouched."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if not mask.any():
        return out
    p = pvals[mask]
    n = p.size
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[mask] = tmp
    return out


def to_numeric_series(values: Any) -> pd.Series:
    """Column values coerced to numeric; anything non-numeric becomes NaN.

    ``pd.to_numeric`` is declared as returning a wide union (Series, scalar,
    ndarray, NaT, Timestamp, ...), so every downstream ``.to_numpy`` /
    ``.dropna`` / ``.astype`` / ``.iloc`` is reported once per union member --
    a single call site produced nine diagnostics. Passing a Series always
    yields a Series at runtime, so the narrowing is contained here instead of
    being repeated, or suppressed, at every call site.
    """
    return cast(pd.Series, pd.to_numeric(values, errors="coerce"))


def to_numeric_array(values: Any, dtype: Any = np.float64) -> np.ndarray:
    """Column values as a numeric array; anything non-numeric becomes NaN."""
    return to_numeric_series(values).to_numpy(dtype=dtype, copy=False)


def gwas_neff(case_n: int, control_n: int) -> float:
    """Effective sample size of a case/control set: ``4 / (1/n_case + 1/n_control)``.

    The harmonic-mean form that association power actually scales with, so an
    unbalanced set is penalised relative to its raw total.
    """
    if case_n <= 0 or control_n <= 0:
        return float("nan")
    return float(4.0 / ((1.0 / float(case_n)) + (1.0 / float(control_n))))


def rgv(coords: np.ndarray) -> float:
    """Residual genetic spread of a sample set over however many axes it carries.

    Root generalized variance, ``det(Sigma) ** (1 / 2d)`` -- the geometric mean
    of the per-axis standard deviations. It captures the variance magnitude and
    the covariance structure in one number, unlike a per-axis variance, and lower
    means a more homogeneous set. The ``1 / 2d`` exponent keeps the value in the
    units of a standard deviation at every ``d``, so an isotropic cloud of width
    ``s`` scores ``s`` whether it is measured on two axes or ten.

    Two values are still only comparable when they share a basis *and* a ``d``:
    different PCA bases put different amounts of structure on their leading axes.

    Requires more samples than axes -- at ``n <= d`` the sample covariance is
    singular and the result would be an artefact of the stabilising ridge rather
    than of the data. Returns NaN there, and for a degenerate covariance.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        return float("nan")
    n_samples, d = int(coords.shape[0]), int(coords.shape[1])
    if d < 1 or n_samples <= d:
        return float("nan")

    cov = np.atleast_2d(np.asarray(np.cov(coords, rowvar=False), dtype=np.float64))
    if not np.all(np.isfinite(cov)):
        return float("nan")

    # Small ridge for numerical stability under near-collinearity.
    trace = float(np.trace(cov))
    scale = trace / d if np.isfinite(trace) else 1.0
    cov = cov + np.eye(d, dtype=np.float64) * (1e-8 * max(1.0, scale))

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        return float("nan")
    # Via the log determinant: at d = 10 a raw det() underflows long before the
    # spread itself is degenerate.
    return float(np.exp(logdet / (2.0 * d)))


def pc12_rgv(xy: np.ndarray) -> float:
    """``rgv`` on exactly two axes, rejecting anything else.

    The global-PCA spread is defined on PC1-PC2 and nothing else, so callers of
    that specific quantity go through this wrapper rather than passing whatever
    width the frame happens to have.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return float("nan")
    return rgv(xy)


class CaseControlSeparation(NamedTuple):
    """How far apart cases and controls sit in ancestry space."""

    #: Mahalanobis distance between the two group means, in pooled within-group
    #: SD units. Scale-free, so it is comparable across sample sets.
    mahalanobis: float
    #: Hotelling's T-squared -- the same distance weighted by sample size. This
    #: is the test statistic; the distance alone says nothing about evidence.
    hotelling_t2: float
    #: p-value of that test, via the exact F transform. Small means the two
    #: groups really do sit at different places on PC1-PC2.
    p_value: float


def pc12_case_ctrl_separation(
    case_xy: np.ndarray, ctrl_xy: np.ndarray
) -> CaseControlSeparation:
    """Case-vs-control ancestry separation on PC1-PC2.

    Reported as a diagnostic, never optimised against. Three reasons, all
    measured on this dataset rather than assumed:

    * It is not monotone in the rank cut (Spearman -0.49 against k, versus
      +1.00 for ``pc12_rgv``) and its minimum falls on the *uncut* set, which is
      also where GWAS_Neff is largest. Selecting on it would put both objectives
      at the same point, leaving no trade-off and so no decision to make.
    * ``E[D^2]`` carries a ``p * (1/n_case + 1/n_control)`` bias that shrinks as
      the set grows -- from 0.037 at the tightest cut to 0.005 at the widest.
      Minimising it therefore partly just selects for a larger set.
    * It is computed from the case/control labels. Choosing the samples that
      minimise it optimises the very quantity the association test measures, so
      a genuinely ancestry-linked risk would be selected away along with the
      confounding.

    What it is good for is the question ``pc12_rgv`` cannot answer: residual
    spread says how wide the retained set is, not whether cases and controls
    are drawn from the same place within it. Only the second is what biases an
    association test.

    Returns NaN for fewer than two samples in either group, or a singular
    pooled covariance.
    """
    case_xy = np.asarray(case_xy, dtype=np.float64)
    ctrl_xy = np.asarray(ctrl_xy, dtype=np.float64)
    nan = CaseControlSeparation(float("nan"), float("nan"), float("nan"))

    if case_xy.ndim != 2 or ctrl_xy.ndim != 2:
        return nan
    if case_xy.shape[1] != 2 or ctrl_xy.shape[1] != 2:
        return nan

    n_case, n_ctrl, n_dim = int(case_xy.shape[0]), int(ctrl_xy.shape[0]), 2
    if n_case < 2 or n_ctrl < 2:
        return nan

    dof = n_case + n_ctrl - 2
    pooled = (
        (n_case - 1) * np.cov(case_xy, rowvar=False)
        + (n_ctrl - 1) * np.cov(ctrl_xy, rowvar=False)
    ) / float(dof)
    pooled = np.asarray(pooled, dtype=np.float64)
    if not np.all(np.isfinite(pooled)):
        return nan

    # Small ridge for numerical stability under near-collinearity.
    trace = float(np.trace(pooled))
    scale = trace / 2.0 if np.isfinite(trace) else 1.0
    pooled = pooled + np.eye(2, dtype=np.float64) * (1e-8 * max(1.0, scale))

    delta = case_xy.mean(axis=0) - ctrl_xy.mean(axis=0)
    d2 = float(delta @ np.linalg.pinv(pooled) @ delta)
    if not np.isfinite(d2) or d2 < 0.0:
        return nan

    t2 = d2 * (n_case * n_ctrl) / float(n_case + n_ctrl)
    df2 = dof - n_dim + 1
    if df2 <= 0:
        return CaseControlSeparation(float(np.sqrt(d2)), float(t2), float("nan"))

    f_stat = t2 * df2 / float(n_dim * dof)
    p_value = float(cast(Any, _f_dist).sf(f_stat, n_dim, df2))
    return CaseControlSeparation(float(np.sqrt(d2)), float(t2), p_value)


def safe_stats(x: np.ndarray) -> dict[str, float]:
    """mean/std/min/max over the finite entries; all-NaN for an empty input."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else float(np.std(x, ddof=0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


# ---------------------------------------------------------------------------
# Logging and templating
# ---------------------------------------------------------------------------


def setup_file_logger(name: str, log_path: Path) -> logging.Logger:
    """Logger writing to ``log_path`` and stdout with no timestamp prefix.

    The bare ``%(message)s`` format is what makes the emitted ``.log`` files
    byte-comparable between runs, which ``tools/verify_results.py`` relies on.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


