"""Shared helpers for the PopGMM step modules.

Import direction is strictly one-way: this module imports nothing from
``scripts.*``, and every other module imports from it. That dissolves the
previous tangle in which ``cluster_all_pcs_kde`` acted as an accidental utility
library for the two ``mainland_*_kde`` modules while itself importing
``_PLOT_STYLE_RC`` back out of ``our_assignment``.

Only helpers that were verified to be behaviourally identical across their
copies live here. Two deliberate exceptions are recorded so the next person does
not "finish the job" and change a figure:

* ``format_pc_axis_label`` takes ``decimals`` because
  ``gmm_component_merging`` formatted variance as ``.1f`` while
  ``gmm_clustering`` and ``hdbscan_filtering`` used ``.2f`` -- i.e. it drew
  ``PC1 (39.2%)`` where the others drew ``PC1 (39.24%)``. The call site in
  the merging module must pass ``decimals=1``.
* ``resolve_pc_columns`` returns *all* PC columns.
  ``mainland_subcluster_only`` needs a 2-PC variant that raises when fewer than
  two are present, so it keeps a thin local wrapper rather than changing this
  one.

``_build_merged_cluster_palette`` and ``_build_premerge_component_palette`` are
NOT here: they have two genuinely different implementations each, and unifying
them would recolour published figures for the sake of ~40 lines.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

#: Shared rcParams for publication figures (dpi=400, large type).
PLOT_STYLE_RC: dict[str, object] = {
    "figure.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "axes.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "legend.fontsize": 22,
    "legend.title_fontsize": 24,
    "figure.titlesize": 40,
}

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


def gwas_neff(case_n: int, control_n: int) -> float:
    """Effective sample size of a case/control set: ``4 / (1/n_case + 1/n_control)``.

    The harmonic-mean form that association power actually scales with, so an
    unbalanced set is penalised relative to its raw total.
    """
    if case_n <= 0 or control_n <= 0:
        return float("nan")
    return float(4.0 / ((1.0 / float(case_n)) + (1.0 / float(control_n))))


def pc12_rgv(xy: np.ndarray) -> float:
    """Residual genetic spread of a sample set on PC1-PC2.

    Root generalized variance: ``det(Sigma) ** (1/4)`` for two dimensions, which
    captures both the variance magnitude and the covariance structure in one
    number, unlike a per-axis variance. Lower means a more homogeneous set.

    Returns NaN for fewer than two samples or a degenerate covariance.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or int(xy.shape[0]) < 2:
        return float("nan")

    cov = np.asarray(np.cov(xy, rowvar=False), dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        return float("nan")

    # Small ridge for numerical stability under near-collinearity.
    trace = float(np.trace(cov))
    scale = trace / 2.0 if np.isfinite(trace) else 1.0
    cov = cov + np.eye(2, dtype=np.float64) * (1e-8 * max(1.0, scale))

    det = float(np.linalg.det(cov))
    if not np.isfinite(det) or det <= 0.0:
        return float("nan")
    return float(det**0.25)


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


def format_template(template: str, *, cluster_id: int, threshold: float) -> str:
    """Expand ``{cluster_id}``/``{threshold*}`` placeholders in a filename."""
    threshold_str = f"{float(threshold):.2f}"
    threshold_tag = threshold_str.replace(".", "p")
    return str(template).format(
        cluster_id=int(cluster_id),
        threshold=float(threshold),
        threshold_str=threshold_str,
        threshold_tag=threshold_tag,
        threshold_pct=int(round(float(threshold) * 100.0)),
    )
