"""Quantify the trade-off that defines the rank cut.

Ranks the major cluster's components by case/control ratio, then walks the
cumulative sets: for each k, the top-k components are merged into one group, the
posteriors are recomputed and renormalized, and samples are reassigned by argmax
-- the same operation the subcluster stage performs, so the reported metrics
describe the sets that would actually be produced.

Two quantities are traded off. ``GWAS_Neff`` is the effective sample size, which
rises as more components are included. Residual genetic spread rises too, and is
reported in two bases that are never compared with each other: ``RGV_Global`` on
the global PCA's PC1-PC2, and ``RGV_Mainland`` on a PCA fitted to the major
cluster, over as many axes as the caller asks for. ``rgv_basis`` picks which one
the Pareto front and the distance-to-ideal score are computed from.

The global PCA's leading axes separate the major cluster from the other regions,
so within it they are nearly constant; a major-cluster PCA spends its axes on the
structure that actually remains. That is the reason for the second basis.

Alongside the trade-off, each cumulative set carries a case/control separation
diagnostic -- how far the two group centroids sit apart within the retained set,
which residual spread cannot express. It is reported in the same two bases and
is never optimised against; ``scripts.common.case_ctrl_separation`` gives the
three reasons why.

This module produces evidence, not a decision. A cut may be forced by the
caller; otherwise the Pareto optimum is reported.

Inputs
------
Cohort assignment results, the merge map, the fitted mixture, case/control lists.

Outputs
-------
Component rank table, cumulative metrics, decision table, the trade-off figure,
and a supplementary case/control separation figure (PNG + PDF).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.rank import plot_methods, plot_selection
from scripts.plotting.style import THEME_RANK, figure_context, save_figure
from scripts.common import gwas_neff as _gwas_neff
from scripts.common import to_numeric_array, to_numeric_series
from scripts.common import case_ctrl_separation as _case_ctrl_separation
from scripts.common import mahalanobis_bias as _mahalanobis_bias
from scripts.common import pc12_case_ctrl_separation as _case_ctrl_separation_pc12
from scripts.common import CaseControlSeparation as _CaseControlSeparation
from scripts.common import pc12_rgv as _global_rgv_pc12
from scripts.common import resolve_pc_columns, rgv as _rgv

#: Stand-in for the rows where no mainland projection was supplied, leaving the
#: mainland separation columns NaN exactly as ``RGV_Mainland`` already is. The
#: noise-floor column reaches NaN by its own route, via ``n_dim = 0``.
_SEPARATION_NAN = _CaseControlSeparation(
    float("nan"), float("nan"), float("nan"), float("nan")
)


#: Weight at which residual spread and case/control separation are given equal
#: say when the intermediate cut is derived. Stated rather than fitted: no data
#: can supply it, and sweeping it (``_weight_sweep``) is what shows the answer
#: does not hinge on the value. Equal weight is the one choice that needs no
#: argument for preferring either measure.
INTERMEDIATE_BLEND_WEIGHT: float = 0.5

#: Operator names written to ``cut_record.tsv``. Which one applies to a
#: cut follows from the geometry of its objective space, not from a choice; see
#: ``_choose_operator``.
OPERATOR_EXCESS_RETURN = "peak_excess_return"
OPERATOR_NEAREST_IDEAL = "nearest_ideal"
OPERATOR_WHOLE_CLUSTER = "whole_cluster"


class RankSelectionOutput(NamedTuple):
    """Rank-progression tables, the recommended cut, and the figure paths."""

    rank_table: pd.DataFrame
    decision_table: pd.DataFrame
    recommended_rank: int | None
    output_dir: Path
    rank_table_path: Path
    decision_table_path: Path
    selection_figure_path: Path | None
    methods_figure_path: Path | None
    #: Variant name -> resolved rank; None is the uncut full set. This is what
    #: the subcluster stage consumes, so the two cannot drift apart.
    rank_cuts: dict[str, int | None]
    cut_selection_table: pd.DataFrame
    cut_selection_path: Path | None


@dataclass(frozen=True)
class RankSelectionConfig:
    """Configuration for the rank-selection trade-off analysis."""

    output_dir: str | Path = "results/03_rank_selection"
    # Two numbered figures, in reading order; tables carry descriptive names
    # because they are reference rather than steps of the argument.
    selection_figure_file: str = "00_selection.png"
    methods_figure_file: str = "01_methods.png"
    rank_table_file: str = "component_ranking.tsv"
    decision_table_file: str = "rank_decision_table.tsv"

    # Supplementary case/control separation figure, in the mainland basis over
    # the same axes as RGV_Mainland. Only written when a mainland projection was
    # supplied, since without one its four columns are all NaN.

    # Variant name -> declared cut: an int pins it, "auto" defers to that
    # variant's rule, "full" is the uncut set. The stage resolves these and
    # reports how, so the next stage never re-derives a cut of its own.
    variant_cuts: "Mapping[str, int | str] | None" = None

    # Consulted when rank_cut_mode is "manual", and reported as the cross-check
    # otherwise. Both are always computed; the mode picks which one is used.
    manual_cuts: "Mapping[str, int] | None" = None

    # "auto" derives narrow and intermediate from their rules; "manual" takes
    # them from manual_cuts. Either way both are evaluated and written to
    # cut_record.tsv, so switching cannot move a cut silently.
    rank_cut_mode: Literal["auto", "manual"] = "auto"

    # Weight given to residual spread when the intermediate cut is derived; the
    # remainder goes to case/control separation. Equal weight by default.
    blend_weight: float = INTERMEDIATE_BLEND_WEIGHT

    cut_selection_file: str = "cut_record.tsv"

    # How many of the ranked mainland components to walk. None means "all of
    # them", discovered from the mainland component list rather than stated as a
    # literal -- the count is a property of the fitted model (17 under the
    # previous float32 run, 16 under float64), so hard-coding it goes stale
    # silently every time the model changes.
    max_rank: int | None = None

    case_label: str = "Case"
    control_label: str = "Control"

    # Residual spread is reported in two bases. RGV_Global is fixed at the global
    # PCA's PC1-PC2 so it stays comparable with earlier runs; RGV_Mainland is
    # computed over this many leading axes of the major-cluster PCA, and only
    # exists when `mainland_coordinates` is supplied.
    mainland_rgv_n_pcs: int = 2

    # Which basis the Pareto front, the normalised utility and the recommended
    # rank are computed from. Defaults to "global" so the stage behaves exactly
    # as before unless a caller opts in.
    rgv_basis: Literal["global", "mainland"] = "global"

    # Last-resort override of the recommended rank, which is otherwise the
    # resolved narrow cut. Prefer pinning "narrow" in variant_cuts: that goes
    # through the same resolver and is recorded in cut_record.tsv, where a
    # value set here is only visible in the config snapshot.
    forced_recommended_rank: int | None = None

    save_plot: bool = True
    show_plot: bool = False
    figure_dpi: int = 600
    verbose: bool = True


def _extract_major_cluster_components(merge_map: pd.DataFrame) -> list[int]:
    if "GMM_Component" not in merge_map.columns:
        raise KeyError("merge_map must contain GMM_Component column.")
    if "Is_Mainland_Merged_Cluster" not in merge_map.columns:
        raise KeyError("merge_map must contain Is_Mainland_Merged_Cluster column.")

    mask = merge_map["Is_Mainland_Merged_Cluster"].astype(bool)
    vals = to_numeric_series(merge_map.loc[mask, "GMM_Component"]).dropna().astype(int)
    return sorted(set(int(v) for v in vals.tolist()))


def _resolve_pc12_columns(df: pd.DataFrame) -> tuple[str, str]:
    preferred_pairs = [
        ("PC1_AVG", "PC2_AVG"),
        ("PC1", "PC2"),
    ]
    for c1, c2 in preferred_pairs:
        if c1 in df.columns and c2 in df.columns:
            return c1, c2

    pc1_candidates = [c for c in df.columns if str(c).upper().startswith("PC1")]
    pc2_candidates = [c for c in df.columns if str(c).upper().startswith("PC2")]
    if pc1_candidates and pc2_candidates:
        return str(pc1_candidates[0]), str(pc2_candidates[0])

    raise KeyError("df_results must contain PC1/PC2 columns (e.g., PC1_AVG and PC2_AVG).")


def _resolve_model_pc_columns(
    df: pd.DataFrame,
    gmm_model: Any,
    gmm_summary: dict[str, Any] | None,
) -> list[str]:
    """Return the PC column names that match the GMM model's feature dimensionality."""
    if isinstance(gmm_summary, dict):
        cols = [c for c in gmm_summary.get("pc_columns_used", []) if c in df.columns]
        if len(cols) >= 2:
            return cols
    n_features = int(getattr(gmm_model, "n_features_in_", 2))
    candidates = sorted(
        [c for c in df.columns if str(c).upper().startswith("PC") and str(c).upper().endswith("_AVG")],
        key=lambda c: int("".join(ch for ch in str(c).split("_AVG")[0] if ch.isdigit()) or "0"),
    )
    if len(candidates) >= n_features:
        return candidates[:n_features]
    any_pc = [c for c in df.columns if str(c).upper().startswith("PC")][:n_features]
    if len(any_pc) < n_features:
        raise KeyError(f"Cannot find {n_features} PC columns in df_results for GMM projection.")
    return any_pc


def _mainland_axes(
    mainland_coordinates: pd.DataFrame,
    n_pcs: int,
) -> list[str]:
    """The leading `n_pcs` PC columns of the major-cluster projection."""
    if "IID" not in mainland_coordinates.columns:
        raise KeyError("mainland_coordinates must contain an IID column.")
    axes = resolve_pc_columns(mainland_coordinates)
    if len(axes) < n_pcs:
        raise ValueError(
            f"mainland_coordinates carries {len(axes)} PC columns but "
            f"mainland_rgv_n_pcs is {n_pcs}."
        )
    return axes[:n_pcs]


def _align_mainland(
    df: pd.DataFrame,
    mainland_coordinates: pd.DataFrame,
    axes: list[str],
) -> np.ndarray:
    """Mainland coordinates as an array row-aligned with `df`, NaN where absent.

    A left merge on IID rather than a reindex: the projection covers only the
    major cluster, so it is expected to be missing rows, and the caller checks
    that the missing ones are never retained.
    """
    right = mainland_coordinates.loc[:, ["IID", *axes]].copy()
    right["IID"] = right["IID"].astype(str)
    if right["IID"].duplicated().any():
        n_dup = int(right["IID"].duplicated().sum())
        raise ValueError(f"mainland_coordinates has {n_dup} duplicated IIDs.")

    merged = df.loc[:, ["IID"]].merge(right, on="IID", how="left", validate="one_to_one")
    return merged[axes].to_numpy(dtype=np.float64, copy=False)


def _safe_minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    span = a_max - a_min
    if (not np.isfinite(span)) or span <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - a_min) / span


#: Grid the blend weight is swept on. Fine enough that a plateau boundary is
#: located to the third decimal, which is well past the precision the choice
#: deserves -- the point is the width of the plateau, not where it ends.
_WEIGHT_GRID_STEP: float = 0.001


def _weight_sweep(
    neff: np.ndarray, het: np.ndarray, sep: np.ndarray, ranks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Which cut wins as the two homogeneity measures are re-weighted.

    Residual spread and case/control separation disagree about which way to go:
    RGV rises monotonically with the cut, while the de-biased separation falls.
    Blending them therefore produces an interior optimum, where either alone
    would only trade against ``GWAS_Neff``.

    For each weight ``w`` the two are min-max normalised and combined as
    ``w * RGV + (1 - w) * separation``, and the cut nearest the ideal corner
    wins -- the same distance-to-ideal score the single-basis recommendation
    uses, so the two are directly comparable.

    ``w`` is a judgement about which kind of residual structure matters, not a
    quantity the data can supply. Sweeping it is the point: a cut that wins
    across a wide band of ``w`` does not depend on having picked one. Note too
    that min-max normalisation is anchored on the observed extremes, so ``w``
    is not an absolute scale -- the plateau boundaries shift if the walk is
    truncated, even though the winner within the balanced band does not.

    Returns the grid and the winning rank at each point on it.
    """
    n_norm = _safe_minmax_norm(neff)
    h_norm = _safe_minmax_norm(het)
    s_norm = _safe_minmax_norm(sep)
    valid = np.isfinite(n_norm) & np.isfinite(h_norm) & np.isfinite(s_norm)
    grid = np.arange(0.0, 1.0 + _WEIGHT_GRID_STEP / 2.0, _WEIGHT_GRID_STEP)
    if not bool(np.any(valid)):
        return grid, np.full(grid.shape, -1, dtype=int)

    n_v, h_v, s_v, r_v = n_norm[valid], h_norm[valid], s_norm[valid], ranks[valid]
    blended = np.outer(grid, h_v) + np.outer(1.0 - grid, s_v)
    dist = np.sqrt(blended ** 2 + (1.0 - n_v) ** 2)
    return grid, r_v[np.argmin(dist, axis=1)].astype(int)



class ObjectiveSpace(NamedTuple):
    """The normalised plane one cut is chosen in.

    Every cut is an optimum over the same two axes -- residual structure to be
    minimised, effective sample size to be maximised -- and the cuts differ only
    in *what is counted as residual structure*. Holding that in one object is
    what makes the three selections one procedure rather than three rules.
    """

    #: What this space counts as residual structure, for the record.
    counted: str
    #: Residual structure per cut, min-max normalised to [0, 1].
    structure: np.ndarray
    #: GWAS_Neff per cut, min-max normalised to [0, 1].
    power: np.ndarray
    #: Whether the structure axis increases strictly with the cut. This decides
    #: which operator is admissible -- see ``_choose_operator``.
    monotone: bool
    #: How far the structure axis travels between the first and last cut. A
    #: monotone axis spans the full [0, 1]; a folded one barely moves.
    chord_span: float


def _objective_space(counted: str, structure: np.ndarray, power: np.ndarray) -> ObjectiveSpace:
    s = _safe_minmax_norm(structure)
    p = _safe_minmax_norm(power)
    finite = np.isfinite(s) & np.isfinite(p)
    mono = bool(finite.all() and s.size > 1 and np.all(np.diff(s[finite]) > 0))
    span = float(abs(s[finite][-1] - s[finite][0])) if int(finite.sum()) > 1 else float("nan")
    return ObjectiveSpace(counted=counted, structure=s, power=p, monotone=mono, chord_span=span)


def _excess_return(
    ranks: np.ndarray, power: np.ndarray, structure: np.ndarray
) -> tuple[int | None, float, float, float, str]:
    """The cut that has bought the most effective sample size for its cost.

    Both quantities rise strictly with the cut here, so the walk has a
    well-defined average exchange rate -- how much ``GWAS_Neff`` one unit of
    residual spread buys, taken end to end:

        rate = (N_K - N_1) / (H_K - H_1)

    Against that benchmark each cut has bought a surplus or a shortfall:

        excess(k) = (N_k - N_1) - rate * (H_k - H_1)

    and the cut where that surplus peaks is the last one that still repays what
    it costs. Beyond it the curve flattens and later components never make the
    accumulated shortfall back.

    Cumulative rather than per-step, because the per-step rate is not monotone:
    on this dataset ranks 10-12 fall below the average and rank 13 springs back
    above it, so "the last step above the average rate" would answer 13. The
    cumulative form answers 9, and is exactly the knee of the curve -- the point
    furthest from the chord joining its ends -- rescaled into N_eff.

    Returns the cut, its excess, its lead over the runner-up, the average rate,
    and the statistic's name. The lead is reported beside the answer, never
    behind it: a peak that leads by a hair is a peak the data does not place.
    """
    ok = np.isfinite(power) & np.isfinite(structure)
    if int(ok.sum()) < 3:
        return None, float("nan"), float("nan"), float("nan"), "excess_neff"

    n_v, h_v, r_v = power[ok], structure[ok], ranks[ok]
    span = float(h_v[-1] - h_v[0])
    if not np.isfinite(span) or span == 0.0:
        return None, float("nan"), float("nan"), float("nan"), "excess_neff"

    rate = float(n_v[-1] - n_v[0]) / span
    excess = (n_v - n_v[0]) - rate * (h_v - h_v[0])
    best = int(np.argmax(excess))
    runner = float(np.sort(excess)[-2]) if excess.size >= 2 else float("nan")
    return (int(r_v[best]), float(excess[best]), float(excess[best] - runner),
            rate, "excess_neff")


def _nearest_ideal(space: ObjectiveSpace, ranks: np.ndarray) -> tuple[int | None, float, str]:
    """The cut closest to the unattainable corner (no structure, all the power).

    What is left when the frontier is not a curve. A folded axis has no turn in
    its exchange rate to find, so there is nothing for ``_knee`` to measure and
    the remaining well-defined choice is proximity to the ideal corner.
    """
    x, y = space.structure, space.power
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 2:
        return None, float("nan"), "distance_to_ideal"
    dist = np.sqrt(x[ok] ** 2 + (1.0 - y[ok]) ** 2)
    return int(ranks[ok][int(np.argmin(dist))]), float(np.min(dist)), "distance_to_ideal"


def _direction_reversals(values: np.ndarray) -> int:
    """How many times a series changes direction.

    Zero for a monotone series. It is what separates the two cuts: residual
    spread never reverses, so a rate can be read off it; case/control separation
    reverses repeatedly, so it has no rate to read and must be combined with
    something that does.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0
    signs = np.sign(np.diff(v))
    signs = signs[signs != 0]
    return int(np.sum(np.diff(signs) != 0)) if signs.size >= 2 else 0


def _blend_margin(
    ranks: np.ndarray, neff: np.ndarray, het: np.ndarray, sep: np.ndarray,
    weight: float, winner: int | None,
) -> float:
    """How far ``weight`` sits from the nearest weight that changes the winner.

    The honest statement of how much the blended answer depends on its weight: a
    winner that changes just outside the chosen value was chosen by the value.
    """
    if winner is None:
        return float("nan")
    grid, won = _weight_sweep(neff, het, sep, ranks)
    on = grid[won == winner]
    return float(min(weight - on.min(), on.max() - weight)) if on.size else float("nan")


def resolve_rank_cuts(
    *,
    decision_table: pd.DataFrame,
    rgv_column: str,
    variant_cuts: "Mapping[str, int | str]",
    manual_cuts: "Mapping[str, int]",
    mode: str,
    blend_weight: float = INTERMEDIATE_BLEND_WEIGHT,
) -> tuple[dict[str, int | None], pd.DataFrame, dict[str, ObjectiveSpace]]:
    """Resolve every variant's cut, and record how each one was arrived at.

    The three cuts are a chain of reasoning, not three settings of one knob:

    ``full``          the major cluster itself -- every component of it. Not an
                      optimum of anything; it is the population the other two
                      are selected *within*.
    ``narrow``        inside it, effective sample size and residual spread both
                      rise strictly with the cut, so the walk has an average
                      exchange rate and ``_excess_return`` can ask which cut has
                      bought the most power for its cost.
    ``intermediate``  case/control separation does *not* fall monotonically as
                      spread improves -- it reverses direction repeatedly -- so
                      there is no rate to read off that axis. The two are
                      combined into one parameter instead, and the cut nearest
                      the ideal corner is taken.

    That is why there are three: one population, one cut priced on the axis that
    has a price, and one that answers the axis that does not.

    Both the rule and the manual value are evaluated whatever the mode, so the
    record can always say whether they agree. The mode selects which is *used*;
    it never changes which are computed, so switching modes cannot move a cut
    without the table showing it moved.

    Returns the cuts keyed by variant -- None being the uncut full set -- one row
    per variant describing the decision, and the spaces themselves for plotting.
    """
    ranks = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=np.float64, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=np.float64, copy=False)
    sep_col = "Mainland_CaseCtrl_D2_Unbiased"
    sep = (
        decision_table[sep_col].to_numpy(dtype=np.float64, copy=False)
        if sep_col in decision_table.columns
        else np.full(ranks.shape, np.nan)
    )
    max_rank = int(ranks.max()) if ranks.size else 0
    w = float(blend_weight)

    spread = _objective_space("residual spread", het, neff)
    blended = _objective_space(
        f"residual spread and case/control separation, weighted {w:g} / {1 - w:g}",
        w * _safe_minmax_norm(het) + (1.0 - w) * _safe_minmax_norm(sep),
        neff,
    )
    spaces: dict[str, ObjectiveSpace] = {"narrow": spread, "intermediate": blended}

    # narrow: priced on the axis that has a price.
    n_rank, n_val, n_margin, exchange_rate, n_stat = _excess_return(ranks, neff, het)
    derived: dict[str, dict[str, Any]] = {
        "narrow": {
            "space": spread, "operator": OPERATOR_EXCESS_RETURN, "rank": n_rank,
            "statistic": n_stat, "value": n_val,
            "margin": n_margin, "margin_kind": "lead_over_runner_up",
        }
    }

    # intermediate: the separation axis reverses, so it is combined rather than
    # priced. The reversal count is what justifies the different treatment, so
    # it is recorded beside the answer.
    i_rank, i_val, i_stat = _nearest_ideal(blended, ranks)
    derived["intermediate"] = {
        "space": blended, "operator": OPERATOR_NEAREST_IDEAL, "rank": i_rank,
        "statistic": i_stat, "value": i_val,
        "margin": _blend_margin(ranks, neff, het, sep, w, i_rank),
        "margin_kind": "weight_to_nearest_boundary",
    }

    # full: the cluster, not an optimum. Its "rank" is the last one walked
    # because that is every component, and nothing was selected to get there.
    derived["full"] = {
        "space": None, "operator": OPERATOR_WHOLE_CLUSTER,
        "rank": max_rank or None, "statistic": "components",
        "value": float(max_rank), "margin": float("nan"), "margin_kind": "",
    }

    reversals = {
        "residual spread": _direction_reversals(het),
        "case/control separation": _direction_reversals(sep),
    }

    cuts: dict[str, int | None] = {}
    rows: list[dict[str, Any]] = []
    for variant, declared in dict(variant_cuts).items():
        if variant not in derived:
            raise ValueError(
                f"variant {variant!r} has no place in the selection framework. Add a "
                f"setting of what counts as residual structure for it, or give it an "
                f"explicit int in params.SUBCLUSTER_VARIANTS."
            )
        spec = derived[variant]
        space = spec["space"]
        manual = manual_cuts.get(variant)
        manual = int(manual) if manual is not None else None
        auto_rank = spec["rank"]

        # A per-variant int overrides the mode in either direction; that is what
        # makes "pin one cut, leave the other derived" expressible.
        if declared == "full":
            source, resolved = "definitional", auto_rank
        elif isinstance(declared, str) and declared == "auto":
            source, resolved = "auto", auto_rank
        elif isinstance(declared, str):
            raise ValueError(
                f"variant {variant!r} declares cut {declared!r}; expected an int, "
                f'"auto", or "full".'
            )
        else:
            source, resolved = "pinned", int(declared)

        if source == "auto" and mode == "manual":
            source, resolved = "manual", manual
        if resolved is None:
            raise ValueError(
                f"variant {variant!r} resolved to no rank under mode {mode!r}: its "
                f"{spec['operator']} rule returned nothing and params carries no "
                f"manual value."
            )
        resolved = int(resolved)
        if not (1 <= resolved <= max_rank):
            raise ValueError(
                f"variant {variant!r} resolved to rank {resolved}, outside the walked "
                f"range 1..{max_rank}."
            )

        # The uncut set is expressed downstream as "exclude nothing" rather than
        # as a rank, which is the same component set by construction.
        cuts[variant] = None if resolved >= max_rank and declared == "full" else resolved
        rows.append({
            "Variant": variant,
            "Structure_Counted": "nothing" if space is None else space.counted,
            "Frontier_Monotone": pd.NA if space is None else bool(space.monotone),
            "Chord_Span": np.nan if space is None else float(space.chord_span),
            "Operator": spec["operator"],
            "Resolved_Rank": resolved,
            "Source": source,
            "Auto_Rank": pd.NA if auto_rank is None else int(auto_rank),
            "Manual_Rank": pd.NA if manual is None else int(manual),
            "Auto_Manual_Agree": (
                pd.NA if (auto_rank is None or manual is None) else bool(auto_rank == manual)
            ),
            "Statistic": spec["statistic"],
            "Value": float(spec["value"]),
            "Axis_Reversals": (
                pd.NA if space is None
                else int(reversals["case/control separation" if variant == "intermediate"
                                   else "residual spread"])
            ),
            "Exchange_Rate": (
                float(exchange_rate) if variant == "narrow" else np.nan
            ),
            "Margin": float(spec["margin"]),
            "Margin_Kind": spec["margin_kind"],
        })

    order = list(dict(variant_cuts).keys())
    table = pd.DataFrame.from_records(rows)
    table["__o"] = table["Variant"].map({v: i for i, v in enumerate(order)})
    table = table.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    return cuts, table, spaces


def run_rank_selection(
    *,
    df_results: pd.DataFrame,
    merge_map: pd.DataFrame,
    case_iids: list[Any],
    control_iids: list[Any],
    gmm_model: Any,
    gmm_summary: dict[str, Any] | None = None,
    mainland_coordinates: pd.DataFrame | None = None,
    config: RankSelectionConfig | None = None,
) -> RankSelectionOutput:
    """Walk the cumulative sets of major-cluster components ranked by case/control ratio.

    Per-cluster ranking uses direct pre-merge MAP assignment from df_results
    (Assigned_Merged_Cluster) to compute case/ctrl ratios for sorting.

    `mainland_coordinates` is an optional frame carrying IID plus the PC columns
    of a major-cluster PCA; supplying it adds the RGV_Mainland column. It is a
    frame rather than a path because every other input to this stage is one.

    Cumulative trade-off metrics (case/ctrl counts, Neff, residual spread) use
    the same composite posterior recomputation the subcluster stage performs: for each k,
    the top-k clusters are merged into a composite group, posteriors are
    re-normalized, and assignment is via argmax.  This ensures consistency
    with the subcluster stage's sample counts.
    """

    config = config or RankSelectionConfig()

    if "IID" not in df_results.columns:
        raise KeyError("df_results must contain IID column.")
    if "Assigned_Merged_Cluster" not in df_results.columns:
        raise KeyError("df_results must contain Assigned_Merged_Cluster column.")

    pc1_col, pc2_col = _resolve_pc12_columns(df_results)
    major_cluster_component_ids = _extract_major_cluster_components(merge_map)
    if len(major_cluster_component_ids) == 0:
        raise ValueError("No mainland clusters identified in merge_map.")

    df = df_results.copy()
    df["IID"] = df["IID"].astype(str)
    df["Assigned_Merged_Cluster"] = to_numeric_series(df["Assigned_Merged_Cluster"]).astype("Int64")
    df[pc1_col] = to_numeric_series(df[pc1_col])
    df[pc2_col] = to_numeric_series(df[pc2_col])

    case_set = set(str(x) for x in case_iids)
    ctrl_set = set(str(x) for x in control_iids)

    df["__is_case"] = df["IID"].isin(list(case_set))
    df["__is_ctrl"] = df["IID"].isin(list(ctrl_set))

    all_cluster_ids = sorted(
        set(int(x) for x in df["Assigned_Merged_Cluster"].dropna().astype(int).tolist())
        | set(int(x) for x in major_cluster_component_ids)
    )

    case_counts: dict[int, int] = {}
    ctrl_counts: dict[int, int] = {}
    total_counts: dict[int, int] = {}
    ratio_map: dict[int, float] = {}

    for cid in all_cluster_ids:
        mask = df["Assigned_Merged_Cluster"] == int(cid)
        case_n = int((mask & df["__is_case"]).sum())
        ctrl_n = int((mask & df["__is_ctrl"]).sum())
        total_n = int(mask.sum())

        case_counts[int(cid)] = case_n
        ctrl_counts[int(cid)] = ctrl_n
        total_counts[int(cid)] = total_n

        if ctrl_n > 0:
            ratio = float(case_n) / float(ctrl_n)
        elif case_n > 0:
            ratio = float("inf")
        else:
            ratio = 0.0
        ratio_map[int(cid)] = ratio

    ranked_components = sorted(
        [int(cid) for cid in major_cluster_component_ids if int(cid) in ratio_map],
        key=lambda cid: (-float(ratio_map[cid]), int(cid)),
    )

    n_mainland = len(ranked_components)
    max_rank = n_mainland if config.max_rank is None else min(int(config.max_rank), n_mainland)
    if max_rank <= 0:
        raise ValueError("No mainland clusters available for ranking.")

    rank_table_records: list[dict[str, Any]] = []
    for rank, cid in enumerate(ranked_components[:max_rank], start=1):
        ratio = ratio_map[int(cid)]
        rank_table_records.append(
            {
                "Rank": int(rank),
                "Cluster": int(cid),
                f"{config.case_label}_Count_MAP": int(case_counts[int(cid)]),
                f"{config.control_label}_Count_MAP": int(ctrl_counts[int(cid)]),
                "Total_Count_MAP": int(total_counts[int(cid)]),
                "Case_Ctrl_Ratio": float(ratio),
            }
        )
    rank_table = pd.DataFrame.from_records(rank_table_records)

    cum_records: list[dict[str, Any]] = []
    selected_clusters = ranked_components[:max_rank]

    # Pre-compute raw posteriors once; the subcluster stage does the same.
    pc_cols_model = _resolve_model_pc_columns(df, gmm_model, gmm_summary)
    x_study = df[pc_cols_model].to_numpy(dtype=np.float64, copy=False)
    probs_original = gmm_model.predict_proba(x_study).astype(np.float64, copy=False)
    old_k = int(probs_original.shape[1])
    is_case_arr = df["__is_case"].to_numpy(dtype=bool, copy=False)
    is_ctrl_arr = df["__is_ctrl"].to_numpy(dtype=bool, copy=False)
    pc1_arr = to_numeric_array(df[pc1_col])
    pc2_arr = to_numeric_array(df[pc2_col])

    mainland_axes: list[str] = []
    mainland_xy: np.ndarray | None = None
    n_mainland_axes: int = 0
    if mainland_coordinates is not None:
        mainland_axes = _mainland_axes(mainland_coordinates, int(config.mainland_rgv_n_pcs))
        mainland_xy = _align_mainland(df, mainland_coordinates, mainland_axes)
        n_mainland_axes = len(mainland_axes)
    elif config.rgv_basis == "mainland":
        raise ValueError(
            'rgv_basis="mainland" needs mainland_coordinates; none was supplied.'
        )

    for k in range(1, max_rank + 1):
        included_clusters = selected_clusters[:k]
        included_set = set(int(c) for c in included_clusters)
        remaining = [c for c in range(old_k) if c not in included_set]

        # Build composite posterior: col 0 = mainland subcluster (sum of included),
        # cols 1..m = individual remaining components, then re-normalize row-wise.
        n_groups = 1 + len(remaining)
        probs_k = np.zeros((probs_original.shape[0], n_groups), dtype=np.float64)
        probs_k[:, 0] = probs_original[:, sorted(included_set)].sum(axis=1)
        for i, cid in enumerate(remaining, start=1):
            probs_k[:, i] = probs_original[:, cid]
        row_sums = probs_k.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        probs_k /= row_sums

        in_sub = np.argmax(probs_k, axis=1) == 0

        case_n = int((in_sub & is_case_arr).sum())
        ctrl_n = int((in_sub & is_ctrl_arr).sum())
        total_n = int(in_sub.sum())

        neff = _gwas_neff(case_n, ctrl_n)

        finite_mask = in_sub & np.isfinite(pc1_arr) & np.isfinite(pc2_arr)
        all_xy = np.stack([pc1_arr[finite_mask], pc2_arr[finite_mask]], axis=1)
        rgv_global = _global_rgv_pc12(all_xy)

        rgv_mainland = float("nan")
        separation_mainland = _SEPARATION_NAN
        if mainland_xy is not None:
            covered = np.isfinite(mainland_xy).all(axis=1)
            # The projection covers the major cluster, and the retained set is
            # nested inside it, so a retained sample without coordinates means the
            # two files disagree. Dropping it would compute the spread and Neff on
            # different sample sets, which is exactly the kind of silent mismatch
            # this stage exists to avoid.
            missing = int((in_sub & ~covered).sum())
            if missing:
                raise ValueError(
                    f"rank {k}: {missing} of {int(in_sub.sum())} retained samples have no "
                    f"mainland coordinates. The projection must cover every sample the "
                    f"walk can retain."
                )
            rgv_mainland = _rgv(mainland_xy[in_sub])
            # Same retained set as RGV_Mainland and GWAS_Neff -- the coverage
            # check above guarantees it, where the global-basis diagnostic below
            # silently drops non-finite rows via finite_mask.
            separation_mainland = _case_ctrl_separation(
                mainland_xy[in_sub & is_case_arr], mainland_xy[in_sub & is_ctrl_arr]
            )

        # Diagnostic only -- reported next to the two selection metrics, never
        # optimised against. See scripts.common.case_ctrl_separation for why
        # minimising it would be the wrong objective.
        separation = _case_ctrl_separation_pc12(
            np.stack([pc1_arr[finite_mask & is_case_arr], pc2_arr[finite_mask & is_case_arr]], axis=1),
            np.stack([pc1_arr[finite_mask & is_ctrl_arr], pc2_arr[finite_mask & is_ctrl_arr]], axis=1),
        )

        cum_records.append(
            {
                "Included_Max_Rank": int(k),
                "Included_Cluster_Count": int(len(included_clusters)),
                "Included_Clusters": ",".join(str(int(x)) for x in included_clusters),
                f"{config.case_label}_Count": int(case_n),
                f"{config.control_label}_Count": int(ctrl_n),
                "Case_Control_Ratio": (float(case_n) / float(ctrl_n)) if ctrl_n > 0 else np.nan,
                "Total_Count": int(total_n),
                "GWAS_Neff": float(neff),
                "RGV_Global": float(rgv_global),
                "RGV_Mainland": float(rgv_mainland),
                "PC12_CaseCtrl_Mahalanobis": float(separation.mahalanobis),
                "PC12_CaseCtrl_HotellingT2": float(separation.hotelling_t2),
                "PC12_CaseCtrl_P": float(separation.p_value),
                "Mainland_CaseCtrl_Mahalanobis": float(separation_mainland.mahalanobis),
                "Mainland_CaseCtrl_D2_Unbiased": float(separation_mainland.d2_unbiased),
                "Mainland_CaseCtrl_HotellingT2": float(separation_mainland.hotelling_t2),
                "Mainland_CaseCtrl_P": float(separation_mainland.p_value),
                "Mainland_CaseCtrl_Noise_Floor": float(_mahalanobis_bias(case_n, ctrl_n, n_mainland_axes)),
            }
        )

    # The cumulative metrics are not written separately: the decision table is
    # this frame plus nine derived columns, so a second file would be a strict
    # subset with identical values.
    decision_table = pd.DataFrame.from_records(cum_records)
    decision_table = decision_table.sort_values("Included_Max_Rank").reset_index(drop=True)

    # One basis drives everything; the other is carried in the table for
    # reference only. The two are on different scales and must never be mixed
    # within a comparison.
    rgv_column = "RGV_Mainland" if config.rgv_basis == "mainland" else "RGV_Global"

    # ===== Resolve the delivered cuts =====
    #
    # Both rules run whatever the mode, so the record can always say whether
    # they agree. Done here rather than in the subcluster stage because the
    # evidence lives here: a cut derived where the decision table is built
    # cannot drift from the table that justifies it.
    rank_cuts, cut_selection_table, objective_spaces = resolve_rank_cuts(
        decision_table=decision_table,
        rgv_column=rgv_column,
        variant_cuts=dict(config.variant_cuts or {}),
        manual_cuts=dict(config.manual_cuts or {}),
        mode=str(config.rank_cut_mode),
        blend_weight=float(config.blend_weight),
    )

    # The recommendation *is* the narrow cut -- the tightest set the evidence
    # supports. It used to be a separate scalarisation (Distance_To_Ideal, which
    # answers a different question and picks a different rank); that column and
    # the other eight the old rules wrote are gone, since nothing read them and
    # they contradicted the derivation the figures now show.
    recommended_rank: int | None = rank_cuts.get("narrow")
    if config.forced_recommended_rank is not None:
        forced = int(config.forced_recommended_rank)
        valid_ranks = decision_table["Included_Max_Rank"].tolist()
        if forced not in valid_ranks:
            raise ValueError(
                f"forced_recommended_rank={forced} is not in the valid rank range "
                f"{valid_ranks[0]}..{valid_ranks[-1]}."
            )
        recommended_rank = forced

    out_dir = Path(str(config.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_table_path = out_dir / str(config.rank_table_file)
    decision_table_path = out_dir / str(config.decision_table_file)
    rank_table.to_csv(rank_table_path, sep="\t", index=False)
    decision_table.to_csv(decision_table_path, sep="\t", index=False)

    cut_selection_path = out_dir / str(config.cut_selection_file)
    cut_selection_table.to_csv(cut_selection_path, sep="\t", index=False)

    # Both figures need the separation column: it is step 3 of the argument and
    # the subject of the methods figure's second half.
    have_sep = (
        mainland_xy is not None
        and bool(np.any(np.isfinite(
            decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=np.float64, copy=False)
        )))
    )

    selection_figure_path: Path | None = None
    methods_figure_path: Path | None = None
    if (bool(config.save_plot) or bool(config.show_plot)) and have_sep:
        neff_arr_f = decision_table["GWAS_Neff"].to_numpy(dtype=np.float64, copy=False)
        het_arr_f = decision_table[rgv_column].to_numpy(dtype=np.float64, copy=False)
        sep_arr_f = decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=np.float64, copy=False)
        rank_arr_f = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
        w_grid, w_winner = _weight_sweep(neff_arr_f, het_arr_f, sep_arr_f, rank_arr_f)

        with figure_context(THEME_RANK):
            fig_sel = plot_selection(
                decision_table=decision_table,
                cut_selection=cut_selection_table,
                rank_table=rank_table,
                objective_spaces=objective_spaces,
                rgv_column=rgv_column,
                mainland_axes=mainland_axes,
                blend_weight=float(config.blend_weight),
                rank_cuts=rank_cuts,
                mode=str(config.rank_cut_mode),
                case_label=str(config.case_label),
                control_label=str(config.control_label),
            )
            if bool(config.save_plot):
                selection_figure_path = out_dir / str(config.selection_figure_file)
                save_figure(fig_sel, selection_figure_path,
                            dpi=int(config.figure_dpi), bbox_inches=None)
            if bool(config.show_plot):
                plt.show()
            else:
                plt.close(fig_sel)

            fig_met = plot_methods(
                decision_table=decision_table,
                cut_selection=cut_selection_table,
                rgv_column=rgv_column,
                mainland_axes=mainland_axes,
                weight_grid=w_grid,
                weight_winner=w_winner,
                blend_weight=float(config.blend_weight),
                rank_cuts=rank_cuts,
                config=config,
            )
            if bool(config.save_plot):
                methods_figure_path = out_dir / str(config.methods_figure_file)
                save_figure(fig_met, methods_figure_path,
                            dpi=int(config.figure_dpi), bbox_inches=None)
            if bool(config.show_plot):
                plt.show()
            else:
                plt.close(fig_met)

    if bool(config.verbose):
        print("\n" + "=" * 92)
        print("RANK SELECTION: EFFECTIVE SAMPLE SIZE vs RESIDUAL SPREAD".center(92))
        print("=" * 92)
        print(f"Mainland clusters ranked (top {max_rank}): {selected_clusters}")
        _rec_source = "(forced)" if config.forced_recommended_rank is not None else "(Pareto-auto)"
        print(f"Recommended rank k   : {recommended_rank}  {_rec_source}")
        print(f"Rank table saved      : {rank_table_path}")
        print(f"Decision table saved  : {decision_table_path}")
        print(f"Cut record saved      : {cut_selection_path}")
        for _label, _path in (("Selection figure", selection_figure_path),
                              ("Methods figure  ", methods_figure_path)):
            if _path is not None:
                print(f"{_label}      : {_path}")
        print("-" * 92)
        print(decision_table.to_string(index=False))
        print("=" * 92 + "\n")

    return RankSelectionOutput(
        rank_table=rank_table,
        decision_table=decision_table,
        recommended_rank=recommended_rank,
        output_dir=out_dir,
        rank_table_path=rank_table_path,
        decision_table_path=decision_table_path,
        selection_figure_path=selection_figure_path,
        methods_figure_path=methods_figure_path,
        rank_cuts=rank_cuts,
        cut_selection_table=cut_selection_table,
        cut_selection_path=cut_selection_path,
    )
