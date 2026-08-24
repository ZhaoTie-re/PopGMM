"""The rank-selection figure: seven formulas, and the three cohorts they fix.

``plot_selection`` is the whole stage on one canvas. Each rung states a
quantity or a rule as a formula, says in one line what it means, and carries
its evidence beside it; the seventh is what falls out -- three cohorts, and why
three rather than one.

It was four figures once, then two, and the reasoning had to be assembled
across them. The formulas are the argument, so they belong on the same page as
the argument.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from scripts.common import to_numeric_array

if TYPE_CHECKING:
    from scripts.rank_selection import RankSelectionConfig


# ---------------------------------------------------------------------------
# One spec for the whole family
# ---------------------------------------------------------------------------
#
# The two figures are read together, so they share a canvas, a column split, a
# title grammar and a palette. Anything set here must not be re-declared inside
# a figure: a second copy is how the same cut ended up drawn in two blues.

#: Canvas every figure in this directory uses.
FIGURE_SIZE: "tuple[float, float]" = (19.0, 13.0)

#: Data on the left, typeset methods on the right, in this ratio.
COLUMN_RATIOS: "tuple[float, float]" = (3.15, 2.85)
COLUMN_WSPACE: float = 0.055

#: Canonical order of the delivered cuts: ascending in k, which is also the
#: nesting order (narrow subset of intermediate subset of full). Every figure
#: that lists all three lists them this way.
CUT_ORDER: "tuple[str, str, str]" = ("narrow", "intermediate", "full")

#: Order the *reasoning* runs in, which is not the order the results list in.
#: ``full`` comes first because it is the population the other two are selected
#: within; the cuts then follow in the order they were arrived at.
NARRATIVE_ORDER: "tuple[str, str, str]" = ("full", "narrow", "intermediate")

#: The four figures, in reading order. Each names its place and its neighbours,
#: so one lifted out of the directory still says where it sits.
FIGURE_SERIES: "tuple[tuple[str, str], ...]" = (
    ("00_problem", "what has to be decided"),
    ("01_narrow", "the first cut"),
    ("02_intermediate", "the obstacle, and the second cut"),
    ("03_cohorts", "the three delivered sets"),
)


def _series_tag(name: str) -> str:
    """"2 of 4 · ... — after X, before Y", for a figure's footer."""
    names = [n for n, _ in FIGURE_SERIES]
    i = names.index(name)
    parts = [f"Rank selection · {i + 1} of {len(names)} · {FIGURE_SERIES[i][1]}"]
    if i > 0:
        parts.append(f"after {names[i - 1]}.png")
    if i < len(names) - 1:
        parts.append(f"before {names[i + 1]}.png")
    return "   —   ".join(parts)


def _series_footer(fig: Figure, name: str) -> None:
    """Stamp the figure's place in the series along the bottom."""
    fig.text(0.5, 0.012, _series_tag(name), fontsize=10.5, color=_DIM,
             ha="center", va="bottom", fontstyle="italic")


#: Text greys, shared by both figures' typeset blocks.
_BK, _GR, _DIM = "#212121", "#424242", "#757575"
_X_INDENT = 0.05

#: Per-cut identity. The only source of these colours.
_TINT = {"narrow": "#E7F1F8", "intermediate": "#E6F4EC", "full": "#FBEAEC"}
_EDGE = {"narrow": "#0571B0", "intermediate": "#008837", "full": "#B2182B"}

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

_PANEL_TITLE_SIZE = 14.0
_SUPTITLE_SIZE = 18.0


def _panel_title(ax: "plt.Axes", letter: str, text: str) -> None:
    """Panel heading, in the one style the family uses."""
    ax.set_title(f"{letter} · {text}", fontsize=_PANEL_TITLE_SIZE,
                 fontweight="bold", loc="left", pad=9)


def _figure_title(fig: Figure, title: str, qualifier: str) -> None:
    """Figure heading: Title Case noun phrase, em dash, lowercase qualifier."""
    fig.suptitle(f"{title} — {qualifier}", fontsize=_SUPTITLE_SIZE,
                 fontweight="bold", y=0.972)


def _reversals(values: np.ndarray) -> int:
    """How many times a series changes direction; zero when monotone.

    Mirrors ``rank_selection._direction_reversals``. Duplicated rather than
    imported because that module imports this one.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0
    signs = np.sign(np.diff(v))
    signs = signs[signs != 0]
    return int(np.sum(np.diff(signs) != 0)) if signs.size >= 2 else 0


def _safe_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; zeros for a degenerate range.

    Mirrors ``rank_selection._safe_minmax_norm``. Duplicated rather than imported
    because that module imports this one.
    """
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    span = hi - lo
    if (not np.isfinite(span)) or span <= 0:
        return np.zeros_like(a)
    return (a - lo) / span


def _note_axis(ax: "plt.Axes") -> None:
    """Turn an axes into a blank sheet with unit coordinates for typeset text."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _symbol_table(
    ax: "plt.Axes", rows: "Sequence[tuple[str, str, str]]", *,
    top: float = 0.92, gap: float = 0.078, title: str = "Symbols",
) -> None:
    """Draw a symbol / meaning / value-on-this-run table.

    The third column is the point of it: a reader who cannot check a symbol's
    value against the plot beside it has to take the formula on trust, which is
    what these figures exist to avoid.
    """
    n = max(len(rows), 1)
    gap = min(gap, (top - 0.04) / n)
    ax.text(0.0, top + 0.070, title, fontsize=12.5, fontweight="bold",
            ha="left", va="center", color=_BK)
    ax.plot([0.0, 1.0], [top + 0.030, top + 0.030], color="#CFCFCF", linewidth=1.0)
    for i, (sym, meaning, value) in enumerate(rows):
        y = top - (i + 0.5) * gap
        ax.text(0.006, y, sym, fontsize=12.0, ha="left", va="center", color=_BK)
        ax.text(0.150, y, meaning, fontsize=11.0, ha="left", va="center", color=_GR)
        ax.text(0.998, y, value, fontsize=11.0, ha="right", va="center",
                color=_BK, fontweight="bold")


def _formula(ax: "plt.Axes", y: float, tex: str, size: float = 16.0,
             indent: float = 0.03) -> None:
    """One displayed formula, left-aligned under its heading."""
    ax.text(indent, y, tex, fontsize=size, ha="left", va="center", color=_BK)


def _says(ax: "plt.Axes", y: float, text: str, width: int = 92,
          colour: str = _GR, size: float = 11.5) -> None:
    """What the formula above it means, wrapped to the column."""
    ax.text(0.03, y, "\n".join(textwrap.wrap(text, width=width)), fontsize=size,
            ha="left", va="top", color=colour, linespacing=1.5)


def _verdict(ax: "plt.Axes", text: str, colour: str) -> None:
    """The step's conclusion, banded across the foot of its evidence."""
    ax.text(0.5, 0.02, text, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", ha="center", va="bottom", color=colour, zorder=9,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=colour, linewidth=1.3, alpha=0.96))


def _stage(n_evidence: int, table_rows: int) -> "tuple[Figure, Any, Any, Any]":
    """The frame every figure in this stage uses.

    Formulas and what they mean on the left; the evidence and then the symbol
    table on the right. The table gets its own axes rather than the foot of the
    prose column -- ten symbols do not fit under three formulas, and putting
    them there is what pushed the longest figure off the page.
    """
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.09)
    ax_text = fig.add_subplot(gs[0, 0])
    _note_axis(ax_text)
    heights = [1.0] * n_evidence + [max(0.30, 0.088 * table_rows)]
    right = gs[0, 1].subgridspec(n_evidence + 1, 1, height_ratios=heights, hspace=0.34)
    ax_tab = fig.add_subplot(right[n_evidence, 0])
    _note_axis(ax_tab)
    return fig, ax_text, right, ax_tab


def plot_problem(
    *,
    decision_table: pd.DataFrame,
    rank_table: pd.DataFrame,
    rgv_column: str,
    mainland_axes: Sequence[str],
    basis: str,
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """1 of 4 — what has to be decided, and the two quantities that decide it.

    Defines the walk first, because ``k`` appears in every formula that follows
    and was never said anywhere: the major cluster's components are ordered by
    case/control ratio, and cut ``k`` keeps the top ``k``. Then the two
    quantities, each derived rather than asserted.
    """
    d = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    n1 = decision_table[f"{case_label}_Count"].to_numpy(dtype=int, copy=False)
    n2 = decision_table[f"{control_label}_Count"].to_numpy(dtype=int, copy=False)
    ratios = rank_table.sort_values("Rank")["Case_Ctrl_Ratio"].to_numpy(dtype=float)

    fig, ax, right, ax_tab = _stage(2, 9)
    ax_r = fig.add_subplot(right[0, 0])
    ax_b = fig.add_subplot(right[1, 0])

    ax.text(0.0, 0.985, "The question", fontsize=14.5, fontweight="bold",
            ha="left", va="center", color=_BK)
    _says(ax, 0.955, f"The major cluster has K = {rank.max()} components. Keeping more of them raises "
                     f"power and raises heterogeneity together. Which do we keep? Order the components "
                     f"by {case_label}/{control_label} ratio, and let cut k keep the top k — one nested "
                     f"set per k. Two quantities change along that walk.")

    ax.text(0.0, 0.815, r"1 ·  Effective sample size  $N_k$", fontsize=13.5,
            fontweight="bold", ha="left", va="center", color=_BK)
    _says(ax, 0.788, "The variance of an allele-frequency difference in an unbalanced set, equated to "
                     "the balanced set that would have the same variance:")
    _formula(ax, 0.700, r"$\mathrm{Var} = f(1-f)\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)"
                        r"\;\equiv\;\dfrac{2f(1-f)}{N_k}"
                        r"\;\;\Longrightarrow\;\; N_k = \dfrac{4\,n_1 n_2}{n_1 + n_2}$", 15.0)
    _says(ax, 0.630, r"$f$ is the allele frequency and cancels, so $N_k$ depends only on the two counts. "
                     r"An unbalanced set is worth less than its raw total: 439 + 2,662 samples give "
                     r"$N_k$ = 1,507, not 3,101.")

    ax.text(0.0, 0.525, rf"2 ·  Residual spread  $H_k$", fontsize=13.5,
            fontweight="bold", ha="left", va="center", color=_BK)
    _says(ax, 0.498, f"How wide the retained set still is on the {d} leading axes of a PCA fitted to the "
                     f"major cluster itself:")
    _formula(ax, 0.412, rf"$H_k = \left|\Sigma_k\right|^{{1/2d}} = "
                        rf"\left(\prod_{{i=1}}^{{d}} \sqrt{{\lambda_i}}\right)^{{1/d}}, \qquad d = {d}$", 15.0)
    _says(ax, 0.348, r"$\Sigma_k$ is the sample covariance of the retained samples and $\lambda_i$ its "
                     r"eigenvalues, so $H_k$ is the geometric mean of the per-axis SDs — one number "
                     r"carrying both the variances and their correlations. Adding samples from further "
                     r"out can only raise it.")

    _symbol_table(ax_tab, (
        (r"$K$", "components in the major cluster", f"{rank.max()}"),
        (r"$k$", f"cut: keep the top $k$ by {case_label}/{control_label} ratio", f"1 … {rank.max()}"),
        (r"$n_1,\ n_2$", f"{case_label} and {control_label} retained at cut $k$",
         f"{n1.min()}→{n1.max()},  {n2.min()}→{n2.max()}"),
        (r"$f$", "allele frequency (cancels)", "—"),
        (r"$N_k$", "effective sample size", f"{neff.min():,.1f} → {neff.max():,.1f}"),
        (r"$d$", f"{basis} PCA axes used", f"{d}"),
        (r"$\Sigma_k$", f"covariance of the retained samples on those {d} axes", f"{d}×{d}"),
        (r"$\lambda_i$", r"its eigenvalues", r"$i = 1 \ldots d$"),
        (r"$H_k$", "residual spread", f"{het.min():.6f} → {het.max():.6f}"),
    ), top=0.90, gap=0.098)

    # ── evidence: the ranking, then the two quantities rising ────────
    ax_r.bar(np.arange(1, len(ratios) + 1), ratios, color="#BDBDBD",
             edgecolor="white", linewidth=0.8)
    ax_r.set_xticks(np.arange(1, len(ratios) + 1)[::2])
    ax_r.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_r.set_ylabel(f"{case_label}/{control_label} ratio", fontsize=10.5)
    ax_r.tick_params(labelsize=9.5)
    _panel_title(ax_r, "A", "the order the walk follows")

    ax_b.plot(rank, _safe_norm(neff), "-o", color=_BK, markersize=4.2, linewidth=1.6,
              markerfacecolor="white", markeredgewidth=1.0, label=r"$N_k$")
    ax_b.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=4.2, linewidth=1.6,
              markerfacecolor="white", markeredgewidth=1.0, label=r"$H_k$")
    ax_b.set_xticks(rank[::2]); ax_b.set_ylim(-0.08, 1.32)
    ax_b.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_b.set_ylabel("min-max normalised", fontsize=10.5)
    ax_b.tick_params(labelsize=9.5)
    ax_b.legend(loc="upper left", fontsize=10.0, ncol=2, frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")
    _panel_title(ax_b, "B", "both rise strictly with $k$")
    _verdict(ax_b, "Neither can be improved without giving up the other", _BK)

    for a in (ax_r, ax_b):
        a.grid(True, alpha=0.30, linewidth=0.7); a.set_axisbelow(True)

    _figure_title(fig, "The Problem", "what has to be decided, and what changes as the set widens")
    fig.subplots_adjust(left=0.030, right=0.985, top=0.905, bottom=0.075)
    _series_footer(fig, "00_problem")
    return fig


def plot_narrow(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rgv_column: str,
    rank_cuts: "Mapping[str, int | None]",
) -> Figure:
    """2 of 4 — the first cut, priced against the walk's own exchange rate."""
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    rate = float(rows["narrow"]["Exchange_Rate"])
    margin = float(rows["narrow"]["Margin"])
    k_nar = int(rows["narrow"]["Resolved_Rank"])
    excess = (neff - neff[0]) - rate * (het - het[0])
    step = np.diff(neff) / np.diff(het)
    above = [int(rank[i + 1]) for i in range(len(step)) if step[i] >= rate]
    colour = _EDGE["narrow"]

    fig, ax, right, ax_tab = _stage(2, 5)
    ax_s = fig.add_subplot(right[0, 0])
    ax_e = fig.add_subplot(right[1, 0])

    ax.text(0.0, 0.985, "Both axes rise, so the walk has one price",
            fontsize=14.5, fontweight="bold", ha="left", va="center", color=colour)
    _says(ax, 0.955, "Taken end to end, the walk exchanges spread for power at a single average rate. "
                     "Each cut can then be scored by how much power it bought above that rate.")
    _formula(ax, 0.858, r"$r = \dfrac{N_K - N_1}{H_K - H_1}$"
                        rf"$\;=\;{rate:,.1f}$", 16.0)
    _formula(ax, 0.762, r"$E_k = (N_k - N_1) \;-\; r\,(H_k - H_1)$", 16.0)
    _says(ax, 0.700, r"$E_k$ is in units of $N_{eff}$: the surplus, in effective samples, that cut $k$ "
                     r"has over paying the average price for the spread it took on. Where $E_k$ peaks, "
                     r"spread stops repaying.")

    ax.text(0.0, 0.590, "Cumulative, not per-step", fontsize=13.5, fontweight="bold",
            ha="left", va="center", color=_BK)
    _says(ax, 0.562, f"The per-step rate is not monotone. Steps {', '.join(str(x) for x in above[-3:])} "
                     f"sit above the average while the ones between them fall below it, so 'the last "
                     f"step above the average rate' would answer k = {max(above)}. The cumulative form "
                     f"asks a different question — has the walk *so far* repaid — and answers "
                     f"k = {k_nar}.")

    ax.text(0.0, 0.430, "The first cut", fontsize=13.5, fontweight="bold",
            ha="left", va="center", color=colour)
    _formula(ax, 0.372, r"$k_{\mathrm{narrow}} = \arg\max_k E_k$"
                        rf"$\; = \;{k_nar}$", 16.0)

    _symbol_table(ax_tab, (
        (r"$N_1,\ H_1$", r"the two quantities at $k = 1$",
         f"{neff[0]:,.1f},  {het[0]:.6f}"),
        (r"$N_K,\ H_K$", rf"and at $k = K = {rank.max()}$",
         f"{neff[-1]:,.1f},  {het[-1]:.6f}"),
        (r"$r$", r"average exchange rate, $N_{eff}$ per unit spread", f"{rate:,.1f}"),
        (r"$E_k$", r"surplus $N_{eff}$ over that rate", f"{excess.min():,.0f} → {excess.max():,.0f}"),
        (r"margin", "peak's lead over the runner-up", f"{margin:.1f} $N_{{eff}}$"),
    ), top=0.90, gap=0.098)

    # ── evidence ─────────────────────────────────────────────────────
    ax_s.axhline(rate, color=colour, linewidth=1.4, linestyle="--")
    ax_s.text(rank[-1], rate, f"  $r$ = {rate:,.0f}", color=colour, fontsize=10.0,
              va="bottom", ha="right", fontweight="bold")
    ax_s.plot(rank[1:], step, "-o", color=_GR, markersize=4.0, linewidth=1.4,
              markerfacecolor="white", markeredgewidth=1.0)
    ax_s.set_yscale("log")
    ax_s.set_xticks(rank[::2])
    ax_s.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_s.set_ylabel(r"per-step $\Delta N/\Delta H$", fontsize=10.5)
    ax_s.tick_params(labelsize=9.5)
    _panel_title(ax_s, "A", f"the per-step rate rebounds — it would answer $k$ = {max(above)}")

    ax_e.axhline(0.0, color=_DIM, linewidth=1.0, linestyle="--")
    ax_e.vlines(rank, 0.0, excess, color="#D6D6D6", linewidth=4.0)
    i_pk = int(np.argmax(excess))
    ax_e.vlines(rank[i_pk], 0.0, excess[i_pk], color=colour, linewidth=4.0)
    ax_e.plot(rank, excess, "-", color=_GR, linewidth=1.3)
    ax_e.plot([rank[i_pk]], [excess[i_pk]], _MARK["narrow"], color=colour, markersize=11.0,
              markeredgecolor="white", markeredgewidth=1.3)
    ax_e.set_xticks(rank[::2])
    ax_e.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_e.set_ylabel(r"$E_k$  (excess $N_{eff}$)", fontsize=10.5)
    ax_e.tick_params(labelsize=9.5)
    _panel_title(ax_e, "B", rf"$E_k$ peaks at $k$ = {k_nar},  +{excess[i_pk]:,.1f}")
    _verdict(ax_e, f"narrow  =  {k_nar}", colour)

    for a in (ax_s, ax_e):
        a.grid(True, alpha=0.30, linewidth=0.7); a.set_axisbelow(True)

    _figure_title(fig, "The First Cut", "where spread stops repaying in power")
    fig.subplots_adjust(left=0.030, right=0.985, top=0.905, bottom=0.075)
    _series_footer(fig, "01_narrow")
    return fig


def plot_intermediate(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    objective_spaces: "Mapping[str, object]",
    rgv_column: str,
    mainland_axes: Sequence[str],
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    blend_weight: float,
    case_label: str = "Case",
    control_label: str = "Control",
    safe_weight_floor: float = 0.5,
) -> Figure:
    """3 of 4 — the obstacle that blocks repeating step 2, and the cut it forces."""
    d = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    d2 = decision_table["Mainland_CaseCtrl_Mahalanobis"].to_numpy(dtype=float, copy=False) ** 2
    floor = decision_table["Mainland_CaseCtrl_Noise_Floor"].to_numpy(dtype=float, copy=False)
    sep = decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float, copy=False)
    n1 = decision_table[f"{case_label}_Count"].to_numpy(dtype=int, copy=False)
    n2 = decision_table[f"{control_label}_Count"].to_numpy(dtype=int, copy=False)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_int = int(rows["intermediate"]["Resolved_Rank"])
    reversals = int(rows["intermediate"]["Axis_Reversals"])
    dist_min = float(rows["intermediate"]["Value"])
    colour = _EDGE["intermediate"]

    x = _safe_norm(het); y = _safe_norm(neff); s_n = _safe_norm(sep)
    u = blend_weight * x + (1.0 - blend_weight) * s_n
    blended = np.asarray(objective_spaces["intermediate"].structure, dtype=float)
    on = weight_grid[weight_winner == k_int]
    lo, hi = (float(on.min()), float(on.max())) if on.size else (float("nan"),) * 2

    fig, ax, right, ax_tab = _stage(3, 11)
    ax_o = fig.add_subplot(right[0, 0])
    ax_g = fig.add_subplot(right[1, 0])
    ax_w = fig.add_subplot(right[2, 0])

    ax.text(0.0, 0.985, "The obstacle — a second kind of residual structure",
            fontsize=14.5, fontweight="bold", ha="left", va="center", color="#B35806")
    _says(ax, 0.955, f"Spread says how wide the retained set is. It does not say whether the two arms "
                     f"sit at different places inside it — and only that biases an association test.")
    _formula(ax, 0.878, r"$S = \dfrac{(n_1-1)C_1 + (n_2-1)C_2}{n_1 + n_2 - 2}, \qquad "
                        r"\hat{D}^2_k = \Delta\bar{x}^{\top} S^{-1} \Delta\bar{x}$", 15.0)
    _formula(ax, 0.800, r"$s_k = \hat{D}^2_k \;-\; d\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)$", 15.0)
    _says(ax, 0.742, r"Two sample means never coincide and $\hat{D}^2$ squares the gap between them, so "
                     r"sampling alone contributes $d(1/n_1 + 1/n_2)$ whatever the truth. Subtracting it "
                     r"is what makes cuts of different sizes comparable; $s_k$ may go negative, which "
                     r"means the gap is below what sampling would give.")
    _says(ax, 0.640, f"But $s_k$ reverses direction {reversals} times along the walk. There is no single "
                     f"rate to read off it, so step 2 cannot be repeated here.", colour="#B35806")

    ax.text(0.0, 0.565, "So combine the two axes", fontsize=14.5, fontweight="bold",
            ha="left", va="center", color=colour)
    _formula(ax, 0.505, r"$u_k(w) = w\,x_k + (1-w)\,\tilde{s}_k, \qquad "
                        r"\tilde{H}_k = \mathrm{minmax}(u_k(w))$", 15.0)
    _formula(ax, 0.432, r"$k^{*}(w) = \arg\min_k \sqrt{\tilde{H}_k^{\,2} + (1 - y_k)^2}$", 15.0)
    _says(ax, 0.376, r"$x_k, y_k, \tilde{s}_k$ are $H_k$, $N_k$ and $s_k$ each min-max scaled to "
                     r"$[0,1]$. The blend is scaled a second time so the distance is measured on a "
                     r"full unit axis, then the cut nearest the unattainable corner $(0,1)$ is taken.")

    ax.text(0.0, 0.286, r"Why $w = \frac{1}{2}$", fontsize=14.5, fontweight="bold",
            ha="left", va="center", color=colour)
    _formula(ax, 0.236, r"$w \geq \frac{1}{2} \;\Longleftrightarrow\; w \geq 1 - w$", 15.0)
    _says(ax, 0.190, f"Below ½ the term built from {case_label}/{control_label} labels would outweigh "
                     f"the one built from genotypes, and minimising that optimises exactly what the "
                     f"association test measures. ½ is the boundary, so the distance carries the most "
                     f"weight it legitimately can — which is what this cut is for.")

    _symbol_table(ax_tab, (
        (r"$C_1,\ C_2$", f"within-group covariance of {case_label}, {control_label}", f"{d}×{d} each"),
        (r"$S$", "the two pooled", rf"$\nu = n_1+n_2-2$"),
        (r"$\Delta\bar{x}$", "difference of the two centroids", f"{d}-vector"),
        (r"$\hat{D}^2_k$", "squared Mahalanobis distance", f"{d2.min():.5f} → {d2.max():.5f}"),
        (r"$d(1/n_1{+}1/n_2)$", "what sampling alone contributes", f"{floor.min():.5f} → {floor.max():.5f}"),
        (r"$s_k$", "de-biased distance", f"{sep.min():+.5f} → {sep.max():+.5f}"),
        (r"$x_k,\ y_k,\ \tilde{s}_k$", r"$H_k$, $N_k$, $s_k$ each min-max scaled", r"$[0,1]$"),
        (r"$u_k(w)$", "the blend before rescaling", f"{u.min():.4f} → {u.max():.4f}"),
        (r"$\tilde{H}_k$", "and after", f"{blended.min():.2f} → {blended.max():.2f}"),
        (r"$w$", "weight on spread", f"{blend_weight:g}"),
        (r"$k^{*}$", "nearest the ideal corner", f"{k_int},  at distance {dist_min:.4f}"),
    ), top=0.90, gap=0.098)

    # ── evidence ─────────────────────────────────────────────────────
    ax_o.axhline(0.0, color=_DIM, linewidth=1.0, linestyle=":")
    ax_o.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=4.0, linewidth=1.5,
              markerfacecolor="white", markeredgewidth=1.0, label=r"$H_k$ — never reverses")
    ax_o.plot(rank, _safe_norm(sep), "-^", color="#B35806", markersize=4.8, linewidth=1.8,
              markerfacecolor="white", markeredgewidth=1.0,
              label=rf"$s_k$ — reverses {reversals}×")
    ax_o.set_xticks(rank[::2]); ax_o.set_ylim(-0.10, 1.34)
    ax_o.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_o.set_ylabel("min-max normalised", fontsize=10.5)
    ax_o.tick_params(labelsize=9.5)
    ax_o.legend(loc="upper center", fontsize=9.5, ncol=2, frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")
    _panel_title(ax_o, "A", "one axis has a rate to read; the other has none")

    # The operator that decides this cut, drawn. It had no panel before: the
    # obstacle and the weight both did, so the figure read as though w were the
    # criterion rather than a parameter of it.
    i_int = int(np.argmin(np.abs(rank - k_int)))
    ax_g.plot(blended, y, "-o", color=_GR, markersize=4.2, linewidth=1.1, alpha=0.75,
              markerfacecolor="white", markeredgewidth=1.0, zorder=3,
              label=f"the {rank.max()} cuts")
    ax_g.plot([0.0], [1.0], "*", color=colour, markersize=19.0, zorder=6,
              markeredgecolor="white", markeredgewidth=1.0,
              label=r"ideal corner $(0,1)$")
    ax_g.plot([0.0, blended[i_int]], [1.0, y[i_int]], "--", color=colour,
              linewidth=2.0, zorder=5)
    ax_g.annotate(rf"$\sqrt{{\tilde{{H}}^2 + (1-y)^2}} = {dist_min:.4f}$",
                  xy=(blended[i_int] / 2.0, (1.0 + y[i_int]) / 2.0),
                  xytext=(14, -20), textcoords="offset points", fontsize=10.5,
                  color=colour, fontstyle="italic", zorder=7)
    ax_g.plot([blended[i_int]], [y[i_int]], _MARK["intermediate"], color=colour,
              markersize=12.0, markeredgecolor="white", markeredgewidth=1.4, zorder=7)
    ax_g.annotate(f"$k$ = {k_int}", xy=(blended[i_int], y[i_int]), xytext=(12, -12),
                  textcoords="offset points", fontsize=11.5, fontweight="bold",
                  color=colour, zorder=8)
    ax_g.set_xlabel(rf"$\tilde{{H}}_k$  at  $w = {blend_weight:g}$", fontsize=10.5, labelpad=1)
    ax_g.set_ylabel(r"$y_k$   ($N_k$ normalised)", fontsize=10.5)
    ax_g.tick_params(labelsize=9.5)
    ax_g.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95,
                edgecolor="#CFCFCF")
    _panel_title(ax_g, "B", "the rule — nearest the unattainable corner")

    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color="#F4C7C3", alpha=0.55, linewidth=0)
    ax_w.text(safe_weight_floor / 2.0, 0.55, "not usable —\nlabels outweigh spread",
              transform=ax_w.get_xaxis_transform(), fontsize=9.5, color="#9B2226",
              ha="center", va="center", fontstyle="italic")
    ax_w.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK, linewidth=2.2)
    if on.size:
        ax_w.plot([lo, hi], [k_int, k_int], color=colour, linewidth=6.0,
                  solid_capstyle="butt", alpha=0.85)
    ax_w.axvline(blend_weight, color=colour, linewidth=1.5, linestyle="-.")
    ax_w.plot([blend_weight], [k_int], _MARK["intermediate"], color=colour, markersize=10.0,
              markeredgecolor="white", markeredgewidth=1.3)
    ax_w.set_ylim(int(won.min()) - 1, int(won.max()) + 1)
    ax_w.set_xlim(0.0, 1.0)
    ax_w.set_yticks(sorted(set(int(v) for v in np.unique(won))))
    ax_w.set_xlabel(r"$w$ — weight on spread;  $1-w$ on distance", fontsize=10.5, labelpad=1)
    ax_w.set_ylabel(r"winning cut  $k^{*}(w)$", fontsize=10.5)
    ax_w.tick_params(labelsize=9.5)
    _panel_title(ax_w, "C", rf"and it holds — $k^{{*}}$ = {k_int} across $w \in [{lo:.2f},\ {hi:.2f}]$")
    _verdict(ax_w, f"intermediate  =  {k_int}", colour)

    for a in (ax_o, ax_g, ax_w):
        a.grid(True, alpha=0.30, linewidth=0.7); a.set_axisbelow(True)

    _figure_title(fig, "The Second Cut", "what blocks repeating step 2, and what replaces it")
    fig.subplots_adjust(left=0.030, right=0.985, top=0.905, bottom=0.075)
    _series_footer(fig, "02_intermediate")
    return fig


def plot_cohorts(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rank_table: pd.DataFrame,
    rgv_column: str,
    rank_cuts: "Mapping[str, int | None]",
    mode: str,
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """4 of 4 — the three sets steps 1-3 deliver, and what each one costs."""
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    rs = rank_table.sort_values("Rank")
    comps = {n: [int(c) for c in rs.loc[rs["Rank"] <= k_of[n], "Cluster"]] for n in k_of}
    order = [n for n in CUT_ORDER if n in k_of]

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        return float(to_numeric_array(r[col])[0]) if col in r.columns and not r.empty else float("nan")

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.18], hspace=0.24)
    ax_loc = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[1, 0])
    _note_axis(ax_t)

    ax_loc.plot(het, neff, "-o", color=_GR, markersize=4.4, linewidth=1.4,
                markerfacecolor="white", markeredgewidth=1.0, zorder=3,
                label=f"the {rank.max()} cumulative cuts")
    for name in order:
        i = int(np.argmin(np.abs(rank - k_of[name])))
        ax_loc.plot([het[i]], [neff[i]], _MARK[name], color=_EDGE[name], markersize=15.0,
                    markeredgecolor="white", markeredgewidth=1.8, zorder=6)
        ax_loc.annotate(f"{name}\n$k$ = {k_of[name]}", xy=(het[i], neff[i]),
                        xytext={"narrow": (-52, -26), "intermediate": (10, 26),
                                "full": (-12, -34)}[name],
                        textcoords="offset points", fontsize=11.5, fontweight="bold",
                        color=_EDGE[name], ha="center", va="center", zorder=7)
    ax_loc.set_xlabel(r"residual spread  $H_k$   $\rightarrow$ less homogeneous", labelpad=6)
    ax_loc.set_ylabel(r"$N_k$   $\rightarrow$ more power")
    ax_loc.legend(loc="lower right", fontsize=11.0, frameon=True, framealpha=0.95,
                  edgecolor="#CFCFCF")
    ax_loc.grid(True, alpha=0.30, linewidth=0.7); ax_loc.set_axisbelow(True)
    ax_loc.margins(x=0.14, y=0.12)
    _panel_title(ax_loc, "A", "where the three sit on the trade-off")

    defs = {
        "narrow": (r"$\arg\max_k E_k$", "step 2 — where spread stops repaying in power"),
        "intermediate": (r"$\arg\min_k \sqrt{\tilde{H}_k(\frac{1}{2})^2 + (1-y_k)^2}$",
                         "step 3 — distance carrying the most weight it may"),
        "full": ("every major-cluster component", "no rule; the population steps 1–3 work inside"),
    }
    _panel_title(ax_t, "B", "what each one is, and what it delivers")
    cols = ("", "definition", "$k$", "components added", case_label, control_label,
            "n", r"$N_k$", r"$H_k$")
    xs = (0.004, 0.105, 0.400, 0.455, 0.605, 0.688, 0.772, 0.850, 0.925)
    for xx, c in zip(xs, cols):
        ax_t.text(xx, 0.845, c, fontsize=11.0, ha="left", va="center", color=_DIM, fontstyle="italic")
    ax_t.plot([0.0, 1.0], [0.785, 0.785], color="#CFCFCF", linewidth=1.0)
    for i, name in enumerate(order):
        yy = 0.645 - i * 0.235
        ax_t.add_patch(FancyBboxPatch((0.0, yy - 0.105), 1.0, 0.212, boxstyle="square,pad=0",
                                      facecolor=_TINT[name], edgecolor="none", alpha=0.60, zorder=1))
        extra = ("" if i == 0 else "+ ") + ", ".join(
            str(c) for c in (comps[name] if i == 0
                             else [c for c in comps[name] if c not in set(comps[order[i - 1]])]))
        vals = (name, defs[name][0], str(k_of[name]), extra,
                f"{val(name, f'{case_label}_Count'):,.0f}",
                f"{val(name, f'{control_label}_Count'):,.0f}",
                f"{val(name, 'Total_Count'):,.0f}",
                f"{val(name, 'GWAS_Neff'):,.0f}",
                f"{val(name, rgv_column):.5f}")
        for j, (xx, t) in enumerate(zip(xs, vals)):
            ax_t.text(xx, yy + 0.042, t, fontsize=13.5 if j == 0 else 11.5, ha="left",
                      va="center", zorder=3, color=_EDGE[name] if j == 0 else _BK,
                      fontweight="bold" if j in (0, 2) else "normal")
        ax_t.text(xs[1], yy - 0.062, defs[name][1], fontsize=10.0, ha="left", va="center",
                  zorder=3, color=_DIM, fontstyle="italic")
    ax_t.text(0.0, -0.015,
              r"narrow $\subset$ intermediate $\subset$ full — nested by construction.   "
              "Neither is the better list: a narrower set buys homogeneity with effective sample "
              f"size, and a broader one the reverse.   Cuts resolved in mode: {mode}.",
              fontsize=10.5, ha="left", va="top", color=_DIM, fontstyle="italic")

    _figure_title(fig, "The Three Cohorts", "what steps 1–3 deliver")
    fig.subplots_adjust(left=0.055, right=0.988, top=0.905, bottom=0.078)
    _series_footer(fig, "03_cohorts")
    return fig
