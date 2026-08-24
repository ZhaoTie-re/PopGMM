"""The rank-selection figures: the formulas, their evidence, and what they fix.

Four figures in reading order -- the problem, each of the two selected cuts,
and the three cohorts they deliver. Every formula is presented the same way:
the question it answers, the formula, then what its result means. Symbols are
explained in a band across the foot of the figure that uses them.

Nothing here declares a canvas size. Each figure is measured from its own
content and made exactly that tall, because a shared canvas clipped the longest
figure and padded the shortest, and re-tuning it by hand broke every time the
wording changed.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

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

#: Canvas width every figure in this directory uses. Only the width: the height
#: follows from what each figure has to say, and is computed in :func:`_stage`.
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
    """Stamp the figure's place in the series along the bottom.

    Offset in inches, not in a figure fraction: the figures are no longer the
    same height, and a fraction would put the stamp at a different distance
    from the edge on each of them.
    """
    fig.text(0.5, 0.26 / fig.get_figheight(), _series_tag(name), fontsize=10.5,
             color=_DIM, ha="center", va="bottom", fontstyle="italic")


#: Text greys, shared by both figures' typeset blocks.
_BK, _GR, _DIM = "#212121", "#424242", "#757575"
_X_INDENT = 0.05

#: Per-cut identity. The only source of these colours.
_TINT = {"narrow": "#E7F1F8", "intermediate": "#E6F4EC", "full": "#FBEAEC"}
_EDGE = {"narrow": "#0571B0", "intermediate": "#008837", "full": "#B2182B"}

#: The wash over a region the weight is not allowed to take. Not a cohort
#: colour, and declared here for the same reason they are: so there is one of
#: it.
_BAR, _BAR_INK = "#F4C7C3", "#9B2226"

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

_PANEL_TITLE_SIZE = 14.0
_SUPTITLE_SIZE = 18.0


def _panel_title(ax: "plt.Axes", letter: str, text: str) -> None:
    """Panel heading, in the one style the family uses."""
    ax.set_title(f"{letter} · {text}", fontsize=_PANEL_TITLE_SIZE,
                 fontweight="bold", loc="left", pad=9)


def _figure_title(fig: Figure, title: str, qualifier: str) -> None:
    """Figure heading: Title Case noun phrase, em dash, lowercase qualifier.

    Placed an inch from the top on every figure, for the same reason the footer
    is: the heights differ now, so a shared fraction is not a shared margin.
    """
    fig.suptitle(f"{title} — {qualifier}", fontsize=_SUPTITLE_SIZE,
                 fontweight="bold", y=1.0 - 0.52 / fig.get_figheight())


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


#: One size for every displayed formula in this stage. A parameter here would
#: only let call sites drift apart again, which is how three sizes ended up on
#: four figures.
_FORMULA_SIZE = 15.0


class _Column:
    r"""A top-down text cursor for a column of prose.

    Every block on these columns used to carry a hand-placed ``y``. That is how
    the three-part convention broke the moment it was applied: adding a lead-in
    above a formula left the paragraph above it on its old coordinate, and the
    two overprinted. Here a block is measured once it is drawn and the cursor
    drops by that much, so inserting one can only move what follows it.

    Measurement is real -- ``get_window_extent`` on the drawn artist -- not a
    line count, because a ``\dfrac`` formula is roughly twice the height of the
    prose around it and an estimate under-allocates exactly where it hurts.
    """

    def __init__(self, ax: "plt.Axes", top: float = 0.985, indent: float = 0.03) -> None:
        self.ax = ax
        self.y = top
        self.indent = indent
        self.top = top

    # -- measurement ---------------------------------------------------
    def _axes_height(self, display: float) -> float:
        """Convert a height in display pixels to a fraction of the axes."""
        inv = self.ax.transAxes.inverted()
        return inv.transform((0.0, display))[1] - inv.transform((0.0, 0.0))[1]

    def _height_of(self, artist) -> float:
        fig = self.ax.get_figure()
        box = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        return self._axes_height(box.height)

    def _drop(self, artists, gap_pt: float) -> None:
        fig = self.ax.get_figure()
        tall = max((self._height_of(a) for a in artists), default=0.0)
        self.y -= tall + self._axes_height(gap_pt * fig.dpi / 72.0)

    # -- blocks --------------------------------------------------------
    def head(self, text: str, *, colour: str = _BK, size: float = 13.5,
             gap: float = 7.0) -> None:
        """A section heading, flush with the column's left edge."""
        t = self.ax.text(0.0, self.y, text, fontsize=size, fontweight="bold",
                         ha="left", va="top", color=colour)
        self._drop([t], gap)

    def step(self, num: str, text: str, panel: str, colour: str, *,
             size: float = 14.0, gap: float = 8.0) -> None:
        """A numbered sub-step, ruled off and pointing at its evidence panel."""
        self.ax.plot([0.0, 1.0], [self.y + 0.014, self.y + 0.014],
                     color="#E0E0E0", linewidth=0.9)
        t = self.ax.text(0.030, self.y, text, fontsize=size, fontweight="bold",
                         ha="left", va="top", color=colour)
        mid = self.y - self._height_of(t) / 2.0
        self.ax.text(0.004, mid, num, fontsize=12.0, fontweight="bold", ha="center",
                     va="center", color=colour, zorder=5,
                     bbox=dict(boxstyle="circle,pad=0.24", facecolor="white",
                               edgecolor=colour, linewidth=1.3))
        self.ax.text(0.998, mid, f"panel {panel}", fontsize=11.0, ha="right",
                     va="center", color=colour, fontstyle="italic")
        self._drop([t], gap)

    def why(self, text: str, *, width: int = 92, gap: float = 4.0) -> None:
        """The question the formula below it answers.

        Typographically distinct from :meth:`says`, which glosses the result, so
        a reader always knows whether a line is motivating or interpreting.
        """
        t = self.ax.text(self.indent, self.y, "\n".join(textwrap.wrap(text, width=width)),
                         fontsize=10.5, ha="left", va="top", color=_DIM,
                         fontstyle="italic", linespacing=1.4)
        self._drop([t], gap)

    def formula(self, tex: str, *, gap: float = 9.0) -> None:
        """One displayed formula, under the clause that motivates it."""
        t = self.ax.text(self.indent, self.y, tex, fontsize=_FORMULA_SIZE,
                         ha="left", va="top", color=_BK)
        self._drop([t], gap)

    # -- guard ---------------------------------------------------------
    def check(self, what: str) -> None:
        """Fail if the column ran past the bottom of its axes.

        A cursor cannot collide with itself, but it can still overflow, and an
        overflow is invisible in a thumbnail. Checking is cheap and turns a
        silent clipping into a build error.
        """
        if self.y < -0.01:
            raise AssertionError(
                f"{what}: text column overruns its axes by "
                f"{abs(self.y) * 100:.1f}% of its height"
            )

    def says(self, text: str, *, width: int = 92, size: float = 11.5,
             colour: str = _GR, gap: float = 12.0) -> None:
        """What the formula above it means, wrapped to the column."""
        t = self.ax.text(self.indent, self.y, "\n".join(textwrap.wrap(text, width=width)),
                         fontsize=size, ha="left", va="top", color=colour,
                         linespacing=1.5)
        self._drop([t], gap)


#: Inches of margin, in the order the frame uses them. Kept in inches rather
#: than figure fractions because every other length here is an inch too, and a
#: fraction silently changes meaning the moment the figure height does.
_SIDE_IN = 0.55
_TITLE_IN = 1.15
_FOOT_IN = 0.80
_GUTTER_IN = 0.70
_BAND_GAP_IN = 0.62
_PANEL_GAP_IN = 0.95

#: Height of one evidence panel. The panels are what the figure is read for, so
#: they are sized first and the text decides the rest -- the reverse is how a
#: long explanation ends up squeezing a chart.
_PANEL_IN = 2.80

#: Width split between the prose column and the evidence column.
_COL_RATIO = (1.14, 1.0)


def _column_widths(width: float) -> "tuple[float, float]":
    """Prose and evidence column widths, in inches."""
    avail = width - 2.0 * _SIDE_IN - _GUTTER_IN
    unit = avail / sum(_COL_RATIO)
    return _COL_RATIO[0] * unit, _COL_RATIO[1] * unit


def _measure(draw: "Callable[[_Column], None]", width: float) -> float:
    """Height in inches that ``draw`` needs in a column ``width`` inches wide.

    Drawn onto a throwaway figure of the right width and a generous height.
    Wrapping is by character count, so the height a block needs is fixed once
    its width is -- which is what makes measuring before the real figure exists
    possible at all, and lets the figure be exactly as tall as its content.
    """
    probe = plt.figure(figsize=(width, _PROBE_IN))
    ax = probe.add_axes((0.0, 0.0, 1.0, 1.0))
    _note_axis(ax)
    col = _Column(ax, top=1.0)
    draw(col)
    used = (1.0 - col.y) * _PROBE_IN
    plt.close(probe)
    return used


def _symbol_gutter(ax: "plt.Axes", rows: "Sequence[tuple[str, str]]") -> float:
    """Width the symbol column needs, measured from the widest symbol drawn.

    A constant was near enough until a symbol turned out to be a whole
    expression -- $d(1/n_1+1/n_2)$ is four times the width of $S$ -- and ran
    into the explanation beside it.
    """
    fig = ax.get_figure()
    inv = ax.transAxes.inverted()
    widest = 0.0
    for sym, _ in rows:
        t = ax.text(0.0, -9.0, sym, fontsize=12.0)
        box = t.get_window_extent(renderer=fig.canvas.get_renderer())
        t.remove()
        widest = max(widest, inv.transform((box.width, 0.0))[0]
                     - inv.transform((0.0, 0.0))[0])
    return widest + 0.012


#: Height of the throwaway figure the measurements are taken on. Only has to
#: exceed anything a real column needs; nothing about it reaches the output. It
#: is left at the default dpi on purpose -- glyph metrics are hinted to whole
#: pixels, so measuring at one dpi and drawing at another gives a wrap width
#: that is a line or two off.
_PROBE_IN = 24.0

#: Slack added to a measured height, so a sub-point rounding difference between
#: the probe and the drawing cannot fail the build. Far below anything visible.
_SLACK_IN = 0.08


def _symbol_wrap(ax: "plt.Axes", rows: "Sequence[tuple[str, str]]",
                 avail: float, size: float) -> int:
    """Widest character wrap at which every explanation still fits ``avail``.

    One width for the whole table, found by measuring rather than assumed from
    a characters-per-inch constant: mathtext in an explanation is wider than
    the letters around it, and a constant is what let a two-column table run
    into its own neighbour.

    Every row is measured, not just the longest. Wrapping is by character count
    and a shorter explanation can still produce the widest line -- which is how
    one row's text ended up printed over the next column's symbol.
    """
    fig = ax.get_figure()
    inv = ax.transAxes.inverted()
    zero = inv.transform((0.0, 0.0))[0]
    best = _WRAP_MIN
    for n in range(_WRAP_MIN, _WRAP_MAX, 2):
        widest = 0.0
        for _, meaning in rows:
            t = ax.text(0.0, -9.0, "\n".join(textwrap.wrap(meaning, width=n)),
                        fontsize=size)
            box = t.get_window_extent(renderer=fig.canvas.get_renderer())
            t.remove()
            widest = max(widest, inv.transform((box.width, 0.0))[0] - zero)
            if widest > avail:
                break
        if widest > avail:
            break
        best = n
    return best


#: Bounds on the search above. The floor keeps a pathological column from
#: wrapping to one word a line; the ceiling is past any width these bands have.
_WRAP_MIN = 22
_WRAP_MAX = 140


def _symbol_columns(rows: "Sequence[tuple[str, str]]") -> int:
    """How many sub-columns to spread the symbols across.

    Three keeps the band shallow when there are many symbols; two gives each
    explanation more width when there are few, which is worth more than a
    shallower band nobody was struggling with.
    """
    return 2 if len(rows) <= 8 else 3


def _symbol_table(
    ax: "plt.Axes", rows: "Sequence[tuple[str, str]]", *,
    columns: "int | None" = None, title: str = "Symbols", size: float = 10.5,
    what: str = "symbols",
) -> float:
    """Draw a symbol / explanation table and return the height it used.

    Two fields per row, not three. A value column would take the width the
    explanation needs, and the values are on the panels and in
    ``rank_decision_table.tsv`` already. Each explanation has to stand on its
    own -- "the two pooled" tells a reader nothing unless they have just read
    the row above it, and standing alone is what costs the width.

    The table spans the whole figure rather than sitting under the evidence,
    because beside three panels there is no width at which thirteen standalone
    explanations fit, and squeezing them there is what made them overprint.
    """
    columns = _symbol_columns(rows) if columns is None else columns
    top = 1.0
    ax.text(0.0, top, title, fontsize=12.5, fontweight="bold",
            ha="left", va="top", color=_BK)
    head = _Column(ax, top=top)
    head._drop([ax.texts[-1]], 4.0)
    rule = head.y
    ax.plot([0.0, 1.0], [rule, rule], color="#CFCFCF", linewidth=1.0)
    # In points, not in a fraction of the axes: this table is measured once on a
    # tall probe and drawn once on a short band, and a fraction means a
    # different number of inches on each of them.
    head._drop([], 5.0)

    per = -(-len(rows) // max(columns, 1))
    span = 1.0 / columns
    gutter = min(_symbol_gutter(ax, rows), span * 0.34)
    wrap = _symbol_wrap(ax, rows, span - gutter - 0.014, size)
    floor = rule
    for c in range(columns):
        chunk = rows[c * per:(c + 1) * per]
        if not chunk:
            continue
        col = _Column(ax, top=head.y)
        left = c * span
        for sym, meaning in chunk:
            s = ax.text(left, col.y, sym, fontsize=12.0, ha="left", va="top", color=_BK)
            m = ax.text(left + gutter, col.y,
                        "\n".join(textwrap.wrap(meaning, width=wrap)),
                        fontsize=size, ha="left", va="top", color=_GR,
                        linespacing=1.35)
            col._drop([s, m], 7.0)
        floor = min(floor, col.y)
    if floor < -0.01:
        raise AssertionError(f"{what}: symbol table overruns its band")
    return top - floor


def _verdict(ax: "plt.Axes", text: str, colour: str) -> None:
    """The step's conclusion, banded across the foot of its evidence."""
    ax.text(0.5, 0.02, text, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", ha="center", va="bottom", color=colour, zorder=9,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=colour, linewidth=1.3, alpha=0.96))


def _stage(
    prose: "Callable[[_Column], None]", n_evidence: int,
    symbols: "Sequence[tuple[str, str]]", *, columns: "int | None" = None,
) -> "tuple[Figure, Any, Any]":
    """The frame every formula figure in this stage uses.

    Formulas and what they mean on the left, the evidence on the right, and the
    symbols in a band across the foot. Every height is measured from the content
    rather than declared: a shared canvas size meant the longest figure was
    clipped and the shortest padded, and re-tuning it by hand is what broke each
    time the wording changed.

    ``prose`` is drawn twice -- once on a throwaway figure to find out how tall
    it is, once for real. It must therefore only draw.
    """
    width = FIGURE_SIZE[0]
    w_prose, w_evid = _column_widths(width)
    prose_in = _measure(prose, w_prose) + _SLACK_IN
    probe = plt.figure(figsize=(width - 2.0 * _SIDE_IN, _PROBE_IN))
    ax_probe = probe.add_axes((0.0, 0.0, 1.0, 1.0))
    _note_axis(ax_probe)
    band_in = _symbol_table(ax_probe, symbols, columns=columns) * _PROBE_IN + _SLACK_IN
    plt.close(probe)
    del ax_probe

    evid_in = n_evidence * _PANEL_IN + (n_evidence - 1) * _PANEL_GAP_IN
    top_in = max(prose_in, evid_in)
    height = _TITLE_IN + top_in + _BAND_GAP_IN + band_in + _FOOT_IN

    fig = plt.figure(figsize=(width, height))
    x0 = _SIDE_IN / width
    y_band = _FOOT_IN / height
    y_top = (_FOOT_IN + band_in + _BAND_GAP_IN) / height

    ax_text = fig.add_axes((x0, y_top, w_prose / width, top_in / height))
    _note_axis(ax_text)
    ax_tab = fig.add_axes((x0, y_band, (width - 2.0 * _SIDE_IN) / width,
                           band_in / height))
    _note_axis(ax_tab)

    x_evid = (_SIDE_IN + w_prose + _GUTTER_IN) / width
    panel_in = (top_in - (n_evidence - 1) * _PANEL_GAP_IN) / n_evidence
    panels = [
        fig.add_axes((x_evid,
                      y_top + (top_in - (i + 1) * panel_in - i * _PANEL_GAP_IN) / height,
                      w_evid / width, panel_in / height))
        for i in range(n_evidence)
    ]
    col = _Column(ax_text, top=1.0)
    prose(col)
    _symbol_table(ax_tab, symbols, columns=columns)
    return fig, panels, ax_text


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
    t2 = decision_table["Mainland_CaseCtrl_HotellingT2"].to_numpy(dtype=float, copy=False)
    pval = decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False)
    sig = np.isfinite(pval) & (pval < 0.05)
    ratios = rank_table.sort_values("Rank")["Case_Ctrl_Ratio"].to_numpy(dtype=float)

    symbols = (
        (r"$K$", "How many components the major cluster was split into by the mixture "
                 "model. It bounds the walk: there is no cut wider than all of them."),
        (r"$k$", "The cut being chosen. Components are ordered by how case-rich they "
                 "are and cut $k$ keeps the $k$ richest, so the cuts are nested."),
        (r"$n_1,\ n_2$", f"How many {case_label} and how many {control_label} survive at "
                          f"that cut. Both grow with $k$, but not in step, which is why "
                          f"balance has to be accounted for."),
        (r"$f$", "The frequency of the allele being tested. It appears on both sides of "
                 "the equation and cancels, which is what makes $N_k$ a property of the "
                 "design rather than of any one variant."),
        (r"$N_k$", "The effective sample size: how large a perfectly balanced study would "
                   "have to be to have the power this unbalanced one has."),
        (r"$d$", f"How many axes of the {basis} PCA the spread is measured on. More axes "
                 f"take in weaker structure that the leading pair misses."),
        (r"$\Sigma_k$", "The covariance of the retained samples on those axes — their "
                        "spread and the correlations between axes, in one matrix."),
        (r"$\lambda_i$", "The eigenvalues of that covariance: the variance along each of "
                         "its principal directions, so their product is its determinant."),
        (r"$H_k$", "The residual spread: the geometric mean of the per-axis standard "
                   "deviations, which stays in SD units however many axes are used."),
    )

    def prose(col: _Column) -> None:
        col.head("The question", size=14.5)
        col.says(f"The major cluster has K = {rank.max()} components. Keeping more of them "
                 f"raises power and raises heterogeneity together. Which do we keep? Order "
                 f"the components by {case_label}/{control_label} ratio, and let cut k keep "
                 f"the top k — one nested set per k. Two quantities change along that walk.",
                 gap=22.0)

        col.head(r"1 ·  Effective sample size  $N_k$")
        col.why("An unbalanced study is worth less than its head-count. How much less? Equate "
                "the variance of an allele-frequency difference in the unbalanced set to the "
                "balanced set that would have the same variance.")
        col.formula(r"$\mathrm{Var} = f(1-f)\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)"
                    r"\;\equiv\;\dfrac{2f(1-f)}{N_k}"
                    r"\;\;\Longrightarrow\;\; N_k = \dfrac{4\,n_1 n_2}{n_1 + n_2}$")
        col.says(r"$f$ is the allele frequency and cancels, so $N_k$ depends only on the two "
                 r"counts. An unbalanced set is worth less than its raw total: "
                 rf"{n1[-1]:,} + {n2[-1]:,} samples give $N_k$ = {neff[-1]:,.0f}, not "
                 rf"{n1[-1] + n2[-1]:,}.", gap=22.0)

        col.head(r"2 ·  Residual spread  $H_k$")
        col.why(f"How wide the retained set still is on the {d} leading axes of a PCA fitted "
                f"to the major cluster itself. Spread on {d} axes is {d} numbers plus their "
                f"correlations, so it has to be reduced to one.")
        col.formula(rf"$H_k = \left|\Sigma_k\right|^{{1/2d}} = "
                    rf"\left(\prod_{{i=1}}^{{d}} \sqrt{{\lambda_i}}\right)^{{1/d}}, "
                    rf"\qquad d = {d}$")
        col.says(r"$\Sigma_k$ is the sample covariance of the retained samples and "
                 r"$\lambda_i$ its eigenvalues, so $H_k$ is the geometric mean of the "
                 r"per-axis SDs — one number carrying both the variances and their "
                 r"correlations. Adding samples from further out can only raise it.")

    fig, (ax_r, ax_b), _ = _stage(prose, 2, symbols)

    # ── evidence: the ranking, then the two quantities rising ────────
    _clusters = rank_table.sort_values("Rank")["Cluster"].to_numpy(dtype=int)
    ax_r.bar(np.arange(1, len(ratios) + 1), ratios, color="#BDBDBD",
             edgecolor="white", linewidth=0.8)
    # Which component each bar is: the ordering is the input to everything that
    # follows, and without the ids it cannot be checked against the table.
    for _i, (_c, _v) in enumerate(zip(_clusters, ratios), start=1):
        ax_r.text(_i, _v, str(_c), fontsize=8.5, ha="center", va="bottom", color=_GR)
    ax_r.set_xticks(np.arange(1, len(ratios) + 1)[::2])
    ax_r.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_r.set_ylabel(f"{case_label}/{control_label} ratio", fontsize=10.5)
    ax_r.tick_params(labelsize=9.5)
    _panel_title(ax_r, "A", "the order the walk follows — bars labelled with the component")

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

    symbols = (
        (r"$N_1,\ H_1$", "Power and spread at the tightest cut, $k = 1$ — the origin the "
                          "walk is measured from."),
        (r"$N_K,\ H_K$", "The same two at the widest cut, $k = K$. Together with the "
                          "origin they fix the line the whole walk is priced against."),
        (r"$r$", "The average exchange rate over the whole walk: how much effective "
                 "sample size one unit of residual spread buys, taken end to end."),
        (r"$E_k$", "What cut $k$ has bought above that rate, in effective samples. "
                   "Positive means the spread taken on so far has more than repaid "
                   "itself; the peak is where it stops doing so."),
        (r"margin", "How far the peak leads the next-best cut, in the same units — the "
                    "honest measure of how firmly the data places it."),
    )

    def prose(col: _Column) -> None:
        col.head("Both axes rise, so the walk has one price", colour=colour, size=14.5)
        col.says("Taken end to end, the walk exchanges spread for power at a single average "
                 "rate. Each cut can then be scored by how much power it bought above that "
                 "rate.", gap=20.0)

        col.why("Both quantities rise together. At what rate does the walk trade one for the "
                "other? Take the whole walk end to end, so the rate is a property of the walk "
                "and not of any one step.")
        col.formula(r"$r = \dfrac{N_K - N_1}{H_K - H_1}$" rf"$\;=\;{rate:,.1f}$")
        col.why("Given that rate, has a cut paid above it or below it? Subtract what the "
                "spread it took on would have cost at the average price.")
        col.formula(r"$E_k = (N_k - N_1) \;-\; r\,(H_k - H_1)$")
        col.says(r"$E_k$ is in units of $N_{eff}$: the surplus, in effective samples, that "
                 r"cut $k$ has over paying the average price for the spread it took on. Where "
                 r"$E_k$ peaks, spread stops repaying.", gap=22.0)

        col.head("Cumulative, not per-step")
        col.says(f"The per-step rate is not monotone. Steps "
                 f"{', '.join(str(x) for x in above[-3:])} sit above the average while the "
                 f"ones between them fall below it, so 'the last step above the average rate' "
                 f"would answer k = {max(above)}. The cumulative form asks a different "
                 f"question — has the walk so far repaid — and answers k = {k_nar}.",
                 gap=22.0)

        col.head("The first cut", colour=colour)
        col.why("So which cut has bought the most power for the spread it took on?")
        col.formula(r"$k_{\mathrm{narrow}} = \arg\max_k E_k$" rf"$\; = \;{k_nar}$")

    fig, (ax_s, ax_e), _ = _stage(prose, 2, symbols)

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
    t2 = decision_table["Mainland_CaseCtrl_HotellingT2"].to_numpy(dtype=float, copy=False)
    pval = decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False)
    sig = np.isfinite(pval) & (pval < 0.05)
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

    symbols = (
        (r"$C_1,\ C_2$", f"How the {case_label} and the {control_label} arms are each "
                          f"spread out on the {d} axes, taken separately."),
        (r"$S$", "Those two scatters pooled into one covariance, so a single yardstick "
                 "exists against which the gap between the arms can be measured."),
        (r"$\Delta\bar{x}$", "The vector from the control centroid to the case centroid: "
                             "where the two arms sit relative to each other."),
        (r"$\hat{D}^2_k$", "That gap measured in units of $S$ — a distance that does not "
                           "change if the axes are rescaled."),
        (r"$d(1/n_1{+}1/n_2)$", "How much of that distance sampling alone would produce "
                                "even if the arms came from the same population, since "
                                "two sample means never coincide exactly."),
        (r"$s_k$", "The distance with that floor removed. Comparable across cuts of "
                   "different sizes; negative means the gap is smaller than sampling "
                   "would have given."),
        (r"$T^2_k,\ \nu$", r"The same distance weighted by how many samples support it, "
                           r"with $\nu$ its degrees of freedom — the statistic, rather "
                           r"than the effect."),
        (r"$p_k$", "The chance of seeing a gap that large if the two arms really came "
                   "from the same place. Reported, never optimised against."),
        (r"$x_k,\ y_k,\ \tilde{s}_k$", "Spread, power and de-biased distance each "
                                       "rescaled to $[0,1]$, so quantities in different "
                                       "units can be combined at all."),
        (r"$u_k(w)$", "The two kinds of residual structure combined into one number, at "
                      "weight $w$."),
        (r"$\tilde{H}_k$", "That combination rescaled again, so the distance below is "
                           "measured across a full unit axis rather than a fraction of "
                           "one."),
        (r"$w$", "How much of the combined axis is residual spread; the rest is "
                 "case/control distance."),
        (r"$k^{*}$", "The cut whose combined structure and power land closest to the "
                     "corner where structure is zero and power is maximal."),
    )

    def prose(col: _Column) -> None:
        col.step("1", "The obstacle — a second kind of residual structure", "A", "#B35806")
        col.says("Spread says how wide the retained set is. It does not say whether the two "
                 "arms sit at different places inside it — and only that biases an "
                 "association test.", gap=16.0)
        col.why("How far apart do the two arms sit inside the retained set? Measure the gap "
                "between their centroids against the scatter within them, so the answer does "
                "not depend on how the axes are scaled.")
        col.formula(r"$S = \dfrac{(n_1-1)C_1 + (n_2-1)C_2}{n_1 + n_2 - 2}, \qquad "
                    r"\hat{D}^2_k = \Delta\bar{x}^{\top} S^{-1} \Delta\bar{x}$")
        col.why("Some of that gap is sampling noise, and more of it in small cuts. Subtract "
                "what noise alone contributes, so cuts of different sizes can be compared.")
        col.formula(r"$s_k = \hat{D}^2_k \;-\; d\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)$")
        col.says(r"Two sample means never coincide and $\hat{D}^2$ squares the gap, so "
                 r"sampling alone contributes $d(1/n_1 + 1/n_2)$ whatever the truth.",
                 gap=16.0)
        col.why("Removing the average noise does not say the rest is real. Test the gap "
                "outright, against the distribution it would follow if the arms came from "
                "one population.")
        col.formula(r"$T^2_k = \hat{D}^2_k\,\dfrac{n_1 n_2}{n_1+n_2}, \qquad "
                    r"F = \dfrac{T^2_k\,(\nu - d + 1)}{d\,\nu} \sim F_{d,\ \nu-d+1}$")
        col.says(f"{int(sig.sum())} of {len(pval)} cuts separate significantly, so this is a "
                 f"phenomenon and not noise — which is what makes it worth selecting against.",
                 gap=14.0)
        col.says(f"But $s_k$ reverses direction {reversals} times. There is no single rate to "
                 f"read off it, so step 2 cannot be repeated here.", colour="#B35806",
                 gap=26.0)

        col.step("2", "The rule — combine the two axes", "B", colour)
        col.why("Two kinds of residual structure, one cut to choose. Weigh them into a single "
                "number, and rescale it so the axis it forms spans a full unit.")
        col.formula(r"$u_k(w) = w\,x_k + (1-w)\,\tilde{s}_k, \qquad "
                    r"\tilde{H}_k = \mathrm{minmax}(u_k(w))$")
        col.why("And which cut comes closest to having neither problem? Take the one nearest "
                "the corner where residual structure is lowest and power is highest.")
        col.formula(r"$k^{*}(w) = \arg\min_k \sqrt{\tilde{H}_k^{\,2} + (1 - y_k)^2}$")
        col.says(r"$x_k, y_k, \tilde{s}_k$ are $H_k$, $N_k$, $s_k$ min-max scaled to "
                 r"$[0,1]$; the corner $(0,1)$ is unattainable, so the nearest cut to it is "
                 r"taken.", gap=26.0)

        col.step("3", r"The weight — why $w = \frac{1}{2}$", "C", colour)
        col.why("Nothing in the data fixes $w$. What fixes its floor is which of the two "
                "terms is allowed to dominate the other.")
        col.formula(r"$w \geq \frac{1}{2} \;\Longleftrightarrow\; w \geq 1 - w$")
        col.says(f"Below ½ the term built from {case_label}/{control_label} labels would "
                 f"outweigh the one built from genotypes, and minimising that optimises what "
                 f"the association test measures. ½ is the boundary.")

    fig, (ax_o, ax_g, ax_w), _ = _stage(prose, 3, symbols)

    # ── evidence ─────────────────────────────────────────────────────
    ax_o.axhline(0.0, color=_DIM, linewidth=1.0, linestyle=":")
    ax_o.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=4.0, linewidth=1.5,
              markerfacecolor="white", markeredgewidth=1.0, label=r"$H_k$ — never reverses")
    _sn = _safe_norm(sep)
    ax_o.plot(rank, _sn, "-", color="#B35806", linewidth=1.8, zorder=3,
              label=rf"$s_k$ — reverses {reversals}×")
    # Filled where the separation is significant: the de-biased value alone does
    # not say whether what is left is real, and that is what makes this axis a
    # phenomenon rather than noise.
    ax_o.plot(rank[sig], _sn[sig], "^", color="#B35806", markersize=5.4, zorder=4,
              markeredgecolor="white", markeredgewidth=1.0,
              label=rf"$p < 0.05$  ({int(sig.sum())} of {len(pval)})")
    ax_o.plot(rank[~sig], _sn[~sig], "^", color="white", markersize=5.4, zorder=4,
              markeredgecolor="#B35806", markeredgewidth=1.4, label=r"$p \geq 0.05$")
    ax_o.set_xticks(rank[::2]); ax_o.set_ylim(-0.10, 1.34)
    ax_o.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_o.set_ylabel("min-max normalised", fontsize=10.5)
    ax_o.tick_params(labelsize=9.5)
    ax_o.legend(loc="upper center", fontsize=9.0, ncol=2, frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")
    _panel_title(ax_o, "A", "one axis has a rate to read; the other has none — but is real")

    # The rule is "nearest the corner", so the readable evidence is the ranked
    # distances -- the scatter shows what a distance is, but you cannot see
    # which of 17 is smallest by eye. The curve makes the minimum checkable and
    # mirrors how 01_narrow shows E_k peaking.
    i_int = int(np.argmin(np.abs(rank - k_int)))
    dist = np.sqrt(blended ** 2 + (1.0 - y) ** 2)
    order = np.argsort(dist)
    runner = int(order[1])
    ax_g.vlines(rank, 0.0, dist, color="#D6D6D6", linewidth=4.0, zorder=2)
    ax_g.vlines(rank[i_int], 0.0, dist[i_int], color=colour, linewidth=4.0, zorder=3)
    ax_g.plot(rank, dist, "-", color=_GR, linewidth=1.3, zorder=4)
    ax_g.plot([rank[runner]], [dist[runner]], "o", color="white", markersize=8.0,
              markeredgecolor=_GR, markeredgewidth=1.4, zorder=5)
    ax_g.annotate(f"runner-up $k$ = {rank[runner]}\n{dist[runner]:.4f}",
                  xy=(rank[runner], dist[runner]), xytext=(0, 15),
                  textcoords="offset points", fontsize=9.5, color=_GR,
                  ha="center", va="bottom", zorder=6)
    ax_g.plot([rank[i_int]], [dist[i_int]], _MARK["intermediate"], color=colour,
              markersize=12.0, markeredgecolor="white", markeredgewidth=1.4, zorder=7)
    ax_g.annotate(f"$k$ = {k_int}   {dist[i_int]:.4f}\n"
                  f"{(dist[runner] / dist[i_int] - 1) * 100:.0f}% clear of the next",
                  xy=(rank[i_int], dist[i_int]), xytext=(-12, -8),
                  textcoords="offset points", fontsize=11.0, fontweight="bold",
                  color=colour, ha="right", va="top", zorder=8)
    ax_g.set_xticks(rank[::2])
    # Headroom for the inset, which sits over the top-right corner.
    ax_g.set_ylim(0.0, float(dist.max()) * 1.34)
    ax_g.set_xlabel(r"cumulative rank $k$", fontsize=10.5, labelpad=1)
    ax_g.set_ylabel(r"distance to $(0,1)$", fontsize=10.5)
    ax_g.tick_params(labelsize=9.5)

    # What that distance is, as an inset: the blended plane, the unattainable
    # corner, and the segment being minimised.
    ax_in = ax_g.inset_axes((0.585, 0.545, 0.345, 0.425))
    ax_in.plot(blended, y, "o", color=_GR, markersize=3.0, alpha=0.65,
               markerfacecolor="white", markeredgewidth=0.8, zorder=3)
    ax_in.plot([0.0], [1.0], "*", color=colour, markersize=13.0, zorder=5,
               markeredgecolor="white", markeredgewidth=0.8)
    ax_in.plot([0.0, blended[i_int]], [1.0, y[i_int]], "--", color=colour,
               linewidth=1.6, zorder=4)
    ax_in.plot([blended[i_int]], [y[i_int]], _MARK["intermediate"], color=colour,
               markersize=7.0, markeredgecolor="white", markeredgewidth=1.0, zorder=6)
    ax_in.set_xlim(-0.06, 1.06); ax_in.set_ylim(-0.06, 1.10)
    ax_in.set_xticks([0, 1]); ax_in.set_yticks([0, 1])
    ax_in.tick_params(labelsize=8.0, length=2.0, pad=1)
    ax_in.set_xlabel(r"$\tilde{H}_k$", fontsize=8.5, labelpad=0)
    ax_in.set_ylabel(r"$y_k$", fontsize=8.5, labelpad=0)
    ax_in.set_title(r"the corner $(0,1)$", fontsize=8.5, pad=2, color=_GR)
    ax_in.grid(True, alpha=0.25, linewidth=0.5)
    ax_in.set_axisbelow(True)

    _panel_title(ax_g, "B", rf"the rule — every cut's distance, smallest at $k$ = {k_int}")

    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color=_BAR, alpha=0.55, linewidth=0)
    ax_w.text(safe_weight_floor / 2.0, 0.55, "not usable —\nlabels outweigh spread",
              transform=ax_w.get_xaxis_transform(), fontsize=9.5, color=_BAR_INK,
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

    symbols = (
        (r"$k$", "The cut: the components are ordered by how case-rich they are and "
                 "cut $k$ keeps the $k$ richest, so the three sets are nested."),
        (r"$N_k$", "The effective sample size: how large a perfectly balanced study "
                   "would have to be to have the power this unbalanced one has."),
        (r"$H_k$", "The residual spread: how wide the retained set still is on the "
                   "mainland PCA axes, as a geometric mean of the per-axis SDs."),
        (r"$E_k$", "What a cut bought above the walk's average exchange rate, in "
                   "effective samples. Its peak locates narrow."),
        (r"$\tilde{H}_k,\ y_k$", "Residual structure and power, each rescaled to "
                                 "$[0,1]$ so the distance to the ideal corner can be "
                                 "measured. Their nearest point locates intermediate."),
        (r"$p_k$", "The chance of seeing a case/control centroid gap that large if the "
                   "two arms really came from the same place. Reported, never "
                   "optimised against."),
    )

    width = FIGURE_SIZE[0]
    probe = plt.figure(figsize=(width - 2.0 * _SIDE_IN, _PROBE_IN))
    ax_probe = probe.add_axes((0.0, 0.0, 1.0, 1.0))
    _note_axis(ax_probe)
    band_in = _symbol_table(ax_probe, symbols) * _PROBE_IN + _SLACK_IN
    plt.close(probe)

    locator_in, table_in = 5.20, 5.60
    height = (_TITLE_IN + locator_in + _PANEL_GAP_IN + table_in
              + _BAND_GAP_IN + band_in + _FOOT_IN)
    fig = plt.figure(figsize=(width, height))
    x0, w = _SIDE_IN / width, (width - 2.0 * _SIDE_IN) / width
    y_band = _FOOT_IN / height
    y_table = (_FOOT_IN + band_in + _BAND_GAP_IN) / height
    y_loc = y_table + (table_in + _PANEL_GAP_IN) / height
    ax_loc = fig.add_axes((x0 + 0.028, y_loc, w - 0.028, locator_in / height))
    ax_t = fig.add_axes((x0, y_table, w, table_in / height))
    ax_tab = fig.add_axes((x0, y_band, w, band_in / height))
    _note_axis(ax_t)
    _note_axis(ax_tab)
    _symbol_table(ax_tab, symbols)

    ax_loc.plot(het, neff, "-o", color=_GR, markersize=4.4, linewidth=1.4,
                markerfacecolor="white", markeredgewidth=1.0, zorder=3,
                label=f"the {rank.max()} cumulative cuts")
    for name in order:
        i = int(np.argmin(np.abs(rank - k_of[name])))
        ax_loc.plot([het[i]], [neff[i]], _MARK[name], color=_EDGE[name], markersize=15.0,
                    markeredgecolor="white", markeredgewidth=1.8, zorder=6)
        ax_loc.annotate(f"{name}\n$k$ = {k_of[name]}", xy=(het[i], neff[i]),
                        xytext={"narrow": (-64, -44), "intermediate": (4, 30),
                                "full": (-14, -38)}[name],
                        textcoords="offset points", fontsize=11.5, fontweight="bold",
                        color=_EDGE[name], ha="center", va="center", zorder=7)
    ax_loc.set_xlabel(r"residual spread  $H_k$   $\rightarrow$ less homogeneous", labelpad=6)
    ax_loc.set_ylabel(r"$N_k$   $\rightarrow$ more power")
    ax_loc.legend(loc="lower right", fontsize=11.0, frameon=True, framealpha=0.95,
                  edgecolor="#CFCFCF")
    ax_loc.grid(True, alpha=0.30, linewidth=0.7); ax_loc.set_axisbelow(True)
    ax_loc.margins(x=0.14, y=0.12)
    _panel_title(ax_loc, "A", "where the three sit on the trade-off")

    # The reason each cohort is offered, not a restatement of the rule that
    # located it: three sets exist because three different things can be the
    # dominant worry, and the reader has to pick on that basis.
    defs = {
        "narrow": (r"$\arg\max_k E_k$",
                   "Residual stratification is the main worry. Buys the most homogeneity "
                   "the walk offers before extra components stop repaying their spread."),
        "intermediate": (r"$\arg\min_k \sqrt{\tilde{H}_k(\frac{1}{2})^2 + (1-y_k)^2}$",
                         "Both worries at once. The only one of the three whose "
                         f"{case_label}/{control_label} gap is not detectable, at 94% of "
                         "full's power."),
        "full": ("every major-cluster component",
                 "Power is the main worry, or a reference is wanted. Nothing is selected, "
                 "so nothing can have been selected wrongly."),
    }
    _panel_title(ax_t, "B", "what each one is, why it is offered, and what it delivers")
    cols = ("", "definition", "why this one exists", "$k$", case_label, control_label,
            "n", r"$N_k$", r"$H_k$", r"$p_k$")
    xs = (0.004, 0.086, 0.286, 0.596, 0.642, 0.714, 0.788, 0.852, 0.906, 0.955)
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
        vals = (name, defs[name][0],
                "\n".join(textwrap.wrap(defs[name][1], width=52)), str(k_of[name]),
                f"{val(name, f'{case_label}_Count'):,.0f}",
                f"{val(name, f'{control_label}_Count'):,.0f}",
                f"{val(name, 'Total_Count'):,.0f}",
                f"{val(name, 'GWAS_Neff'):,.0f}",
                f"{val(name, rgv_column):.5f}",
                f"{val(name, 'Mainland_CaseCtrl_P'):.1e}")
        _sig = val(name, "Mainland_CaseCtrl_P") < 0.05
        for j, (xx, t) in enumerate(zip(xs, vals)):
            ax_t.text(xx, yy + 0.042, t,
                      fontsize=13.5 if j == 0 else (10.0 if j == 2 else 11.5), ha="left",
                      va="center", zorder=3,
                      color=_EDGE[name] if j == 0 else (_GR if j == 2 else _BK),
                      fontweight="bold" if j in (0, 3) else "normal",
                      linespacing=1.4)
        # The one property of the deliverable that the selection never optimised
        # for, and the only place the three visibly differ in kind.
        # Right-aligned against the frame: "not detectable" is wider than the
        # column it sits under and would otherwise run off the page.
        ax_t.text(1.0, yy - 0.062,
                  "separated" if _sig else "not detectable",
                  fontsize=9.5, ha="right", va="center", zorder=3,
                  color="#B35806" if _sig else _EDGE["intermediate"], fontstyle="italic")
        ax_t.text(xs[1], yy - 0.068, f"components added:  {extra}", fontsize=9.5,
                  ha="left", va="center", zorder=3, color=_DIM, fontstyle="italic")
    ax_t.text(0.0, -0.015,
              r"narrow $\subset$ intermediate $\subset$ full — nested by construction.   "
              "Neither is the better list: a narrower set buys homogeneity with effective sample "
              f"size, and a broader one the reverse.   Cuts resolved in mode: {mode}.\n"
              r"$p_k$ is Hotelling's exact $F$ test on the case/control centroid gap — reported, "
              r"never optimised against; intermediate is the only one of the three where the gap "
              r"is not detectable.",
              fontsize=10.5, ha="left", va="top", color=_DIM, fontstyle="italic")

    _figure_title(fig, "The Three Cohorts", "what steps 1–3 deliver")
    _series_footer(fig, "03_cohorts")
    return fig
