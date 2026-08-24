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

import re
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
    fig.text(0.5, 0.26 / fig.get_figheight(), _series_tag(name),
             fontsize=_SUPP["foot"], color=_DIM, ha="center", va="bottom",
             fontstyle="italic")


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

#: Two type scales, and only two.
#:
#: ``_MAIN`` sets the band that gets presented -- the rule, the panels, the
#: captions. It is sized for a figure eighteen inches wide shown on a screen,
#: which is what the old 9.5 pt tick labels were not.
#:
#: ``_SUPP`` sets the apparatus below the crop line, which is read at a desk and
#: can afford to be smaller.
#:
#: Every ``fontsize=`` in this module reads from one of these. Passing a bare
#: number is how three different formula sizes ended up on four figures.
_MAIN = {
    "suptitle": 27.0, "rule": 21.0, "answer": 19.0, "panel": 17.0,
    "axis": 15.0, "tick": 13.0, "annot": 14.0, "verdict": 15.5,
    "caption": 14.0, "legend": 13.0, "inset": 11.0, "bar": 11.0,
    "heading": 19.0, "why": 15.0, "says": 15.0, "formula": 21.0,
    "step": 19.0, "badge": 14.0, "pointer": 14.0,
}
_SUPP = {
    "heading": 15.0, "step": 15.0, "badge": 12.5, "pointer": 12.0,
    "why": 12.5, "formula": 16.0, "says": 12.5,
    "symbol": 13.5, "explain": 12.0, "title": 13.5, "foot": 12.0,
}


def _panel_title(ax: "plt.Axes", letter: str, text: str) -> None:
    """Panel heading, in the one style the family uses.

    Wrapped to the panel it sits over. Three panels in a row are narrow enough
    that a title set as one line runs into its neighbour's.
    """
    fig = ax.get_figure()
    ax.set_title(_wrap_to_width(fig, f"{letter} · {text}",
                                ax.get_position().width, _MAIN["panel"]),
                 fontsize=_MAIN["panel"], fontweight="bold", loc="left", pad=10)


def _figure_title(fig: Figure, title: str, qualifier: str) -> None:
    """Figure heading: Title Case noun phrase, em dash, lowercase qualifier.

    Placed an inch from the top on every figure, for the same reason the footer
    is: the heights differ now, so a shared fraction is not a shared margin.
    """
    fig.suptitle(f"{title} — {qualifier}", fontsize=_MAIN["suptitle"],
                 fontweight="bold", y=1.0 - 0.62 / fig.get_figheight())


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


def _wrap(text: str, width: int) -> "list[str]":
    r"""Wrap ``text`` without ever breaking inside a ``$...$`` span.

    ``textwrap`` counts characters and will happily put a newline between the
    two dollars of a formula. Matplotlib then fails to parse the whole string
    and falls back to printing it literally -- which is how "$w \in [0.37,\ 
    0.71]$" ended up on a figure as its own source code. Math spans are swapped
    for placeholders of the same length, wrapped, and put back.
    """
    spans = re.findall(r"\$[^$]*\$", text)
    if not spans:
        return textwrap.wrap(text, width=width)
    holes = {}
    masked = text
    for i, span in enumerate(spans):
        key = f"\x00{i:03d}" + "\x01" * max(len(span) - 5, 0)
        holes[key] = span
        masked = masked.replace(span, key, 1)
    lines = textwrap.wrap(masked, width=width, break_long_words=False)
    return [_unmask(line, holes) for line in lines]


def _unmask(line: str, holes: "Mapping[str, str]") -> str:
    for key, span in holes.items():
        line = line.replace(key, span)
    return line


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

    ``scale`` is one of the two type scales. The same class sets the rule strip
    above the crop line and the derivations below it, at different sizes, from
    one place -- so "the presented band is larger" is a property of the frame
    rather than of every call site remembering.
    """

    def __init__(self, ax: "plt.Axes", top: float = 0.985, indent: float = 0.03,
                 scale: "Mapping[str, float]" = _SUPP) -> None:
        self.ax = ax
        self.y = top
        self.indent = indent
        self.top = top
        self.scale = scale

    # -- measurement ---------------------------------------------------
    def _axes_width(self, display: float) -> float:
        """Convert a width in display pixels to a fraction of the axes."""
        inv = self.ax.transAxes.inverted()
        return inv.transform((display, 0.0))[0] - inv.transform((0.0, 0.0))[0]

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

    def _fit(self, text: str, size: float) -> str:
        """Wrap ``text`` to this column's width, measured rather than counted.

        A character count is not a width, and these columns range from a third
        of the page to all of it. Every block that wraps goes through here, so
        a wide column uses its width instead of inheriting a number chosen for
        a narrow one.
        """
        fig = self.ax.get_figure()
        inv = self.ax.transAxes.inverted()
        zero = inv.transform((0.0, 0.0))[0]
        avail = 1.0 - self.indent
        best = _wrap(text, _WRAP_MIN)
        for n in range(_WRAP_MIN, _WRAP_MAX, 4):
            lines = _wrap(text, n)
            t = self.ax.text(0.0, -9.0, "\n".join(lines), fontsize=size)
            box = t.get_window_extent(renderer=fig.canvas.get_renderer())
            t.remove()
            if inv.transform((box.width, 0.0))[0] - zero > avail:
                break
            best = lines
        return "\n".join(best)

    # -- blocks --------------------------------------------------------
    def head(self, text: str, *, colour: str = _BK, gap: float = 7.0) -> None:
        """A section heading, flush with the column's left edge."""
        t = self.ax.text(0.0, self.y, text, fontsize=self.scale["heading"],
                         fontweight="bold", ha="left", va="top", color=colour)
        self._drop([t], gap)

    def step(self, num: str, text: str, panel: str, colour: str, *,
             gap: float = 8.0) -> None:
        """A numbered sub-step, ruled off and pointing at its evidence panel.

        The badge's width is measured, not assumed: the columns are no longer
        all the same width, and a fixed fraction that cleared the circle in a
        wide one printed the title straight through it in a narrow one.
        """
        self.ax.plot([0.0, 1.0], [self.y + 0.014, self.y + 0.014],
                     color="#E0E0E0", linewidth=0.9)
        badge = 0.5 * self._axes_width(
            self.scale["badge"] * 2.2 * self.ax.get_figure().dpi / 72.0)
        t = self.ax.text(badge * 2.0 + 0.012, self.y, text,
                         fontsize=self.scale["step"], fontweight="bold",
                         ha="left", va="top", color=colour)
        mid = self.y - self._height_of(t) / 2.0
        self.ax.text(badge, mid, num, fontsize=self.scale["badge"],
                     fontweight="bold", ha="center", va="center", color=colour,
                     zorder=5, bbox=dict(boxstyle="circle,pad=0.24",
                                         facecolor="white", edgecolor=colour,
                                         linewidth=1.3))
        self.ax.text(0.998, mid, f"panel {panel}", fontsize=self.scale["pointer"],
                     ha="right", va="center", color=colour, fontstyle="italic")
        self._drop([t], gap)

    def why(self, text: str, *, gap: float = 4.0) -> None:
        """The question the formula below it answers.

        Typographically distinct from :meth:`says`, which glosses the result, so
        a reader always knows whether a line is motivating or interpreting.
        """
        t = self.ax.text(self.indent, self.y, self._fit(text, self.scale["why"]),
                         fontsize=self.scale["why"], ha="left", va="top",
                         color=_DIM, fontstyle="italic", linespacing=1.4)
        self._drop([t], gap)

    def formula(self, tex: str, *, gap: float = 9.0) -> None:
        """One displayed formula, under the clause that motivates it."""
        t = self.ax.text(self.indent, self.y, tex, fontsize=self.scale["formula"],
                         ha="left", va="top", color=_BK)
        self._drop([t], gap)

    def says(self, text: str, *, colour: str = _GR, gap: float = 12.0) -> None:
        """What the formula above it means, wrapped to the column."""
        t = self.ax.text(self.indent, self.y, self._fit(text, self.scale["says"]),
                         fontsize=self.scale["says"], ha="left", va="top",
                         color=colour, linespacing=1.5)
        self._drop([t], gap)

    def rule(self, tex: str, *, gap: float = 11.0, indent: float = 0.0) -> None:
        """One formula in the strip above the panels, at presentation size."""
        t = self.ax.text(indent, self.y, tex, fontsize=self.scale["rule"],
                         ha="left", va="top", color=_BK)
        self._drop([t], gap)

    def answer(self, text: str, colour: str, *, gap: float = 6.0) -> None:
        """The conclusion the rule reaches, banded so it reads as the point.

        Drawn into its own axes beside the formulas when the frame gave the
        column one, and centred against them. Stacked under the formulas it had
        a whole line to itself and left the right half of the strip empty.
        """
        into = getattr(self, "answer_at", None)
        if into is None:
            t = self.ax.text(0.0, self.y, text, fontsize=self.scale["answer"],
                             fontweight="bold", ha="left", va="top", color=colour,
                             bbox=dict(boxstyle="round,pad=0.42",
                                       facecolor="white", edgecolor=colour,
                                       linewidth=1.6))
            self._drop([t], gap)
            return
        into.text(1.0, (1.0 + self.y) / 2.0, _wrap_to_width(
                      into.get_figure(), text,
                      into.get_position().width * 0.92, self.scale["answer"]),
                  fontsize=self.scale["answer"], fontweight="bold", ha="right",
                  va="center", color=colour, linespacing=1.4,
                  bbox=dict(boxstyle="round,pad=0.46", facecolor="white",
                            edgecolor=colour, linewidth=1.8))

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


#: Inches of margin, in the order the frame uses them. Kept in inches rather
#: than figure fractions because every other length here is an inch too, and a
#: fraction silently changes meaning the moment the figure height does.
_SIDE_IN = 0.62
_TITLE_IN = 1.45
_FOOT_IN = 0.80
_GUTTER_IN = 0.62
_BAND_GAP_IN = 0.70
_RULE_GAP_IN = 0.34

#: Room a panel needs outside its own box: for its title above, and for its tick
#: labels and axis label below. Reserved rather than hoped for -- with
#: ``add_axes`` these hang outside the rectangle, and the caption underneath was
#: printed straight through them when it was not.
_PANEL_TOP_IN = 0.52
_XLABEL_IN = 0.68
_CAPTION_IN = 0.52
_TITLE_LINE_IN = 0.32

#: Height of the band that gets presented -- the rule, the panels, the captions
#: -- and therefore where the crop line falls. The same on every figure in the
#: series on purpose: one crop setting has to work for all four, which is worth
#: a little slack on the figures with less to show. :func:`_frame` fails if a
#: figure cannot fit inside it.
_MAIN_IN = 7.60

#: Room under the crop rule for the label that names what follows it.
_SUPP_HEAD_IN = 0.52

#: How much of the rule strip the formulas take; the answer takes the rest,
#: beside them rather than beneath.
_RULE_SHARE = 0.72

#: Width split when a figure puts text beside its evidence rather than above it.
_COL_RATIO = (1.14, 1.0)


def _crop_offset(dpi: float) -> "tuple[float, int]":
    """Where to cut, in inches from the top of the figure and in pixels.

    The whole point of the two-band frame: this is the same number on all four
    figures, so one crop removes the apparatus from the set.
    """
    inches = _TITLE_IN + _MAIN_IN + _RULE_GAP_IN
    return inches, int(round(inches * dpi))


def _column_widths(width: float) -> "tuple[float, float]":
    """Prose and evidence column widths, in inches."""
    avail = width - 2.0 * _SIDE_IN - _GUTTER_IN
    unit = avail / sum(_COL_RATIO)
    return _COL_RATIO[0] * unit, _COL_RATIO[1] * unit


def _measure(draw: "Callable[[_Column], None]", width: float, *,
             scale: "Mapping[str, float]" = _SUPP, indent: float = 0.0) -> float:
    """Height in inches that ``draw`` needs in a column ``width`` inches wide.

    Drawn onto a throwaway figure of the right width and a generous height.
    Wrapping is by character count, so the height a block needs is fixed once
    its width is -- which is what makes measuring before the real figure exists
    possible at all, and lets the figure be exactly as tall as its content.
    """
    probe = plt.figure(figsize=(width, _PROBE_IN))
    ax = probe.add_axes((0.0, 0.0, 1.0, 1.0))
    _note_axis(ax)
    col = _Column(ax, top=1.0, indent=indent, scale=scale)
    draw(col)
    used = (1.0 - col.y) * _PROBE_IN
    plt.close(probe)
    return used


#: How narrow and how wide an apparatus column may be set, in inches. Roughly
#: 42 and 88 characters at the apparatus size -- the bounds inside which prose
#: stays readable. Balancing must not buy a flat bottom with a measure nobody
#: can follow.
_COL_MIN_IN = 3.55
_COL_MAX_IN = 7.60


#: Vertical space between two topics sharing a column, in points.
_TOPIC_GAP_PT = 26.0


def _measured(fn, width: float, _memo: dict = {}) -> float:
    """:func:`_measure`, remembered.

    The packing below measures the same topic at a handful of widths while it
    searches. Laying text out is the expensive part of drawing these figures,
    and the same question asked twice has the same answer.
    """
    key = (id(fn), round(width, 2))
    if key not in _memo:
        _memo[key] = _measure(fn, width)
    return _memo[key]


def _partitions(n: int) -> "list[list[int]]":
    """Every way of cutting ``n`` topics into contiguous runs, longest first.

    Contiguous because the topics are numbered steps and a reader follows them
    in order; a packing that put step 3 above step 1 would balance the band and
    ruin it.
    """
    out = []

    def walk(start: int, acc: "list[int]") -> None:
        if start == n:
            out.append(list(acc)); return
        for take in range(1, n - start + 1):
            acc.append(take); walk(start + take, acc); acc.pop()

    walk(0, [])
    return sorted(out, key=len, reverse=True)


def _size_columns(
    groups: "Sequence[Sequence[Callable[[_Column], None]]]",
    total: float, gutter: float, gap_in: float,
) -> "tuple[list[float], list[float]]":
    """Widths that make these columns end at about the same depth.

    A column's height at a given width is very nearly its area divided by that
    width, so allocating width in proportion to area levels the bottoms. Two
    passes converge; a third is not worth the layout time.
    """
    n = len(groups)
    avail = total - (n - 1) * gutter

    def heights(ws: "Sequence[float]") -> "list[float]":
        return [sum(_measured(fn, w) for fn in g) + (len(g) - 1) * gap_in
                for g, w in zip(groups, ws)]

    widths = [avail / n] * n
    hs = heights(widths)
    for _ in range(2):
        areas = [h * w for h, w in zip(hs, widths)]
        widths = _clamp_to_total([avail * a / sum(areas) for a in areas], avail)
        hs = heights(widths)
    return widths, hs


def _clamp_to_total(widths: "Sequence[float]", avail: float) -> "list[float]":
    """Hold every column inside the readable measure, still summing to ``avail``.

    Clamping alone would change the total, so whatever a clamp took or gave is
    pushed onto the columns that are still free to move.
    """
    out = [min(max(w, _COL_MIN_IN), _COL_MAX_IN) for w in widths]
    for _ in range(4):
        slack = avail - sum(out)
        free = [i for i, w in enumerate(out) if _COL_MIN_IN < w < _COL_MAX_IN]
        if abs(slack) < 1e-6 or not free:
            break
        for i in free:
            out[i] = min(max(out[i] + slack / len(free), _COL_MIN_IN), _COL_MAX_IN)
    return out


def _balance_columns(
    draws: "Sequence[Callable[[_Column], None]]", total: float, gutter: float,
) -> "tuple[list[list], list[float], list[float]]":
    """Pack the apparatus topics into columns that end at the same depth.

    Equal-width, one-topic-per-column sized the band to the tallest column, so
    the short ones left a void -- on the three-column figure that was 6.9 in,
    forty per cent of the band. Widening the tall column alone does not close
    it either: that topic is half again the next even at the widest measure
    prose stays readable in.

    So both are searched together -- how many columns, which topics share one,
    and how wide each is -- and the packing with the shallowest band wins.
    Topics stay whole and stay in order: a newspaper flow would level the
    bottoms exactly, but it can put a formula and the gloss that reads it on
    opposite sides of a gutter, which costs more than the void did.

    Returns the topic groups, their widths, and their heights.
    """
    gap_in = _TOPIC_GAP_PT / 72.0
    best = None
    for parts in _partitions(len(draws)):
        groups, at = [], 0
        for take in parts:
            groups.append(list(draws[at:at + take])); at += take
        widths, hs = _size_columns(groups, total, gutter, gap_in)
        if min(widths) < _COL_MIN_IN - 1e-6:
            continue
        score = (round(max(hs), 2), round(sum(max(hs) - h for h in hs), 2))
        if best is None or score < best[0]:
            best = (score, groups, widths, hs)
    if best is None:
        raise AssertionError("no apparatus packing fits the readable measure")
    return best[1], best[2], best[3]


def _measure_rule(rule: "Callable[[_Column], None]",
                  w_formula: float, w_answer: float) -> float:
    """Height the rule strip needs with the answer beside the formulas.

    Measuring it with :func:`_measure` alone counts the answer as if it were
    stacked under the formulas, because that is what a column with nowhere to
    put it does -- and the strip then reserved half an inch it never used. Both
    halves are laid out here and the taller one wins.
    """
    probe = plt.figure(figsize=(w_formula + w_answer, _PROBE_IN))
    frac = w_formula / (w_formula + w_answer)
    ax_f = probe.add_axes((0.0, 0.0, frac, 1.0))
    ax_a = probe.add_axes((frac, 0.0, 1.0 - frac, 1.0))
    _note_axis(ax_f)
    _note_axis(ax_a)
    col = _Column(ax_f, top=1.0, indent=0.0, scale=_MAIN)
    col.answer_at = ax_a
    rule(col)
    used = (1.0 - col.y) * _PROBE_IN
    for t in ax_a.texts:
        box = t.get_window_extent(renderer=probe.canvas.get_renderer())
        used = max(used, box.height / probe.dpi + 0.30)
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
        t = ax.text(0.0, -9.0, sym, fontsize=_SUPP["symbol"])
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
            t = ax.text(0.0, -9.0, "\n".join(_wrap(meaning, n)),
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
    columns: "int | None" = None, title: str = "Symbols",
    size: float = _SUPP["explain"],
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
    ax.text(0.0, top, title, fontsize=_SUPP["title"], fontweight="bold",
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
            s = ax.text(left, col.y, sym, fontsize=_SUPP["symbol"], ha="left",
                        va="top", color=_BK)
            m = ax.text(left + gutter, col.y,
                        "\n".join(_wrap(meaning, wrap)),
                        fontsize=size, ha="left", va="top", color=_GR,
                        linespacing=1.35)
            col._drop([s, m], 7.0)
        floor = min(floor, col.y)
    if floor < -0.01:
        raise AssertionError(f"{what}: symbol table overruns its band")
    return top - floor


def _verdict(ax: "plt.Axes", text: str, colour: str) -> None:
    """The step's conclusion, banded across the foot of its evidence."""
    ax.text(0.5, 0.02, text, transform=ax.transAxes, fontsize=_MAIN["verdict"],
            fontweight="bold", ha="center", va="bottom", color=colour, zorder=9,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=colour, linewidth=1.3, alpha=0.96))


def _crop_rule(fig: Figure, y_in: float, height: float, width: float) -> None:
    """The labelled cut between what is presented and what supports it.

    Drawn, not implied. A reader has to be able to see where the figure ends and
    the apparatus begins, and so does whoever is cropping it.
    """
    x0, x1 = _SIDE_IN / width, 1.0 - _SIDE_IN / width
    for dy in (0.0, 0.030):
        fig.add_artist(plt.Line2D([x0, x1], [(y_in - dy) / height] * 2,
                                  color="#BDBDBD", linewidth=1.0,
                                  transform=fig.transFigure, zorder=1))
    fig.text(x0, (y_in - 0.088) / height,
             "supporting detail — derivations, caveats, and symbols",
             fontsize=_SUPP["title"], color=_DIM, ha="left", va="top",
             fontstyle="italic")


def _captions(fig: Figure, panels: "Sequence[plt.Axes]",
              texts: "Sequence[str]", colours: "Sequence[str]") -> None:
    """One line under each panel saying what that panel establishes.

    The panel title says what is plotted; this says what it settles. Together
    they are what lets the presented band stand on its own once the derivation
    below the crop line has been cropped away.

    Drawn by the frame, into the strip the frame reserved, wrapped to the panel
    it belongs to -- when the panels drew their own the text ran through the
    axis labels above it and into the neighbouring panel beside it.
    """
    for ax, text, colour in zip(panels, texts, colours):
        box = ax.get_position()
        wide = _wrap_to_width(fig, text, box.width, _MAIN["caption"])
        fig.text(box.x0, box.y0 - (_XLABEL_IN / fig.get_figheight()), wide,
                 fontsize=_MAIN["caption"], color=colour, ha="left", va="top",
                 linespacing=1.35)


def _wrap_to_width(fig: Figure, text: str, avail: float, size: float) -> str:
    """Wrap ``text`` so no line exceeds ``avail`` as a fraction of the figure.

    Measured, like everything else here. A character count is not a width: the
    same forty characters are a different length with a subscript in them.
    """
    best = _wrap(text, _WRAP_MIN)
    for n in range(_WRAP_MIN, _WRAP_MAX, 2):
        lines = _wrap(text, n)
        t = fig.text(0.0, -9.0, "\n".join(lines), fontsize=size)
        box = t.get_window_extent(renderer=fig.canvas.get_renderer())
        t.remove()
        if box.width / (fig.get_figwidth() * fig.dpi) > avail:
            break
        best = lines
    return "\n".join(best)


def _title_lines(text: str, frac: float, letter: str) -> int:
    """How many lines a panel title takes at ``frac`` of the figure width."""
    probe = plt.figure(figsize=(FIGURE_SIZE[0], 2.0))
    n = len(_wrap_to_width(probe, f"{letter} · {text}", frac,
                           _MAIN["panel"]).split("\n"))
    plt.close(probe)
    return n


def _frame(
    rule: "Callable[[_Column], None]",
    apparatus: "Sequence[Callable[[_Column], None]]",
    panel_text: "Sequence[tuple[str, str, str]]",
    symbols: "Sequence[tuple[str, str]]",
    *,
    columns: "int | None" = None,
) -> "tuple[Figure, list]":
    """The two-band frame every formula figure in this stage uses.

    Above the crop line: the rule, the evidence panels in a row, and a caption
    under each. Below it: the derivations in columns, then the symbols. The band
    above is the same height on all four figures so that one crop setting serves
    the set; the band below is measured and is whatever its content needs.

    Panels sit in a row rather than a column because every one of them plots a
    seventeen-point sequence against ``k``: wide and short is the right shape
    for that, and it gives the cropped band a proportion that fits a slide.

    ``panel_text`` is one ``(letter, title, caption)`` per panel. Every callable
    is drawn twice -- once on a throwaway figure to find out how tall it is,
    once for real -- so all of them must only draw.
    """
    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    n_evidence = len(panel_text)

    rule_in = _measure_rule(rule, inner * _RULE_SHARE - _GUTTER_IN,
                            inner * (1.0 - _RULE_SHARE)) + _SLACK_IN
    groups, w_cols, h_cols = _balance_columns(apparatus, inner, _GUTTER_IN)
    text_in = max(h_cols, default=0.0) + _SLACK_IN

    probe = plt.figure(figsize=(inner, _PROBE_IN))
    ax_probe = probe.add_axes((0.0, 0.0, 1.0, 1.0))
    _note_axis(ax_probe)
    band_in = _symbol_table(ax_probe, symbols, columns=columns) * _PROBE_IN + _SLACK_IN
    plt.close(probe)

    # How much room the titles need is not a constant: three panels in a row
    # are narrow enough that a title wraps, and a wrapped one printed straight
    # through the answer box above it.
    w_panel = (inner - (n_evidence - 1) * _GUTTER_IN) / n_evidence
    title_lines = max(
        _title_lines(t, w_panel / width, letter) for letter, t, _ in panel_text)
    top_in = _PANEL_TOP_IN + (title_lines - 1) * _TITLE_LINE_IN
    panel_in = (_MAIN_IN - rule_in - top_in - _XLABEL_IN - _CAPTION_IN)
    if panel_in < 2.0:
        raise AssertionError(
            f"the rule strip needs {rule_in:.2f}in, which leaves only "
            f"{panel_in:.2f}in of panel inside the {_MAIN_IN:.2f}in band"
        )

    support_in = text_in + _BAND_GAP_IN + band_in
    height = (_TITLE_IN + _MAIN_IN + _RULE_GAP_IN + _SUPP_HEAD_IN + support_in
              + _FOOT_IN)

    fig = plt.figure(figsize=(width, height))
    x0 = _SIDE_IN / width
    y_main = (height - _TITLE_IN - _MAIN_IN) / height

    # The formulas take the left of the strip and the answer sits beside them,
    # not under them: stacked, the strip used barely half the width and the
    # answer had a whole line to itself.
    ax_rule = fig.add_axes((x0, (height - _TITLE_IN - rule_in) / height,
                            inner * _RULE_SHARE / width, rule_in / height))
    _note_axis(ax_rule)
    col_rule = _Column(ax_rule, top=1.0, indent=0.0, scale=_MAIN)
    col_rule.answer_at = fig.add_axes(
        (x0 + inner * _RULE_SHARE / width + _GUTTER_IN / width,
         (height - _TITLE_IN - rule_in) / height,
         inner * (1.0 - _RULE_SHARE) / width - _GUTTER_IN / width,
         rule_in / height))
    _note_axis(col_rule.answer_at)
    rule(col_rule)
    col_rule.check("rule strip")

    panels = [
        fig.add_axes(((_SIDE_IN + i * (w_panel + _GUTTER_IN)) / width,
                      y_main + (_CAPTION_IN + _XLABEL_IN) / height,
                      w_panel / width, panel_in / height))
        for i in range(n_evidence)
    ]
    for ax, (letter, title, _) in zip(panels, panel_text):
        _panel_title(ax, letter, title)
    _captions(fig, panels, [c for _, _, c in panel_text],
              [_GR] * n_evidence)

    _crop_rule(fig, height - _TITLE_IN - _MAIN_IN - _RULE_GAP_IN, height, width)

    y_text = (_FOOT_IN + band_in + _BAND_GAP_IN) / height
    lefts = [_SIDE_IN + sum(w_cols[:i]) + i * _GUTTER_IN for i in range(len(w_cols))]
    for i, (group, w_col, left) in enumerate(zip(groups, w_cols, lefts)):
        ax_c = fig.add_axes((left / width, y_text, w_col / width,
                             text_in / height))
        _note_axis(ax_c)
        col = _Column(ax_c, top=1.0, indent=0.0)
        for j, fn in enumerate(group):
            if j:
                col._drop([], _TOPIC_GAP_PT)
            fn(col)
        col.check(f"apparatus column {i + 1}")

    ax_tab = fig.add_axes((x0, _FOOT_IN / height, inner / width, band_in / height))
    _note_axis(ax_tab)
    _symbol_table(ax_tab, symbols, columns=columns)

    return fig, panels


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

    def why_neff(col: _Column) -> None:
        col.head(r"Where  $N_k$  comes from")
        col.why("An unbalanced study is worth less than its head-count. How much less? "
                "Equate the variance of an allele-frequency difference in the unbalanced "
                "set to the balanced set that would have the same variance.")
        col.formula(r"$\mathrm{Var} = f(1-f)\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)$")
        col.formula(r"$\qquad\equiv\;\dfrac{2f(1-f)}{N_k}"
                    r"\;\;\Longrightarrow\;\; N_k = \dfrac{4\,n_1 n_2}{n_1 + n_2}$")
        col.says(r"$f$ is the allele frequency and cancels, so $N_k$ depends only on the "
                 r"two counts. An unbalanced set is worth less than its raw total: "
                 rf"{n1[-1]:,} + {n2[-1]:,} samples give $N_k$ = {neff[-1]:,.0f}, not "
                 rf"{n1[-1] + n2[-1]:,}.")

    def why_spread(col: _Column) -> None:
        col.head(r"Where  $H_k$  comes from")
        col.why(f"How wide the retained set still is on the {d} leading axes of a PCA "
                f"fitted to the major cluster itself. Spread on {d} axes is {d} numbers "
                f"plus their correlations, so it has to be reduced to one.")
        col.formula(rf"$H_k = \left|\Sigma_k\right|^{{1/2d}} = "
                    rf"\left(\prod_{{i=1}}^{{d}} \sqrt{{\lambda_i}}\right)^{{1/d}}$")
        col.says(r"$\Sigma_k$ is the sample covariance of the retained samples and "
                 r"$\lambda_i$ its eigenvalues, so $H_k$ is the geometric mean of the "
                 r"per-axis SDs — one number carrying both the variances and their "
                 r"correlations. Adding samples from further out can only raise it.")

    def the_walk(col: _Column) -> None:
        col.head("What the walk is")
        col.why("Every formula in this series is indexed by $k$, so what $k$ selects has "
                "to be fixed before any of them mean anything.")
        col.says(f"The major cluster has K = {rank.max()} components. Order them by "
                 f"{case_label}/{control_label} ratio and let cut $k$ keep the top $k$: "
                 f"one nested set per $k$, {rank.max()} of them, and $k$ is the only thing "
                 f"being chosen. Keeping more raises power and raises heterogeneity "
                 f"together, which is why there is a decision at all.")

    def rule(col: _Column) -> None:
        col.why(f"The major cluster splits into K = {rank.max()} components. Order them by "
                f"{case_label}/{control_label} ratio and let cut $k$ keep the top $k$; two "
                f"quantities change along that walk, and they change together.")
        col.rule(r"$N_k = \dfrac{4\,n_1 n_2}{n_1 + n_2}$"
                 r"$\qquad\qquad$"
                 r"$H_k = \left|\Sigma_k\right|^{1/2d} = "
                 rf"\left(\prod_{{i=1}}^{{d}} \sqrt{{\lambda_i}}\right)^{{1/d}}, "
                 rf"\quad d = {d}$")
        col.answer("Neither can be improved without giving up the other — so $k$ has to "
                   "be chosen, not computed", _BK)

    fig, (ax_r, ax_b) = _frame(
        rule, (the_walk, why_neff, why_spread),
        (("A", "the order the walk follows",
          "Each bar is one component, labelled with its id; cut $k$ keeps the "
          "leftmost $k$."),
         ("B", r"both rise strictly with $k$",
          "Power and spread, min-max scaled onto one axis. Widening the set buys "
          "one and costs the other.")),
        symbols)

    # ── evidence: the ranking, then the two quantities rising ────────
    _clusters = rank_table.sort_values("Rank")["Cluster"].to_numpy(dtype=int)
    ax_r.bar(np.arange(1, len(ratios) + 1), ratios, color="#BDBDBD",
             edgecolor="white", linewidth=0.8)
    # Which component each bar is: the ordering is the input to everything that
    # follows, and without the ids it cannot be checked against the table.
    for _i, (_c, _v) in enumerate(zip(_clusters, ratios), start=1):
        ax_r.text(_i, _v, str(_c), fontsize=_MAIN["bar"], ha="center", va="bottom", color=_GR)
    ax_r.set_xticks(np.arange(1, len(ratios) + 1)[::2])
    ax_r.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_r.set_ylabel(f"{case_label}/{control_label} ratio", fontsize=_MAIN["axis"])
    ax_r.tick_params(labelsize=_MAIN["tick"])

    ax_b.plot(rank, _safe_norm(neff), "-o", color=_BK, markersize=4.2, linewidth=1.6,
              markerfacecolor="white", markeredgewidth=1.0, label=r"$N_k$")
    ax_b.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=4.2, linewidth=1.6,
              markerfacecolor="white", markeredgewidth=1.0, label=r"$H_k$")
    ax_b.set_xticks(rank[::2]); ax_b.set_ylim(-0.08, 1.32)
    ax_b.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_b.set_ylabel("min-max normalised", fontsize=_MAIN["axis"])
    ax_b.tick_params(labelsize=_MAIN["tick"])
    ax_b.legend(loc="upper left", fontsize=_MAIN["legend"], ncol=2, frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")

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
    i_pk = int(np.argmax(excess))
    excess_peak = float(excess[i_pk])
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

    def why_rate(col: _Column) -> None:
        col.head("Why the walk has one price", colour=colour)
        col.why("Both quantities rise together. At what rate does the walk trade one for "
                "the other? Take the whole walk end to end, so the rate is a property of "
                "the walk and not of any one step.")
        col.formula(r"$r = \dfrac{N_K - N_1}{H_K - H_1}$" rf"$\;=\;{rate:,.1f}$")
        col.why("Given that rate, has a cut paid above it or below it? Subtract what the "
                "spread it took on would have cost at the average price.")
        col.formula(r"$E_k = (N_k - N_1) - r\,(H_k - H_1)$")
        col.says(r"$E_k$ is in units of $N_{eff}$: the surplus, in effective samples, that "
                 r"cut $k$ has over paying the average price for the spread it took on. "
                 r"Where $E_k$ peaks, spread stops repaying.")

    def why_cumulative(col: _Column) -> None:
        col.head("Cumulative, not per-step")
        col.why("The same question asked one step at a time gives a different answer. Why "
                "the cumulative form is the right one.")
        col.says(f"The per-step rate is not monotone. Steps "
                 f"{', '.join(str(x) for x in above[-3:])} sit above the average while the "
                 f"ones between them fall below it, so 'the last step above the average "
                 f"rate' would answer k = {max(above)}. The cumulative form asks a "
                 f"different question — has the walk so far repaid what it took on — and "
                 f"answers k = {k_nar}.")
        col.says(f"Panel A is that per-step rate, and it is shown precisely because it "
                 f"disagrees: a rule that looks equally reasonable answers "
                 f"{max(above)} rather than {k_nar}.")

    def why_margin(col: _Column) -> None:
        col.head("How firmly it is placed", colour=colour)
        col.why("A peak is only worth taking if the next-best cut is meaningfully behind "
                "it. How far back is it?")
        col.says(f"The margin is the lead over the runner-up in the same units, "
                 f"{margin:,.1f} effective samples. It is reported rather than optimised: "
                 f"nothing about the rule changes if the margin is small, but how much "
                 f"weight the answer carries does.")

    def rule(col: _Column) -> None:
        col.why("Both axes rise together, so the whole walk exchanges spread for power at "
                "one average rate — and every cut can then be scored by how much power it "
                "bought above that rate.")
        col.rule(r"$r = \dfrac{N_K - N_1}{H_K - H_1}$" rf"$\;=\;{rate:,.0f}$"
                 r"$\qquad\qquad$"
                 r"$E_k = (N_k - N_1) - r\,(H_k - H_1)$")
        col.answer(rf"narrow  =  $\arg\max_k E_k$  =  {k_nar}", colour)

    fig, (ax_s, ax_e) = _frame(
        rule, (why_rate, why_cumulative, why_margin),
        (("A", f"the per-step rate rebounds — it would answer $k$ = {max(above)}",
          "Drawn because it disagrees: a rule that looks just as reasonable "
          "picks a different cut."),
         ("B", rf"$E_k$ peaks at $k$ = {k_nar},  +{excess_peak:,.1f}",
          f"Every cut scored, so the maximum can be checked rather than taken on "
          f"faith. It leads the next by {margin:,.1f}.")),
        symbols)

    # ── evidence ─────────────────────────────────────────────────────
    ax_s.axhline(rate, color=colour, linewidth=1.4, linestyle="--")
    ax_s.text(rank[-1], rate, f"  $r$ = {rate:,.0f}", color=colour, fontsize=_MAIN["annot"],
              va="bottom", ha="right", fontweight="bold")
    ax_s.plot(rank[1:], step, "-o", color=_GR, markersize=4.0, linewidth=1.4,
              markerfacecolor="white", markeredgewidth=1.0)
    ax_s.set_yscale("log")
    ax_s.set_xticks(rank[::2])
    ax_s.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_s.set_ylabel(r"per-step $\Delta N/\Delta H$", fontsize=_MAIN["axis"])
    ax_s.tick_params(labelsize=_MAIN["tick"])

    ax_e.axhline(0.0, color=_DIM, linewidth=1.0, linestyle="--")
    ax_e.vlines(rank, 0.0, excess, color="#D6D6D6", linewidth=4.0)
    ax_e.vlines(rank[i_pk], 0.0, excess[i_pk], color=colour, linewidth=4.0)
    ax_e.plot(rank, excess, "-", color=_GR, linewidth=1.3)
    ax_e.plot([rank[i_pk]], [excess[i_pk]], _MARK["narrow"], color=colour, markersize=11.0,
              markeredgecolor="white", markeredgewidth=1.3)
    ax_e.set_xticks(rank[::2])
    ax_e.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_e.set_ylabel(r"$E_k$  (excess $N_{eff}$)", fontsize=_MAIN["axis"])
    ax_e.tick_params(labelsize=_MAIN["tick"])

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

    def why_distance(col: _Column) -> None:
        col.step("1", "The obstacle", "A", "#B35806")
        col.says("Spread says how wide the retained set is. It does not say whether the "
                 "two arms sit at different places inside it — and only that biases an "
                 "association test.", gap=14.0)
        col.why("How far apart do the two arms sit inside the retained set? Measure the "
                "gap between their centroids against the scatter within them, so the "
                "answer does not depend on how the axes are scaled.")
        col.formula(r"$S = \dfrac{(n_1-1)C_1 + (n_2-1)C_2}{n_1 + n_2 - 2}$")
        col.formula(r"$\hat{D}^2_k = \Delta\bar{x}^{\top} S^{-1} \Delta\bar{x}$")
        col.why("Some of that gap is sampling noise, and more of it in small cuts. "
                "Subtract what noise alone contributes, so cuts of different sizes can be "
                "compared.")
        col.formula(r"$s_k = \hat{D}^2_k - d\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)$")
        col.says(r"Two sample means never coincide and $\hat{D}^2$ squares the gap, so "
                 r"sampling alone contributes $d(1/n_1 + 1/n_2)$ whatever the truth.")

    def why_real(col: _Column) -> None:
        col.step("1", "…and is it real?", "A", "#B35806")
        col.why("De-biasing removes what sampling gives on average. It does not say the "
                "remainder is anything, so test the gap outright against the distribution "
                "it would follow if the two arms came from one population.")
        col.formula(r"$T^2_k = \hat{D}^2_k\,\dfrac{n_1 n_2}{n_1+n_2}$")
        col.formula(r"$F = \dfrac{T^2_k\,(\nu - d + 1)}{d\,\nu} \sim F_{d,\ \nu-d+1}$")
        col.says(f"{int(sig.sum())} of {len(pval)} cuts separate significantly, so this is "
                 f"a phenomenon and not noise — which is what makes it worth selecting "
                 f"against.", gap=14.0)
        col.says(f"But $s_k$ reverses direction {reversals} times. There is no single rate "
                 f"to read off it, so the pricing argument of the first cut cannot be "
                 f"repeated here.", colour="#B35806")

    def why_rule(col: _Column) -> None:
        col.step("2", "The rule", "B", colour)
        col.says("Two kinds of residual structure, one cut to choose. They are in "
                 "different units, so neither can be traded against the other directly.",
                 gap=14.0)
        col.why("Weigh them into a single number, and rescale it so the axis it forms "
                "spans a full unit — otherwise the distance below is measured across a "
                "fraction of one axis and the whole of the other.")
        col.formula(r"$u_k(w) = w\,x_k + (1-w)\,\tilde{s}_k$")
        col.formula(r"$\tilde{H}_k = \mathrm{minmax}(u_k(w))$")
        col.why("And which cut comes closest to having neither problem? Take the one "
                "nearest the corner where residual structure is lowest and power is "
                "highest.")
        col.formula(r"$k^{*}(w) = \arg\min_k \sqrt{\tilde{H}_k^{\,2} + (1 - y_k)^2}$")
        col.says(r"$x_k, y_k, \tilde{s}_k$ are $H_k$, $N_k$, $s_k$ min-max scaled to "
                 r"$[0,1]$. The corner $(0,1)$ is unattainable — no cut has zero structure "
                 r"and full power — so the nearest cut to it is taken.", gap=14.0)
        col.says(f"The rule is the proximity to that corner. The weight only decides how "
                 f"the two kinds of structure are mixed before it is measured.")

    def why_weight(col: _Column) -> None:
        col.step("3", r"The weight", "C", colour)
        col.says(f"Nothing in the data fixes $w$, so the honest thing is to say what "
                 f"bounds it and then show the answer does not turn on the choice.",
                 gap=14.0)
        col.why("What fixes its floor is which of the two terms is allowed to dominate "
                "the other.")
        col.formula(r"$w \geq \frac{1}{2} \;\Longleftrightarrow\; w \geq 1 - w$")
        col.says(f"Below ½ the term built from {case_label}/{control_label} labels would "
                 f"outweigh the one built from genotypes, and minimising that optimises "
                 f"what the association test is meant to measure. ½ is the boundary, not "
                 f"a tuned value.", gap=14.0)
        col.says(f"Panel C sweeps $w$ across its whole range. The answer holds across "
                 f"$w \\in [{lo:.2f},\\ {hi:.2f}]$, so ½ sits inside a plateau rather than "
                 f"on a knife edge.")

    def rule(col: _Column) -> None:
        col.why(f"A second kind of residual structure — how far apart the two arms sit "
                f"inside the retained set — has no rate to price against: it reverses "
                f"{reversals} times. So the two axes are combined and the cut nearest the "
                f"unattainable corner is taken.")
        col.rule(r"$s_k = \hat{D}^2_k - d\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)$"
                 r"$\qquad$"
                 r"$\tilde{H}_k = \mathrm{minmax}\left(w\,x_k + "
                 r"(1-w)\,\tilde{s}_k\right)$"
                 r"$\qquad$"
                 r"$k^{*} = \arg\min_k \sqrt{\tilde{H}_k^{\,2} + (1 - y_k)^2}$")
        col.answer(rf"intermediate  =  {k_int}   at  $w = \frac{{1}}{{2}}$,   "
                   rf"distance {dist_min:.4f}", colour)

    fig, (ax_o, ax_g, ax_w) = _frame(
        rule, (why_distance, why_real, why_rule, why_weight),
        (("A", "one axis has a rate; the other has none",
          f"$s_k$ reverses {reversals}×, so it cannot be priced — but it is real: "
          f"filled markers are the {int(sig.sum())} cuts that separate at "
          f"$p<0.05$."),
         ("B", rf"every cut's distance, smallest at $k$ = {k_int}",
          "All 17 scored, so the minimum is checkable. Inset: what a distance is, "
          "on the blended plane."),
         ("C", rf"it holds across $w \in [{lo:.2f},\ {hi:.2f}]$",
          r"The answer over the whole sweep of $w$. Shaded: where labels would "
          r"outweigh genotypes.")),
        symbols)

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
    ax_o.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_o.set_ylabel("min-max normalised", fontsize=_MAIN["axis"])
    ax_o.tick_params(labelsize=_MAIN["tick"])
    ax_o.legend(loc="upper center", fontsize=_MAIN["legend"], ncol=2, frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")

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
                  xy=(rank[runner], dist[runner]), xytext=(0, -14),
                  textcoords="offset points", fontsize=_MAIN["annot"], color=_GR,
                  ha="center", va="top", zorder=6)
    ax_g.plot([rank[i_int]], [dist[i_int]], _MARK["intermediate"], color=colour,
              markersize=12.0, markeredgecolor="white", markeredgewidth=1.4, zorder=7)
    ax_g.annotate(f"$k$ = {k_int}   {dist[i_int]:.4f}\n"
                  f"{(dist[runner] / dist[i_int] - 1) * 100:.0f}% clear of the next",
                  xy=(rank[i_int], dist[i_int]), xytext=(-12, -8),
                  textcoords="offset points", fontsize=_MAIN["annot"], fontweight="bold",
                  color=colour, ha="right", va="top", zorder=8)
    ax_g.set_xticks(rank[::2])
    # Headroom for the inset, which sits over the top-right corner.
    ax_g.set_ylim(0.0, float(dist.max()) * 1.34)
    ax_g.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_g.set_ylabel(r"distance to $(0,1)$", fontsize=_MAIN["axis"])
    ax_g.tick_params(labelsize=_MAIN["tick"])

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
    ax_in.tick_params(labelsize=_MAIN["inset"], length=3.0, pad=2)
    ax_in.set_xlabel(r"$\tilde{H}_k$", fontsize=_MAIN["inset"], labelpad=1)
    ax_in.set_ylabel(r"$y_k$", fontsize=_MAIN["inset"], labelpad=1)
    ax_in.set_title(r"the corner $(0,1)$", fontsize=_MAIN["inset"], pad=3, color=_GR)
    ax_in.grid(True, alpha=0.25, linewidth=0.5)
    ax_in.set_axisbelow(True)


    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color=_BAR, alpha=0.55, linewidth=0)
    ax_w.text(safe_weight_floor / 2.0, 0.55, "not usable —\nlabels outweigh spread",
              transform=ax_w.get_xaxis_transform(), fontsize=_MAIN["annot"], color=_BAR_INK,
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
    ax_w.set_xlabel(r"$w$ — weight on spread;  $1-w$ on distance", fontsize=_MAIN["axis"], labelpad=3)
    ax_w.set_ylabel(r"winning cut  $k^{*}(w)$", fontsize=_MAIN["axis"])
    ax_w.tick_params(labelsize=_MAIN["tick"])

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

    def why_three(col: _Column) -> None:
        col.head("Why three, and not one")
        col.why("A single cohort would need one worry to dominate. Three different things "
                "can, so three sets are delivered and the reason for each is stated.")
        col.says(r"The three are nested — narrow $\subset$ intermediate $\subset$ full — "
                 r"so this is a choice of where to stop along one walk, not a choice "
                 r"between three different lists. A narrower set buys homogeneity with "
                 r"effective sample size; a broader one does the reverse.")
        col.says(f"Cuts resolved in mode: {mode}. The automatic and manual answers agree "
                 f"on both selected cuts; cut_record.tsv carries that comparison.")
        col.head("What each one contains")
        for i, name in enumerate(order):
            added = (comps[name] if i == 0
                     else [c for c in comps[name]
                           if c not in set(comps[order[i - 1]])])
            col.says(f"{name} — " + ("" if i == 0 else "adds ")
                     + ", ".join(str(c) for c in added)
                     + (f"  ({len(comps[name])} components)" if i else ""),
                     colour=_EDGE[name], gap=5.0)

    def why_p(col: _Column) -> None:
        col.head(r"What  $p_k$  is, and is not")
        col.why("The last column is the only property of a delivered cohort that no step "
                "of the selection optimised for, so it needs saying plainly what it is.")
        col.says(r"$p_k$ is Hotelling's exact $F$ test on the case/control centroid gap "
                 r"inside the retained set. It is reported, never optimised against — no "
                 r"cut was chosen because its $p$ was small or large.")
        col.says("Of the three, intermediate is the only one where that gap is not "
                 "detectable, and narrow sits at the strongest separation anywhere in the "
                 "walk. Both are consequences of where the cuts fell, not reasons they "
                 "fell there.")

    symbol_cols = 3
    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    apparatus = (why_three, why_p)
    groups, w_cols, h_cols = _balance_columns(apparatus, inner, _GUTTER_IN)
    text_in = max(h_cols) + _SLACK_IN

    probe = plt.figure(figsize=(inner, _PROBE_IN))
    ax_probe = probe.add_axes((0.0, 0.0, 1.0, 1.0))
    _note_axis(ax_probe)
    band_in = _symbol_table(ax_probe, symbols, columns=symbol_cols) * _PROBE_IN + _SLACK_IN
    plt.close(probe)

    # Same presented band as the other three, split differently inside it. The
    # locator and the three reasons sit side by side, so the locator gets a
    # panel's proportions instead of the 7:1 strip it had when it spanned the
    # page; the numeric table keeps the full width beneath them, and is now only
    # numbers, the reasons having moved into the cards.
    top_in = 3.30
    table_in = (_MAIN_IN - top_in - 2.0 * _PANEL_TOP_IN - _XLABEL_IN
                - _CAPTION_IN)
    support_in = text_in + _BAND_GAP_IN + band_in
    height = (_TITLE_IN + _MAIN_IN + _RULE_GAP_IN + _SUPP_HEAD_IN + support_in
              + _FOOT_IN)

    fig = plt.figure(figsize=(width, height))
    x0, w = _SIDE_IN / width, inner / width
    y_main = (height - _TITLE_IN - _MAIN_IN) / height
    y_top = y_main + (_CAPTION_IN + table_in + _PANEL_TOP_IN + _XLABEL_IN) / height
    w_loc = inner * 0.56
    ax_loc = fig.add_axes(((_SIDE_IN + 0.55) / width, y_top,
                           (w_loc - 0.55) / width, top_in / height))
    ax_card = fig.add_axes(((_SIDE_IN + w_loc + _GUTTER_IN) / width, y_top,
                            (inner - w_loc - _GUTTER_IN) / width, top_in / height))
    ax_t = fig.add_axes((x0, y_main + _CAPTION_IN / height, w, table_in / height))
    _note_axis(ax_card)
    _note_axis(ax_t)

    _crop_rule(fig, height - _TITLE_IN - _MAIN_IN - _RULE_GAP_IN, height, width)

    y_text = (_FOOT_IN + band_in + _BAND_GAP_IN) / height
    lefts = [_SIDE_IN + sum(w_cols[:i]) + i * _GUTTER_IN for i in range(len(w_cols))]
    for i, (group, w_col, left) in enumerate(zip(groups, w_cols, lefts)):
        ax_c = fig.add_axes((left / width, y_text, w_col / width,
                             text_in / height))
        _note_axis(ax_c)
        c = _Column(ax_c, top=1.0, indent=0.0)
        for j, fn in enumerate(group):
            if j:
                c._drop([], _TOPIC_GAP_PT)
            fn(c)
        c.check(f"03 apparatus column {i + 1}")

    ax_tab = fig.add_axes((x0, _FOOT_IN / height, w, band_in / height))
    _note_axis(ax_tab)
    _symbol_table(ax_tab, symbols, columns=symbol_cols)

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
                        textcoords="offset points", fontsize=_MAIN["annot"], fontweight="bold",
                        color=_EDGE[name], ha="center", va="center", zorder=7)
    ax_loc.set_xlabel(r"residual spread  $H_k$   $\rightarrow$ less homogeneous",
                      fontsize=_MAIN["axis"], labelpad=4)
    ax_loc.set_ylabel(r"$N_k$   $\rightarrow$ more power", fontsize=_MAIN["axis"])
    ax_loc.tick_params(labelsize=_MAIN["tick"])
    ax_loc.legend(loc="lower right", fontsize=_MAIN["legend"], frameon=True, framealpha=0.95,
                  edgecolor="#CFCFCF")
    ax_loc.grid(True, alpha=0.30, linewidth=0.7); ax_loc.set_axisbelow(True)
    ax_loc.margins(x=0.14, y=0.12)
    _panel_title(ax_loc, "A", "where the three sit on the trade-off")

    # The reason each cohort is offered, not a restatement of the rule that
    # located it: three sets exist because three different things can be the
    # dominant worry, and the reader has to pick on that basis.
    defs = {
        "narrow": (r"$\arg\max_k E_k$",
                   "Residual stratification is the main worry. Buys the most "
                   "homogeneity the walk offers before spread stops repaying."),
        "intermediate": (r"$\arg\min_k \sqrt{\tilde{H}_k(\frac{1}{2})^2 + (1-y_k)^2}$",
                         "Both worries at once. The only one of the three whose "
                         f"{case_label}/{control_label} gap is not detectable, at 94% of "
                         "full's power."),
        "full": ("every major-cluster component",
                 "Power is the main worry, or a reference is wanted. Nothing was "
                 "selected, so nothing can have been selected wrongly."),
    }
    # The reasons, beside the locator rather than inside the table. Read as
    # three cards they are a decision the reader makes; read as a wrapped cell
    # in a numeric row they were something to skip past.
    _panel_title(ax_card, "B", "which one to use")
    card_h = 0.94 / len(order)
    for i, name in enumerate(order):
        top = 0.96 - i * card_h
        ax_card.add_patch(FancyBboxPatch(
            (0.0, top - card_h + 0.030), 1.0, card_h - 0.045,
            boxstyle="round,pad=0.004", facecolor=_TINT[name],
            edgecolor=_EDGE[name], linewidth=1.4, alpha=0.85, zorder=1))
        ax_card.plot([0.014, 0.014], [top - card_h + 0.050, top - 0.022],
                     color=_EDGE[name], linewidth=5.0, solid_capstyle="butt",
                     zorder=3)
        ax_card.text(0.042, top - 0.055, name, fontsize=_MAIN["panel"],
                     fontweight="bold", ha="left", va="top", color=_EDGE[name],
                     zorder=3)
        ax_card.text(0.99, top - 0.055, f"$k$ = {k_of[name]}",
                     fontsize=_MAIN["panel"], fontweight="bold", ha="right",
                     va="top", color=_EDGE[name], zorder=3)
        ax_card.text(0.042, top - 0.055 - card_h * 0.30,
                     _wrap_to_width(fig, defs[name][1],
                                    ax_card.get_position().width * 0.93,
                                    _MAIN["caption"]),
                     fontsize=_MAIN["caption"], ha="left", va="top", color=_BK,
                     linespacing=1.4, zorder=3)

    _panel_title(ax_t, "C", "and what each one delivers")
    cols = ("", "definition", "$k$", case_label, control_label,
            "n", r"$N_k$", r"$H_k$", r"$p_k$", "")
    xs = (0.004, 0.108, 0.448, 0.520, 0.606, 0.692, 0.762, 0.828, 0.892, 0.999)
    for xx, c in zip(xs, cols):
        ax_t.text(xx, 0.880, c, fontsize=_MAIN["caption"], ha="left", va="center",
                  color=_DIM, fontstyle="italic")
    ax_t.plot([0.0, 1.0], [0.815, 0.815], color="#CFCFCF", linewidth=1.0)
    for i, name in enumerate(order):
        yy = 0.640 - i * 0.268
        ax_t.add_patch(FancyBboxPatch((0.0, yy - 0.120), 1.0, 0.250, boxstyle="square,pad=0",
                                      facecolor=_TINT[name], edgecolor="none", alpha=0.60, zorder=1))
        vals = (name, defs[name][0], str(k_of[name]),
                f"{val(name, f'{case_label}_Count'):,.0f}",
                f"{val(name, f'{control_label}_Count'):,.0f}",
                f"{val(name, 'Total_Count'):,.0f}",
                f"{val(name, 'GWAS_Neff'):,.0f}",
                f"{val(name, rgv_column):.5f}",
                f"{val(name, 'Mainland_CaseCtrl_P'):.1e}",
                "separated" if val(name, "Mainland_CaseCtrl_P") < 0.05
                else "not detectable")
        _sig = val(name, "Mainland_CaseCtrl_P") < 0.05
        for j, (xx, t) in enumerate(zip(xs, vals)):
            last = j == len(xs) - 1
            ax_t.text(xx, yy + 0.008, t,
                      fontsize=(_MAIN["panel"] if j == 0
                                else (_MAIN["bar"] if last else _MAIN["annot"])),
                      ha="right" if last else "left", va="center", zorder=3,
                      color=(_EDGE[name] if j == 0 else
                             (("#B35806" if _sig else _EDGE["intermediate"])
                              if last else _BK)),
                      fontweight="bold" if j in (0, 2) else "normal",
                      fontstyle="italic" if last else "normal")
    box = ax_t.get_position()
    fig.text(box.x0, box.y0 - 0.006,
             _wrap_to_width(fig,
                            "narrow $\\subset$ intermediate $\\subset$ full — nested by "
                            "construction, so this is a choice of where to stop along one "
                            "walk, not a choice between three different lists.",
                            box.width, _MAIN["caption"]),
             fontsize=_MAIN["caption"], color=_GR, ha="left", va="top")

    _figure_title(fig, "The Three Cohorts", "what steps 1–3 deliver")
    _series_footer(fig, "03_cohorts")
    return fig
