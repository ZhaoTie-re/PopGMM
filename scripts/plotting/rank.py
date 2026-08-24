"""The rank-selection figures: the equations, their evidence, and what they fix.

Four figures in reading order -- the problem, each of the two selected cuts, and
the three cohorts they deliver. Each is in two bands, divided at the same height
on all four so that one crop yields four presentable figures: above the rule,
the numbered equations, the cut they reach, the evidence panels and their
captions; below it, the basis for each of those choices.

Each thing is stated once. A symbol is defined in the line under the equations;
what an equation computes is stated by the equation; why it is that equation and
not another is argued below the rule. The detail is deliberately uneven --
standard steps get an equation and a clause, contestable ones get a column.

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
    """"2 of 4 · what the figure is about", for a figure's footer.

    The neighbouring filenames used to be stamped here too. That is navigation
    between files rather than anything the figure says, and the numbering
    already carries the order.
    """
    names = [n for n, _ in FIGURE_SERIES]
    i = names.index(name)
    return f"Rank selection · {i + 1} of {len(names)} · {FIGURE_SERIES[i][1]}"


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
    "heading": 19.0, "lead": 15.0, "gloss": 15.0, "formula": 21.0, "note": 13.0,
    "step": 19.0, "badge": 14.0, "pointer": 14.0,
}
_SUPP = {
    "heading": 15.0, "step": 15.0, "badge": 12.5, "pointer": 12.0,
    "lead": 12.5, "formula": 16.0, "gloss": 12.5,
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

    def lead(self, text: str, *, gap: float = 4.0) -> None:
        """The clause that sets up the displayed equation below it.

        Typographically distinct from :meth:`gloss`, which reads the result off
        the equation, so a line is never ambiguous between the two roles.
        """
        t = self.ax.text(self.indent, self.y, self._fit(text, self.scale["lead"]),
                         fontsize=self.scale["lead"], ha="left", va="top",
                         color=_DIM, fontstyle="italic", linespacing=1.4)
        self._drop([t], gap)

    def formula(self, tex: str, *, gap: float = 9.0) -> None:
        """One displayed formula, under the clause that motivates it."""
        t = self.ax.text(self.indent, self.y, tex, fontsize=self.scale["formula"],
                         ha="left", va="top", color=_BK)
        self._drop([t], gap)

    def gloss(self, text: str, *, colour: str = _GR, gap: float = 12.0) -> None:
        """Reads the result off the equation above it, wrapped to the column."""
        t = self.ax.text(self.indent, self.y, self._fit(text, self.scale["gloss"]),
                         fontsize=self.scale["gloss"], ha="left", va="top",
                         color=colour, linespacing=1.5)
        self._drop([t], gap)

    def rule(self, tex: str, *, eq: "int | None" = None, gap: float = 11.0,
             indent: float = 0.0) -> None:
        """One numbered equation in the strip above the panels.

        Numbered per figure, right-aligned, one to a line, so a caption can say
        "scored by (3)" instead of "the rule" -- and so three formulas are not
        set side by side across one line at presentation size.
        """
        t = self.ax.text(indent, self.y, tex, fontsize=self.scale["rule"],
                         ha="left", va="top", color=_BK)
        if eq is not None:
            self.ax.text(1.0, self.y - self._height_of(t) / 2.0, f"({eq})",
                         fontsize=self.scale["answer"], ha="right", va="center",
                         color=_GR)
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
        into.text(0.988, (1.0 + self.y) / 2.0, _wrap_to_width(
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
_MAIN_IN = 8.55

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


#: Height of the throwaway figure the measurements are taken on. Only has to
#: exceed anything a real column needs; nothing about it reaches the output. It
#: is left at the default dpi on purpose -- glyph metrics are hinted to whole
#: pixels, so measuring at one dpi and drawing at another gives a wrap width
#: that is a line or two off.
_PROBE_IN = 24.0

#: Slack added to a measured height, so a sub-point rounding difference between
#: the probe and the drawing cannot fail the build. Far below anything visible.
_SLACK_IN = 0.08


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


#: Bounds on the search above. The floor keeps a pathological column from
#: wrapping to one word a line; the ceiling is past any width these bands have.
_WRAP_MIN = 22
_WRAP_MAX = 140


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
             "Methodological notes — the basis for each choice above",
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


def _note_height(text: str, width: float) -> float:
    """Inches the definition line needs at ``width``. Measured, like the rest.

    It was a constant, which fitted two lines and printed a third straight
    through the panel title beneath it.
    """
    probe = plt.figure(figsize=(width, _PROBE_IN))
    t = probe.text(0.0, 0.5, _wrap_to_width(probe, text, 1.0, _MAIN["note"]),
                   fontsize=_MAIN["note"], linespacing=1.4)
    box = t.get_window_extent(renderer=probe.canvas.get_renderer())
    used = box.height / probe.dpi
    plt.close(probe)
    return used + 0.22


def _symbol_note(fig: Figure, x0: float, y_in: float, height: float,
                 width: float, text: str) -> None:
    """The one line of definitions that keeps a cropped figure self-contained.

    Everything above the crop line has to stand on its own, and a formula whose
    symbols are only explained below the cut does not. This is the only place a
    symbol is defined: a table at the foot said the same thing again at greater
    length, and the derivations below said it a third time with a justification
    attached, so what a reader met three times they now meet once.
    """
    fig.text(x0, y_in / height,
             _wrap_to_width(fig, text, 1.0 - 2.0 * x0, _MAIN["note"]),
             fontsize=_MAIN["note"], color=_GR, ha="left", va="top",
             linespacing=1.4)


def _frame(
    rule: "Callable[[_Column], None]",
    apparatus: "Sequence[Callable[[_Column], None]]",
    panel_text: "Sequence[tuple[str, str, str]]",
    *,
    note: str = "",
) -> "tuple[Figure, list]":
    """The two-band frame every formula figure in this stage uses.

    Above the crop line: the rule, the evidence panels in a row, and a caption
    under each. Below it: the derivations, in columns. The band
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

    # How much room the titles need is not a constant: three panels in a row
    # are narrow enough that a title wraps, and a wrapped one printed straight
    # through the answer box above it.
    w_panel = (inner - (n_evidence - 1) * _GUTTER_IN) / n_evidence
    title_lines = max(
        _title_lines(t, w_panel / width, letter) for letter, t, _ in panel_text)
    top_in = _PANEL_TOP_IN + (title_lines - 1) * _TITLE_LINE_IN
    note_in = _note_height(note, inner) if note else 0.0
    panel_in = (_MAIN_IN - rule_in - note_in - top_in - _XLABEL_IN - _CAPTION_IN)
    if panel_in < 2.0:
        raise AssertionError(
            f"the rule strip needs {rule_in:.2f}in, which leaves only "
            f"{panel_in:.2f}in of panel inside the {_MAIN_IN:.2f}in band"
        )

    support_in = text_in
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

    if note:
        _symbol_note(fig, x0, height - _TITLE_IN - rule_in - 0.10, height, width, note)

    _crop_rule(fig, height - _TITLE_IN - _MAIN_IN - _RULE_GAP_IN, height, width)

    y_text = _FOOT_IN / height
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

    def note_basis(col: _Column) -> None:
        col.head(rf"Basis and dimension of $H$")
        col.gloss(f"$H$ could be measured on the global PCA's leading pair, and earlier "
                 f"versions of this analysis were. It is measured instead on {d} axes of "
                 f"a PCA fitted to the major cluster itself, and the choice changes which "
                 f"cut wins.", gap=14.0)
        col.gloss("The global PC1–PC2 are dominated by the split between the major cluster "
                 "and everything outside it. Inside the major cluster — which is all these "
                 "cuts ever contain — that pair carries little of the remaining structure, "
                 "so spread measured on it is close to flat along the walk.", gap=14.0)
        col.gloss(f"A basis fitted to the cluster puts the residual structure on its own "
                 f"leading axes, and {d} of them rather than 2 because the pair alone "
                 f"leaves visible structure on the next two. The exponent $1/2d$ keeps the "
                 f"result in SD units at any $d$, so (2) is comparable across that choice "
                 f"even though its value is not.", gap=14.0)
        col.gloss("Two values of $H$ are only comparable when they share a basis and a $d$. "
                 "Everything on these four figures shares both.")

    def note_neff(col: _Column) -> None:
        col.head(r"Effective size against head-count")
        col.lead(r"(1) is the standard result — equate the variance of an allele-frequency "
                r"difference in an unbalanced set to the balanced design that would have "
                r"the same variance, and the frequency $p$ cancels. Against the "
                r"head-count and the imbalance it reads")
        col.formula(r"$N_{\mathrm{eff}} = N_{\mathrm{tot}} \cdot "
                    r"\dfrac{4r}{(1+r)^2}, \qquad r = N_{\mathrm{case}}/"
                    r"N_{\mathrm{ctrl}}$", gap=13.0)
        col.gloss(rf"The result is a fair axis. The walk adds {case_label} and "
                 rf"{control_label} at very different rates — the ratio $r$ runs from "
                 rf"{n1[0] / max(n2[0], 1):.2f} at $k = 1$ to "
                 rf"{n1[-1] / max(n2[-1], 1):.2f} at $k = K$ — so raw totals would credit "
                 rf"a cut for samples that add almost nothing to power.", gap=14.0)
        col.gloss(rf"At the widest cut that is the difference between "
                 rf"{n1[-1] + n2[-1]:,} samples and {neff[-1]:,.0f} effective ones.")

    def rule(col: _Column) -> None:
        col.lead(f"Order the major cluster's {rank.max()} components by "
                f"{case_label}/{control_label} ratio and let cut $k$ keep the top $k$. "
                f"Two quantities change along that walk, and they change together.")
        col.rule(r"$N_{\mathrm{eff},k} = \dfrac{4\,N_{\mathrm{case}} "
                 r"N_{\mathrm{ctrl}}}{N_{\mathrm{case}} + N_{\mathrm{ctrl}}}$",
                 eq=1)
        col.rule(rf"$H_k = \left|\Sigma_k\right|^{{1/2d}} = "
                 rf"\left(\prod_{{i=1}}^{{d}} \sqrt{{\lambda_i}}\right)^{{1/d}}, "
                 rf"\qquad d = {d}$", eq=2)
        col.answer("Neither can be improved without giving up the other — so $k$ has to "
                   "be chosen, not computed", _BK)

    fig, (ax_r, ax_b) = _frame(
        rule, (note_basis, note_neff),
        (("A", "the order the walk follows",
          "Each bar is one component, labelled with its id; cut $k$ keeps the "
          "leftmost $k$."),
         ("B", r"both rise strictly with $k$",
          "Power and spread, min-max scaled onto one axis. Widening the set buys "
          "one and costs the other.")),
        note=(rf"$k$ the cut, one of $K = {rank.max()}$  ·  "
              rf"$N_{{\mathrm{{case}}}}, N_{{\mathrm{{ctrl}}}}$ samples per arm at that "
              rf"cut  ·  $N_{{\mathrm{{eff}},k}}$ the balanced study of equal power  ·  "
              rf"$H_k$ the residual spread of the retained set  ·  $\Sigma_k$ its "
              rf"covariance on the $d = {d}$ mainland PCA axes, $\lambda_i$ the "
              rf"eigenvalues of that covariance"))

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
              markerfacecolor="white", markeredgewidth=1.0, label=r"$N_{\mathrm{eff},k}$")
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

    _figure_title(fig, "The Problem",
                  "the decision, and the two quantities that move with it")
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
    colour = _EDGE["narrow"]

    def note_pricing(col: _Column) -> None:
        col.head("End-to-end pricing", colour=colour)
        col.gloss(r"(1) takes a single rate from the two ends of the walk, and (2) scores "
                 r"every cut against it. The alternative is to price each step against "
                 r"its own predecessor.", gap=14.0)
        col.gloss("That alternative asks a different question, and gets a different "
                 "answer. A per-step rate is not monotone here: it crosses the average "
                 "repeatedly, so 'the last step that paid above the average' lands on a "
                 "late cut for no reason beyond where the noise in one step fell.",
                 gap=14.0)
        col.gloss(r"The cumulative form asks whether the walk up to that cut has repaid what it "
                 r"took on, which is a property of the retained set rather than of the "
                 r"one component that entered last. Panel A draws that as a chord: (2) "
                 r"is the vertical gap between the curve and the straight line.")

    def note_margin(col: _Column) -> None:
        col.head("Interpretation of the margin", colour=colour)
        col.gloss(f"The peak leads the runner-up by {margin:,.1f} effective samples out of "
                 f"{excess_peak:,.1f}, so the neighbouring cuts price about the same.",
                 gap=14.0)
        col.gloss("That supports the reading that any cut in the neighbourhood is "
                  "defensible on this criterion. It does not support treating the peak "
                  "as sharp, which is why panel B scores every cut rather than marking "
                  "the winner alone.", gap=14.0)
        col.gloss("The margin is reported, never optimised. Nothing about the rule would "
                 "change if it were larger or smaller; only how much weight the answer "
                 "carries would.")

    def rule(col: _Column) -> None:
        col.lead("Both axes rise together, so the walk has one average price — and every "
                "cut can be scored by how much power it bought above it.")
        col.rule(r"$\gamma = \dfrac{N_{\mathrm{eff},K} - N_{\mathrm{eff},1}}"
                 r"{H_K - H_1}$" rf"$\;=\;{rate:,.0f}$", eq=1)
        col.rule(r"$E_k = (N_{\mathrm{eff},k} - N_{\mathrm{eff},1}) "
                 r"- \gamma\,(H_k - H_1)$", eq=2)
        col.answer(rf"narrow  =  $\arg\max_k E_k$  =  {k_nar}", colour)

    fig, (ax_s, ax_e) = _frame(
        rule, (note_pricing, note_margin),
        (("A", r"the walk, and the rate (1) prices across it",
          r"$E_k$ is the vertical gap between the curve and the chord — how much "
          r"power a cut bought above the average price."),
         ("B", rf"$E_k$ peaks at $k$ = {k_nar},  +{excess_peak:,.1f}",
          f"Every cut scored, so the maximum can be checked rather than taken on "
          f"faith. It leads the next by {margin:,.1f}.")),
        note=(r"$N_{\mathrm{eff},k}, H_k$ power and spread at cut $k$  ·  "
              r"$k = 1$ the tightest cut, $k = K$ the widest  ·  "
              r"$\gamma$ the average exchange rate over the whole walk  ·  "
              r"$E_k$ what a cut bought above it, in effective samples"))

    # ── evidence ─────────────────────────────────────────────────────
    # The walk itself, with the straight line the average rate describes drawn
    # from end to end. E_k is the vertical distance between the two, so the
    # panel shows what the formula says rather than asserting it.
    ax_s.plot([het[0], het[-1]], [neff[0], neff[-1]], "--", color=colour,
              linewidth=1.6, zorder=2,
              label=rf"the average rate  $\gamma$ = {rate:,.0f}")
    ax_s.plot(het, neff, "-o", color=_BK, markersize=5.0, linewidth=1.6,
              markerfacecolor="white", markeredgewidth=1.2, zorder=3,
              label=f"the {rank.max()} cumulative cuts")
    ax_s.vlines(het[i_pk], neff[0] + rate * (het[i_pk] - het[0]), neff[i_pk],
                color=colour, linewidth=5.0, alpha=0.75, zorder=4)
    ax_s.plot([het[i_pk]], [neff[i_pk]], _MARK["narrow"], color=colour,
              markersize=15.0, markeredgecolor="white", markeredgewidth=1.8,
              zorder=6)
    ax_s.annotate(rf"$k$ = {k_nar}" "\n" rf"$E_k$ = {excess_peak:,.1f}",
                  xy=(het[i_pk], neff[i_pk]), xytext=(14, -6),
                  textcoords="offset points", fontsize=_MAIN["annot"],
                  fontweight="bold", color=colour, ha="left", va="top", zorder=7)
    ax_s.set_xlabel(r"residual spread  $H_k$   $\rightarrow$ less homogeneous",
                    fontsize=_MAIN["axis"], labelpad=3)
    ax_s.set_ylabel(r"$N_{\mathrm{eff},k}$   $\rightarrow$ more power",
                    fontsize=_MAIN["axis"])
    ax_s.tick_params(labelsize=_MAIN["tick"])
    ax_s.legend(loc="lower right", fontsize=_MAIN["legend"], frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")
    ax_s.margins(x=0.10, y=0.12)

    ax_e.axhline(0.0, color=_DIM, linewidth=1.0, linestyle="--")
    ax_e.vlines(rank, 0.0, excess, color="#D6D6D6", linewidth=4.0)
    ax_e.vlines(rank[i_pk], 0.0, excess[i_pk], color=colour, linewidth=4.0)
    ax_e.plot(rank, excess, "-", color=_GR, linewidth=1.3)
    ax_e.plot([rank[i_pk]], [excess[i_pk]], _MARK["narrow"], color=colour, markersize=11.0,
              markeredgecolor="white", markeredgewidth=1.3)
    ax_e.set_xticks(rank[::2])
    ax_e.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_e.set_ylabel(r"$E_k$  (excess $N_{\mathrm{eff}}$)", fontsize=_MAIN["axis"])
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

    def note_second_axis(col: _Column) -> None:
        col.step("1", "The second axis", "A", "#B35806")
        col.gloss("Spread says how wide the retained set is. It does not say whether the "
                 "two arms sit at different places inside it, and only that biases an "
                 "association test — so a set can be homogeneous and still be the wrong "
                 "one to run on.", gap=14.0)
        col.gloss(r"(1) collects the standard pieces: the case/control centroid gap "
                  r"as a "
                 r"Mahalanobis distance $\hat{D}^2_k$ against the pooled within-arm "
                 r"scatter, less the floor $d(1/N_{\mathrm{case}}+1/N_{\mathrm{ctrl}})$ "
                 r"that two sample means show even drawn from one population.", gap=14.0)
        col.gloss(f"The floor matters at this scale. The retained set grows more than "
                  f"tenfold along the walk, so a raw $\\hat{{D}}^2$ would fall across it "
                  f"for arithmetic reasons alone and the axis would be an artefact of "
                  f"sample size rather than of ancestry.")

    def note_rescaling(col: _Column) -> None:
        col.step("2", "The second rescaling in (2)", "B", colour)
        col.gloss(r"$x_k$ and $\tilde{s}_k$ are already on $[0,1]$, so the blend inside "
                 r"(2) is too — but only in the sense that it cannot leave the interval. "
                 r"Its own range is narrower, because the two terms peak at different "
                 r"cuts and the average of them never reaches either end.", gap=14.0)
        col.gloss(f"(3) then measures a distance in a unit square. Left unrescaled, one "
                 f"of its two axes would span a fraction of that square and the other "
                 f"all of it, so the vertical and horizontal parts of the same distance "
                 f"would not be in the same units.", gap=14.0)
        col.gloss(f"This is not cosmetic. Without the second rescale the minimum is "
                 f"0.3446 rather than the recorded {dist_min:.4f}, and it does not fall "
                 f"at the same cut. It is the step of the whole procedure most likely "
                 f"to be dropped by someone reimplementing it from the formulas.",
                 colour=colour)

    def note_weight(col: _Column) -> None:
        col.step("3", r"Admissible range of $w$", "C", colour)
        col.lead("Nothing in the data fixes $w$, so the honest thing is to say what "
                "bounds it rather than to fit it.")
        col.formula(r"$w \geq \frac{1}{2} \;\Longleftrightarrow\; w \geq 1 - w$")
        col.gloss(f"Below ½ the term built from {case_label}/{control_label} labels "
                 f"outweighs the one built from genotypes — and minimising that is "
                 f"optimising the very thing the association test is meant to measure. "
                 f"½ is where that stops being true, not a value that was tuned.",
                 gap=14.0)
        col.gloss(f"Panel C sweeps $w$ across its whole range. The answer holds on "
                 f"$w \\in [{lo:.2f},\\ {hi:.2f}]$, so ½ sits inside a plateau rather "
                  f"than on a knife edge. The plateau supports the claim that the cut "
                  f"does not turn on the weight; it does not make ½ optimal.")

    def note_significance(col: _Column) -> None:
        col.step("4", r"Role of $P_k$", "A", "#B35806")
        col.gloss(f"De-biasing removes what sampling gives on average; it does not say "
                 f"the remainder is anything. Hotelling's exact $F$ test does, and "
                 f"{int(sig.sum())} of {len(pval)} cuts separate at $P<0.05$ — which is "
                 f"what makes this axis a phenomenon rather than noise.", gap=14.0)
        col.gloss("It is reported everywhere and selected on nowhere. Choosing the cut "
                 "with the largest $P$ would be choosing the set that best hides a real "
                 "difference between the arms, which is the opposite of what a cohort "
                 "is for.", gap=14.0)
        col.gloss(f"$s_k$ also reverses direction {reversals} times along the walk. There "
                 f"is no single rate to read off it, which is why the pricing argument "
                 f"of the first cut cannot simply be repeated on this axis.")

    def rule(col: _Column) -> None:
        col.lead(f"A second kind of residual structure has no rate to price against: it "
                f"reverses {reversals} times. The two axes are combined instead, and the "
                f"cut nearest the unattainable corner is taken.")
        col.rule(r"$s_k = \hat{D}^2_k - d\left(\dfrac{1}{N_{\mathrm{case}}}"
                 r"+\dfrac{1}{N_{\mathrm{ctrl}}}\right)$", eq=1)
        col.rule(r"$\tilde{H}_k = \mathrm{minmax}\left(w\,x_k + "
                 r"(1-w)\,\tilde{s}_k\right)$", eq=2)
        col.rule(r"$k^{*} = \arg\min_k \sqrt{\tilde{H}_k^{\,2} "
                 r"+ (1 - y_k)^2}$", eq=3)
        col.answer(rf"intermediate  =  {k_int}   at  $w = \frac{{1}}{{2}}$,   "
                   rf"distance {dist_min:.4f}", colour)

    fig, (ax_o, ax_g, ax_w) = _frame(
        rule, (note_second_axis, note_rescaling, note_weight, note_significance),
        (("A", "one axis has a rate; the other has none",
          f"$s_k$ from (1) reverses {reversals}×, so it cannot be priced — but it "
          f"is real: filled markers are the {int(sig.sum())} cuts separating at "
          f"$P<0.05$."),
         ("B", r"the plane (3) minimises over",
          rf"Every cut placed by structure and power; the dashed segment is the "
          rf"distance (3) makes smallest. Inset: all 17 scored."),
         ("C", rf"$k^{{*}}$ unchanged across $w \in [{lo:.2f},\ {hi:.2f}]$",
          r"The answer over the whole sweep of $w$. Shaded: where labels would "
          r"outweigh genotypes.")),
        note=(r"$\hat{D}^2_k$ centroid gap, pooled-scatter units  ·  $d$ axes, "
              r"$N_{\mathrm{case}}, N_{\mathrm{ctrl}}$ samples  ·  $s_k$ the gap less "
              r"its sampling floor  ·  $x_k, y_k, \tilde{s}_k$ spread, power, $s_k$ on "
              r"$[0,1]$  ·  $w$ weight on spread  ·  $\tilde{H}_k$ the blend, rescaled "
              r" ·  $k^{*}$ the cut chosen"))

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
              label=rf"$P < 0.05$  ({int(sig.sum())} of {len(pval)})")
    ax_o.plot(rank[~sig], _sn[~sig], "^", color="white", markersize=5.4, zorder=4,
              markeredgecolor="#B35806", markeredgewidth=1.4, label=r"$P \geq 0.05$")
    ax_o.set_xticks(rank[::2]); ax_o.set_ylim(-0.10, 1.34)
    ax_o.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_o.set_ylabel("min-max normalised", fontsize=_MAIN["axis"])
    ax_o.tick_params(labelsize=_MAIN["tick"])
    ax_o.legend(loc="upper center", fontsize=_MAIN["legend"], ncol=2, frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")

    # The rule is a distance in a plane, so the plane is the panel. The ranked
    # distances say which of the seventeen is smallest, which the eye cannot
    # read off a scatter -- but that is an audit trail, and it belongs in the
    # inset the geometry used to be squeezed into.
    i_int = int(np.argmin(np.abs(rank - k_int)))
    dist = np.sqrt(blended ** 2 + (1.0 - y) ** 2)
    order = np.argsort(dist)
    runner = int(order[1])

    ax_g.plot(blended, y, "o", color=_GR, markersize=8.0, alpha=0.75,
              markerfacecolor="white", markeredgewidth=1.4, zorder=3)
    # Only the cuts a reader needs to find are labelled. Seventeen labels over a
    # cluster this tight is not a scatter plot, it is a smudge.
    for _i in (0, int(np.argmax(rank))):
        ax_g.annotate(rf"$k$ = {int(rank[_i])}", xy=(blended[_i], y[_i]),
                      xytext=(9, 0), textcoords="offset points",
                      fontsize=_MAIN["annot"], color=_DIM, ha="left",
                      va="center", zorder=4)
    ax_g.plot([0.0, blended[i_int]], [1.0, y[i_int]], "--", color=colour,
              linewidth=2.2, zorder=5)
    ax_g.plot([0.0], [1.0], "*", color=colour, markersize=26.0, zorder=6,
              markeredgecolor="white", markeredgewidth=1.6)
    ax_g.annotate("the unattainable corner $(0,\\,1)$", xy=(0.0, 1.0),
                  xytext=(16, 6), textcoords="offset points",
                  fontsize=_MAIN["annot"], color=colour, ha="left", va="bottom",
                  zorder=7)
    ax_g.plot([blended[runner]], [y[runner]], "o", color="white", markersize=11.0,
              markeredgecolor=_GR, markeredgewidth=1.8, zorder=6)
    ax_g.annotate(f"runner-up $k$ = {rank[runner]}", xy=(blended[runner], y[runner]),
                  xytext=(11, 6), textcoords="offset points",
                  fontsize=_MAIN["annot"], color=_GR, ha="left", va="bottom",
                  zorder=7)
    ax_g.plot([blended[i_int]], [y[i_int]], _MARK["intermediate"], color=colour,
              markersize=15.0, markeredgecolor="white", markeredgewidth=1.6,
              zorder=8)
    ax_g.annotate(rf"$k^{{*}}$ = {k_int}", xy=(blended[i_int], y[i_int]),
                  xytext=(0, -14), textcoords="offset points",
                  fontsize=_MAIN["annot"], fontweight="bold", color=colour,
                  ha="center", va="top", zorder=8)
    ax_g.set_xlim(-0.07, 1.10); ax_g.set_ylim(-0.10, 1.20)
    ax_g.set_xlabel(r"$\tilde{H}_k$   $\rightarrow$ more residual structure",
                    fontsize=_MAIN["axis"], labelpad=3)
    ax_g.set_ylabel(r"$y_k$   $\rightarrow$ more power", fontsize=_MAIN["axis"])
    ax_g.tick_params(labelsize=_MAIN["tick"])

    # The audit trail, in the corner the cuts leave empty: every cut scored, so
    # "smallest" is checkable rather than a judgement about a scatter.
    ax_in = ax_g.inset_axes((0.055, 0.055, 0.44, 0.34))
    ax_in.vlines(rank, 0.0, dist, color="#D6D6D6", linewidth=2.6, zorder=2)
    ax_in.vlines(rank[i_int], 0.0, dist[i_int], color=colour, linewidth=2.6, zorder=3)
    ax_in.plot(rank, dist, "-", color=_GR, linewidth=1.0, zorder=4)
    ax_in.set_xticks([1, k_int, int(rank.max())])
    ax_in.set_ylim(0.0, float(dist.max()) * 1.10)
    ax_in.tick_params(labelsize=_MAIN["inset"], length=3.0, pad=2)
    ax_in.set_title(rf"(3) for every cut $k$ — least at {k_int}",
                    fontsize=_MAIN["inset"], pad=3, color=_GR)
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

    _figure_title(fig, "The Second Cut",
                  "the obstacle to repeating step 2, and the rule that replaces it")
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

    def note_cohorts(col: _Column) -> None:
        col.head("Three cohorts rather than one")
        col.lead("A single cohort would need one worry to dominate. Three different things "
                "can, so three sets are delivered and the reason for each is stated.")
        col.gloss(r"The three are nested — narrow $\subset$ intermediate $\subset$ full — "
                 r"so this is a choice of where to stop along one walk, not a choice "
                 r"between three different lists. A narrower set buys homogeneity with "
                 r"effective sample size; a broader one does the reverse.")
        col.gloss(f"Cuts resolved in mode: {mode}. The automatic and manual answers agree "
                 f"on both selected cuts; cut_record.tsv carries that comparison.")
        col.head("Composition of each cohort")
        for i, name in enumerate(order):
            added = (comps[name] if i == 0
                     else [c for c in comps[name]
                           if c not in set(comps[order[i - 1]])])
            col.gloss(f"{name} — " + ("" if i == 0 else "adds ")
                     + ", ".join(str(c) for c in added)
                     + (f"  ({len(comps[name])} components)" if i else ""),
                     colour=_EDGE[name], gap=5.0)

    def note_interpretation(col: _Column) -> None:
        col.head(r"Interpretation of  $P_k$")
        col.lead("The last column is the only property of a delivered cohort that no step "
                "of the selection optimised for, so it needs saying plainly what it is.")
        col.gloss(r"$P_k$ is Hotelling's exact $F$ test on the case/control centroid gap "
                 r"inside the retained set. It is reported, never optimised against — no "
                 r"cut was chosen because its $P$ was small or large.")
        col.gloss("Of the three, intermediate is the only one where that gap is not "
                 "detectable, and narrow sits at the strongest separation anywhere in the "
                 "walk. Both are consequences of where the cuts fell, not reasons they "
                 "fell there.")

    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    apparatus = (note_cohorts, note_interpretation)
    groups, w_cols, h_cols = _balance_columns(apparatus, inner, _GUTTER_IN)
    text_in = max(h_cols) + _SLACK_IN

    # Same presented band as the other three, split differently inside it. The
    # locator and the three reasons sit side by side, so the locator gets a
    # panel's proportions instead of the 7:1 strip it had when it spanned the
    # page; the numeric table keeps the full width beneath them, and is now only
    # numbers, the reasons having moved into the cards.
    note = (r"$k$ the cut  ·  $N_{\mathrm{eff},k}$ effective sample size  ·  "
            r"$H_k$ residual spread  ·  $E_k$ excess over the walk's average rate  ·  "
            r"$\tilde{H}_k, y_k$ structure and power on $[0,1]$  ·  "
            r"$P_k$ separation p-value")
    note_in = _note_height(note, inner)
    top_in = 3.10
    table_in = (_MAIN_IN - note_in - top_in - 2.0 * _PANEL_TOP_IN - _XLABEL_IN
                - _CAPTION_IN)
    support_in = text_in
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

    # 03 has no rule strip, so its definitions sit under the title instead.
    _symbol_note(fig, x0, height - _TITLE_IN, height, width, note)

    _crop_rule(fig, height - _TITLE_IN - _MAIN_IN - _RULE_GAP_IN, height, width)

    y_text = _FOOT_IN / height
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
    ax_loc.set_ylabel(r"$N_{\mathrm{eff},k}$   $\rightarrow$ more power", fontsize=_MAIN["axis"])
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
    _panel_title(ax_card, "B", "basis for choosing among the three")
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

    _panel_title(ax_t, "C", "composition and cost of each cohort")
    cols = ("", "definition", "$k$", case_label, control_label,
            "n", r"$N_{\mathrm{eff},k}$", r"$H_k$", r"$P_k$", "")
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

    _figure_title(fig, "The Three Cohorts", "the sets steps 1–3 deliver")
    _series_footer(fig, "03_cohorts")
    return fig
