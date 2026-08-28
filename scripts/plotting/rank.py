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
             fontsize=_MAIN["caption"], color=_DIM, ha="center", va="bottom",
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

#: What each cut is chosen by, in the words used on the criteria panel. Two are
#: optimisations of a stated quantity; the third is a definition, and saying so
#: is the point of putting all three on one axis.
_CRITERION = {
    "narrow": r"$\max_k E_k$ — excess return",
    "intermediate": r"$\min_k$ distance to $(0,1)$",
    "full": "every component — no optimum",
}

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

#: One type scale for the whole stage. It is sized for a figure eighteen inches
#: wide shown on a screen, which is what the old 9.5 pt tick labels were not.
#: There were two while a second band of smaller print sat below a crop line;
#: that band is now prose in ``docs/outputs.md`` and the second scale went with
#: it.
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
                 scale: "Mapping[str, float]" = _MAIN) -> None:
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

#: Room a panel needs outside its own box: for its title above, and for its tick
#: labels and axis label below. Reserved rather than hoped for -- with
#: ``add_axes`` these hang outside the rectangle, and the caption underneath was
#: printed straight through them when it was not.
_PANEL_TOP_IN = 0.52
_XLABEL_IN = 0.68
_CAPTION_IN = 0.52
_TITLE_LINE_IN = 0.32

#: Height of one evidence panel. The figure is now just its content, so this is
#: what sets the shape of it. The same on every figure in the
#: series on purpose: one crop setting has to work for all four, which is worth
#: a little slack on the figures with less to show. :func:`_frame` fails if a
#: figure cannot fit inside it.
_PANEL_IN = 4.10


#: How much of the rule strip the formulas take; the answer takes the rest,
#: beside them rather than beneath.
_RULE_SHARE = 0.72

#: Width split when a figure puts text beside its evidence rather than above it.
_COL_RATIO = (1.14, 1.0)


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
             scale: "Mapping[str, float]" = _MAIN, indent: float = 0.0) -> float:
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


def _note_height(text: str, width: float) -> float:
    """Inches the definition line needs at ``width``. Measured, like the rest."""
    probe = plt.figure(figsize=(width, _PROBE_IN))
    t = probe.text(0.0, 0.5, _wrap_to_width(probe, text, 1.0, _MAIN["note"]),
                   fontsize=_MAIN["note"], linespacing=1.4)
    box = t.get_window_extent(renderer=probe.canvas.get_renderer())
    used = box.height / probe.dpi
    plt.close(probe)
    return used + 0.22


def _frame(
    rule: "Callable[[_Column], None]",
    panel_text: "Sequence[tuple[str, str, str]]",
    *,
    note: str = "",
) -> "tuple[Figure, list]":
    """The frame every formula figure in this stage uses.

    The equations and the cut they reach, one line of symbol definitions, the
    evidence panels in a row, and a caption under each. That is the whole
    figure.

    It used to carry a second band below a crop rule holding the argument for
    each choice. That band grew to two thirds of the text on these figures --
    prose set as pictures -- and it now lives in ``docs/outputs.md``, where
    prose belongs. Nothing here needs cropping any more: the whole figure is the
    presentable part.

    Panels sit in a row rather than a column because every one of them plots a
    seventeen-point sequence against ``k``: wide and short is the right shape
    for that, and it gives the figure a proportion that fits a slide.

    ``panel_text`` is one ``(letter, title, caption)`` per panel. ``rule`` is
    drawn twice -- once on a throwaway figure to measure it -- so it must only
    draw.
    """
    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    n_evidence = len(panel_text)

    rule_in = _measure_rule(rule, inner * _RULE_SHARE - _GUTTER_IN,
                            inner * (1.0 - _RULE_SHARE)) + _SLACK_IN

    # How much room the titles need is not a constant: three panels in a row
    # are narrow enough that a title wraps, and a wrapped one printed straight
    # through the answer box above it.
    w_panel = (inner - (n_evidence - 1) * _GUTTER_IN) / n_evidence
    title_lines = max(
        _title_lines(t, w_panel / width, letter) for letter, t, _ in panel_text)
    top_in = _PANEL_TOP_IN + (title_lines - 1) * _TITLE_LINE_IN
    note_in = _note_height(note, inner) if note else 0.0

    height = (_TITLE_IN + rule_in + note_in + top_in + _PANEL_IN + _XLABEL_IN
              + _CAPTION_IN + _FOOT_IN)

    fig = plt.figure(figsize=(width, height))
    x0 = _SIDE_IN / width

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

    if note:
        _symbol_note(fig, x0, height - _TITLE_IN - rule_in - 0.10, height, width, note)

    panels = [
        fig.add_axes(((_SIDE_IN + i * (w_panel + _GUTTER_IN)) / width,
                      (_FOOT_IN + _CAPTION_IN + _XLABEL_IN) / height,
                      w_panel / width, _PANEL_IN / height))
        for i in range(n_evidence)
    ]
    for ax, (letter, title, _) in zip(panels, panel_text):
        _panel_title(ax, letter, title)
    _captions(fig, panels, [c for _, _, c in panel_text], [_GR] * n_evidence)

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

    fig, (ax_r, ax_n, ax_b) = _frame(
        rule,
        (("A", "the order the walk follows",
          "Each bar is one component, labelled with its id; cut $k$ keeps the "
          "leftmost $k$."),
         ("B", r"what the imbalance costs — (1)",
          rf"Raw head-count against effective size. The shaded gap is what the "
          rf"{case_label}/{control_label} imbalance gives up; at $k$ = "
          rf"{int(rank.max())} it is {n1[-1] + n2[-1] - neff[-1]:,.0f} samples."),
         ("C", r"(1) and (2) both rise strictly with $k$",
          "Min-max scaled onto one axis. Widening the set buys one and costs "
          "the other.")),
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

    # (1) drawn rather than asserted: the whole point of an effective size is
    # that it is below the head-count, and by how much.
    ax_n.fill_between(rank, neff, n1 + n2, color="#D6D6D6", alpha=0.75, zorder=2,
                      label="what imbalance gives up")
    ax_n.plot(rank, n1 + n2, "-", color=_GR, linewidth=1.8, zorder=3,
              label=r"$N_{\mathrm{tot}}$ — raw head-count")
    ax_n.plot(rank, neff, "-o", color=_BK, markersize=4.2, linewidth=1.8,
              markerfacecolor="white", markeredgewidth=1.0, zorder=4,
              label=r"$N_{\mathrm{eff},k}$ — effective")
    ax_n.set_xticks(rank[::2])
    ax_n.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_n.set_ylabel("samples", fontsize=_MAIN["axis"])
    ax_n.tick_params(labelsize=_MAIN["tick"])
    ax_n.legend(loc="upper left", fontsize=_MAIN["legend"], frameon=True,
                framealpha=0.95, edgecolor="#CFCFCF")

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

    for a in (ax_r, ax_n, ax_b):
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

    def rule(col: _Column) -> None:
        col.lead("Both axes rise together, so the walk has one average price — and every "
                "cut can be scored by how much power it bought above it.")
        col.rule(r"$\gamma = \dfrac{N_{\mathrm{eff},K} - N_{\mathrm{eff},1}}"
                 r"{H_K - H_1}$" rf"$\;=\;{rate:,.0f}$", eq=1)
        col.rule(r"$E_k = (N_{\mathrm{eff},k} - N_{\mathrm{eff},1}) "
                 r"- \gamma\,(H_k - H_1)$", eq=2)
        col.answer(rf"narrow  =  $\arg\max_k E_k$  =  {k_nar}", colour)

    fig, (ax_s, ax_e) = _frame(
        rule,
        (("A", r"the walk, and the rate (1) prices across it",
          r"$E_k$ is the vertical gap between the curve and the chord — how much "
          r"power a cut bought above the average price."),
         ("B", rf"(2) peaks at $k$ = {k_nar},  $E_k$ = +{excess_peak:,.1f}",
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

    def rule(col: _Column) -> None:
        col.lead(f"A second kind of residual structure has no rate to price against: it "
                f"reverses {reversals} times. The two axes are combined instead, and the "
                f"cut nearest the unattainable corner is taken.")
        col.rule(r"$s_k = \hat{D}^2_k - d\left(\dfrac{1}{N_{\mathrm{case}}}"
                 r"+\dfrac{1}{N_{\mathrm{ctrl}}}\right)$", eq=1)
        col.rule(r"$u_k(w) = w\,\tilde{H}_k + (1-w)\,\tilde{s}_k, \qquad "
                 r"\tilde{u}_k = \mathrm{minmax}\left(u_k(w)\right)$", eq=2)
        col.rule(r"$k^{*} = \arg\min_k \sqrt{\tilde{u}_k^{\,2} "
                 r"+ (1 - \tilde{N}_k)^2}$", eq=3)
        col.answer(rf"intermediate  =  {k_int}   at  $w = \frac{{1}}{{2}}$,   "
                   rf"distance {dist_min:.4f}", colour)

    fig, (ax_o, ax_g, ax_w) = _frame(
        rule,
        (("A", "one axis has a rate, one has none — so (2)",
          f"$\\tilde{{s}}_k$ from (1) reverses {reversals}×, so it cannot be priced "
          f"— but it is real: filled markers are the {int(sig.sum())} cuts "
          f"separating at $P<0.05$. $\\tilde{{u}}_k$ is their blend at $w$ = ½."),
         ("B", r"the plane (3) minimises over",
          rf"Every cut placed by structure and power; the dashed segment is the "
          rf"distance (3) makes smallest. Inset: all 17 scored."),
         ("C", rf"(3) gives the same cut across $w \in [{lo:.2f},\ {hi:.2f}]$",
          r"(3) re-run at every $w$ in (2). Shaded: below ½ the labels would "
          r"outweigh the genotypes.")),
        note=(r"$\hat{D}^2_k$ centroid gap, pooled-scatter units  ·  $d$ axes, "
              r"$N_{\mathrm{case}}, N_{\mathrm{ctrl}}$ samples  ·  $s_k$ the gap less "
              r"its sampling floor  ·  a tilde is that quantity min-max scaled to "
              r"$[0,1]$, so $\tilde{H}_k, \tilde{N}_k, \tilde{s}_k$ are spread, power "
              r"and $s_k$ on a common scale  ·  $w$ weight on spread  ·  $u_k(w)$ their "
              r"blend, $\tilde{u}_k$ it rescaled  ·  $k^{*}$ the cut chosen"))

    # ── evidence ─────────────────────────────────────────────────────
    ax_o.axhline(0.0, color=_DIM, linewidth=1.0, linestyle=":")
    ax_o.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=4.0, linewidth=1.5,
              markerfacecolor="white", markeredgewidth=1.0,
              label=r"$\tilde{H}_k$ — never reverses")
    _sn = _safe_norm(sep)
    ax_o.plot(rank, _sn, "-", color="#B35806", linewidth=1.8, zorder=3,
              label=rf"$\tilde{{s}}_k$ — reverses {reversals}×")
    # Filled where the separation is significant: the de-biased value alone does
    # not say whether what is left is real, and that is what makes this axis a
    # phenomenon rather than noise.
    ax_o.plot(rank[sig], _sn[sig], "^", color="#B35806", markersize=5.4, zorder=4,
              markeredgecolor="white", markeredgewidth=1.0,
              label=rf"$P < 0.05$  ({int(sig.sum())} of {len(pval)})")
    ax_o.plot(rank[~sig], _sn[~sig], "^", color="white", markersize=5.4, zorder=4,
              markeredgecolor="#B35806", markeredgewidth=1.4, label=r"$P \geq 0.05$")
    # (2) has no panel of its own, but its two inputs are already on this one --
    # so drawing the blend here is drawing the equation being made.
    ax_o.plot(rank, blended, "-", color=colour, linewidth=2.4, zorder=5,
              label=r"$\tilde{u}_k$ — the blend, (2)")
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
    ax_g.annotate("unattainable $(0,\\,1)$", xy=(0.0, 1.0),
                  xytext=(15, 7), textcoords="offset points",
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
    ax_g.set_xlim(-0.07, 1.10); ax_g.set_ylim(-0.34, 1.20)
    ax_g.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_g.set_xlabel(r"$\tilde{u}_k$   $\rightarrow$ more residual structure",
                    fontsize=_MAIN["axis"], labelpad=3)
    ax_g.set_ylabel(r"$\tilde{N}_k$   $\rightarrow$ more power", fontsize=_MAIN["axis"])
    ax_g.tick_params(labelsize=_MAIN["tick"])

    # The audit trail, in the corner the cuts leave empty: every cut scored, so
    # "smallest" is checkable rather than a judgement about a scatter.
    ax_in = ax_g.inset_axes((0.050, 0.040, 0.40, 0.29))
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
    ax_w.text(safe_weight_floor / 2.0, 0.13, "not usable —\nlabels outweigh spread",
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
    objective_spaces: "Mapping[str, Any]",
    blend_weight: float,
    mode: str,
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """4 of 4 — the three criteria, the sets they pick, and what each one costs."""
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

    # The two selection rules, on one axis and both pointing the same way, so
    # that each is read the same way: higher is better, and the peak is the
    # answer. Their optima are checked against cut_record rather than trusted.
    _rate = float(rows["narrow"]["Exchange_Rate"])
    excess = (neff - neff[0]) - _rate * (het - het[0])
    _blend = np.asarray(objective_spaces["intermediate"].structure, dtype=float)
    _power = _safe_norm(neff)
    _dist = np.sqrt(_blend ** 2 + (1.0 - _power) ** 2)
    score = {"narrow": _safe_norm(excess), "intermediate": 1.0 - _safe_norm(_dist)}
    for _n, _sc in score.items():
        if int(rank[int(np.argmax(_sc))]) != k_of[_n]:
            raise AssertionError(
                f"the {_n} criterion drawn on 03 peaks at "
                f"{int(rank[int(np.argmax(_sc))])}, not the recorded {k_of[_n]}")

    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    # A criteria and B locator side by side, then the cards and the numeric
    # table. The locator was an inset inside A and sat on nine of A's own
    # points; with the argument prose moved to the docs there is room for it to
    # be a panel.
    note = (r"$k$ the cut  ·  $N_{\mathrm{eff},k}$ effective sample size  ·  "
            r"$H_k$ residual spread  ·  $E_k$ excess over the walk's average rate  ·  "
            r"a tilde is that quantity min-max scaled to $[0,1]$: $\tilde{u}_k$ "
            r"blended structure, $\tilde{N}_k$ power  ·  "
            r"$P_k$ separation p-value")
    note_in = _note_height(note, inner)
    top_in, card_in, table_in = 2.80, 2.28, 2.10
    height = (_TITLE_IN + note_in + _PANEL_TOP_IN + top_in + _XLABEL_IN
              + _PANEL_TOP_IN + card_in + _PANEL_TOP_IN + table_in
              + _CAPTION_IN + _FOOT_IN)

    fig = plt.figure(figsize=(width, height))
    x0, w = _SIDE_IN / width, inner / width
    y_table = (_FOOT_IN + _CAPTION_IN) / height
    y_card = y_table + (table_in + _PANEL_TOP_IN) / height
    y_top = y_card + (card_in + _PANEL_TOP_IN + _XLABEL_IN) / height
    w_half = (inner - _GUTTER_IN) / 2.0
    ax_loc = fig.add_axes(((_SIDE_IN + 0.55) / width, y_top,
                           (w_half - 0.55) / width, top_in / height))
    ax_tr = fig.add_axes(((_SIDE_IN + w_half + _GUTTER_IN + 0.55) / width, y_top,
                          (w_half - 0.55) / width, top_in / height))
    ax_card = fig.add_axes((x0, y_card, w, card_in / height))
    ax_t = fig.add_axes((x0, y_table, w, table_in / height))
    _note_axis(ax_card)
    _note_axis(ax_t)

    # 03 has no rule strip, so its definitions sit under the title instead.
    _symbol_note(fig, x0, height - _TITLE_IN, height, width, note)

    # The three criteria on one axis, both rules mapped so that higher is
    # better, so each is read the same way and each peaks at its own answer.
    # Until now they lived on separate figures and the comparison had to be
    # assembled by the reader.
    for name in ("narrow", "intermediate"):
        ax_loc.plot(rank, score[name], "-", color=_EDGE[name], linewidth=2.4,
                    zorder=3, label=_CRITERION[name])
        i = int(np.argmin(np.abs(rank - k_of[name])))
        ax_loc.plot([rank[i]], [score[name][i]], _MARK[name], color=_EDGE[name],
                    markersize=15.0, markeredgecolor="white", markeredgewidth=1.8,
                    zorder=6)
        ax_loc.annotate(f"{name}\n$k$ = {k_of[name]}", xy=(rank[i], score[name][i]),
                        xytext=(0, 15), textcoords="offset points",
                        fontsize=_MAIN["annot"], fontweight="bold",
                        color=_EDGE[name], ha="center", va="bottom", zorder=7)
    # full is not an optimum of anything, and drawing three curves would imply
    # it were. It is the end of the walk, taken whole.
    ax_loc.axvline(k_of["full"], color=_EDGE["full"], linewidth=2.0, linestyle="--",
                   zorder=4, label=_CRITERION["full"])
    ax_loc.annotate(f"full\n$k$ = {k_of['full']}", xy=(k_of["full"], 1.52),
                    xytext=(-8, 0), textcoords="offset points",
                    fontsize=_MAIN["annot"], fontweight="bold",
                    color=_EDGE["full"], ha="right", va="center", zorder=7)
    ax_loc.set_xticks(rank[::2]); ax_loc.set_ylim(-0.06, 1.72)
    ax_loc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_loc.set_xlabel(r"cumulative rank $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax_loc.set_ylabel("criterion score\n" r"$\rightarrow$ better", fontsize=_MAIN["axis"])
    ax_loc.tick_params(labelsize=_MAIN["tick"])
    ax_loc.legend(loc="upper left", fontsize=_MAIN["legend"], frameon=True,
                  framealpha=0.95, edgecolor="#CFCFCF")
    ax_loc.grid(True, alpha=0.30, linewidth=0.7); ax_loc.set_axisbelow(True)
    _panel_title(ax_loc, "A", "three criteria, three answers")

    # B: where those answers land on the trade-off the whole series is about.
    ax_tr.plot(het, neff, "-o", color=_GR, markersize=4.4, linewidth=1.4,
               markerfacecolor="white", markeredgewidth=1.0, zorder=3,
               label=f"the {rank.max()} cumulative cuts")
    for name in order:
        i = int(np.argmin(np.abs(rank - k_of[name])))
        ax_tr.plot([het[i]], [neff[i]], _MARK[name], color=_EDGE[name],
                   markersize=15.0, markeredgecolor="white", markeredgewidth=1.8,
                   zorder=6)
        ax_tr.annotate(f"{name}\n$k$ = {k_of[name]}", xy=(het[i], neff[i]),
                       xytext={"narrow": (-58, -34), "intermediate": (4, 28),
                               "full": (-16, -34)}[name],
                       textcoords="offset points", fontsize=_MAIN["annot"],
                       fontweight="bold", color=_EDGE[name], ha="center",
                       va="center", zorder=7)
    ax_tr.set_xlabel(r"residual spread  $H_k$   $\rightarrow$ less homogeneous",
                     fontsize=_MAIN["axis"], labelpad=3)
    ax_tr.set_ylabel(r"$N_{\mathrm{eff},k}$   $\rightarrow$ more power",
                     fontsize=_MAIN["axis"])
    ax_tr.tick_params(labelsize=_MAIN["tick"])
    ax_tr.margins(x=0.16, y=0.16)
    ax_tr.grid(True, alpha=0.30, linewidth=0.7); ax_tr.set_axisbelow(True)
    _panel_title(ax_tr, "B", "where those answers land")

    # The reason each cohort is offered, not a restatement of the rule that
    # located it: three sets exist because three different things can be the
    # dominant worry, and the reader has to pick on that basis.
    defs = {
        "narrow": (r"$\arg\max_k E_k$",
                   "Residual stratification is the main worry. Buys the most "
                   "homogeneity the walk offers before spread stops repaying."),
        "intermediate": (r"$\arg\min_k \sqrt{\tilde{u}_k(\frac{1}{2})^2 + (1-\tilde{N}_k)^2}$",
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
    _panel_title(ax_card, "C", "basis for choosing among the three")
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

    _panel_title(ax_t, "D", "composition and cost of each cohort")
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
