"""How the three cohorts were chosen, in three figures that read as one.

The argument is in three parts, so it is in three figures:

1. ``00_problem``  -- the problem, and the three quantities we watch along it
2. ``01_tradeoff`` -- how those quantities are traded off against each other
3. ``02_cohorts``  -- the three cohorts that falls out, and when to use each

Every one of them is laid out by :func:`_story_row`: a row of cells, each a
heading, a small plot, the equations that plot draws, and the answer they reach,
joined left to right by arrows. Four figures once carried four different
layouts and the reader had to assemble the line through them; here the line is
the layout, and it is the same line on all three pages.

Equations stay on the figures because they *are* the reasoning. What is not
here is the reviewer's layer -- what each choice rules out, and the
alternatives it was taken against -- which is the *Methodological notes* in
``docs/outputs.md``, alongside ``cut_record.tsv`` and
``rank_decision_table.tsv``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from scripts.common import to_numeric_array

if TYPE_CHECKING:
    from scripts.rank_selection import RankSelectionConfig


# ---------------------------------------------------------------------------
# One spec for the figure
# ---------------------------------------------------------------------------

#: Canvas. Every figure in this stage is exactly this size: the three are meant
#: to be read one after another, and a deck that has to rescale them page by
#: page loses the thing that makes them one argument. The plot band absorbs
#: whatever the equations do not use, so a fixed height costs no white space.
FIGURE_SIZE: "tuple[float, float]" = (21.0, 8.6)

#: Narrow to full, which is the order they nest in and the order they are read.
CUT_ORDER: "tuple[str, str, str]" = ("narrow", "intermediate", "full")

#: Text greys.
_BK, _GR, _DIM = "#212121", "#424242", "#757575"

#: Per-cut identity. The only source of these colours.
_TINT = {"narrow": "#E7F1F8", "intermediate": "#E6F4EC", "full": "#FBEAEC"}
_EDGE = {"narrow": "#0571B0", "intermediate": "#008837", "full": "#B2182B"}

#: Not cohort colours, declared here for the same reason those are: so there is
#: one of each. Muted greys for unselected marks, the orange for the third
#: quantity (which belongs to no cohort), the wash over the weights that are not
#: admissible, and the hairline every legend frame uses.
_MUTED = "#D6D6D6"
_THIRD = "#B35806"
_BARRED = "#F4C7C3"
_HAIR = "#CFCFCF"

#: The page furniture: the ground a card sits on, the fill of a card, the rule
#: that draws every border and every row line, and the band behind a table head.
_PAGE = "#F7F9FC"
_CARD = "#FFFFFF"
_RULE = "#DDE3EA"
_HEAD_FILL = "#EDF1F6"

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

#: What each figure is, for the footer that numbers them.
FIGURE_SERIES: "tuple[tuple[str, str], ...]" = (
    ("00_problem", "the problem, and what we watch"),
    ("01_tradeoff", "how the three are traded off"),
    ("02_cohorts", "the three cohorts"),
)

#: What the left card of each figure holds. Names the stage rather than
#: repeating the page title -- on ``02`` the two were the same words twice.
_SECTION = {
    "00_problem": "What we watch",
    "01_tradeoff": "How we trade off",
    "02_cohorts": "What we get",
}

#: When each cohort is the right one -- the decision the reader actually makes.
_WHEN = {
    "narrow": "stratification is the main worry",
    "intermediate": "both worries matter",
    "full": "power matters most, or a reference is wanted",
}

#: One type scale. Every ``fontsize=`` in this module reads from it; passing a
#: bare number is how several different sizes once ended up on one figure.
_MAIN = {
    "title": 24.0, "section": 17.0, "subtitle": 13.0, "step": 17.0,
    "axis": 13.0, "tick": 11.5, "annot": 13.0, "equation": 15.0,
    "answer": 15.5, "card_name": 17.0, "card_when": 13.5, "card_num": 13.0,
    "foot": 12.5, "note_head": 15.0, "note_sub": 11.5, "note_col": 12.0,
    "note_sym": 11.5, "note_gloss": 11.0,
}

#: Inches. Every length on this figure is one, so a change to one cannot
#: silently mean something different somewhere else.
_SIDE_IN = 0.62
_TITLE_IN = 0.92
_FOOT_IN = 0.55
_CARD_PAD_IN = 0.26
_SECT_IN = 0.40
_SUB_IN = 0.34
_STEP_IN = 0.40
_XLABEL_IN = 0.56
_EQ_LINE_IN = 0.44
_ANSWER_IN = 0.50
_GUTTER_IN = 0.50
_CARD_GAP_IN = 0.44
_BOX_PAD_IN = 0.16

#: A plot needs room for its y-axis label and tick labels, which are drawn
#: outside its own rectangle; at the text padding they hung out of the box and
#: over the card behind it.
_AX_LEFT_IN = 0.80
_AX_RIGHT_IN = 0.20
_AX_TOP_IN = 0.10

#: The notation card, and the two columns of the table inside it.
_NOTE_W_IN = 5.21
_NOTE_SYM_IN = 2.33
_NOTE_DEF_IN = 2.36
_TBL_PAD_IN = 0.14
_TBL_HEAD_IN = 0.34
_TBL_LINE_IN = 0.165
_TBL_ROWPAD_IN = 0.115

#: The last step of ``02`` is three cards rather than a plot, and needs the
#: width for them; the plot beside it takes what is left.
_CARD_W_IN = 5.60


def _note_axis(ax: "plt.Axes") -> None:
    """Turn an axes into a blank sheet with unit coordinates for typeset text."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _panel(fig: Figure, x_in: float, y_in: float, w_in: float, h_in: float,
           *, face: str, edge: str, lw: float = 1.2,
           zorder: float = -5.0) -> None:
    """A rounded card, placed in inches on the page.

    ``mutation_aspect`` keeps the corners round: the figure is more than twice
    as wide as it is tall, so a radius in figure fractions is an ellipse without
    it. The zorder is negative because a figure-level patch and an axes both
    default to zero and the patch is drawn second -- at zero these cards cover
    the plots they are meant to sit behind.
    """
    w_fig, h_fig = fig.get_size_inches()
    fig.patches.append(FancyBboxPatch(
        (x_in / w_fig, y_in / h_fig), w_in / w_fig, h_in / h_fig,
        boxstyle="round,pad=0,rounding_size=0.006",
        mutation_aspect=w_fig / h_fig, transform=fig.transFigure,
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
        clip_on=False))


def _rule(fig: Figure, x0_in: float, x1_in: float, y_in: float,
          *, colour: str = _RULE, lw: float = 1.0) -> None:
    """One horizontal hairline, placed in inches."""
    w_fig, h_fig = fig.get_size_inches()
    fig.lines.append(plt.Line2D([x0_in / w_fig, x1_in / w_fig],
                                [y_in / h_fig] * 2, transform=fig.transFigure,
                                color=colour, linewidth=lw, zorder=-3.5))


def _arrow(fig: Figure, x_in: float, y_in: float) -> None:
    """The join between two steps -- the story line, drawn.

    In the gutter between two cell boxes. Before the boxes existed this sat on
    top of the next panel's y-axis label and two of three arrows were
    unreadable; the box edges now keep the gutter clear.
    """
    w_fig, h_fig = fig.get_size_inches()
    fig.text(x_in / w_fig, y_in / h_fig, "→",
             fontsize=_MAIN["step"] * 1.45, color=_DIM, ha="center",
             va="center")


def _wrap_to(fig: Figure, text: str, avail_in: float, size: float) -> str:
    """Wrap so no line is wider than ``avail_in``, measured rather than counted.

    A character count is not a width -- the same forty characters are a
    different length with a subscript in them -- and an unwrapped lead line is
    what pushed these figures four inches past their own canvas.

    Math spans are atoms. ``textwrap`` split one across two lines and mathtext
    then printed the raw source of both halves into the notation table.
    """
    import re

    words: "list[str]" = []
    glue = False        # whether this piece continues the word before it
    for piece in re.split(r"(\$[^$]*\$)", text):
        if not piece:
            continue
        if piece.startswith("$"):
            if glue and words:
                words[-1] += piece
            else:
                words.append(piece)
            glue = True
        else:
            parts = piece.split()
            if parts and glue and not piece[0].isspace() and words:
                words[-1] += parts.pop(0)
            words.extend(parts)
            glue = not piece[-1].isspace()

    def width_of(line: str) -> float:
        t = fig.text(0.0, -9.0, line, fontsize=size)
        w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
        t.remove()
        return w / fig.dpi

    lines: "list[str]" = []
    for word in words:
        if lines and width_of(f"{lines[-1]} {word}") <= avail_in:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return "\n".join(lines)


def _measure(fig: Figure, text: str, size: float) -> "tuple[float, float]":
    """Width and height of a string in inches, at the dpi it will be drawn at."""
    t = fig.text(0.0, -9.0, text, fontsize=size)
    box = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return box.width / fig.dpi, box.height / fig.dpi


def _notation(fig: Figure, name: str, entries: "Sequence[tuple[str, str]]",
              left_in: float, bottom_in: float, height_in: float) -> None:
    """The symbols this figure uses, as a two-column table.

    The left column carries the symbol *and*, where the formula is the
    definition, the formula: that is what keeps the cells to one headline
    expression each instead of a stack of five. The right column is a short
    noun phrase, not a sentence -- written as sentences the same fourteen rows
    need 6.98in of a 5.91in card.

    Row heights are measured, not assumed: a row whose symbol is a built-up
    fraction is taller than one whose symbol is a letter, and at a fixed height
    the fractions printed over the row beneath them.

    Every symbol here appears on this figure and nowhere else in the series; an
    earlier version carried one general table repeated on all four pages, which
    is why that one was deleted.
    """
    w_fig, h_fig = fig.get_size_inches()
    x0 = left_in + _CARD_PAD_IN
    tbl_w = _NOTE_W_IN - 2.0 * _CARD_PAD_IN
    sym_x = x0 + _TBL_PAD_IN
    def_x = x0 + _NOTE_SYM_IN + _TBL_PAD_IN
    sym_room = _NOTE_SYM_IN - 2.0 * _TBL_PAD_IN
    def_room = _NOTE_DEF_IN - 2.0 * _TBL_PAD_IN
    top = bottom_in + height_in

    y = top - _CARD_PAD_IN
    fig.text(x0 / w_fig, (y - _SECT_IN * 0.62) / h_fig,
             "Notation and definitions", fontsize=_MAIN["note_head"],
             fontweight="bold", color=_BK, ha="left", va="baseline")
    y -= _SECT_IN
    fig.text(x0 / w_fig, (y - _SUB_IN * 0.52) / h_fig,
             "Symbols and formulas used on this figure",
             fontsize=_MAIN["note_sub"], color=_GR, ha="left", va="baseline")
    y -= _SUB_IN

    rows: "list[tuple[str, str, float]]" = []
    for sym, gloss in entries:
        wrapped = _wrap_to(fig, gloss, def_room, _MAIN["note_gloss"])
        sym_w, sym_h = _measure(fig, sym, _MAIN["note_sym"])
        if sym_w > sym_room:
            raise AssertionError(
                f"{name}: notation entry {sym!r} is {sym_w:.2f}in in a "
                f"{sym_room:.2f}in column")
        gloss_h = _TBL_LINE_IN * (wrapped.count("\n") + 1)
        rows.append((sym, wrapped,
                     max(gloss_h, sym_h + 0.05) + _TBL_ROWPAD_IN))

    need = _TBL_HEAD_IN + sum(h for _, _, h in rows)
    have = y - (bottom_in + _CARD_PAD_IN)
    if need > have:
        raise AssertionError(
            f"{name}: the notation table needs {need:.2f}in for "
            f"{len(entries)} rows but the card has {have:.2f}in")

    _panel(fig, x0, y - need, tbl_w, need, face=_CARD, edge=_RULE, lw=1.0,
           zorder=-4.0)
    _panel(fig, x0, y - _TBL_HEAD_IN, tbl_w, _TBL_HEAD_IN, face=_HEAD_FILL,
           edge=_RULE, lw=1.0, zorder=-3.8)
    for label, x in (("Notation", sym_x), ("Definition", def_x)):
        fig.text(x / w_fig, (y - _TBL_HEAD_IN * 0.68) / h_fig, label,
                 fontsize=_MAIN["note_col"], fontweight="bold", color=_BK,
                 ha="left", va="baseline")
    y -= _TBL_HEAD_IN
    body_top = y

    for i, (sym, gloss, row_h) in enumerate(rows):
        mid = y - row_h / 2.0
        fig.text(sym_x / w_fig, mid / h_fig, sym, fontsize=_MAIN["note_sym"],
                 color=_BK, ha="left", va="center")
        fig.text(def_x / w_fig, mid / h_fig, gloss,
                 fontsize=_MAIN["note_gloss"], color=_GR, ha="left",
                 va="center", linespacing=1.30)
        y -= row_h
        if i < len(rows) - 1:
            _rule(fig, x0, x0 + tbl_w, y)

    fig.lines.append(plt.Line2D(
        [(x0 + _NOTE_SYM_IN) / w_fig] * 2, [y / h_fig, body_top / h_fig],
        transform=fig.transFigure, color=_RULE, linewidth=1.0, zorder=-3.5))


def _series_footer(fig: Figure, name: str, x_in: float, y_in: float) -> None:
    """Stamp which of the three this is, so the three read as one argument."""
    w_fig, h_fig = fig.get_size_inches()
    names = [n for n, _ in FIGURE_SERIES]
    i = names.index(name)
    fig.text(x_in / w_fig, y_in / h_fig,
             f"{i + 1} of {len(names)} · {FIGURE_SERIES[i][1]}",
             fontsize=_MAIN["foot"], color=_DIM, ha="left", va="baseline",
             fontstyle="italic")


class _Cell:
    """One step of the argument: a heading, a plot, one equation, an answer.

    Filled in by the figure that owns it; :func:`_story_row` places all of them
    and is the only thing that knows where anything goes. The equation is the
    headline one; the derivation behind it belongs in the notation table.
    """

    def __init__(self, head: str, *, colour: str = _BK, plot: bool = True,
                 width: float | None = None) -> None:
        self.head = head
        self.colour = colour
        self.plot = plot
        self.width = width
        self.equations: "list[str]" = []
        self.answer: "tuple[str, str] | None" = None
        self.ax: "plt.Axes | None" = None

    def says(self, *lines: str) -> "_Cell":
        self.equations.extend(lines)
        return self

    def concludes(self, text: str, colour: str | None = None) -> "_Cell":
        self.answer = (text, colour or self.colour)
        return self


def _story_row(
    name: str,
    title: str,
    lead: str,
    cells: "Sequence[_Cell]",
    notation: "Sequence[tuple[str, str]]",
    *,
    eq_lines: int = 1,
) -> Figure:
    """Lay out one figure of the argument: two cards on one page.

    Every figure in this stage goes through here. That is what makes the three
    read as one thing -- four figures each with its own layout was what made an
    earlier version of this argument impossible to follow.

    Left card: a row of cell boxes joined by arrows, each a heading, a plot,
    the headline equation and the answer it reaches. Right card: the notation
    table. The canvas is the same size on all three, and the plot band takes
    whatever the equations leave, so nothing has to be rescaled between pages.

    Each cell is measured against its own box; a heading, an equation or an
    answer that would run past it fails the build rather than being drawn over
    its neighbour.
    """
    width, height = FIGURE_SIZE
    n = len(cells)
    body_h = height - _TITLE_IN - _FOOT_IN
    body_top = height - _TITLE_IN
    left_w = width - 2.0 * _SIDE_IN - _NOTE_W_IN - _CARD_GAP_IN
    usable = left_w - 2.0 * _CARD_PAD_IN
    fixed = sum(c.width for c in cells if c.width is not None)
    flex = [c for c in cells if c.width is None]
    w_flex = (usable - (n - 1) * _GUTTER_IN - fixed) / max(len(flex), 1)
    widths = [c.width if c.width is not None else w_flex for c in cells]
    lefts = [_SIDE_IN + _CARD_PAD_IN + sum(widths[:i]) + i * _GUTTER_IN
             for i in range(n)]

    box_h = body_h - 2.0 * _CARD_PAD_IN - _SECT_IN - _SUB_IN
    eq_in = _EQ_LINE_IN * eq_lines
    plot_in = (box_h - 2.0 * _BOX_PAD_IN - _STEP_IN - _XLABEL_IN - eq_in
               - _ANSWER_IN)
    if plot_in < 1.8:
        raise AssertionError(
            f"{name}: {eq_lines} equation lines leave only {plot_in:.2f}in "
            f"for the plots on a {height:.1f}in page")

    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(_CARD)
    box_bottom = _FOOT_IN + _CARD_PAD_IN
    _panel(fig, _SIDE_IN, _FOOT_IN, left_w, body_h, face=_PAGE, edge=_RULE)
    _panel(fig, width - _SIDE_IN - _NOTE_W_IN, _FOOT_IN, _NOTE_W_IN, body_h,
           face=_PAGE, edge=_RULE)

    def fits(t, i: int, what: str) -> None:
        room = widths[i] - 2.0 * _BOX_PAD_IN
        used = t.get_window_extent(
            renderer=fig.canvas.get_renderer()).width / fig.dpi
        if used > room + 0.04:
            raise AssertionError(
                f"{name} cell {i + 1}: {what} is {used:.2f}in wide in a "
                f"{room:.2f}in box — {t.get_text()[:44]!r}")

    # the page: title, its rule, and the footer
    fig.text(_SIDE_IN / width, (height - _TITLE_IN * 0.60) / height, title,
             fontsize=_MAIN["title"], fontweight="bold", color=_BK, ha="left",
             va="baseline")
    _rule(fig, _SIDE_IN, width - _SIDE_IN, height - _TITLE_IN + 0.16,
          colour=_HAIR, lw=1.3)
    _series_footer(fig, name, _SIDE_IN, _FOOT_IN * 0.46)

    # the left card: what this page is, then the row of steps
    y = body_top - _CARD_PAD_IN
    fig.text((_SIDE_IN + _CARD_PAD_IN) / width, (y - _SECT_IN * 0.62) / height,
             _SECTION[name], fontsize=_MAIN["section"], fontweight="bold",
             color=_BK, ha="left", va="baseline")
    y -= _SECT_IN
    sub = _wrap_to(fig, lead, usable, _MAIN["subtitle"])
    if "\n" in sub:
        raise AssertionError(
            f"{name}: the subtitle wraps to {sub.count(chr(10)) + 1} lines in "
            f"{usable:.2f}in; it is meant to be one")
    fig.text((_SIDE_IN + _CARD_PAD_IN) / width, (y - _SUB_IN * 0.52) / height,
             sub, fontsize=_MAIN["subtitle"], color=_GR, ha="left",
             va="baseline")

    y_ans = box_bottom + _BOX_PAD_IN
    y_eq = y_ans + _ANSWER_IN
    y_plot = y_eq + eq_in + _XLABEL_IN
    y_step = y_plot + plot_in + _STEP_IN * 0.28

    for i, c in enumerate(cells):
        _panel(fig, lefts[i], box_bottom, widths[i], box_h, face=_CARD,
               edge=c.colour if c.colour != _BK else _RULE, lw=1.3)
        c.ax = fig.add_axes((
            (lefts[i] + (_BOX_PAD_IN if not c.plot else _AX_LEFT_IN)) / width,
            y_plot / height,
            (widths[i] - (2.0 * _BOX_PAD_IN if not c.plot
                          else _AX_LEFT_IN + _AX_RIGHT_IN)) / width,
            (plot_in - (0.0 if not c.plot else _AX_TOP_IN)) / height))
        if not c.plot:
            _note_axis(c.ax)
        t = fig.text((lefts[i] + _BOX_PAD_IN) / width, y_step / height,
                     f"{i + 1} ·  {c.head}", fontsize=_MAIN["step"],
                     fontweight="bold", ha="left", va="bottom", color=c.colour)
        fits(t, i, "a heading")
        if i:
            _arrow(fig, lefts[i] - _GUTTER_IN / 2.0, box_bottom + box_h / 2.0)

    def draw_text() -> None:
        for i, c in enumerate(cells):
            if len(c.equations) > eq_lines:
                raise AssertionError(
                    f"{name} cell {i + 1}: {len(c.equations)} equations in a "
                    f"{eq_lines}-line band")
            for j, tex in enumerate(c.equations):
                t = fig.text((lefts[i] + _BOX_PAD_IN) / width,
                             (y_eq + eq_in - _EQ_LINE_IN * (j + 0.5)) / height,
                             tex, fontsize=_MAIN["equation"], ha="left",
                             va="center", color=_BK)
                fits(t, i, "an equation")
            if c.answer is not None:
                text, colour = c.answer
                t = fig.text((lefts[i] + _BOX_PAD_IN) / width,
                             (y_ans + 0.08) / height, text,
                             fontsize=_MAIN["answer"], fontweight="bold",
                             ha="left", va="bottom", color=colour)
                fits(t, i, "the answer")

    _notation(fig, name, notation, width - _SIDE_IN - _NOTE_W_IN, _FOOT_IN,
              body_h)
    fig._draw_story_text = draw_text  # type: ignore[attr-defined]
    return fig


def _walk(decision_table: pd.DataFrame, rgv_column: str):
    """The three quantities every figure here is about, in walk order.

    Plus the two the third is built from: the raw squared distance and the
    floor subtracted off it, which ``00_problem`` draws so the subtraction is
    something the reader sees rather than something they are told.
    """
    return (
        decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False),
        decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False),
        decision_table[rgv_column].to_numpy(dtype=float, copy=False),
        decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float,
                                                                copy=False),
        decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False),
        decision_table["Mainland_CaseCtrl_Mahalanobis"].to_numpy(
            dtype=float, copy=False) ** 2,
        decision_table["Mainland_CaseCtrl_Noise_Floor"].to_numpy(dtype=float,
                                                                copy=False),
    )


def _norm(a: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; zeros for a degenerate range."""
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if np.isfinite(hi - lo) and hi > lo else np.zeros_like(a)


def _reversals(a: np.ndarray) -> int:
    """How many times a series changes direction; zero when monotone."""
    v = a[np.isfinite(a)]
    signs = np.sign(np.diff(v))
    signs = signs[signs != 0]
    return int(np.sum(np.diff(signs) != 0)) if signs.size >= 2 else 0


def _grid(ax: "plt.Axes") -> None:
    ax.grid(True, alpha=0.30, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=_MAIN["tick"])


def plot_problem(
    *,
    decision_table: pd.DataFrame,
    rank_table: pd.DataFrame,
    rgv_column: str,
    mainland_axes: "Sequence[str]",
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """1 of 3 — the problem, and the three quantities we watch along the walk."""
    rank, neff, het, sep, pval, d2, floor = _walk(decision_table, rgv_column)
    n1 = decision_table[f"{case_label}_Count"].to_numpy(dtype=float, copy=False)
    n2 = decision_table[f"{control_label}_Count"].to_numpy(dtype=float, copy=False)
    d = len(mainland_axes)
    sig = np.isfinite(pval) & (pval < 0.05)
    flips = _reversals(sep)

    power = _Cell("statistical power").says(
        r"$N_{\mathrm{eff},k} = 4N_{\mathrm{case}}N_{\mathrm{ctrl}}"
        r"/N_{\mathrm{tot}}$").concludes(
        f"rises to {neff[-1]:,.0f} effective")
    spread = _Cell("residual stratification").says(
        rf"$H_k = \left|\Sigma_k\right|^{{1/2d}},\; d = {d}$").concludes(
        "rises with $k$ too")
    shift = _Cell(f"{case_label}$-${control_label} shift", colour=_THIRD).says(
        r"$s_k = \hat{D}^2_k - b_k$").concludes(
        f"reverses {flips}×, not monotone", _THIRD)

    # The left column carries the formula wherever the formula is the
    # definition. That is what lets each cell above keep to one headline
    # expression: the derivation behind s_k, and the F test the significance
    # marks come from, are here rather than stacked under the third plot.
    notation = [
        ("$k$", "Cut index; the top $k$ of "
                f"{int(rank.max())} ranked components."),
        (f"$d = {d}$", "Mainland PCA axes retained."),
        (r"$N_{\mathrm{case}},\,N_{\mathrm{ctrl}},\,N_{\mathrm{tot}}$",
         "Arm sizes and their sum."),
        (r"$N_{\mathrm{eff},k}$", "Effective sample size."),
        (r"$\Sigma_k,\;\lambda_i$",
         "Retained-set covariance and eigenvalues."),
        ("$H_k$", "Root generalized variance."),
        (r"$S,\;\Delta\bar{x}$",
         "Pooled within-arm covariance; centroid gap."),
        (r"$\hat{D}^2_k = \Delta\bar{x}^{\mathsf{T}}S^{-1}\Delta\bar{x}$",
         "Squared Mahalanobis gap."),
        (r"$b_k = d\left(\frac{1}{N_{\mathrm{case}}}"
         r"+\frac{1}{N_{\mathrm{ctrl}}}\right)$",
         "Sampling floor under that gap."),
        (r"$\mathbb{E}[\hat{D}^2_k] = D^2_k + b_k$",
         "Why the floor is subtracted."),
        (r"$T^2 = \hat{D}^2_k N_{\mathrm{case}}N_{\mathrm{ctrl}}"
         r"/N_{\mathrm{tot}}$", "Hotelling's two-sample statistic."),
        (r"$F = \frac{\nu-d+1}{d\nu}T^2 \sim F_{d,\nu-d+1}$",
         r"Exact $F$ transform; $\nu = N_{\mathrm{tot}}-2$."),
        (r"$P_k = \Pr(F_{d,\nu-d+1} > F)$",
         "Tail probability under one population."),
    ]

    fig = _story_row(
        "00_problem",
        "The problem — a wider set has more power and more residual structure",
        f"{int(rank.max())} components ranked by {case_label}/{control_label} "
        f"ratio; cut $k$ keeps the top $k$. No cut is best on all three.",
        (power, spread, shift), notation, eq_lines=1)

    ax = power.ax
    ax.fill_between(rank, neff, n1 + n2, color=_MUTED, alpha=0.75, zorder=2,
                    label="lost to imbalance")
    ax.plot(rank, n1 + n2, "-", color=_GR, linewidth=1.6, zorder=3,
            label=r"$N_{\mathrm{tot}}$")
    ax.plot(rank, neff, "-o", color=_BK, markersize=4.0, linewidth=1.7,
            markerfacecolor="white", markeredgewidth=1.0, zorder=4,
            label=r"$N_{\mathrm{eff},k}$")
    ax.set_xticks(rank[::4])
    ax.set_xlabel("cut $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel("samples", fontsize=_MAIN["axis"])
    ax.legend(loc="upper left", fontsize=_MAIN["tick"], frameon=True,
              framealpha=0.95, edgecolor=_HAIR)
    _grid(ax)

    ax = spread.ax
    ax.plot(rank, het, "-s", color=_BK, markersize=4.4, linewidth=1.7,
            markerfacecolor="white", markeredgewidth=1.0)
    ax.set_xticks(rank[::4])
    ax.set_xlabel("cut $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel(r"$H_k$", fontsize=_MAIN["axis"])
    _grid(ax)

    # Three curves, not one: the floor falls 6.6x along this walk, so a raw
    # distance would fall across it for arithmetic reasons alone. Drawing the
    # floor is what makes the subtraction in s_k an observation.
    ax = shift.ax
    ax.axhline(0.0, color=_DIM, linewidth=0.9, linestyle=":")
    ax.fill_between(rank, 0.0, floor, color=_MUTED, alpha=0.7, linewidth=0,
                    zorder=1)
    ax.plot(rank, d2, "-", color=_GR, linewidth=1.5, zorder=3,
            label=r"$\hat{D}^2_k$  raw")
    ax.plot(rank, floor, "--", color=_DIM, linewidth=1.5, zorder=3,
            label=r"$b_k$  sampling floor")
    ax.plot(rank, sep, "-", color=_THIRD, linewidth=2.0, zorder=4,
            label=r"$s_k = \hat{D}^2_k - b_k$")
    ax.plot(rank[sig], sep[sig], "^", color=_THIRD, markersize=6.0, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0, linestyle="none",
            label=rf"$P < 0.05$ ({int(sig.sum())} of {len(pval)})")
    ax.plot(rank[~sig], sep[~sig], "^", color="white", markersize=6.0, zorder=5,
            markeredgecolor=_THIRD, markeredgewidth=1.4, linestyle="none")
    ax.set_xticks(rank[::4])
    ax.set_ylim(top=float(np.nanmax(d2)) * 1.62)
    ax.set_xlabel("cut $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel("squared distance", fontsize=_MAIN["axis"])
    ax.legend(loc="upper right", fontsize=_MAIN["tick"], frameon=True,
              framealpha=0.95, edgecolor=_HAIR)
    _grid(ax)

    fig._draw_story_text()
    return fig


def plot_tradeoff(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rgv_column: str,
    objective_spaces: "Mapping[str, Any]",
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    blend_weight: float,
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """2 of 3 — how the three quantities are traded off against each other."""
    rank, neff, het, sep, *_ = _walk(decision_table, rgv_column)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_nar = int(rows["narrow"]["Resolved_Rank"])
    k_int = int(rows["intermediate"]["Resolved_Rank"])
    gamma = float(rows["narrow"]["Exchange_Rate"])
    excess = (neff - neff[0]) - gamma * (het - het[0])
    blended = np.asarray(objective_spaces["intermediate"].structure, dtype=float)
    power = np.asarray(objective_spaces["intermediate"].power, dtype=float)
    dist = np.sqrt(blended ** 2 + (1.0 - power) ** 2)
    flips = _reversals(sep)

    # A figure that disagreed with the record it illustrates would be worse than
    # no figure, so the two peaks are checked before anything is drawn.
    for what, got, want in (("narrow", int(rank[int(np.argmax(excess))]), k_nar),
                            ("intermediate", int(rank[int(np.argmin(dist))]), k_int)):
        if got != want:
            raise AssertionError(
                f"the {what} criterion drawn here peaks at k = {got}, but "
                f"cut_record.tsv records k = {want}")

    nar, inter = _EDGE["narrow"], _EDGE["intermediate"]
    price = _Cell("price the walk", colour=nar).says(
        r"$E_k = (N_{\mathrm{eff},k} - N_{\mathrm{eff},1})"
        r" - \gamma(H_k - H_1)$").concludes(
        rf"$\Rightarrow$  narrow, $k$ = {k_nar}")
    blend = _Cell(rf"blend at $w = {blend_weight:g}$", colour=inter).says(
        r"$\tilde{u}_k = \mathrm{minmax}(w\tilde{H}_k + (1-w)\tilde{s}_k)$",
        r"$k^{*} = \arg\min_k \sqrt{\tilde{u}_k^{2} "
        r"+ (1-\tilde{N}_k)^{2}}$").concludes(
        rf"$\Rightarrow$  intermediate, $k$ = {k_int}")
    weight = _Cell(r"$w$ does not decide it", colour=inter).says(
        r"$w \geq \frac{1}{2} \Longleftrightarrow w \geq 1 - w$").concludes(
        rf"$k^{{*}}$ = {k_int} on a plateau of $w$")

    notation = [
        ("$k$", f"Cut index; the top $k$ of {int(rank.max())} components."),
        (r"$N_{\mathrm{eff},k},\;H_k,\;s_k$",
         "The three quantities of figure 1 of 3."),
        (r"$\gamma = \frac{N_{\mathrm{eff},K}-N_{\mathrm{eff},1}}{H_K-H_1}$",
         "Average exchange rate over the walk."),
        ("$E_k$", "Excess over that rate, end to end."),
        (r"$k_{\mathrm{narrow}} = \arg\max_k E_k$",
         "Cut that beats the rate by most."),
        (r"$\tilde{x}$", r"Min-max onto $[0,1]$ over the 17 cuts."),
        ("$w$", r"Weight on $\tilde{H}$; $1-w$ on $\tilde{s}$."),
        (r"$\tilde{u}_k(w)$", "The blend, rescaled again."),
        (r"$\tilde{N}_k$", "Effective size on the same scale."),
        (r"$k^{*}(w)$", "Cut nearest the ideal corner."),
    ]

    fig = _story_row(
        "01_tradeoff",
        "The trade-off — one rate prices two of them, and nothing prices the third",
        f"Power and stratification move together and share one rate; the "
        f"{case_label}$-${control_label} shift reverses {flips} times and has "
        f"none. Hence two criteria, not one.",
        (price, blend, weight), notation, eq_lines=2)

    ax = price.ax
    i_nar = int(np.argmin(np.abs(rank - k_nar)))
    ax.plot([het[0], het[-1]], [neff[0], neff[-1]], "--", color=nar,
            linewidth=1.7, zorder=2, label=rf"the average rate $\gamma$")
    ax.plot(het, neff, "-o", color=_BK, markersize=4.2, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.0, zorder=3,
            label="the walk")
    ax.vlines(het[i_nar], neff[0] + gamma * (het[i_nar] - het[0]), neff[i_nar],
              color=nar, linewidth=5.0, alpha=0.8, zorder=4)
    ax.plot([het[i_nar]], [neff[i_nar]], _MARK["narrow"], color=nar,
            markersize=13.0, markeredgecolor="white", markeredgewidth=1.5,
            zorder=6)
    ax.annotate(rf"$E_k$ at $k$ = {k_nar}", xy=(het[i_nar], neff[i_nar]),
                xytext=(9, -4), textcoords="offset points",
                fontsize=_MAIN["annot"], fontweight="bold", color=nar,
                ha="left", va="top", zorder=7)
    ax.set_xlabel(r"$H_k$   $\rightarrow$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel(r"$N_{\mathrm{eff},k}$   $\rightarrow$", fontsize=_MAIN["axis"])
    ax.legend(loc="lower right", fontsize=_MAIN["tick"], frameon=True,
              framealpha=0.95, edgecolor=_HAIR)
    ax.margins(x=0.12, y=0.14)
    _grid(ax)

    ax = blend.ax
    i_int = int(np.argmin(np.abs(rank - k_int)))
    ax.plot(blended, power, "o", color=_GR, markersize=6.0, alpha=0.8,
            markerfacecolor="white", markeredgewidth=1.2)
    ax.plot([0.0, blended[i_int]], [1.0, power[i_int]], "--", color=inter,
            linewidth=2.0)
    ax.plot([0.0], [1.0], "*", color=inter, markersize=19.0,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.plot([blended[i_int]], [power[i_int]], _MARK["intermediate"],
            color=inter, markersize=13.0, markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)
    ax.annotate("ideal", xy=(0.0, 1.0), xytext=(11, -3),
                textcoords="offset points", fontsize=_MAIN["annot"],
                color=inter, ha="left", va="top")
    ax.set_xlim(-0.08, 1.10); ax.set_ylim(-0.10, 1.16)
    ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(rf"$\tilde{{u}}_k({blend_weight:g})$   $\rightarrow$",
                  fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel(r"$\tilde{N}_k$   $\rightarrow$", fontsize=_MAIN["axis"])
    _grid(ax)

    ax = weight.ax
    won = weight_winner[weight_winner > 0]
    on = weight_grid[weight_winner == k_int]
    lo, hi = (float(on.min()), float(on.max())) if on.size else (np.nan,) * 2
    ax.fill_between([0.0, 0.5], -100, 100, color=_BARRED, alpha=0.5,
                    linewidth=0)
    ax.text(0.25, 0.12, "not admissible\n$w < \\frac{1}{2}$",
            transform=ax.transAxes, fontsize=_MAIN["annot"], color=_GR,
            ha="center", va="bottom", zorder=2)
    ax.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK,
            linewidth=2.0)
    if on.size:
        ax.plot([lo, hi], [k_int, k_int], color=inter, linewidth=6.0,
                solid_capstyle="butt", alpha=0.85)
    ax.axvline(blend_weight, color=inter, linewidth=1.5, linestyle="-.")
    ax.plot([blend_weight], [k_int], _MARK["intermediate"], color=inter,
            markersize=11.0, markeredgecolor="white", markeredgewidth=1.3,
            zorder=5)
    ax.annotate(f"panel 2 is\nthe slice at $w$ = {blend_weight:g}",
                xy=(blend_weight, k_int), xycoords="data",
                xytext=(0.56, 0.88), textcoords="axes fraction",
                fontsize=_MAIN["annot"], fontweight="bold", color=inter,
                ha="left", va="top", zorder=6,
                arrowprops=dict(arrowstyle="->", color=inter, linewidth=1.4,
                                shrinkB=7))
    ax.set_xlim(0.0, 1.0); ax.set_ylim(int(won.min()) - 1, int(won.max()) + 1)
    ax.set_yticks(sorted({int(v) for v in np.unique(won)}))
    ax.set_xlabel("$w$   weight on $H$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel(r"winning $k^{*}$", fontsize=_MAIN["axis"])
    _grid(ax)
    weight.answer = (rf"$k^{{*}}$ = {k_int} on $w \in [{lo:.2f},\ {hi:.2f}]$",
                     inter)

    fig._draw_story_text()
    return fig


def plot_cohorts(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rgv_column: str,
    mode: str,
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """3 of 3 — the three cohorts, and when to use each."""
    rank, neff, het, *_ = _walk(decision_table, rgv_column)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    order = [n for n in CUT_ORDER if n in k_of]
    at = {n: int(np.argmin(np.abs(rank - k_of[n]))) for n in order}

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        if col not in r.columns or r.empty:
            return float("nan")
        return float(to_numeric_array(r[col])[0])

    notation = [
        ("$k$", f"Cut index; the top $k$ of {int(rank.max())} components."),
        (r"$N_{\mathrm{eff},k}$", "Effective sample size."),
        ("$H_k$", "Residual spread on 4 mainland axes."),
        (r"$\subset$", "Nesting of the three cohorts."),
        ("narrow", r"Stops at $\arg\max_k E_k$."),
        ("intermediate", r"Stops at $k^{*}$, nearest the ideal corner."),
        ("full", "Every component of the major cluster."),
    ]

    where = _Cell("where the three stop")
    which = _Cell("which one to use", plot=False, width=_CARD_W_IN).concludes(
        r"narrow $\subset$ intermediate $\subset$ full")

    fig = _story_row(
        "02_cohorts", "The three cohorts",
        "Two criteria stop the walk in two places; taking every component is "
        f"the third. They are nested. Cuts resolved in mode: {mode}.",
        (where, which), notation, eq_lines=0)

    ax = where.ax
    ax.plot(het, neff, "-o", color=_GR, markersize=4.6, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.0, zorder=3)
    for name in order:
        i = at[name]
        ax.plot([het[i]], [neff[i]], _MARK[name], color=_EDGE[name],
                markersize=16.0, markeredgecolor="white", markeredgewidth=1.9,
                zorder=6)
        ax.annotate(f"{name}\n$k$ = {k_of[name]}", xy=(het[i], neff[i]),
                    xytext={"narrow": (-58, -30), "intermediate": (2, 32),
                            "full": (-16, -34)}[name],
                    textcoords="offset points", fontsize=_MAIN["annot"],
                    fontweight="bold", color=_EDGE[name], ha="center",
                    va="center", zorder=7)
    ax.set_xlabel(r"residual spread $H_k$   $\rightarrow$",
                  fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel(r"$N_{\mathrm{eff},k}$   $\rightarrow$",
                  fontsize=_MAIN["axis"])
    ax.margins(x=0.16, y=0.14)
    _grid(ax)

    ax = which.ax
    card_h = 1.0 / len(order)
    for i, name in enumerate(order):
        top = 1.0 - i * card_h
        ax.add_patch(FancyBboxPatch(
            (0.0, top - card_h + 0.030), 1.0, card_h - 0.048,
            boxstyle="round,pad=0.004", facecolor=_TINT[name],
            edgecolor=_EDGE[name], linewidth=1.4, alpha=0.85, zorder=1))
        ax.plot([0.018, 0.018], [top - card_h + 0.052, top - 0.020],
                color=_EDGE[name], linewidth=5.5, solid_capstyle="butt",
                zorder=3)
        ax.text(0.055, top - 0.062, name, fontsize=_MAIN["card_name"],
                fontweight="bold", ha="left", va="top", color=_EDGE[name],
                zorder=3)
        ax.text(0.982, top - 0.062, f"$k$ = {k_of[name]}",
                fontsize=_MAIN["card_name"], fontweight="bold", ha="right",
                va="top", color=_EDGE[name], zorder=3)
        ax.text(0.055, top - 0.062 - card_h * 0.29, _WHEN[name],
                fontsize=_MAIN["card_when"], ha="left", va="top", color=_BK,
                zorder=3)
        ax.text(0.055, top - 0.062 - card_h * 0.56,
                f"{val(name, f'{case_label}_Count'):,.0f}"
                f" + {val(name, f'{control_label}_Count'):,.0f}"
                f" = {val(name, 'Total_Count'):,.0f}"
                f"   effective {val(name, 'GWAS_Neff'):,.0f}",
                fontsize=_MAIN["card_num"], ha="left", va="top", color=_GR,
                zorder=3)

    fig._draw_story_text()
    return fig
