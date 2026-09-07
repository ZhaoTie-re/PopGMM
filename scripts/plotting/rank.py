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

#: Canvas width. The height follows from the content and is computed below.
FIGURE_SIZE: "tuple[float, float]" = (19.0, 13.0)

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

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

#: What each figure is, for the footer that numbers them.
FIGURE_SERIES: "tuple[tuple[str, str], ...]" = (
    ("00_problem", "the problem, and what we watch"),
    ("01_tradeoff", "how the three are traded off"),
    ("02_cohorts", "the three cohorts"),
)

#: When each cohort is the right one -- the decision the reader actually makes.
_WHEN = {
    "narrow": "stratification is the main worry",
    "intermediate": "both worries matter",
    "full": "power matters most, or a reference is wanted",
}

#: One type scale. Every ``fontsize=`` in this module reads from it; passing a
#: bare number is how several different sizes once ended up on one figure.
_MAIN = {
    "suptitle": 26.0, "lead": 15.0, "step": 17.0, "axis": 13.5, "tick": 12.0,
    "annot": 13.5, "equation": 15.0, "answer": 15.5, "card_name": 17.0,
    "card_when": 13.5, "card_num": 13.0, "foot": 13.0,
}

#: Inches. Every length on this figure is one, so a change to one cannot
#: silently mean something different somewhere else.
_SIDE_IN = 0.62
_TITLE_IN = 1.05
_LEAD_IN = 0.52
_STEP_IN = 0.46
_PLOT_IN = 3.05
_XLABEL_IN = 0.62
_EQ_IN = 1.66
_ANSWER_IN = 0.52
_FOOT_IN = 0.85
_GUTTER_IN = 0.72

#: The fourth step is three cards rather than a plot, and needs the width for
#: them; the three plots share what is left.
_CARD_W_IN = 5.20


def _note_axis(ax: "plt.Axes") -> None:
    """Turn an axes into a blank sheet with unit coordinates for typeset text."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _arrow(fig: Figure, x: float, y: float) -> None:
    """The join between two steps -- the story line, drawn.

    On the heading row rather than beside the plots: level with the plots it sat
    on top of the next panel's y-axis label, and two of the three arrows were
    unreadable.
    """
    fig.text(x, y, "→", fontsize=_MAIN["step"] * 1.35, color=_DIM,
             ha="center", va="bottom")



def _wrap_to(fig: Figure, text: str, avail_in: float, size: float) -> str:
    """Wrap so no line is wider than ``avail_in``, measured rather than counted.

    A character count is not a width -- the same forty characters are a
    different length with a subscript in them -- and an unwrapped lead line is
    what pushed these figures four inches past their own canvas.
    """
    import textwrap

    best = textwrap.wrap(text, width=40)
    for n in range(40, 400, 6):
        lines = textwrap.wrap(text, width=n)
        t = fig.text(0.0, -9.0, "\n".join(lines), fontsize=size)
        box = t.get_window_extent(renderer=fig.canvas.get_renderer())
        t.remove()
        if box.width / fig.dpi > avail_in:
            break
        best = lines
    return "\n".join(best)


def _series_footer(fig: Figure, name: str, x0: float, y: float) -> None:
    """Stamp which of the three this is, so the three read as one argument."""
    names = [n for n, _ in FIGURE_SERIES]
    i = names.index(name)
    fig.text(x0, y,
             f"{i + 1} of {len(names)} · {FIGURE_SERIES[i][1]}",
             fontsize=_MAIN["foot"], color=_DIM, ha="left", va="bottom",
             fontstyle="italic")


class _Cell:
    """One step of the argument: a heading, a plot, equations, an answer.

    Filled in by the figure that owns it; :func:`_story_row` places all of them
    and is the only thing that knows where anything goes.
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
    *,
    plot_in: float = _PLOT_IN,
    eq_lines: int = 2,
) -> Figure:
    """Lay out one figure of the argument: a row of cells joined by arrows.

    Every figure in this stage goes through here. That is what makes the three
    read as one thing -- four figures each with its own layout was what made an
    earlier version of this argument impossible to follow.

    Each cell is measured against its own column; an equation or an answer that
    would run into the next cell fails the build rather than being drawn over
    its neighbour.
    """
    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    n = len(cells)
    fixed = sum(c.width for c in cells if c.width is not None)
    flex = [c for c in cells if c.width is None]
    w_flex = (inner - (n - 1) * _GUTTER_IN - fixed) / max(len(flex), 1)
    widths = [c.width if c.width is not None else w_flex for c in cells]

    eq_in = 0.28 + 0.46 * eq_lines
    probe = plt.figure(figsize=(width, 3.0))
    lead_lines = _wrap_to(probe, lead, inner, _MAIN["lead"]).count("\n") + 1
    plt.close(probe)
    lead_in = _LEAD_IN + 0.30 * (lead_lines - 1)
    height = (_TITLE_IN + lead_in + _STEP_IN + plot_in + _XLABEL_IN
              + eq_in + _ANSWER_IN + _FOOT_IN)

    fig = plt.figure(figsize=(width, height))
    x0 = _SIDE_IN / width
    lefts = [_SIDE_IN + sum(widths[:i]) + i * _GUTTER_IN for i in range(n)]
    y_ans = _FOOT_IN / height
    y_eq = y_ans + _ANSWER_IN / height
    y_plot = y_eq + (eq_in + _XLABEL_IN) / height
    y_step = y_plot + (plot_in + _STEP_IN * 0.34) / height

    def fits(t, i: int, what: str) -> None:
        box = t.get_window_extent(renderer=fig.canvas.get_renderer())
        used = box.width / fig.dpi
        if used > widths[i] + 0.04:
            raise AssertionError(
                f"{name} cell {i + 1}: {what} is {used:.2f}in wide in a "
                f"{widths[i]:.2f}in column — {t.get_text()[:44]!r}")

    for i, c in enumerate(cells):
        c.ax = fig.add_axes((lefts[i] / width, y_plot, widths[i] / width,
                             plot_in / height))
        if not c.plot:
            _note_axis(c.ax)
        fig.text(lefts[i] / width, y_step, f"{i + 1} ·  {c.head}",
                 fontsize=_MAIN["step"], fontweight="bold", ha="left",
                 va="bottom", color=c.colour)
        if i:
            _arrow(fig, (lefts[i] - _GUTTER_IN / 2.0) / width, y_step)

    def draw_text() -> None:
        for i, c in enumerate(cells):
            for j, tex in enumerate(c.equations):
                t = fig.text(lefts[i] / width,
                             y_eq + (eq_in - 0.28 - j * 0.46) / height, tex,
                             fontsize=_MAIN["equation"], ha="left",
                             va="center", color=_BK)
                fits(t, i, "an equation")
            if c.answer is not None:
                text, colour = c.answer
                t = fig.text(lefts[i] / width, y_ans + 0.10 / height, text,
                             fontsize=_MAIN["answer"], fontweight="bold",
                             ha="left", va="bottom", color=colour)
                fits(t, i, "the answer")

    fig.suptitle(title, fontsize=_MAIN["suptitle"], fontweight="bold",
                 y=1.0 - 0.46 / height)
    fig.text(x0, 1.0 - (_TITLE_IN + 0.16) / height,
             _wrap_to(fig, lead, inner, _MAIN["lead"]),
             fontsize=_MAIN["lead"], color=_GR, ha="left", va="top",
             linespacing=1.4)
    _series_footer(fig, name, x0, 0.30 / height)
    fig._draw_story_text = draw_text  # type: ignore[attr-defined]
    return fig

def _walk(decision_table: pd.DataFrame, rgv_column: str):
    """The three quantities every figure here is about, in walk order."""
    return (
        decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False),
        decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False),
        decision_table[rgv_column].to_numpy(dtype=float, copy=False),
        decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float,
                                                                copy=False),
        decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False),
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
    rank, neff, het, sep, pval = _walk(decision_table, rgv_column)
    n1 = decision_table[f"{case_label}_Count"].to_numpy(dtype=float, copy=False)
    n2 = decision_table[f"{control_label}_Count"].to_numpy(dtype=float, copy=False)
    d = len(mainland_axes)
    sig = np.isfinite(pval) & (pval < 0.05)
    flips = _reversals(sep)

    power = _Cell("statistical power").says(
        r"$N_{\mathrm{eff},k} = \dfrac{4\,N_{\mathrm{case}}N_{\mathrm{ctrl}}}"
        r"{N_{\mathrm{case}} + N_{\mathrm{ctrl}}}$").concludes(
        f"rises with $k$ — {n1[-1] + n2[-1]:,.0f} samples, "
        f"{neff[-1]:,.0f} effective")
    spread = _Cell("residual stratification").says(
        rf"$H_k = \left|\Sigma_k\right|^{{1/2d}}, \quad d = {d}$").concludes(
        "rises with $k$ too")
    shift = _Cell(f"{case_label}/{control_label} shift", colour=_THIRD).says(
        r"$s_k = \hat{D}^2_k - d\left(\frac{1}{N_{\mathrm{case}}}"
        r"+\frac{1}{N_{\mathrm{ctrl}}}\right)$").concludes(
        f"reverses {flips}×; {int(sig.sum())} of {len(pval)} real",
        _THIRD)

    fig = _story_row(
        "00_problem",
        "The problem — a wider set has more power and more residual structure",
        f"The major cluster splits into {int(rank.max())} components. Order them "
        f"by {case_label}/{control_label} ratio and let cut $k$ keep the top $k$: "
        f"one nested set per $k$. Three quantities move along that walk, and no "
        f"cut is best on all three.",
        (power, spread, shift), eq_lines=2)

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

    ax = shift.ax
    ax.axhline(0.0, color=_DIM, linewidth=0.9, linestyle=":")
    ax.plot(rank, sep, "-", color=_THIRD, linewidth=1.8, zorder=3)
    ax.plot(rank[sig], sep[sig], "^", color=_THIRD, markersize=6.0, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0,
            label=rf"$P < 0.05$ ({int(sig.sum())})")
    ax.plot(rank[~sig], sep[~sig], "^", color="white", markersize=6.0, zorder=4,
            markeredgecolor=_THIRD, markeredgewidth=1.4,
            label=r"$P \geq 0.05$")
    ax.set_xticks(rank[::4])
    ax.set_xlabel("cut $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel(r"$s_k$", fontsize=_MAIN["axis"])
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
    rank, neff, het, sep, _ = _walk(decision_table, rgv_column)
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
        r"$\gamma = \dfrac{N_{\mathrm{eff},K} - N_{\mathrm{eff},1}}{H_K - H_1}$"
        rf"$\;=\;{gamma:,.0f}$",
        r"$E_k = (N_{\mathrm{eff},k} - N_{\mathrm{eff},1}) - \gamma(H_k - H_1)$",
        r"$k_{\mathrm{narrow}} = \arg\max_k E_k$").concludes(
        rf"$\Rightarrow$  narrow, $k$ = {k_nar}")
    blocked = _Cell("but not the third", colour=_THIRD).concludes(
        f"no single rate to read off", _THIRD)
    blend = _Cell("so blend, then take the nearest", colour=inter).says(
        r"$\tilde{u}_k = \mathrm{minmax}\!\left(w\tilde{H}_k "
        r"+ (1-w)\tilde{s}_k\right)$",
        r"$k^{*} = \arg\min_k \sqrt{\tilde{u}_k^{2} + (1-\tilde{N}_k)^{2}}$").concludes(
        rf"$\Rightarrow$  intermediate, $k$ = {k_int}")
    weight = _Cell("and the weight does not decide it", colour=inter).says(
        r"$w \geq \frac{1}{2} \;\Longleftrightarrow\; w \geq 1 - w$").concludes(
        rf"$k^{{*}}$ = {k_int} on a plateau of $w$")

    fig = _story_row(
        "01_tradeoff",
        "The trade-off — one rate prices two of them, and nothing prices the third",
        "Power and residual stratification move together, so the walk has a "
        "single average rate and every cut can be scored against it. The "
        f"{case_label}/{control_label} shift does not: it reverses, so it has no "
        "rate. That is why there are two criteria and not one.",
        (price, blocked, blend, weight), eq_lines=3)

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

    ax = blocked.ax
    ax.axhline(0.0, color=_DIM, linewidth=0.9, linestyle=":")
    ax.plot(rank, _norm(het), "-s", color=_GR, markersize=4.0, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.0,
            label=r"$\tilde{H}_k$ — never reverses")
    ax.plot(rank, _norm(sep), "-", color=_THIRD, linewidth=2.0,
            label=rf"$\tilde{{s}}_k$ — reverses {flips}×")
    ax.set_xticks(rank[::4]); ax.set_ylim(-0.10, 1.34)
    ax.set_xlabel("cut $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax.set_ylabel("on $[0,1]$", fontsize=_MAIN["axis"])
    ax.legend(loc="upper center", fontsize=_MAIN["tick"], frameon=True,
              framealpha=0.95, edgecolor=_HAIR)
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
    ax.set_xlabel(r"$\tilde{u}_k$   $\rightarrow$", fontsize=_MAIN["axis"],
                  labelpad=3)
    ax.set_ylabel(r"$\tilde{N}_k$   $\rightarrow$", fontsize=_MAIN["axis"])
    _grid(ax)

    ax = weight.ax
    won = weight_winner[weight_winner > 0]
    on = weight_grid[weight_winner == k_int]
    lo, hi = (float(on.min()), float(on.max())) if on.size else (np.nan,) * 2
    ax.fill_between([0.0, 0.5], -100, 100, color=_BARRED, alpha=0.5,
                    linewidth=0)
    ax.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK,
            linewidth=2.0)
    if on.size:
        ax.plot([lo, hi], [k_int, k_int], color=inter, linewidth=6.0,
                solid_capstyle="butt", alpha=0.85)
    ax.axvline(blend_weight, color=inter, linewidth=1.5, linestyle="-.")
    ax.plot([blend_weight], [k_int], _MARK["intermediate"], color=inter,
            markersize=11.0, markeredgecolor="white", markeredgewidth=1.3)
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
    rank, neff, het, _, _ = _walk(decision_table, rgv_column)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    order = [n for n in CUT_ORDER if n in k_of]
    at = {n: int(np.argmin(np.abs(rank - k_of[n]))) for n in order}

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        if col not in r.columns or r.empty:
            return float("nan")
        return float(to_numeric_array(r[col])[0])

    where = _Cell("where the three stop")
    which = _Cell("which one to use", plot=False, width=_CARD_W_IN).concludes(
        r"narrow $\subset$ intermediate $\subset$ full")

    fig = _story_row(
        "02_cohorts", "The three cohorts",
        "Two criteria stop the walk in two places; taking every component is the "
        "third. They are nested, so this is a choice of where to stop along one "
        f"walk. Cuts resolved in mode: {mode}.",
        (where, which), plot_in=3.60, eq_lines=1)

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
