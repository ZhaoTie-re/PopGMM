"""How the three cohorts were chosen, as one line of reasoning.

Four steps across one canvas, joined by arrows: the walk and the two quantities
that move along it, the criterion that fixes ``narrow``, the criterion that
fixes ``intermediate``, and the three delivered sets. Each step is a heading, a
small plot, the equation that plot draws, and the answer it reaches.

The equations stay on the figure because they *are* the reasoning -- moving them
to prose leaves a picture that asserts three numbers without showing where they
came from. What is not here is the reviewer's detail: the weight sweep, the
significance test, the per-cut margins. ``cut_record.tsv``,
``rank_decision_table.tsv`` and the *Methodological notes* in ``docs/outputs.md``
carry those.
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

#: The unselected bars in step 2. Not a cohort colour, declared here for the
#: same reason they are: so there is one of it.
_MUTED = "#D6D6D6"

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

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
    """The reasoning that fixed the three cohorts, in four steps on one canvas."""
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    order = [n for n in CUT_ORDER if n in k_of]
    at = {n: int(np.argmin(np.abs(rank - k_of[n]))) for n in order}

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        if col not in r.columns or r.empty:
            return float("nan")
        return float(to_numeric_array(r[col])[0])

    # Step 2 and step 3 draw the two criteria. Both are recomputed here from the
    # same inputs the selector used, then checked against cut_record -- a figure
    # that disagreed with the record it illustrates would be worse than no
    # figure.
    gamma = float(rows["narrow"]["Exchange_Rate"])
    excess = (neff - neff[0]) - gamma * (het - het[0])
    blended = np.asarray(objective_spaces["intermediate"].structure, dtype=float)
    power = np.asarray(objective_spaces["intermediate"].power, dtype=float)
    dist = np.sqrt(blended ** 2 + (1.0 - power) ** 2)
    for name, got in (("narrow", int(rank[int(np.argmax(excess))])),
                      ("intermediate", int(rank[int(np.argmin(dist))]))):
        if got != k_of[name]:
            raise AssertionError(
                f"the {name} criterion drawn here peaks at k = {got}, but "
                f"cut_record.tsv records k = {k_of[name]}")

    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    height = (_TITLE_IN + _LEAD_IN + _STEP_IN + _PLOT_IN + _XLABEL_IN
              + _EQ_IN + _ANSWER_IN + _FOOT_IN)
    w_plot = (inner - 3.0 * _GUTTER_IN - _CARD_W_IN) / 3.0
    widths = [w_plot, w_plot, w_plot, _CARD_W_IN]

    fig = plt.figure(figsize=(width, height))
    x0 = _SIDE_IN / width
    lefts = [_SIDE_IN + sum(widths[:i]) + i * _GUTTER_IN for i in range(4)]
    y_ans = _FOOT_IN / height
    y_eq = y_ans + _ANSWER_IN / height
    y_plot = y_eq + (_EQ_IN + _XLABEL_IN) / height
    y_step = y_plot + (_PLOT_IN + _STEP_IN * 0.34) / height

    def cell(i: int, plot: bool = True):
        """One step's axes: its plot (or blank sheet) at a fixed place."""
        ax = fig.add_axes((lefts[i] / width, y_plot, widths[i] / width,
                           _PLOT_IN / height))
        if not plot:
            _note_axis(ax)
        return ax

    def head(i: int, num: str, text: str, colour: str = _BK) -> None:
        fig.text(lefts[i] / width, y_step, f"{num} ·  {text}",
                 fontsize=_MAIN["step"], fontweight="bold", ha="left",
                 va="bottom", color=colour)

    def _fits(t, i: int, what: str) -> None:
        """Fail if a string runs out of its own column and into the next one."""
        box = t.get_window_extent(renderer=fig.canvas.get_renderer())
        used = box.width / fig.dpi
        if used > widths[i] + 0.04:
            raise AssertionError(
                f"step {i + 1}: {what} is {used:.2f}in wide in a "
                f"{widths[i]:.2f}in column — {t.get_text()[:48]!r}")

    def equation(i: int, *lines: str) -> None:
        for j, tex in enumerate(lines):
            t = fig.text(lefts[i] / width,
                         y_eq + (_EQ_IN - 0.28 - j * 0.46) / height, tex,
                         fontsize=_MAIN["equation"], ha="left", va="center",
                         color=_BK)
            _fits(t, i, "an equation")

    def answer(i: int, text: str, colour: str) -> None:
        t = fig.text(lefts[i] / width, y_ans + 0.10 / height, text,
                     fontsize=_MAIN["answer"], fontweight="bold", ha="left",
                     va="bottom", color=colour)
        _fits(t, i, "the answer")

    for i in range(3):
        _arrow(fig, (lefts[i] + widths[i] + _GUTTER_IN / 2.0) / width, y_step)

    # ── ① the walk ───────────────────────────────────────────────────
    ax1 = cell(0)
    head(0, "1", "one walk, two quantities")
    ax1.plot(het, neff, "-o", color=_GR, markersize=4.4, linewidth=1.4,
             markerfacecolor="white", markeredgewidth=1.0)
    ax1.set_xlabel("residual spread   $\\rightarrow$", fontsize=_MAIN["axis"],
                   labelpad=3)
    ax1.set_ylabel("effective size   $\\rightarrow$", fontsize=_MAIN["axis"])
    ax1.tick_params(labelsize=_MAIN["tick"])
    ax1.grid(True, alpha=0.30, linewidth=0.7)
    ax1.set_axisbelow(True)
    ax1.margins(x=0.12, y=0.12)
    equation(0,
             r"$N_{\mathrm{eff},k} = \dfrac{4\,N_{\mathrm{case}}"
             r"N_{\mathrm{ctrl}}}{N_{\mathrm{case}} + N_{\mathrm{ctrl}}}$",
             r"$H_k = \left|\Sigma_k\right|^{1/2d}$")
    answer(0, "both rise with $k$", _BK)

    # ── ② the first criterion ────────────────────────────────────────
    ax2 = cell(1)
    head(1, "2", "price the walk", _EDGE["narrow"])
    i_nar = at["narrow"]
    ax2.vlines(rank, 0.0, excess, color=_MUTED, linewidth=5.0)
    ax2.vlines(rank[i_nar], 0.0, excess[i_nar], color=_EDGE["narrow"],
               linewidth=5.0)
    ax2.plot(rank, excess, "-", color=_GR, linewidth=1.2)
    ax2.plot([rank[i_nar]], [excess[i_nar]], _MARK["narrow"],
             color=_EDGE["narrow"], markersize=13.0, markeredgecolor="white",
             markeredgewidth=1.5, zorder=5)
    ax2.axhline(0.0, color=_DIM, linewidth=0.9, linestyle="--")
    ax2.set_xticks(rank[::4])
    ax2.set_xlabel("cut $k$", fontsize=_MAIN["axis"], labelpad=3)
    ax2.set_ylabel("$E_k$", fontsize=_MAIN["axis"])
    ax2.tick_params(labelsize=_MAIN["tick"])
    ax2.grid(True, alpha=0.30, linewidth=0.7)
    ax2.set_axisbelow(True)
    equation(1,
             r"$\gamma = \dfrac{N_{\mathrm{eff},K} - N_{\mathrm{eff},1}}"
             r"{H_K - H_1}$" + rf"$\;=\;{gamma:,.0f}$",
             r"$E_k = (N_{\mathrm{eff},k} - N_{\mathrm{eff},1}) "
             r"- \gamma\,(H_k - H_1)$",
             r"$k_{\mathrm{narrow}} = \arg\max_k E_k$")
    answer(1, rf"$\Rightarrow$  narrow, $k$ = {k_of['narrow']}", _EDGE["narrow"])

    # ── ③ the second criterion ───────────────────────────────────────
    ax3 = cell(2)
    head(2, "3", "a second kind of structure", _EDGE["intermediate"])
    i_int = at["intermediate"]
    ax3.plot(blended, power, "o", color=_GR, markersize=6.5, alpha=0.8,
             markerfacecolor="white", markeredgewidth=1.2)
    ax3.plot([0.0, blended[i_int]], [1.0, power[i_int]], "--",
             color=_EDGE["intermediate"], linewidth=2.0)
    ax3.plot([0.0], [1.0], "*", color=_EDGE["intermediate"], markersize=20.0,
             markeredgecolor="white", markeredgewidth=1.3)
    ax3.plot([blended[i_int]], [power[i_int]], _MARK["intermediate"],
             color=_EDGE["intermediate"], markersize=13.0,
             markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax3.annotate("ideal", xy=(0.0, 1.0), xytext=(11, -4),
                 textcoords="offset points", fontsize=_MAIN["annot"],
                 color=_EDGE["intermediate"], ha="left", va="top")
    ax3.set_xlim(-0.08, 1.10)
    ax3.set_ylim(-0.10, 1.16)
    ax3.set_xticks([0.0, 0.5, 1.0])
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set_xlabel(r"$\tilde{u}_k$   structure $\rightarrow$",
                   fontsize=_MAIN["axis"], labelpad=3)
    ax3.set_ylabel(r"$\tilde{N}_k$   power $\rightarrow$",
                   fontsize=_MAIN["axis"])
    ax3.tick_params(labelsize=_MAIN["tick"])
    ax3.grid(True, alpha=0.30, linewidth=0.7)
    ax3.set_axisbelow(True)
    equation(2,
             r"$s_k = \hat{D}^2_k - d\left(\frac{1}{N_{\mathrm{case}}}"
             r"+\frac{1}{N_{\mathrm{ctrl}}}\right)$",
             r"$\tilde{u}_k = \mathrm{minmax}(w\tilde{H}_k + (1-w)\tilde{s}_k)$",
             r"$k^{*} = \arg\min_k \sqrt{\tilde{u}_k^{2} "
             r"+ (1-\tilde{N}_k)^{2}}$")
    answer(2, rf"$\Rightarrow$  intermediate, $k$ = {k_of['intermediate']}",
           _EDGE["intermediate"])

    # ── ④ what that delivers ─────────────────────────────────────────
    ax4 = cell(3, plot=False)
    head(3, "4", "three cohorts")
    card_h = 1.0 / len(order)
    for i, name in enumerate(order):
        top = 1.0 - i * card_h
        ax4.add_patch(FancyBboxPatch(
            (0.0, top - card_h + 0.030), 1.0, card_h - 0.048,
            boxstyle="round,pad=0.004", facecolor=_TINT[name],
            edgecolor=_EDGE[name], linewidth=1.4, alpha=0.85, zorder=1))
        ax4.plot([0.018, 0.018], [top - card_h + 0.052, top - 0.020],
                 color=_EDGE[name], linewidth=5.5, solid_capstyle="butt",
                 zorder=3)
        ax4.text(0.055, top - 0.062, name, fontsize=_MAIN["card_name"],
                 fontweight="bold", ha="left", va="top", color=_EDGE[name],
                 zorder=3)
        ax4.text(0.982, top - 0.062, f"$k$ = {k_of[name]}",
                 fontsize=_MAIN["card_name"], fontweight="bold", ha="right",
                 va="top", color=_EDGE[name], zorder=3)
        ax4.text(0.055, top - 0.062 - card_h * 0.29, _WHEN[name],
                 fontsize=_MAIN["card_when"], ha="left", va="top", color=_BK,
                 zorder=3)
        ax4.text(0.055, top - 0.062 - card_h * 0.56,
                 f"{val(name, f'{case_label}_Count'):,.0f}"
                 f" + {val(name, f'{control_label}_Count'):,.0f}"
                 f" = {val(name, 'Total_Count'):,.0f}"
                 f"   effective {val(name, 'GWAS_Neff'):,.0f}",
                 fontsize=_MAIN["card_num"], ha="left", va="top", color=_GR,
                 zorder=3)
    answer(3, r"narrow $\subset$ intermediate $\subset$ full", _BK)

    fig.suptitle("How the three cohorts were chosen",
                 fontsize=_MAIN["suptitle"], fontweight="bold",
                 y=1.0 - 0.46 / height)
    fig.text(x0, 1.0 - (_TITLE_IN + 0.16) / height,
             f"The major cluster's {int(rank.max())} components are ordered by "
             f"{case_label}/{control_label} ratio, and cut $k$ keeps the top "
             f"$k$ — one nested set per $k$. Two quantities move along that "
             f"walk, and the two criteria below each stop it somewhere.",
             fontsize=_MAIN["lead"], color=_GR, ha="left", va="top")
    fig.text(x0, 0.30 / height,
             f"$w = \\frac{{1}}{{2}}$; a tilde is that quantity min-max scaled "
             f"to $[0,1]$.   Cuts resolved in mode: {mode}.   "
             f"Full derivation, and what each choice rules out: docs/outputs.md.",
             fontsize=_MAIN["foot"], color=_DIM, ha="left", va="bottom",
             fontstyle="italic")
    return fig
