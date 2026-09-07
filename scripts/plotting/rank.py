"""The cohort-selection figure: three nested sets, and when to use each.

One figure. The walk of cumulative cuts is drawn as a trade-off curve with the
three delivered cohorts marked on it, and beside it three cards saying what each
one is for and what it costs. That is the whole decision a reader has to make.

It was four figures once -- the problem, each selected cut, and the cohorts --
carrying eight hundred words of derivation between them. The derivation is not
lost: ``docs/outputs.md`` carries the argument for every choice under
*Methodological notes*, ``rank_decision_table.tsv`` carries every number at
every ``k``, and ``cut_record.tsv`` carries the operator and value behind each
cut. None of that belongs on the figure someone reads to pick a cohort.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

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

#: Marker shape per cut, so the three stay distinguishable without colour.
_MARK = {"narrow": "D", "intermediate": "s", "full": "o"}

#: When each cohort is the right one. This is what the figure is for.
_WHEN = {
    "narrow": "residual stratification is the main worry",
    "intermediate": "both worries matter",
    "full": "power is the main worry, or a reference is wanted",
}

#: One type scale. Every ``fontsize=`` in this module reads from it; passing a
#: bare number is how several different sizes once ended up on one figure.
_MAIN = {
    "suptitle": 27.0, "card_name": 21.0, "card_k": 21.0, "card_when": 15.0,
    "card_num": 15.0, "panel": 17.0, "axis": 15.0, "tick": 13.0, "annot": 15.0,
    "foot": 13.5,
}

#: Inches. Everything on this figure is placed in them, so a change to one
#: length cannot silently mean something different somewhere else.
_SIDE_IN = 0.62
_TITLE_IN = 1.15
_FOOT_IN = 0.95
_GUTTER_IN = 0.70
_PANEL_TOP_IN = 0.52
_XLABEL_IN = 0.72
_PLOT_IN = 4.60

#: Width the trade-off plot takes; the cards take the rest.
_PLOT_SHARE = 0.56


def _note_axis(ax: "plt.Axes") -> None:
    """Turn an axes into a blank sheet with unit coordinates for typeset text."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _panel_title(ax: "plt.Axes", letter: str, text: str) -> None:
    """Panel heading, in the one style this figure uses."""
    ax.set_title(f"{letter} · {text}", fontsize=_MAIN["panel"],
                 fontweight="bold", loc="left", pad=10)


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
    """The three cohorts on the trade-off they were chosen along, and their cost.

    Left: every cumulative cut as a point, the three delivered ones marked.
    Right: one card per cohort -- what it is for, and what it contains.
    """
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    order = [n for n in CUT_ORDER if n in k_of]

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        if col not in r.columns or r.empty:
            return float("nan")
        return float(to_numeric_array(r[col])[0])

    # The marked cuts come from cut_record, never from a recomputation here, so
    # the figure cannot drift from the record it is illustrating.
    at = {n: int(np.argmin(np.abs(rank - k_of[n]))) for n in order}
    for n, i in at.items():
        if int(rank[i]) != k_of[n]:
            raise AssertionError(
                f"{n} is recorded at k = {k_of[n]}, which is not on the walk")

    width = FIGURE_SIZE[0]
    inner = width - 2.0 * _SIDE_IN
    height = (_TITLE_IN + _PANEL_TOP_IN + _PLOT_IN + _XLABEL_IN + _FOOT_IN)

    fig = plt.figure(figsize=(width, height))
    x0 = _SIDE_IN / width
    y_plot = (_FOOT_IN + _XLABEL_IN) / height
    w_plot = inner * _PLOT_SHARE
    ax = fig.add_axes(((_SIDE_IN + 0.55) / width, y_plot,
                       (w_plot - 0.55) / width, _PLOT_IN / height))
    ax_card = fig.add_axes(((_SIDE_IN + w_plot + _GUTTER_IN) / width, y_plot,
                            (inner - w_plot - _GUTTER_IN) / width,
                            _PLOT_IN / height))
    _note_axis(ax_card)

    # ── the walk, with the three stopping points on it ───────────────
    ax.plot(het, neff, "-o", color=_GR, markersize=5.0, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.1, zorder=3,
            label=f"the {int(rank.max())} cumulative cuts")
    for name in order:
        i = at[name]
        ax.plot([het[i]], [neff[i]], _MARK[name], color=_EDGE[name],
                markersize=17.0, markeredgecolor="white", markeredgewidth=2.0,
                zorder=6)
        ax.annotate(f"{name}\n$k$ = {k_of[name]}", xy=(het[i], neff[i]),
                    xytext={"narrow": (-62, -30), "intermediate": (2, 34),
                            "full": (-18, -36)}[name],
                    textcoords="offset points", fontsize=_MAIN["annot"],
                    fontweight="bold", color=_EDGE[name], ha="center",
                    va="center", zorder=7)
    ax.set_xlabel(r"residual spread   $\rightarrow$ less homogeneous",
                  fontsize=_MAIN["axis"], labelpad=4)
    ax.set_ylabel(r"effective sample size   $\rightarrow$ more power",
                  fontsize=_MAIN["axis"])
    ax.tick_params(labelsize=_MAIN["tick"])
    ax.legend(loc="lower right", fontsize=_MAIN["tick"], frameon=True,
              framealpha=0.95, edgecolor="#CFCFCF")
    ax.grid(True, alpha=0.30, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.16, y=0.14)
    _panel_title(ax, "A", "the walk, and where the three stop")

    # ── one card per cohort ──────────────────────────────────────────
    _panel_title(ax_card, "B", "which one to use")
    card_h = 1.0 / len(order)
    for i, name in enumerate(order):
        top = 1.0 - i * card_h
        ax_card.add_patch(FancyBboxPatch(
            (0.0, top - card_h + 0.035), 1.0, card_h - 0.055,
            boxstyle="round,pad=0.004", facecolor=_TINT[name],
            edgecolor=_EDGE[name], linewidth=1.5, alpha=0.85, zorder=1))
        ax_card.plot([0.016, 0.016], [top - card_h + 0.058, top - 0.022],
                     color=_EDGE[name], linewidth=6.0, solid_capstyle="butt",
                     zorder=3)
        ax_card.text(0.048, top - 0.070, name, fontsize=_MAIN["card_name"],
                     fontweight="bold", ha="left", va="top", color=_EDGE[name],
                     zorder=3)
        ax_card.text(0.985, top - 0.070, f"$k$ = {k_of[name]}",
                     fontsize=_MAIN["card_k"], fontweight="bold", ha="right",
                     va="top", color=_EDGE[name], zorder=3)
        ax_card.text(0.048, top - 0.070 - card_h * 0.30, _WHEN[name],
                     fontsize=_MAIN["card_when"], ha="left", va="top",
                     color=_BK, zorder=3)
        ax_card.text(
            0.048, top - 0.070 - card_h * 0.56,
            f"{val(name, f'{case_label}_Count'):,.0f} {case_label}"
            f"  +  {val(name, f'{control_label}_Count'):,.0f} {control_label}"
            f"  =  {val(name, 'Total_Count'):,.0f}"
            f"      effective {val(name, 'GWAS_Neff'):,.0f}",
            fontsize=_MAIN["card_num"], ha="left", va="top", color=_GR,
            zorder=3)

    fig.suptitle("Cohort Selection — three nested sets, and when to use each",
                 fontsize=_MAIN["suptitle"], fontweight="bold",
                 y=1.0 - 0.52 / height)
    fig.text(x0, 0.34 / height,
             "narrow $\\subset$ intermediate $\\subset$ full — three stopping "
             "points on one walk, not three different lists.   "
             "How each $k$ was fixed, and why: docs/outputs.md.",
             fontsize=_MAIN["foot"], color=_DIM, ha="left", va="bottom",
             fontstyle="italic")
    return fig
