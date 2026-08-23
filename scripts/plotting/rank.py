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
FIGURE_SIZE: "tuple[float, float]" = (21.0, 14.5)

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


def plot_selection(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rank_table: pd.DataFrame,
    objective_spaces: "Mapping[str, object]",
    rgv_column: str,
    mainland_axes: Sequence[str],
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    blend_weight: float,
    rank_cuts: "Mapping[str, int | None]",
    mode: str,
    basis: str,
    case_label: str = "Case",
    control_label: str = "Control",
    safe_weight_floor: float = 0.5,
) -> Figure:
    """Return the selection as a chain of formulas, each with what it means.

    Seven numbered steps, read top to bottom. Six define the quantities and the
    two rules; the seventh is what falls out of them -- three cohorts, and why
    there are three rather than one. Evidence sits beside the step it belongs
    to, so a formula and the plot that justifies it are never on separate rows.

    Every input parameter appears inside the formula that uses it (``d`` in the
    spread, ``w`` in the blend, the basis in the meaning line), so nothing the
    reader has to take on trust is left off the page.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    sep = decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float, copy=False)

    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    # Derived once when the cuts were resolved; read, never recomputed.
    rate = float(rows["narrow"]["Exchange_Rate"])
    reversals = int(rows["intermediate"]["Axis_Reversals"])
    excess = (neff - neff[0]) - rate * (het - het[0])
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    idx = {n: int(np.argmin(np.abs(rank - k_of[n]))) for n in k_of}
    rank_sorted = rank_table.sort_values("Rank")
    comps = {n: [int(c) for c in rank_sorted.loc[rank_sorted["Rank"] <= k_of[n], "Cluster"]]
             for n in k_of}

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        return float(to_numeric_array(r[col])[0]) if col in r.columns and not r.empty else float("nan")

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(7, 2, height_ratios=[0.90, 1.12, 1.12, 1.12, 0.90, 1.12, 2.28],
                          width_ratios=[1.62, 1.0], wspace=0.045, hspace=0.38)

    def _text_row(i: int, span: bool) -> "plt.Axes":
        ax = fig.add_subplot(gs[i, :] if span else gs[i, 0])
        _note_axis(ax)
        return ax

    def _step(ax: "plt.Axes", num: str, title: str, formula: str, meaning: str,
              value: str, colour: str = _BK, wrap: int = 96) -> None:
        """One rung: number, name, the formula, what it means, what it comes to.

        ``wrap`` differs by row because the rows carrying evidence are only as
        wide as the left column; an unwrapped meaning line runs under the plot.
        """
        meaning = "\n".join(textwrap.wrap(meaning, width=wrap))
        ax.text(0.004, 0.905, num, fontsize=13.0, fontweight="bold", ha="center",
                va="center", color=colour, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.26", facecolor="white",
                          edgecolor=colour, linewidth=1.4))
        ax.text(0.026, 0.905, title, fontsize=13.5, fontweight="bold", ha="left",
                va="center", color=colour)
        ax.text(0.026, 0.545, formula, fontsize=15.0, ha="left", va="center", color=_BK)
        ax.text(0.026, 0.235, meaning, fontsize=11.5, ha="left", va="top",
                color=_GR, linespacing=1.45)
        if value:
            ax.text(0.996, 0.905, value, fontsize=12.0, fontweight="bold", ha="right",
                    va="center", color=colour)

    def _mini(i: int, xlabel: str = "") -> "plt.Axes":
        ax = fig.add_subplot(gs[i, 1])
        ax.grid(True, alpha=0.30, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9.5, length=3.0, pad=2)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10.0, labelpad=1)
        return ax

    # ══ 1 · effective sample size ════════════════════════════════════
    _step(_text_row(0, True), "1", "Effective sample size — how much power a cut has",
          r"$N_k \;=\; \dfrac{4\,n_1 n_2}{n_1 + n_2}$",
          f"The size a balanced {case_label}/{control_label} study would need for the same power. "
          f"Every component added brings samples, so it only rises.",
          f"{neff.min():,.0f}  →  {neff.max():,.0f}")

    # ══ 2 · residual spread ══════════════════════════════════════════
    _step(_text_row(1, False), "2", "Residual spread — how heterogeneous a cut still is",
          rf"$H_k \;=\; \left|\Sigma_k\right|^{{1/2d}}, \quad d = {n_dim}$",
          f"Geometric mean of the per-axis SDs over the leading {n_dim} {basis} PCs — one number for "
          f"how wide the retained set is. Widening the set can only raise it.",
          f"{het.min():.5f}  →  {het.max():.5f}", wrap=58)
    ax2 = _mini(1)
    ax2.plot(rank, _safe_norm(neff), "-o", color=_BK, markersize=3.8, linewidth=1.5,
             markerfacecolor="white", markeredgewidth=1.0, label=r"$N_k$")
    ax2.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=3.8, linewidth=1.5,
             markerfacecolor="white", markeredgewidth=1.0, label=r"$H_k$")
    ax2.set_xticks(rank[::2]); ax2.set_ylim(-0.08, 1.30)
    ax2.legend(loc="upper left", fontsize=9.0, frameon=True, framealpha=0.95,
               edgecolor="#CFCFCF", ncol=2, handlelength=1.4)
    ax2.set_title("both rise strictly with $k$ — so the walk has one exchange rate",
                  fontsize=10.5, loc="left", pad=4, color=_GR)

    # ══ 3 · excess N_eff ═════════════════════════════════════════════
    _step(_text_row(2, False), "3", "Excess power — what a cut bought above that rate",
          r"$E_k = (N_k - N_1) - r\,(H_k - H_1), \quad "
          rf"r = \frac{{N_K - N_1}}{{H_K - H_1}} = {rate:,.0f}$",
          f"$r$ is what one unit of spread buys on average across the whole walk. $E_k$ is how much "
          f"more than that cut $k$ has bought. Where it peaks, spread stops repaying.",
          f"peak +{excess.max():,.0f} at $k$={rank[int(np.argmax(excess))]}",
          _EDGE["narrow"], wrap=58)
    ax3 = _mini(2)
    ax3.axhline(0.0, color=_DIM, linewidth=1.0, linestyle="--")
    ax3.vlines(rank, 0.0, excess, color="#D6D6D6", linewidth=3.6)
    ax3.vlines(rank[idx["narrow"]], 0.0, excess[idx["narrow"]], color=_EDGE["narrow"], linewidth=3.6)
    ax3.plot(rank, excess, "-", color=_GR, linewidth=1.3)
    ax3.plot([rank[idx["narrow"]]], [excess[idx["narrow"]]], _MARK["narrow"],
             color=_EDGE["narrow"], markersize=10.0, markeredgecolor="white", markeredgewidth=1.3)
    ax3.set_xticks(rank[::2])
    ax3.set_title(rf"$E_k$ peaks at $k$ = {k_of['narrow']}", fontsize=10.5, loc="left",
                  pad=4, color=_EDGE["narrow"], fontweight="bold")

    # ══ 4 · case/control distance ════════════════════════════════════
    _step(_text_row(3, False), "4", f"{case_label}/{control_label} distance — is the residual structure shared?",
          r"$s_k = \hat{D}^2_k - p\left(\frac{1}{n_1}+\frac{1}{n_2}\right),\quad "
          r"\hat{D}^2_k = \Delta\bar{x}^{\top} S^{-1}\Delta\bar{x}$",
          f"Gap between the two centroids on the same {n_dim} axes, with the gap sampling alone "
          f"would give removed. Spread says how wide; this says whether the arms sit apart.",
          f"reverses {reversals}×", "#B35806", wrap=58)
    ax4 = _mini(3)
    ax4.axhline(0.0, color=_DIM, linewidth=1.0, linestyle=":")
    ax4.plot(rank, sep, "-^", color="#B35806", markersize=4.6, linewidth=1.7,
             markerfacecolor="white", markeredgewidth=1.0)
    ax4.set_xticks(rank[::2])
    ax4.set_title(f"not monotone — {reversals} direction changes, so no rate to read",
                  fontsize=10.5, loc="left", pad=4, color="#B35806", fontweight="bold")

    # ══ 5 · the combined criterion ═══════════════════════════════════
    _step(_text_row(4, True), "5", "Combining them — one axis carrying both",
          r"$H_k(w) = w\,x_k + (1-w)\,s_k \qquad "
          r"k^{*}(w) = \arg\min_k \sqrt{H_k(w)^2 + (1 - y_k)^2}$",
          r"$x_k,\ y_k,\ s_k$ are $H_k$, $N_k$ and $s_k$ min-max scaled to $[0,1]$. Step 3 cannot be "
          r"repeated on step 4's axis, so the two are blended and the cut nearest the ideal "
          r"corner $(0,1)$ is taken.", "")

    # ══ 6 · why w = 1/2 ══════════════════════════════════════════════
    on = weight_grid[weight_winner == k_of["intermediate"]]
    lo, hi = (float(on.min()), float(on.max())) if on.size else (float("nan"),) * 2
    _step(_text_row(5, False), "6", r"Why $w = \frac{1}{2}$ — the most weight the distance may carry",
          rf"$w \geq \frac{{1}}{{2}} \;\Longleftrightarrow\; w \geq 1-w$",
          f"Below ½ the label-derived term outweighs the genotype-derived one, and minimising "
          f"that optimises what the association test measures. ½ is the boundary.",
          rf"stable on $[{lo:.2f},\ {hi:.2f}]$", _EDGE["intermediate"], wrap=58)
    ax6 = _mini(5, r"$w$")
    ax6.fill_between([0.0, safe_weight_floor], -100, 100, color="#F4C7C3", alpha=0.55, linewidth=0)
    ax6.text(safe_weight_floor / 2.0, 0.5, "not usable", transform=ax6.get_xaxis_transform(),
             fontsize=9.5, color="#9B2226", ha="center", va="center", fontstyle="italic")
    ax6.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK, linewidth=2.0)
    if on.size:
        ax6.plot([lo, hi], [k_of["intermediate"]] * 2, color=_EDGE["intermediate"],
                 linewidth=5.0, solid_capstyle="butt", alpha=0.85)
    ax6.axvline(blend_weight, color=_EDGE["intermediate"], linewidth=1.5, linestyle="-.")
    ax6.plot([blend_weight], [k_of["intermediate"]], _MARK["intermediate"],
             color=_EDGE["intermediate"], markersize=9.0, markeredgecolor="white", markeredgewidth=1.3)
    won = weight_winner[weight_winner > 0]
    ax6.set_ylim(int(won.min()) - 1, int(won.max()) + 1)
    ax6.set_xlim(0.0, 1.0)
    ax6.set_yticks([int(v) for v in (won.min(), k_of["intermediate"], won.max())])
    ax6.set_title(rf"$k^{{*}}(w)$, swept at {float(np.diff(weight_grid)[0]):.3f} — "
                  rf"at $w=\frac{{1}}{{2}}$ the answer is $k$ = {k_of['intermediate']}",
                  fontsize=10.5, loc="left", pad=4, color=_EDGE["intermediate"], fontweight="bold")

    # ══ 7 · the three cohorts ════════════════════════════════════════
    ax7 = fig.add_subplot(gs[6, :])
    _note_axis(ax7)
    ax7.text(0.004, 0.945, "7", fontsize=13.0, fontweight="bold", ha="center", va="center",
             color=_BK, zorder=5,
             bbox=dict(boxstyle="circle,pad=0.26", facecolor="white", edgecolor=_BK, linewidth=1.4))
    ax7.text(0.026, 0.945, "So there are three — one population, and the two cuts steps 3 and 6 fix",
             fontsize=13.5, fontweight="bold", ha="left", va="center", color=_BK)
    defs = {
        "full": ("every component of the mainland cluster — no rule, nothing selected",
                 r"the population steps 2–6 work inside"),
        "narrow": (r"$\arg\max_k E_k$   (step 3)",
                   r"the last cut whose spread still repays in power"),
        "intermediate": (rf"$\arg\min_k \sqrt{{H_k(\frac{{1}}{{2}})^2 + (1-y_k)^2}}$   (steps 5–6)",
                         r"the cut when distance carries all the weight it may"),
    }
    cols = ("", "definition", "$k$", "components", case_label, control_label, "n",
            r"$N_{eff}$", r"$H$")
    xs = (0.026, 0.130, 0.470, 0.530, 0.660, 0.740, 0.822, 0.888, 0.952)
    for xx, c in zip(xs, cols):
        ax7.text(xx, 0.775, c, fontsize=10.5, ha="left", va="center", color=_DIM, fontstyle="italic")
    ax7.plot([0.020, 0.995], [0.715, 0.715], color="#CFCFCF", linewidth=1.0)
    order = [n for n in CUT_ORDER if n in k_of]
    for i, name in enumerate(order):
        yy = 0.560 - i * 0.212
        ax7.add_patch(FancyBboxPatch((0.020, yy - 0.098), 0.975, 0.196,
                                     boxstyle="square,pad=0", facecolor=_TINT[name],
                                     edgecolor="none", alpha=0.60, zorder=1))
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
            ax7.text(xx, yy + 0.038, t, fontsize=13.0 if j == 0 else 11.5, ha="left",
                     va="center", zorder=3, color=_EDGE[name] if j == 0 else _BK,
                     fontweight="bold" if j in (0, 2) else "normal")
        ax7.text(xs[1], yy - 0.062, defs[name][1], fontsize=10.0, ha="left", va="center",
                 zorder=3, color=_DIM, fontstyle="italic")
    ax7.text(0.020, -0.055,
             r"narrow $\subset$ intermediate $\subset$ full — nested by construction.   "
             "Neither is the better list: a narrower set buys homogeneity with effective sample "
             f"size, and a broader one the reverse.   Cuts resolved in mode: {mode}.",
             fontsize=10.5, ha="left", va="bottom", color=_DIM, fontstyle="italic")

    _figure_title(fig, "Rank Selection", "why there are three cohorts")
    fig.subplots_adjust(left=0.026, right=0.988, top=0.940, bottom=0.028)
    fig.text(0.5, 0.006,
             f"Derivations and checks: docs/method.md § 6.   "
             f"Every number here is in rank_decision_table.tsv and cut_record.tsv.",
             fontsize=10.0, color=_DIM, ha="center", va="bottom", fontstyle="italic")
    return fig
