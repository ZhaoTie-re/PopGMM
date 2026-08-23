"""The two rank-selection figures: the argument, and what supports it.

``plot_selection`` carries the whole reasoning on one canvas -- one panel per
step, read left to right, then the three cohorts it produces. It was four
figures once, and every step was told twice across them; a reader had to open
four files and know the order to reconstruct one chain.

``plot_methods`` holds what that figure deliberately leaves out: the definitions
of the two selection quantities and of the separation statistic, the
significance test, and what each answer depends on. The split exists so the
first figure can be read as an argument rather than a derivation.
"""

from __future__ import annotations

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
FIGURE_SIZE: "tuple[float, float]" = (21.0, 11.0)

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

#: The two figures, in reading order. Each names its own place and its
#: neighbour, so one lifted out of the directory still says where it sits.
FIGURE_SERIES: "tuple[tuple[str, str], ...]" = (
    ("00_selection", "why there are three cohorts"),
    ("01_methods", "the definitions and checks behind it"),
)

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


def _split_figure(n_left: int = 1, **kwargs: "Any") -> "tuple[Figure, Any, Any]":
    """A figure in the family layout: ``n_left`` stacked data panels, then the column.

    Returns the figure, the left-hand gridspec and the note axes. Callers that
    need a different left-hand arrangement subdivide the gridspec themselves.
    """
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=list(COLUMN_RATIOS), wspace=COLUMN_WSPACE)
    gs_left = gs[0, 0].subgridspec(n_left, 1, **kwargs) if n_left > 1 else gs[0, 0]
    return fig, gs_left, fig.add_subplot(gs[0, 1])


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


def _series_tag(name: str) -> str:
    """"1 of 2 · ... — before X", for a figure's footer."""
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

    A footer rather than a subtitle: it is metadata about where to read this
    figure, not part of what the figure says.
    """
    fig.text(0.5, 0.013, _series_tag(name), fontsize=10.5, color=_DIM,
             ha="center", va="bottom", fontstyle="italic")


def _note_axis(ax: "plt.Axes") -> None:
    """Turn an axes into a blank sheet with unit coordinates for typeset text."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _mark_cuts(
    axes: "Sequence[plt.Axes]", cuts: "Mapping[str, int]", label_on: "plt.Axes | None",
    rank_lo: int, rank_hi: int,
) -> None:
    """Drop a dashed rule at each delivered cut, labelled once across the stack."""
    for name, k in sorted(cuts.items(), key=lambda kv: kv[1]):
        for axis in axes:
            axis.axvline(k, color=_DIM, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
        if label_on is None:
            continue
        # Axes-fraction y so the strip stays put whatever the data range; the end
        # cuts anchor inwards or they overrun the frame.
        ha = "left" if k <= rank_lo else "right" if k >= rank_hi else "center"
        label_on.text(
            k, 0.985, f"{name}\nk={k}", color=_BK, fontsize=11.0, fontweight="bold",
            va="top", ha=ha, transform=label_on.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.8, alpha=0.92),
        )



def plot_selection(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rank_table: pd.DataFrame,
    objective_spaces: "Mapping[str, object]",
    rgv_column: str,
    mainland_axes: Sequence[str],
    blend_weight: float,
    rank_cuts: "Mapping[str, int | None]",
    mode: str,
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """Return the whole argument on one figure: four steps, then the result.

    One panel per step of the reasoning, read left to right, and each panel
    shows exactly one thing. Panel 1 introduces the two axes that 2 and 4 reuse,
    so the chain is cumulative rather than four separate exhibits.

    No algebra here. Definitions, the significance test and the robustness
    checks are in the methods figure, which exists so this one can stay an
    argument.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    sep = decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float, copy=False)
    blended = np.asarray(objective_spaces["intermediate"].structure, dtype=float)
    y_norm = _safe_norm(neff)

    rate = float(neff[-1] - neff[0]) / float(het[-1] - het[0])
    excess = (neff - neff[0]) - rate * (het - het[0])
    reversals = _reversals(sep)

    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in rows}
    rank_sorted = rank_table.sort_values("Rank")
    comps = {n: [int(c) for c in rank_sorted.loc[rank_sorted["Rank"] <= k_of[n], "Cluster"]]
             for n in k_of}
    idx = {n: int(np.argmin(np.abs(rank - k_of[n]))) for n in k_of}

    def val(name: str, col: str) -> float:
        r = decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]]
        return float(to_numeric_array(r[col])[0]) if col in r.columns and not r.empty else float("nan")

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.40], wspace=0.24, hspace=0.30)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[0, i]) for i in range(4))
    ax_res = fig.add_subplot(gs[1, :])

    def _step(ax: "plt.Axes", num: str, name: str | None, head: str, tail: str) -> None:
        """Numbered heading over a panel, in the cut's colour where it has one.

        Both lines are placed in axes coordinates rather than one being a title:
        a title plus a text block below it collide as soon as either wraps.
        """
        edge = _EDGE[name] if name else _DIM
        ax.text(0.0, 1.175, f"{num} · {head}", transform=ax.transAxes,
                fontsize=13.0, fontweight="bold", ha="left", va="bottom", color=edge)
        ax.text(0.0, 1.020, tail, transform=ax.transAxes, fontsize=10.5,
                ha="left", va="bottom", color=_GR, linespacing=1.5)

    # ══ 1 · the population, and the two things that move inside it ═══
    ax1.plot(het, neff, "-o", color=_GR, markersize=4.6, linewidth=1.5,
             markerfacecolor="white", markeredgewidth=1.1, zorder=3)
    ax1.plot([het[idx["full"]]], [neff[idx["full"]]], _MARK["full"], color=_EDGE["full"],
             markersize=15.0, markeredgecolor="white", markeredgewidth=1.6, zorder=6)
    ax1.annotate(f"full\n$k$ = {k_of['full']}   n = {val('full', 'Total_Count'):,.0f}",
                 xy=(het[idx["full"]], neff[idx["full"]]), xytext=(-8, -34),
                 textcoords="offset points", fontsize=11.0, fontweight="bold",
                 color=_EDGE["full"], ha="right", va="top", zorder=7)
    ax1.set_xlabel("residual spread  (RGV)")
    ax1.set_ylabel(r"GWAS $N_{eff}$")
    _step(ax1, "1", "full", "The population",
          "every component of the mainland cluster —\nthe other two are chosen inside it")

    # ══ 2 · pricing the walk, which is where narrow comes from ═══════
    ax2.axhline(0.0, color=_DIM, linewidth=1.1, linestyle="--", zorder=2)
    ax2.vlines(rank, 0.0, excess, color="#D6D6D6", linewidth=4.4, zorder=2)
    ax2.vlines(rank[idx["narrow"]], 0.0, excess[idx["narrow"]], color=_EDGE["narrow"],
               linewidth=4.4, zorder=3)
    ax2.plot(rank, excess, "-", color=_GR, linewidth=1.4, zorder=4)
    ax2.plot([rank[idx["narrow"]]], [excess[idx["narrow"]]], _MARK["narrow"],
             color=_EDGE["narrow"], markersize=13.0, markeredgecolor="white",
             markeredgewidth=1.5, zorder=6)
    ax2.annotate(f"narrow\n$k$ = {k_of['narrow']}   +{excess[idx['narrow']]:,.0f}",
                 xy=(rank[idx["narrow"]], excess[idx["narrow"]]), xytext=(15, -16),
                 textcoords="offset points", fontsize=11.0, fontweight="bold",
                 color=_EDGE["narrow"], ha="left", va="top", zorder=7)
    ax2.set_xticks(rank[::2])
    ax2.set_xlabel(r"cumulative rank $k$")
    ax2.set_ylabel(r"excess $N_{eff}$")
    _step(ax2, "2", "narrow", "Both axes rise, so price the walk",
          f"one average rate ({rate:,.0f} $N_{{eff}}$ per unit spread);\n"
          f"the surplus over it peaks here")

    # ══ 3 · the observation that forces a different method ═══════════
    ax3.plot(rank, _safe_norm(het), "-s", color="#8C8C8C", markersize=4.6, linewidth=1.7,
             markerfacecolor="white", markeredgewidth=1.1, zorder=3,
             label="residual spread — never reverses")
    ax3.plot(rank, _safe_norm(sep), "-^", color="#B35806", markersize=5.2, linewidth=1.9,
             markerfacecolor="white", markeredgewidth=1.1, zorder=4,
             label=f"case/control distance — reverses {reversals}×")
    ax3.set_xticks(rank[::2])
    ax3.set_ylim(-0.06, 1.30)
    ax3.set_xlabel(r"cumulative rank $k$")
    ax3.set_ylabel("min-max normalised")
    ax3.legend(loc="upper center", frameon=True, framealpha=0.95,
               edgecolor="#CFCFCF", fontsize=9.5)
    _step(ax3, "3", None, "But the second axis has no rate",
          "case/control distance does not fall as\nspread improves — so step 2 cannot be repeated")

    # ══ 4 · combining them, which is where intermediate comes from ═══
    ax4.plot(blended, y_norm, "-o", color=_GR, markersize=4.4, linewidth=1.1, alpha=0.75,
             markerfacecolor="white", markeredgewidth=1.1, zorder=3)
    ax4.plot([0.0], [1.0], "*", color=_EDGE["intermediate"], markersize=19.0, zorder=6,
             markeredgecolor="white", markeredgewidth=1.0, label="ideal corner")
    ax4.plot([0.0, blended[idx["intermediate"]]], [1.0, y_norm[idx["intermediate"]]], "--",
             color=_EDGE["intermediate"], linewidth=2.0, zorder=5)
    ax4.plot([blended[idx["intermediate"]]], [y_norm[idx["intermediate"]]],
             _MARK["intermediate"], color=_EDGE["intermediate"], markersize=13.0,
             markeredgecolor="white", markeredgewidth=1.5, zorder=7)
    ax4.annotate(f"intermediate\n$k$ = {k_of['intermediate']}",
                 xy=(blended[idx["intermediate"]], y_norm[idx["intermediate"]]),
                 xytext=(12, -8), textcoords="offset points", fontsize=11.0,
                 fontweight="bold", color=_EDGE["intermediate"], ha="left", va="top", zorder=8)
    ax4.set_xlabel(rf"spread and distance combined, $w={blend_weight:g}$")
    ax4.set_ylabel(r"$N_{eff}$  (normalised)")
    ax4.legend(loc="lower right", frameon=True, framealpha=0.95,
               edgecolor="#CFCFCF", fontsize=9.5)
    _step(ax4, "4", "intermediate", "So combine the two axes",
          "one parameter carrying both, and take the\ncut nearest the unattainable corner")

    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, alpha=0.30, linewidth=0.7)
        ax.set_axisbelow(True)

    # ══ The result ═══════════════════════════════════════════════════
    _note_axis(ax_res)
    _panel_title(ax_res, "5", "What that delivers")
    cols = ("cohort", "$k$", "components (added to the row above)", case_label, control_label,
            "n", r"$N_{eff}$", "RGV")
    xs = (0.010, 0.118, 0.190, 0.330, 0.440, 0.552, 0.660, 0.775)
    for xx, c in zip(xs, cols):
        ax_res.text(xx, 0.845, c, fontsize=11.0, ha="left", va="top",
                    color=_DIM, fontstyle="italic")
    ax_res.plot([0.0, 1.0], [0.760, 0.760], color="#CFCFCF", linewidth=1.0)
    # Each row lists what it adds to the row above, not the whole set: the full
    # list is 17 ids wide and the increments are what carry the nesting.
    _ordered = [n for n in CUT_ORDER if n in k_of]
    added: dict[str, str] = {}
    for j, name in enumerate(_ordered):
        if j == 0:
            added[name] = ", ".join(str(c) for c in comps[name])
        else:
            extra = [c for c in comps[name] if c not in set(comps[_ordered[j - 1]])]
            added[name] = "+ " + ", ".join(str(c) for c in extra)

    for i, name in enumerate(_ordered):
        yy = 0.640 - i * 0.215
        ax_res.add_patch(FancyBboxPatch(
            (0.0, yy - 0.088), 1.0, 0.180, boxstyle="square,pad=0",
            facecolor=_TINT[name], edgecolor="none", alpha=0.60, zorder=1))
        vals = (name, str(k_of[name]), added[name],
                f"{val(name, f'{case_label}_Count'):,.0f}",
                f"{val(name, f'{control_label}_Count'):,.0f}",
                f"{val(name, 'Total_Count'):,.0f}",
                f"{val(name, 'GWAS_Neff'):,.0f}",
                f"{val(name, rgv_column):.5f}")
        for j, (xx, t) in enumerate(zip(xs, vals)):
            ax_res.text(xx, yy, t, fontsize=12.0 if j == 0 else 11.0, ha="left",
                        va="center", zorder=3,
                        color=_EDGE[name] if j == 0 else _BK,
                        fontweight="bold" if j in (0, 1) else "normal",
                        fontstyle="italic" if j == 2 else "normal")
    ax_res.text(0.0, 0.020,
                r"narrow $\subset$ intermediate $\subset$ full — nested by construction.   "
                "Neither is the better list: a narrower set buys homogeneity with effective "
                "sample size, and a broader one the reverse.",
                fontsize=10.5, ha="left", va="bottom", color=_DIM, fontstyle="italic")

    _figure_title(fig, "Rank Selection",
                  "why there are three cohorts, and what each one is")
    fig.subplots_adjust(left=0.048, right=0.992, top=0.845, bottom=0.055)
    _series_footer(fig, "00_selection")
    return fig


def plot_methods(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rgv_column: str,
    mainland_axes: Sequence[str],
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    blend_weight: float,
    rank_cuts: "Mapping[str, int | None]",
    config: "RankSelectionConfig",
    safe_weight_floor: float = 0.5,
) -> Figure:
    """Return the supporting material the argument figure deliberately omits.

    Definitions of the two selection quantities and of the separation
    statistic, the significance test behind step 3, and what each answer
    depends on. Separated so the argument figure can be read as an argument.
    """
    n_dim = len(mainland_axes)
    rgv_dim = n_dim if config.rgv_basis == "mainland" else 2
    rgv_axis_label = (f"mainland PCA, PC1-PC{rgv_dim}" if config.rgv_basis == "mainland"
                      else "global PCA, PC1-PC2")
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    pval = decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False)
    cuts = {str(n): int(k) for n, k in rank_cuts.items() if k is not None}
    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_blend = int(rows["intermediate"]["Resolved_Rank"])
    k_narrow = int(rows["narrow"]["Resolved_Rank"])

    fig, gs_l, ax_note = _split_figure(2, height_ratios=[1.0, 0.92], hspace=0.34)
    ax_p = fig.add_subplot(gs_l[0, 0])
    ax_w = fig.add_subplot(gs_l[1, 0])

    # ══ A · is the separation in step 3 real? ════════════════════════
    sig = np.isfinite(pval) & (pval < 0.05)
    ax_p.plot(rank, pval, "-", color=_GR, linewidth=1.5, zorder=3)
    ax_p.scatter(rank[sig], pval[sig], s=78, color="#D7191C", edgecolor="white",
                 linewidth=1.0, zorder=5, label=r"$p < 0.05$")
    ax_p.scatter(rank[~sig], pval[~sig], s=78, color="#FFFFFF", edgecolor=_GR,
                 linewidth=1.5, zorder=5, label=r"$p \geq 0.05$")
    ax_p.axhline(0.05, color="#D7191C", linewidth=1.1, linestyle="--", zorder=2)
    ax_p.set_yscale("log")
    ax_p.set_xticks(rank)
    ax_p.set_xlabel(r"cumulative rank $k$")
    ax_p.set_ylabel(r"Hotelling's $T^2$  $p$")
    _panel_title(ax_p, "A", "Is that case/control distance real? — the exact $F$ test")
    ax_p.legend(loc="lower left", frameon=True, framealpha=0.95,
                edgecolor="#CFCFCF", fontsize=11.0, ncol=2)
    _mark_cuts((ax_p,), cuts, ax_p, int(rank.min()), int(rank.max()))

    # ══ B · what the two answers depend on ═══════════════════════════
    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color="#F4C7C3", alpha=0.55,
                      linewidth=0, zorder=1)
    ax_w.text(safe_weight_floor / 2.0, 0.05,
              "not usable — weights the case/control labels\nabove the residual spread",
              transform=ax_w.get_xaxis_transform(), fontsize=10.5, color="#9B2226",
              ha="center", va="bottom", fontstyle="italic", zorder=6)
    ax_w.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK,
              linewidth=2.6, zorder=4, solid_joinstyle="miter")
    on = weight_grid[weight_winner == k_blend]
    if on.size:
        lo, hi = float(on.min()), float(on.max())
        ax_w.plot([lo, hi], [k_blend, k_blend], color=_EDGE["intermediate"], linewidth=7.0,
                  solid_capstyle="butt", alpha=0.85, zorder=5)
        ax_w.axvline(blend_weight, color=_EDGE["intermediate"], linewidth=1.6,
                     linestyle="-.", zorder=6)
        ax_w.plot([blend_weight], [k_blend], _MARK["intermediate"],
                  color=_EDGE["intermediate"], markersize=11.0, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=8)
        ax_w.annotate(
            rf"intermediate holds on $w \in [{lo:.2f},\ {hi:.2f}]$ — "
            rf"evaluated {min(blend_weight - lo, hi - blend_weight):.2f} from the nearest edge",
            xy=(blend_weight, k_blend), xytext=(0, 16), textcoords="offset points",
            fontsize=11.0, color=_BK, ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=_EDGE["intermediate"], linewidth=1.2, alpha=0.96), zorder=9)
    if not np.any(weight_winner == k_narrow):
        ax_w.axhline(k_narrow, color=_EDGE["narrow"], linewidth=1.1, linestyle=":",
                     alpha=0.9, zorder=2)
        ax_w.text(0.995, k_narrow,
                  rf"narrow ($k$={k_narrow}) — fixed in step 2; "
                  rf"peak leads the runner-up by {float(rows['narrow']['Margin']):.1f} $N_{{eff}}$",
                  color=_EDGE["narrow"], fontsize=10.5, va="center", ha="right",
                  fontstyle="italic", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="none", alpha=0.92))
    ax_w.set_ylim(int(won.min()) - 1, max(int(won.max()), max(cuts.values(), default=0)) + 1)
    ax_w.set_xlim(0.0, 1.0)
    ax_w.set_yticks(sorted(set(int(v) for v in np.unique(won)) | set(cuts.values())))
    ax_w.set_ylabel(r"winning cut  $k^{*}(w)$")
    ax_w.set_xlabel(r"$w$ — weight on residual spread;  $1-w$ on case/control distance")
    _panel_title(ax_w, "B", "What the answers depend on — only step 4 has a weight to depend on")

    for ax in (ax_p, ax_w):
        ax.grid(True, alpha=0.30, linewidth=0.7)
        ax.set_axisbelow(True)

    _note_axis(ax_note)

    # ── The column: the two selection quantities, then the statistic ──
    _X = _X_INDENT

    def _rule(y: float) -> None:
        ax_note.plot([0.0, 1.0], [y, y], color="#E0E0E0", linewidth=0.9)

    def _h(y: float, txt: str) -> None:
        ax_note.text(_X, y, txt, fontsize=13.5, fontweight="bold", va="top",
                     ha="left", color=_BK)

    def _m(y: float, txt: str, size: float = 12.5, indent: float = 0.0) -> None:
        ax_note.text(_X + indent, y, txt, fontsize=size, va="top", ha="left", color=_BK)

    def _t(y: float, txt: str, size: float = 11.0, color: str = _GR,
           style: str = "normal") -> None:
        ax_note.text(_X, y, txt, fontsize=size, va="top", ha="left", color=color,
                     fontstyle=style)

    # ══ N_eff, as published ══════════════════════════════════════════
    _rule(0.982)
    _h(0.966, r"Effective Sample Size  $N_{eff}$")
    _t(0.936, "Variance of allele-frequency difference (unbalanced design):",
       size=10.5, color=_DIM, style="italic")
    _m(0.906, r"$\mathrm{Var}=p(1-p)\!\left(\dfrac{1}{N_{\mathrm{case}}}"
              r"+\dfrac{1}{N_{\mathrm{ctrl}}}\right)$", 12.5, 0.04)
    _t(0.850, r"Equate to balanced design  ($N_{eff}/2$ per arm):",
       size=10.5, color=_DIM, style="italic")
    _m(0.820, r"$=\dfrac{2\,p(1-p)}{N_{eff}}$", 12.5, 0.08)
    _m(0.760, r"$\Rightarrow\;N_{eff}=\dfrac{4\,N_{\mathrm{case}}\,N_{\mathrm{ctrl}}}{"
              r"N_{\mathrm{case}}+N_{\mathrm{ctrl}}}=N_{tot}\cdot\dfrac{4r}{(1+r)^{2}}$", 13.0)
    _t(0.698, r"Equivalent total $N$ of a balanced (1:1) case-control study")
    _t(0.672, "with the same statistical test power.")
    _t(0.646, r"($r=N_{\mathrm{case}}/N_{\mathrm{ctrl}},\;\;"
              r"N_{tot}=N_{\mathrm{case}}+N_{\mathrm{ctrl}}$)", size=10.0, color=_DIM)

    # ══ Residual spread, as published ════════════════════════════════
    _rule(0.622)
    _h(0.606, f"Residual Spread  $H$  ({rgv_axis_label})")
    _m(0.572, rf"$H=\left|\,\Sigma\,\right|^{{1/{2 * rgv_dim}}}$", 14.0, 0.04)
    # The closed form below only holds in two dimensions, so it is shown only
    # when the basis actually has two axes.
    if rgv_dim == 2:
        _det = (r"$|\Sigma|=\sigma^{2}_{PC1}\cdot\sigma^{2}_{PC2}-\sigma^{2}_{PC1,PC2}"
                r"=\sigma^{2}_{PC1}\cdot\sigma^{2}_{PC2}(1-\rho^{2})$")
        _closed = r"$\Rightarrow H=(\sigma_{PC1}\cdot\sigma_{PC2})^{1/2}(1-\rho^{2})^{1/4}$"
        _mean = r"Root Generalized Variance (2D): joint spread in PC1/PC2,"
        _tail = r"attenuated by inter-PC correlation $\rho$."
    else:
        _det = r"$|\Sigma|=\prod_{i=1}^{d}\lambda_i$" + f",  $d={rgv_dim}$"
        _closed = r"$\Rightarrow H=\left(\prod_i \sqrt{\lambda_i}\right)^{1/d}$"
        _mean = f"Root Generalized Variance ({rgv_dim}D): joint spread over PC1-PC{rgv_dim},"
        _tail = r"the geometric mean of the per-axis SDs."
    _m(0.518, _det, 11.5)
    _m(0.458, _closed, 12.5)
    _t(0.418, _mean)
    _t(0.392, _tail)

    # ══ The separation statistic step 3 reads off ════════════════════
    _rule(0.368)
    _h(0.352, r"Case/Control Distance  $D$")
    _m(0.318, r"$D^2 = (\bar{x}_{1} - \bar{x}_{2})^{\top}\,"
              r"S_{\mathrm{pooled}}^{-1}\,(\bar{x}_{1} - \bar{x}_{2})$", 13.0, 0.04)
    _m(0.262, r"$T^2 = D^2\,\dfrac{n_1 n_2}{n_1 + n_2}$"
              r"$\qquad F = \dfrac{T^2\,(\nu - p + 1)}{p\,\nu}"
              r"\ \sim\ F_{p,\ \nu-p+1}$", 12.0)
    _t(0.204, r"Mahalanobis distance between the case and control centroids under")
    _t(0.178, r"the pooled within-group covariance, $\nu = n_1 + n_2 - 2$, in pooled-SD")
    _t(0.152, rf"units — on the same mainland PC1-PC{n_dim} axes the spread uses.")

    # ══ Why step 3 plots the de-biased form ══════════════════════════
    _rule(0.128)
    _h(0.112, "Why step 3 plots the de-biased form")
    _m(0.080, r"$E\left[\hat{D}^2\right] = D^2_{\mathrm{true}}"
              r" + p\left(\dfrac{1}{n_1} + \dfrac{1}{n_2}\right)$", 12.5, 0.04)
    _t(0.026, r"Two means never coincide and $D^2$ squares the gap, so scatter only adds;")
    _t(0.000, r"the set grows tenfold, so a raw $\hat{D}^2$ would drift down on its own.")

    _figure_title(fig, "Methods", "definitions, significance, and what the answers depend on")
    fig.subplots_adjust(left=0.062, right=0.990, top=0.905, bottom=0.098)
    _series_footer(fig, "01_methods")
    return fig
