"""The two rank-selection figures, one per question the stage answers.

``plot_rank_tradeoff`` is the decision figure: effective sample size against
residual spread, one point per cumulative rank cut, with the Pareto front
marked. Its right-hand column is a typeset methods note rather than a data
panel.

``plot_casectrl_separation`` is the supplementary diagnostic: whether cases and
controls sit at different places *within* each retained set, which residual
spread cannot express. Kept a sibling rather than a third panel -- the two
figures answer different questions and only the first selects a cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from scripts.common import to_numeric_array

if TYPE_CHECKING:
    from scripts.rank_selection import RankSelectionConfig


def plot_rank_tradeoff(
    *,
    decision_table: pd.DataFrame,
    rgv_column: str,
    mainland_axes: Sequence[str],
    config: "RankSelectionConfig",
    cut_label: str = "recommended",
) -> Figure:
    """Return the trade-off figure; the caller styles, saves and closes it.

    ``cut_label`` names the marked cut. It is the variant name -- "narrow" --
    rather than "recommended", which would read as a claim that this cut is
    better than the other two when the whole point is that each buys one thing
    with another.
    """

    # ── Figure & axes layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(18.0, 9.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.8], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.075, right=0.990, top=0.882, bottom=0.115)

    rank_vals = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff_vals = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het_vals = decision_table[rgv_column].to_numpy(dtype=float, copy=False)
    # The axis label has to name the basis: two RGV columns exist and they are
    # on different scales, so an unlabelled axis would be unreadable.
    if config.rgv_basis == "mainland":
        rgv_dim = len(mainland_axes)
        rgv_axis_label = f"mainland PCA, PC1-PC{rgv_dim}"
    else:
        rgv_dim = 2
        rgv_axis_label = "global PCA, PC1-PC2"
    pareto_vals = decision_table["Is_Pareto"].to_numpy(dtype=bool, copy=False)
    rec_vals = decision_table["Is_Recommended"].to_numpy(dtype=bool, copy=False)

    valid_mask = np.isfinite(neff_vals) & np.isfinite(het_vals)
    if bool(np.any(valid_mask)):
        x_valid = het_vals[valid_mask]
        y_valid = neff_vals[valid_mask]
        rank_valid = rank_vals[valid_mask]
        p_valid = pareto_vals[valid_mask]
        r_valid = rec_vals[valid_mask]

        order = np.argsort(rank_valid)
        x_plot = x_valid[order]
        y_plot = y_valid[order]
        rank_plot = rank_valid[order]
        p_plot = p_valid[order]
        r_plot = r_valid[order]

        x_range = float(x_plot.max() - x_plot.min()) if len(x_plot) > 1 else 1.0
        y_range = float(y_plot.max() - y_plot.min()) if len(y_plot) > 1 else 1.0

        # ── All-rank connecting line ──────────────────────────────────
        ax.plot(
            x_plot, y_plot,
            color="#BDBDBD", linewidth=1.2, alpha=0.85,
            zorder=1, solid_capstyle="round",
        )
        # ── All rank scatter points ───────────────────────────────────
        ax.scatter(
            x_plot, y_plot,
            color="#616161", s=60, alpha=0.85,
            edgecolors="white", linewidths=0.7, zorder=2,
        )
        # ── Pareto frontier ───────────────────────────────────────────
        if bool(np.any(p_plot)):
            pf_order = np.argsort(x_plot[p_plot])
            x_pf = x_plot[p_plot][pf_order]
            y_pf = y_plot[p_plot][pf_order]
            ax.plot(
                x_pf, y_pf,
                color="#D32F2F", linewidth=1.8, alpha=0.95,
                zorder=3, solid_capstyle="round",
            )
            ax.scatter(
                x_pf, y_pf,
                s=130, facecolors="none",
                edgecolors="#D32F2F", linewidths=1.6,
                label="Pareto-optimal frontier", zorder=4,
            )
        # ── Recommended rank highlight ────────────────────────────────
        if bool(np.any(r_plot)):
            rec_idx = int(np.where(r_plot)[0][0])
            ax.scatter(
                x_plot[r_plot], y_plot[r_plot],
                marker="D", s=180,
                facecolors="none", edgecolors="#1565C0", linewidths=2.2,
                label=f"{cut_label} cut", zorder=5,
            )
            # build sample-size annotation, placed in lower-right empty space
            rec_k = int(rank_plot[rec_idx])
            rec_row = decision_table[decision_table["Included_Max_Rank"] == rec_k]
            _case_col = f"{config.case_label}_Count"
            _ctrl_col = f"{config.control_label}_Count"
            def _rec_count(col: str) -> int | None:
                """The recommended row's value in a count column, if present."""
                if col not in rec_row.columns:
                    return None
                return int(to_numeric_array(rec_row[col])[0])

            _case_n = _rec_count(_case_col)
            _ctrl_n = _rec_count(_ctrl_col)
            _total_n = _rec_count("Total_Count")
            if _case_n is not None and _ctrl_n is not None and _total_n is not None:
                _ann_lines = [
                    f"k = {rec_k}  ({cut_label})",
                    f"{config.case_label}: {_case_n:,}  |  {config.control_label}: {_ctrl_n:,}",
                    f"Total: {_total_n:,}  (composite posterior)",
                ]
            else:
                _ann_lines = [f"k = {rec_k}  ({cut_label})"]
            # place annotation box in the lower-right empty area of the plot
            ax.annotate(
                "\n".join(_ann_lines),
                xy=(float(x_plot[rec_idx]), float(y_plot[rec_idx])),
                xycoords="data",
                xytext=(0.68, 0.12),
                textcoords="axes fraction",
                fontsize=11.5, fontweight="normal",
                color="#0D2B6E",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "#EEF2FF", "ec": "#1565C0",
                    "alpha": 0.97, "lw": 1.1,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#1565C0",
                    "lw": 1.1,
                    "connectionstyle": "arc3,rad=-0.25",
                },
            )
        # ── Rank number labels (skip recommended k — already annotated) ──
        for i, (xv, yv, kv) in enumerate(zip(x_plot, y_plot, rank_plot)):
            if r_plot[i]:   # recommended k has its own annotation box
                continue
            ax.text(
                float(xv) + x_range * 0.007,
                float(yv) + y_range * 0.012,
                str(int(kv)),
                fontsize=11, ha="left", va="bottom",
                color="#424242", zorder=6,
            )

        # ── Legend + label caption ────────────────────────────────────
        ax.legend(
            loc="lower right", frameon=False,
            handlelength=1.8, handleheight=1.2,
            fontsize=13, borderpad=0.6,
        )
        # Caption explaining the numeric label — italic, top-left, no box
        ax.text(
            0.018, 0.970,
            r"Label = cumulative rank $k$  (top-1 $\ldots$ top-$k$ mainland clusters included)",
            transform=ax.transAxes,
            fontsize=11, color="#757575",
            ha="left", va="top", style="italic",
        )

    ax.set_xlabel(f"Residual spread  $H$  (RGV on {rgv_axis_label})", labelpad=10)
    ax.set_ylabel(r"GWAS  $N_{eff}$", labelpad=10)
    ax.set_title(
        r"Trade-off: Residual Spread vs. GWAS $N_{eff}$",
        loc="left", fontweight="bold", pad=11, fontsize=15,
    )
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.30, color="#9E9E9E")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", length=4.0, width=0.9, pad=5)

    # ── Right panel: Methods / Formulas ──────────────────────────────
    ax_note.set_axis_off()
    ax_note.set_xlim(0.0, 1.0)
    ax_note.set_ylim(0.0, 1.0)

    _BK  = "#212121"   # near-black for headers
    _GR  = "#424242"   # body text
    _DIM = "#757575"   # secondary/dim text
    _X   = 0.05        # left indent

    def _rule(y: float) -> None:
        ax_note.plot([0.0, 1.0], [y, y], color="#E0E0E0", linewidth=0.9)

    # ══ Section 1: N_eff ════════════════════════════════════════════
    _rule(0.978)
    ax_note.text(
        _X, 0.958,
        r"Effective Sample Size  $N_{eff}$",
        fontsize=14.5, fontweight="bold", va="top", ha="left", color=_BK,
    )
    # Step 1
    ax_note.text(
        _X, 0.910,
        "Variance of allele-frequency difference (unbalanced design):",
        fontsize=11.0, fontstyle="italic", va="top", ha="left", color=_DIM,
    )
    ax_note.text(
        _X + 0.04, 0.870,
        r"$\mathrm{Var}=p(1-p)\!\left(\dfrac{1}{N_{\mathrm{case}}}+\dfrac{1}{N_{\mathrm{ctrl}}}\right)$",
        fontsize=13.0, va="top", ha="left", color=_GR,
    )
    # Step 2
    ax_note.text(
        _X, 0.798,
        r"Equate to balanced design  ($N_{eff}/2$ per arm):",
        fontsize=11.0, fontstyle="italic", va="top", ha="left", color=_DIM,
    )
    ax_note.text(
        _X + 0.04, 0.758,
        r"$=\dfrac{2\,p(1-p)}{N_{eff}}$",
        fontsize=13.0, va="top", ha="left", color=_GR,
    )
    # Main result
    ax_note.text(
        _X, 0.695,
        r"$\Rightarrow\;N_{eff}=\dfrac{4\,N_{\mathrm{case}}\,N_{\mathrm{ctrl}}}{"
        r"N_{\mathrm{case}}+N_{\mathrm{ctrl}}}=N_{tot}\cdot\dfrac{4r}{(1+r)^{2}}$",
        fontsize=14.5, va="top", ha="left", color=_BK,
    )
    # Meaning
    ax_note.text(
        _X, 0.612,
        r"Equivalent total $N$ of a balanced (1:1) case-control study",
        fontsize=12.0, va="top", ha="left", color=_GR,
    )
    ax_note.text(
        _X, 0.578,
        "with the same statistical test power.",
        fontsize=12.0, va="top", ha="left", color=_GR,
    )
    ax_note.text(
        _X, 0.542,
        r"($r=N_{\mathrm{case}}/N_{\mathrm{ctrl}},\;\;N_{tot}=N_{\mathrm{case}}+N_{\mathrm{ctrl}}$)",
        fontsize=11.0, va="top", ha="left", color=_DIM,
    )

    _rule(0.505)

    # ══ Section 2: Heterogeneity ════════════════════════════════════
    ax_note.text(
        _X, 0.482,
        f"Residual Spread  $H$  ({rgv_axis_label})",
        fontsize=14.5, fontweight="bold", va="top", ha="left", color=_BK,
    )
    # Main formula
    ax_note.text(
        _X + 0.04, 0.430,
        rf"$H=\left|\,\Sigma\,\right|^{{1/{2 * rgv_dim}}}$",
        fontsize=15.5, va="top", ha="left", color=_BK,
    )
    # Determinant expansion. The closed form below only holds in two
    # dimensions, so it is shown only when the basis actually has two axes.
    if rgv_dim == 2:
        _det_expansion = (
            r"$|\Sigma|=\sigma^{2}_{PC1}\cdot\sigma^{2}_{PC2}-\sigma^{2}_{PC1,PC2}"
            r"=\sigma^{2}_{PC1}\cdot\sigma^{2}_{PC2}(1-\rho^{2})$"
        )
        _closed_form = r"$\Rightarrow H=(\sigma_{PC1}\cdot\sigma_{PC2})^{1/2}(1-\rho^{2})^{1/4}$"
        _meaning = r"Root Generalized Variance (2D): joint spread in PC1/PC2,"
    else:
        _det_expansion = r"$|\Sigma|=\prod_{i=1}^{d}\lambda_i$" + f",  $d={rgv_dim}$"
        _closed_form = r"$\Rightarrow H=\left(\prod_i \sqrt{\lambda_i}\right)^{1/d}$"
        _meaning = f"Root Generalized Variance ({rgv_dim}D): joint spread over PC1-PC{rgv_dim},"
    ax_note.text(
        _X, 0.355, _det_expansion,
        fontsize=11.5, va="top", ha="left", color=_GR,
    )
    # Closed form
    ax_note.text(
        _X, 0.292, _closed_form,
        fontsize=13.0, va="top", ha="left", color=_GR,
    )
    # Meaning
    ax_note.text(
        _X, 0.220, _meaning,
        fontsize=12.0, va="top", ha="left", color=_GR,
    )
    ax_note.text(
        _X, 0.186,
        r"attenuated by inter-PC correlation $\rho$."
        if rgv_dim == 2 else r"the geometric mean of the per-axis SDs.",
        fontsize=12.0, va="top", ha="left", color=_GR,
    )

    _rule(0.072)

    # ══ Section 3: Counting methods ═══════════════════════════════════
    ax_note.text(
        _X, 0.057,
        "Counting methods:",
        fontsize=10.5, fontweight="bold", va="top", ha="left", color=_BK,
    )
    ax_note.text(
        _X, 0.035,
        r"Rank table: per-component argmax (MAP), as in the assignment figure.",
        fontsize=9.5, fontstyle="italic", va="top", ha="left", color=_DIM,
    )
    ax_note.text(
        _X, 0.011,
        r"Scatter: composite posterior recomputation (top-$k$ merged), as in the subcluster stage.",
        fontsize=9.5, fontstyle="italic", va="top", ha="left", color=_DIM,
    )

    fig.suptitle(
        "Mainland Rank-Cumulative Trade-off Analysis",
        fontsize=18, fontweight="bold", y=0.972,
    )
    _series_footer(fig, "02_tradeoff")
    return fig


# Shared with the trade-off figure's note column so the two read as one document.
_BK, _GR, _DIM = "#212121", "#424242", "#757575"
_X_INDENT = 0.05


#: The three rank-selection figures, in reading order. Each names its own place
#: and its neighbours, so one lifted out of the directory still says where it
#: sits in the argument.
FIGURE_SERIES: "tuple[tuple[str, str], ...]" = (
    ("00_overview", "the cohorts and the basis for each"),
    ("02_tradeoff", "what is being traded"),
    ("03_separation", "the second homogeneity axis"),
    ("04_cut_selection", "how the two of them fix the cuts"),
)


def _series_footer(fig: Figure, name: str) -> None:
    """Stamp the figure's place in the series along the bottom.

    A footer rather than a subtitle: it is metadata about where to read this
    figure, not part of what the figure says, and the titles are already
    carrying the argument.
    """
    fig.text(0.5, 0.013, _series_tag(name), fontsize=10.5, color=_DIM,
             ha="center", va="bottom", fontstyle="italic")


def _series_tag(name: str) -> str:
    """"2 of 3 · ... — after X, before Y", for a figure's subtitle."""
    names = [n for n, _ in FIGURE_SERIES]
    i = names.index(name)
    parts = [f"Rank selection · {i + 1} of {len(names)} · {FIGURE_SERIES[i][1]}"]
    if i > 0:
        parts.append(f"after {names[i - 1]}.png")
    if i < len(names) - 1:
        parts.append(f"before {names[i + 1]}.png")
    return "   —   ".join(parts)


#: Per-cut identity, shared by every figure that names a cut.
_TINT = {"full": "#FBEAEC", "narrow": "#E7F1F8", "intermediate": "#E6F4EC"}
_EDGE = {"full": "#B2182B", "narrow": "#0571B0", "intermediate": "#008837"}


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


def plot_casectrl_separation(
    *,
    decision_table: pd.DataFrame,
    mainland_axes: Sequence[str],
    variant_cuts: "Mapping[str, int] | None" = None,
) -> Figure:
    """Return the case/control separation diagnostic.

    Purely a diagnostic: it carries the question residual spread cannot answer
    -- whether cases and controls are drawn from the same place *within* the
    retained set -- and no cut is selected on it. The reasons that would make
    selecting on it wrong are in ``scripts.common.case_ctrl_separation``; the
    cuts themselves are derived in ``plot_rank_cut_selection``.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    d_raw = decision_table["Mainland_CaseCtrl_Mahalanobis"].to_numpy(dtype=float, copy=False)
    d2_unb = decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float, copy=False)
    floor = decision_table["Mainland_CaseCtrl_Noise_Floor"].to_numpy(dtype=float, copy=False)
    pval = decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False)
    d2_raw = d_raw ** 2
    cuts = dict(variant_cuts or {})

    _RAW, _UNB, _FLOOR, _SIG = "#7B3294", "#008837", "#BDBDBD", "#D7191C"

    fig = plt.figure(figsize=(19.0, 9.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 2.75], wspace=0.055)
    gs_l = gs[0, 0].subgridspec(2, 1, height_ratios=[1.06, 0.94], hspace=0.20)
    ax_d = fig.add_subplot(gs_l[0, 0])
    ax_p = fig.add_subplot(gs_l[1, 0], sharex=ax_d)
    ax_note = fig.add_subplot(gs[0, 1])

    # ══ A · How far apart, and how much of that is sampling ══════════
    ax_d.fill_between(
        rank, 0.0, floor, color=_FLOOR, alpha=0.55, linewidth=0, zorder=1,
        label=rf"Sampling floor  $p\,(1/n_1 + 1/n_2)$,  $p={n_dim}$",
    )
    ax_d.axhline(0.0, color=_DIM, linewidth=0.9, linestyle=":", zorder=2)
    ax_d.plot(rank, d2_raw, "-o", color=_RAW, markersize=6.0, linewidth=1.9,
              markeredgecolor="white", markeredgewidth=0.9, zorder=4,
              label=r"Observed  $\hat{D}^2$")
    ax_d.plot(rank, d2_unb, "-s", color=_UNB, markersize=5.6, linewidth=1.9,
              markeredgecolor="white", markeredgewidth=0.9, zorder=5,
              label=r"De-biased  $\hat{D}^2 - E[D^2]$")
    ax_d.set_ylabel(r"Mahalanobis  $D^2$")
    ax_d.set_title("A · Separation of the case and control centroids",
                   fontsize=14, fontweight="bold", loc="left", pad=9)
    ax_d.legend(loc="center right", frameon=True, framealpha=0.95,
                edgecolor="#CFCFCF", fontsize=11.5)
    ax_d.tick_params(labelbottom=False)

    # ══ B · The evidence, which needs no bias correction ═════════════
    sig = np.isfinite(pval) & (pval < 0.05)
    ax_p.plot(rank, pval, "-", color=_GR, linewidth=1.5, zorder=3)
    ax_p.scatter(rank[sig], pval[sig], s=78, color=_SIG, edgecolor="white",
                 linewidth=1.0, zorder=5, label=r"$p < 0.05$")
    ax_p.scatter(rank[~sig], pval[~sig], s=78, color="#FFFFFF", edgecolor=_GR,
                 linewidth=1.5, zorder=5, label=r"$p \geq 0.05$")
    ax_p.axhline(0.05, color=_SIG, linewidth=1.1, linestyle="--", zorder=2)
    ax_p.set_yscale("log")
    ax_p.set_ylabel(r"Hotelling's $T^2$  $p$")
    ax_p.set_title(r"B · Evidence for it. The exact $F$ test already accounts for sampling.",
                   fontsize=14, fontweight="bold", loc="left", pad=9)
    ax_p.legend(loc="lower left", frameon=True, framealpha=0.95,
                edgecolor="#CFCFCF", fontsize=11.5, ncol=2)
    ax_p.set_xlabel(r"Cumulative rank $k$   (top-1 … top-$k$ mainland components included)")
    ax_p.set_xticks(rank)

    _mark_cuts((ax_d, ax_p), cuts, ax_d, int(rank.min()), int(rank.max()))
    for axis in (ax_d, ax_p):
        axis.set_xlim(rank.min() - 0.4, rank.max() + 0.4)
        axis.grid(True, alpha=0.30, linewidth=0.7)
        axis.set_axisbelow(True)

    # ══ Right column: methods ════════════════════════════════════════
    _note_axis(ax_note)

    def _rule(y: float) -> None:
        ax_note.plot([0.0, 1.0], [y, y], color="#E0E0E0", linewidth=0.9)

    def _h(y: float, txt: str) -> None:
        ax_note.text(_X_INDENT, y, txt, fontsize=14, fontweight="bold", va="top",
                     ha="left", color=_BK)

    def _m(y: float, txt: str, size: float = 15.0) -> None:
        ax_note.text(0.5, y, txt, fontsize=size, va="top", ha="center", color=_BK)

    def _b(y: float, txt: str, size: float = 12.0, style: str = "normal",
           color: str = _GR) -> None:
        ax_note.text(_X_INDENT, y, txt, fontsize=size, va="top", ha="left",
                     color=color, fontstyle=style)

    _rule(0.980)
    _h(0.964, "The statistic")
    _m(0.918, r"$D^2 = (\bar{x}_{1} - \bar{x}_{2})^{\top}\,"
              r"S_{\mathrm{pooled}}^{-1}\,(\bar{x}_{1} - \bar{x}_{2})$")
    _m(0.862, r"$T^2 = D^2\,\dfrac{n_1 n_2}{n_1 + n_2}$"
              r"$\qquad F = \dfrac{T^2\,(\nu - p + 1)}{p\,\nu}\ \sim\ F_{p,\ \nu-p+1}$", 13.5)
    _b(0.790, r"$\bar{x}_1, \bar{x}_2$ are the case and control centroids and")
    _b(0.766, r"$S_{\mathrm{pooled}}$ the within-group covariance pooled over both,")
    _b(0.742, r"with $\nu = n_1 + n_2 - 2$. $D$ is in pooled-SD units and so is")
    _b(0.718, r"scale-free; $T^2$ weights it by sample size, which is what")
    _b(0.694, r"turns a distance into evidence.")
    _b(0.664, rf"Mainland PCA, PC1-PC{n_dim} — the axes RGV uses, on the same",
       color=_DIM, style="italic")
    _b(0.640, "retained samples.", color=_DIM, style="italic")

    _rule(0.618)
    _h(0.602, r"Why a raw $D^2$ cannot be compared across cuts")
    _m(0.554, r"$E\left[\hat{D}^2\right] = D^2_{\mathrm{true}}"
              r" + p\left(\dfrac{1}{n_1} + \dfrac{1}{n_2}\right)$", 14.0)
    _b(0.482, "Two sample means never coincide, and $D^2$ squares the gap, so")
    _b(0.458, "sampling scatter can only add: the axis count multiplies the")
    _b(0.434, "contribution and the group sizes divide it. The retained set")
    _b(0.410, r"grows more than tenfold across the walk, so this floor falls")
    _b(0.386, r"with it and a raw $\hat{D}^2$ drifts down for arithmetic reasons")
    _b(0.362, "alone. The de-biased column subtracts it, and goes negative")
    _b(0.338, "where the separation sits below the floor.")
    _b(0.308, r"$T^2$ and $p$ need no correction — the $F$ test is exact.",
       color=_DIM, style="italic")

    _rule(0.286)
    _h(0.270, "Diagnostic only — never optimised against")
    _b(0.233, r"Minimising it would optimise the quantity the association test")
    _b(0.209, r"measures, and it bottoms out near the uncut set where $N_{eff}$ is")
    _b(0.185, r"largest — so it would collapse the trade-off rather than inform it.")
    _b(0.155, r"The full argument: docs/method.md § 6.", color=_DIM, style="italic")
    _b(0.115, r"One of two inputs to the intermediate cut is the de-biased",
       size=11.0, color=_BK)
    _b(0.091, r"column above; this $p$-value is an input to nothing.",
       size=11.0, color=_BK)

    fig.suptitle("Supplementary · Case/Control Ancestry Separation Across the Rank Walk",
                 fontsize=18, fontweight="bold", y=0.972)
    fig.subplots_adjust(left=0.058, right=0.992, top=0.905, bottom=0.098)
    _series_footer(fig, "03_separation")
    return fig


def _flow_box(
    ax: "plt.Axes", x: float, y: float, w: float, h: float, text: str,
    *, face: str, edge: str, bold: bool = False, size: float = 10.0,
) -> tuple[float, float]:
    """One node of the decision flow. Returns its bottom-centre anchor."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2.0, y - h / 2.0), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=1.1, zorder=3,
    ))
    ax.text(x, y, text, fontsize=size, ha="center", va="center", zorder=4,
            color=_BK, fontweight="bold" if bold else "normal", linespacing=1.35)
    return x, y - h / 2.0


def _flow_arrow(ax: "plt.Axes", x: float, y0: float, y1: float, colour: str) -> None:
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.3,
                                shrinkA=0, shrinkB=0))


def plot_rank_cut_selection(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    objective_spaces: "Mapping[str, object]",
    mainland_axes: Sequence[str],
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    blend_weight: float,
    rank_cuts: "Mapping[str, int | None]",
    mode: str,
    case_label: str = "Case",
    control_label: str = "Control",
    safe_weight_floor: float = 0.5,
) -> Figure:
    """Return the figure that derives the delivered cuts.

    Three data panels -- the two derivations, then what they depend on -- beside
    a column that *draws* the procedure rather than describing it. The three
    cuts are one question asked three ways, and a branching diagram carries that
    in a glance where paragraphs did not. Everything the diagram omits is in
    ``docs/method.md``; nothing here needs to restate it.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    y = decision_table["Neff_Norm"].to_numpy(dtype=float, copy=False)
    x = decision_table["RGV_Norm"].to_numpy(dtype=float, copy=False)
    spread = objective_spaces["narrow"]
    blended = objective_spaces["intermediate"]
    b = np.asarray(blended.structure, dtype=float)

    cuts = {str(n): int(k) for n, k in rank_cuts.items() if k is not None}
    row = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_full = int(row["full"]["Resolved_Rank"])
    k_knee = int(row["narrow"]["Resolved_Rank"])
    k_blend = int(row["intermediate"]["Resolved_Rank"])

    _NARROW, _INTER, _FULL = "#0571B0", "#008837", "#B2182B"
    _CHORD = "#B0B0B0"

    fig = plt.figure(figsize=(19.5, 11.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 2.95], wspace=0.055)
    gs_l = gs[0, 0].subgridspec(2, 2, height_ratios=[1.16, 0.84], hspace=0.30, wspace=0.20)
    ax_mo = fig.add_subplot(gs_l[0, 0])
    ax_bl = fig.add_subplot(gs_l[0, 1])
    ax_w = fig.add_subplot(gs_l[1, :])
    ax_note = fig.add_subplot(gs[0, 1])

    # ══ A · Counting spread only — a monotone frontier ═══════════════
    ax_mo.plot([x[0], x[-1]], [y[0], y[-1]], "-", color=_CHORD, linewidth=2.0,
               zorder=2, label="chord joining the ends")
    ax_mo.plot(x, y, "-o", color=_GR, markersize=5.0, linewidth=1.6,
               markerfacecolor="white", markeredgewidth=1.2, zorder=4)
    i_knee = int(np.argmin(np.abs(rank - k_knee)))
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    t = ((x[i_knee] - x[0]) * dx + (y[i_knee] - y[0]) * dy) / (dx ** 2 + dy ** 2)
    ax_mo.plot([x[i_knee], x[0] + t * dx], [y[i_knee], y[0] + t * dy], "-",
               color=_NARROW, linewidth=2.6, zorder=5)
    ax_mo.plot([x[i_knee]], [y[i_knee]], "D", color=_NARROW, markersize=12.0,
               markeredgecolor="white", markeredgewidth=1.4, zorder=7,
               label=rf"narrow — knee, $k$ = {k_knee}")
    i_full = int(np.argmin(np.abs(rank - k_full)))
    ax_mo.plot([x[i_full]], [y[i_full]], "o", color=_FULL, markersize=12.0,
               markeredgecolor="white", markeredgewidth=1.4, zorder=7,
               label=rf"full — $\arg\max N_{{eff}}$, $k$ = {k_full}")
    for i, k in enumerate(rank):
        if i in (i_knee, i_full):
            continue
        ax_mo.annotate(str(int(k)), (x[i], y[i]), textcoords="offset points",
                       xytext=(0, -14) if i % 2 == 0 else (0, 8), fontsize=9.0,
                       color=_DIM, ha="center")
    ax_mo.set_xlabel("residual spread  (normalised)")
    ax_mo.set_ylabel(r"$N_{eff}$  (normalised)")
    ax_mo.set_title(f"A · Counting spread — monotone, chord spans {spread.chord_span:.3f}",
                    fontsize=13.5, fontweight="bold", loc="left", pad=9)
    ax_mo.legend(loc="lower right", frameon=True, framealpha=0.95,
                 edgecolor="#CFCFCF", fontsize=10.0)

    # ══ B · Counting both — a folded axis, so the knee cannot apply ══
    ax_bl.plot([b[0], b[-1]], [y[0], y[-1]], "-", color=_CHORD, linewidth=2.0,
               zorder=2, label="chord — near vertical, no leverage")
    ax_bl.plot(b, y, "-o", color=_GR, markersize=5.0, linewidth=1.2, alpha=0.75,
               markerfacecolor="white", markeredgewidth=1.2, zorder=4)
    i_bl = int(np.argmin(np.abs(rank - k_blend)))
    ax_bl.plot([0.0], [1.0], "*", color=_INTER, markersize=20.0, zorder=6,
               markeredgecolor="white", markeredgewidth=1.0,
               label="ideal corner (unattainable)")
    ax_bl.plot([0.0, b[i_bl]], [1.0, y[i_bl]], "--", color=_INTER, linewidth=2.2, zorder=5)
    ax_bl.plot([b[i_bl]], [y[i_bl]], "s", color=_INTER, markersize=12.0,
               markeredgecolor="white", markeredgewidth=1.4, zorder=7,
               label=rf"intermediate — nearest, $k$ = {k_blend}")
    for i, k in enumerate(rank):
        if i == i_bl:
            continue
        ax_bl.annotate(str(int(k)), (b[i], y[i]), textcoords="offset points",
                       xytext=(0, -14) if i % 2 == 0 else (0, 8), fontsize=9.0,
                       color=_DIM, ha="center")
    ax_bl.set_xlabel(rf"$H(w)$ at $w = {blend_weight:g}$  (normalised)")
    ax_bl.set_title(f"B · Counting both — folded, chord spans only {blended.chord_span:.3f}",
                    fontsize=13.5, fontweight="bold", loc="left", pad=9)
    ax_bl.legend(loc="lower right", frameon=True, framealpha=0.95,
                 edgecolor="#CFCFCF", fontsize=10.0)

    # ══ C · What each answer depends on ══════════════════════════════
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
        ax_w.plot([lo, hi], [k_blend, k_blend], color=_INTER, linewidth=7.0,
                  solid_capstyle="butt", alpha=0.85, zorder=5)
        ax_w.axvline(blend_weight, color=_INTER, linewidth=1.6, linestyle="-.", zorder=6)
        ax_w.plot([blend_weight], [k_blend], "s", color=_INTER, markersize=11.0,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=8)
        ax_w.annotate(
            rf"$k$ = {k_blend} holds on $w \in [{lo:.2f},\ {hi:.2f}]$ — "
            rf"evaluated {min(blend_weight - lo, hi - blend_weight):.2f} from the nearest edge",
            xy=(blend_weight, k_blend), xytext=(0, 16), textcoords="offset points",
            fontsize=11.0, color=_BK, ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=_INTER,
                      linewidth=1.2, alpha=0.96), zorder=9,
        )
    for name, k in sorted(cuts.items(), key=lambda kv: kv[1]):
        if np.any(weight_winner == k):
            continue
        colour = _EDGE[name]
        ax_w.axhline(k, color=colour, linewidth=1.1, linestyle=":", alpha=0.9, zorder=2)
        why = (rf"{name} ($k$={k}) — fixed in panel A, lead {float(row[name]['Margin']):+.4f}"
               if name == "narrow" else rf"{name} ($k$={k}) — fixed in panel A")
        ax_w.text(0.995, k, why, color=colour, fontsize=10.5, va="center", ha="right",
                  fontstyle="italic", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="none", alpha=0.92))
    ax_w.set_ylim(int(won.min()) - 1, max(int(won.max()), max(cuts.values(), default=0)) + 1)
    ax_w.set_xlim(0.0, 1.0)
    ax_w.set_yticks(sorted(set(int(v) for v in np.unique(won)) | set(cuts.values())))
    ax_w.set_ylabel(r"winning cut  $k^{*}(w)$")
    ax_w.set_xlabel(r"$w$ — weight on residual spread;  $1-w$ on case/control separation")
    ax_w.set_title("C · What the answers depend on — only the blend has a weight to depend on",
                   fontsize=13.5, fontweight="bold", loc="left", pad=9)

    for axis in (ax_mo, ax_bl, ax_w):
        axis.grid(True, alpha=0.30, linewidth=0.7)
        axis.set_axisbelow(True)

    # ══ Right column: the procedure, drawn ═══════════════════════════
    _note_axis(ax_note)
    ax_note.text(0.5, 0.985, "One question, asked three ways",
                 fontsize=14.5, fontweight="bold", ha="center", va="top", color=_BK)
    ax_note.text(0.5, 0.949, "what counts as residual structure?",
                 fontsize=12.0, ha="center", va="top", color=_GR, fontstyle="italic")

    # Branch geometry: one column per cut, four tiers down the column.
    lanes = {"full": 0.175, "narrow": 0.500, "intermediate": 0.825}
    tiers = (0.868, 0.758, 0.648, 0.534)
    counted = {"full": "nothing", "narrow": "residual\nspread",
               "intermediate": "spread  +\ncase/control\nseparation"}
    geometry = {
        "full": "no trade\nto make",
        "narrow": f"monotone\nchord spans {spread.chord_span:.3f}",
        "intermediate": f"folded\nchord spans {blended.chord_span:.3f}",
    }
    operator = {"full": r"$\arg\max N_{eff}$", "narrow": "knee",
                "intermediate": "nearest ideal"}
    answer = {"full": k_full, "narrow": k_knee, "intermediate": k_blend}

    ax_note.annotate("", xy=(lanes["full"], 0.925), xytext=(lanes["intermediate"], 0.925),
                     arrowprops=dict(arrowstyle="-", color="#CFCFCF", linewidth=1.1), zorder=1)
    for name, cx in lanes.items():
        tint, edge = _TINT[name], _EDGE[name]
        _flow_arrow(ax_note, cx, 0.925, tiers[0] + 0.038, edge)
        prev = None
        for tier, label, bold, size in (
            (tiers[0], counted[name], False, 10.0),
            (tiers[1], geometry[name], False, 10.0),
            (tiers[2], operator[name], True, 11.0),
            (tiers[3], f"{name}\n$k$ = {answer[name]}", True, 12.5),
        ):
            _, bottom = _flow_box(ax_note, cx, tier, 0.275,
                                  0.086 if tier == tiers[3] else 0.072, label,
                                  face=tint, edge=edge, bold=bold, size=size)
            if prev is not None:
                _flow_arrow(ax_note, cx, prev,
                            tier + (0.043 if tier == tiers[3] else 0.036), edge)
            prev = bottom

    # The defining equations only; the derivations are in docs/method.md.
    ax_note.plot([0.0, 1.0], [0.470, 0.470], color="#E0E0E0", linewidth=0.9)
    ax_note.text(_X_INDENT, 0.454, "Definitions", fontsize=12.5, fontweight="bold",
                 va="top", ha="left", color=_BK)
    ax_note.text(0.5, 0.410,
                 r"$x_k,\ y_k,\ s_k$ : spread, $N_{eff}$, de-biased $D^2$, each min-max to $[0,1]$",
                 fontsize=11.0, ha="center", va="top", color=_GR)
    ax_note.text(0.5, 0.358,
                 r"$d_k = \dfrac{\left|\Delta y\,x_k - \Delta x\,y_k "
                 r"+ x_K y_1 - y_K x_1\right|}{\sqrt{\Delta x^2 + \Delta y^2}}$"
                 r"$\qquad$"
                 r"$H_k = \dfrac{x_k + s_k}{2}$",
                 fontsize=12.5, ha="center", va="top", color=_BK)
    ax_note.text(0.5, 0.276,
                 r"knee $= \arg\max_k d_k$"
                 r"$\qquad$"
                 r"nearest ideal $= \arg\min_k \sqrt{H_k^2 + (1-y_k)^2}$",
                 fontsize=12.0, ha="center", va="top", color=_BK)
    ax_note.text(_X_INDENT, 0.228,
                 "Derivations, margins and caveats: docs/method.md § 6b",
                 fontsize=10.0, va="top", ha="left", color=_DIM, fontstyle="italic")

    # ── What the procedure produced ──────────────────────────────────
    ax_note.plot([0.0, 1.0], [0.204, 0.204], color="#E0E0E0", linewidth=0.9)
    ax_note.text(_X_INDENT, 0.188, f"What it produced   (mode: {mode})",
                 fontsize=12.5, fontweight="bold", va="top", ha="left", color=_BK)
    hdr = ("cut", "k", case_label, control_label, "n", r"$N_{eff}$", "RGV")
    xs = (0.05, 0.230, 0.330, 0.470, 0.610, 0.725, 0.855)
    for xx, t in zip(xs, hdr):
        ax_note.text(xx, 0.148, t, fontsize=10.0, va="top", ha="left", color=_DIM,
                     fontstyle="italic")
    ax_note.plot([0.0, 1.0], [0.129, 0.129], color="#E0E0E0", linewidth=0.9)
    for i, (_, r) in enumerate(cut_selection.iterrows()):
        name = str(r["Variant"])
        k = int(r["Resolved_Rank"])
        d_row = decision_table.loc[decision_table["Included_Max_Rank"] == k]
        def _v(col: str) -> str:
            if col not in d_row.columns or d_row.empty:
                return "—"
            val = float(to_numeric_array(d_row[col])[0])
            return f"{val:,.0f}" if col != "RGV_Mainland" else f"{val:.5f}"
        yy = 0.110 - i * 0.032
        vals = (name, str(k),
                _v(f"{case_label}_Count"), _v(f"{control_label}_Count"),
                _v("Total_Count"), _v("GWAS_Neff"), _v("RGV_Mainland"))
        for j, (xx, t) in enumerate(zip(xs, vals)):
            ax_note.text(xx, yy, t, fontsize=10.5, va="top", ha="left",
                         color=_EDGE.get(name, _BK) if j == 0 else _GR,
                         fontweight="bold" if j == 0 else "normal")

    fig.suptitle(f"Deriving the Delivered Cuts — Mainland PCA, PC1-PC{n_dim}",
                 fontsize=18, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.918, bottom=0.082)
    _series_footer(fig, "04_cut_selection")
    return fig


def plot_cut_overview(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rank_table: pd.DataFrame,
    rgv_column: str,
    mainland_axes: Sequence[str],
    case_label: str = "Case",
    control_label: str = "Control",
) -> Figure:
    """Return the one-glance summary of what was chosen and on what basis.

    The other three figures each carry one step of the argument. This one is
    read first and answers the two questions a reader actually arrives with:
    *what are the three cohorts*, and *what fixed each of them*. Everything on
    it is drawn from the same decision table the later figures use, so it
    restates rather than adds.
    """
    n_dim = len(mainland_axes)
    ranks = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["GWAS_Neff"].to_numpy(dtype=float, copy=False)
    het = decision_table[rgv_column].to_numpy(dtype=float, copy=False)

    rows = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    order = sorted(rows, key=lambda n: int(rows[n]["Resolved_Rank"]))   # narrow -> full
    k_of = {n: int(rows[n]["Resolved_Rank"]) for n in order}
    rank_sorted = rank_table.sort_values("Rank")
    comps = {
        n: [int(c) for c in rank_sorted.loc[rank_sorted["Rank"] <= k_of[n], "Cluster"]]
        for n in order
    }

    def d_row(name: str) -> "pd.Series":
        return decision_table.loc[decision_table["Included_Max_Rank"] == k_of[name]].iloc[0]

    def val(name: str, col: str) -> float:
        r = d_row(name)
        return float(to_numeric_array(r[[col]])[0]) if col in r.index else float("nan")

    fig = plt.figure(figsize=(19.0, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], width_ratios=[1.0, 1.12],
                          hspace=0.24, wspace=0.14)
    ax_set = fig.add_subplot(gs[0, 0])
    ax_loc = fig.add_subplot(gs[0, 1])
    ax_tab = fig.add_subplot(gs[1, :])

    # ══ A · The three cohorts are nested ═════════════════════════════
    _note_axis(ax_set)
    ax_set.set_title("A · The three cohorts are nested sets",
                     fontsize=14, fontweight="bold", loc="left", pad=10)
    # Outermost first, so each smaller set is drawn over the one containing it.
    for depth, name in enumerate(reversed(order)):
        inset = 0.085 * depth
        x0, y0 = 0.045 + inset, 0.075 + inset
        w, h = 0.910 - 2 * inset, 0.815 - 2 * inset
        ax_set.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
            facecolor=_TINT[name], edgecolor=_EDGE[name], linewidth=2.0,
            zorder=2 + depth,
        ))
        ax_set.text(
            x0 + 0.018, y0 + h - 0.030,
            f"{name}   $k$ = {k_of[name]}",
            fontsize=13.5, fontweight="bold", ha="left", va="top",
            color=_EDGE[name], zorder=10 + depth,
        )
        ax_set.text(
            x0 + w - 0.018, y0 + h - 0.030,
            f"{len(comps[name])} components   n = {val(name, 'Total_Count'):,.0f}",
            fontsize=11.5, ha="right", va="top", color=_GR, zorder=10 + depth,
        )

    # What each widening step buys, written in the band it opens up.
    for depth, name in enumerate(order[:-1]):
        outer = order[depth + 1]
        added = sorted(set(comps[outer]) - set(comps[name]))
        d_n = val(outer, "Total_Count") - val(name, "Total_Count")
        band_y = 0.075 + 0.085 * (len(order) - 2 - depth) + 0.030
        ax_set.text(
            0.5, band_y,
            f"+{len(added)} components ({', '.join(str(c) for c in added)})"
            f"   +{d_n:,.0f} samples",
            fontsize=11.0, ha="center", va="bottom", color=_EDGE[outer],
            zorder=20, fontstyle="italic",
        )
    inner = order[0]
    ax_set.text(
        0.5, 0.505,
        "components, in rank order\n" + "   ".join(str(c) for c in comps[inner]),
        fontsize=12.0, ha="center", va="center", color=_EDGE[inner],
        zorder=15, linespacing=1.9, fontweight="bold",
    )
    ax_set.text(0.5, 0.028,
                "Every sample in a smaller cohort is in every larger one — "
                "the cuts are nested by construction.",
                fontsize=10.5, ha="center", va="bottom", color=_DIM, fontstyle="italic")

    # ══ B · Where each sits on the trade-off ═════════════════════════
    ax_loc.plot(het, neff, "-o", color=_GR, markersize=4.6, linewidth=1.5,
                markerfacecolor="white", markeredgewidth=1.1, zorder=3,
                label="the 17 cumulative cuts")
    for name in order:
        r = d_row(name)
        ax_loc.plot([float(r[rgv_column])], [float(r["GWAS_Neff"])], "o",
                    color=_EDGE[name], markersize=15.0, markeredgecolor="white",
                    markeredgewidth=1.8, zorder=6)
        ax_loc.annotate(
            f"{name}\n$k$ = {k_of[name]}",
            xy=(float(r[rgv_column]), float(r["GWAS_Neff"])),
            # Placed off the curve on the side each point has room on; the
            # cuts sit close together on the flat arm.
            xytext={"narrow": (-22, -36), "intermediate": (10, 26),
                    "full": (-10, -30)}.get(name, (14, -4)),
            textcoords="offset points", fontsize=11.5, fontweight="bold",
            color=_EDGE[name], ha={"intermediate": "left"}.get(name, "right"),
            va="center", zorder=7,
        )
    ax_loc.set_xlabel(rf"residual spread — RGV on mainland PC1-PC{n_dim}   $\rightarrow$ less homogeneous")
    ax_loc.set_ylabel(r"GWAS $N_{eff}$   $\rightarrow$ more power")
    ax_loc.set_title("B · and where each sits on the trade-off",
                     fontsize=14, fontweight="bold", loc="left", pad=10)
    ax_loc.legend(loc="lower right", frameon=True, framealpha=0.95,
                  edgecolor="#CFCFCF", fontsize=11.0)
    ax_loc.grid(True, alpha=0.30, linewidth=0.7)
    ax_loc.set_axisbelow(True)
    ax_loc.margins(x=0.16, y=0.10)

    # ══ C · The basis, beside the consequence ════════════════════════
    _note_axis(ax_tab)
    ax_tab.set_title("C · What fixed each cut, and what it delivers",
                     fontsize=14, fontweight="bold", loc="left", pad=10)
    basis = {
        "narrow": "counts residual spread — knee of the trade-off curve",
        "intermediate": "counts spread and case/control separation — nearest the ideal corner",
        "full": "counts nothing — every major-cluster component",
    }
    cols = ("cohort", "basis", "$k$", case_label, control_label,
            "n", r"$N_{eff}$", "RGV")
    xs = (0.008, 0.098, 0.505, 0.606, 0.694, 0.784, 0.858, 0.932)
    ax_tab.plot([0.0, 1.0], [0.865, 0.865], color="#CFCFCF", linewidth=1.0)
    for xx, c in zip(xs, cols):
        ax_tab.text(xx, 0.815, c, fontsize=11.0, ha="left", va="top",
                    color=_DIM, fontstyle="italic")
    ax_tab.plot([0.0, 1.0], [0.700, 0.700], color="#CFCFCF", linewidth=1.0)
    for i, name in enumerate(order):
        yy = 0.600 - i * 0.200
        ax_tab.add_patch(FancyBboxPatch(
            (0.0, yy - 0.072), 1.0, 0.150,
            boxstyle="square,pad=0", facecolor=_TINT[name], edgecolor="none",
            alpha=0.55, zorder=1,
        ))
        vals = (
            name, basis[name], str(k_of[name]),
            f"{val(name, f'{case_label}_Count'):,.0f}",
            f"{val(name, f'{control_label}_Count'):,.0f}",
            f"{val(name, 'Total_Count'):,.0f}",
            f"{val(name, 'GWAS_Neff'):,.0f}",
            f"{val(name, rgv_column):.5f}",
        )
        for j, (xx, t) in enumerate(zip(xs, vals)):
            ax_tab.text(xx, yy, t, fontsize=12.0 if j == 0 else 11.5,
                        ha="left", va="center", zorder=3,
                        color=_EDGE[name] if j == 0 else _BK,
                        fontweight="bold" if j in (0, 2) else "normal",
                        fontstyle="italic" if j == 1 else "normal")
    ax_tab.text(0.0, 0.010,
                "Neither cut is the better list: a narrower set buys homogeneity with "
                "effective sample size, and a broader one the reverse. "
                "Derivations: 04_cut_selection.png.",
                fontsize=10.5, ha="left", va="bottom", color=_DIM, fontstyle="italic")

    fig.suptitle("Rank Selection — the three delivered cohorts and the basis for each",
                 fontsize=18, fontweight="bold", y=0.972)
    fig.subplots_adjust(left=0.050, right=0.988, top=0.905, bottom=0.070)
    _series_footer(fig, "00_overview")
    return fig
