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
    _series_footer(fig, "rank_selection_tradeoff")
    return fig


# Shared with the trade-off figure's note column so the two read as one document.
_BK, _GR, _DIM = "#212121", "#424242", "#757575"
_X_INDENT = 0.05


#: The three rank-selection figures, in reading order. Each names its own place
#: and its neighbours, so one lifted out of the directory still says where it
#: sits in the argument.
FIGURE_SERIES: "tuple[tuple[str, str], ...]" = (
    ("rank_selection_tradeoff", "what is being traded"),
    ("casectrl_separation", "the second homogeneity axis"),
    ("rank_cut_selection", "the cuts the two of them fix"),
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
    _b(0.233, r"1.  It is computed from the case/control labels, so minimising")
    _b(0.209, r"     it optimises the very quantity the association test measures;")
    _b(0.185, r"     an ancestry-linked risk would be selected away with the")
    _b(0.161, r"     confounding.")
    _b(0.131, r"2.  It is not monotone in $k$ ($\rho = -0.73$, against $+1.00$ for")
    _b(0.107, r"     RGV) and bottoms out near the uncut set, where $N_{eff}$ is")
    _b(0.083, r"     largest — so it would collapse the trade-off, not inform it.")
    _b(0.045, r"The cuts are derived in rank_cut_selection.png, which uses the",
       size=11.0, color=_BK)
    _b(0.021, r"de-biased column as one of two inputs but never this $p$.",
       size=11.0, color=_BK)

    fig.suptitle("Supplementary · Case/Control Ancestry Separation Across the Rank Walk",
                 fontsize=18, fontweight="bold", y=0.972)
    fig.subplots_adjust(left=0.058, right=0.992, top=0.905, bottom=0.098)
    _series_footer(fig, "casectrl_separation")
    return fig


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
    safe_weight_floor: float = 0.5,
) -> Figure:
    """Return the figure that derives the delivered cuts.

    Laid out as the argument runs -- define the space, derive, derive, validate:

    ``A`` establishes the three quantities and shows which are monotone, which
    is what decides the operator. ``B`` and ``C`` sit side by side because they
    are the same procedure at two settings of what counts as residual structure.
    ``D`` reports what each answer depends on.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    y = decision_table["Neff_Norm"].to_numpy(dtype=float, copy=False)
    x = decision_table["RGV_Norm"].to_numpy(dtype=float, copy=False)
    spread = objective_spaces["narrow"]
    blended = objective_spaces["intermediate"]
    b = np.asarray(blended.structure, dtype=float)
    s_raw = _safe_norm(decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float, copy=False))

    cuts = {str(n): int(k) for n, k in rank_cuts.items() if k is not None}
    row = {str(r["Variant"]): r for _, r in cut_selection.iterrows()}
    k_full = int(row["full"]["Resolved_Rank"])
    k_knee = int(row["narrow"]["Resolved_Rank"])
    k_blend = int(row["intermediate"]["Resolved_Rank"])

    _SPREAD, _POWER, _SEP = "#0571B0", "#212121", "#008837"
    _NARROW, _INTER, _FULL = "#0571B0", "#008837", "#B2182B"
    _CHORD, _UNSAFE = "#B0B0B0", "#F4C7C3"

    fig = plt.figure(figsize=(20.0, 13.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 2.75], wspace=0.055)
    gs_l = gs[0, 0].subgridspec(3, 2, height_ratios=[0.78, 1.12, 0.80], hspace=0.36, wspace=0.20)
    ax_sp = fig.add_subplot(gs_l[0, :])
    ax_mo = fig.add_subplot(gs_l[1, 0])
    ax_bl = fig.add_subplot(gs_l[1, 1])
    ax_w = fig.add_subplot(gs_l[2, :])
    ax_note = fig.add_subplot(gs[0, 1])

    # ══ A · The objective space ══════════════════════════════════════
    ax_sp.plot(rank, y, "-o", color=_POWER, markersize=5.0, linewidth=1.8,
               markerfacecolor="white", markeredgewidth=1.2, zorder=5,
               label=r"$N_{eff}$  (maximise)  — monotone")
    ax_sp.plot(rank, x, "-s", color=_SPREAD, markersize=5.0, linewidth=1.8,
               markerfacecolor="white", markeredgewidth=1.2, zorder=4,
               label="residual spread  (minimise)  — monotone")
    ax_sp.plot(rank, s_raw, "-^", color=_SEP, markersize=5.4, linewidth=1.8,
               markerfacecolor="white", markeredgewidth=1.2, zorder=3,
               label=r"case/control separation  (minimise)  — $\bf{not}$ monotone")
    ax_sp.set_ylabel("min-max normalised")
    ax_sp.set_xlabel(r"Cumulative rank $k$", labelpad=1)
    ax_sp.set_xticks(rank)
    ax_sp.set_ylim(-0.06, 1.20)
    ax_sp.set_title(
        "A · The objective space — every cut below is an optimum over these, and they differ only in what is counted",
        fontsize=13.5, fontweight="bold", loc="left", pad=9)
    ax_sp.legend(loc="upper right", frameon=True, framealpha=0.95,
                 edgecolor="#CFCFCF", fontsize=10.5, ncol=3, columnspacing=1.1)

    # ══ B · Counting spread only — a monotone frontier ═══════════════
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
    ax_mo.set_title(f"B · Counting spread — monotone, chord spans {spread.chord_span:.3f}",
                    fontsize=13.5, fontweight="bold", loc="left", pad=9)
    ax_mo.legend(loc="lower right", frameon=True, framealpha=0.95,
                 edgecolor="#CFCFCF", fontsize=10.0)

    # ══ C · Counting both — a folded axis, so the knee cannot apply ══
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
    ax_bl.set_title(f"C · Counting both — folded, chord spans only {blended.chord_span:.3f}",
                    fontsize=13.5, fontweight="bold", loc="left", pad=9)
    ax_bl.legend(loc="lower right", frameon=True, framealpha=0.95,
                 edgecolor="#CFCFCF", fontsize=10.0)

    # ══ D · What each answer depends on ══════════════════════════════
    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color=_UNSAFE, alpha=0.55,
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
            rf"intermediate $k$ = {k_blend} holds on $w \in [{lo:.2f},\ {hi:.2f}]$"
            "\n"
            rf"evaluated at $w={blend_weight:g}$, {min(blend_weight - lo, hi - blend_weight):.2f} from the nearest edge",
            xy=(blend_weight, k_blend), xytext=(0, 16), textcoords="offset points",
            fontsize=11.0, color=_BK, ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=_INTER,
                      linewidth=1.2, alpha=0.96), zorder=9,
        )
    for name, k in sorted(cuts.items(), key=lambda kv: kv[1]):
        if np.any(weight_winner == k):
            continue
        colour = _NARROW if name == "narrow" else _FULL
        ax_w.axhline(k, color=colour, linewidth=1.1, linestyle=":", alpha=0.9, zorder=2)
        why = (rf"{name} ($k$={k}) — fixed in panel B, lead {float(row[name]['Margin']):+.4f}"
               if name == "narrow" else rf"{name} ($k$={k}) — fixed in panel B")
        ax_w.text(0.995, k, why, color=colour, fontsize=10.5, va="center", ha="right",
                  fontstyle="italic", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                            edgecolor="none", alpha=0.92))
    ax_w.set_ylim(int(won.min()) - 1, max(int(won.max()), max(cuts.values(), default=0)) + 1)
    ax_w.set_xlim(0.0, 1.0)
    ax_w.set_yticks(sorted(set(int(v) for v in np.unique(won)) | set(cuts.values())))
    ax_w.set_ylabel(r"winning cut  $k^{*}(w)$")
    ax_w.set_xlabel(r"$w$ — weight on residual spread;  $1-w$ on case/control separation")
    ax_w.set_title("D · What the answers depend on — the blend swept over its weight, the knee's lead marked",
                   fontsize=13.5, fontweight="bold", loc="left", pad=9)

    for axis in (ax_sp, ax_mo, ax_bl, ax_w):
        axis.grid(True, alpha=0.30, linewidth=0.7)
        axis.set_axisbelow(True)

    # ══ Right column: the framework, in the panels' order ════════════
    _note_axis(ax_note)

    def _rule(yy: float) -> None:
        ax_note.plot([0.0, 1.0], [yy, yy], color="#E0E0E0", linewidth=0.9)

    def _h(yy: float, txt: str) -> None:
        ax_note.text(_X_INDENT, yy, txt, fontsize=13.5, fontweight="bold", va="top",
                     ha="left", color=_BK)

    def _m(yy: float, txt: str, size: float = 13.0) -> None:
        ax_note.text(0.5, yy, txt, fontsize=size, va="top", ha="center", color=_BK)

    def _b(yy: float, txt: str, size: float = 11.5, style: str = "normal",
           color: str = _GR) -> None:
        ax_note.text(_X_INDENT, yy, txt, fontsize=size, va="top", ha="left",
                     color=color, fontstyle=style)

    _rule(0.984)
    _h(0.971, "One procedure, three settings")
    _b(0.940, r"Every cut minimises residual structure against $N_{eff}$ over the")
    _b(0.918, r"same normalised axes. The cuts differ only in what is counted as")
    _b(0.896, r"structure, and the operator then follows from the geometry that")
    _b(0.874, r"produces — it is a property of the space, not a preference.")
    _m(0.838, r"$x_k = \dfrac{H_k - \min_j H_j}{\max_j H_j - \min_j H_j}$"
              r"$\qquad$"
              r"$y_k = \dfrac{N_k - \min_j N_j}{\max_j N_j - \min_j N_j}$", 12.5)

    _rule(0.772)
    _h(0.759, r"full — nothing counted   (panel B, circle)")
    _m(0.726, r"$k_{\mathrm{full}} = \arg\max_k\ y_k$", 12.5)
    _b(0.692, r"With no homogeneity term there is no trade to make, so the")
    _b(0.670, r"optimum is the largest set: every major-cluster component.")
    _b(0.648, r"Derived, not asserted — which is what places it in this family.")

    _rule(0.628)
    _h(0.615, r"narrow — residual spread counted   (panel B, diamond)")
    _m(0.578, r"$k_{\mathrm{narrow}} = \arg\max_k\ d_k,\quad$"
              r"$d_k = \dfrac{\left|\Delta y\,x_k - \Delta x\,y_k + x_K y_1 - y_K x_1\right|}"
              r"{\sqrt{\Delta x^2 + \Delta y^2}}$", 11.5)
    _b(0.522, r"Spread is strictly monotone in $k$, so the frontier is a curve and")
    _b(0.500, r"the chord spans it. The knee is the turn in the exchange rate —")
    _b(0.478, r"the last cut before $N_{eff}$ stops repaying the spread it costs")
    _b(0.456, r"(Satopää et al. 2011). Preferred to proximity-to-a-corner wherever")
    _b(0.434, r"it applies, because only it reads in those terms.")
    _b(0.406, rf"Lead over the runner-up is {float(row['narrow']['Margin']):+.4f}. The turn is real but",
       color=_DIM, style="italic")
    _b(0.384, r"not sharply placed; panel D marks it.", color=_DIM, style="italic")

    _rule(0.364)
    _h(0.351, r"intermediate — spread and separation counted   (panel C, square)")
    _m(0.316, r"$H_k(w) = w\,x_k + (1-w)\,s_k,\qquad w = 1/2$", 12.5)
    _m(0.274, r"$k_{\mathrm{inter}} = \arg\min_k \sqrt{H_k(w)^2 + (1 - y_k)^2}$", 12.5)
    _b(0.232, r"Separation falls with $k$ while spread rises, so the blended axis")
    _b(0.210, r"folds: it travels " + f"{blended.chord_span:.3f}" + r" between the ends against spread's")
    _b(0.188, f"{spread.chord_span:.3f}" + r", leaving the chord near vertical. There is no turn to")
    _b(0.166, r"find, so the knee is inadmissible here and proximity to the ideal")
    _b(0.144, r"corner is what remains defined.")
    _b(0.116, r"$w$ cannot be estimated from data. It is fixed at equal weight and",
       color=_DIM, style="italic")
    _b(0.094, r"swept in panel D, which reports where the answer would change.",
       color=_DIM, style="italic")

    # ── The resolved cuts, as delivered ──────────────────────────────
    _rule(0.074)
    _b(0.062, f"Resolved   (mode: {mode})", size=11.0, color=_BK)
    hdr = ("variant", "operator", "k", "auto", "manual", "agree")
    xs = (0.05, 0.230, 0.500, 0.585, 0.680, 0.800)
    for xx, t in zip(xs, hdr):
        ax_note.text(xx, 0.040, t, fontsize=9.5, va="top", ha="left", color=_DIM,
                     fontstyle="italic")
    for i, (_, r) in enumerate(cut_selection.iterrows()):
        yy = 0.021 - i * 0.018
        agree = r.get("Auto_Manual_Agree")
        vals = (
            str(r["Variant"]),
            str(r["Operator"]).replace("_", " "),
            "—" if pd.isna(r["Resolved_Rank"]) else str(int(r["Resolved_Rank"])),
            "—" if pd.isna(r["Auto_Rank"]) else str(int(r["Auto_Rank"])),
            "—" if pd.isna(r["Manual_Rank"]) else str(int(r["Manual_Rank"])),
            "—" if pd.isna(agree) else ("yes" if bool(agree) else "NO"),
        )
        for j, (xx, t) in enumerate(zip(xs, vals)):
            ax_note.text(xx, yy, t, fontsize=9.5, va="top", ha="left",
                         color=_BK if j == 0 else _GR,
                         fontweight="bold" if j == 0 or t == "NO" else "normal")

    fig.suptitle(f"Deriving the Delivered Cuts — Mainland PCA, PC1-PC{n_dim}",
                 fontsize=18, fontweight="bold", y=0.977)
    fig.subplots_adjust(left=0.052, right=0.992, top=0.930, bottom=0.070)
    _series_footer(fig, "rank_cut_selection")
    return fig
