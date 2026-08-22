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
) -> Figure:
    """Return the trade-off figure; the caller styles, saves and closes it."""

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
                label="Recommended k", zorder=5,
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
                    f"k = {rec_k}  (recommended)",
                    f"{config.case_label}: {_case_n:,}  |  {config.control_label}: {_ctrl_n:,}",
                    f"Total: {_total_n:,}  (composite posterior)",
                ]
            else:
                _ann_lines = [f"k = {rec_k}  (recommended)"]
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
    return fig


# Shared with the trade-off figure's note column so the two read as one document.
_BK, _GR, _DIM = "#212121", "#424242", "#757575"
_X_INDENT = 0.05


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
    fig.subplots_adjust(left=0.058, right=0.992, top=0.905, bottom=0.075)
    return fig


def plot_rank_cut_selection(
    *,
    decision_table: pd.DataFrame,
    cut_selection: pd.DataFrame,
    rgv_column: str,
    mainland_axes: Sequence[str],
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    blend_weight: float,
    rank_cuts: "Mapping[str, int | None]",
    mode: str,
    safe_weight_floor: float = 0.5,
) -> Figure:
    """Return the figure that derives the delivered cuts.

    One panel per rule, plus the construction each rule is read off, so a
    reader can check the arithmetic rather than take the number: the knee is
    drawn as the chord and the perpendicular it maximises, and the blend as the
    sweep over the weight it is evaluated at. The uncut full set appears in the
    summary but has no panel -- it is definitional, not selected.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    neff = decision_table["Neff_Norm"].to_numpy(dtype=float, copy=False)
    het = decision_table["RGV_Norm"].to_numpy(dtype=float, copy=False)

    dx, dy = het[-1] - het[0], neff[-1] - neff[0]
    chord = float(np.hypot(dx, dy))
    perp = np.abs(dy * het - dx * neff + het[-1] * neff[0] - neff[-1] * het[0]) / chord

    cuts = {str(n): int(k) for n, k in rank_cuts.items() if k is not None}
    k_knee = cuts.get("narrow", int(rank[int(np.argmax(perp))]))
    k_blend = cuts.get("intermediate")

    _KNEE, _CHORD, _BLEND, _UNSAFE = "#0571B0", "#B0B0B0", "#008837", "#F4C7C3"

    fig = plt.figure(figsize=(19.5, 12.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 2.75], wspace=0.055)
    gs_l = gs[0, 0].subgridspec(2, 2, height_ratios=[1.0, 0.82],
                               width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.20)
    ax_k = fig.add_subplot(gs_l[0, 0])
    ax_pf = fig.add_subplot(gs_l[0, 1])
    ax_w = fig.add_subplot(gs_l[1, :])
    ax_note = fig.add_subplot(gs[0, 1])

    # ══ A · The knee construction ════════════════════════════════════
    ax_k.plot([het[0], het[-1]], [neff[0], neff[-1]], "-", color=_CHORD,
              linewidth=2.0, zorder=2, label="Chord joining the ends")
    ax_k.plot(het, neff, "-o", color=_GR, markersize=5.2, linewidth=1.6,
              markerfacecolor="white", markeredgewidth=1.3, zorder=4,
              label="Cumulative cuts")
    i_knee = int(np.argmax(perp))
    # Foot of the perpendicular, so the distance being maximised is visible as a
    # segment rather than asserted in a caption.
    t = ((het[i_knee] - het[0]) * dx + (neff[i_knee] - neff[0]) * dy) / (chord ** 2)
    foot = (het[0] + t * dx, neff[0] + t * dy)
    ax_k.plot([het[i_knee], foot[0]], [neff[i_knee], foot[1]], "-", color=_KNEE,
              linewidth=2.4, zorder=5)
    ax_k.plot([het[i_knee]], [neff[i_knee]], "D", color=_KNEE, markersize=11.5,
              markeredgecolor="white", markeredgewidth=1.4, zorder=6,
              label=rf"Knee — $k$ = {k_knee}")
    # Alternating offsets: the cuts crowd together on the flat arm, where a
    # single offset stacks the labels on top of each other.
    for i, k in enumerate(rank):
        if i == i_knee:
            offset, colour, weight = (15, -3), _KNEE, "bold"
        else:
            offset, colour, weight = ((0, -15) if i % 2 == 0 else (0, 9)), _DIM, "normal"
        ax_k.annotate(str(int(k)), (het[i], neff[i]), textcoords="offset points",
                      xytext=offset, fontsize=9.5, color=colour, ha="center",
                      fontweight=weight)
    ax_k.set_xlabel(r"Residual spread  (min-max normalised)")
    ax_k.set_ylabel(r"$N_{eff}$  (min-max normalised)")
    ax_k.set_title("A · The knee, drawn", fontsize=14, fontweight="bold", loc="left", pad=9)
    ax_k.legend(loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="#CFCFCF", fontsize=10.5)

    # ══ B · The profile it maximises ═════════════════════════════════
    order = np.argsort(-perp)
    runner = int(order[1])
    ax_pf.vlines(rank, 0.0, perp, color="#D6D6D6", linewidth=5.0, zorder=2)
    ax_pf.vlines(rank[i_knee], 0.0, perp[i_knee], color=_KNEE, linewidth=5.0, zorder=3)
    ax_pf.plot(rank, perp, "o", color=_GR, markersize=4.6, markerfacecolor="white",
               markeredgewidth=1.2, zorder=4)
    ax_pf.plot([rank[i_knee]], [perp[i_knee]], "D", color=_KNEE, markersize=10.0,
               markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    ax_pf.annotate(
        f"lead over $k$ = {int(rank[runner])}:\n{perp[i_knee] - perp[runner]:+.4f}"
        f"  ({(perp[i_knee] / perp[runner] - 1) * 100:.2f}%)",
        xy=(rank[i_knee], perp[i_knee]), xytext=(14, -6), textcoords="offset points",
        fontsize=10.5, color=_BK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=_KNEE,
                  linewidth=1.0, alpha=0.95), zorder=6,
    )
    ax_pf.set_xlabel(r"Cumulative rank $k$")
    ax_pf.set_ylabel(r"Distance to the chord  $d(k)$")
    ax_pf.set_title("B · and the profile it maximises", fontsize=14, fontweight="bold",
                    loc="left", pad=9)
    ax_pf.set_xticks(rank[::2])

    # ══ C · The blend, swept over its weight ═════════════════════════
    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color=_UNSAFE, alpha=0.55,
                      linewidth=0, zorder=1)
    ax_w.text(safe_weight_floor / 2.0, 0.05,
              "not usable — weights the case/control labels\nabove the residual spread",
              transform=ax_w.get_xaxis_transform(), fontsize=10.5, color="#9B2226",
              ha="center", va="bottom", fontstyle="italic", zorder=6)
    ax_w.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK,
              linewidth=2.6, zorder=4, solid_joinstyle="miter")

    if k_blend is not None:
        on = weight_grid[weight_winner == k_blend]
        lo, hi = float(on.min()), float(on.max())
        ax_w.plot([lo, hi], [k_blend, k_blend], color=_BLEND, linewidth=7.0,
                  solid_capstyle="butt", alpha=0.85, zorder=5)
        ax_w.axvline(blend_weight, color=_BLEND, linewidth=1.6, linestyle="-.", zorder=6)
        ax_w.plot([blend_weight], [k_blend], "o", color=_BLEND, markersize=11.0,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=8)
        ax_w.annotate(
            f"intermediate  $k$ = {k_blend}\n"
            rf"evaluated at $w = {blend_weight:g}$, constant on $[{lo:.2f},\ {hi:.2f}]$"
            "\n"
            rf"nearest boundary {min(blend_weight - lo, hi - blend_weight):.2f} away",
            xy=(blend_weight, k_blend), xytext=(0, 17), textcoords="offset points",
            fontsize=11.0, color=_BK, ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=_BLEND,
                      linewidth=1.2, alpha=0.96), zorder=9,
        )
    for name, k in sorted(cuts.items(), key=lambda kv: kv[1]):
        if not np.any(weight_winner == k):
            ax_w.axhline(k, color=_DIM, linewidth=1.0, linestyle=":", alpha=0.85, zorder=2)
            ax_w.text(0.995, k, f"{name} (k={k}) — fixed by another rule", color=_DIM,
                      fontsize=10.5, va="center", ha="right", fontstyle="italic", zorder=6,
                      bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                edgecolor="none", alpha=0.92))
    ax_w.set_ylim(int(won.min()) - 1, max(int(won.max()), max(cuts.values(), default=0)) + 1)
    ax_w.set_xlim(0.0, 1.0)
    ax_w.set_yticks(sorted(set(int(v) for v in np.unique(won)) | set(cuts.values())))
    ax_w.set_ylabel(r"Winning cut  $k^{*}(w)$")
    ax_w.set_xlabel(r"$w$ — weight on residual spread;  $1-w$ on case/control separation")
    ax_w.set_title("C · The blend, swept over the weight it is evaluated at",
                   fontsize=14, fontweight="bold", loc="left", pad=9)

    for axis in (ax_k, ax_pf, ax_w):
        axis.grid(True, alpha=0.30, linewidth=0.7)
        axis.set_axisbelow(True)

    # ══ Right column: the two rules, formally ════════════════════════
    _note_axis(ax_note)

    def _rule(y: float) -> None:
        ax_note.plot([0.0, 1.0], [y, y], color="#E0E0E0", linewidth=0.9)

    def _h(y: float, txt: str) -> None:
        ax_note.text(_X_INDENT, y, txt, fontsize=14, fontweight="bold", va="top",
                     ha="left", color=_BK)

    def _m(y: float, txt: str, size: float = 14.0) -> None:
        ax_note.text(0.5, y, txt, fontsize=size, va="top", ha="center", color=_BK)

    def _b(y: float, txt: str, size: float = 12.0, style: str = "normal",
           color: str = _GR) -> None:
        ax_note.text(_X_INDENT, y, txt, fontsize=size, va="top", ha="left",
                     color=color, fontstyle=style)

    _rule(0.982)
    _h(0.968, "Normalisation, common to both rules")
    _m(0.928, r"$x_k = \dfrac{H_k - \min_j H_j}{\max_j H_j - \min_j H_j}$"
              r"$\qquad$"
              r"$y_k = \dfrac{N_k - \min_j N_j}{\max_j N_j - \min_j N_j}$", 13.0)
    _b(0.862, r"$H_k$ is the residual spread of cut $k$ and $N_k$ its $N_{eff}$;")
    _b(0.838, r"both are mapped to $[0,1]$ so neither unit can dominate.")

    _rule(0.816)
    _h(0.802, r"1 · narrow — the knee of the trade-off curve")
    _m(0.756, r"$k_{\mathrm{narrow}} = \arg\max_k\ d_k,\quad$"
              r"$d_k = \dfrac{\left|\Delta y\,x_k - \Delta x\,y_k + x_K y_1 - y_K x_1\right|}"
              r"{\sqrt{\Delta x^2 + \Delta y^2}}$", 12.5)
    _b(0.692, r"the point of the curve furthest from the chord joining its two")
    _b(0.668, r"ends, with $\Delta x = x_K - x_1$ and $\Delta y = y_K - y_1$ over the")
    _b(0.644, r"$K$ walked cuts (Satopää et al. 2011). It formalises reading the")
    _b(0.620, r"corner off the trade-off curve: the last cut before $N_{eff}$ stops")
    _b(0.596, r"repaying the spread it costs.")
    _b(0.566, r"Both quantities are monotone in $k$ here, so the chord runs",
       color=_DIM, style="italic")
    _b(0.542, r"$(0,0)\!\rightarrow\!(1,1)$ and $d_k = |y_k - x_k|/\sqrt{2}$ — a rescaling of the",
       color=_DIM, style="italic")
    _b(0.518, r"Utility_Neff_minus_RGV column the table already carries.",
       color=_DIM, style="italic")

    _rule(0.496)
    _h(0.482, r"2 · intermediate — equal weight on both homogeneity measures")
    _m(0.438, r"$H_k(w) = w\,x_k + (1-w)\,s_k$", 13.0)
    _m(0.392, r"$k_{\mathrm{inter}} = \arg\min_k \sqrt{H_k(w)^2 + (1-y_k)^2}\,,"
              r"\quad w = 1/2$", 13.0)
    _b(0.334, r"with $s_k$ the min-max normalised de-biased $D^2$. Residual spread")
    _b(0.310, r"rises strictly with $k$ ($\rho = +1.00$) while the separation falls")
    _b(0.286, r"($\rho = -0.73$), so the blend has an interior optimum that neither")
    _b(0.262, r"measure has alone — either one only trades against $N_{eff}$.")
    _b(0.232, r"$w$ encodes which residual structure is judged to matter and no")
    _b(0.208, r"data can supply it. It is fixed at equal weight and swept in")
    _b(0.184, r"panel C, which reports the interval on which the answer is")
    _b(0.160, r"constant rather than asserting the value is correct.")

    _rule(0.138)
    _h(0.124, ("3 · full — every major-cluster component.  Definitional."))

    # ── The resolved cuts, as delivered ──────────────────────────────
    _b(0.090, f"Resolved cuts   (mode: {mode})", size=11.5, color=_BK)
    hdr = ("variant", "rule", "k", "auto", "manual", "agree")
    xs = (0.05, 0.235, 0.505, 0.585, 0.680, 0.800)
    for x, t in zip(xs, hdr):
        ax_note.text(x, 0.064, t, fontsize=10.0, va="top", ha="left", color=_DIM,
                     fontstyle="italic")
    for i, (_, row) in enumerate(cut_selection.iterrows()):
        y = 0.042 - i * 0.020
        agree = row.get("Auto_Manual_Agree")
        vals = (
            str(row["Variant"]),
            str(row["Rule"]).replace("_", " "),
            "—" if pd.isna(row["Resolved_Rank"]) else str(int(row["Resolved_Rank"])),
            "—" if pd.isna(row["Auto_Rank"]) else str(int(row["Auto_Rank"])),
            "—" if pd.isna(row["Manual_Rank"]) else str(int(row["Manual_Rank"])),
            "—" if pd.isna(agree) else ("yes" if bool(agree) else "NO"),
        )
        for j, (x, t) in enumerate(zip(xs, vals)):
            ax_note.text(x, y, t, fontsize=10.0, va="top", ha="left",
                         color=_BK if j == 0 else _GR,
                         fontweight="bold" if j == 0 or t == "NO" else "normal")

    fig.suptitle(f"Deriving the Delivered Cuts — Mainland PCA, PC1-PC{n_dim}",
                 fontsize=18, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.918, bottom=0.062)
    return fig
