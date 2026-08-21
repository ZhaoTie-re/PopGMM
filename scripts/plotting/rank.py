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


def plot_casectrl_separation(
    *,
    decision_table: pd.DataFrame,
    mainland_axes: Sequence[str],
    weight_grid: np.ndarray,
    weight_winner: np.ndarray,
    variant_cuts: "Mapping[str, int] | None" = None,
    safe_weight_floor: float = 0.5,
) -> Figure:
    """Return the supplementary separation figure: three data panels and a note column.

    A sibling of ``plot_rank_tradeoff``, not an extension of it. That figure
    prices one exchange -- effective sample size against residual spread -- and
    is what selects a cut. This one carries the question residual spread cannot
    answer (do cases and controls sit at different places *within* the retained
    set) and the re-weighting argument behind the intermediate cut. Neither is
    an input to the recommendation; see ``scripts.common.case_ctrl_separation``.
    """
    n_dim = len(mainland_axes)
    rank = decision_table["Included_Max_Rank"].to_numpy(dtype=int, copy=False)
    d_raw = decision_table["Mainland_CaseCtrl_Mahalanobis"].to_numpy(dtype=float, copy=False)
    d2_unb = decision_table["Mainland_CaseCtrl_D2_Unbiased"].to_numpy(dtype=float, copy=False)
    floor = decision_table["Mainland_CaseCtrl_Noise_Floor"].to_numpy(dtype=float, copy=False)
    pval = decision_table["Mainland_CaseCtrl_P"].to_numpy(dtype=float, copy=False)
    d2_raw = d_raw ** 2
    cuts = dict(variant_cuts or {})

    _BK, _GR, _DIM = "#212121", "#424242", "#757575"
    _RAW, _UNB, _FLOOR = "#7B3294", "#008837", "#BDBDBD"
    _SIG, _UNSAFE = "#D7191C", "#F4C7C3"
    _X = 0.05

    fig = plt.figure(figsize=(19.5, 12.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 2.75], wspace=0.055)
    gs_l = gs[0, 0].subgridspec(3, 1, height_ratios=[1.08, 0.90, 0.86], hspace=0.33)
    ax_d = fig.add_subplot(gs_l[0, 0])
    ax_p = fig.add_subplot(gs_l[1, 0], sharex=ax_d)
    ax_w = fig.add_subplot(gs_l[2, 0])
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
    ax_p.set_xlabel(r"Cumulative rank $k$", labelpad=2)
    ax_p.set_xticks(rank)

    # ── The delivered cuts, on both rank panels ──────────────────────
    for name, k in sorted(cuts.items(), key=lambda kv: kv[1]):
        for axis in (ax_d, ax_p):
            axis.axvline(k, color=_DIM, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
        # Axes-fraction y so the strip stays put whatever the data range; the
        # end cuts anchor inwards or they overrun the frame.
        ha = "left" if k <= rank.min() else "right" if k >= rank.max() else "center"
        ax_d.text(
            k, 0.985, f"{name}\nk={k}", color=_BK, fontsize=11.0, fontweight="bold",
            va="top", ha=ha, transform=ax_d.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.8, alpha=0.92),
        )
    for axis in (ax_d, ax_p):
        axis.set_xlim(rank.min() - 0.4, rank.max() + 0.4)
        axis.grid(True, alpha=0.30, linewidth=0.7)
        axis.set_axisbelow(True)

    # ══ C · Which cut wins as the two measures are re-weighted ═══════
    won = weight_winner[weight_winner > 0]
    ax_w.fill_between([0.0, safe_weight_floor], -100, 100, color=_UNSAFE, alpha=0.55,
                      linewidth=0, zorder=1)
    ax_w.text(safe_weight_floor / 2.0, 0.045,
              "not usable — weights the labels\nabove the spread",
              transform=ax_w.get_xaxis_transform(), fontsize=10.5, color="#9B2226",
              ha="center", va="bottom", fontstyle="italic", zorder=6)
    ax_w.plot(weight_grid, weight_winner, drawstyle="steps-post", color=_BK,
              linewidth=2.6, zorder=4, solid_joinstyle="miter")

    # The cuts fixed by other rules are drawn but never win here; saying so on
    # the panel is the point, since a reader will look for them.
    for name, k in sorted(cuts.items(), key=lambda kv: kv[1]):
        if not np.any(weight_winner == k):
            ax_w.axhline(k, color=_DIM, linewidth=1.0, linestyle=":", alpha=0.85, zorder=2)
            ax_w.text(0.995, k, f"{name} (k={k}) — fixed by another rule", color=_DIM,
                      fontsize=10.5, va="center", ha="right", fontstyle="italic", zorder=6,
                      bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                edgecolor="none", alpha=0.92))

    # The winning plateau, and how wide it is.
    for name, k in cuts.items():
        on = weight_grid[weight_winner == k]
        if on.size == 0:
            continue
        lo, hi = float(on.min()), float(on.max())
        ax_w.plot([lo, hi], [k, k], color=_UNB, linewidth=6.5, solid_capstyle="butt",
                  alpha=0.85, zorder=5)
        ax_w.annotate(
            f"{name}  k={k}\n" rf"wins across  $w \in [{lo:.2f},\ {hi:.2f}]$",
            xy=((lo + hi) / 2.0, k), xytext=(0, 15), textcoords="offset points",
            fontsize=11.5, fontweight="bold", color=_BK, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=_UNB, linewidth=1.1, alpha=0.95), zorder=7,
        )

    y_lo, y_hi = int(won.min()) - 1, max(int(won.max()), max(cuts.values(), default=0)) + 1
    ax_w.set_ylim(y_lo, y_hi)
    ax_w.set_xlim(0.0, 1.0)
    ax_w.set_yticks(sorted(set(int(v) for v in np.unique(won)) | set(cuts.values())))
    ax_w.set_ylabel(r"Winning cut  $k^{*}(w)$")
    ax_w.set_xlabel(r"$w$ — weight on residual spread; $1-w$ on case/control separation")
    ax_w.set_title("C · Why k = 12 is the intermediate cut: it wins across the whole balanced band",
                   fontsize=14, fontweight="bold", loc="left", pad=9)
    ax_w.grid(True, alpha=0.30, linewidth=0.7)
    ax_w.set_axisbelow(True)

    # ══ Right column: methods / formulas ═════════════════════════════
    ax_note.set_axis_off()
    ax_note.set_xlim(0.0, 1.0)
    ax_note.set_ylim(0.0, 1.0)

    def _rule(y: float) -> None:
        ax_note.plot([0.0, 1.0], [y, y], color="#E0E0E0", linewidth=0.9)

    def _h(y: float, txt: str) -> None:
        ax_note.text(_X, y, txt, fontsize=14, fontweight="bold", va="top",
                     ha="left", color=_BK)

    def _m(y: float, txt: str, size: float = 15.0) -> None:
        ax_note.text(0.5, y, txt, fontsize=size, va="top", ha="center", color=_BK)

    def _b(y: float, txt: str, size: float = 12.0, style: str = "normal",
           color: str = _GR) -> None:
        ax_note.text(_X, y, txt, fontsize=size, va="top", ha="left", color=color,
                     fontstyle=style)

    # ── Section 1: the statistic ──────────────────────────────────────
    _rule(0.980)
    _h(0.966, "Case/Control Separation")
    _m(0.926, r"$D^2 = (\bar{x}_{1} - \bar{x}_{2})^{\top}\,"
              r"S_{\mathrm{pooled}}^{-1}\,(\bar{x}_{1} - \bar{x}_{2})$")
    _m(0.876, r"$T^2 = D^2\,\dfrac{n_1 n_2}{n_1 + n_2}$"
              r"$\qquad p:\ F = \dfrac{T^2 (\nu - p + 1)}{p\,\nu}$", 14.0)
    _b(0.812, "Mahalanobis distance between the two group centroids under")
    _b(0.789, "the pooled within-group covariance, in SD units. $T^2$ weights")
    _b(0.766, r"it by sample size: the distance alone is not evidence.")
    _b(0.743, rf"Computed on mainland PC1-PC{n_dim} — the axes RGV uses, on the",
       color=_DIM, style="italic")
    _b(0.720, "same retained samples.", color=_DIM, style="italic")

    # ── Section 2: the sampling floor ─────────────────────────────────
    _rule(0.700)
    _h(0.686, "Why a raw $D^2$ cannot be compared across cuts")
    _m(0.644, r"$E\left[\hat{D}^2\right] = D^2_{\mathrm{true}}"
              r" + p\left(\dfrac{1}{n_1} + \dfrac{1}{n_2}\right)$", 14.0)
    _b(0.580, "Two sample means never coincide, and $D^2$ squares the gap, so")
    _b(0.557, "scatter can only add. The retained set grows more than tenfold")
    _b(0.534, "across the walk, so this floor falls with it and a raw $\\hat{D}^2$ drifts")
    _b(0.511, "down for arithmetic reasons alone. The de-biased column subtracts")
    _b(0.488, r"it, and goes negative where separation is below the floor.")
    _b(0.465, r"$T^2$ and $p$ need no correction — the $F$ test is exact.",
       color=_DIM, style="italic")

    # ── Section 3: the intermediate cut ───────────────────────────────
    _rule(0.445)
    _h(0.431, "Why $k = 12$ for the intermediate cut")
    _b(0.394, "The two homogeneity measures disagree about direction: RGV rises")
    _b(0.371, r"with $k$ ($\rho = +1.00$), the de-biased separation falls")
    _b(0.348, r"($\rho = -0.73$). Blending them gives an interior optimum, where")
    _b(0.325, "either alone only trades against $N_{eff}$.")
    _m(0.286, r"$H(w) = w\cdot\mathrm{RGV} + (1-w)\cdot \hat{D}^2_{\mathrm{unb}}$"
              r"$\quad$(min-max)", 13.5)
    _m(0.238, r"$k^{*}(w) = \arg\min_k \sqrt{H(w)^2 + \left(1 - N_{eff}\right)^2}$", 13.5)
    _b(0.190, r"$w$ is a judgement about which residual structure matters, not a")
    _b(0.167, "value the data can supply — which is why it is swept rather than")
    _b(0.144, "picked. Panel C is the result: one cut owns the entire balanced")
    _b(0.121, "band, so the choice does not rest on having chosen a $w$.")

    # ── Section 4: what fixes each cut ────────────────────────────────
    _rule(0.101)
    _b(0.087, "What fixes each cut:", size=11.0, color=_BK)
    for i, (label, why) in enumerate((
        (r"narrow", r"knee of the $N_{eff}$-vs-RGV curve"),
        (r"intermediate", r"the re-weighting sweep in panel C"),
        (r"full", r"every major-cluster component — definitional"),
    )):
        y = 0.065 - i * 0.019
        ax_note.text(_X + 0.02, y, label, fontsize=10.5, fontweight="bold",
                     va="top", ha="left", color=_GR)
        ax_note.text(_X + 0.20, y, why, fontsize=10.5, va="top", ha="left",
                     color=_DIM, fontstyle="italic")
    ax_note.text(_X, 0.005, "Separation is diagnostic throughout: no cut is selected on it.",
                 fontsize=10.5, va="top", ha="left", color=_BK, fontweight="bold")

    fig.suptitle("Supplementary · Case/Control Ancestry Separation, and the Intermediate Cut",
                 fontsize=18, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.058, right=0.992, top=0.928, bottom=0.058)
    return fig
