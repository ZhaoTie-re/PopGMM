"""The rank-selection trade-off figure: effective sample size against residual spread.

One point per cumulative rank cut, with the Pareto front marked. The right-hand
column is a typeset methods note rather than a data panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

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
