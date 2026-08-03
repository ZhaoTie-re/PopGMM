"""Panel-level drawing helpers shared by more than one figure.

The reuse axis in this codebase is the *panel*, not the figure: the two helpers
below existed as byte-identical copies inside the plot blocks of
``cohort_assignment``, ``subcluster_assignment`` and ``subcluster_view``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

from scripts.plotting.style import REFERENCE_ALPHA, REFERENCE_COLOR


def attach_colorbar(
    ax: Axes,
    mappable: ScalarMappable,
    *,
    label: str,
    label_size: float | None = None,
    tick_size: float | None = None,
    width: float = 0.018,
    pad: float = 0.02,
    fmt: str = "%.2f",
    n_ticks: int = 6,
) -> Colorbar:
    """Slim colourbar pinned to the right edge of ``ax``.

    Uses ``ax.inset_axes``, whose position is a transform tied to the parent, so
    it tracks the panel whatever the layout does. The four call sites previously
    used ``fig.add_axes`` with coordinates read from ``ax.get_position()``, which
    is only correct after the layout is final -- and is invisible to a layout
    engine, so under ``constrained_layout`` the bar neither moves nor reserves
    room and can land on top of the next column.
    """
    figure = ax.get_figure()
    if figure is None:  # pragma: no cover - an axes always has a figure here
        raise RuntimeError("axes is not attached to a figure")
    cax = ax.inset_axes((1.0 + pad, 0.04, width, 0.92))
    cbar = figure.colorbar(mappable, cax=cax)
    # Sizes default to the active theme rather than to a shared scale: each
    # figure keeps its own typography, and this helper only fixes *where* the
    # bar sits.
    cbar.set_label(label, fontsize=label_size if label_size is not None else plt.rcParams["axes.labelsize"])
    cbar.ax.tick_params(labelsize=tick_size if tick_size is not None else plt.rcParams["ytick.labelsize"])
    cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(n_ticks))
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
    return cbar


def apply_equal_centered_limits(ax: Axes, *, x_center: float, y_center: float, span: float) -> None:
    """Square axes of side ``span`` centred on ``(x_center, y_center)``.

    Every PC1-PC2 panel in a figure gets the same window, so panels are directly
    comparable and a cluster does not appear to move between them. The aspect is
    locked so one unit of PC1 is one unit of PC2 -- without that, the eye reads
    the elongation of the axes rather than of the data.
    """
    ax.set_xlim(x_center - span / 2.0, x_center + span / 2.0)
    ax.set_ylim(y_center - span / 2.0, y_center + span / 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_anchor("C")


def add_reference_background(
    ax: Axes,
    pc1: np.ndarray,
    pc2: np.ndarray,
    *,
    label: str = "BBJ",
    color: str = REFERENCE_COLOR,
    alpha: float = REFERENCE_ALPHA,
    size: float = 20.0,
) -> None:
    """Draw the reference panel as a grey cloud beneath everything else.

    ``rasterized=True`` matters: this is ~183k points, and leaving it vector
    would dominate the file size of any PDF export while adding nothing a raster
    layer does not already show.

    The empty guard came from ``subcluster_view``, the only copy that had it. It
    is not merely defensive -- an empty ``scatter`` still registers its label, so
    without the guard a basis whose projection covers no reference sample would
    produce a legend entry for a cloud that is not drawn.
    """
    if pc1.size == 0:
        return
    ax.scatter(
        pc1,
        pc2,
        c=color,
        s=size,
        alpha=alpha,
        label=label,
        rasterized=True,
        zorder=0,
    )
