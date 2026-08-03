"""The frozen palette, the figure themes, and the one figure saver.

Two properties this module exists to guarantee:

**1. Style is hermetic.** Before this, style was applied by mutating global
rcParams inside each plot block, and only 3 of 9 modules reset first. A replay
of the fresh-run sequence measured **27 of 28 renders** inheriting state from
whichever figure drew before them. Two examples of what that cost:

* ``sns.set_context("talk", font_scale=1.08)`` in the denoising figure sets
  ``lines.linewidth`` 1.5 -> 2.25 and ``lines.markersize`` 6 -> 9. Nothing
  resets them -- ``plt.style.use("seaborn-v0_8-whitegrid")`` does not touch
  context keys -- so the merging dendrogram's ``LineCollection`` was drawn at
  the denoising figure's line width. That is the mechanism behind the 3.4 %
  pixel shift recorded in ``scripts/artifacts.py``.
* ``subcluster_view`` rendered at ``figure.titlesize`` 18.0 in the global basis
  and 40.0 in the mainland basis -- same code, same loop, same run -- because a
  KDE figure ran in between and left ``PLOT_STYLE_RC`` behind.

``figure_context`` reseeds from matplotlib's defaults every time and reverts on
exit, so a figure looks the same no matter what ran before it. That is also what
makes figures from a ``resume`` run trustworthy.

**2. The cohort colours never move.** They are frozen constants here, and one of
them is not merely cosmetic: ``COMPOSITE_GROUP_COLOR`` is serialised into
``subcluster_summary.json`` as ``custom_group_color``, which
``tools/verify_results.py`` compares as a hard class. Changing it fails
verification as a *numeric* artifact.

Note on ``plt.style.use("seaborn-v0_8-whitegrid")``: it is kept as a call rather
than replaced by ``sns.axes_style("whitegrid")``. The two are not the same --
they disagree on 6 keys (``image.cmap``, ``patch.edgecolor``,
``patch.force_edgecolor``, ``xtick.bottom``, ``ytick.left``, sans-serif order)
and seaborn's omits 6 more that the bundled style sets (``axes.linewidth``,
``legend.frameon``, the tick sizes). ``sns.plotting_context`` *is* exactly
equivalent to ``sns.set_context`` and is pure, so contexts are resolved to a
dict instead of being applied globally.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Frozen palette
# ---------------------------------------------------------------------------
#
# FROZEN. These are the identity of the published figures and must not be
# changed without an explicit instruction. Entries 6, 2 and 4 of ColorBrewer
# "Paired" plus a neutral grey.

#: Case cohort (params.CASE_LABEL).
CASE_COLOR: Final[str] = "#E31A1C"

#: Control cohort (params.CONTROL_LABEL).
CONTROL_COLOR: Final[str] = "#1F78B4"

#: Study samples in neither cohort.
OTHER_COLOR: Final[str] = "#33A02C"

#: The reference-panel background cloud.
REFERENCE_COLOR: Final[str] = "#B0B0B0"
REFERENCE_ALPHA: Final[float] = 0.20

#: The composite subcluster group. Also written to subcluster_summary.json as
#: ``custom_group_color`` -- a hard-class artifact, so this is doubly frozen.
COMPOSITE_GROUP_COLOR: Final[str] = "#E67E22"

#: Sequential map for posterior confidence. Colourblind-safe and monotonic in
#: lightness. The three *scalings* it is used with are deliberately different
#: per figure (percentile 1/99, min/max, and a fixed PowerNorm) and are set at
#: the call sites, not here.
CONFIDENCE_CMAP: Final[str] = "cividis"

#: Sequential map for the inter-component Mahalanobis distance matrix.
DISTANCE_CMAP: Final[str] = "YlGnBu"


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """A complete, order-independent description of how a figure is styled.

    Applied as: matplotlib defaults -> ``seaborn-v0_8-whitegrid`` ->
    ``sns.plotting_context(context, font_scale)`` -> ``rc``. Every step is
    explicit, so the result never depends on what drew before.
    """

    #: seaborn context name, or None to leave the context keys at their defaults.
    context: str | None = None
    font_scale: float = 1.0
    #: rcParams applied last, overriding both the style and the context.
    rc: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self) -> dict[str, Any]:
        """The rcParams this theme produces, as a plain dict."""
        with plt.rc_context():
            _seed()
            if self.context is not None:
                plt.rcParams.update(sns.plotting_context(self.context, font_scale=self.font_scale))
            plt.rcParams.update(dict(self.rc))
            return dict(plt.rcParams)


def _seed() -> None:
    """Reset to matplotlib defaults, then apply the shared base style."""
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-whitegrid")


@contextlib.contextmanager
def figure_context(theme: Theme) -> Iterator[None]:
    """Apply ``theme`` for the duration of the block, then restore.

    Build *and* draw the figure inside this block. rcParams read at draw time
    (fonts, line widths, grid) are captured here; rcParams read at *save* time
    are handled separately by :func:`save_figure`.
    """
    with plt.rc_context():
        _seed()
        if theme.context is not None:
            plt.rcParams.update(sns.plotting_context(theme.context, font_scale=theme.font_scale))
        plt.rcParams.update(dict(theme.rc))
        yield


# --- The per-figure themes, carried over verbatim from the step modules. ----
#
# They stay distinct here rather than being collapsed into one: the blocks
# genuinely differ (boxed vs open spines, 18 pt vs 22 pt, coloured vs default
# axis furniture), and unifying them is a typography decision, not a
# refactoring one.

_SANS: Final[dict[str, Any]] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}

#: Large open-spine type. Was ``common.PLOT_STYLE_RC``.
_LARGE_OPEN: Final[dict[str, Any]] = {
    **_SANS,
    "figure.dpi": 400,
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "axes.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "legend.fontsize": 22,
    "legend.title_fontsize": 24,
    "figure.titlesize": 40,
}

#: Boxed-spine 18 pt type, shared by the mixture-model and merging overviews.
_BOXED_18: Final[dict[str, Any]] = {
    **_SANS,
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.title_fontsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 30,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
}

THEME_DENOISING: Final[Theme] = Theme(
    context="talk",
    font_scale=1.08,
    rc={
        **_SANS,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#3A3A3A",
        # No "axes.linewidth" here on purpose. The original block set it to 1.0
        # but then called sns.set_context("talk", ...) *afterwards*, and
        # axes.linewidth is a context key -- so seaborn's value always won and
        # the 1.0 never took effect. Themes apply the context first, so keeping
        # the 1.0 would silently change the figure.
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#C9CED6",
        "grid.linestyle": "--",
        "grid.alpha": 0.40,
        "legend.frameon": True,
        "legend.framealpha": 0.97,
        "legend.edgecolor": "#C3C7CE",
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFD",
    },
)

THEME_MIXTURE: Final[Theme] = Theme(rc=_BOXED_18)
THEME_MERGING: Final[Theme] = Theme(rc=_BOXED_18)
THEME_ASSIGNMENT: Final[Theme] = Theme(context="paper", font_scale=2.5, rc=_LARGE_OPEN)
THEME_KDE: Final[Theme] = Theme(context="paper", font_scale=2.0, rc=_LARGE_OPEN)

#: The subcluster assignment and view figures set only a context today. Their
#: previous appearance additionally depended on leaked rcParams; making them
#: hermetic necessarily changes them, which is expected and recorded.
THEME_SUBCLUSTER: Final[Theme] = Theme(context="paper", font_scale=2.5)

#: The rank-selection figure never called ``plt.style.use`` at all, so it
#: inherited 27 keys from whatever drew before it. Seeded like the rest now.
THEME_RANK: Final[Theme] = Theme(
    rc={
        **_SANS,
        "axes.linewidth": 0.9,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 18,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    },
)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

#: rcParams that ``savefig`` reads at save time rather than at draw time. They
#: must be in force around the ``savefig`` call itself -- setting them only
#: inside ``figure_context`` would leave PDFs with Type 3 fonts, which journals
#: reject. ``rank_selection`` was the only module that set fonttype, and it is
#: the one figure that was never exported as a PDF.
SAVE_RC: Final[dict[str, Any]] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "pdf.compression": 6,
}


def save_figure(
    fig: Figure,
    path: Path | str,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 400,
    bbox_inches: str | None = "tight",
    facecolor: Any = "auto",
) -> list[Path]:
    """Write ``fig`` to ``path`` once per entry in ``formats``.

    ``path``'s suffix is replaced per format, so callers pass the intended
    filename (``.../subcluster_view.png``) and get siblings for free.

    Takes a full path rather than a stem plus a directory: the merging figure is
    emitted seven times from one call path, once per robustness threshold, each
    into a different ``output_dir``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with plt.rc_context(SAVE_RC):
        for fmt in formats:
            out = target.with_suffix(f".{fmt}")
            fig.savefig(
                out,
                format=fmt,
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor=facecolor,
            )
            written.append(out)
    return written
