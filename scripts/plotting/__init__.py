"""Figure rendering, separated from the computation that feeds it.

The step modules under ``scripts/`` compute; this package draws. The split is
not cosmetic -- before it, ``cohort_assignment`` computed the cluster ranking
*inside* ``if save_plot or show_plot:`` and wrote
``major_cluster_component_ranks.tsv`` from there, so turning plotting off
silently dropped a data deliverable.

Three modules, split along the axis that actually gets reused:

* ``style``  -- the frozen palette, the themes, and the one figure saver.
* ``panels`` -- panel-level helpers that were duplicated across step modules.
* ``figures`` -- one ``plot_*`` per figure, each returning a ``Figure``.
"""

from __future__ import annotations
