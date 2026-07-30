"""Single source of truth for PopGMM pipeline paths and parameters.

Everything the notebook would otherwise hard-code twice lives here. Two
duplications in particular were live bug sources:

* the mainland rank cut ``k = 9`` was written once as
  ``forced_recommended_rank`` in STEP4_tmp and again as
  ``_step5_included_rank`` in STEP5, with nothing keeping them in sync;
* the confidence threshold tuple was written verbatim in both STEP8 and STEP9.

This is a Python module rather than a YAML file on purpose. The step configs
are already typed ``@dataclass(frozen=True)`` objects, so YAML would need an
extra mapping-and-validation layer -- more code, plus a place for
``forced_recommended_rank: "9"`` to silently become a string. Half the values
here are derived expressions, which are natural in Python and awkward in YAML.
And ``git blame scripts/params.py`` yields the parameter change history, which
is exactly what a reviewer asks for.

Paths are ``Path`` objects. The step modules all normalise via
``Path(str(config.output_dir)).as_posix()``, so passing a ``Path`` produces
byte-identical log output to passing the original string literal.

The two environment overrides exist so a verification run can be written to a
scratch tree without touching ``results/``:

    POPGMM_RESULTS_ROOT=results_verify jupyter nbconvert --execute ...

That is a pure path substitution and cannot change any number.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(os.environ.get("POPGMM_RESULTS_ROOT", "results"))
DATA_ROOT = Path(os.environ.get("POPGMM_DATA_ROOT", "data"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

EIGENVAL_PATH = DATA_ROOT / "bbj.pca_base.eigenval"
SSCORE_PATH = DATA_ROOT / "cteph_agp3k_v6_wgs_merged.sample_qc.variant_qc.bbjproj.sscore"

# ---------------------------------------------------------------------------
# Per-step output directories
# ---------------------------------------------------------------------------

STEP1_DIR = RESULTS_ROOT / "01_hdbscan_filtering"
STEP2_DIR = RESULTS_ROOT / "02_gmm_clustering"
STEP3_DIR = RESULTS_ROOT / "03_gmm_component_merging"
STEP4_DIR = RESULTS_ROOT / "04_our_assignment"
STEP5_DIR = RESULTS_ROOT / "05_customize_cluster_assignment"
STEP6_DIR = RESULTS_ROOT / "06_mainland_subcluster_only"
STEP7_DIR = RESULTS_ROOT / "07_mainland_subcluster_confidence_distribution"
STEP8_DIR = RESULTS_ROOT / "08_mainland_subcluster_confidence_screening"
STEP9_DIR = RESULTS_ROOT / "09_threshold_sample_exports"

STEP4_TMP_DIR = STEP4_DIR / "STEP4_tmp"

# ---------------------------------------------------------------------------
# Cohort labels
# ---------------------------------------------------------------------------

CASE_LABEL = "CTEPH"
CONTROL_LABEL = "AGP3K"

# ---------------------------------------------------------------------------
# STEP2 -- GMM search
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# STEP3 -- component merging
# ---------------------------------------------------------------------------

MERGE_THRESHOLD_MAIN = 6.0

# Sensitivity runs, as {merge_threshold: output subdirectory}. The subdirectory
# name is not derivable from the threshold, so it is stated rather than built.
MERGE_THRESHOLD_SENSITIVITY: dict[float, str] = {2.5: "STEP3_tmp"}

# ---------------------------------------------------------------------------
# STEP4_tmp / STEP5 -- mainland rank progression and the cut
# ---------------------------------------------------------------------------

# There is deliberately no MAINLAND_MAX_RANK constant. How many mainland
# components exist is a property of the fitted model -- 17 under the previous
# float32 run, 16 under float64 -- so STEP4_tmp discovers it from the mainland
# component list (which gmm_component_merging already derives automatically via
# "merged cluster with the most pre-merge components, ties to the smallest id").
# A literal here went stale the moment the model changed.

# The rank cut: how many of the ranked mainland components are merged into the
# Mainland Subcluster.
#
# This is a HUMAN decision, stated here and nowhere else. STEP5 reads the rank
# back off STEP4_tmp's output and asserts it matches, so the value cannot drift
# between the two steps.
#
# Set to None to hand the choice to STEP4_tmp's Pareto analysis instead
# (Neff vs PC1-2 heterogeneity). That is a one-line change and the notebook
# handles it -- no assertion fires, the chosen rank is printed instead.
#
# Currently forced to 9. Note that under the float64 model the Pareto analysis
# would pick rank 5 (both Distance_To_Ideal and Utility_NeffMinusHet point
# there); 9 is a deliberate override, not an oversight. See README.
MAINLAND_RANK_K: int | None = 9

# ---------------------------------------------------------------------------
# STEP8 / STEP9 -- confidence thresholds
# ---------------------------------------------------------------------------

# Shared by the screening step and the export step; they must agree, or the
# summary table and the emitted keep-lists describe different cutoffs.
CONFIDENCE_THRESHOLDS: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 0.99)


def merge_threshold_dir(subdir: str) -> Path:
    """Output directory for a STEP3 sensitivity run."""
    return STEP3_DIR / subdir
