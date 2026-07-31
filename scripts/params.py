"""Single source of truth for pipeline paths and parameters.

Everything a run depends on is stated here exactly once: where the inputs live,
where each stage writes, the random seed, the cohort labels, and the three
selection parameters that define the deliverable sample lists.

This is a Python module rather than a YAML file on purpose. The stage configs
are already typed frozen dataclasses, so YAML would need an extra
mapping-and-validation layer -- more code, plus a place for an int parameter to
silently become a string. Half the values here are derived expressions, which
are natural in Python and awkward in YAML. And ``git blame`` on this file yields
the parameter history a reviewer will ask for.

Two things are deliberately absent, because they are properties of the fitted
model rather than choices: which components form the major cluster (derived by
the merging stage) and how many of them there are (the rank-selection stage
walks all of them). Literals for either go stale the moment the model changes.

Paths are ``Path`` objects. Every stage normalises via
``Path(str(config.output_dir)).as_posix()``, so passing a ``Path`` produces
byte-identical log output to passing a string literal.

The two environment overrides exist so a verification run can be written to a
scratch tree without touching ``results/``:

    POPGMM_RESULTS_ROOT=results_verify jupyter nbconvert --execute ...

That is a pure path substitution and cannot change any number.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(os.environ.get("POPGMM_RESULTS_ROOT", "results"))
DATA_ROOT = Path(os.environ.get("POPGMM_DATA_ROOT", "data"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

#: PCA eigenvalues of the reference panel, one per line, no header. Assumed to
#: cover every PC in the score file -- variance-explained is computed as
#: eigenvalue / sum(eigenvalues), so a truncated file makes every axis label wrong.
EIGENVAL_PATH = DATA_ROOT / "bbj.pca_base.eigenval"

#: PLINK2 --score output holding both cohorts:
#: `#FID IID PHENO1 ALLELE_CT NAMED_ALLELE_DOSAGE_SUM PC1_AVG ... PCn_AVG`
SSCORE_PATH = DATA_ROOT / "cteph_agp3k_v6_wgs_merged.sample_qc.variant_qc.bbjproj.sscore"

#: Samples whose IID starts with this prefix form the reference panel; the rest
#: are the study cohort.
REFERENCE_IID_PREFIX = "bbj_"

# ---------------------------------------------------------------------------
# Cohort labels
# ---------------------------------------------------------------------------
#
# The only place the specific study appears. Everything downstream refers to
# "case cohort" and "control cohort"; these two values put the study's names on
# the figures and in the count columns of the rank-selection tables.

CASE_LABEL = "CTEPH"
CONTROL_LABEL = "AGP3K"

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
#
# The deliverable sits at the top so it is unmistakable; the supporting stages
# are numbered in dependency order.

#: The deliverable: the cohort sample lists, the reference panel's own list,
#: and a table comparing them.
KEEP_LIST_DIR = RESULTS_ROOT / "keep_lists"

#: Modelling the reference panel: denoise, fit the mixture, merge components.
REFERENCE_MODEL_DIR = RESULTS_ROOT / "01_reference_model"
DENOISING_DIR = REFERENCE_MODEL_DIR / "denoising"
MIXTURE_DIR = REFERENCE_MODEL_DIR / "mixture_model"
MERGING_DIR = REFERENCE_MODEL_DIR / "component_merging"
THRESHOLD_ROBUSTNESS_DIR = MERGING_DIR / "threshold_robustness"

#: Projecting the study cohort into the mixture and assigning it.
ASSIGNMENT_DIR = RESULTS_ROOT / "02_cohort_assignment"

#: Evidence for the rank cut: effective sample size against residual spread.
RANK_SELECTION_DIR = RESULTS_ROOT / "03_rank_selection"

#: One subdirectory per subcluster variant (see SUBCLUSTER_VARIANTS).
SUBCLUSTER_DIR = RESULTS_ROOT / "04_subcluster_variants"

#: Environment and configuration snapshots for the run.
PROVENANCE_DIR = RESULTS_ROOT / "provenance"

# ---------------------------------------------------------------------------
# Reference-panel modelling
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

#: Dendrogram cut on the pairwise Mahalanobis distances between component means.
MERGE_THRESHOLD = 6.0

#: Additional thresholds run purely as robustness evidence for the major-cluster
#: identification. The mixture is already fitted, so each extra threshold costs
#: one merge plus one figure. Results are compared in
#: `threshold_robustness/major_cluster_robustness.tsv`.
MERGE_THRESHOLD_ROBUSTNESS: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 4.5, 8.0)

# ---------------------------------------------------------------------------
# The deliverable: three sample lists
# ---------------------------------------------------------------------------
#
# The major cluster is split into nested variants by a rank cut. Components are
# ranked by case/control ratio; including the top-k of them trades effective
# sample size (GWAS_Neff) against residual genetic spread (RGV, the root
# generalized variance of the retained samples on PC1-PC2). The rank-selection
# stage tabulates and plots that trade-off; the cut itself is a human decision.
#
#   full     -- every major-cluster component. No cut.
#   refined  -- the primary analysis set.
#   expanded -- a looser cut between refined and full, for sensitivity analysis.
#
# All three go through the same subcluster stage, so each gets the same figures
# and tables under `04_subcluster_variants/<variant>/` and is directly
# comparable with the others.

#: Rank cut for a variant: an int keeps the components ranked 1..k, "full" keeps
#: every major-cluster component, and "pareto" defers to the rank-selection
#: analysis (the notebook then prints the chosen rank rather than asserting one).
RankCut = int | Literal["full", "pareto"]

#: Primary analysis set. Currently a deliberate override: under this model the
REFINED_RANK_K: RankCut = 9

#: Sensitivity set, between refined and full.
EXPANDED_RANK_K: RankCut = 12

#: Variant name -> rank cut. Drives the subcluster stage and the keep-list names.
SUBCLUSTER_VARIANTS: dict[str, RankCut] = {
    "full": "full",
    "refined": REFINED_RANK_K,
    "expanded": EXPANDED_RANK_K,
}

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

#: What the major cluster is called on figures and in reports. The code names it
#: by its definition (the merged cluster holding the most components); this is
#: the population-genetic interpretation of that cluster, and is an assumption
#: the pipeline does not itself verify.
MAJOR_CLUSTER_DISPLAY_NAME = "Mainland"


def subcluster_dir(variant: str) -> Path:
    """Output directory for one subcluster variant."""
    return SUBCLUSTER_DIR / variant


def threshold_robustness_dir(threshold: float) -> Path:
    """Output directory for one robustness threshold."""
    return THRESHOLD_ROBUSTNESS_DIR / f"threshold_{threshold:.1f}".replace(".", "p")


