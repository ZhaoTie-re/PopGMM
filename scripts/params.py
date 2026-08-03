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

#: A second PCA, built on the major cluster alone, with the major-cluster study
#: samples projected into it. Optional: absent, the pipeline reports residual
#: spread on the global PCA only.
#:
#: The global PCA's leading axes separate the major cluster from the other
#: regions, so they are nearly constant *within* it and resolve residual
#: heterogeneity poorly -- which is exactly what the rank cut trades against. A
#: PCA fitted to the major cluster spends its axes on the structure that remains
#: (PC1 explains 13.2% here against the global 39.2%).
#:
#: Same format as SSCORE_PATH, and it must cover every sample the rank walk can
#: retain; the stage raises rather than dropping a sample it cannot place.
MAINLAND_SSCORE_PATH = DATA_ROOT / "mainland" / "mainland.bbjproj.sscore"
MAINLAND_EIGENVAL_PATH = DATA_ROOT / "mainland" / "mainland.pca_base.eigenval"

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
# generalized variance of the retained samples). The rank-selection stage
# tabulates and plots that trade-off; the cut itself is a human decision.
#
# RGV is reported in two bases, side by side and never compared with each other:
# RGV_Global on the global PCA's PC1-PC2 (always two axes, so the number stays
# comparable with earlier runs), and RGV_Mainland on the major-cluster PCA over
# MAINLAND_RGV_N_PCS axes. RGV_BASIS picks which one the Pareto front and the
# recommendation are computed from.
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

#: Leading axes of the major-cluster PCA that RGV_Mainland is computed over,
#: as ``det(Sigma)**(1/2d)`` with ``d = MAINLAND_RGV_N_PCS``. RGV_Global is fixed
#: at PC1-PC2; only this one is configurable.
#:
#: At d = 2 the mainland axes still track the global PC1-PC2 residual almost
#: proportionally (ratio 1.07-1.27 across every rank), so the two bases rank the
#: cuts identically and the second basis adds nothing. More axes take in the
#: weaker structure that the leading pair misses, which is the point of fitting a
#: PCA to the major cluster at all.
#:
#: The value is not comparable across settings of this parameter: it is a
#: geometric-mean SD over however many axes are included, so it falls as d rises
#: (0.00744 / 0.00654 / 0.00551 / 0.00466 at d = 2 / 3 / 5 / 10 for the full set).
MAINLAND_RGV_N_PCS: int = 4

#: Which basis the Pareto front and the recommended rank are computed from.
#: "mainland" needs MAINLAND_SSCORE_PATH to be present and loaded.
RGV_BASIS: Literal["global", "mainland"] = "mainland"

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


def pc_space_dir(stage_dir: Path, basis: str) -> Path:
    """Output directory for one PC basis under a stage directory.

    The analyses that can be computed in more than one PCA basis live in sibling
    directories named for the basis; anything basis-independent, and anything
    that exists only in the fitted model's own space, stays at the stage root.
    """
    return stage_dir / f"pc_space_{basis}"


def threshold_robustness_dir(threshold: float) -> Path:
    """Output directory for one robustness threshold."""
    return THRESHOLD_ROBUSTNESS_DIR / f"threshold_{threshold:.1f}".replace(".", "p")


