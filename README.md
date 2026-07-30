# PopGMM

**Probabilistic ancestry inference and population-stratification control via PCA + Gaussian mixture models.**

A CTEPH case / AGP3K control cohort is projected onto the BioBank Japan (BBJ)
PCA reference space. The reference is denoised with HDBSCAN and modelled with a
full-covariance Gaussian mixture; study samples then receive posterior
membership probabilities in that mixture. The mainland Japanese cluster is
narrowed to a homogeneous subcluster and filtered by posterior confidence,
producing PLINK-ready keep-lists for downstream association analysis.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate popgmm
python -m ipykernel install --user --name popgmm --display-name popgmm

# then: open workflow.ipynb and Restart & Run All   (~13 minutes)
```

The notebook runs top to bottom as a linear dependency chain, and must be
executed with the repository root as the working directory.

---

## Pipeline

`workflow.ipynb` is the only orchestrator; `scripts/` holds the step modules,
each exposing a frozen `…Config` dataclass and a single `run_*` entry point.

| Step | Module | What it does | Output |
|---|---|---|---|
| STEP0 | `data_loading` | Split the shared `.sscore` by `IID` prefix into BBJ reference and study cohort; derive case/control lists from `PHENO1` | in memory |
| STEP1 | `hdbscan_filtering` | HDBSCAN denoising of the reference in PC1–PC2 space | `01_hdbscan_filtering/` |
| STEP2 | `gmm_clustering`, `gmm_search_audit` | Fit `k = 2..100` full-covariance GMMs, select minimum BIC with no empty cluster | `02_gmm_clustering/` |
| STEP3 | `gmm_component_merging` | Merge components by Mahalanobis distance (`S_ij = ½(Σ_i + Σ_j)`) + average linkage | `03_gmm_component_merging/` |
| STEP3_tmp | *(same module)* | Merge-threshold sensitivity run | `03_gmm_component_merging/STEP3_tmp/` |
| STEP4 | `our_assignment`, `mainland_all_pcs_kde` | Posterior assignment of study samples; 20-PC case/control KDE within mainland | `04_our_assignment/` |
| STEP4_tmp | `step4_tmp_mainland_rank_progression` | Rank mainland components by case/control ratio; cumulative Neff vs heterogeneity trade-off | `04_our_assignment/STEP4_tmp/` |
| STEP5 | `customize_cluster_assignment` | Merge the top-ranked mainland components into one group and reassign globally | `05_customize_cluster_assignment/` |
| STEP6 | `mainland_subcluster_only`, `mainland_subcluster_all_pcs_kde` | Subcluster-only scatter, 20-PC KDE, unfiltered export | `06_mainland_subcluster_only/` |
| STEP7 | `mainland_subcluster_confidence_distribution` | Confidence histogram and CDF | `07_…_confidence_distribution/` |
| STEP8 | `mainland_subcluster_confidence_threshold_screening` | Retained/removed rates per threshold | `08_…_confidence_screening/` |
| STEP9 | `mainland_subcluster_threshold_sample_export` | **Emit the PLINK keep-lists** | `09_threshold_sample_exports/` |

Supporting modules: `params` (all paths and parameters), `common` (shared
helpers), `artifacts` (step caching).

The GMM/EM mathematics and convergence criteria are documented separately in
[`gmm_convergence_diagram.EN.md`](gmm_convergence_diagram.EN.md) and
[`gmm_convergence_diagram.CN.md`](gmm_convergence_diagram.CN.md).

---

## Inputs

| File | Format |
|---|---|
| `data/bbj.pca_base.eigenval` | plain text, one eigenvalue per line (20 PCs) |
| `data/cteph_agp3k_v6_wgs_merged.sample_qc.variant_qc.bbjproj.sscore` | PLINK2 `--score` output: `#FID IID PHENO1 ALLELE_CT NAMED_ALLELE_DOSAGE_SUM PC1_AVG … PC20_AVG` |

Cohorts are split by the `bbj_` prefix on `IID`; `PHENO1` is 2 = case, 1 = control.
`data/` is not tracked in git.

## Deliverables

`results/09_threshold_sample_exports/retained_all_thr_*.fid_iid.txt` — headerless
`FID IID` keep-lists:

```bash
plink2 --pfile <dataset> \
       --keep results/09_threshold_sample_exports/retained_all_thr_0.9000.fid_iid.txt \
       --make-pgen --out <dataset>.ancestry_qc
```

---

## Key results

| Stage | Value |
|---|---|
| BBJ reference loaded | 183,013 |
| After HDBSCAN denoising | 181,815 (1,198 noise, 0.65 %) |
| Selected mixture | **k = 26** by minimum BIC (BIC = −2,682,560.08) |
| After merging at threshold 6.0 | 6 clusters |
| Mainland cluster | 16 pre-merge components |
| Mainland samples (study cohort) | 3,099 / 3,571 — 434 cases, 2,665 controls |
| Rank cut | k = 9 → components {0, 2, 4, 5, 12, 13, 16, 19, 22} |
| Mainland Subcluster | 2,138 / 3,571 — 408 cases, 1,730 controls |

Retained after confidence filtering:

| Threshold | Cases | Controls |
|---|---|---|
| 0.3575 (case minimum) | 407 | 1,721 |
| 0.80 | 320 | 962 |
| 0.85 | 301 | 809 |
| 0.90 | 272 | 650 |
| 0.95 | 199 | 428 |
| 0.99 | 53 | 102 |

The pipeline computes in float64. An earlier version cast to float32, which made
the model search sensitive to BLAS reduction order and produced a different
(and less reproducible) analysis; that version is preserved at the git tag
`results-float32-final`. See
[`docs/reproducibility_probe.md`](docs/reproducibility_probe.md) for the
measurement and the full before/after comparison.

---

## Configuration

Every path, seed, label and threshold lives in
[`scripts/params.py`](scripts/params.py). Do not hard-code them in the notebook
— two parameters used to be written in two cells each with nothing keeping them
in sync.

The mainland rank cut is stated once as `params.MAINLAND_RANK_K`; STEP5 reads
`recommended_rank` back off STEP4_tmp's output and asserts it matches, so a
divergence fails loudly instead of silently analysing the wrong components. Set
it to `None` to let STEP4_tmp's Pareto analysis choose instead.

Two things are deliberately **not** constants, because they are properties of the
fitted model and go stale the moment it changes: the mainland component ids
(derived by `gmm_component_merging` as "the merged cluster with the most
pre-merge components, ties to the smallest id") and how many of them there are
(STEP4_tmp walks all of them when `max_rank` is left at `None`). Under float32
there were 17; under float64 there are 16.

Two environment overrides, both pure path substitutions:

```bash
POPGMM_RESULTS_ROOT=results_verify   # write a run somewhere other than results/
POPGMM_DATA_ROOT=/path/to/data
```

### Run modes

`RUN_MODE` at the top of the notebook:

- `"fresh"` (default) — execute every step and write all of its output files.
  **Use this for any run whose results you intend to keep, publish or verify.**
- `"resume"` — reuse cached STEP0–STEP2 artifacts to skip the `k = 2..100` GMM
  search, which dominates the runtime. Under the previous float32 pipeline this
  cut a run from 6 min 52 s to 47 s; the float64 search is ~2.3× more expensive,
  so the saving is larger still (not separately measured).

`"resume"` is a development convenience with two consequences, both measured:

1. A cache hit skips the function, so STEP1 and STEP2 write nothing — their 12
   output files are simply absent from the results tree.
2. **The figures of later steps change.** Data is unaffected — every file a
   resume run does write, including all keep-lists, is byte-identical — but
   `gmm_component_merging_overview.png` shifts by ~3.4 % of its pixels
   (measured under float32; the mechanism is dtype-independent). The cause is the
   `rcParams` leak described under Known issues: skipping STEP1/2 means their
   plotting code never runs, so STEP3 onward inherits a different global style
   state.

The mode is recorded in `run_environment.json`, and `verify_results.py` refuses
to verify a tree produced with it.

The cache lives in `.cache/popgmm/` (untracked), keyed on the step config,
upstream step keys, input fingerprints and library versions — so any change to
parameters, inputs or dependencies misses. The results-root prefix is normalised
out of the key, so a run into `results_verify` reuses a cache populated by a run
into `results`; genuine subdirectory differences (`STEP3_tmp`) still miss.

---

## Verification

The results are final, so the repository ships an oracle for proving a change
did not alter them:

```bash
# full run into a scratch tree, then compare
POPGMM_RESULTS_ROOT=results_verify jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=python3 --output-dir=/tmp workflow.ipynb
python -m tools.verify_results --baseline results --candidate results_verify \
       --report tools/verify_report.md

# or, without a baseline tree present (e.g. a fresh clone):
python -m tools.verify_results --manifest tools/baseline_manifest.json \
       --candidate results_verify
```

Exit code 0 means nothing failed. The pipeline is deterministic, so **there is no
per-file numeric tolerance**: any difference in a numeric artifact or a figure is
a failure. Only two things are normalised away — embedded UTC timestamps in the
audit logs and PDF, and the results-root path that a run records in its `.log`
files and config snapshot. One extra invariant is asserted for free: the k
minimising BIC must not move.

A cheap pre-flight that needs no pipeline run:

```bash
diff results/run_config_snapshot.json results_verify/run_config_snapshot.json
```

### Reproducibility status

**Fully deterministic.** Two independent full runs in the same environment give
60 pass / 0 warn / 0 fail with no numeric tolerance applied. 52 of 60 artifacts
are raw byte-identical, including **all 12 figures**; the other 8 differ only in
embedded timestamps or the results-root path they record. No numeric value
varies. Thread pinning is not required — parallelism stays on.

This is a stronger claim than the previous float32 pipeline could support, where
15–18 of the 99 candidate model fits varied by up to 11.27 BIC units between
runs. Measurement, mechanism and the full before/after comparison are in
[`docs/reproducibility_probe.md`](docs/reproducibility_probe.md).

---

## Environment

Pinned in [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml);
the machine that produced `results/` is recorded in
[`docs/environment_snapshot.txt`](docs/environment_snapshot.txt).

Python 3.12.6 · numpy 2.4.2 · pandas 2.2.3 · scikit-learn 1.8.0 · scipy 1.17.0 ·
hdbscan 0.8.42 · matplotlib 3.9.2 · seaborn 0.13.2

Every run records its own provenance to `results/run_environment.json`, including
the HDBSCAN backend that actually executed and the BLAS thread settings.

Two pins are load-bearing rather than hygiene:

- **`hdbscan` is required, not optional.** `hdbscan_filtering` imports
  python-hdbscan first and *silently falls back* to `sklearn.cluster.HDBSCAN`
  when it is absent. The two backends disagree on the noise set, so an install
  without it yields a different cohort.
- **`scikit-learn`** drives `GaussianMixture` with `init_params="kmeans"`, whose
  RNG stream and `n_init` defaults have changed across minor releases — that
  would move `best_k` and renumber every component id.

---

## Repository layout

```
workflow.ipynb              the pipeline (14 code cells, run top to bottom)
scripts/
  params.py                 all paths, seeds, labels, thresholds
  common.py                 shared helpers (PC columns, palettes, BH-FDR, logging)
  artifacts.py              content-addressed cache for STEP0-STEP2
  data_loading.py           STEP0
  hdbscan_filtering.py      STEP1
  gmm_clustering.py         STEP2
  gmm_search_audit.py       STEP2 convergence/BIC audit logs
  gmm_component_merging.py  STEP3
  our_assignment.py         STEP4
  mainland_all_pcs_kde.py   STEP4 20-PC KDE
  step4_tmp_mainland_rank_progression.py                  STEP4_tmp
  customize_cluster_assignment.py                         STEP5
  mainland_subcluster_only.py                             STEP6
  mainland_subcluster_all_pcs_kde.py                      STEP6 20-PC KDE
  mainland_subcluster_confidence_distribution.py          STEP7
  mainland_subcluster_confidence_threshold_screening.py   STEP8
  mainland_subcluster_threshold_sample_export.py          STEP9
tools/
  verify_results.py         result-comparison oracle
  baseline_manifest.json    committed fingerprints of results/
docs/
  reproducibility_probe.md  evidence that a re-run reproduces results/
  environment_snapshot.txt  interpreter, BLAS and package provenance
results/                    per-step outputs (see note below)
data/                       inputs (not tracked)
```

Six large regenerable intermediates under `results/` are untracked — four
redundant copies of the same 181,815 × 20 PC matrix plus two posterior arrays,
218 MB in total. They are reproduced exactly by re-running the notebook and
their checksums are in `tools/baseline_manifest.json`, so
`verify_results --manifest` validates a run without them.

---

## Known issues

- **The plotting modules mutate global `plt.rcParams` without restoring them**,
  and four of them call `plt.style.use()` without setting the shared style dict,
  so they inherit whatever the previously executed cell left behind. The figures
  therefore depend on *which steps ran before them*, not only on their own data.
  Demonstrated by the `RUN_MODE="resume"` measurement above: skipping STEP1–2
  leaves every data file byte-identical while shifting 3.4 % of the pixels in
  `gmm_component_merging_overview.png`.

  This is deliberately **not** fixed — the committed figures encode the current
  state, and wrapping the mutations in `rc_context` would change them. Run the
  notebook top to bottom in `"fresh"` mode, as intended, and the figures
  reproduce exactly. If the figures are ever regenerated for publication, fix
  this first and regenerate all of them together.
- **The rank cut is a human override, not the model's own choice.**
  `params.MAINLAND_RANK_K = 9` forces STEP4_tmp's recommendation, whereas the
  current model's Pareto analysis points to rank 5 (both `Distance_To_Ideal` and
  `Utility_NeffMinusHet` in `STEP4_tmp/mainland_rank_decision_table.tsv`). This
  is deliberate, but it means the decision table and the applied cut disagree —
  state the override explicitly in Methods. Setting `MAINLAND_RANK_K = None`
  hands the choice back to the Pareto analysis; the notebook prints the chosen
  rank instead of asserting one, and nothing else needs changing.
- `GMMConfig.use_zscale=True` is not usable end to end — `our_assignment` raises
  because the training scaler is never persisted. The committed run uses
  `use_zscale=False`.
