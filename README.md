# PopGMM

**Ancestry-homogeneous sample selection for association studies, via PCA + Gaussian mixture models.**

A study cohort is projected onto a population PCA reference panel. The panel is
denoised with HDBSCAN and modelled with a full-covariance Gaussian mixture;
study samples then receive posterior membership probabilities in that mixture.
The dominant ("major") cluster is narrowed by an explicit trade-off between
effective sample size and residual genetic spread, producing the sample lists
that association analysis should run on.

The pipeline is generic. The specific study it is currently configured for —
cohort labels, input filenames, the selected rank cuts — lives entirely in
[`scripts/params.py`](scripts/params.py).

---

## The deliverable: three sample lists

Everything the pipeline exists to produce is in `results/keep_lists/`:

| List | Definition | Use |
|---|---|---|
| `full_mainland` | every component of the major cluster | the widest defensible set |
| `refined_mainland` | rank cut chosen on effective sample size vs residual spread | **primary analysis** |
| `expanded_mainland` | a looser cut between refined and full | sensitivity analysis |

Each is a headerless, tab-separated `FID IID` file:

```bash
plink2 --pfile <dataset> \
       --keep results/keep_lists/refined_mainland.fid_iid.txt \
       --make-pgen --out <dataset>.ancestry_qc
```

`keep_list_summary.tsv` compares them on size, case/control balance, `GWAS_Neff`
and `PC12_RGV` — the table to reproduce in a Methods section.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate popgmm
python -m ipykernel install --user --name popgmm --display-name popgmm

# then: open workflow.ipynb and Restart & Run All   (~12 minutes)
```

Runs top to bottom as a linear dependency chain; the working directory must be
the repository root.

---

## Pipeline

`workflow.ipynb` is the only orchestrator. Each module in `scripts/` exposes one
frozen `…Config` dataclass and one `run_*` entry point.

| Stage | Module | Output |
|---|---|---|
| Data loading | `data_loading` | in memory |
| Reference panel — denoising | `hdbscan_filtering` | `01_reference_model/denoising/` |
| Reference panel — mixture model | `gmm_clustering`, `gmm_search_audit` | `01_reference_model/mixture_model/` |
| Reference panel — component merging | `gmm_component_merging` | `01_reference_model/component_merging/` |
| Major-cluster robustness | *(same module)* | `…/component_merging/threshold_robustness/` |
| Cohort assignment | `cohort_assignment`, `major_cluster_all_pcs_kde` | `02_cohort_assignment/` |
| Rank selection | `rank_selection` | `03_rank_selection/` |
| Subcluster variants | `subcluster_assignment`, `subcluster_view`, `subcluster_all_pcs_kde` | `04_subcluster_variants/{refined,expanded}/` |
| **Keep lists** | `keep_lists` | **`keep_lists/`** |

Supporting modules: `params` (all paths and parameters), `common` (shared
helpers and metrics), `artifacts` (stage caching).

GMM/EM mathematics and convergence criteria are documented separately in
[`gmm_convergence_diagram.EN.md`](gmm_convergence_diagram.EN.md) and
[`gmm_convergence_diagram.CN.md`](gmm_convergence_diagram.CN.md).

---

## How the selection works

**The major cluster is derived, not configured.** Component merging picks the
merged cluster holding the most pre-merge components (ties to the smallest id).
Currently that is 16 of 26 components and 92.2 % of the reference panel, so the
choice is unambiguous — but the *population-genetic interpretation* of that
cluster is an assumption the pipeline does not verify.
`params.MAJOR_CLUSTER_DISPLAY_NAME` is what appears on figures.

`threshold_robustness/major_cluster_robustness.tsv` tests whether the
identification is stable across dendrogram cut heights. At every threshold tried
the selected component set is a **strict subset** of the main analysis
(Jaccard 0.44 → 1.00 as the cut loosens): tightening the cut carves the same
region more finely rather than jumping elsewhere.

**The rank cut is a human decision, supported by evidence.** Components are
ranked by case/control ratio; including the top-k trades `GWAS_Neff`
(`4 / (1/n_case + 1/n_control)`) against `PC12_RGV` (residual genetic spread,
`det(Sigma)**0.25` on PC1–PC2). `rank_selection` tabulates and plots the whole
frontier; `params.REFINED_RANK_K` and `params.EXPANDED_RANK_K` record the chosen
cuts. Set either to `None` to delegate the choice to the Pareto optimum, in
which case the notebook prints the chosen rank instead of asserting one.

---

## Inputs

| File | Format |
|---|---|
| `data/bbj.pca_base.eigenval` | plain text, one eigenvalue per line; must cover every PC in the score file |
| `data/*.sscore` | PLINK2 `--score` output: `#FID IID PHENO1 ALLELE_CT NAMED_ALLELE_DOSAGE_SUM PC1_AVG … PCn_AVG` |

Cohorts are split by `params.REFERENCE_IID_PREFIX` on `IID`; case/control comes
from the phenotype column. PC columns are auto-detected (`PC<n>` or
`PC<n>_AVG`), so the PC count is whatever the input provides. `data/` is not
tracked in git.

---

## Key results

| Stage | Value |
|---|---|
| Reference panel loaded | 183,013 |
| After denoising | 181,815 (1,198 noise, 0.65 %) |
| Selected mixture | **k = 26** by minimum BIC (BIC = −2,682,560.08) |
| After merging at threshold 6.0 | 6 clusters |
| Major cluster | 16 components, 92.2 % of the panel |

| Keep list | Rank | Components | n | Cases | Controls | GWAS_Neff | PC12_RGV |
|---|---|---|---|---|---|---|---|
| `full` | — | 16 | 3,099 | 434 | 2,665 | 1492.88 | 0.006962 |
| `refined` | 9 | 9 | 2,138 | 408 | 1,730 | 1320.56 | 0.005040 |
| `expanded` | 12 | 12 | 2,637 | 430 | 2,207 | 1439.53 | 0.005864 |

---

## Configuration

Every path, seed, label and cut lives in [`scripts/params.py`](scripts/params.py).
Two things are deliberately **not** constants there, because they are properties
of the fitted model and go stale the moment it changes: which components form
the major cluster, and how many there are.

Two environment overrides, both pure path substitutions:

```bash
POPGMM_RESULTS_ROOT=results_verify   # write a run somewhere other than results/
POPGMM_DATA_ROOT=/path/to/data
```

### Run modes

`RUN_MODE` at the top of the notebook:

- `"fresh"` (default) — execute every stage and write all output. **The only mode
  valid for publication or verification.**
- `"resume"` — reuse cached upstream artifacts to skip the mixture search while
  iterating. Two measured consequences: the cached stages write nothing, and the
  **figures of later stages change** (data stays byte-identical, but skipping
  them means their plotting code never runs, so later stages inherit a different
  global `rcParams` state). Never publish figures from a resume run;
  `verify_results.py` refuses to verify such a tree.

The cache lives in `.cache/popgmm/` (untracked), keyed on the stage config,
upstream keys, input fingerprints and library versions.

---

## Verification

The pipeline is deterministic, so re-run-and-diff is a valid acceptance test
with **no numeric tolerance**:

```bash
POPGMM_RESULTS_ROOT=results_verify jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=python3 --output-dir=/tmp workflow.ipynb
python -m tools.verify_results --baseline results --candidate results_verify

# without a baseline tree present (e.g. a fresh clone):
python -m tools.verify_results --manifest tools/baseline_manifest.json \
       --candidate results_verify
```

Exit code 0 means nothing failed. Any difference in a numeric artifact or a
figure is a failure; only embedded timestamps and the results-root path are
normalised away. One extra invariant is asserted for free: the k minimising BIC
must not move.

Two independent full runs agree on every numeric artifact and every figure — the
only files that differ carry a timestamp or the output path. Measurement and the
float32 → float64 history are in
[`docs/reproducibility_probe.md`](docs/reproducibility_probe.md); the previous
float32 analysis is preserved at the git tag `results-float32-final`.

---

## Environment

Pinned in [`environment.yml`](environment.yml);
the machine that produced `results/` is recorded in
[`docs/environment_snapshot.txt`](docs/environment_snapshot.txt), and every run
writes its own `results/provenance/run_environment.json`.

Python 3.12.6 · numpy 2.4.2 · pandas 2.2.3 · scikit-learn 1.8.0 · scipy 1.17.0 ·
hdbscan 0.8.42 · matplotlib 3.9.2 · seaborn 0.13.2

Two pins are load-bearing rather than hygiene:

- **`hdbscan` is required, not optional.** `hdbscan_filtering` is pinned to
  python-hdbscan and raises rather than falling back to
  `sklearn.cluster.HDBSCAN`: the two disagree on the noise set, so a silent
  substitution would change the cohort while the summary still reported the
  requested parameters. Every configured parameter is verified against the
  constructed estimator for the same reason.
- **`scikit-learn`** drives `GaussianMixture` with `init_params="kmeans"`, whose
  RNG stream and `n_init` defaults have changed across minor releases — that
  would move the selected component count and renumber every component.

Computation is float64 throughout. Under float32 the log-likelihood sum was
sensitive to BLAS reduction order and the mixture search log was not reproducible
between runs even with the seed fixed.

---

## Repository layout

```
workflow.ipynb              the pipeline; run top to bottom
scripts/
  params.py                 all paths, labels, seeds, rank cuts
  common.py                 shared helpers (PC columns, palettes, BH-FDR, Neff, RGV)
  artifacts.py              content-addressed cache for the upstream stages
  keep_lists.py             emits the deliverable
  data_loading.py           cohort split
  hdbscan_filtering.py      reference-panel denoising
  gmm_clustering.py         mixture fit + BIC selection
  gmm_search_audit.py       search audit trail
  gmm_component_merging.py  merging, major cluster, threshold robustness
  cohort_assignment.py      posterior assignment of the study cohort
  major_cluster_all_pcs_kde.py   all-PC case/control comparison
  rank_selection.py         Neff vs RGV trade-off
  subcluster_assignment.py  composite-group reassignment
  subcluster_view.py        PC1-PC2 view
  subcluster_all_pcs_kde.py all-PC comparison per variant
tools/
  verify_results.py         result-comparison oracle
  baseline_manifest.json    committed fingerprints of results/
docs/
  reproducibility_probe.md  determinism measurement and dtype history
  environment_snapshot.txt  interpreter, BLAS and package provenance
results/                    see note below
data/                       inputs (not tracked)
```

Large regenerable intermediates under `results/` are untracked — several
redundant copies of the same PC matrix plus the posterior arrays. They are
reproduced exactly by re-running, and their checksums are in
`tools/baseline_manifest.json`, so `verify_results --manifest` validates a run
without them.

---

## Known issues

- **The plotting modules mutate global `plt.rcParams` without restoring them**,
  and several call `plt.style.use()` without setting the shared style dict, so
  they inherit whatever the previously executed cell left behind. Figures
  therefore depend on *which stages ran before them*, not only on their own data
  — demonstrated by the `RUN_MODE="resume"` measurement above. Deliberately not
  fixed: the committed figures encode the current state. Run the notebook top to
  bottom in `"fresh"` mode and they reproduce exactly. If the figures are ever
  regenerated for publication, fix this first and regenerate all of them together.
- **The rank cut is a human override, not the model's own choice.**
  `params.REFINED_RANK_K = 9` forces `rank_selection`'s recommendation, whereas
  the current model's Pareto analysis points to rank 5. This is deliberate, but
  it means the decision table and the applied cut disagree — state the override
  explicitly in Methods. Setting it to `None` hands the choice back.
- `GMMConfig.use_zscale=True` is not usable end to end — `cohort_assignment`
  raises because the training scaler is never persisted. The committed run uses
  `use_zscale=False`.
