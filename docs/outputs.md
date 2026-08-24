# Outputs

Every file the pipeline writes, and what it is for. Stage numbers in brackets
refer to [`method.md`](method.md).

Everything lands under `results/`, or under whatever `POPGMM_RESULTS_ROOT` points
at. Nothing outside that directory is written.

---

## The deliverable — `keep_lists/`

| File | Contents |
|---|---|
| `full_mainland.fid_iid.txt` | Study samples in the complete major cluster |
| `narrow_mainland.fid_iid.txt` | Tightest cut — least residual spread, fewest samples |
| `intermediate_mainland.fid_iid.txt` | Between narrow and full |
| `reference_full_mainland.fid_iid.txt` | Reference-panel samples in the major cluster |
| `keep_list_summary.tsv` | The lists side by side: counts, balance, $N_{\mathrm{eff}}$, RGV, components |

Each `.fid_iid.txt` is headerless and tab-separated, `FID` then `IID`, which is
what PLINK/PLINK2 `--keep` expects. `mainland` in the filenames is the configured
display label (`params.MAJOR_CLUSTER_DISPLAY_NAME`), not a model output.

`reference_full_mainland` is not a cohort deliverable — it is the same selection
applied to the reference panel, for anyone who needs the panel side of it (to
re-derive a PCA, or as an ancestry-matched external control set). Its case and
control counts are zero by construction; its RGV is the useful number, being the
residual spread of the region the cohort variants approximate.

---

## Full tree

```text
results/
├── keep_lists/                          the deliverable, above
├── 01_reference_model/
│   ├── denoising/                 [2]
│   ├── mixture_model/             [3]
│   │   └── tmp/                         per-fit search audit
│   └── component_merging/         [4]
│       └── threshold_robustness/        one directory per alternative cut
├── 02_cohort_assignment/          [5]
│   ├── pc_space_global/           [8]
│   └── pc_space_mainland/         [8]
├── 03_rank_selection/             [6]
├── 04_subcluster_variants/        [7]
│   └── <variant>/                       narrow · intermediate · full
│       ├── pc_space_global/       [8]
│       └── pc_space_mainland/     [8]
└── provenance/
```

### The `pc_space_<basis>/` rule

Anything computable in more than one PC basis lives under `pc_space_<basis>/`.
Anything basis-independent, and anything that exists only in the fitted model's
own space, stays at the stage root. So the posterior tables and the assignment
overview figure sit at the top of `02_cohort_assignment/`, while the all-PC
comparison — which depends entirely on which axes you look along — is duplicated
per basis.

---

## Stage by stage

### `01_reference_model/denoising/` [2]

| File | Contents |
|---|---|
| `reference_samples_denoised.tsv` | The retained panel, with cluster label and noise flag |
| `denoising_summary.json` | Configuration and the resulting noise counts |
| `denoising_overview.png` | PCA, the noise call, and the retained clusters |

### `01_reference_model/mixture_model/` [3]

| File | Contents |
|---|---|
| `bic_search.tsv` | One row per candidate $K$: BIC, AIC, empty-component flag, timing |
| `component_summary.tsv` | Per-component weight, size and mean |
| `reference_samples_clustered.tsv` | The panel with its assigned component |
| `mixture_model_summary.json` | The selected $K$ and the fit configuration |
| `mixture_model_overview.png` | BIC curve, structure, cluster sizes, confidence |
| `tmp/search_report.md`, `tmp/search_*.jsonl` | Per-fit audit trail from the order search |

The fitted model itself is not persisted — it is the one artifact that exists
only in memory, which is why the notebook caches it (see `scripts/artifacts.py`).

### `01_reference_model/component_merging/` [4]

| File | Contents |
|---|---|
| `component_mahalanobis_distance.tsv` | The $D = [d_{ij}]$ matrix components are clustered on |
| `component_merge_map.tsv` | $k \mapsto c(k)$, plus which merged cluster is the major one |
| `merged_cluster_summary.tsv` | Per-merged-cluster sizes |
| `merged_posterior_probabilities.npy` | Posterior mass after the group-sum |
| `reference_samples_merged.tsv` | The panel with its merged-cluster label |
| `major_cluster_reference.{tsv,json}` | The major cluster's members and definition |
| `merge_summary.json` | Threshold, linkage, and resulting counts |
| `component_merging_overview.png` | Distance matrix, dendrogram, merged clusters, confidence |
| `threshold_robustness/` | The same merge repeated at each alternative threshold, plus `major_cluster_robustness.tsv` comparing which components the major cluster picks up |

The robustness table is what shows the major-cluster identification is stable: a
subset relationship across thresholds means a tighter cut subdivides the same
region rather than jumping elsewhere.

### `02_cohort_assignment/` [5]

| File | Contents |
|---|---|
| `cohort_posterior_probabilities.tsv` | Per-sample responsibilities, assignment, confidence |
| `major_cluster_component_ranks.tsv` | Per-sample component and rank, restricted to the major cluster |
| `cohort_cluster_statistics.tsv` | Per-cluster case/control counts, ratio and rank |
| `cohort_assignment_overview.png` | Study cohort, assigned component, confidence, statistics |
| `pc_space_<basis>/all_pcs_kde.{png,log}` | Case/control densities on every PC |
| `pc_space_<basis>/all_pcs_kde_tests.tsv` | Welch $t$ and Mann-Whitney per PC, BH-adjusted |

### `03_rank_selection/` [6]

Four figures and three tables. The figures are numbered in reading order and
answer one question each. Every formula is presented the same way — the question
it answers, the formula, then how to read its result — and every symbol in it
has a row in a band across the foot of that figure saying what it is. No figure
declares a canvas size: each is measured from its own content and made exactly
that tall.

| File | Contents |
|---|---|
| `00_problem.png` | What has to be decided, and the two quantities that decide it — $N_k$ and $H_k$, each derived |
| `01_narrow.png` | The first cut: pricing the walk against its own average exchange rate |
| `02_intermediate.png` | Three numbered sub-steps, each with the panel that backs it: the obstacle (A), the rule and every cut's distance (B), the weight (C) |
| `03_cohorts.png` | The three delivered sets, where they sit, why each one exists, and what each costs |
| `component_ranking.tsv` | Major-cluster components ordered by case/control ratio — the order the walk follows |
| `cut_record.tsv` | How each cut was arrived at, and whether the automatic and manual answers agree |
| `rank_decision_table.tsv` | Every number the figures draw |

The argument runs across the four in order:

1. The major cluster has $K = 17$ components. Order them by case/control ratio;
   cut $k$ keeps the top $k$. Along that walk
   $N_k = 4n_1n_2/(n_1+n_2)$ rises — the allele frequency cancels, so an
   unbalanced set is worth less than its raw total — and so does
   $H_k = |\Sigma_k|^{1/2d} = (\prod\sqrt{\lambda_i})^{1/d}$ with $d = 4$.
2. Because both rise, the walk has one average exchange rate
   $r = (N_K-N_1)/(H_K-H_1) = 381{,}519$, and each cut can be scored by
   $E_k = (N_k-N_1) - r(H_k-H_1)$ — its surplus in effective samples over paying
   that price. **`narrow` $= \arg\max_k E_k = 9$.** Cumulative rather than
   per-step: the per-step rate rebounds and would answer 13.
3. Case/control centroid distance
   $s_k = \hat{D}^2_k - d(1/n_1+1/n_2)$ reverses direction **7 times**, so it has
   no rate and step 2 cannot be repeated on it. The two axes are blended
   instead — $u_k(w) = wx_k + (1-w)\tilde{s}_k$, rescaled to $\tilde{H}_k$, then
   $k^{*}(w) = \arg\min_k\sqrt{\tilde{H}_k^2+(1-y_k)^2}$ — at
   $w = \tfrac{1}{2}$, which is the boundary of $w \geq 1-w$: below it the term
   built from phenotype labels would outweigh the one built from genotypes.
   **`intermediate` $= 12$**, at distance 0.3596 from the corner — 21% clear of
   the runner-up ($k = 15$, 0.4346), with nothing else within 10% — and stable
   across $w \in [0.37, 0.71]$.

   Panel B plots that distance for *every* cut rather than only the winner, so
   "nearest" can be checked rather than taken on faith; the scatter that shows
   what the distance is geometrically sits as an inset. It mirrors how
   `01_narrow.png` shows $E_k$ peaking.

   The selection rule is the proximity to the corner, not the weight. `02`
   draws all three in that order — the obstacle, then the rule, then the
   weight's robustness — because with only the first and last drawn it read as
   though $w$ were the criterion.
4. **`full`** is every major-cluster component — no rule; the population the
   other two are chosen inside.

Three are delivered rather than one because three different things can be the
dominant worry, and `03_cohorts.png` states that per row rather than restating
the rule that located each cut. Pick `narrow` when residual stratification is
the main worry: it buys the most homogeneity the walk offers before extra
components stop repaying their spread. Pick `intermediate` when both worries
apply at once: it is the only one of the three whose case/control gap is not
detectable, at 94% of `full`'s power. Pick `full` when power is the main worry
or a reference is wanted: nothing is selected, so nothing can have been selected
wrongly. The three are nested — `narrow` $\subset$ `intermediate` $\subset$
`full` — so this is a choice of where to stop, not of which list.

Significance is reported throughout and optimised against nowhere. The de-biasing
removes what sampling contributes on average; it does not say whether what is
left is real, and Hotelling's exact $F$ test does — 12 of the 17 cuts separate
significantly, which is what makes the second axis a phenomenon rather than
noise. Of the three delivered cohorts, **`intermediate` is the only one where
the gap is not detectable** ($p = 0.15$, against $3.9\times10^{-8}$ for `narrow`
and $0.0035$ for `full`); `narrow` sits at the strongest separation in the whole
walk. Neither fact selected a cut — both are properties of the deliverable,
reported in `03_cohorts.png`.

The second rescaling in step 3 is not cosmetic: without it the recorded distance
0.3596 is not reproducible (you get 0.3446). Derived values on the figures are
read from `cut_record.tsv` rather than recomputed, so figures and tables cannot
disagree. Nothing downstream reads any file here — the subcluster stage consumes
the resolved cuts in memory.

### `04_subcluster_variants/<variant>/` [7]

| File | Contents |
|---|---|
| `subcluster_posterior_probabilities.tsv` | Per-sample assignment under the recomputed composite posterior |
| `subcluster_group_statistics.tsv` | Per-group case/control counts |
| `subcluster_summary.json` | Which components the composite group contains |
| `subcluster_assignment_overview.png` | Cohort, recomputed assignment, confidence, statistics |
| `pc_space_<basis>/subcluster_view.png` | The PC1–PC2 view in that basis |
| `pc_space_<basis>/subcluster_view_counts.tsv` | The per-group counts the view draws |
| `pc_space_<basis>/all_pcs_kde.{png,log}`, `all_pcs_kde_tests.tsv` | All-PC comparison within the variant |

All three variants run through identical code, so they are directly comparable.

### `provenance/`

| File | Contents |
|---|---|
| `run_config_snapshot.json` | Every stage config, plus the derived quantities (major-cluster components, recommended rank, variant definitions) |
| `run_environment.json` | Run mode, Python and library versions, platform, threading environment, HDBSCAN backend |

Diffing two config snapshots proves a refactor did not alter a parameter without
spending a full run to find out.

---

## Checking a run

`tools/verify_results.py` compares two result trees, or one tree against
`tools/baseline_manifest.json`:

```bash
# compare two trees
python -m tools.verify_results --baseline results --candidate results_other

# check one tree against the committed fingerprints
python -m tools.verify_results --candidate results --manifest tools/baseline_manifest.json

# regenerate the fingerprints after an intended change
python -m tools.verify_results --baseline results --write-manifest tools/baseline_manifest.json
```

`--candidate` is the tree being checked and is always required except when
writing a manifest; `--baseline` names the trusted tree to compare against, and
is only meaningful together with `--candidate` or `--write-manifest`.

Comparison is by file kind: TSV numerically, JSON value-by-value, `.npy`
elementwise, logs and keep-lists byte-for-byte, with the results-root path
normalised so a run into a different directory still matches. Timestamped audit
files are compared with the volatile keys removed rather than skipped.

It refuses to verify a tree produced with `RUN_MODE="resume"`. A resumed run
reuses cached upstream results, and a cached stage writes none of its output
files — so the tree would be a mix of this run and a previous one.

---

## Not tracked in git

Some large regenerable intermediates are gitignored: the per-threshold copies of
the merged panel and the posterior arrays. Their checksums are in
`tools/baseline_manifest.json`, so a run can still be validated without them.
