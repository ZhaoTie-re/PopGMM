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

Three figures and three tables. The argument is in three parts, so it is in
three figures, and all three are laid out by the same row framework — a row of
cells, each a heading, a small plot, the equations that plot draws, and the
answer they reach, joined left to right by arrows. An earlier version had four
figures with four different layouts and the reader had to assemble the line
through them; here the line *is* the layout.

| File | Contents |
|---|---|
| `00_problem.png` | The problem, and the three quantities we watch: statistical power `N_eff`, residual stratification `H`, and the case/control shift `s_k` |
| `01_tradeoff.png` | How they are traded off: one average rate prices power against `H` and fixes `narrow`; `s_k` reverses so it has no rate, which is why it is blended instead and fixes `intermediate`; and the weight does not decide the answer |
| `02_cohorts.png` | The three cohorts, where they stop on the trade-off, and when to use each |
| `component_ranking.tsv` | Major-cluster components ordered by case/control ratio — the order the walk follows |
| `cut_record.tsv` | How each cut was arrived at, and whether the automatic and manual answers agree |
| `rank_decision_table.tsv` | Every number at every `k` |

The equations are on the figures because they are the reasoning. Two things
fail the build rather than producing a wrong figure: either criterion drawn on
`01` peaking somewhere other than the cut `cut_record.tsv` records, and any
equation or answer running out of its own column.

The notes below are the reviewer's layer — what each choice rules out, and the
alternatives it was taken against. None of it is needed to read the figures.

#### Methodological notes

These were carried on the figures themselves until they grew to two thirds of
the text on them. They are the argument for each choice, and prose belongs here.

**Basis and dimension of $H$.** $H$ could be measured on the global PCA's
leading pair, and earlier versions of this analysis were; it is measured instead
on 4 axes of a PCA fitted to the major cluster, and the choice changes which cut
wins. The global PC1–PC2 are dominated by the split between the major cluster
and everything outside it, so inside the cluster — which is all these cuts ever
contain — that pair carries little of the remaining structure and spread on it is
close to flat along the walk. A basis fitted to the cluster puts the residual
structure on its own leading axes, and 4 of them rather than 2 because the pair
alone leaves visible structure on the next two. The $1/2d$ exponent keeps the
result in SD units at any $d$. Two values of $H$ are comparable only when they
share a basis *and* a $d$; everything in this stage shares both.

**Effective size against head-count.** The walk adds cases and controls at very
different rates, so raw totals would credit a cut for samples that add almost
nothing to power. At the widest cut that is the difference between 3,101 samples
and 1,507 effective ones.

**End-to-end pricing.** $\gamma$ takes a single rate from the two ends of the
walk and $E_k$ scores every cut against it. The alternative — pricing each step
against its predecessor — asks a different question and gets a different answer:
a per-step rate is not monotone here, it crosses the average repeatedly, so "the
last step that paid above the average" lands on a late cut for no reason beyond
where the noise in one step fell. The cumulative form asks whether the walk up to
that cut has repaid what it took on, which is a property of the retained set
rather than of the component that entered last.

**Interpretation of the margin.** The peak leads the runner-up by 3.1 effective
samples out of 525.1, so neighbouring cuts price about the same. That supports
the reading that any cut in the neighbourhood is defensible on this criterion; it
does not support treating the peak as sharp: `rank_decision_table.tsv` carries
$E_k$ at every cut, not only at the winner. The margin is reported, never
optimised.

**The second axis, and whether it is real.** Spread says how wide the retained
set is, not whether the two arms sit at different places inside it — and only
that biases an association test, so a set can be homogeneous and still be the
wrong one to run on. The sampling floor $d(1/N_{case}+1/N_{ctrl})$ matters at
this scale: the retained set grows more than tenfold along the walk, so a raw
$\hat{D}^2$ would fall across it for arithmetic reasons alone. Subtracting the
floor does not say the remainder is anything; Hotelling's exact $F$ test does,
and 12 of 17 cuts separate at $P<0.05$, which is what makes this axis a
phenomenon rather than noise. $s_k$ also reverses direction 7 times, which is why
the pricing argument of the first cut cannot be repeated on it.

**Why (2) rescales twice.** $\tilde H_k$ and $\tilde s_k$ are already on $[0,1]$,
so $u_k(w)$ cannot leave the interval — but its own range is narrower, because
the two terms peak at different cuts and their average never reaches either end.
$k^{*}$ then measures a distance in a unit square; left unrescaled one of its two
axes would span a fraction of that square and the other all of it, so the
vertical and horizontal parts of the same distance would not be in the same
units. This is not cosmetic: without the second rescale the minimum is 0.3446
rather than 0.3596 and does not fall at the same cut. It is the step most likely
to be dropped by someone reimplementing this from the formulas.

**Admissible range of $w$.** Nothing in the data fixes $w$, so the honest thing
is to bound it rather than fit it: $w \geq \frac12 \iff w \geq 1-w$. Below ½ the
term built from case/control labels outweighs the one built from genotypes, and
minimising that is optimising the very thing the association test is meant to
measure. ½ is where that stops being true, not a tuned value. The answer holds on
$w \in [0.37, 0.71]$, so ½ sits inside a plateau; the plateau supports the claim
that the cut does not turn on the weight, and does not make ½ optimal.

**Three cohorts rather than one.** A single cohort would need one worry to
dominate; three different things can, so three sets are delivered and the reason
for each is stated on `03`. They are nested, so this is a choice of where to stop
along one walk rather than between three lists.

**Role of $P_k$.** It is Hotelling's exact $F$ test on the case/control centroid
gap inside the retained set — reported everywhere, selected on nowhere. Choosing
the cut with the largest $P$ would be choosing the set that best hides a real
difference between the arms, which is the opposite of what a cohort is for. Of
the three, `intermediate` is the only one where the gap is not detectable and
`narrow` sits at the strongest separation in the walk; both are consequences of
where the cuts fell, not reasons they fell there.

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
