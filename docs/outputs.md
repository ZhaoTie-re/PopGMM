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
answer one question each.

**Each figure is in two bands, and the lower one is meant to be cropped away.**
Above a labelled rule: the governing equations — **numbered `(1)`, `(2)`, …
per figure** so captions and prose can cite them — the answer they reach, the
evidence panels in a row, and one caption per panel. Everything needed for the
figure to stand on its own in a talk. Below it, under the heading *Methodological
notes*, the basis for each of those choices.

Each thing is said in exactly one place. Symbols are defined once, in the line
under the equations; what an equation computes is said once, by the equation;
why it is that equation and not another is said once, below the crop line. A
symbol table used to sit at the foot repeating all three, and the figures said
"the corner (0,1)" twelve times between them.

**The rule is at the same place on all four**, so one crop setting serves the
set — at the 600 dpi these are written, **cut at row 5886** (±1 px from
`bbox_inches="tight"`), i.e. 10.34 in from the top. The cropped band is roughly
1.9:1. `scripts.plotting.rank._crop_offset(dpi)` returns that offset for any dpi.

Below the rule the justifications are packed into columns whose widths follow
their content, so the columns end at about the same depth rather than the band
being sized to the longest and the rest left blank. Topics stay whole and stay in
order: a column may hold two of them, but none is broken across a gutter.

The detail is deliberately uneven. Textbook steps — equating two variances for
$N_{eff}$, the pooled $S$ and Mahalanobis $\hat D^2$, Hotelling's $T^2$, min-max
scaling — get the equation and a clause. The steps a reviewer would question get
a column each: why $d = 4$ mainland axes rather than global PC1–PC2, why (2)
rescales twice, why $w = \tfrac12$ is a floor rather than a fitted value, and
why $P_k$ is reported and never selected on.

Notation follows the slide deck these figures sit in: `p` for allele frequency,
`N_case` / `N_ctrl` / `N_tot` / `N_eff` for the counts, `H` for residual spread.
Two symbols the deck already spends are kept free — the walk's exchange rate is
`γ`, not `r`, and the separation p-value is `P`, not `p`. The cut index is a
subscript throughout (`N_eff,k`, `H_k`, `E_k`, `s_k`, `P_k`), because every one
of these figures is a walk over `k`.

Each figure carries one line of symbol definitions **above** the crop line,
scoped to exactly the symbols its own formulas use, so a cropped figure is
self-contained; the standalone explanations stay in the band below.

Every equation is presented the same way — a clause setting it up, the equation,
then a gloss reading the result off it. Headings are noun phrases, not questions:
*Basis and dimension of H*, *The second rescaling in (2)*, *Admissible range of
w*, *Role of P_k*. No figure declares a canvas size: each is measured
from its own content and made exactly that tall, and the presented band is set
at a size meant to be read across a room rather than at a desk.

| File | Contents |
|---|---|
| `00_problem.png` | What has to be decided, and the two quantities that decide it — $N_k$ and $H_k$ |
| `01_narrow.png` | The first cut: the walk with its average rate drawn as a chord (A), and $E_k$ — the gap between them — peaking at $k$ = 9 (B) |
| `02_intermediate.png` | The obstacle (A), the $(\tilde H_k, y_k)$ plane that (3) minimises over (B), and the weight sweep (C); four justification columns below the crop rule |
| `03_cohorts.png` | Where the three sit on the trade-off (A), why you would pick each one (B), and what each delivers (C) |
| `component_ranking.tsv` | Major-cluster components ordered by case/control ratio — the order the walk follows |
| `cut_record.tsv` | How each cut was arrived at, and whether the automatic and manual answers agree |
| `rank_decision_table.tsv` | Every number the figures draw |

The argument runs across the four in order:

1. The major cluster has $K = 17$ components. Order them by case/control ratio;
   cut $k$ keeps the top $k$. Along that walk
   $N_{eff} = 4N_{case}N_{ctrl}/(N_{case}+N_{ctrl}) = N_{tot}\cdot 4r/(1+r)^2$ rises — the allele frequency cancels, so an
   unbalanced set is worth less than its raw total — and so does
   $H_k = |\Sigma_k|^{1/2d} = (\prod\sqrt{\lambda_i})^{1/d}$ with $d = 4$.
2. Because both rise, the walk has one average exchange rate
   $\gamma = (N_{eff,K}-N_{eff,1})/(H_K-H_1) = 381{,}519$, and each cut can be
   scored by $E_k = (N_{eff,k}-N_{eff,1}) - \gamma(H_k-H_1)$ — its surplus in
   effective samples over paying that price. **`narrow` $= \arg\max_k E_k = 9$.**
   Panel A draws the walk and that rate as a straight chord across it, so $E_k$
   is visible as the vertical gap between the two rather than asserted.
3. Case/control centroid distance
   $s_k = \hat{D}^2_k - d(1/N_{case}+1/N_{ctrl})$ reverses direction **7 times**, so it has
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
the gap is not detectable** ($P = 0.15$, against $3.9\times10^{-8}$ for `narrow`
and $0.0035$ for `full`); `narrow` sits at the strongest separation in the whole
walk. Neither fact selected a cut — both are properties of the deliverable,
reported in `03_cohorts.png`.

One thing to reconcile in the deck rather than here: its trade-off slide computes
`H` as `RGV_Global` on PC1–PC2 (`d = 2`), while the cut is selected on
`RGV_Mainland` with `mainland_rgv_n_pcs = 4`. Its k = 9 counts are 411 / 1,782 /
2,193 against the 419 / 1,776 / 2,195 here, so that slide predates this run. The
formula is the same either way — `det(Σ)^(1/2d)` — only `d` differs.

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
