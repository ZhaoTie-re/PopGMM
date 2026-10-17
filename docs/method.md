# Method

Why each stage of PopGMM exists and what it computes. For the operational side —
install, run, use the output — see [`../README.md`](../README.md); for what every
produced file contains, see [`outputs.md`](outputs.md).

---

## Motivation

### Stratification confounding

Allele frequencies vary across populations. If cases and controls differ in
ancestry composition, a variant can appear associated with disease because it
tracks ancestry rather than biology — inflating test statistics, creating
false-positive loci, and obscuring genuine signals.

Principal-component covariates and mixed models adjust for structure *during*
association testing. PopGMM addresses a complementary question earlier in the
analysis:

> Which samples form a sufficiently coherent ancestry subset for the intended
> case/control analysis, and what is lost or gained when that subset is made
> narrower or broader?

A single hard ancestry label cannot express that uncertainty, and a manually
drawn PCA boundary is difficult to reproduce or audit.

### From density elements to population structure

PCA turns genome-wide similarity into a coordinate system but does not decide
where one population region ends and another begins. Real PCA clouds are skewed,
elongated, overlapping, or joined by gradients, so one named population may
require several Gaussian components to describe its geometry.

PopGMM therefore works at two levels:

- a **component** $k$ is a local density element — on its own it makes no
  population claim;
- an **ancestry cluster** $c$ is a coherent region obtained by merging nearby
  components, and it is at this level that the model recovers documented
  structure ([§4](#4-merging-components-into-ancestry-regions)).

### Uncertainty is information

Samples near a population boundary are not as confidently placed as samples at a
cluster center. PopGMM assigns posterior membership probabilities rather than
nearest-centroid labels, so each study sample carries a distribution over the
fitted components alongside its assignment and confidence.

### Residual structure versus power

A narrow sample set reduces residual structure but discards usable cases and
controls; a broad set preserves sample size but retains more of it. In an
unbalanced study, raw count also overstates power — adding many controls to few
cases contributes less than the total suggests.

PopGMM makes the exchange visible by reporting effective sample size against
residual structure, and emits three nested cohorts rather than hiding the choice
behind one threshold.

---

## The eight stages

### 1. A shared PCA coordinate system

PopGMM starts from PCA scores generated upstream. Study samples must be projected
onto the reference loadings so that both datasets share axes, scale, and
orientation: a study point can only be evaluated under the reference density if
its coordinates carry the same meaning. The score table is split by IID
convention into a reference panel and a study cohort; case/control status is kept
for later assessment but never used to fit the density.

Density estimation uses the leading two principal components, so each sample is a
point $x_n \in \mathbb{R}^2$. Higher PCs are read from the input and used for
post-assignment diagnostics ([§8](#8-all-pc-casecontrol-diagnostics)), never for
fitting.

### 2. Denoising the reference panel

A Gaussian mixture must explain every observation, so sparse isolated samples
attract a spurious component or inflate a covariance, and the distortion
propagates through merging and assignment. HDBSCAN removes them: dense reference
structure is retained, sparse off-manifold observations are marked as noise.

Density clustering compares distances across axes and is scale-sensitive, so it
runs on z-scaled PCs; the mixture is fitted on raw PCs, where a full covariance
absorbs any per-axis rescaling. Only the reference panel is denoised — study
samples keep their place until their probabilities have been evaluated.

### 3. Learning the reference density

The denoised panel is modelled as a mixture of $K$ components with full covariances,

```math
p(x \mid \theta) \;=\; \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k),
\qquad \theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^{K}
```

fitted by expectation–maximization. The order $K$ is selected over a candidate
range by the Bayesian Information Criterion,

```math
\hat{K} \;=\; \arg\min_{K} \mathrm{BIC}(K),
\qquad \mathrm{BIC}(K) \;=\; -2\,\ell(\hat{\theta}_K) \;+\; p_K \log N
```

which penalises the parameter count $p_K$ against the maximised log-likelihood.
Each component contributes a weight, a center, and a covariance describing the
orientation and spread of a local ancestry cloud. $\hat{K}$ is a
density-modelling choice, not an estimate of how many human populations exist.

> The EM iteration behind the fit — initialization, E-step, M-step,
> regularization, convergence test — is derived in
> [`gmm_convergence_diagram.EN.md`](gmm_convergence_diagram.EN.md).

### 4. Merging components into ancestry regions

Components are compared by the Mahalanobis distance between their means under
their pooled covariance,

```math
S_{ij} \;=\; \tfrac{1}{2}\left(\Sigma_i + \Sigma_j\right),
\qquad d_{ij} \;=\; \sqrt{(\mu_i - \mu_j)^{\top} S_{ij}^{-1} (\mu_i - \mu_j)}
```

which — unlike Euclidean distance — accounts for scale, elongation, and
orientation. The matrix $D = [d_{ij}]$ is clustered by hierarchical linkage and
cut at a threshold, giving a map $k \mapsto c(k)$ from components to ancestry
clusters. Posterior mass follows the map,

```math
r_{nc} \;=\; \sum_{k \,:\, c(k) = c} r_{nk}
```

so probability is preserved as the description moves from local density elements
to broader regions. The **major cluster** follows an explicit rule: the merged
group containing the most components.

Merging is also what makes the model checkable. On a BioBank Japan panel the
merged clusters line up with independently published population structure, and a
tighter cut subdivides the same regions rather than rearranging them — so the
components do carry ancestry information, even though no single component is a
population.

![Component merging at distance thresholds 6 and 3](figures/merged_clusters.png)

*Merged clusters at distance thresholds 6 (left) and 3 (right). Top: the
average-linkage dendrogram of the 27 components; the dashed line is the cut.
Bottom: the merged clusters on PC1–PC2. The published structure they are compared
with is Fig. 1b–c of Yamamoto et al. (2024),* Genetic legacy of ancient
hunter-gatherer Jomon in Japanese populations, *Nature Communications 15, 9780
([doi:10.1038/s41467-024-54052-0](https://doi.org/10.1038/s41467-024-54052-0)).
That article is licensed CC BY-NC-ND 4.0, so its panels are not reproduced here.*

### 5. Projecting the study cohort

The cohort is evaluated under the fitted model without refitting it. Each study
point $x_n$ receives responsibilities over the original components,

```math
r_{nk} \;=\; \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}
{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
```

evaluated at the fixed estimate $\hat{\theta}$, together with an assignment
$\arg\max_k r_{nk}$ and a confidence $\max_k r_{nk}$.

The projection is one-way by design: cohort composition, disease status, and
case/control imbalance cannot move the learned boundaries. It also keeps
ambiguous samples visible — a low-confidence assignment stays available for
review instead of disappearing behind a hard label.

### 6. Three quantities along one walk

The major cluster can still contain internal structure. Its components are ranked
by observed case/control ratio, and cut $k$ keeps the top $k$ of them — one nested
set per $k$. That ordered traverse is **the walk**, and every cut in §6b is a
place to stop on it.

Each set along the walk is scored on **three** quantities. Two are defined here;
the third, the case/control shift, is defined immediately below.

```math
N_{\mathrm{eff},k} \;=\; \frac{4}{n_{\mathrm{case}}^{-1} + n_{\mathrm{ctrl}}^{-1}},
\qquad
H_k \;=\; \left(\det \Sigma_k\right)^{1/2d}
```

$N_{\mathrm{eff},k}$, the **effective sample size**, is the harmonic form that
association power actually scales with, so an unbalanced set is penalised against
its raw total. $H_k$, the **residual structure** of the retained set, is the root
generalized variance over $d$ axes: the geometric mean of the per-axis standard
deviations, which accounts for correlation between the axes where a per-axis
variance would not. Both rise strictly with $k$.

RGV is reported in two bases, never compared with each other:

| Column | Basis | Axes |
|---|---|---|
| `RGV_Global` | the global PCA | PC1–PC2, always |
| `RGV_Mainland` | a PCA fitted to the major cluster | `params.MAINLAND_RGV_N_PCS` |

The global PCA's leading axes are what separate the major cluster from the other
regions, so *within* it they are nearly constant and resolve residual
heterogeneity poorly — exactly the quantity the cut trades against. A PCA fitted
to the major cluster spends its axes on the structure that actually remains.
`params.RGV_BASIS` selects which column $H_k$ is read from, and so which column
both criteria in §6b are computed against.

The walk does not name a correct subset on its own; it prices the exchange — how
much effective sample size is gained as the region broadens, and how much
residual structure comes with it. §6b takes two criteria on that pricing, and
§"Automatic and manual selection" says how each is resolved and recorded.

#### The case/control shift

$H_k$ says how *wide* the retained set is. It does not say whether cases and
controls are drawn from the same place within it — and only the second biases an
association test. So each set along the walk also carries the Mahalanobis
distance between the two arm centroids under the pooled within-group covariance,
with Hotelling's $T^2$ and its exact $F$ $p$-value:

```math
D^2 = (\bar{x}_{\mathrm{case}} - \bar{x}_{\mathrm{ctrl}})^{\top}
      S_{\mathrm{pooled}}^{-1}
      (\bar{x}_{\mathrm{case}} - \bar{x}_{\mathrm{ctrl}})
\qquad
T^2 = D^2\,\frac{n_1 n_2}{n_1 + n_2}
```

Reported in both bases, matching the RGV columns: `PC12_CaseCtrl_*` on the
global PC1–PC2, and `Mainland_CaseCtrl_*` over the same
`params.MAINLAND_RGV_N_PCS` mainland axes the trade-off is judged on.

A raw $\hat{D}^2$ cannot be compared across the walk. Two sample means never
coincide, and $D^2$ squares the gap between them, so sampling scatter can only
add to it:

```math
E\!\left[\hat{D}^2_k\right] \;=\; D^2_k \;+\; b_k,
\qquad
b_k \;=\; d\left(\frac{1}{n_{\mathrm{case}}} + \frac{1}{n_{\mathrm{ctrl}}}\right)
```

The retained set grows more than tenfold across the walk, so that floor $b_k$
falls from 0.070 to 0.011 — a factor of 6.6 — and a raw $\hat{D}^2_k$ drifts
downwards for arithmetic reasons alone. Subtracting it gives the third quantity,
the **case/control shift**:

```math
s_k \;=\; \hat{D}^2_k - b_k
```

`Mainland_CaseCtrl_D2_Unbiased` is the column it is read from, and it is the only
one of the three separation columns that compares fairly between cuts; it goes
negative where the shift sits below the floor. $T^2$ and its $p$-value need no
such correction — the exact $F$ test already accounts for sampling.

**$P_k$ is reported everywhere and selected on nowhere.** It is computed from the
case/control labels, so choosing the cut with the largest $P_k$ would be choosing
the set that best hides a real difference between the arms, and a genuinely
ancestry-linked risk would be selected away along with the confounding.

$s_k$ is also non-monotone in $k$, unlike the other two quantities: Spearman
$-0.78$ in $k$ for the raw `Mainland_CaseCtrl_Mahalanobis` and $-0.73$ for the
de-biased `Mainland_CaseCtrl_D2_Unbiased`, against $+1.00$ for both
$N_{\mathrm{eff},k}$ and $H_k$. It reverses direction seven times along the walk
and bottoms out near the *uncut* set, where $N_{\mathrm{eff},k}$ is largest — so
it has no exchange rate to read off and cannot be priced the way the other two
are.

$s_k$ is nonetheless an *input* to one of the two criteria, as long as it is not
the quantity that criterion minimises on its own. That distinction is what the
next section formalises.

### 6b. The two criteria, and the three cuts

Two criteria cut the walk, and `full` is not one of them. The two are **one
procedure at two settings of a single question**: what counts as residual
structure. Each minimises that against $N_{\mathrm{eff}}$ over the same
normalised axes —

```math
\tilde{H}_k = \frac{H_k - \min_j H_j}{\max_j H_j - \min_j H_j},
\qquad
\tilde{N}_k = \frac{N_{\mathrm{eff},k} - \min_j N_{\mathrm{eff},j}}
                   {\max_j N_{\mathrm{eff},j} - \min_j N_{\mathrm{eff},j}}
```

— where a tilde always means min-max onto $[0,1]$ over the 17 cuts. The two
differ only in what enters the structure axis. Widening what is counted moves the
cut outwards, because $s_k$ falls with $k$ while $H_k$ rises.

| Cut | Structure counted | Frontier | Operator | $k$ | Weighs |
|---|---|---|---|---|---|
| `full` | nothing | — | `whole_cluster` | 17 | none of the three |
| `narrow` | residual structure | monotone, chord spans 1.000 | `peak_excess_return` | 9 | $N_{\mathrm{eff},k}$, $H_k$ |
| `intermediate` | residual structure **and** the shift | folded, chord spans 0.238 | `nearest_ideal` | 12 | all three |

The `Operator`, `Frontier_Monotone` and `Chord_Span` values transcribe
`cut_record.tsv`. That table's own `Structure_Counted` column still spells the
axis `residual spread` and the shift `case/control separation`; those two string
literals predate the rename and are left as they are so the committed record
stays byte-identical.

**The operator is not interchangeable.** Each follows from the geometry its
setting produces, and the stage records that geometry beside the answer — hence
`Frontier_Monotone` and `Chord_Span` — so the pairing can be checked against the
table rather than taken on trust. The pairing itself is written into
`resolve_rank_cuts`; it is not inferred at run time. Swapping the two gives the
wrong answer in both directions:

| | monotone frontier | folded axis |
|---|---|---|
| peak excess return | **9** | 2 — degenerate |
| nearest ideal | 7 | **12** |

**`full` — nothing counted.** With nothing counted there is no trade to make, and
so nothing to optimise. `full` is the major cluster itself — every one of its 17
components, which is every mainland sample — and it is the population the other
two are cut from rather than the answer to a rule. `cut_record.tsv` records it as
`whole_cluster` / `definitional`, and its card on `02_cohorts.png` carries three
hollow marks, one per quantity it does not weigh.

**`narrow` — priced against the walk's average exchange rate.** Both quantities
rise strictly with $k$, so the walk end to end has one average rate — how much
$N_{\mathrm{eff}}$ a unit of residual structure buys:

```math
\gamma = \frac{N_{\mathrm{eff},K} - N_{\mathrm{eff},1}}{H_K - H_1} = 381{,}519,
\qquad
k_{\mathrm{narrow}} = \arg\max_k\ E_k,
\qquad
E_k = (N_{\mathrm{eff},k} - N_{\mathrm{eff},1}) - \gamma\,(H_k - H_1)
```

$E_k$ is the **excess** $N_{\mathrm{eff}}$ cut $k$ has bought over that rate. It
peaks at $k = 9$ at $+525.1$; beyond it the curve flattens and later components
never make the accumulated shortfall back. The case/control shift plays no part
in this criterion — `_excess_return(ranks, neff, het)` is not passed it.

Cumulative, not per-step, because the per-step rate is *not* monotone: ranks
10–12 fall below the average and rank 13 springs back to 469,219. "The last step
above the average rate" would answer **13**. The cumulative form answers 9, and
is exactly the knee of the curve — $E_k$ is the vertical distance from cut $k$ to
the chord joining the walk's ends, so maximising it *is* taking the point
furthest from that chord, expressed in $N_{\mathrm{eff}}$, a unit the reader can
price.

The margin belongs beside the answer: the peak leads its runner-up by 3.1
$N_{\mathrm{eff}}$ out of 525.1, so the turn is real but not sharply placed.

**`intermediate` — residual structure and the shift counted.** Adding $s_k$ folds
the structure axis, because the two disagree about direction: Spearman $+1.00$ in
$k$ for $H_k$ against $-0.73$ for `Mainland_CaseCtrl_D2_Unbiased`.

```math
\tilde{u}_k(w) = \mathrm{minmax}\!\left(w\,\tilde{H}_k + (1-w)\,\tilde{s}_k\right),
\qquad
k_{\mathrm{inter}} = \arg\min_k \sqrt{\tilde{u}_k(w)^2 + \left(1 - \tilde{N}_k\right)^2},
\qquad w = \tfrac{1}{2}
```

The outer min-max is not decoration. A blend of two $[0,1]$ axes does not itself
span $[0,1]$, and rescaling puts it back on $\tilde{N}_k$'s scale before the
distance is taken. It is what the recorded number is: `cut_record.tsv` gives the
winning distance as 0.3596, against 0.3446 without it. On this walk both pick the
same cut, so the rescale fixes the score rather than the answer — but a
reimplementation that drops it will not reproduce the record.

The blended axis travels 0.238 between its ends against the structure axis's
1.000, leaving the chord near vertical. There is no turn in the exchange rate to
find, so peak excess return is inadmissible and proximity to the ideal corner is
what remains defined. This gives $k = 12$.

$w$ encodes which kind of residual structure is judged to matter; no data can
supply it. It is fixed at equal weight — the one value that needs no argument
for preferring either measure — and then *swept*, which is what shows the answer
is not an artefact of the value: $k = 12$ wins across $w \in [0.372, 0.715]$, so
the evaluation point sits 0.128 from the nearest boundary. Three limits belong
with that:

- **The sweep and the rule score slightly differently.** `_weight_sweep` blends
  the two normalised axes and measures the distance directly; the resolver
  rescales the blend first, as the equation above does. Both put $k = 12$ at
  $w = \tfrac{1}{2}$, but the plateau they report is not identical — $[0.372,
  0.715]$ swept against $[0.364, 0.666]$ under the rule's own score. The interval
  quoted here and on `01_tradeoff.png` is the swept one.
- **$w < 0.5$ is not usable.** It weights the case/control labels above residual
  structure, which is the failure mode above. The cuts that win there
  ($k = 15, 16$) are artefacts of optimising the association signal.
- **$w$ is not an absolute scale.** Min-max normalisation is anchored on the
  observed extremes, so the interval shifts if the walk is truncated —
  $[0.45, 0.78]$ without $k=1$, $[0.31, 0.66]$ without $k=17$. The winner at
  $w = \tfrac{1}{2}$ is $k = 12$ in every case tested; only the edges move.

This is also why `narrow` counts one measure and not two. One measure needs no
weight; a second introduces a parameter with no principled value, which is why
the blend fixes the intermediate cut and nothing else.

#### Automatic and manual selection

`params.RANK_CUT_MODE` selects which answer is *used*:

| Mode | Effect |
|---|---|
| `"auto"` (default) | each cut comes from its rule above |
| `"manual"` | each comes from `params.MANUAL_RANK_CUTS` |

**Both are always computed, whichever mode is set.** The mode never changes
which quantities are evaluated, only which column is consumed, so switching it
cannot move a cut without `cut_record.tsv` recording that it moved — the
table carries `Resolved_Rank`, `Auto_Rank`, `Manual_Rank` and
`Auto_Manual_Agree` side by side. On the committed run the two agree for both
cuts, so the mode is currently inert.

A single variant can also override the mode *upwards*: writing an integer in
`params.SUBCLUSTER_VARIANTS` pins that cut whatever the mode is, which makes "pin
narrow, leave intermediate on the rule" expressible without a global switch.
Writing `"auto"` there does **not** do the reverse — under
`RANK_CUT_MODE = "manual"` a variant left on `"auto"` still takes its
`MANUAL_RANK_CUTS` value, and `cut_record.tsv` records `Source = manual`
(`resolve_rank_cuts`, `scripts/rank_selection.py`).

The resolution happens in the rank-selection stage rather than in the subcluster
stage, because that is where the evidence is: a cut derived beside the table
that justifies it cannot drift from that table. The subcluster stage consumes
`rank_out.rank_cuts` and derives nothing of its own.

What the diagnostic is good for beyond that is auditing a cut after the fact.
Two things it surfaces on the committed run, neither visible in RGV:

- The `full` set is indistinguishable from homogeneous on the global PC1–PC2
  ($p = 0.39$) but separated on the four mainland axes ($p = 0.0035$). The
  structure is in mainland PC3–PC4, which the global diagnostic cannot see.
- The `narrow` cut has the smallest $P_k$ of all seventeen
  ($P = 3.9 \times 10^{-8}$), and the second largest $s_k$ after $k = 1$. It is
  the peak of $E_k$ on the $N_{\mathrm{eff}}$-versus-$H_k$ walk that selects it,
  and the shift plays no part in that criterion; the separation is a property to
  be aware of, not a reason to move the cut.

### 7. Composite-posterior reassignment

A selected component set $\mathcal{G}$ is treated as one composite group.
Responsibilities are computed over the **original** components first, then summed
across $\mathcal{G}$, with every unselected component still competing:

```math
\tilde{r}_{n\mathcal{G}} \;=\; \sum_{k \in \mathcal{G}} r_{nk}
\qquad\text{and}\qquad
\hat{g}_n = \mathcal{G}
\;\;\Longleftrightarrow\;\;
\tilde{r}_{n\mathcal{G}} \;>\; \max_{k \notin \mathcal{G}} r_{nk}
```

> With $r_n = (0.30, 0.30, 0.40)$ over components $(A, B, C)$ and
> $\mathcal{G} = \lbrace A, B \rbrace$, the sample joins $\mathcal{G}$ at $0.60$ — not $C$.

The order matters. Merging first and projecting into the coarsened mixture would
discard exactly the joint evidence that places a borderline sample in the broader
region, so membership is a property of the group rather than of any single
component.

### 8. All-PC case/control diagnostics

After assignment, case and control distributions are compared across all
available PCs, not only the two used for fitting. These checks surface residual
structure invisible in the fitting view; they are reported for quality control
and never redefine the model.

Each PC-space diagnostic is produced in **both bases** of §6 and written to a
directory named for the basis. The global PCA's leading axes separate the major
cluster from the other regions, so within it they are close to constant and
resolve its residual structure poorly; a PCA fitted to the major cluster spends
its axes on the structure that remains. The two are never compared with each
other — they are different coordinate systems, and each figure names the one it
is drawn in.

A basis is a single coordinate source: study samples and reference cloud are
resolved through the same lookup. Both projections name their columns
identically, so a frame cannot be asked which basis it belongs to, and mixing
one basis with another would render without error.

---

## Interpretation and limitations

- PopGMM describes structure *within a supplied reference PCA space*. It does not
  establish ethnicity, identity, or ancestry independently of that reference.
- The major cluster is defined algorithmically. Its display name is an analyst's
  interpretation, requiring validation against the reference-panel design and
  prior population knowledge — not a model-discovered label.
- Responsibilities quantify uncertainty *under the fitted mixture*, not
  uncertainty in genotype processing, panel construction, or PCA projection.
- The narrow and intermediate cuts are analysis choices on the walk that trades
  effective sample size against residual structure, not biological boundaries.
  Neither is the "better" list: a narrower set buys less residual structure with
  effective sample size, and a broader one the reverse.
- A more homogeneous set removes one source of stratification, not all residual
  structure. Association-model covariates, relatedness handling, and sensitivity
  analyses may still be required.
- Case/control labels are excluded from model fitting but used afterwards to
  assess balance and effective sample size: ancestry learning is unsupervised,
  the final subset design is study-aware.

---

## Implementation

The notebook orchestrates; these modules implement the stages.

| Stage | Module |
|---|---|
| — | `scripts/data_loading.py` — input loading, cohort separation |
| **2** | `scripts/hdbscan_filtering.py` — reference denoising |
| **3** | `scripts/gmm_clustering.py` — mixture fitting and order selection |
| **4** | `scripts/gmm_component_merging.py` — merging and robustness |
| **5** | `scripts/cohort_assignment.py` — posterior assignment |
| **6** | `scripts/rank_selection.py` — $N_{\mathrm{eff}}$ vs RGV assessment |
| **7** | `scripts/subcluster_assignment.py` — composite-group reassignment |
| **8** | `scripts/subcluster_view.py`, `scripts/major_cluster_all_pcs_kde.py`, `scripts/subcluster_all_pcs_kde.py` — per-basis views and all-PC diagnostics |
| — | `scripts/keep_lists.py` — PLINK keep-list generation |

Each module computes; none of them draws. Figure code lives in
`scripts/plotting/`, one module per figure plus `style.py` (the cohort palette
and the per-figure themes) and `panels.py` (helpers shared between figures).

The split is not cosmetic. A figure's style is applied through a context that
reseeds from matplotlib's defaults and reverts on exit, so a figure looks the
same regardless of what was drawn before it — previously style was applied by
mutating global state inside each plot block, and a measured 27 of 28 renders
inherited settings from whichever figure ran first. The separation also removed
a defect in which a data table was written from inside the plotting branch, so
disabling figures silently dropped it.

All parameters — paths, cohort labels, the merge threshold, the rank cuts, the
RGV basis — are centralised in [`../scripts/params.py`](../scripts/params.py).
Two things are deliberately absent from it, because they are properties of the
fitted model rather than choices: which components form the major cluster, and
how many of them there are.
