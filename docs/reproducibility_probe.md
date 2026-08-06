# Reproducibility

**Verdict:** the pipeline is fully deterministic. Two independent full runs in
the same environment agree on every numeric artifact and every figure; the only
differences are embedded timestamps and the results-root path a run was written
to.

This was not true of the original float32 pipeline. The measurement that
established the difference, and the reason for the switch, are recorded below —
they are the justification for the dtype change and for the numbers in the
manuscript having moved.

---

## Current state (float64)

Two independent full runs, `RUN_MODE="fresh"`, default multi-threaded BLAS:

| | |
|---|---|
| Artifacts compared | 139 |
| Verification result | **139 pass, 0 warn, 0 fail — with no numeric tolerance of any kind** |
| Figures byte-identical | **28 / 28** |

The counts above are from the current layout. The probe was originally run
against an 86-artifact tree; the pipeline has since gained the per-basis
PC-space analyses, which multiplied the diagnostic outputs. The conclusion is
unchanged and has been re-confirmed on the current tree.

The 9 artifacts that are not raw byte-identical differ *only* in:

- embedded UTC timestamps — the four `01_reference_model/mixture_model/tmp/` audit files
  (`timestamp_utc`, `generated_at_utc`, `elapsed_seconds`, and the
  `- Generated (UTC):` line of the report);
- the results-root path they record — the `*_all_pcs_kde.log` files echo
  their own `output_dir`, and `provenance/run_config_snapshot.json` stores it per stage.

No numeric value differs anywhere. `tools/verify_results.py` normalises exactly
those two things and nothing else.

**Cost:** float64 makes the `k = 2..100` search about 2.3× more compute; a full
run went from ~7 min to ~13 min. Determinism does *not* require pinning BLAS
thread counts — parallelism stays on.

---

## Why the switch happened (float32)

The original pipeline cast its inputs to float32
(`gmm_clustering.py` and `hdbscan_filtering.py`). float32 carries ~7 decimal
digits, so summing 181,817 per-sample log-likelihoods surfaced the reduction
order used by multi-threaded OpenBLAS across the 6-process search pool. Fixing
`random_state=42` did not help: the seed governs which random choices the
algorithm makes, not the order in which floats are added, and floating-point
addition is not associative.

Across three measured run pairs, 15–18 of the 99 candidate fits disagreed:

| run pair | candidates differing | max \|d\| | relative |
|---|---|---|---|
| probe vs committed | 15 / 99 | 1.387154 | 5.2e-07 |
| verify 1 vs committed | 16 / 99 | 1.21 | 4.5e-07 |
| verify 2 vs committed | 18 / 99 | **11.2706** (k=58) | 4.2e-06 |

Two mechanisms, only the first of which is bounded by ULP analysis:

1. **Float-floor quantisation.** Most differences were exact multiples of
   **0.173394** BIC units — one float32 ULP of `average_log_likelihood`
   (4.77e-7 at magnitude 7.38) times 2 × 181,817 samples.
2. **Restart selection flipping.** With `n_init=3`, EM is restarted three times
   and the best kept, so a last-bit difference could flip *which restart wins*,
   producing the ~11-unit jumps.

It never propagated: `best_k`, the fitted model and all downstream artifacts
were byte-identical in every float32 run pair. But it forced the verification
tool to carry a per-file tolerance and a "drift budget" tied to the
winner-to-runner-up margin, plus a written argument for why the selection could
not move. Removing that machinery was the point of the change.

Direct isolation of the cause (5 independent processes per condition, same seed,
same data, k=29 and k=35):

| | result |
|---|---|
| float32, multi-threaded | **not reproducible** — runs disagreed (e.g. k=35 gave two different values) |
| float32, `OMP_NUM_THREADS=1` | reproducible, but ~5% slower **and it changes `best_bic`** |
| **float64, multi-threaded** | **reproducible, 5/5 bit-identical** |

---

## What the dtype change did to the results

This was a re-analysis, not a refactor.

| | float32 (former `results-float32-final` tag) | float64 (current) |
|---|---|---|
| Denoised | 181,817 / 183,013 (1,196 noise) | **181,815** (1,198 noise) |
| `best_k` | 26 | **26** (unchanged) |
| `best_bic` | −2,682,580.328806532 | **−2,682,560.0780631886** |
| Merged clusters | 6 | 6 |
| Mainland components | 17 | **16** |
| Mainland component ids | 0,2,3,4,7,8,11,12,13,14,15,16,18,20,22,24,25 | 0,2,4,5,6,8,10,11,12,13,16,18,19,20,21,22 |
| Subcluster (rank ≤ 9) | 2,193 samples | **2,138** |
| Cohort | 411 CTEPH / 1,782 AGP3K | **408 / 1,730** |
| Retained at 0.90 | 300 / 890 | **272 / 650** |
| Case-minimum threshold | 0.423998624086 | **0.357533** |

The last two rows refer to the confidence-threshold stages, which were removed
in the restructure described at the end of this document.

Component ids are renumbered wholesale, so the hand-picked rank selection refers
to entirely different components. At threshold 0.90 the retained control group is
27% smaller — a real difference in statistical power, not a relabelling.

The float32 tree was pinned by a tag `results-float32-final`, which has since
been deleted: that tree contained individual-level posterior tables with
per-sample PC coordinates, and those are no longer distributed. The numbers
quoted above are the record of that comparison.

---

## Two claims made during the investigation that were wrong

Recorded because both are the kind of thing that would otherwise be rediscovered:

1. **"float64 moves `best_k` from 26 to 25."** That came from running a float64
   GMM on the *float32-denoised* sample table — a mixed-precision configuration
   the pipeline never produces. In the coherent float64 pipeline, HDBSCAN is also
   float64, retains 181,815 samples, and `best_k` remains 26.
2. **"PNGs cannot be compared byte-for-byte because of embedded matplotlib
   metadata."** The metadata is stable within a pinned environment. Under float32
   the figures differed because the *plotted data* differed. All 28 figures are
   now byte-identical across runs, and `verify_results.py` treats any pixel
   difference as a failure.

---

## Method

```bash
git status                       # clean
rm -rf results_run_a .cache
POPGMM_RESULTS_ROOT=results_run_a jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output-dir=<scratch> --output run_a.ipynb workflow.ipynb
# repeat into results_run_b, then:
python -m tools.verify_results --baseline results_run_a --candidate results_run_b
```

Environment as recorded in `docs/environment_snapshot.txt` and
`results/provenance/run_environment.json` (Python 3.12.6, numpy 2.4.2, scikit-learn 1.8.0,
hdbscan 0.8.42 via python-hdbscan, OpenBLAS 0.3.31, macOS arm64).

## Implications

1. Re-run-and-diff is a valid acceptance test, with **no tolerance** — a
   stronger claim than the float32 pipeline could support.
2. The version pins in `environment.yml` are backed by evidence and can be
   cited in Methods.
3. `verify_results.py` keeps one zero-cost invariant beyond exact comparison:
   the k minimising BIC must not move. It costs nothing and is the single thing
   most worth failing loudly on if numeric behaviour ever changes again.

---

## Restructure around the three keep-lists

The pipeline was later reorganised so that its output layout expresses its
purpose: three deliverable sample lists in `results/keep_lists/`, with the
supporting stages numbered in dependency order. The threshold-screening stages
were removed, the two rank variants (`narrow`, `intermediate`) are produced by one
parameterised loop, and emitting the lists became one module's job rather than
four stages each writing one somewhere.

That restructure is expression, not arithmetic, and was verified as such:

- **16 upstream numeric artifacts byte-identical** to the pre-restructure tree
  (denoising table and summary, BIC search, clustered table, component summary,
  mixture summary, Mahalanobis distances, merge map, merged table, merged
  summary, posterior `.npy`, major-cluster reference, cohort posteriors,
  component ranks).
- **`full_mainland` and `narrow_mainland` are the identical row sets** as the
  lists they replace (3,099 and 2,138 samples).
- `intermediate_mainland`, `keep_list_summary.tsv` and
  `major_cluster_robustness.tsv` are new.

The keep-list summary's `GWAS_Neff` and `PC12_RGV` values agree to the last digit
with the independently computed rank-selection decision table, which
cross-checks the new module against the existing one.
