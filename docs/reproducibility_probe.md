# Reproducibility probe

**Date:** 2026-07-30
**Verdict:** reproducible — the selected model and every downstream deliverable are bit-identical across independent runs. A non-propagating numerical wobble exists in the BIC search log; it is characterised below over three run pairs and bounded by an explicit guard in the verification tool.

## Why this exists

The scientific results in `results/` are final. Before refactoring `scripts/` or
`workflow.ipynb` for maintainability, we had to establish that "re-run and diff"
is a valid oracle — otherwise there is no way to prove a refactor changed
nothing. This document records that check. It is also the provenance record for
the version pins in `requirements.txt`: those pins are only defensible in a
Methods section because this probe passed.

## Method

No code was modified. The committed tree was moved aside and the unmodified
notebook was executed end-to-end into a clean directory:

```bash
git status                      # clean
mv results results_baseline
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output-dir=<scratch> --output probe_run.ipynb workflow.ipynb
python -m tools.verify_results --baseline results_baseline --candidate results
```

Wall time: **6 min 53 s** (01:40:58 → 01:47:51), exit code 0, 58 artifacts
produced — the same count as the baseline. Environment as recorded in
`docs/environment_snapshot.txt` (Python 3.12.6, numpy 2.4.2, scikit-learn 1.8.0,
hdbscan 0.8.42, OpenBLAS 0.3.31, macOS arm64).

Afterwards `results_baseline` was restored as `results`, so the committed tree
is still the original one that the manuscript refers to. The probe run was
discarded once compared.

## Result

| | |
|---|---|
| Artifacts compared | 58 |
| Byte-identical | 54 |
| Differing | 3 files + 1 figure, all inside the STEP2 BIC search log |

**Bit-identical across runs**, i.e. the things that matter:

- `gmm_summary.json` — `best_k = 26`, `best_bic = -2682580.328806532` (identical to the last bit)
- `bbj_samples_gmm_clustered.tsv` — every BBJ sample's component assignment
- `hdbscan_summary.json` and the denoised sample table (183,013 → 181,817; 1,196 noise / 0.65 %)
- all of STEP3–STEP9: merge map, Mahalanobis distances, posterior `.npy` arrays,
  the rank progression tables, and **all seven `*.fid_iid.txt` keep-lists** —
  the actual deliverables consumed by `plink --keep`

**Differing:** `02_gmm_clustering/gmm_bic_search.tsv`, the two
`tmp/gmm_search_*.jsonl` audit logs, and `gmm_fixed_pcs_overview.png`
(0.001 % of pixels, max channel delta 6 — the BIC curve redrawn through the
perturbed points).

## Diagnosis of the wobble

Three independent run pairs have now been measured (the probe above, and two
verification runs of the refactored pipeline). In each, 15–18 of the 99
candidate fits report a slightly different BIC:

| run pair | candidates differing | max \|d\| | relative |
|---|---|---|---|
| probe vs committed | 15 / 99 | 1.387154 | 5.2e-07 |
| verify 1 vs committed | 16 / 99 | 1.21 | 4.5e-07 |
| verify 2 vs committed | 18 / 99 | **11.2706** (at k=58) | 4.2e-06 |

**Two distinct mechanisms**, which the first measurement alone did not reveal:

1. *Float-floor quantisation.* Most perturbations are exact multiples of
   **0.173394** BIC units. That number is not arbitrary: the convergence log
   shows `average_log_likelihood` differing as `7.382948398590088` vs
   `7.382948875427246` — 4.77e-7, which is **one float32 ULP** at that
   magnitude. Multiplied by 2 × 181,817 samples it is exactly 0.1734. The cause
   is float reduction order varying between runs: the search runs `k=2..100`
   across a 6-process pool over OpenBLAS-backed operations, and neither the
   process-to-k assignment nor the BLAS thread count is pinned.
2. *Restart selection flipping.* Occasionally a single candidate jumps by an
   order of magnitude more — 11.27 units at k=58 in the third run pair. This is
   not ULP noise: with `n_init=3`, EM is restarted three times and the best fit
   kept, so a small float difference can flip **which restart wins**, landing
   the candidate in a different local optimum.

The first version of this document reported a maximum of 1.39 units based on a
single comparison. That understated the effect; 11.27 is the measured maximum
over three pairs, and the second mechanism means the bound is statistical rather
than structural.

## Why it does not propagate

- The margin between the winner (k=26) and the runner-up (k=29) is **155.92 BIC
  units** — still a **14× margin** over the largest wobble observed.
- The most-perturbed candidate, k=58, sits **1,759 BIC units** away from the
  winner. It is not competing for the selection under any plausible noise.
- Empirically, across all three run pairs: `best_k = 26` and
  `best_bic = -2682580.328806532` are identical to the last bit,
  `bbj_samples_gmm_clustered.tsv` is byte-identical (so the fitted model and
  every sample's component assignment are identical), and **all 57 downstream
  artifacts including every keep-list are byte-identical**.

The wobble is confined to the diagnostic log of the search itself.

## How this is encoded in the verification tool

`tools/verify_results.py` does not paper over this with an arbitrary relative
tolerance. For the BIC search table it checks what actually carries the science:

1. `argmin(BIC)` — which k is selected — must not move;
2. the winning BIC value must be identical;
3. no candidate may drift by more than **10 % of the winner-to-runner-up
   margin**, which the tool derives from the baseline at comparison time. At the
   observed 155.92-unit margin that admits 15.59 units — above the 11.27 seen in
   practice, and far below anything that could reorder the top of the ranking.

The two JSONL audit logs carry the same quantities under a matching 1e-5
relative tolerance. All three guards were tested in both directions: a change
letting k=29 overtake k=26 fails, a 20-unit drift at an unrelated k fails, and a
10-unit drift within budget passes. No other file in the tree is given any
tolerance.

## If exact search-log reproduction is ever needed

Set `OMP_NUM_THREADS=1` and `search_workers=1`. This costs roughly 6× wall time
and changes nothing about the selected model, so it is not enabled by default.

## Implications

1. Refactoring may proceed with re-run-and-diff as the acceptance test. It has:
   the config/caching/notebook refactor and the deduplication of `scripts/` were
   both accepted on this basis, each reproducing all 57 pre-existing artifacts.
2. The version pins in `requirements.txt` are backed by evidence and can be
   cited in Methods.
