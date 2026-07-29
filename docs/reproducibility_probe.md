# Reproducibility probe

**Date:** 2026-07-30
**Verdict:** reproducible — the selected model and every downstream deliverable are bit-identical across independent runs. A bounded, non-propagating float-precision wobble exists in the BIC search log and is documented below.

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

15 of the 99 candidate fits report a slightly different BIC:

| k | baseline BIC | re-run BIC | \|d\| | relative |
|---|---|---|---|---|
| 92 | −2678534.919556 | −2678536.306709 | 1.387154 | 5.2e-07 |
| 58 | −2680821.022340 | −2680821.542523 | 0.520183 | 1.9e-07 |
| 41, 45, 84, 85, 88, 94, 95 | | | 0.346788 | 1.3e-07 |
| 35, 51, 66, 86, 89, 93 | | | 0.173394 | 6.5e-08 |

The perturbations are quantised as exact multiples of **0.173394**. That number
is not arbitrary: the convergence log shows `average_log_likelihood` differing
as `7.382948398590088` vs `7.382948875427246` — a difference of 4.77e-7, which
is **one float32 ULP** at that magnitude. Multiplied by 2 × 181,817 samples this
is exactly 0.1734. So the entire effect is the last representable bit of the
per-sample log-likelihood, amplified by the sample count.

The cause is float reduction order varying between runs — the BIC search runs
`k=2..100` across a 6-process pool over OpenBLAS-backed matrix operations, and
neither the process-to-k assignment nor the BLAS thread count is pinned.

## Why it does not propagate

- The largest perturbation is **1.39 BIC units**.
- The margin between the winner (k=26) and the runner-up (k=29) is **155.92 BIC
  units** — a **112× safety margin**.
- The best of the 15 perturbed candidates is still **420 BIC units** worse than
  the winner, so none of them is anywhere near competing.

Consequently `best_k` never moves, the fitted model is identical, and all 54
downstream artifacts come out byte-for-byte the same. The wobble is visible only
in the diagnostic log of the search itself.

## How this is encoded in the verification tool

`tools/verify_results.py` treats the three affected files under a documented
relative tolerance of 1e-6 (`NOISE_TOLERANT_FILES`) rather than pretending they
match. Independently of that tolerance it asserts the invariant that actually
carries the science (`ARGMIN_INVARIANT`): **the k minimising BIC must not
change**. Both directions were tested — injecting a 5.8e-5 relative change that
lets k=29 overtake k=26 is reported as a hard failure, while the observed
float-floor noise passes. No other file in the tree is given any tolerance.

## Implications

1. Refactoring may proceed with re-run-and-diff as the acceptance test.
2. The version pins in `requirements.txt` are backed by evidence and can be
   cited in Methods.
3. If exact reproduction of the *search log* is ever required, pin the BLAS
   thread count (`OMP_NUM_THREADS=1`) and set `search_workers=1`. This costs
   roughly 6× wall time and changes nothing about the selected model, so it is
   not enabled by default.
