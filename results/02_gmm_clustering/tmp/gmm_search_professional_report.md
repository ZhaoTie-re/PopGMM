# GMM One-k Convergence, BIC, and LR Audit Report

- Generated (UTC): 2026-04-08T06:06:04.025602+00:00
- Focus component count ($k_{\mathrm{best}}$): 21
- Search range ($k$): 2 to 30
- Search convergence: 29/29 fits converged
- Non-empty clusters during search: 29/29

## 1. Key Findings
- $k_{\mathrm{best}}=21$ is selected by minimum BIC on search data.
- Search fit at $k_{\mathrm{best}}$ converged: True in 5 EM iterations.
- BIC consistency check: abs(sklearn - manual) = 0.0 (should be near 0).
- Final refit on full data converged: True in 5 iterations.
- LR evidence around $k_{\mathrm{best}}$ is provided as a diagnostic; classical chi-square LRT assumptions are not strictly valid for GMM component-count testing.

### 1.1 Decision Snapshot
| Question | Answer | Evidence |
|---|---|---|
| Best component count? | $k_{\mathrm{best}}=21$ | Section 6.2 |
| Did EM converge at $k_{\mathrm{best}}$? | Yes, in 5 iterations | Section 5.6 |
| Is BIC computation self-consistent? | Yes, `bic_abs_diff = 0.0` | Section 6.3 |
| Is LR used as formal hypothesis test? | No, diagnostics only | Section 7.0-7.1 |

## 2. Output Files and Usage
Use files in this order for fastest understanding:
- `gmm_search_professional_report.md`: human-readable full report (this file).
- `gmm_search_convergence_log.jsonl`: per-k convergence variables and stop reasons.
- `gmm_search_bic_log.jsonl`: per-k BIC/AIC calculation variables.
- `gmm_best_k_components_log.json`: detailed component descriptors under $k_{\mathrm{best}}$.

## 3. How to Read This Report
This report is organized from decision to evidence: first the selected model conclusion, then convergence proof, then BIC/LR quantitative support, and finally raw audit-field mappings for reproducibility.
- If you only need the final conclusion, read Sections 1, 6.2, and 7.1.
- If you need methodological validity, read Sections 4, 5, and 6 in order.
- If you need machine-log mapping, read Section 8.
- If you need component-level interpretation, read Section 9.

## 4. Data Flow
- Input table: `bbj_samples_filtered`.
- Feature selection: `selected_pc_cols` gives $X_{\mathrm{full}}$.
- Search matrix: $X_{\mathrm{search}}$ (same features, optional row subsampling).
- For each fixed $k$: fit one GMM on $X_{\mathrm{search}}$ and emit one `search_iteration` event.
- If selected as $k_{\mathrm{best}}$: refit on $X_{\mathrm{full}}$ and emit `final_model` event.

## 5. EM Convergence Evidence (at $k_{\mathrm{best}}$)

### 5.1 Core Symbols and Notation
All mathematical notation used in this section:

- $n$: sample size used by a fit.
- $d$: number of selected PC features.
- $k$: number of Gaussian components.
- $x_i \in \mathbb{R}^d$: sample $i$.
- $\theta=\{\pi_c,\mu_c,\Sigma_c\}_{c=1}^{k}$: complete parameter set of the GMM.
- $\pi_c$: mixture weight of component $c$, with $\sum_{c=1}^{k}\pi_c=1$.
- $\mu_c \in \mathbb{R}^d$: mean vector of component $c$.
- $\Sigma_c \in \mathbb{R}^{d\times d}$: covariance matrix of component $c$.
- $\mathcal{N}(x\mid\mu_c,\Sigma_c)$: Gaussian probability density with mean $\mu_c$ and covariance $\Sigma_c$.
- $\ell$: total log-likelihood, where $\ell=\log L = \log p(X\mid\theta)$.
- $p$: number of free parameters (varies by covariance_type).
- $r_{ic}=p(z_i=c\mid x_i,\theta)$: posterior responsibility (soft assignment) of sample $i$ to component $c$.
- $N_c=\sum_{i=1}^{n}r_{ic}$: effective sample size of component $c$.
- $\text{ELBO}_t$: Evidence Lower Bound at iteration $t$, used for convergence monitoring.
- $\Delta$: relative improvement in ELBO between iterations.
- $\mathrm{tol}$: convergence tolerance (default 0.001).
- $n_{\mathrm{iter,max}}$: maximum iteration count (default 200).

### 5.2 EM Algorithm Overview
EM (Expectation-Maximization) is an iterative algorithm that alternates between two steps until convergence:

1. **E-step (Expectation)**: Compute posterior responsibility of each component for each sample.
2. **M-step (Maximization)**: Update component parameters using weighted statistics.
3. **Convergence check**: Monitor a lower bound on the log-likelihood; stop when improvement stalls.

### 5.3 E-step: Compute Responsibilities
For each sample $i$ and component $c$:
$$
r_{ic}=\frac{\pi_c\,\mathcal{N}(x_i\mid\mu_c,\Sigma_c)}{\sum_{j=1}^{k}\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}
$$
where $r_{ic}=p(z_i=c\mid x_i,\theta)$ is the posterior probability (responsibility) and $\sum_{c=1}^{k}r_{ic}=1$.

### 5.4 M-step: Update Parameters
Using responsibilities, update mixture weights, means, and covariances:
$$
N_c=\sum_{i=1}^{n}r_{ic},\quad \pi_c=\frac{N_c}{n},\quad \mu_c=\frac{1}{N_c}\sum_{i=1}^{n}r_{ic}x_i
$$
$$
\Sigma_c=\frac{1}{N_c}\sum_{i=1}^{n}r_{ic}(x_i-\mu_c)(x_i-\mu_c)^\top
$$
where $N_c$ is the effective sample size of component $c$.
(Covariance update varies by `covariance_type`: full, tied, diag, spherical.)

### 5.5 Convergence Check: Lower Bound
The **lower bound** (also called ELBO, Evidence Lower Bound) is a mathematical quantity derived from the E-step:

$$
\text{ELBO}_t = \sum_{i=1}^{n} \sum_{c=1}^{k} r_{ic} \left[ \log\left(\pi_c \, \mathcal{N}(x_i\mid\mu_c,\Sigma_c)\right) - \log(r_{ic}) \right]
$$

**Key property:** After each M-step update, $\text{ELBO}_{t+1} \geq \text{ELBO}_t$ (monotonic increase or plateau).

**Convergence rule in sklearn:**
- Compute relative improvement: $\Delta = \frac{|\text{ELBO}_t - \text{ELBO}_{t-1}|}{|\text{ELBO}_{t-1}|}$
- If $\Delta < \mathrm{tol}$ (default 0.001): declare **converged** and stop.
- If iteration count reaches $n_{\mathrm{iter,max}}$ (default 200): stop (may be unconverged).
- Report `converged=true` if stopped due to $\Delta < \mathrm{tol}$; otherwise `converged=false`.

**Per-sample lower bound:** sklearn reports `lower_bound` as the average ELBO divided by $n$ (easier to compare across datasets with different sizes).

### 5.6 Convergence Results at $k_{\mathrm{best}}$

- $\mathrm{converged}$: True
- $n_{\mathrm{iter}}$: 5
- $\mathrm{tol}$: 0.001
- $n_{\mathrm{iter,max}}$: 200
- $\mathrm{lower\_bound}$: 7.460114479064941
- Final refit on $X_{\mathrm{full}}$ converged: True
- Final refit $n_{\mathrm{iter}}$: 5
- Final refit $\mathrm{tol}/n_{\mathrm{iter,max}}$: 0.001 / 200

## 6. BIC Evidence (at $k_{\mathrm{best}}$)

### 6.0 Core Concepts and Symbols for This Section

**Data and Model Dimensions:**
- $n$: sample size (number of rows in $X_{\mathrm{search}}$ or $X_{\mathrm{full}}$).
- $d$: number of selected PC features (dimension).
- $k$: number of Gaussian components being compared.
- $X_{\mathrm{search}}$: subset of samples used for model search.
- $X_{\mathrm{full}}$: complete sample set (used for final refit).

**Likelihood and Parameters:**
- $L(k)$ or $\mathcal{L}_k = p(X \mid \theta_k)$: marginal likelihood (probability of data given model with $k$ components).
- $\ell(k) = \log L(k)$: log-likelihood.
- $\theta_k = \{\pi_c, \mu_c, \Sigma_c\}_{c=1}^{k}$: parameter set for model with $k$ components.
- $p$ or $p(k)$: total number of free parameters in model with $k$ components.
  - $p_{\mathrm{mean}} = kd$: parameters for all component means.
  - $p_{\mathrm{weight}} = k-1$: free parameters for mixture weights (sum constraint).
  - $p_{\mathrm{cov}}(k)$: parameters for covariance matrices (depends on `covariance_type`).
  - $p(k) = p_{\mathrm{mean}} + p_{\mathrm{cov}} + p_{\mathrm{weight}}$.

**Information Criteria:**
- $\mathrm{BIC} = -2\ell + p\log n$: Bayesian Information Criterion (balances fit and complexity).
- $\mathrm{AIC} = -2\ell + 2p$: Akaike Information Criterion (alternative complexity penalty).
- Higher $\ell$ favors better fit; higher $p$ increases penalty; lower BIC/AIC is preferred.

**Selection Principle:**
- $k_{\mathrm{best}} = \arg\min_k \mathrm{BIC}(k)$: component count that minimizes BIC.

BIC definition used by the audit:

$$
\mathrm{BIC}=-2\ell + p\log n
$$

where $\ell$ is total log-likelihood, $n$ is sample size, and $p$ is free-parameter count.

### 6.1 Parameter Count $p$
$$
p=p_{\mathrm{mean}}+p_{\mathrm{cov}}+p_{\mathrm{weight}}
$$
$$
p_{\mathrm{mean}}=kd,\quad p_{\mathrm{weight}}=k-1
$$
with $p_{\mathrm{weight}}=k-1$ because $\sum_{c=1}^{k}\pi_c=1$.
$$
\text{full: }p_{\mathrm{cov}}=k\frac{d(d+1)}{2},\quad \text{tied: }p_{\mathrm{cov}}=\frac{d(d+1)}{2}
$$
$$
\text{diag: }p_{\mathrm{cov}}=kd,\quad \text{spherical: }p_{\mathrm{cov}}=k
$$

### 6.2 Numeric Snapshot at $k_{\mathrm{best}}$
- $k_{\mathrm{best}}$: 21
- $n$: 180270
- $d$: 2
- covariance_type: full
- $p$: 125
- $p_{\mathrm{mean}}=kd$: 42
- $p_{\mathrm{cov}}$ (full): 63
- $p_{\mathrm{weight}}=k-1$: 20
- $p_{\mathrm{check}}=p_{\mathrm{mean}}+p_{\mathrm{cov}}+p_{\mathrm{weight}}$: 125
- average log-likelihood ($\ell/n$): 7.4602580070495605
- total log-likelihood ($\ell$): 1344860.7109308243
- fit term ($-2\ell$): -2689721.4218616486
- penalty term ($p\log n$): 1512.7763757495104
- recomposed BIC ($-2\ell + p\log n$): -2688208.645485899
- bic_sklearn: -2688208.645485899
- bic_manual: -2688208.645485899
- bic_abs_diff: 0.0

### 6.3 Why BIC Is Trustworthy Here
- `bic_manual` is recomputed independently from $\ell$, $n$, and $p$.
- `bic_abs_diff\approx 0` confirms consistency with sklearn's BIC implementation.
- This creates an explicit audit trail from formula to numeric decision.

## 7. LR Diagnostics Around $k_{\mathrm{best}}$

### 7.0 Notation and Definitions for This Section

**Model Comparison:**
- $k_{\mathrm{hi}}, k_{\mathrm{lo}}$: higher and lower component counts being compared (e.g., $k_{\mathrm{hi}}=27, k_{\mathrm{lo}}=26$).
- $L(k_{\mathrm{hi}}), L(k_{\mathrm{lo}})$: marginal likelihoods for models with $k_{\mathrm{hi}}$ and $k_{\mathrm{lo}}$ components.
- $\ell(k_{\mathrm{hi}}), \ell(k_{\mathrm{lo}})$: log-likelihoods (logarithms of above).
- $\theta_{\mathrm{hi}} = \{\pi_c,\mu_c,\Sigma_c\}_{c=1}^{k_{\mathrm{hi}}}$: fitted parameters at $k_{\mathrm{hi}}$.
- $\theta_{\mathrm{lo}} = \{\pi_c,\mu_c,\Sigma_c\}_{c=1}^{k_{\mathrm{lo}}}$: fitted parameters at $k_{\mathrm{lo}}$.

**Likelihood Ratio and Complexity:**
- $\mathrm{LR} = 2[\ell(k_{\mathrm{hi}}) - \ell(k_{\mathrm{lo}})]$: twice the log-likelihood difference (positive if $k_{\mathrm{hi}}$ fits better).
- $\Delta p = p(k_{\mathrm{hi}}) - p(k_{\mathrm{lo}})$: increase in free parameters when adding components.
- Note: Unlike standard LRT, $\mathrm{LR}$ under the null hypothesis (true $k=k_{\mathrm{lo}}$) does NOT follow $\chi^2_{\Delta p}$ exactly due to boundary effects (component weights $\pi_c \geq 0$).

**Information Criteria Differences:**
- $\Delta \mathrm{BIC} = \mathrm{BIC}(k_{\mathrm{hi}}) - \mathrm{BIC}(k_{\mathrm{lo}})$: BIC comparison.
  - Negative $\Delta \mathrm{BIC}$ favors $k_{\mathrm{hi}}$ (more components).
  - Positive $\Delta \mathrm{BIC}$ favors $k_{\mathrm{lo}}$ (fewer components).
- $\mathrm{BF}_{\mathrm{hi,lo}} = \exp\left(-\frac{\Delta \mathrm{BIC}}{2}\right)$: approximate Bayes Factor (BIC-based), ratio of posterior odds.
- $\log_{10}(\mathrm{BF}_{\mathrm{hi,lo}})$: log10-scale Bayes Factor (positive favors $k_{\mathrm{hi}}$, negative favors $k_{\mathrm{lo}}$).

**Interpretation Caveat:**
- For GMM component-count selection, classical chi-square LRT p-values are unreliable (mixture model regularity conditions violated).
- $\mathrm{LR}$ and $\Delta\mathrm{BIC}$ are reported as diagnostics to understand local model behavior.
- **Primary decision rule:** BIC, not hypothesis test.

$$
\mathrm{LR}=2\left[\log L(k_{hi})-\log L(k_{lo})\right]
$$
$$
\Delta p=p(k_{hi})-p(k_{lo})
$$
where $\mathrm{LR}$ measures fit gain and $\Delta p$ measures added complexity.

### 7.1 Local LR Diagnostics
- Lower-neighbor comparison: $k_{hi}=21$ vs $k_{lo}=20$
- $\log L(k_{hi})$: 1344860.7109308243
- $\log L(k_{lo})$: 1344479.9965953827
- $\mathrm{LR}$: 761.4286708831787
- $\Delta p$: 6
- $\Delta\mathrm{BIC}$: -688.8154048472643
- $\log_{10}(\mathrm{BF}_{hi,lo})$ (BIC approximation): 149.57436468756063
- Upper-neighbor comparison: $k_{hi}=22$ vs $k_{lo}=21$
- $\log L(k_{hi})$: 1344866.72809124
- $\log L(k_{lo})$: 1344860.7109308243
- $\mathrm{LR}$: 12.034320831298828
- $\Delta p$: 6
- $\Delta\mathrm{BIC}$: 60.57894520461559
- $\log_{10}(\mathrm{BF}_{hi,lo})$ (BIC approximation): -13.154550810942004

## 8. Audit Field Dictionary
Convergence/BIC dedicated logs are simplified views derived from these events.
- `gmm_search_convergence_log.jsonl` now includes `stop_reason`, `convergence_rule`, and EM parameter snapshot fields.
- `gmm_search_bic_log.jsonl` focuses only on likelihood/parameter/BIC-AIC terms used for criteria calculations.

### 8.1 `run_start` event
- `selected_pc_cols`: selected feature columns.
- `n_rows_full`: row count of $X_{\mathrm{full}}$.
- `n_rows_search`: row count of $X_{\mathrm{search}}$.
- `search_fraction`: ratio `n_rows_search / n_rows_full`.
- `config`: full hyper-parameter snapshot.

### 8.2 `search_iteration` event (one record per $k$)
- `k`: current component count.
- `converged`, `n_iter`, `lower_bound`: EM convergence diagnostics.
- `tol`, `max_iter`: stopping controls.
- `n_samples`, `n_features`: matrix shape for this fit.
- `bic_sklearn`, `aic_sklearn`: sklearn criteria.
- `n_parameters`: parameter count $p$.
- `average_log_likelihood`, `total_log_likelihood`: likelihood terms.
- `bic_manual`, `aic_manual`: audit recomputation.
- `bic_abs_diff`, `aic_abs_diff`: consistency checks.
- `has_empty_clusters`, `n_non_empty_clusters`: occupancy diagnostics.
- `weights_min`, `weights_max`, `weights_entropy`: weight diagnostics.
- `components`: per-component details (`component_id`, `weight`, `mean`, `covariance_matrix`, `mean_l2_norm`, `cov_trace`).

### 8.2a Convergence Log Fields (`gmm_search_convergence_log.jsonl`)
- `converged`, `n_iter`, `tol`, `max_iter`, `lower_bound`: direct convergence outcome and decision thresholds.
- `convergence_rule`: textual rule used by optimizer stopping logic.
- `stop_reason`: inferred stop category, e.g. `tol_reached_before_max_iter` or `max_iter_reached_without_convergence`.
- `decision_explanation`: simplified status (`accepted_converged` or `not_converged`).
- `covariance_type`, `n_init`, `init_params`, `reg_covar`, `random_state`: key EM/GMM configuration that affects convergence behavior.

### 8.3 `best_model_selection` event
- `best_k`, `best_bic`: selected optimum by BIC.
- `require_non_empty_clusters`: non-empty eligibility rule.
- `search_non_empty_model_count`, `search_model_count`: candidate pool counts.

### 8.4 `final_model` event
- `best_k`: selected $k$ used for final refit.
- `converged`, `n_iter`, `lower_bound`: final-fit convergence diagnostics.
- `bic_full_data`, `aic_full_data`: criteria on $X_{\mathrm{full}}$.
- `cluster_counts`: final sample count per cluster.

## 9. $k_{\mathrm{best}}$ Component Profile and Meaning

### 9.0 Notation and Definitions for This Section

**Index and Posterior Responsibility:**
- $c \in \{1, 2, \ldots, k_{\mathrm{best}}\}$: component index.
- $r_{ic} = p(z_i=c \mid x_i, \theta)$: posterior responsibility of sample $i$ for component $c$ (soft cluster assignment).
  - Computed from E-step: $r_{ic} = \frac{\pi_c \mathcal{N}(x_i \mid \mu_c, \Sigma_c)}{\sum_{j=1}^{k}\pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$.
  - Constraint: $\sum_{c=1}^{k}r_{ic} = 1$ for each sample $i$ (sample is distributed across all components).

**Component Statistics:**
- $N_c = \sum_{i=1}^{n}r_{ic}$: effective sample size (total soft weight) for component $c$.
- $\pi_c = \frac{N_c}{n}$: mixture weight or proportion of component $c$ in the fitted mixture.
  - Updated in M-step: $N_c$ aggregates responsibilities, then normalizes to $\pi_c$.
  - Constraint: $\pi_c \geq 0$ and $\sum_{c=1}^{k}\pi_c = 1$.
- $\mu_c \in \mathbb{R}^d$: component-specific mean vector, fitted in M-step.
- $\Sigma_c \in \mathbb{R}^{d \times d}$: component-specific covariance matrix, fitted in M-step.

**Summary Metrics:**
- $\|\mu_c\|_2 = \sqrt{\sum_{j=1}^{d}(\mu_c)_j^2}$: Euclidean norm (distance from origin in PC space), reported as `Mean L2 Norm`.
- $\mathrm{tr}(\Sigma_c) = \sum_{j=1}^{d}(\Sigma_c)_{jj}$: trace (sum of diagonal variance terms), reported as `Covariance Trace`.
  - Note: Only captures marginal variances; does not include correlations or off-diagonal covariance terms.

**Distance Metric:**
- $d_M(c_1,c_2) = \sqrt{(\mu_{c_1}-\mu_{c_2})^\top S_{c_1,c_2}^{-1}(\mu_{c_1}-\mu_{c_2})}$: Mahalanobis distance with pair-specific pooled covariance between two components.
- $S_{c_1,c_2}=\tfrac{1}{2}(\Sigma_{c_1}+\Sigma_{c_2})$: pair-specific pooled covariance used for the component-pair distance.
  - Requires full $\mu_c$ and $\Sigma_c$ (or inverse); $\|\mu_c\|_2$ and $\mathrm{tr}(\Sigma_c)$ alone are insufficient.

**Table Columns:**
- `Component`: numeric ID for $c$ (index $1, 2, \ldots, k_{\mathrm{best}}$).
- `Weight`: estimate of $\pi_c$.
- `Mean L2 Norm`: value of $\|\mu_c\|_2$.
- `Covariance Trace`: value of $\mathrm{tr}(\Sigma_c)$.

### 9.1 How `Weight` Is Computed and Interpreted
- Source in EM: responsibilities $r_{ic}=p(z_i=c\mid x_i,\theta)$ from E-step.
- Effective count: $N_c=\sum_{i=1}^{n} r_{ic}$.
- M-step update: $\pi_c=N_c/n$.
- Constraint: $\pi_c\ge 0$ and $\sum_{c=1}^{k}\pi_c=1$.
- Interpretation: `Weight` is a soft-assignment expected proportion, not a hard-cluster fraction from argmax labels.

### 9.2 How to Read the Three Summary Columns
- `Weight`: component size/probability mass in the fitted mixture.
- `Mean L2 Norm`: $\|\mu_c\|_2=\sqrt{\sum_{j=1}^{d}\mu_{cj}^2}$, a location summary in PC space.
- `Covariance Trace`: $\mathrm{tr}(\Sigma_c)=\sum_{j=1}^{d}\Sigma_{c,jj}$, a spread summary (marginal variances only).
- These are compact descriptors of $\mu_c$ and $\Sigma_c$; they do not capture full directional covariance structure.

### 9.3 Relation to Component-Component Mahalanobis Distance
- Exact component-pair distance uses full means and covariances of two components:
  $d_M(c_1,c_2)=\sqrt{(\mu_{c_1}-\mu_{c_2})^\top S_{c_1,c_2}^{-1}(\mu_{c_1}-\mu_{c_2})}$, with $S_{c_1,c_2}=\tfrac{1}{2}(\Sigma_{c_1}+\Sigma_{c_2})$.
- `Mean L2 Norm` and `Covariance Trace` help interpretation but cannot replace full-parameter component-distance computation.

### 9.4 Practical Interpretation Tips
- A component with high `Weight` and low `Covariance Trace` is both common and compact.
- A component with very low `Weight` but high `Covariance Trace` may represent a rare and diffuse subgroup or potential instability candidate.
- `Mean L2 Norm` is reported on the same PC scale used for model fitting in this workflow.
- Use `Weight` together with `Covariance Trace`: tiny weight + very large trace often suggests weakly supported broad components.
- For scientific reporting, avoid interpreting a single column alone; combine size (`Weight`), location (`Mean L2 Norm`), and spread (`Covariance Trace`).

| Component | Weight | Mean L2 Norm | Covariance Trace |
|---|---|---|---|
| 0 | 0.052795 | 0.013100 | 0.000018 |
| 1 | 0.015650 | 0.053964 | 0.000022 |
| 2 | 0.109032 | 0.007977 | 0.000015 |
| 3 | 0.042312 | 0.007420 | 0.000025 |
| 4 | 0.001004 | 0.071907 | 0.000021 |
| 5 | 0.029381 | 0.060658 | 0.000021 |
| 6 | 0.049396 | 0.010606 | 0.000020 |
| 7 | 0.105670 | 0.006693 | 0.000017 |
| 8 | 0.002393 | 0.033714 | 0.000045 |
| 9 | 0.155289 | 0.005092 | 0.000014 |
| 10 | 0.045257 | 0.017621 | 0.000019 |
| 11 | 0.015693 | 0.067385 | 0.000027 |
| 12 | 0.017836 | 0.017825 | 0.000026 |
| 13 | 0.002466 | 0.027358 | 0.000026 |
| 14 | 0.081466 | 0.005655 | 0.000016 |
| 15 | 0.058934 | 0.002380 | 0.000020 |
| 16 | 0.073578 | 0.007174 | 0.000021 |
| 17 | 0.070079 | 0.011359 | 0.000018 |
| 18 | 0.000377 | 0.074344 | 0.000333 |
| 19 | 0.003814 | 0.051669 | 0.000010 |
| 20 | 0.067578 | 0.007747 | 0.000016 |

## 10. Quick Validation Checklist
- Confirm `search_iteration` at $k_{\mathrm{best}}$ has `converged=true`.
- Confirm `final_model` has `converged=true` on full data.
- Confirm `bic_abs_diff` is near zero.
- Confirm local LR diagnostics are directionally consistent with BIC differences.
- Inspect `weights_min` and empty-cluster diagnostics for degeneracy signals.

## 11. Output Files
- Human report (md, overwritten each run): `gmm_search_professional_report.md`
- Convergence variables log (jsonl): `gmm_search_convergence_log.jsonl`
- BIC variables log (jsonl): `gmm_search_bic_log.jsonl`
- Best-k component description log (json): `gmm_best_k_components_log.json`
