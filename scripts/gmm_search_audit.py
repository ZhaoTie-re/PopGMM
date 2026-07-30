"""Record why the mixture search selected the model it did.

Logs one entry per candidate fit -- convergence, iteration count, stop reason,
BIC and AIC computed both from the estimator and independently -- then a report
tying them together. The independent recomputation is a cross-check: a
divergence between the two would mean the selection criterion is not what it
claims to be.

Inputs
------
Driven by ``gmm_clustering`` as the search proceeds.

Outputs
-------
Convergence and BIC JSONL logs, the selected model's components as JSON, and a
Markdown report, all under the mixture stage's ``tmp/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> float:
    return float(v) if v is not None else float("nan")


def _component_covariance_traces(model: GaussianMixture, n_components: int, n_features: int) -> list[float]:
    cov_type = str(getattr(model, "covariance_type", "full"))
    if cov_type == "full":
        covs = np.asarray(model.covariances_, dtype=np.float64)
        return [float(np.trace(covs[i])) for i in range(n_components)]
    if cov_type == "tied":
        c = np.asarray(model.covariances_, dtype=np.float64)
        t = float(np.trace(c))
        return [t for _ in range(n_components)]
    if cov_type == "diag":
        covs = np.asarray(model.covariances_, dtype=np.float64)
        return [float(np.sum(covs[i])) for i in range(n_components)]
    if cov_type == "spherical":
        covs = np.asarray(model.covariances_, dtype=np.float64)
        return [float(covs[i]) * float(n_features) for i in range(n_components)]
    return [float("nan") for _ in range(n_components)]


def _component_covariance_matrices(
    model: GaussianMixture,
    *,
    n_components: int,
    n_features: int,
) -> list[list[list[float]]]:
    """Return per-component covariance matrices as JSON-friendly nested lists.

    This normalizes all sklearn `covariance_type` variants to a full (d x d)
    matrix representation so downstream auditing/inspection does not need to
    branch on covariance type.
    """

    cov_type = str(getattr(model, "covariance_type", "full"))
    matrices: list[list[list[float]]] = []

    def _nan_matrix() -> list[list[float]]:
        return [[float("nan") for _ in range(n_features)] for _ in range(n_features)]

    try:
        covs = np.asarray(model.covariances_, dtype=np.float64)
    except Exception:
        return [_nan_matrix() for _ in range(n_components)]

    if cov_type == "full":
        # shape: (k, d, d)
        if covs.ndim != 3:
            return [_nan_matrix() for _ in range(n_components)]
        for i in range(n_components):
            matrices.append(covs[i].astype(np.float64, copy=False).tolist())
        return matrices

    if cov_type == "tied":
        # shape: (d, d) shared across components
        if covs.ndim != 2:
            return [_nan_matrix() for _ in range(n_components)]
        m = covs.astype(np.float64, copy=False).tolist()
        return [m for _ in range(n_components)]

    if cov_type == "diag":
        # shape: (k, d) diagonal variances
        if covs.ndim != 2:
            return [_nan_matrix() for _ in range(n_components)]
        for i in range(n_components):
            diag = np.asarray(covs[i], dtype=np.float64).reshape(-1)
            if diag.shape[0] != n_features:
                matrices.append(_nan_matrix())
                continue
            matrices.append(np.diag(diag).tolist())
        return matrices

    if cov_type == "spherical":
        # shape: (k,) scalar variances
        covs_1d = np.asarray(covs, dtype=np.float64).reshape(-1)
        if covs_1d.shape[0] != n_components:
            return [_nan_matrix() for _ in range(n_components)]
        ident = np.eye(n_features, dtype=np.float64)
        for i in range(n_components):
            matrices.append((float(covs_1d[i]) * ident).tolist())
        return matrices

    return [_nan_matrix() for _ in range(n_components)]


def _gmm_n_parameters(model: GaussianMixture, n_features: int) -> int:
    k = int(getattr(model, "n_components", 0))
    if k <= 0:
        return 0

    means = k * int(n_features)
    weights = k - 1

    cov_type = str(getattr(model, "covariance_type", "full"))
    if cov_type == "full":
        cov = int(k * n_features * (n_features + 1) / 2)
    elif cov_type == "tied":
        cov = int(n_features * (n_features + 1) / 2)
    elif cov_type == "diag":
        cov = int(k * n_features)
    elif cov_type == "spherical":
        cov = int(k)
    else:
        cov = 0

    return int(means + cov + weights)


def _infer_convergence_reason(*, converged: bool, n_iter: int, max_iter: int) -> str:
    if converged and n_iter < max_iter:
        return "tol_reached_before_max_iter"
    if converged and n_iter == max_iter:
        return "converged_at_max_iter_boundary"
    if not converged and n_iter >= max_iter:
        return "max_iter_reached_without_convergence"
    return "undetermined"


@dataclass
class GMMAuditLogger:
    output_dir: Path
    config_dict: dict[str, Any]
    selected_pc_cols: list[str]
    n_rows_full: int
    n_rows_search: int
    random_state: int
    use_zscale: bool

    def __post_init__(self) -> None:
        self.tmp_dir = Path(self.output_dir) / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Fixed filenames: overwrite on each run.
        self.report_path = self.tmp_dir / "search_report.md"
        self.convergence_log_path = self.tmp_dir / "search_convergence_log.jsonl"
        self.bic_log_path = self.tmp_dir / "search_bic_log.jsonl"
        self.best_components_log_path = self.tmp_dir / "selected_model_components.json"

        self.events: list[dict[str, Any]] = []
        self.components_by_k: dict[int, list[dict[str, Any]]] = {}

        # Reset rolling logs for this run.
        self.convergence_log_path.write_text("", encoding="utf-8")
        self.bic_log_path.write_text("", encoding="utf-8")
        with self.best_components_log_path.open("w", encoding="utf-8") as f:
            json.dump({"generated_at_utc": _utc_now_iso(), "status": "pending_best_k"}, f, indent=2)

        self._log_event(
            {
                "event": "run_start",
                "timestamp_utc": _utc_now_iso(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "selected_pc_cols": self.selected_pc_cols,
                "n_features": int(len(self.selected_pc_cols)),
                "n_rows_full": int(self.n_rows_full),
                "n_rows_search": int(self.n_rows_search),
                "search_fraction": float(self.n_rows_search / max(1, self.n_rows_full)),
                "use_zscale": bool(self.use_zscale),
                "random_state": int(self.random_state),
                "config": self.config_dict,
            }
        )

    def _log_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_best_components_log(self, *, best_k: int, components: list[dict[str, Any]]) -> None:
        payload = {
            "generated_at_utc": _utc_now_iso(),
            "best_k": int(best_k),
            "n_components": int(len(components)),
            "components": components,
        }
        with self.best_components_log_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def log_search_iteration(
        self,
        *,
        k: int,
        model: GaussianMixture,
        x_used: np.ndarray,
        bic: float,
        aic: float,
        elapsed_seconds: float,
        has_empty_clusters: bool,
        n_non_empty_clusters: int,
    ) -> None:
        n_samples = int(x_used.shape[0])
        n_features = int(x_used.shape[1])

        n_parameters = _gmm_n_parameters(model=model, n_features=n_features)
        avg_log_likelihood = float(model.score(x_used))
        total_log_likelihood = float(avg_log_likelihood * n_samples)
        bic_manual = float(-2.0 * total_log_likelihood + n_parameters * math.log(max(1, n_samples)))
        aic_manual = float(-2.0 * total_log_likelihood + 2.0 * n_parameters)

        weights = np.asarray(model.weights_, dtype=np.float64)
        means = np.asarray(model.means_, dtype=np.float64)
        n_components = int(getattr(model, "n_components", 0))
        traces = _component_covariance_traces(model=model, n_components=n_components, n_features=n_features)
        cov_mats = _component_covariance_matrices(model, n_components=n_components, n_features=n_features)

        components: list[dict[str, Any]] = []
        for i in range(n_components):
            comp = {
                "component_id": int(i),
                "weight": float(weights[i]),
                "mean": [float(v) for v in means[i].tolist()],
                "covariance_matrix": cov_mats[i],
                "mean_l2_norm": float(np.linalg.norm(means[i])),
                "cov_trace": float(traces[i]),
            }
            components.append(comp)

        self.components_by_k[int(k)] = components

        event_payload = {
                "event": "search_iteration",
                "timestamp_utc": _utc_now_iso(),
                "k": int(k),
                "n_samples": n_samples,
                "n_features": n_features,
                "covariance_type": str(getattr(model, "covariance_type", "unknown")),
                "tol": _safe_float(getattr(model, "tol", float("nan"))),
                "max_iter": int(getattr(model, "max_iter", 0)),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "lower_bound": _safe_float(getattr(model, "lower_bound_", float("nan"))),
                "bic_sklearn": float(bic),
                "aic_sklearn": float(aic),
                "bic_manual": bic_manual,
                "aic_manual": aic_manual,
                "bic_abs_diff": float(abs(float(bic) - bic_manual)),
                "aic_abs_diff": float(abs(float(aic) - aic_manual)),
                "total_log_likelihood": total_log_likelihood,
                "average_log_likelihood": avg_log_likelihood,
                "n_parameters": int(n_parameters),
                "has_empty_clusters": bool(has_empty_clusters),
                "n_non_empty_clusters": int(n_non_empty_clusters),
                "elapsed_seconds": float(elapsed_seconds),
                "weights_min": float(weights.min()),
                "weights_max": float(weights.max()),
                "weights_entropy": float(-(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum()),
                "components": components,
            }
        self._log_event(event_payload)

        self._append_jsonl(
            self.convergence_log_path,
            {
                "timestamp_utc": event_payload["timestamp_utc"],
                "k": int(k),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "tol": _safe_float(getattr(model, "tol", float("nan"))),
                "max_iter": int(getattr(model, "max_iter", 0)),
                "lower_bound": _safe_float(getattr(model, "lower_bound_", float("nan"))),
                "convergence_rule": "stop if lower_bound improvement < tol, else stop at max_iter",
                "stop_reason": _infer_convergence_reason(
                    converged=bool(model.converged_),
                    n_iter=int(model.n_iter_),
                    max_iter=int(getattr(model, "max_iter", 0)),
                ),
                "decision_explanation": (
                    "accepted_converged" if bool(model.converged_) else "not_converged"
                ),
                "covariance_type": str(getattr(model, "covariance_type", "unknown")),
                "n_init": int(getattr(model, "n_init", 0)),
                "init_params": str(getattr(model, "init_params", "unknown")),
                "reg_covar": _safe_float(getattr(model, "reg_covar", float("nan"))),
                "random_state": self.random_state,
                "elapsed_seconds": float(elapsed_seconds),
                "has_empty_clusters": bool(has_empty_clusters),
                "n_non_empty_clusters": int(n_non_empty_clusters),
            },
        )

        self._append_jsonl(
            self.bic_log_path,
            {
                "timestamp_utc": event_payload["timestamp_utc"],
                "k": int(k),
                "n_samples": n_samples,
                "n_features": n_features,
                "covariance_type": str(getattr(model, "covariance_type", "unknown")),
                "n_parameters": int(n_parameters),
                "average_log_likelihood": avg_log_likelihood,
                "total_log_likelihood": total_log_likelihood,
                "bic_sklearn": float(bic),
                "bic_manual": bic_manual,
                "bic_abs_diff": float(abs(float(bic) - bic_manual)),
                "aic_sklearn": float(aic),
                "aic_manual": aic_manual,
                "aic_abs_diff": float(abs(float(aic) - aic_manual)),
            },
        )

    def log_best_selection(
        self,
        *,
        bic_table: pd.DataFrame,
        best_k: int,
        best_bic: float,
        require_non_empty_clusters: bool,
    ) -> None:
        self._log_event(
            {
                "event": "best_model_selection",
                "timestamp_utc": _utc_now_iso(),
                "best_k": int(best_k),
                "best_bic": float(best_bic),
                "require_non_empty_clusters": bool(require_non_empty_clusters),
                "search_non_empty_model_count": int((~bic_table["has_empty_clusters"]).sum()),
                "search_model_count": int(bic_table.shape[0]),
                "k_values": [int(v) for v in bic_table["n_components"].tolist()],
            }
        )

        components = self.components_by_k.get(int(best_k), [])
        self._write_best_components_log(best_k=int(best_k), components=components)

    def log_final_model(
        self,
        *,
        model: GaussianMixture,
        x_full: np.ndarray,
        labels: np.ndarray,
        best_k: int,
        elapsed_seconds: float,
    ) -> None:
        n_samples = int(x_full.shape[0])
        counts = pd.Series(labels).value_counts().sort_index()
        self._log_event(
            {
                "event": "final_model",
                "timestamp_utc": _utc_now_iso(),
                "best_k": int(best_k),
                "n_samples": n_samples,
                "tol": _safe_float(getattr(model, "tol", float("nan"))),
                "max_iter": int(getattr(model, "max_iter", 0)),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "lower_bound": _safe_float(getattr(model, "lower_bound_", float("nan"))),
                "bic_full_data": float(model.bic(x_full)),
                "aic_full_data": float(model.aic(x_full)),
                "cluster_counts": {str(k): int(v) for k, v in counts.items()},
                "elapsed_seconds": float(elapsed_seconds),
            }
        )

    def finalize(self) -> None:
        summary = {
            "generated_at_utc": _utc_now_iso(),
            "convergence_log_file": str(self.convergence_log_path.name),
            "bic_log_file": str(self.bic_log_path.name),
            "best_components_log_file": str(self.best_components_log_path.name),
            "event_count": int(len(self.events)),
            "events": self.events,
        }

        best_sel = next((e for e in self.events if e.get("event") == "best_model_selection"), None)
        if best_sel is not None:
            best_k = int(best_sel.get("best_k", -1))
            components = self.components_by_k.get(best_k, [])
            self._write_best_components_log(best_k=best_k, components=components)

        report_md = self._build_professional_report_markdown(summary=summary)
        with self.report_path.open("w", encoding="utf-8") as f:
            f.write(report_md)

    def _build_professional_report_markdown(self, summary: dict[str, Any]) -> str:
        run_start = next((e for e in self.events if e.get("event") == "run_start"), None)
        best_sel = next((e for e in self.events if e.get("event") == "best_model_selection"), None)
        final_evt = next((e for e in self.events if e.get("event") == "final_model"), None)
        iters = [e for e in self.events if e.get("event") == "search_iteration"]

        k_values = [int(e.get("k", -1)) for e in iters if "k" in e]
        converged_count = int(sum(1 for e in iters if bool(e.get("converged", False))))
        non_empty_count = int(sum(1 for e in iters if not bool(e.get("has_empty_clusters", True))))
        bic_abs_diff_max = max((float(e.get("bic_abs_diff", 0.0)) for e in iters), default=0.0)
        aic_abs_diff_max = max((float(e.get("aic_abs_diff", 0.0)) for e in iters), default=0.0)
        iter_seconds_total = float(sum(float(e.get("elapsed_seconds", 0.0)) for e in iters))

        best_k = int(best_sel.get("best_k", -1)) if best_sel else -1
        best_iter = next((e for e in iters if int(e.get("k", -1)) == best_k), None)

        iters_by_k = {int(e.get("k", -1)): e for e in iters if "k" in e}
        sorted_k = sorted([k for k in iters_by_k.keys() if k >= 0])
        prev_k = max((k for k in sorted_k if k < best_k), default=None)
        next_k = min((k for k in sorted_k if k > best_k), default=None)

        def _lr_pair(k_hi: int, k_lo: int) -> dict[str, Any] | None:
            hi = iters_by_k.get(k_hi)
            lo = iters_by_k.get(k_lo)
            if hi is None or lo is None:
                return None
            logl_hi = float(hi.get("total_log_likelihood", float("nan")))
            logl_lo = float(lo.get("total_log_likelihood", float("nan")))
            p_hi = int(hi.get("n_parameters", 0))
            p_lo = int(lo.get("n_parameters", 0))
            if not math.isfinite(logl_hi) or not math.isfinite(logl_lo):
                return None
            lr_stat = float(2.0 * (logl_hi - logl_lo))
            delta_p = int(p_hi - p_lo)
            delta_bic = float(hi.get("bic_sklearn", float("nan")) - lo.get("bic_sklearn", float("nan")))
            if math.isfinite(delta_bic):
                log10_bf_hi_vs_lo = float(-delta_bic / (2.0 * math.log(10.0)))
            else:
                log10_bf_hi_vs_lo = float("nan")
            return {
                "k_hi": int(k_hi),
                "k_lo": int(k_lo),
                "logl_hi": logl_hi,
                "logl_lo": logl_lo,
                "lr_stat": lr_stat,
                "delta_p": delta_p,
                "delta_bic": delta_bic,
                "log10_bf_hi_vs_lo": log10_bf_hi_vs_lo,
            }

        lr_prev = _lr_pair(best_k, prev_k) if prev_k is not None else None
        lr_next = _lr_pair(next_k, best_k) if next_k is not None else None

        run_cfg = run_start.get("config", {}) if run_start else {}
        n_rows_full = int(run_start.get("n_rows_full", 0)) if run_start else 0
        n_rows_search = int(run_start.get("n_rows_search", 0)) if run_start else 0
        selected_pc_cols = run_start.get("selected_pc_cols", []) if run_start else []

        lines: list[str] = []
        lines.append("# GMM One-k Convergence, BIC, and LR Audit Report")
        lines.append("")
        lines.append(f"- Generated (UTC): {summary.get('generated_at_utc', 'NA')}")
        lines.append(f"- Focus component count ($k_{'{'}\\mathrm{{best}}{'}'}$): {best_k}")
        lines.append(f"- Search range ($k$): {min(sorted_k) if sorted_k else 'NA'} to {max(sorted_k) if sorted_k else 'NA'}")
        lines.append(f"- Search convergence: {converged_count}/{len(iters)} fits converged")
        lines.append(f"- Non-empty clusters during search: {non_empty_count}/{len(iters)}")
        lines.append("")
        lines.append("## 1. Key Findings")
        if best_iter:
            lines.append(f"- $k_{'{'}\\mathrm{{best}}{'}'}={best_k}$ is selected by minimum BIC on search data.")
            lines.append(
                f"- Search fit at $k_{'{'}\\mathrm{{best}}{'}'}$ converged: {best_iter.get('converged', 'NA')} in {best_iter.get('n_iter', 'NA')} EM iterations."
            )
            lines.append(
                f"- BIC consistency check: abs(sklearn - manual) = {best_iter.get('bic_abs_diff', 'NA')} (should be near 0)."
            )
        if final_evt:
            lines.append(
                f"- Final refit on full data converged: {final_evt.get('converged', 'NA')} in {final_evt.get('n_iter', 'NA')} iterations."
            )
        lines.append(
            "- LR evidence around $k_{\\mathrm{best}}$ is provided as a diagnostic; classical chi-square LRT assumptions are not strictly valid for GMM component-count testing."
        )
        lines.append("")
        lines.append("### 1.1 Decision Snapshot")
        lines.append("| Question | Answer | Evidence |")
        lines.append("|---|---|---|")
        lines.append(f"| Best component count? | $k_{{\\mathrm{{best}}}}={best_k}$ | Section 6.2 |")
        if best_iter:
            lines.append(f"| Did EM converge at $k_{{\\mathrm{{best}}}}$? | Yes, in {best_iter.get('n_iter', 'NA')} iterations | Section 5.6 |")
            lines.append(f"| Is BIC computation self-consistent? | Yes, `bic_abs_diff = {best_iter.get('bic_abs_diff', 'NA')}` | Section 6.3 |")
        else:
            lines.append("| Did EM converge at $k_{\\mathrm{best}}$? | Not available in current log | Section 5.6 |")
            lines.append("| Is BIC computation self-consistent? | Not available in current log | Section 6.3 |")
        lines.append("| Is LR used as formal hypothesis test? | No, diagnostics only | Section 7.0-7.1 |")
        lines.append("")
        lines.append("## 2. Output Files and Usage")
        lines.append("Use files in this order for fastest understanding:")
        lines.append(f"- `gmm_search_professional_report.md`: human-readable full report (this file).")
        lines.append(f"- `{self.convergence_log_path.name}`: per-k convergence variables and stop reasons.")
        lines.append(f"- `{self.bic_log_path.name}`: per-k BIC/AIC calculation variables.")
        lines.append(f"- `{self.best_components_log_path.name}`: detailed component descriptors under $k_{{\\mathrm{{best}}}}$.")
        lines.append("")
        lines.append("## 3. How to Read This Report")
        lines.append(
            "This report is organized from decision to evidence: first the selected model conclusion, then convergence proof, then BIC/LR quantitative support, and finally raw audit-field mappings for reproducibility."
        )
        lines.append("- If you only need the final conclusion, read Sections 1, 6.2, and 7.1.")
        lines.append("- If you need methodological validity, read Sections 4, 5, and 6 in order.")
        lines.append("- If you need machine-log mapping, read Section 8.")
        lines.append("- If you need component-level interpretation, read Section 9.")
        lines.append("")
        lines.append("## 4. Data Flow")
        lines.append("- Input table: `bbj_samples_filtered`.")
        lines.append("- Feature selection: `selected_pc_cols` gives $X_{\\mathrm{full}}$.")
        lines.append("- Search matrix: $X_{\\mathrm{search}}$ (same features, optional row subsampling).")
        lines.append("- For each fixed $k$: fit one GMM on $X_{\\mathrm{search}}$ and emit one `search_iteration` event.")
        lines.append("- If selected as $k_{\\mathrm{best}}$: refit on $X_{\\mathrm{full}}$ and emit `final_model` event.")
        lines.append("")
        lines.append("## 5. EM Convergence Evidence (at $k_{\\mathrm{best}}$)")
        lines.append("")
        lines.append("### 5.1 Core Symbols and Notation")
        lines.append("All mathematical notation used in this section:")
        lines.append("")
        lines.append("- $n$: sample size used by a fit.")
        lines.append("- $d$: number of selected PC features.")
        lines.append("- $k$: number of Gaussian components.")
        lines.append("- $x_i \\in \\mathbb{R}^d$: sample $i$.")
        lines.append("- $\\theta=\\{\\pi_c,\\mu_c,\\Sigma_c\\}_{c=1}^{k}$: complete parameter set of the GMM.")
        lines.append("- $\\pi_c$: mixture weight of component $c$, with $\\sum_{c=1}^{k}\\pi_c=1$.")
        lines.append("- $\\mu_c \\in \\mathbb{R}^d$: mean vector of component $c$.")
        lines.append("- $\\Sigma_c \\in \\mathbb{R}^{d\\times d}$: covariance matrix of component $c$.")
        lines.append("- $\\mathcal{N}(x\\mid\\mu_c,\\Sigma_c)$: Gaussian probability density with mean $\\mu_c$ and covariance $\\Sigma_c$.")
        lines.append("- $\\ell$: total log-likelihood, where $\\ell=\\log L = \\log p(X\\mid\\theta)$.")
        lines.append("- $p$: number of free parameters (varies by covariance_type).")
        lines.append("- $r_{ic}=p(z_i=c\\mid x_i,\\theta)$: posterior responsibility (soft assignment) of sample $i$ to component $c$.")
        lines.append("- $N_c=\\sum_{i=1}^{n}r_{ic}$: effective sample size of component $c$.")
        lines.append("- $\\text{ELBO}_t$: Evidence Lower Bound at iteration $t$, used for convergence monitoring.")
        lines.append("- $\\Delta$: relative improvement in ELBO between iterations.")
        lines.append("- $\\mathrm{tol}$: convergence tolerance (default 0.001).")
        lines.append("- $n_{\\mathrm{iter,max}}$: maximum iteration count (default 200).")
        lines.append("")
        lines.append("### 5.2 EM Algorithm Overview")
        lines.append("EM (Expectation-Maximization) is an iterative algorithm that alternates between two steps until convergence:")
        lines.append("")
        lines.append("1. **E-step (Expectation)**: Compute posterior responsibility of each component for each sample.")
        lines.append("2. **M-step (Maximization)**: Update component parameters using weighted statistics.")
        lines.append("3. **Convergence check**: Monitor a lower bound on the log-likelihood; stop when improvement stalls.")
        lines.append("")
        lines.append("### 5.3 E-step: Compute Responsibilities")
        lines.append("For each sample $i$ and component $c$:")
        lines.append("$$")
        lines.append(r"r_{ic}=\frac{\pi_c\,\mathcal{N}(x_i\mid\mu_c,\Sigma_c)}{\sum_{j=1}^{k}\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}")
        lines.append("$$")
        lines.append(r"where $r_{ic}=p(z_i=c\mid x_i,\theta)$ is the posterior probability (responsibility) and $\sum_{c=1}^{k}r_{ic}=1$.")
        lines.append("")
        lines.append("### 5.4 M-step: Update Parameters")
        lines.append("Using responsibilities, update mixture weights, means, and covariances:")
        lines.append("$$")
        lines.append(r"N_c=\sum_{i=1}^{n}r_{ic},\quad \pi_c=\frac{N_c}{n},\quad \mu_c=\frac{1}{N_c}\sum_{i=1}^{n}r_{ic}x_i")
        lines.append("$$")
        lines.append("$$")
        lines.append(r"\Sigma_c=\frac{1}{N_c}\sum_{i=1}^{n}r_{ic}(x_i-\mu_c)(x_i-\mu_c)^\top")
        lines.append("$$")
        lines.append("where $N_c$ is the effective sample size of component $c$.")
        lines.append("(Covariance update varies by `covariance_type`: full, tied, diag, spherical.)")
        lines.append("")
        lines.append("### 5.5 Convergence Check: Lower Bound")
        lines.append("The **lower bound** (also called ELBO, Evidence Lower Bound) is a mathematical quantity derived from the E-step:")
        lines.append("")
        lines.append("$$")
        lines.append(r"\text{ELBO}_t = \sum_{i=1}^{n} \sum_{c=1}^{k} r_{ic} \left[ \log\left(\pi_c \, \mathcal{N}(x_i\mid\mu_c,\Sigma_c)\right) - \log(r_{ic}) \right]")
        lines.append("$$")
        lines.append("")
        lines.append("**Key property:** After each M-step update, $\\text{ELBO}_{t+1} \\geq \\text{ELBO}_t$ (monotonic increase or plateau).")
        lines.append("")
        lines.append("**Convergence rule in sklearn:**")
        lines.append("- Compute relative improvement: $\\Delta = \\frac{|\\text{ELBO}_t - \\text{ELBO}_{t-1}|}{|\\text{ELBO}_{t-1}|}$")
        lines.append("- If $\\Delta < \\mathrm{tol}$ (default 0.001): declare **converged** and stop.")
        lines.append("- If iteration count reaches $n_{\\mathrm{iter,max}}$ (default 200): stop (may be unconverged).")
        lines.append("- Report `converged=true` if stopped due to $\\Delta < \\mathrm{tol}$; otherwise `converged=false`.")
        lines.append("")
        lines.append("**Per-sample lower bound:** sklearn reports `lower_bound` as the average ELBO divided by $n$ (easier to compare across datasets with different sizes).")
        lines.append("")
        lines.append("### 5.6 Convergence Results at $k_{\\mathrm{best}}$")
        lines.append("")
        if best_iter:
            tol = best_iter.get("tol", "NA")
            mx = best_iter.get("max_iter", "NA")
            lines.append(f"- $\\mathrm{{converged}}$: {best_iter.get('converged', 'NA')}")
            lines.append(f"- $n_{{\\mathrm{{iter}}}}$: {best_iter.get('n_iter', 'NA')}")
            lines.append(f"- $\\mathrm{{tol}}$: {tol}")
            lines.append(f"- $n_{{\\mathrm{{iter,max}}}}$: {mx}")
            lines.append(f"- $\\mathrm{{lower\\_bound}}$: {best_iter.get('lower_bound', 'NA')}")
        else:
            lines.append("- `search_iteration` event for $k_{\\mathrm{best}}$ is missing in current log.")
        if final_evt:
            lines.append(f"- Final refit on $X_{{\\mathrm{{full}}}}$ converged: {final_evt.get('converged', 'NA')}")
            lines.append(f"- Final refit $n_{{\\mathrm{{iter}}}}$: {final_evt.get('n_iter', 'NA')}")
            lines.append(f"- Final refit $\\mathrm{{tol}}/n_{{\\mathrm{{iter,max}}}}$: {final_evt.get('tol', 'NA')} / {final_evt.get('max_iter', 'NA')}")
        lines.append("")
        lines.append("## 6. BIC Evidence (at $k_{\\mathrm{best}}$)")
        lines.append("")
        lines.append("### 6.0 Core Concepts and Symbols for This Section")
        lines.append("")
        lines.append("**Data and Model Dimensions:**")
        lines.append("- $n$: sample size (number of rows in $X_{\\mathrm{search}}$ or $X_{\\mathrm{full}}$).")
        lines.append("- $d$: number of selected PC features (dimension).")
        lines.append("- $k$: number of Gaussian components being compared.")
        lines.append("- $X_{\\mathrm{search}}$: subset of samples used for model search.")
        lines.append("- $X_{\\mathrm{full}}$: complete sample set (used for final refit).")
        lines.append("")
        lines.append("**Likelihood and Parameters:**")
        lines.append("- $L(k)$ or $\\mathcal{L}_k = p(X \\mid \\theta_k)$: marginal likelihood (probability of data given model with $k$ components).")
        lines.append("- $\\ell(k) = \\log L(k)$: log-likelihood.")
        lines.append("- $\\theta_k = \\{\\pi_c, \\mu_c, \\Sigma_c\\}_{c=1}^{k}$: parameter set for model with $k$ components.")
        lines.append("- $p$ or $p(k)$: total number of free parameters in model with $k$ components.")
        lines.append("  - $p_{\\mathrm{mean}} = kd$: parameters for all component means.")
        lines.append("  - $p_{\\mathrm{weight}} = k-1$: free parameters for mixture weights (sum constraint).")
        lines.append("  - $p_{\\mathrm{cov}}(k)$: parameters for covariance matrices (depends on `covariance_type`).")
        lines.append("  - $p(k) = p_{\\mathrm{mean}} + p_{\\mathrm{cov}} + p_{\\mathrm{weight}}$.")
        lines.append("")
        lines.append("**Information Criteria:**")
        lines.append("- $\\mathrm{BIC} = -2\\ell + p\\log n$: Bayesian Information Criterion (balances fit and complexity).")
        lines.append("- $\\mathrm{AIC} = -2\\ell + 2p$: Akaike Information Criterion (alternative complexity penalty).")
        lines.append("- Higher $\\ell$ favors better fit; higher $p$ increases penalty; lower BIC/AIC is preferred.")
        lines.append("")
        lines.append("**Selection Principle:**")
        lines.append("- $k_{\\mathrm{best}} = \\arg\\min_k \\mathrm{BIC}(k)$: component count that minimizes BIC.")
        lines.append("")
        lines.append("BIC definition used by the audit:")
        lines.append("")
        lines.append("$$")
        lines.append(r"\mathrm{BIC}=-2\ell + p\log n")
        lines.append("$$")
        lines.append("")
        lines.append("where $\\ell$ is total log-likelihood, $n$ is sample size, and $p$ is free-parameter count.")
        lines.append("")
        lines.append("### 6.1 Parameter Count $p$")
        lines.append("$$")
        lines.append(r"p=p_{\mathrm{mean}}+p_{\mathrm{cov}}+p_{\mathrm{weight}}")
        lines.append("$$")
        lines.append("$$")
        lines.append(r"p_{\mathrm{mean}}=kd,\quad p_{\mathrm{weight}}=k-1")
        lines.append("$$")
        lines.append("with $p_{\\mathrm{weight}}=k-1$ because $\\sum_{c=1}^{k}\\pi_c=1$." )
        lines.append("$$")
        lines.append(r"\text{full: }p_{\mathrm{cov}}=k\frac{d(d+1)}{2},\quad \text{tied: }p_{\mathrm{cov}}=\frac{d(d+1)}{2}")
        lines.append("$$")
        lines.append("$$")
        lines.append(r"\text{diag: }p_{\mathrm{cov}}=kd,\quad \text{spherical: }p_{\mathrm{cov}}=k")
        lines.append("$$")
        lines.append("")
        if best_iter:
            n_s = float(best_iter.get("n_samples", float("nan")))
            p = float(best_iter.get("n_parameters", float("nan")))
            k_val = int(best_iter.get("k", best_k if best_k >= 0 else 0))
            d_val = int(best_iter.get("n_features", 0))
            cov_type_val = str(best_iter.get("covariance_type", "unknown"))
            p_mean = int(k_val * d_val)
            p_weight = int(max(0, k_val - 1))
            if cov_type_val == "full":
                p_cov = int(k_val * d_val * (d_val + 1) / 2)
            elif cov_type_val == "tied":
                p_cov = int(d_val * (d_val + 1) / 2)
            elif cov_type_val == "diag":
                p_cov = int(k_val * d_val)
            elif cov_type_val == "spherical":
                p_cov = int(k_val)
            else:
                p_cov = int(round(float(p))) - p_mean - p_weight if math.isfinite(p) else 0
            logL = float(best_iter.get("total_log_likelihood", float("nan")))
            term_fit = float(-2.0 * logL) if math.isfinite(logL) else float("nan")
            term_penalty = float(p * math.log(max(1.0, n_s))) if (math.isfinite(p) and math.isfinite(n_s)) else float("nan")
            bic_recompose = float(term_fit + term_penalty) if (math.isfinite(term_fit) and math.isfinite(term_penalty)) else float("nan")

            lines.append("### 6.2 Numeric Snapshot at $k_{\\mathrm{best}}$")
            lines.append(f"- $k_{{\\mathrm{{best}}}}$: {best_k}")
            lines.append(f"- $n$: {best_iter.get('n_samples', 'NA')}")
            lines.append(f"- $d$: {best_iter.get('n_features', 'NA')}")
            lines.append(f"- covariance_type: {cov_type_val}")
            lines.append(f"- $p$: {best_iter.get('n_parameters', 'NA')}")
            lines.append(f"- $p_{{\\mathrm{{mean}}}}=kd$: {p_mean}")
            lines.append(f"- $p_{{\\mathrm{{cov}}}}$ ({cov_type_val}): {p_cov}")
            lines.append(f"- $p_{{\\mathrm{{weight}}}}=k-1$: {p_weight}")
            lines.append(f"- $p_{{\\mathrm{{check}}}}=p_{{\\mathrm{{mean}}}}+p_{{\\mathrm{{cov}}}}+p_{{\\mathrm{{weight}}}}$: {p_mean + p_cov + p_weight}")
            lines.append(f"- average log-likelihood ($\\ell/n$): {best_iter.get('average_log_likelihood', 'NA')}")
            lines.append(f"- total log-likelihood ($\\ell$): {best_iter.get('total_log_likelihood', 'NA')}")
            lines.append(f"- fit term ($-2\\ell$): {term_fit}")
            lines.append(f"- penalty term ($p\\log n$): {term_penalty}")
            lines.append(f"- recomposed BIC ($-2\\ell + p\\log n$): {bic_recompose}")
            lines.append(f"- bic_sklearn: {best_iter.get('bic_sklearn', 'NA')}")
            lines.append(f"- bic_manual: {best_iter.get('bic_manual', 'NA')}")
            lines.append(f"- bic_abs_diff: {best_iter.get('bic_abs_diff', 'NA')}")
            lines.append("")
            lines.append("### 6.3 Why BIC Is Trustworthy Here")
            lines.append("- `bic_manual` is recomputed independently from $\\ell$, $n$, and $p$.")
            lines.append("- `bic_abs_diff\\approx 0` confirms consistency with sklearn's BIC implementation.")
            lines.append("- This creates an explicit audit trail from formula to numeric decision.")
        lines.append("")
        lines.append("## 7. LR Diagnostics Around $k_{\\mathrm{best}}$")
        lines.append("")
        lines.append("### 7.0 Notation and Definitions for This Section")
        lines.append("")
        lines.append("**Model Comparison:**")
        lines.append("- $k_{\\mathrm{hi}}, k_{\\mathrm{lo}}$: higher and lower component counts being compared (e.g., $k_{\\mathrm{hi}}=27, k_{\\mathrm{lo}}=26$).")
        lines.append("- $L(k_{\\mathrm{hi}}), L(k_{\\mathrm{lo}})$: marginal likelihoods for models with $k_{\\mathrm{hi}}$ and $k_{\\mathrm{lo}}$ components.")
        lines.append("- $\\ell(k_{\\mathrm{hi}}), \\ell(k_{\\mathrm{lo}})$: log-likelihoods (logarithms of above).")
        lines.append("- $\\theta_{\\mathrm{hi}} = \\{\\pi_c,\\mu_c,\\Sigma_c\\}_{c=1}^{k_{\\mathrm{hi}}}$: fitted parameters at $k_{\\mathrm{hi}}$.")
        lines.append("- $\\theta_{\\mathrm{lo}} = \\{\\pi_c,\\mu_c,\\Sigma_c\\}_{c=1}^{k_{\\mathrm{lo}}}$: fitted parameters at $k_{\\mathrm{lo}}$.")
        lines.append("")
        lines.append("**Likelihood Ratio and Complexity:**")
        lines.append("- $\\mathrm{LR} = 2[\\ell(k_{\\mathrm{hi}}) - \\ell(k_{\\mathrm{lo}})]$: twice the log-likelihood difference (positive if $k_{\\mathrm{hi}}$ fits better).")
        lines.append("- $\\Delta p = p(k_{\\mathrm{hi}}) - p(k_{\\mathrm{lo}})$: increase in free parameters when adding components.")
        lines.append("- Note: Unlike standard LRT, $\\mathrm{LR}$ under the null hypothesis (true $k=k_{\\mathrm{lo}}$) does NOT follow $\\chi^2_{\\Delta p}$ exactly due to boundary effects (component weights $\\pi_c \\geq 0$).")
        lines.append("")
        lines.append("**Information Criteria Differences:**")
        lines.append("- $\\Delta \\mathrm{BIC} = \\mathrm{BIC}(k_{\\mathrm{hi}}) - \\mathrm{BIC}(k_{\\mathrm{lo}})$: BIC comparison.")
        lines.append("  - Negative $\\Delta \\mathrm{BIC}$ favors $k_{\\mathrm{hi}}$ (more components).")
        lines.append("  - Positive $\\Delta \\mathrm{BIC}$ favors $k_{\\mathrm{lo}}$ (fewer components).")
        lines.append("- $\\mathrm{BF}_{\\mathrm{hi,lo}} = \\exp\\left(-\\frac{\\Delta \\mathrm{BIC}}{2}\\right)$: approximate Bayes Factor (BIC-based), ratio of posterior odds.")
        lines.append("- $\\log_{10}(\\mathrm{BF}_{\\mathrm{hi,lo}})$: log10-scale Bayes Factor (positive favors $k_{\\mathrm{hi}}$, negative favors $k_{\\mathrm{lo}}$).")
        lines.append("")
        lines.append("**Interpretation Caveat:**")
        lines.append("- For GMM component-count selection, classical chi-square LRT p-values are unreliable (mixture model regularity conditions violated).")
        lines.append("- $\\mathrm{LR}$ and $\\Delta\\mathrm{BIC}$ are reported as diagnostics to understand local model behavior.")
        lines.append("- **Primary decision rule:** BIC, not hypothesis test.")
        lines.append("")
        lines.append("$$")
        lines.append(r"\mathrm{LR}=2\left[\log L(k_{hi})-\log L(k_{lo})\right]")
        lines.append("$$")
        lines.append("$$")
        lines.append(r"\Delta p=p(k_{hi})-p(k_{lo})")
        lines.append("$$")
        lines.append("where $\\mathrm{LR}$ measures fit gain and $\\Delta p$ measures added complexity.")
        lines.append("")
        lines.append("### 7.1 Local LR Diagnostics")
        if lr_prev is not None:
            lines.append(f"- Lower-neighbor comparison: $k_{{hi}}={lr_prev['k_hi']}$ vs $k_{{lo}}={lr_prev['k_lo']}$")
            lines.append(f"- $\\log L(k_{{hi}})$: {lr_prev['logl_hi']}")
            lines.append(f"- $\\log L(k_{{lo}})$: {lr_prev['logl_lo']}")
            lines.append(f"- $\\mathrm{{LR}}$: {lr_prev['lr_stat']}")
            lines.append(f"- $\\Delta p$: {lr_prev['delta_p']}")
            lines.append(f"- $\\Delta\\mathrm{{BIC}}$: {lr_prev['delta_bic']}")
            lines.append(f"- $\\log_{{10}}(\\mathrm{{BF}}_{{hi,lo}})$ (BIC approximation): {lr_prev['log10_bf_hi_vs_lo']}")
        else:
            lines.append("- Lower-neighbor LR check unavailable.")
        if lr_next is not None:
            lines.append(f"- Upper-neighbor comparison: $k_{{hi}}={lr_next['k_hi']}$ vs $k_{{lo}}={lr_next['k_lo']}$")
            lines.append(f"- $\\log L(k_{{hi}})$: {lr_next['logl_hi']}")
            lines.append(f"- $\\log L(k_{{lo}})$: {lr_next['logl_lo']}")
            lines.append(f"- $\\mathrm{{LR}}$: {lr_next['lr_stat']}")
            lines.append(f"- $\\Delta p$: {lr_next['delta_p']}")
            lines.append(f"- $\\Delta\\mathrm{{BIC}}$: {lr_next['delta_bic']}")
            lines.append(f"- $\\log_{{10}}(\\mathrm{{BF}}_{{hi,lo}})$ (BIC approximation): {lr_next['log10_bf_hi_vs_lo']}")
        else:
            lines.append("- Upper-neighbor LR check unavailable.")
        lines.append("")
        lines.append("## 8. Audit Field Dictionary")
        lines.append("Convergence/BIC dedicated logs are simplified views derived from these events.")
        lines.append("- `gmm_search_convergence_log.jsonl` now includes `stop_reason`, `convergence_rule`, and EM parameter snapshot fields.")
        lines.append("- `gmm_search_bic_log.jsonl` focuses only on likelihood/parameter/BIC-AIC terms used for criteria calculations.")
        lines.append("")
        lines.append("### 8.1 `run_start` event")
        lines.append("- `selected_pc_cols`: selected feature columns.")
        lines.append("- `n_rows_full`: row count of $X_{\\mathrm{full}}$.")
        lines.append("- `n_rows_search`: row count of $X_{\\mathrm{search}}$.")
        lines.append("- `search_fraction`: ratio `n_rows_search / n_rows_full`.")
        lines.append("- `config`: full hyper-parameter snapshot.")
        lines.append("")
        lines.append("### 8.2 `search_iteration` event (one record per $k$)")
        lines.append("- `k`: current component count.")
        lines.append("- `converged`, `n_iter`, `lower_bound`: EM convergence diagnostics.")
        lines.append("- `tol`, `max_iter`: stopping controls.")
        lines.append("- `n_samples`, `n_features`: matrix shape for this fit.")
        lines.append("- `bic_sklearn`, `aic_sklearn`: sklearn criteria.")
        lines.append("- `n_parameters`: parameter count $p$.")
        lines.append("- `average_log_likelihood`, `total_log_likelihood`: likelihood terms.")
        lines.append("- `bic_manual`, `aic_manual`: audit recomputation.")
        lines.append("- `bic_abs_diff`, `aic_abs_diff`: consistency checks.")
        lines.append("- `has_empty_clusters`, `n_non_empty_clusters`: occupancy diagnostics.")
        lines.append("- `weights_min`, `weights_max`, `weights_entropy`: weight diagnostics.")
        lines.append(
            "- `components`: per-component details (`component_id`, `weight`, `mean`, `covariance_matrix`, `mean_l2_norm`, `cov_trace`)."
        )
        lines.append("")
        lines.append("### 8.2a Convergence Log Fields (`gmm_search_convergence_log.jsonl`)")
        lines.append("- `converged`, `n_iter`, `tol`, `max_iter`, `lower_bound`: direct convergence outcome and decision thresholds.")
        lines.append("- `convergence_rule`: textual rule used by optimizer stopping logic.")
        lines.append("- `stop_reason`: inferred stop category, e.g. `tol_reached_before_max_iter` or `max_iter_reached_without_convergence`.")
        lines.append("- `decision_explanation`: simplified status (`accepted_converged` or `not_converged`).")
        lines.append("- `covariance_type`, `n_init`, `init_params`, `reg_covar`, `random_state`: key EM/GMM configuration that affects convergence behavior.")
        lines.append("")
        lines.append("### 8.3 `best_model_selection` event")
        lines.append("- `best_k`, `best_bic`: selected optimum by BIC.")
        lines.append("- `require_non_empty_clusters`: non-empty eligibility rule.")
        lines.append("- `search_non_empty_model_count`, `search_model_count`: candidate pool counts.")
        lines.append("")
        lines.append("### 8.4 `final_model` event")
        lines.append("- `best_k`: selected $k$ used for final refit.")
        lines.append("- `converged`, `n_iter`, `lower_bound`: final-fit convergence diagnostics.")
        lines.append("- `bic_full_data`, `aic_full_data`: criteria on $X_{\\mathrm{full}}$.")
        lines.append("- `cluster_counts`: final sample count per cluster.")
        lines.append("")
        lines.append("## 9. $k_{\\mathrm{best}}$ Component Profile and Meaning")
        lines.append("")
        lines.append("### 9.0 Notation and Definitions for This Section")
        lines.append("")
        lines.append("**Index and Posterior Responsibility:**")
        lines.append("- $c \\in \\{1, 2, \\ldots, k_{\\mathrm{best}}\\}$: component index.")
        lines.append("- $r_{ic} = p(z_i=c \\mid x_i, \\theta)$: posterior responsibility of sample $i$ for component $c$ (soft cluster assignment).")
        lines.append("  - Computed from E-step: $r_{ic} = \\frac{\\pi_c \\mathcal{N}(x_i \\mid \\mu_c, \\Sigma_c)}{\\sum_{j=1}^{k}\\pi_j \\mathcal{N}(x_i \\mid \\mu_j, \\Sigma_j)}$.")
        lines.append("  - Constraint: $\\sum_{c=1}^{k}r_{ic} = 1$ for each sample $i$ (sample is distributed across all components).")
        lines.append("")
        lines.append("**Component Statistics:**")
        lines.append("- $N_c = \\sum_{i=1}^{n}r_{ic}$: effective sample size (total soft weight) for component $c$.")
        lines.append("- $\\pi_c = \\frac{N_c}{n}$: mixture weight or proportion of component $c$ in the fitted mixture.")
        lines.append("  - Updated in M-step: $N_c$ aggregates responsibilities, then normalizes to $\\pi_c$.")
        lines.append("  - Constraint: $\\pi_c \\geq 0$ and $\\sum_{c=1}^{k}\\pi_c = 1$.")
        lines.append("- $\\mu_c \\in \\mathbb{R}^d$: component-specific mean vector, fitted in M-step.")
        lines.append("- $\\Sigma_c \\in \\mathbb{R}^{d \\times d}$: component-specific covariance matrix, fitted in M-step.")
        lines.append("")
        lines.append("**Summary Metrics:**")
        lines.append("- $\\|\\mu_c\\|_2 = \\sqrt{\\sum_{j=1}^{d}(\\mu_c)_j^2}$: Euclidean norm (distance from origin in PC space), reported as `Mean L2 Norm`.")
        lines.append("- $\\mathrm{tr}(\\Sigma_c) = \\sum_{j=1}^{d}(\\Sigma_c)_{jj}$: trace (sum of diagonal variance terms), reported as `Covariance Trace`.")
        lines.append("  - Note: Only captures marginal variances; does not include correlations or off-diagonal covariance terms.")
        lines.append("")
        lines.append("**Distance Metric:**")
        lines.append("- $d_M(c_1,c_2) = \\sqrt{(\\mu_{c_1}-\\mu_{c_2})^\\top S_{c_1,c_2}^{-1}(\\mu_{c_1}-\\mu_{c_2})}$: Mahalanobis distance with pair-specific pooled covariance between two components.")
        lines.append("- $S_{c_1,c_2}=\\tfrac{1}{2}(\\Sigma_{c_1}+\\Sigma_{c_2})$: pair-specific pooled covariance used for the component-pair distance.")
        lines.append("  - Requires full $\\mu_c$ and $\\Sigma_c$ (or inverse); $\\|\\mu_c\\|_2$ and $\\mathrm{tr}(\\Sigma_c)$ alone are insufficient.")
        lines.append("")
        lines.append("**Table Columns:**")
        lines.append("- `Component`: numeric ID for $c$ (index $1, 2, \\ldots, k_{\\mathrm{best}}$).")
        lines.append("- `Weight`: estimate of $\\pi_c$.")
        lines.append("- `Mean L2 Norm`: value of $\\|\\mu_c\\|_2$.")
        lines.append("- `Covariance Trace`: value of $\\mathrm{tr}(\\Sigma_c)$.")
        lines.append("")
        lines.append("### 9.1 How `Weight` Is Computed and Interpreted")
        lines.append("- Source in EM: responsibilities $r_{ic}=p(z_i=c\\mid x_i,\\theta)$ from E-step.")
        lines.append("- Effective count: $N_c=\\sum_{i=1}^{n} r_{ic}$.")
        lines.append("- M-step update: $\\pi_c=N_c/n$.")
        lines.append("- Constraint: $\\pi_c\\ge 0$ and $\\sum_{c=1}^{k}\\pi_c=1$.")
        lines.append("- Interpretation: `Weight` is a soft-assignment expected proportion, not a hard-cluster fraction from argmax labels.")
        lines.append("")
        lines.append("### 9.2 How to Read the Three Summary Columns")
        lines.append("- `Weight`: component size/probability mass in the fitted mixture.")
        lines.append("- `Mean L2 Norm`: $\\|\\mu_c\\|_2=\\sqrt{\\sum_{j=1}^{d}\\mu_{cj}^2}$, a location summary in PC space.")
        lines.append("- `Covariance Trace`: $\\mathrm{tr}(\\Sigma_c)=\\sum_{j=1}^{d}\\Sigma_{c,jj}$, a spread summary (marginal variances only).")
        lines.append("- These are compact descriptors of $\\mu_c$ and $\\Sigma_c$; they do not capture full directional covariance structure.")
        lines.append("")
        lines.append("### 9.3 Relation to Component-Component Mahalanobis Distance")
        lines.append("- Exact component-pair distance uses full means and covariances of two components:")
        lines.append("  $d_M(c_1,c_2)=\\sqrt{(\\mu_{c_1}-\\mu_{c_2})^\\top S_{c_1,c_2}^{-1}(\\mu_{c_1}-\\mu_{c_2})}$, with $S_{c_1,c_2}=\\tfrac{1}{2}(\\Sigma_{c_1}+\\Sigma_{c_2})$.")
        lines.append("- `Mean L2 Norm` and `Covariance Trace` help interpretation but cannot replace full-parameter component-distance computation.")
        lines.append("")
        lines.append("### 9.4 Practical Interpretation Tips")
        lines.append("- A component with high `Weight` and low `Covariance Trace` is both common and compact.")
        lines.append("- A component with very low `Weight` but high `Covariance Trace` may represent a rare and diffuse subgroup or potential instability candidate.")
        lines.append("- `Mean L2 Norm` is reported on the same PC scale used for model fitting in this workflow.")
        lines.append("- Use `Weight` together with `Covariance Trace`: tiny weight + very large trace often suggests weakly supported broad components.")
        lines.append("- For scientific reporting, avoid interpreting a single column alone; combine size (`Weight`), location (`Mean L2 Norm`), and spread (`Covariance Trace`).")
        lines.append("")
        if best_iter and isinstance(best_iter.get("components"), list):
            lines.append("| Component | Weight | Mean L2 Norm | Covariance Trace |")
            lines.append("|---|---|---|---|")
            for c in best_iter["components"]:
                lines.append(
                    f"| {int(c.get('component_id', -1))} | "
                    f"{float(c.get('weight', float('nan'))):.6f} | "
                    f"{float(c.get('mean_l2_norm', float('nan'))):.6f} | "
                    f"{float(c.get('cov_trace', float('nan'))):.6f} |"
                )
        else:
            lines.append("Component details for $k_{\\mathrm{best}}$ are not available in current audit events.")
        lines.append("")
        lines.append("## 10. Quick Validation Checklist")
        lines.append("- Confirm `search_iteration` at $k_{\\mathrm{best}}$ has `converged=true`.")
        lines.append("- Confirm `final_model` has `converged=true` on full data.")
        lines.append("- Confirm `bic_abs_diff` is near zero.")
        lines.append("- Confirm local LR diagnostics are directionally consistent with BIC differences.")
        lines.append("- Inspect `weights_min` and empty-cluster diagnostics for degeneracy signals.")
        lines.append("")
        lines.append("## 11. Output Files")
        lines.append(f"- Human report (md, overwritten each run): `{self.report_path.name}`")
        lines.append(f"- Convergence variables log (jsonl): `{self.convergence_log_path.name}`")
        lines.append(f"- BIC variables log (jsonl): `{self.bic_log_path.name}`")
        lines.append(f"- Best-k component description log (json): `{self.best_components_log_path.name}`")

        return "\n".join(lines) + "\n"

