"""Fit the reference-panel mixture and select the component count by BIC.

Searches a range of component counts, keeping the minimum-BIC model that leaves
no component empty, then refits on the full denoised panel. Search fits run
across a process pool; the per-fit audit trail is written by ``gmm_search_audit``.

Computation is float64. Under float32 the log-likelihood sum over ~180k samples
was sensitive to BLAS reduction order, so the search log differed between runs
even with the seed fixed, and with several EM restarts a last-bit difference
could flip which restart won.

Inputs
------
Denoised reference panel; eigenvalues for axis labels.

Outputs
-------
Clustered sample table, BIC search table, per-component summary, run summary
JSON, an overview figure, and the audit logs under ``tmp/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import time

from scripts.gmm_search_audit import GMMAuditLogger

from scripts.plotting.mixture import plot_mixture_overview
from scripts.plotting.style import THEME_MIXTURE, figure_context, save_figure
from scripts.common import (
    COMPUTE_DTYPE,
    STORE_DTYPE,
    resolve_pc_columns as _resolve_pc_columns,
)


class GMMClusteringOutput(NamedTuple):
    """Container for fixed-PC GMM clustering results."""

    labels: np.ndarray
    probabilities: np.ndarray
    bic_table: pd.DataFrame
    reference_samples_with_cluster: pd.DataFrame
    cluster_summary: pd.DataFrame
    selected_pc_cols: list[str]
    summary: dict[str, Any]
    model: GaussianMixture


@dataclass(frozen=True)
class GMMConfig:
    """Configuration for fixed-PC GMM clustering.

    Parameters
    ----------
    fixed_n_pcs : int
        Number of leading PCs used for both search and final model.
    k_min : int
        Minimum number of mixture components to evaluate.
    k_max : int
        Maximum number of mixture components to evaluate (inclusive).
    use_zscale : bool
        Whether to z-score selected PCs before fitting.
    covariance_type : str
        Covariance type for GaussianMixture.
    n_init : int
        Number of initializations for each GMM fit.
    init_params : str
        Initialization method for GaussianMixture.
    reg_covar : float
        Non-negative regularization added to covariance diagonal.
    max_iter : int
        Maximum number of EM iterations for each fit.
    random_state : int
        Random seed for reproducibility.
    search_max_samples : int
        Maximum samples used during BIC search. Final fit still uses all rows.
    search_workers : int
        Number of parallel workers used during BIC search (futures-based).
        Set to 1 to disable parallelism.
    require_non_empty_clusters : bool
        If True, select best k from models without empty clusters only.
    output_dir : str
        Directory for plot/table/json outputs.
    save_plot : bool
        Save BIC curve + clustering overview plot.
    save_tables : bool
        Save tables and JSON summary.
    verbose : bool
        Print structured logs.
    """

    fixed_n_pcs: int = 5
    k_min: int = 2
    k_max: int = 15
    use_zscale: bool = False
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    n_init: int = 10
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "k-means++"
    reg_covar: float = 1e-7
    max_iter: int = 500
    random_state: int = 42
    search_max_samples: int = 60000
    search_workers: int = 4
    require_non_empty_clusters: bool = True
    output_dir: str | Path = "results/01_reference_model/mixture_model"
    save_plot: bool = True
    save_tables: bool = True
    verbose: bool = True


_GMM_SEARCH_X: np.ndarray | None = None
_GMM_SEARCH_PARAMS: dict[str, Any] | None = None


def _init_gmm_search_worker(x_search: np.ndarray, params: dict[str, Any]) -> None:
    """Initializer for ProcessPool workers to avoid re-pickling x_search per task."""

    global _GMM_SEARCH_X, _GMM_SEARCH_PARAMS
    _GMM_SEARCH_X = x_search
    _GMM_SEARCH_PARAMS = params


def _fit_gmm_search_for_k(k: int) -> dict[str, Any]:
    """Fit one GMM for a given k using globals set by _init_gmm_search_worker."""

    if _GMM_SEARCH_X is None or _GMM_SEARCH_PARAMS is None:
        raise RuntimeError("GMM search worker not initialized. This is a bug.")

    x_search = _GMM_SEARCH_X
    params = _GMM_SEARCH_PARAMS

    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=int(k),
        covariance_type=cast(Literal["full", "tied", "diag", "spherical"], params["covariance_type"]),
        random_state=int(params["random_state"]),
        n_init=int(params["n_init"]),
        init_params=cast(
            Literal["kmeans", "k-means++", "random", "random_from_data"],
            params["init_params"],
        ),
        reg_covar=float(params["reg_covar"]),
        max_iter=int(params["max_iter"]),
    )
    model.fit(x_search)
    labels_search = model.predict(x_search).astype(np.int32, copy=False)
    n_non_empty = int(np.unique(labels_search).shape[0])
    has_empty = n_non_empty < int(k)
    bic = float(model.bic(x_search))
    aic = float(model.aic(x_search))
    elapsed = float(time.perf_counter() - t0)

    return {
        "n_components": int(k),
        "bic": float(bic),
        "aic": float(aic),
        "has_empty_clusters": bool(has_empty),
        "n_non_empty_clusters": int(n_non_empty),
        "elapsed_seconds": float(elapsed),
        "model": model,
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    means = x.mean(axis=0, dtype=np.float64)
    stds = x.std(axis=0, dtype=np.float64)
    stds[stds == 0] = 1.0
    return ((x - means) / stds).astype(COMPUTE_DTYPE)


def _build_gmm(n_components: int, config: GMMConfig) -> GaussianMixture:
    """Create GMM model from one shared parameter set.

    This builder is used by both model search and final model extraction,
    ensuring strict parameter consistency.
    """

    return GaussianMixture(
        n_components=n_components,
        covariance_type=config.covariance_type,
        random_state=config.random_state,
        n_init=config.n_init,
        init_params=config.init_params,
        reg_covar=config.reg_covar,
        max_iter=config.max_iter,
    )


def run_gmm_fixed_pcs(
    reference_samples_filtered: pd.DataFrame,
    eigenval: pd.DataFrame | None = None,
    config: GMMConfig | None = None,
) -> GMMClusteringOutput:
    """Run fixed-PC GMM search and return clustering outputs.

    Workflow:
    1) Select fixed number of leading PCs.
    2) Optionally z-scale selected PCs.
    3) Search k in [k_min, k_max] using BIC.
    4) Choose model with minimum BIC (single supported strategy).
    5) Export summary/plot/tables.
    """

    config = config or GMMConfig()

    if reference_samples_filtered.empty:
        raise ValueError("bbj_samples_filtered is empty; cannot run GMM clustering.")

    pc_cols = _resolve_pc_columns(reference_samples_filtered)
    if not pc_cols:
        raise ValueError("No PC columns found in bbj_samples_filtered.")

    n_pcs = min(max(2, int(config.fixed_n_pcs)), len(pc_cols))
    selected_pc_cols = pc_cols[:n_pcs]

    x = reference_samples_filtered[selected_pc_cols].to_numpy(dtype=COMPUTE_DTYPE, copy=True)
    if config.use_zscale:
        x = _standardize(x)

    if config.search_max_samples < 1000:
        raise ValueError("search_max_samples should be >= 1000 for stable BIC search.")

    x_search = x
    if x.shape[0] > config.search_max_samples:
        rng = np.random.default_rng(config.random_state)
        idx = rng.choice(x.shape[0], size=int(config.search_max_samples), replace=False)
        x_search = x[idx]

    if config.k_min < 2 or config.k_max < config.k_min:
        raise ValueError("Invalid k range. Require 2 <= k_min <= k_max.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = GMMAuditLogger(
        output_dir=output_dir,
        config_dict={
            "fixed_n_pcs": int(config.fixed_n_pcs),
            "k_min": int(config.k_min),
            "k_max": int(config.k_max),
            "use_zscale": bool(config.use_zscale),
            "covariance_type": str(config.covariance_type),
            "n_init": int(config.n_init),
            "init_params": str(config.init_params),
            "reg_covar": float(config.reg_covar),
            "max_iter": int(config.max_iter),
            "random_state": int(config.random_state),
            "search_max_samples": int(config.search_max_samples),
            "search_workers": int(config.search_workers),
            "require_non_empty_clusters": bool(config.require_non_empty_clusters),
        },
        selected_pc_cols=selected_pc_cols,
        n_rows_full=int(x.shape[0]),
        n_rows_search=int(x_search.shape[0]),
        random_state=int(config.random_state),
        use_zscale=bool(config.use_zscale),
    )

    k_values = list(range(int(config.k_min), int(config.k_max) + 1))
    search_rows: list[dict[str, Any]] = []

    use_parallel = int(getattr(config, "search_workers", 1)) > 1 and len(k_values) > 1
    if use_parallel:
        search_params = {
            "covariance_type": str(config.covariance_type),
            "random_state": int(config.random_state),
            "n_init": int(config.n_init),
            "init_params": str(config.init_params),
            "reg_covar": float(config.reg_covar),
            "max_iter": int(config.max_iter),
        }

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=int(config.search_workers),
            initializer=_init_gmm_search_worker,
            initargs=(x_search, search_params),
        ) as ex:
            fut_map = {ex.submit(_fit_gmm_search_for_k, int(k)): int(k) for k in k_values}
            for fut in as_completed(fut_map):
                results.append(fut.result())

        # Keep deterministic output/log order.
        results.sort(key=lambda r: int(r["n_components"]))
        for row in results:
            k = int(row["n_components"])
            bic = float(row["bic"])
            aic = float(row["aic"])
            has_empty = bool(row["has_empty_clusters"])
            n_non_empty = int(row["n_non_empty_clusters"])
            elapsed = float(row["elapsed_seconds"])
            model = row["model"]

            search_rows.append(
                {
                    "n_components": int(k),
                    "n_pcs": int(n_pcs),
                    "bic": bic,
                    "aic": aic,
                    "has_empty_clusters": bool(has_empty),
                    "n_non_empty_clusters": int(n_non_empty),
                }
            )
            audit.log_search_iteration(
                k=int(k),
                model=model,
                x_used=x_search,
                bic=float(bic),
                aic=float(aic),
                elapsed_seconds=float(elapsed),
                has_empty_clusters=bool(has_empty),
                n_non_empty_clusters=int(n_non_empty),
            )
    else:
        for k in k_values:
            t0 = time.perf_counter()
            model = _build_gmm(n_components=k, config=config)
            model.fit(x_search)
            labels_search = model.predict(x_search).astype(np.int32, copy=False)
            n_non_empty = int(np.unique(labels_search).shape[0])
            has_empty = n_non_empty < int(k)
            bic = float(model.bic(x_search))
            aic = float(model.aic(x_search))
            elapsed = float(time.perf_counter() - t0)
            search_rows.append(
                {
                    "n_components": int(k),
                    "n_pcs": int(n_pcs),
                    "bic": bic,
                    "aic": aic,
                    "has_empty_clusters": bool(has_empty),
                    "n_non_empty_clusters": int(n_non_empty),
                }
            )
            audit.log_search_iteration(
                k=int(k),
                model=model,
                x_used=x_search,
                bic=float(bic),
                aic=float(aic),
                elapsed_seconds=elapsed,
                has_empty_clusters=bool(has_empty),
                n_non_empty_clusters=int(n_non_empty),
            )

    bic_table = pd.DataFrame(search_rows).sort_values("n_components").reset_index(drop=True)
    if config.require_non_empty_clusters:
        eligible_table = bic_table.loc[~bic_table["has_empty_clusters"]].copy()
        if eligible_table.empty:
            raise RuntimeError(
                "No non-empty-cluster model found in search range. "
                "Try reducing k_max or increasing search_max_samples."
            )
    else:
        eligible_table = bic_table

    eligible_bic = eligible_table["bic"].to_numpy(dtype=np.float64)
    eligible_k = eligible_table["n_components"].to_numpy(dtype=np.int32)
    best_pos = int(np.argmin(eligible_bic))
    best_k = int(eligible_k[best_pos])
    best_bic = float(eligible_bic[best_pos])
    audit.log_best_selection(
        bic_table=bic_table,
        best_k=int(best_k),
        best_bic=float(best_bic),
        require_non_empty_clusters=bool(config.require_non_empty_clusters),
    )

    # Final model uses full data with the exact same parameter set as search.
    t_final = time.perf_counter()
    final_model = _build_gmm(n_components=best_k, config=config)
    final_model.fit(x)
    labels = final_model.predict(x).astype(np.int32, copy=False)
    final_elapsed = float(time.perf_counter() - t_final)
    audit.log_final_model(
        model=final_model,
        x_full=x,
        labels=labels,
        best_k=int(best_k),
        elapsed_seconds=final_elapsed,
    )
    audit.finalize()

    probabilities_raw = final_model.predict_proba(x)
    probabilities = np.asarray(probabilities_raw, dtype=STORE_DTYPE)

    reference_samples_with_cluster = reference_samples_filtered.copy()
    reference_samples_with_cluster["GMM_Cluster"] = labels

    cluster_summary = (
        pd.Series(labels, name="GMM_Cluster")
        .value_counts()
        .sort_index()
        .rename_axis("GMM_Cluster")
        .reset_index(name="Samples")
    )
    cluster_summary["Share(%)"] = cluster_summary["Samples"] / float(cluster_summary["Samples"].sum()) * 100.0

    summary: dict[str, Any] = {
        "input_rows": int(x.shape[0]),
        "pc_columns_used": selected_pc_cols,
        "fixed_n_pcs": int(n_pcs),
        "k_min": int(config.k_min),
        "k_max": int(config.k_max),
        "best_k": int(best_k),
        "best_bic": float(best_bic),
        "covariance_type": config.covariance_type,
        "n_init": int(config.n_init),
        "init_params": config.init_params,
        "reg_covar": float(config.reg_covar),
        "max_iter": int(config.max_iter),
        "random_state": int(config.random_state),
        "use_zscale": bool(config.use_zscale),
        "search_rows": int(x_search.shape[0]),
        "full_rows": int(x.shape[0]),
        "search_max_samples": int(config.search_max_samples),
        "require_non_empty_clusters": bool(config.require_non_empty_clusters),
        "search_non_empty_model_count": int((~bic_table["has_empty_clusters"]).sum()),
    }

    if config.save_plot:
        with figure_context(THEME_MIXTURE):
            fig = plot_mixture_overview(
                x=x,
                labels=labels,
                probabilities=probabilities,
                bic_table=bic_table,
                best_k=best_k,
                selected_pc_cols=selected_pc_cols,
                eigenval=eigenval,
            )
            if fig is not None:
                save_figure(fig, output_dir / "mixture_model_overview.png", dpi=400)
                plt.close(fig)

    if config.save_tables:
        bic_table.to_csv(output_dir / "bic_search.tsv", sep="\t", index=False)
        reference_samples_with_cluster.to_csv(output_dir / "reference_samples_clustered.tsv", sep="\t", index=False)
        cluster_summary.to_csv(output_dir / "component_summary.tsv", sep="\t", index=False)
        with (output_dir / "mixture_model_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if config.verbose:
        print("\n" + "=" * 80)
        print("MIXTURE MODEL (FIXED PCs, MIN-BIC SELECTION)".center(80))
        print("=" * 80)
        print("\n[CONFIGURATION]")
        print("-" * 80)
        print(f"  fixed_n_pcs           : {n_pcs}")
        print(f"  k_range               : {config.k_min}..{config.k_max}")
        print(f"  covariance_type       : {config.covariance_type}")
        print(f"  n_init                : {config.n_init}")
        print(f"  init_params           : {config.init_params}")
        print(f"  reg_covar             : {config.reg_covar}")
        print(f"  max_iter              : {config.max_iter}")
        print(f"  random_state          : {config.random_state}")
        print(f"  search_rows           : {x_search.shape[0]:,}")
        print(f"  full_rows             : {x.shape[0]:,}")
        print(f"  search_workers        : {config.search_workers}")
        print(f"  require_non_empty     : {config.require_non_empty_clusters}")
        print(f"  use_zscale            : {config.use_zscale}")
        print(f"  output_dir            : {output_dir}")
        print("\n[RESULTS]")
        print("-" * 80)
        print(f"  input_rows            : {summary['input_rows']:,}")
        print(f"  best_k                : {summary['best_k']}")
        print(f"  best_bic              : {summary['best_bic']:,.2f}")
        print(f"  non_empty_models      : {summary['search_non_empty_model_count']}")
        print(f"  clusters_found        : {cluster_summary.shape[0]}")
        print("=" * 80)

    return GMMClusteringOutput(
        labels=labels,
        probabilities=probabilities,
        bic_table=bic_table,
        reference_samples_with_cluster=reference_samples_with_cluster,
        cluster_summary=cluster_summary,
        selected_pc_cols=selected_pc_cols,
        summary=summary,
        model=final_model,
    )
