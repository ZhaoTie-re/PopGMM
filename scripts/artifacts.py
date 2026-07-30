"""Content-addressed cache for expensive pipeline steps.

Motivation: the fitted ``GaussianMixture`` lives only in the notebook variable
``gmm_model`` and is never persisted, yet STEP3, STEP3_tmp, STEP4, STEP4_tmp
and STEP5 all need it. Restarting the kernel therefore forces a full refit of
the ``k = 2..100`` BIC search (~5 minutes across 6 processes). Everything else
that matters is already on disk as TSV and round-trips exactly, so the model is
the one genuinely unpersisted artifact -- but caching STEP0 and STEP1 alongside
it makes a warm restart near-instant.

Design constraints, in order of importance:

1. **It must not be able to change a number.** Nothing here is imported by the
   step modules; the cache is applied at the call site in the notebook, so the
   numeric code paths are untouched. A cache hit returns the identical object
   graph that the function returned, or it is a miss.
2. **A stale hit must be impossible.** The key covers the step name, the full
   config, the keys of the upstream steps, fingerprints of the DataFrame and
   file inputs, and the versions of every library that could move a float. Any
   change misses.
3. **A hit must never be mistaken for a run.** Hits skip the function, so the
   step's side-effect files (tables, figures, audit logs) are *not* written.
   ``RUN_MODE="fresh"`` -- the default -- therefore always executes, and only
   populates the cache. ``"resume"`` is a development convenience, is recorded
   in ``run_environment.json``, and ``tools/verify_results.py`` refuses to
   verify a tree produced with it.

   This is not a theoretical concern. A measured resume run leaves all 40 data
   files it writes byte-identical -- every keep-list included -- but shifts
   3.4 % of the pixels in ``gmm_component_merging_overview.png``, because
   skipping STEP1/STEP2 means their plotting code never runs and later steps
   inherit a different global ``rcParams`` state. Never publish figures from a
   resume run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

DEFAULT_CACHE_ROOT = Path(".cache/popgmm")

# Libraries whose version could plausibly move a floating-point result. A
# version bump in any of them must invalidate every cached artifact.
_VERSIONED = ("numpy", "pandas", "scikit-learn", "scipy", "hdbscan", "joblib")


def library_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in _VERSIONED:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "absent"
    return out


def run_environment(run_mode: str) -> dict[str, Any]:
    """Provenance blob for ``<RESULTS_ROOT>/run_environment.json``.

    ``gmm_search_audit`` collects this kind of information into an event it
    never writes, so before this the committed results had no version record.
    """
    import os

    from scripts.hdbscan_filtering import HDBSCAN_BACKEND

    return {
        "run_mode": run_mode,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
        # Which clustering implementation actually ran. hdbscan_summary.json
        # echoes config values rather than what the estimator received, so
        # without this there was no record of the backend at all.
        "hdbscan_backend": HDBSCAN_BACKEND,
        "threading": {
            var: os.environ.get(var)
            for var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _normalize_roots(value: Any) -> Any:
    """Collapse the results/data root prefix in any path-like config value.

    Step configs carry ``output_dir``, which moves when ``POPGMM_RESULTS_ROOT``
    is set -- so without this, a verification run into ``results_verify`` would
    miss every cache entry a run into ``results`` had populated, purely because
    of where the *side-effect* files go. Where output is written cannot affect
    what a step returns: the step output NamedTuples carry no paths and the
    summary dicts contain no path keys (both checked).

    The prefix is replaced rather than dropped, so a genuine change of
    subdirectory -- ``03_gmm_component_merging`` vs its ``STEP3_tmp`` -- still
    produces a different key.
    """
    import scripts.params as params

    if isinstance(value, Path) or (isinstance(value, str) and ("/" in value or "\\" in value)):
        text = Path(str(value)).as_posix()
        for root, token in (
            (params.RESULTS_ROOT.as_posix(), "<RESULTS_ROOT>"),
            (params.DATA_ROOT.as_posix(), "<DATA_ROOT>"),
        ):
            if root and text.startswith(root):
                return token + text[len(root):]
        return text
    if isinstance(value, dict):
        return {k: _normalize_roots(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_roots(v) for v in value]
    return value


def _fingerprint_config(config: Any) -> Any:
    if config is None:
        return None
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return _normalize_roots(dataclasses.asdict(config))
    return _normalize_roots(config)


def _fingerprint_frame(df: pd.DataFrame) -> str:
    """Hash values, column names and dtypes -- a renamed column is a new input."""
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(json.dumps([str(t) for t in df.dtypes]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def _fingerprint_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprint_any(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return {"frame": _fingerprint_frame(obj)}
    if isinstance(obj, pd.Series):
        return {"frame": _fingerprint_frame(obj.to_frame())}
    if isinstance(obj, np.ndarray):
        return {"array": hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()}
    if isinstance(obj, (list, tuple, set)):
        return [_fingerprint_any(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ArtifactCache:
    """Keyed joblib store for step outputs.

    ``mode`` is ``"fresh"`` (always execute, then store) or ``"resume"``
    (return a stored result when the key matches).
    """

    def __init__(
        self,
        mode: str = "fresh",
        root: Path | str = DEFAULT_CACHE_ROOT,
        verbose: bool = True,
    ) -> None:
        if mode not in {"fresh", "resume"}:
            raise ValueError(f"mode must be 'fresh' or 'resume', got {mode!r}")
        self.mode = mode
        self.root = Path(root)
        self.verbose = verbose
        self._keys: dict[str, str] = {}

    # -- key construction ---------------------------------------------------

    def key(
        self,
        name: str,
        *,
        config: Any = None,
        upstream: Sequence[str] = (),
        frames: Iterable[Any] = (),
        files: Iterable[Path | str] = (),
    ) -> str:
        missing = [u for u in upstream if u not in self._keys]
        if missing:
            raise KeyError(
                f"step {name!r} declares upstream {missing} which has not run in this session"
            )
        material = {
            "step": name,
            "config": _fingerprint_config(config),
            "upstream": [self._keys[u] for u in upstream],
            "frames": [_fingerprint_any(f) for f in frames],
            "files": [_fingerprint_file(Path(p)) for p in files],
            "libraries": library_versions(),
        }
        blob = json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _paths(self, name: str, key: str) -> tuple[Path, Path]:
        base = self.root / name
        return base / f"{key}.joblib", base / f"{key}.meta.json"

    # -- main entry point ---------------------------------------------------

    def compute(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        config: Any = None,
        upstream: Sequence[str] = (),
        frames: Iterable[Any] = (),
        files: Iterable[Path | str] = (),
        writes_side_effects: bool = True,
    ) -> Any:
        """Run ``fn`` (or return its cached result) and register the step key.

        ``fn`` takes no arguments -- pass a lambda closing over the real call.
        That keeps this module ignorant of every step's signature, which is why
        no step module needs to be modified.
        """
        key = self.key(name, config=config, upstream=upstream, frames=frames, files=files)
        self._keys[name] = key
        blob_path, meta_path = self._paths(name, key)

        if self.mode == "resume" and blob_path.exists():
            result = joblib.load(blob_path)
            if self.verbose:
                note = (
                    "  (side-effect files NOT rewritten)" if writes_side_effects else ""
                )
                print(f"[cache] hit  {name} {key[:12]}{note}")
            return result

        if self.verbose:
            why = "fresh mode" if self.mode == "fresh" else "miss"
            print(f"[cache] run  {name} {key[:12]} ({why})")
        started = time.time()
        result = fn()
        elapsed = time.time() - started

        try:
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, blob_path, compress=3)
            meta_path.write_text(
                json.dumps(
                    {
                        "step": name,
                        "key": key,
                        "elapsed_seconds": round(elapsed, 3),
                        "upstream": list(upstream),
                        "config": _fingerprint_config(config),
                        "libraries": library_versions(),
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            if self.verbose:
                size = blob_path.stat().st_size / 1048576
                print(f"[cache] save {name} {key[:12]} ({elapsed:.1f}s, {size:.1f} MB)")
        except Exception as exc:
            # A cache write failure must never fail the pipeline.
            if self.verbose:
                print(f"[cache] WARN could not store {name}: {type(exc).__name__}: {exc}")

        return result
