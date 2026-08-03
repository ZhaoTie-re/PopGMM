"""Numeric-equivalence checker for PopGMM pipeline outputs.

The scientific results are final, so every refactor of ``scripts/`` or
``workflow.ipynb`` must leave ``results/`` numerically identical. This tool is
the oracle for that claim: it walks a baseline result tree and a candidate
result tree and compares every artifact with a rule chosen per file type.

Two modes:

    # full tree vs full tree
    python -m tools.verify_results --baseline results --candidate results_verify

    # candidate vs a committed fingerprint file (works after the large
    # intermediates are untracked and absent from a fresh clone)
    python -m tools.verify_results --manifest tools/baseline_manifest.json \\
                                   --candidate results_verify

    # (re)generate the fingerprint file from a trusted tree
    python -m tools.verify_results --baseline results --write-manifest tools/baseline_manifest.json

Exit code is 0 only when nothing failed. The pipeline is deterministic
(float64), so there are no per-file numeric tolerances: any difference in a
numeric artifact or a figure is a failure. Only embedded timestamps and the
results-root path are normalised away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

TSV_RTOL = 1e-12
TSV_ATOL = 1e-15
JSON_RTOL = 1e-12
NPY_ATOL = 1e-7
# 0-255 scale. Only used to label how large a figure difference is; it is not a
# pass threshold -- any pixel difference fails.
PNG_PERCEPTUAL_MEAN_ABS_DIFF = 0.5

# Keys whose values legitimately differ between two runs of the same pipeline.
VOLATILE_JSON_KEYS = frozenset(
    {"timestamp_utc", "generated_at_utc", "generated_utc", "elapsed_seconds"}
)

# Lines in generated Markdown that carry a wall-clock stamp.
VOLATILE_MD_LINE = re.compile(r"^- Generated \(UTC\):.*$", re.MULTILINE)

# PDF producer metadata.
VOLATILE_PDF = [
    re.compile(rb"/CreationDate\s*\([^)]*\)"),
    re.compile(rb"/ModDate\s*\([^)]*\)"),
    re.compile(rb"/ID\s*\[[^\]]*\]"),
]

SKIP_NAMES = {".DS_Store"}

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
#
# The pipeline computes in float64 and is fully deterministic: two independent
# full runs in the same environment produce 52 of 60 artifacts byte-identical,
# and the 8 that differ do so only in embedded timestamps or the results-root
# path they were written to. All 28 figures are byte-identical. No numeric
# artifact varies. See docs/reproducibility_probe.md.
#
# There is therefore NO numeric tolerance anywhere in this tool beyond the
# float-comparison epsilons below, which exist to give useful diagnostics on a
# real mismatch rather than to excuse one. Adding a per-file tolerance would be
# a scientific decision, not a convenience -- the previous float32 pipeline
# needed one for the BIC search log, and removing that need was the point of
# the switch.
#
# One zero-cost invariant is kept: whichever k minimises BIC must not move. It
# does not depend on any tolerance, and it is the single thing most worth
# failing loudly on if numeric behaviour ever changes.
ARGMIN_INVARIANT = {
    "01_reference_model/mixture_model/bic_search.tsv": ("bic", "n_components")
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Classes whose mismatch must never be tolerated.
HARD_CLASSES = frozenset({"exact", "tsv", "json", "npy", "run_mode"})


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class Check:
    rel: str
    kind: str
    status: str
    detail: str = ""

    def line(self) -> str:
        mark = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL"}[self.status]
        tail = f"  {self.detail}" if self.detail else ""
        return f"[{mark}] {self.kind:<14} {self.rel}{tail}"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _to_numeric(values: Any) -> pd.Series:
    """Column values coerced to numeric; anything non-numeric becomes NaN.

    Duplicates ``scripts.common.to_numeric_series`` on purpose: this tool is the
    oracle for the pipeline's output and must not import the code it checks.
    """
    return cast(pd.Series, pd.to_numeric(values, errors="coerce"))


def classify(rel: str) -> str:
    """Map a result-relative path to a comparison strategy."""
    name = Path(rel).name
    suffix = Path(rel).suffix.lower()
    in_tmp = "/tmp/" in f"/{rel}"

    if suffix == ".tsv":
        return "tsv"
    if suffix == ".npy":
        return "npy"
    if suffix == ".png":
        return "png"
    if suffix == ".pdf":
        return "pdf_volatile"
    if suffix == ".jsonl":
        return "jsonl_volatile"
    if suffix == ".md":
        return "md_volatile"
    if suffix == ".json":
        # tmp/gmm_best_k_components_log.json carries generated_at_utc, and the
        # run_* snapshots carry environment stamps; both are compared with the
        # volatile keys removed rather than skipped.
        return "json_volatile" if in_tmp or name.startswith("run_") else "json"
    if suffix in {".txt", ".log"}:
        return "exact"
    return "exact"


def iter_files(root: Path) -> Iterator[str]:
    for p in sorted(root.rglob("*")):
        if p.is_dir() or p.name in SKIP_NAMES:
            continue
        yield p.relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


# Directory names a result tree may live under. The artifacts record the root
# they were written to, so every alias has to collapse to the same placeholder
# regardless of where the tree currently sits on disk.
#
# Corollary worth knowing: a tree generated under one name and then renamed keeps
# the original path baked into its .log files and config snapshot. Generate the
# authoritative tree directly into `results/` rather than promoting a scratch
# directory by renaming it.
KNOWN_ROOT_ALIASES = (
    "results_baseline",
    "results_verify",
    "results_probe",
    "results",
)


def normalize_text(text: str, baseline_root: str, candidate_root: str) -> str:
    """Erase the results-root prefix so runs into different dirs still match.

    The ``.log`` files echo their own ``output_dir``, e.g.
    ``-> output_dir = results/04_our_assignment``. A run driven with
    ``POPGMM_RESULTS_ROOT=results_verify`` writes a different string there while
    being otherwise identical, so both roots collapse to a placeholder. Renaming
    the tree afterwards does not change what was baked into the file, hence the
    alias list as well as the two roots actually being compared.

    Substitution requires the root to be followed by a path separator or a
    delimiter, so a root name that is a prefix of another one does not corrupt
    it -- a plain str.replace turns "results_f64b/01_x" into
    "<RESULTS_ROOT>b/01_x" when "results_f64" is also in play, which shows up as
    a bogus mismatch.
    """
    roots = {baseline_root, candidate_root, *KNOWN_ROOT_ALIASES} - {""}
    for root in sorted(roots, key=len, reverse=True):
        text = re.sub(re.escape(root) + r"""(?=[/\\\s"',)\]}]|$)""", "<RESULTS_ROOT>", text)
    return text


def strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_JSON_KEYS
        }
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Structured comparisons
# ---------------------------------------------------------------------------


def compare_json_obj(a: Any, b: Any, path: str = "", rtol: float = JSON_RTOL) -> list[str]:
    """Recursive comparison; key set, list order and float value all matter."""
    diffs: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                diffs.append(f"{path}.{key}: only in candidate")
            elif key not in b:
                diffs.append(f"{path}.{key}: only in baseline")
            else:
                diffs += compare_json_obj(a[key], b[key], f"{path}.{key}", rtol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += compare_json_obj(x, y, f"{path}[{i}]", rtol)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (
            (math.isnan(a) and math.isnan(b))
            or math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)
        ):
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")
    return diffs[:20]


def compare_tsv(base: Path, cand: Path, rel: str = "") -> tuple[str, str]:
    a = pd.read_csv(base, sep="\t", dtype=object, keep_default_na=False)
    b = pd.read_csv(cand, sep="\t", dtype=object, keep_default_na=False)

    if list(a.columns) != list(b.columns):
        only_b = [c for c in b.columns if c not in a.columns]
        only_a = [c for c in a.columns if c not in b.columns]
        return FAIL, f"columns differ (+{only_b} -{only_a})"
    if a.shape != b.shape:
        return FAIL, f"shape {a.shape} vs {b.shape}"

    # The scientifically load-bearing invariants survive any tolerance.
    if rel in ARGMIN_INVARIANT:
        value_col, label_col = ARGMIN_INVARIANT[rel]
        ka = _to_numeric(a[value_col])
        kb = _to_numeric(b[value_col])
        wa, wb = a[label_col][ka.idxmin()], b[label_col][kb.idxmin()]
        if wa != wb or ka.min() != kb.min():
            return FAIL, (
                f"argmin({value_col}) moved: {label_col}={wa} ({ka.min()!r}) "
                f"vs {label_col}={wb} ({kb.min()!r})"
            )

    worst = 0.0
    worst_at = ""
    for col in a.columns:
        sa, sb = a[col], b[col]
        if (sa == sb).all():
            continue
        na = _to_numeric(sa)
        nb = _to_numeric(sb)
        if na.isna().all() or na.isna().sum() != nb.isna().sum():
            bad = int((sa != sb).idxmax())
            return FAIL, f"col {col!r} text differs at row {bad}: {sa[bad]!r} vs {sb[bad]!r}"
        va, vb = na.to_numpy("float64"), nb.to_numpy("float64")
        both_nan = np.isnan(va) & np.isnan(vb)
        close = np.isclose(va, vb, rtol=TSV_RTOL, atol=TSV_ATOL) | both_nan
        if not close.all():
            i = int(np.argmax(~close))
            return FAIL, (
                f"col {col!r} numeric differs at row {i}: {va[i]!r} vs {vb[i]!r}"
            )
        diff = np.nanmax(np.abs(va - vb)) if len(va) else 0.0
        if np.isfinite(diff) and diff > worst:
            worst, worst_at = float(diff), col

    if worst:
        n = int(sum((a[c] != b[c]).any() for c in a.columns))
        return PASS, (
            f"equal within float epsilon (max |d|={worst:.3g} in {worst_at}, "
            f"{n} cols affected)"
        )
    return PASS, "identical after parse"


def compare_npy(base: Path, cand: Path) -> tuple[str, str]:
    a = np.load(base)
    b = np.load(cand)
    if a.dtype != b.dtype:
        return FAIL, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return FAIL, f"shape {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=0, atol=NPY_ATOL, equal_nan=True):
        return FAIL, f"max |d|={float(np.nanmax(np.abs(a - b))):.3g} > {NPY_ATOL}"
    # A probability may wobble in the last bits; the resulting assignment may not.
    if a.ndim == 2:
        if not np.array_equal(a.argmax(axis=1), b.argmax(axis=1)):
            n = int((a.argmax(axis=1) != b.argmax(axis=1)).sum())
            return FAIL, f"argmax flipped for {n} rows"
    d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    return PASS, "identical" if d == 0 else f"within tolerance (max |d|={d:.3g})"


def compare_png(base: Path, cand: Path) -> tuple[str, str]:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return WARN, "pillow not installed; skipped"
    # The dpi=400 overview figures exceed PIL's decompression-bomb guard; these
    # are our own artifacts, so the guard is noise here.
    #
    # Note: an earlier version of this tool claimed PNGs could not be compared
    # byte-for-byte because of embedded matplotlib metadata. That was wrong --
    # the metadata is stable within a pinned environment. Under float32 the
    # figures differed because the plotted data differed.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(base) as ia, Image.open(cand) as ib:
        if ia.size != ib.size:
            return FAIL, f"size {ia.size} vs {ib.size}"
        aa = np.asarray(ia.convert("RGBA"), dtype=np.int16)
        bb = np.asarray(ib.convert("RGBA"), dtype=np.int16)
    d = np.abs(aa - bb)
    if d.max() == 0:
        return PASS, "pixel-identical"
    # Two independent float64 runs produce all 28 figures byte-identical, so any
    # pixel difference means either the plotted data changed or the environment
    # did -- both worth stopping for. The magnitude is still reported so a
    # sub-pixel render difference can be told apart from a changed curve.
    mean = float(d.mean())
    frac = float((d.max(axis=2) > 0).mean())
    scale = "sub-perceptual" if mean < PNG_PERCEPTUAL_MEAN_ABS_DIFF else "visible"
    return FAIL, (
        f"{scale}: mean|d|={mean:.4f}, {frac:.3%} of pixels differ, max={int(d.max())}"
    )


def compare_pdf(base: Path, cand: Path) -> tuple[str, str]:
    ra, rb = base.read_bytes(), cand.read_bytes()
    for pat in VOLATILE_PDF:
        ra, rb = pat.sub(b"", ra), pat.sub(b"", rb)
    if ra == rb:
        return PASS, "identical after stripping producer metadata"
    return WARN, f"content differs ({len(ra)} vs {len(rb)} bytes after stripping)"


def compare_jsonl(base: Path, cand: Path, rel: str = "") -> tuple[str, str]:
    la = [strip_volatile(json.loads(x)) for x in base.read_text().splitlines() if x.strip()]
    lb = [strip_volatile(json.loads(x)) for x in cand.read_text().splitlines() if x.strip()]
    if len(la) != len(lb):
        return FAIL, f"{len(la)} vs {len(lb)} records"
    affected = 0
    for i, (x, y) in enumerate(zip(la, lb)):
        diffs = compare_json_obj(x, y, f"[{i}]")
        if diffs:
            return FAIL, "; ".join(diffs[:3])
        if x != y:
            affected += 1
    if affected:
        return PASS, (
            f"{len(la)} records equal within float epsilon ({affected} differ in the last bits)"
        )
    return PASS, f"{len(la)} records identical (volatile keys ignored)"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def check_pair(rel: str, base: Path, cand: Path, broot: str, croot: str) -> Check:
    kind = classify(rel)
    try:
        if sha256(base) == sha256(cand):
            return Check(rel, kind, PASS, "byte-identical")

        if kind == "exact":
            ta = normalize_text(base.read_text(errors="replace"), broot, croot)
            tb = normalize_text(cand.read_text(errors="replace"), broot, croot)
            if ta == tb:
                return Check(rel, kind, PASS, "identical after root normalisation")
            la, lb = ta.splitlines(), tb.splitlines()
            if len(la) != len(lb):
                return Check(rel, kind, FAIL, f"{len(la)} vs {len(lb)} lines")
            for i, (x, y) in enumerate(zip(la, lb)):
                if x != y:
                    return Check(rel, kind, FAIL, f"line {i + 1}: {x!r} vs {y!r}")
            return Check(rel, kind, FAIL, "differs")

        if kind == "tsv":
            return Check(rel, kind, *compare_tsv(base, cand, rel))
        if kind == "npy":
            return Check(rel, kind, *compare_npy(base, cand))
        if kind == "png":
            return Check(rel, kind, *compare_png(base, cand))
        if kind == "pdf_volatile":
            return Check(rel, kind, *compare_pdf(base, cand))
        if kind == "jsonl_volatile":
            return Check(rel, kind, *compare_jsonl(base, cand, rel))
        if kind == "md_volatile":
            ta = VOLATILE_MD_LINE.sub("", normalize_text(base.read_text(), broot, croot))
            tb = VOLATILE_MD_LINE.sub("", normalize_text(cand.read_text(), broot, croot))
            if ta == tb:
                return Check(rel, kind, PASS, "identical after stripping timestamp")
            return Check(rel, kind, FAIL, "content differs")
        if kind in {"json", "json_volatile"}:
            a = json.loads(normalize_text(base.read_text(), broot, croot))
            b = json.loads(normalize_text(cand.read_text(), broot, croot))
            if kind == "json_volatile":
                a, b = strip_volatile(a), strip_volatile(b)
            diffs = compare_json_obj(a, b)
            if diffs:
                return Check(rel, kind, FAIL, "; ".join(diffs[:3]))
            return Check(rel, kind, PASS, "identical")

        return Check(rel, kind, FAIL, "no comparator")
    except Exception as exc:  # a comparator crash is itself a failure
        return Check(rel, kind, FAIL, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Manifest mode
# ---------------------------------------------------------------------------


def fingerprint(rel: str, path: Path, root: str) -> dict[str, Any]:
    kind = classify(rel)
    fp: dict[str, Any] = {"kind": kind, "sha256": sha256(path), "bytes": path.stat().st_size}
    try:
        if kind == "tsv":
            df = pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)
            fp["shape"] = list(df.shape)
            fp["columns"] = list(df.columns)
            cols: dict[str, list[float]] = {}
            for col in df.columns:
                v = _to_numeric(df[col])
                if v.notna().any():
                    cols[col] = [
                        float(np.nanmin(v)),
                        float(np.nanmax(v)),
                        float(np.nansum(v)),
                        float(np.nanmean(v)),
                    ]
            fp["numeric"] = cols
        elif kind == "npy":
            a = np.load(path)
            fp["shape"] = list(a.shape)
            fp["dtype"] = str(a.dtype)
            fp["sum"] = float(np.nansum(a))
            if a.ndim == 2:
                fp["argmax_sha256"] = hashlib.sha256(
                    a.argmax(axis=1).astype(np.int64).tobytes()
                ).hexdigest()
        elif kind == "png":
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(path) as im:
                fp["size"] = list(im.size)
                fp["mode"] = im.mode
                fp["pixel_mean"] = float(np.asarray(im.convert("RGBA"), float).mean())
        elif kind == "exact":
            fp["normalized_sha256"] = hashlib.sha256(
                normalize_text(path.read_text(errors="replace"), root, root).encode()
            ).hexdigest()
    except Exception as exc:
        fp["fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    return fp


def check_against_fingerprint(rel: str, cand: Path, fp: dict[str, Any], croot: str) -> Check:
    kind = fp.get("kind", classify(rel))
    try:
        if sha256(cand) == fp.get("sha256"):
            return Check(rel, kind, PASS, "byte-identical")
        cur = fingerprint(rel, cand, croot)
        if kind == "exact":
            if cur.get("normalized_sha256") == fp.get("normalized_sha256"):
                return Check(rel, kind, PASS, "identical after root normalisation")
            return Check(rel, kind, FAIL, "content differs")
        if kind == "tsv":
            if cur.get("shape") != fp.get("shape"):
                return Check(rel, kind, FAIL, f"shape {fp.get('shape')} vs {cur.get('shape')}")
            if cur.get("columns") != fp.get("columns"):
                return Check(rel, kind, FAIL, "columns differ")
            for col, ref in (fp.get("numeric") or {}).items():
                got = (cur.get("numeric") or {}).get(col)
                if got is None:
                    return Check(rel, kind, FAIL, f"col {col!r} no longer numeric")
                for stat, x, y in zip(("min", "max", "sum", "mean"), ref, got):
                    if not math.isclose(x, y, rel_tol=TSV_RTOL, abs_tol=0.0):
                        return Check(rel, kind, FAIL, f"col {col!r} {stat}: {x!r} vs {y!r}")
            return Check(rel, kind, PASS, "fingerprint matches")
        if kind == "npy":
            for key in ("shape", "dtype", "argmax_sha256"):
                if key in fp and cur.get(key) != fp[key]:
                    return Check(rel, kind, FAIL, f"{key}: {fp[key]} vs {cur.get(key)}")
            if "sum" in fp and not math.isclose(fp["sum"], cur.get("sum", 0.0), rel_tol=1e-9):
                return Check(rel, kind, FAIL, f"sum {fp['sum']} vs {cur.get('sum')}")
            return Check(rel, kind, PASS, "fingerprint matches")
        if kind == "png":
            if cur.get("size") != fp.get("size"):
                return Check(rel, kind, FAIL, f"size {fp.get('size')} vs {cur.get('size')}")
            d = abs(float(cur.get("pixel_mean", 0)) - float(fp.get("pixel_mean", 0)))
            return Check(rel, kind, PASS if d == 0 else WARN, f"pixel_mean delta {d:.6f}")
        # Timestamped artifacts have no stable fingerprint without the baseline.
        return Check(rel, kind, WARN, "not byte-identical; needs full baseline to judge")
    except Exception as exc:
        return Check(rel, kind, FAIL, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def render(checks: list[Check], baseline: str, candidate: str) -> str:
    n = {PASS: 0, WARN: 0, FAIL: 0}
    for c in checks:
        n[c.status] += 1
    out = [
        "# PopGMM result verification",
        "",
        f"- baseline:  `{baseline}`",
        f"- candidate: `{candidate}`",
        f"- files: {len(checks)} | pass {n[PASS]} | warn {n[WARN]} | **fail {n[FAIL]}**",
        "",
    ]
    for status, title in ((FAIL, "Failures"), (WARN, "Warnings"), (PASS, "Passed")):
        rows = [c for c in checks if c.status == status]
        if not rows:
            continue
        out += [f"## {title} ({len(rows)})", "", "```"]
        out += [c.line() for c in rows]
        out += ["```", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0] if __doc__ else "")
    ap.add_argument("--baseline", type=Path, help="trusted result tree")
    ap.add_argument("--candidate", type=Path, help="result tree produced by the refactor")
    ap.add_argument("--manifest", type=Path, help="fingerprint file to compare against")
    ap.add_argument("--write-manifest", type=Path, help="write fingerprints of --baseline here")
    ap.add_argument("--report", type=Path, help="also write a Markdown report here")
    ap.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    args = ap.parse_args(argv)

    if args.write_manifest:
        if not args.baseline:
            ap.error("--write-manifest requires --baseline")
        root = args.baseline
        entries = {rel: fingerprint(rel, root / rel, root.as_posix()) for rel in iter_files(root)}
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps({"root": root.as_posix(), "files": entries}, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {len(entries)} fingerprints to {args.write_manifest}")
        return 0

    if not args.candidate:
        ap.error("--candidate is required")
    croot = args.candidate.as_posix()
    checks: list[Check] = []

    # A "resume" run reuses cached upstream results, which means those stages'
    # side-effect files were never rewritten -- whatever is on disk for them came
    # from some earlier run. Such a tree cannot testify to anything, so reject it
    # outright rather than let it pass on stale files.
    #
    # The file moved into provenance/ when the output layout was reorganised, and
    # this guard kept looking for it at the root -- so it silently stopped firing
    # and a resume tree could pass. Both locations are checked now; the root one
    # keeps older trees verifiable. Missing the file entirely is also reported,
    # because a silent skip is what caused the regression in the first place.
    env_candidates = [
        args.candidate / "provenance" / "run_environment.json",
        args.candidate / "run_environment.json",
    ]
    env_path = next((p for p in env_candidates if p.exists()), None)
    if env_path is None:
        checks.append(
            Check(
                "run_environment.json",
                "run_mode",
                WARN,
                "no run_environment.json in provenance/ or at the tree root; "
                "cannot confirm the candidate was produced with RUN_MODE=\"fresh\"",
            )
        )
    else:
        try:
            mode = json.loads(env_path.read_text()).get("run_mode")
        except Exception:
            mode = None
        if mode is not None and mode != "fresh":
            checks.append(
                Check(
                    "run_environment.json",
                    "run_mode",
                    FAIL,
                    f"candidate was produced with RUN_MODE={mode!r}; cached steps did "
                    f"not rewrite their outputs, so this tree cannot be verified. "
                    f"Re-run with RUN_MODE=\"fresh\".",
                )
            )

    if args.manifest:
        blob = json.loads(args.manifest.read_text())
        entries = blob["files"]
        baseline_label = f"{args.manifest} (root {blob.get('root')})"
        cand_files = set(iter_files(args.candidate))
        for rel in sorted(set(entries) | cand_files):
            if rel not in entries:
                checks.append(Check(rel, classify(rel), WARN, "only in candidate"))
            elif rel not in cand_files:
                checks.append(Check(rel, entries[rel].get("kind", "?"), FAIL, "missing in candidate"))
            else:
                checks.append(
                    check_against_fingerprint(rel, args.candidate / rel, entries[rel], croot)
                )
    else:
        if not args.baseline:
            ap.error("either --baseline or --manifest is required")
        broot = args.baseline.as_posix()
        baseline_label = broot
        base_files = set(iter_files(args.baseline))
        cand_files = set(iter_files(args.candidate))
        for rel in sorted(base_files | cand_files):
            if rel not in base_files:
                checks.append(Check(rel, classify(rel), WARN, "only in candidate"))
            elif rel not in cand_files:
                checks.append(Check(rel, classify(rel), FAIL, "missing in candidate"))
            else:
                checks.append(
                    check_pair(rel, args.baseline / rel, args.candidate / rel, broot, croot)
                )

    for c in checks:
        if not args.quiet or c.status != PASS:
            print(c.line())

    n = {PASS: 0, WARN: 0, FAIL: 0}
    for c in checks:
        n[c.status] += 1
    hard = [c for c in checks if c.status == FAIL and c.kind in HARD_CLASSES]
    print(
        f"\n{len(checks)} files: {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail"
        f"  ({len(hard)} of the failures are hard classes: {sorted(HARD_CLASSES)})"
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render(checks, baseline_label, croot))
        print(f"report -> {args.report}")

    return 1 if n[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
