"""Build results-adi/report.md from the ADI solver comparison (run_adi_benchmarks.sh).

Inputs:
    <dir>/<condition>/adi.json          src/benchmark_adi.py
    <dir>/nsys/*_nvtx_kern_sum.csv      nsys stats --report nvtx_kern_sum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_cusolver import FAMILIES, env_section, load_kernel_ranges

CONDITIONS = ["cuda13-native", "cuda13-emu", "cuda13-emu-eager"]
SOLVERS = ["tridiag-thomas", "tridiag-cusparse", "tridiag-cusparse-pivot", "propagator", "propagator-lu", "dense-lu"]
TRIDIAG = [s for s in SOLVERS if s.startswith("tridiag")]
EMU = [c for c in CONDITIONS if c != "cuda13-native"]


def lookup(payload: dict) -> dict[tuple[str, int], dict]:
    return {(r["solver"], r["n"]): r for r in payload["results"] if r.get("status") == "ok"}


def ms(v: float | None) -> str:
    return "-" if v is None else f"{v * 1e3:.4g}"


def ratio(a: float | None, b: float | None) -> str:
    return "-" if not a or not b else f"{a / b:.2f}x"


def sci(v: float | None) -> str:
    return "-" if v is None else f"{v:.2e}"


def per_step(luts: dict, cond: str, solver: str, n: int) -> float | None:
    r = luts.get(cond, {}).get((solver, n))
    return r["time_per_step_s"] if r else None


def best(luts: dict, conds: list[str], solvers: list[str], n: int) -> tuple[float | None, str]:
    """Fastest time per step over the given conditions and solvers, with its label."""
    cands = [(t, f"{s}, {c}") for c in conds for s in solvers if (t := per_step(luts, c, s, n)) is not None]
    return min(cands) if cands else (None, "-")


def headline_section(luts: dict, sizes: list[int]) -> str:
    lines = [
        "| N | tridiagonal [ms] | cuSPARSE [ms] | propagator, FP64 [ms] | propagator, Ozaki [ms] | dense LU, FP64 [ms] "
        "| dense LU, Ozaki [ms] | propagator Ozaki / tridiagonal | propagator Ozaki / cuSPARSE "
        "| dense LU Ozaki / tridiagonal |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sizes:
        tri, _ = best(luts, ["cuda13-native"], TRIDIAG, n)
        lib, _ = best(luts, ["cuda13-native"], [s for s in TRIDIAG if "cusparse" in s], n)
        prop_n = per_step(luts, "cuda13-native", "propagator", n)
        prop_e, _ = best(luts, EMU, ["propagator"], n)
        lu_n = per_step(luts, "cuda13-native", "dense-lu", n)
        lu_e, _ = best(luts, EMU, ["dense-lu"], n)
        lines.append(
            f"| {n} | {ms(tri)} | {ms(lib)} | {ms(prop_n)} | {ms(prop_e)} | {ms(lu_n)} | {ms(lu_e)} "
            f"| {ratio(prop_e, tri)} | {ratio(prop_e, lib)} | {ratio(lu_e, tri)} |"
        )
    return "\n".join(lines)


def time_section(luts: dict, sizes: list[int]) -> str:
    conds = [c for c in CONDITIONS if c in luts]
    lines = [
        "| solver | N | " + " | ".join(f"{c} [ms]" for c in conds) + " | native / emu | native / emu-eager |",
        "|---|---:|" + "---:|" * (len(conds) + 2),
    ]
    for solver in SOLVERS:
        for n in sizes:
            t = {c: per_step(luts, c, solver, n) for c in conds}
            if not any(t.values()):
                continue
            base = t.get("cuda13-native")
            lines.append(
                f"| {solver} | {n} | " + " | ".join(ms(t[c]) for c in conds)
                + f" | {ratio(base, t.get('cuda13-emu'))} | {ratio(base, t.get('cuda13-emu-eager'))} |"
            )
    return "\n".join(lines)


def accuracy_section(luts: dict, sizes: list[int]) -> str:
    conds = [c for c in CONDITIONS if c in luts]
    lines = [
        "| solver | N | " + " | ".join(conds) + " | discretization error |",
        "|---|---:|" + "---:|" * (len(conds) + 1),
    ]
    for solver in SOLVERS:
        for n in sizes:
            recs = {c: luts[c].get((solver, n)) for c in conds}
            if not any(recs.values()):
                continue
            first = next(r for r in recs.values() if r)
            lines.append(
                f"| {solver} | {n} | "
                + " | ".join(sci(r["vs_discrete_exact"]["max_rel_err"] if r else None) for r in recs.values())
                + f" | {sci(first['discretization_err_vs_continuous']['max_rel_err'])} |"
            )
    return "\n".join(lines)


def setup_section(luts: dict, sizes: list[int]) -> str:
    conds = [c for c in CONDITIONS if c in luts]
    rows = [
        ("dense-lu", "setup_getrf_s", "setup: getrf"),
        ("propagator-lu", "setup_getrf_getrs_s", "setup: getrf + getrs (N right-hand sides)"),
        ("dense-lu", "phases.gemm_x2_s", "one step: 2 GEMM"),
        ("dense-lu", "phases.getrs_x2_s", "one step: 2 getrs (N right-hand sides)"),
    ]
    lines = [
        "| solver | part | N | " + " | ".join(f"{c} [ms]" for c in conds) + " | native / emu | native / emu-eager |",
        "|---|---|---:|" + "---:|" * (len(conds) + 2),
    ]

    def get(rec: dict | None, key: str) -> float | None:
        for part in key.split("."):
            rec = rec.get(part) if isinstance(rec, dict) else None
        return rec

    for solver, key, label in rows:
        for n in sizes:
            t = {c: get(luts[c].get((solver, n)), key) for c in conds}
            if not any(t.values()):
                continue
            base = t.get("cuda13-native")
            lines.append(
                f"| {solver} | {label} | {n} | " + " | ".join(ms(t[c]) for c in conds)
                + f" | {ratio(base, t.get('cuda13-emu'))} | {ratio(base, t.get('cuda13-emu-eager'))} |"
            )
    return "\n".join(lines)


def kernel_section(path: Path) -> str:
    ranges = load_kernel_ranges(path)
    lines = [
        "| range | kernel ms | " + " | ".join(FAMILIES) + " | top kernels (share) |",
        "|---|---:|" + "---:|" * len(FAMILIES) + "---|",
    ]
    for name, r in ranges.items():
        shares = " | ".join(f"{r['share'].get(f, 0.0):.0%}" for f in FAMILIES)
        top = "<br>".join(f"`{k}` ({s:.0%})" for k, s in r["top"])
        lines.append(f"| {name} | {r['kernel_ms']:.2f} | {shares} | {top} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results-adi"))
    args = p.parse_args()
    d = args.results_dir

    payloads = {c: json.loads(f.read_text(encoding="utf-8")) for c in CONDITIONS if (f := d / c / "adi.json").exists()}
    if "cuda13-native" not in payloads:
        raise SystemExit(f"missing {d / 'cuda13-native' / 'adi.json'}")
    luts = {c: lookup(v) for c, v in payloads.items()}
    cfg = payloads["cuda13-native"]["config"]
    sizes = sorted({n for lut in luts.values() for _, n in lut})
    modes = ", ".join(f"{c}: {v['cusolver_math_mode']['mode']}" for c, v in payloads.items())

    parts = [
        "# ADI for the 2D heat equation: dense propagator, dense LU and tridiagonal solvers",
        "",
        f"Generated by `src/report_adi.py` from `{d}/` (`run_adi_benchmarks.sh`).",
        "",
        "Every solver advances the same Peaceman-Rachford ADI scheme (U <- M U M, M = A^-1 B) on the same grid,",
        f"with dt = {cfg['dt']:g} and {cfg['steps']} steps. Times are per step, medians of {cfg['repeats']} runs of "
        f"{cfg['steps']} steps after a warm-up step.",
        "",
        "- `tridiag-thomas`: banded multiply + Thomas algorithm in custom CUDA kernels, O(N^2)",
        "- `tridiag-cusparse` / `tridiag-cusparse-pivot`: banded multiply + cuSPARSE `gtsv2_nopivot` / `gtsv2`, O(N^2)",
        "- `propagator`: 2 GEMM with the dense M from the known eigen-decomposition (the `adi-gemm` scheme), O(N^3)",
        "- `propagator-lu`: 2 GEMM with M built on the GPU by getrf + getrs, O(N^3)",
        "- `dense-lu`: getrf once, then 2 GEMM + 2 getrs (N right-hand sides) per step, O(N^3)",
        "",
        f"cuSOLVER math mode set by the benchmark to follow the cuBLAS environment variables: {modes}.",
        "",
        "## Environment",
        "",
        env_section(payloads),
        "",
        "## Headline: time per step",
        "",
        "*tridiagonal* = fastest of the three tridiagonal solvers, *cuSPARSE* = faster of the two cuSPARSE solvers",
        "(both cuda13-native). *Ozaki* = faster of `cuda13-emu` and `cuda13-emu-eager`. The ratio columns say how many",
        "times slower (above 1x) or faster (below 1x) the dense Ozaki runs are.",
        "",
        headline_section(luts, sizes),
        "",
        "## Time per step, all solvers and conditions",
        "",
        "The tridiagonal solvers call neither cuBLAS nor cuSOLVER, so their ratios are a control.",
        "",
        time_section(luts, sizes),
        "",
        "## Rounding error: max relative error against the discrete exact solution",
        "",
        f"After {cfg['steps']} steps. The discretization error (discrete exact vs continuous solution) is shown for scale.",
        "",
        accuracy_section(luts, sizes),
        "",
        "## Setup and parts of the dense LU step",
        "",
        setup_section(luts, sizes),
        "",
    ]
    csvs = sorted((d / "nsys").glob("*_nvtx_kern_sum.csv"))
    if csvs:
        parts += [
            "## Kernels (Nsight Systems)",
            "",
            "One step per solver inside an NVTX range, after a warm-up step. Kernel families as in",
            "`results-cusolver/report.md`; the custom kernels and the cuSPARSE kernels fall under `other`.",
            "",
        ]
        for path in csvs:
            parts += [f"### {path.name.removesuffix('_nvtx_kern_sum.csv')}", "", kernel_section(path), ""]
    out = d / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
