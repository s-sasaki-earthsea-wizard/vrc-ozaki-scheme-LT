"""Build results-cusolver/report.md from the cuSOLVER / cuBLAS level-3 runs.

Inputs (written by run_cusolver_benchmarks.sh):
    <dir>/<condition>/cusolver.json   src/benchmark_cusolver.py
    <dir>/<condition>/blas3.json      src/benchmark_blas3.py
    <dir>/nsys/*_nvtx_kern_sum.csv    nsys stats --report nvtx_kern_sum

Kernels are grouped by name: "oz2" = Ozaki-II pre/post-processing,
nvjet "biu" = INT8 tensor-core GEMM (the core of Ozaki-II), "d884gemm" /
"d884syrk" = native FP64 GEMM / SYRK (FP64 tensor core, DMMA).
"cublasLt_imma_emu::...d884gemm" is an FP64 DMMA GEMM launched by the
emulation path itself; in plain DGEMM runs it takes a few microseconds, so
it is presumably the conditional fall-back to native FP64 (inferred from the
name and the timings, not documented).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from summarize_nsys import short_name

CUSOLVER_MODES = ["default", "emu-performant", "emu-eager", "default-again"]
EMU_MODES = ["emu-performant", "emu-eager"]
ROUTINE_ORDER = ["gemm", "lu_solve", "cholesky", "qr", "eigh"]
BLAS3_CONDITIONS = ["cuda13-native", "cuda13-emu", "cuda13-emu-eager"]
FAMILIES = ["ozaki2", "int8-mma", "emu-fp64", "fp64-gemm", "fp64-syrk", "trsm", "other"]
ACCURACY_COLUMNS = {
    "lu_solve": [("backward_err_in_u", "backward error / u")],
    "cholesky": [("recon_rel_err_in_u", "‖LLᵀ−S‖/‖S‖ / u")],
    "qr": [("recon_rel_err_in_u", "‖QR−A‖/‖A‖ / u"), ("orth_err", "‖QᵀQ−I‖/√N")],
    "eigh": [("recon_rel_err_in_u", "‖HV−VΛ‖/‖H‖ / u"), ("orth_err", "‖VᵀV−I‖/√N")],
}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cusolver_lookup(payload: dict) -> dict[tuple[str, int, str], dict]:
    return {(r["routine"], r["n"], r["mode"]): r for r in payload["results"] if r.get("status") == "ok"}


def fmt_s(v: float | None) -> str:
    return "-" if v is None else f"{v:.4g}"


def fmt_x(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}x"


def fmt_e(v: float | None) -> str:
    return "-" if v is None else f"{v:.3g}"


def env_section(payloads: dict[str, dict]) -> str:
    lines = [
        "| condition | GPU | driver | CUDA | cuBLAS | cuSOLVER | CuPy | cuBLAS emulation env | SM clock at start |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cond, p in payloads.items():
        e = p["env"]
        smi = e.get("nvidia_smi", {})
        emu = ", ".join(f"{k.removeprefix('CUBLAS_')}={v}" for k, v in e["emulation_env"].items() if v is not None)
        lines.append(
            f"| {cond} | {e['gpu_name']} | {smi.get('driver_version', '-')} | {e.get('cuda_toolkit')} "
            f"| {e['cublas_version']} | {e.get('cusolver_version', '-')} | {e['cupy']} | {emu or '-'} "
            f"| {smi.get('clocks.sm', '-')} |"
        )
    return "\n".join(lines)


def timing_section(payload: dict) -> str:
    lut = cusolver_lookup(payload)
    sizes = sorted({n for _, n, _ in lut})
    lines = [
        "| routine | N | " + " | ".join(CUSOLVER_MODES) + " | best emu / default | best emu mode |",
        "|---|---:|" + "---:|" * len(CUSOLVER_MODES) + "---:|---|",
    ]
    for routine in ROUTINE_ORDER:
        for n in sizes:
            row = [lut.get((routine, n, m)) for m in CUSOLVER_MODES]
            if not any(row):
                continue
            base = row[0]["median_s"] if row[0] else None
            emu = {m: lut[(routine, n, m)]["median_s"] for m in EMU_MODES if (routine, n, m) in lut}
            best_mode = min(emu, key=emu.get) if emu else None
            speedup = base / emu[best_mode] if base and best_mode else None
            cells = " | ".join(fmt_s(r["median_s"] if r else None) for r in row)
            lines.append(f"| {routine} | {n} | {cells} | {fmt_x(speedup)} | {best_mode or '-'} |")
    return "\n".join(lines)


def accuracy_section(payload: dict) -> str:
    lut = cusolver_lookup(payload)
    sizes = sorted({n for _, n, _ in lut})
    modes = CUSOLVER_MODES[:3]
    lines = ["| routine | metric | N | " + " | ".join(modes) + " |", "|---|---|---:|" + "---:|" * len(modes)]
    for routine, cols in ACCURACY_COLUMNS.items():
        for key, label in cols:
            for n in sizes:
                vals = [lut.get((routine, n, m), {}).get("accuracy", {}).get(key) for m in modes]
                if any(v is not None for v in vals):
                    lines.append(f"| {routine} | {label} | {n} | " + " | ".join(fmt_e(v) for v in vals) + " |")
    return "\n".join(lines)


def env_var_section(native: dict, emu: dict) -> str:
    """cuda13-emu (cuBLAS env on) vs cuda13-native, both in the default cuSOLVER math mode."""
    a, b = cusolver_lookup(native), cusolver_lookup(emu)
    sizes = sorted({n for _, n, _ in a})
    lines = [
        "| routine | N | cuda13-native, default | cuda13-emu, default | ratio | cuda13-emu, emu-performant |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for routine in ROUTINE_ORDER:
        for n in sizes:
            ra, rb = a.get((routine, n, "default")), b.get((routine, n, "default"))
            rc = b.get((routine, n, "emu-performant"))
            if not (ra and rb):
                continue
            ratio = ra["median_s"] / rb["median_s"]
            lines.append(
                f"| {routine} | {n} | {fmt_s(ra['median_s'])} | {fmt_s(rb['median_s'])} | {fmt_x(ratio)} "
                f"| {fmt_s(rc['median_s'] if rc else None)} |"
            )
    return "\n".join(lines)


def blas3_section(payloads: dict[str, dict]) -> str:
    luts = {c: {r["op"]: r for r in p["results"] if r.get("status") == "ok"} for c, p in payloads.items()}
    conds = [c for c in BLAS3_CONDITIONS if c in luts]
    ops = list(dict.fromkeys(op for c in conds for op in luts[c]))
    n = next(iter(next(iter(payloads.values()))["results"]))["n"]
    head = " | ".join(f"{c} [s]" for c in conds)
    ratio_conds = [c for c in conds if c != "cuda13-native"]
    lines = [
        f"N = {n}.",
        "",
        f"| op | {head} | " + " | ".join(f"{c} / native" for c in ratio_conds) + " |",
        "|---|" + "---:|" * (len(conds) + len(ratio_conds)),
    ]
    for op in ops:
        t = {c: luts[c].get(op, {}).get("median_s") for c in conds}
        base = t.get("cuda13-native")
        ratios = [fmt_x(base / t[c]) if base and t.get(c) else "-" for c in ratio_conds]
        lines.append(f"| {op} | " + " | ".join(fmt_s(t[c]) for c in conds) + " | " + " | ".join(ratios) + " |")
    return "\n".join(lines)


def kernel_family(name: str) -> str:
    """Coarse kernel family from its name (see the module docstring)."""
    s = name.lower()
    if "oz2" in s:
        return "ozaki2"
    if "imma_emu" in s and "d884" in s:
        return "emu-fp64"
    if ("nvjet" in s and ("biu" in s or "i8" in s)) or "imma" in s:
        return "int8-mma"
    if "syrk" in s:
        return "fp64-syrk"
    if "d884gemm" in s or "dgemm" in s or "dmma" in s or "f64" in s:
        return "fp64-gemm"
    if "trsm" in s:
        return "trsm"
    return "other"


def load_kernel_ranges(path: Path) -> dict[str, dict]:
    """Group an nvtx_kern_sum CSV by NVTX range and compute the time share per kernel family."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["NVTX Range"].strip(":")]
    ranges: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ranges[r["NVTX Range"].lstrip(":")].append(r)
    out = {}
    for name, rs in ranges.items():
        total = sum(float(r["Total Time (ns)"]) for r in rs)
        fam: dict[str, float] = defaultdict(float)
        for r in rs:
            fam[kernel_family(r["Kernel Name"])] += float(r["Total Time (ns)"])
        top = sorted(rs, key=lambda r: float(r["Total Time (ns)"]), reverse=True)[:3]
        out[name] = {
            "kernel_ms": total / 1e6,
            "share": {k: v / total for k, v in fam.items()} if total else {},
            "top": [(short_name(r["Kernel Name"], 60), float(r["Total Time (ns)"]) / total) for r in top],
        }
    return out


def kernel_section(path: Path) -> str:
    ranges = load_kernel_ranges(path)
    lines = [
        f"`{path.name}`: one call per range after a warm-up call.",
        "",
        "| range | kernel ms | " + " | ".join(FAMILIES) + " | top kernels (share) |",
        "|---|---:|" + "---:|" * len(FAMILIES) + "---|",
    ]
    for name, r in ranges.items():
        shares = " | ".join(f"{r['share'].get(f, 0.0):.0%}" for f in FAMILIES)
        top = "<br>".join(f"`{k}` ({s:.0%})" for k, s in r["top"])
        lines.append(f"| {name} | {r['kernel_ms']:.1f} | {shares} | {top} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results-cusolver"))
    args = p.parse_args()
    d = args.results_dir

    cusolver = {c: v for c in BLAS3_CONDITIONS if (v := load_json(d / c / "cusolver.json"))}
    blas3 = {c: v for c in BLAS3_CONDITIONS if (v := load_json(d / c / "blas3.json"))}
    if "cuda13-native" not in cusolver:
        raise SystemExit(f"missing {d / 'cuda13-native' / 'cusolver.json'}")
    native = cusolver["cuda13-native"]
    cfg = native["config"]

    parts = [
        "# cuSOLVER and cuBLAS level-3 with FP64 emulation",
        "",
        f"Generated by `src/report_cusolver.py` from `{d}/` (`run_cusolver_benchmarks.sh`).",
        "",
        "## Environment",
        "",
        env_section({**cusolver, **{c: v for c, v in blas3.items() if c not in cusolver}}),
        "",
        "## cuSOLVER: time per call [s] (median), condition cuda13-native",
        "",
        "The math mode of CuPy's cuSOLVER handle is switched in-process (`cusolverDnSetMathMode`):",
        "`default` = native FP64, `emu-*` = `CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH` with the emulation strategy",
        "`performant` / `eager`, `default-again` = native again at the end (reversibility check).",
        f"Inputs: A as in the DGEMM benchmark (phi = {cfg['phi']}), S = A Aᵀ + N I, H = (A + Aᵀ) / 2, seed {cfg['seed']}.",
        f"Medians of {cfg['repeats']} calls ({cfg['repeats_large']} for qr / eigh) after {cfg['warmup']} warm-up call.",
        "`gemm` runs on CuPy's cuBLAS handle and is the control: the cuSOLVER math mode must not change it.",
        "",
        timing_section(native),
        "",
        "## cuSOLVER: accuracy, condition cuda13-native",
        "",
        "Residuals of one call, evaluated in FP64 with cuBLAS (native in this condition). u = 2⁻⁵³.",
        "",
        accuracy_section(native),
        "",
    ]
    if "cuda13-emu" in cusolver:
        parts += [
            "## The cuBLAS environment variables do not reach cuSOLVER",
            "",
            "Same benchmark under `cuda13-emu` (`CUBLAS_EMULATE_DOUBLE_PRECISION=1`, `performant`).",
            "A ratio near 1.00x in the default cuSOLVER math mode means that cuSOLVER ignored the environment variables;",
            "`gemm` (CuPy's own cuBLAS handle) does pick them up.",
            "",
            env_var_section(native, cusolver["cuda13-emu"]),
            "",
        ]
    if blas3:
        parts += [
            "## cuBLAS level-3 building blocks [s] (median)",
            "",
            "Switched by the cuBLAS environment variables of each condition (CuPy's cuBLAS handle).",
            "`gemm_rank<k>` = P Pᵀ with P of shape N × k, i.e. a trailing update with block size k.",
            "",
            blas3_section(blas3),
            "",
        ]
    nsys_csvs = sorted((d / "nsys").glob("*_nvtx_kern_sum.csv"))
    if nsys_csvs:
        parts += [
            "## Kernels (Nsight Systems)",
            "",
            "Share of GPU kernel time per family: `ozaki2` = kernels with `oz2` in the name (Ozaki-II splitting and",
            "reconstruction), `int8-mma` = INT8 tensor-core GEMM (`nvjet_*_biu_mma`), `emu-fp64` = FP64 DMMA GEMM",
            "launched by the emulation path (`cublasLt_imma_emu::...d884gemm`, presumably the fall-back to native",
            "FP64), `fp64-gemm` / `fp64-syrk` = native FP64 (`d884gemm` / `d884syrk`), `trsm` = triangular solves,",
            "`other` = solver-specific kernels (panel factorization, pivoting, Householder, tridiagonal eigensolver, ...).",
            "",
        ]
        for path in nsys_csvs:
            parts += [f"### {path.name.removesuffix('_nvtx_kern_sum.csv')}", "", kernel_section(path), ""]
    out = d / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
