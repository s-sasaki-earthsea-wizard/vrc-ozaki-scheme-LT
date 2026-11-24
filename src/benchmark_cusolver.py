"""Dense FP64 solvers in cuSOLVER with and without FP64 emulation (Ozaki scheme).

cuSOLVER does not follow the cuBLAS environment variables: the cuSOLVER
documentation (section "Floating Point Emulation") states that they do not
affect the cuBLAS handles inside cuSOLVERDn. Emulation is switched per cuSOLVER
handle with cusolverDnSetMathMode() and cusolverDnSetEmulationStrategy().
CuPy 14 has no binding for these, so this script calls libcusolver through
ctypes on CuPy's own cuSOLVER handle and switches the math mode in-process:

    default         CUSOLVER_DEFAULT_MATH (native FP64)
    emu-performant  CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH, strategy "performant"
    emu-eager       CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH, strategy "eager"
    default-again   native again, to check that the switch is reversible

Routines, called through CuPy (which calls cuSOLVERDn):

    gemm       A @ A on CuPy's cuBLAS handle (control: the cuSOLVER math
               mode must not change it; the cuBLAS environment variables do)
    lu_solve   cupy.linalg.solve(A, b)   getrf + getrs, one right-hand side
    cholesky   cupy.linalg.cholesky(S)   potrf
    qr         cupy.linalg.qr(A)         geqrf + orgqr
    eigh       cupy.linalg.eigh(H)       Xsyevd, eigenvalues and eigenvectors

Inputs are generated on the CPU with NumPy, so they are bit-identical across
conditions: A as in benchmark_dgemm.py (phi = 0.5), S = A A^T + n I (well
conditioned), H = (A + A^T) / 2.

Accuracy is measured by backward-error style residuals evaluated in FP64 on
the GPU with CuPy's cuBLAS handle. Those residual products carry their own
rounding error (and are emulated under the cuda13-emu conditions), so compare
modes within one condition only.

With --profile the script runs one call per (routine, mode) inside an NVTX
range "<routine>/<mode>" for Nsight Systems and skips timing and accuracy.
"""

from __future__ import annotations

import argparse
import ctypes
import math
from typing import Callable

import numpy as np

from benchmark_dgemm import make_matrix
from common import (
    UNIT_ROUNDOFF,
    _loaded_libraries,
    collect_env_info,
    condition_name,
    free_gpu_memory,
    log,
    summarize_times,
    time_gpu,
    write_json,
)

# cusolverMathMode_t (cusolver_common.h) and cudaEmulationStrategy_t (library_types.h)
CUSOLVER_DEFAULT_MATH = 1
CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH = 4
CUDA_EMULATION_STRATEGY_DEFAULT = 0
CUDA_EMULATION_STRATEGY_PERFORMANT = 1
CUDA_EMULATION_STRATEGY_EAGER = 2

MODES = {
    "default": (CUSOLVER_DEFAULT_MATH, CUDA_EMULATION_STRATEGY_DEFAULT),
    "emu-performant": (CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH, CUDA_EMULATION_STRATEGY_PERFORMANT),
    "emu-eager": (CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH, CUDA_EMULATION_STRATEGY_EAGER),
    "default-again": (CUSOLVER_DEFAULT_MATH, CUDA_EMULATION_STRATEGY_DEFAULT),
}
ROUTINES = ("gemm", "lu_solve", "cholesky", "qr", "eigh")
# Routines whose cost or memory grows fastest; capped by --qr-eigh-max-n.
LARGE_ROUTINES = ("qr", "eigh")


class CusolverMathMode:
    """Switch the math mode of CuPy's cuSOLVER handle through ctypes."""

    def __init__(self) -> None:
        from cupy.cuda import device

        self.handle = ctypes.c_void_p(device.get_cusolver_handle())
        loaded = _loaded_libraries("libcusolver.so")
        # Open the very library CuPy loaded, so that the handle is valid for it.
        self.lib = ctypes.CDLL(loaded[0] if loaded else "libcusolver.so.12")
        self.library_path = loaded[0] if loaded else None

    def _check(self, status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} returned cusolverStatus_t {status}")

    def set(self, math_mode: int, strategy: int) -> dict[str, int]:
        """Set the math mode and emulation strategy, then read both back."""
        self._check(self.lib.cusolverDnSetMathMode(self.handle, ctypes.c_int(math_mode)), "cusolverDnSetMathMode")
        self._check(
            self.lib.cusolverDnSetEmulationStrategy(self.handle, ctypes.c_int(strategy)),
            "cusolverDnSetEmulationStrategy",
        )
        got_mode, got_strategy = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.cusolverDnGetMathMode(self.handle, ctypes.byref(got_mode)), "cusolverDnGetMathMode")
        self._check(
            self.lib.cusolverDnGetEmulationStrategy(self.handle, ctypes.byref(got_strategy)),
            "cusolverDnGetEmulationStrategy",
        )
        return {"math_mode": got_mode.value, "emulation_strategy": got_strategy.value}

    def version(self) -> int:
        """Return cusolverGetVersion() (e.g. 12304 for 12.3.4)."""
        v = ctypes.c_int()
        self._check(self.lib.cusolverGetVersion(ctypes.byref(v)), "cusolverGetVersion")
        return v.value


def make_inputs(n: int, phi: float, seed: int, need_large: bool) -> dict[str, np.ndarray]:
    """Generate the host inputs for one size (deterministic for a given seed)."""
    rng = np.random.default_rng([seed, n, int(round(phi * 1000)), 1])
    a = make_matrix(rng, n, phi)
    inputs = {"a": a, "b": rng.random(n) - 0.5}
    inputs["s"] = a @ a.T + n * np.eye(n)
    if need_large:
        inputs["h"] = (a + a.T) / 2
    return inputs


def _fro(x) -> float:
    import cupy as cp

    return float(cp.linalg.norm(x))


def routine_calls(d: dict) -> dict[str, Callable[[], object]]:
    """Return the callables to time, keyed by routine name."""
    import cupy as cp

    calls = {
        "gemm": lambda: d["a"] @ d["a"],
        "lu_solve": lambda: cp.linalg.solve(d["a"], d["b"]),
        "cholesky": lambda: cp.linalg.cholesky(d["s"]),
    }
    if "h" in d:
        calls["qr"] = lambda: cp.linalg.qr(d["a"])
        calls["eigh"] = lambda: cp.linalg.eigh(d["h"])
    return calls


def accuracy(routine: str, d: dict) -> dict[str, float]:
    """Backward-error style residuals of one call of the routine."""
    import cupy as cp

    n = d["a"].shape[0]
    if routine == "lu_solve":
        a, b = d["a"], d["b"]
        x = cp.linalg.solve(a, b)
        # Normwise backward error ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf)
        eta = float(cp.abs(b - a @ x).max() / (cp.abs(a).sum(axis=1).max() * cp.abs(x).max() + cp.abs(b).max()))
        return {"backward_err": eta, "backward_err_in_u": eta / UNIT_ROUNDOFF}
    if routine == "cholesky":
        s = d["s"]
        low = cp.linalg.cholesky(s)
        rel = _fro(low @ low.T - s) / _fro(s)
        return {"recon_rel_err": rel, "recon_rel_err_in_u": rel / UNIT_ROUNDOFF}
    if routine == "qr":
        a = d["a"]
        q, r = cp.linalg.qr(a)
        recon = _fro(q @ r - a) / _fro(a)
        orth = _fro(q.T @ q - cp.eye(n)) / math.sqrt(n)
        return {"recon_rel_err": recon, "orth_err": orth, "recon_rel_err_in_u": recon / UNIT_ROUNDOFF}
    if routine == "eigh":
        h = d["h"]
        w, v = cp.linalg.eigh(h)
        recon = _fro(h @ v - v * w) / _fro(h)
        orth = _fro(v.T @ v - cp.eye(n)) / math.sqrt(n)
        return {"recon_rel_err": recon, "orth_err": orth, "recon_rel_err_in_u": recon / UNIT_ROUNDOFF}
    return {}


def repeats_for(routine: str, args: argparse.Namespace) -> int:
    return args.repeats_large if routine in LARGE_ROUTINES else args.repeats


def run_size(n: int, args: argparse.Namespace, switch: CusolverMathMode) -> list[dict]:
    """Benchmark every (routine, mode) pair at one size."""
    import cupy as cp

    need_large = n <= args.qr_eigh_max_n
    log(f"n={n}: generating inputs on the CPU")
    host = make_inputs(n, args.phi, args.seed, need_large)
    d = {k: cp.asarray(v) for k, v in host.items()}
    del host
    records = []
    try:
        for routine, fn in routine_calls(d).items():
            if routine not in args.routines:
                continue
            for mode in args.modes:
                rec = {"n": n, "routine": routine, "mode": mode}
                rec["readback"] = switch.set(*MODES[mode])
                try:
                    times = time_gpu(fn, args.warmup, repeats_for(routine, args))
                    rec.update(status="ok", times_s=times, **summarize_times(times))
                    rec["accuracy"] = accuracy(routine, d)
                    log(f"n={n} {routine:9s} {mode:15s} median {rec['median_s']:.4f} s {rec['accuracy']}")
                except Exception as exc:  # noqa: BLE001 - e.g. OOM; keep the sweep going
                    rec.update(status="failed", error=repr(exc))
                    log(f"n={n} {routine} {mode}: FAILED ({exc!r})")
                finally:
                    free_gpu_memory()
                records.append(rec)
    finally:
        switch.set(*MODES["default"])
        d.clear()
        free_gpu_memory()
    return records


def profile(n: int, args: argparse.Namespace, switch: CusolverMathMode) -> None:
    """One call per (routine, mode) inside an NVTX range, for Nsight Systems."""
    import cupy as cp
    from cupy.cuda import nvtx

    host = make_inputs(n, args.phi, args.seed, need_large=True)
    d = {k: cp.asarray(v) for k, v in host.items()}
    del host
    for routine, fn in routine_calls(d).items():
        if routine not in args.routines:
            continue
        for mode in args.modes:
            switch.set(*MODES[mode])
            fn()  # warm-up: handle setup, workspace for this math mode
            cp.cuda.Device().synchronize()
            nvtx.RangePush(f"{routine}/{mode}")
            fn()
            cp.cuda.Device().synchronize()
            nvtx.RangePop()
            free_gpu_memory()
            log(f"profiled n={n} {routine} {mode}")
    switch.set(*MODES["default"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192, 12000])
    p.add_argument("--routines", nargs="+", choices=ROUTINES, default=list(ROUTINES))
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--qr-eigh-max-n", type=int, default=8192, help="skip qr / eigh above this size")
    p.add_argument("--phi", type=float, default=0.5, help="exponent-range parameter of A")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5, help="timed calls for gemm / lu_solve / cholesky")
    p.add_argument("--repeats-large", type=int, default=3, help="timed calls for qr / eigh")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--profile", type=int, default=None, metavar="N", help="NVTX-annotated single calls at size N")
    p.add_argument("--output", default=None, help="default: results-cusolver/<condition>/cusolver.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    switch = CusolverMathMode()
    if args.profile:
        profile(args.profile, args, switch)
        return

    output = args.output or f"results-cusolver/{condition_name()}/cusolver.json"
    env = collect_env_info()
    env["cusolver_version"] = switch.version()
    env["libcusolver_loaded"] = _loaded_libraries("libcusolver")
    log(f"cuSOLVER {env['cusolver_version']}, cuBLAS {env['cublas_version']}, emu={env['emulation_env']}")

    payload = {
        "benchmark": "cusolver",
        "condition": condition_name(),
        "env": env,
        "config": vars(args),
        "modes": {k: {"math_mode": m, "emulation_strategy": s} for k, (m, s) in MODES.items()},
        "results": [],
    }
    for n in args.sizes:
        payload["results"].extend(run_size(n, args, switch))
        write_json(output, payload)  # checkpoint after every size
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
