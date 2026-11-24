"""One problem, several linear-algebra back ends: Peaceman-Rachford ADI for the 2D heat equation.

benchmark_pde.py runs ADI with a precomputed dense propagator M, which is a model
that turns the implicit step into two GEMMs. A production code would solve the
tridiagonal systems instead, in O(N^2) per step. This benchmark runs the *same*
scheme (same grid, time step, initial condition and discrete exact solution) with
different solvers, so that they can be compared on equal terms:

    propagator        U <- M U M with the dense M = A^-1 B          2 GEMM      O(N^3)
                      (M from the known eigen-decomposition, as in benchmark_pde.py)
    propagator-lu     same, but M is built on the GPU by getrf + getrs with
                      N right-hand sides, as for a general dense A  2 GEMM      O(N^3)
    dense-lu          A and B as general dense matrices: getrf once
                      (setup), then per step 2 GEMM + 2 getrs       cuSOLVER    O(N^3)
    tridiag-thomas    banded multiply + Thomas algorithm, one GPU
                      thread per grid line (custom CUDA kernels)                O(N^2)
    tridiag-cusparse  banded multiply + cuSPARSE gtsv2_nopivot                  O(N^2)
    tridiag-cusparse-pivot   same with cuSPARSE gtsv2 (pivoting)                O(N^2)

with A = I - a L, B = I + a L, a = alpha*dt/2 and L the 1D Dirichlet Laplacian.
All of them compute U <- B A^-1 U B A^-1 = M U M in exact arithmetic.

FP64 emulation (Ozaki scheme) follows the cuBLAS environment variables of the
condition: cuBLAS reads them itself, and this script sets the matching math mode
on CuPy's cuSOLVER handle (cuSOLVER ignores the environment variables). The
tridiagonal solvers use neither cuBLAS nor cuSOLVER and serve as a control.

Layout note: the arrays are C-ordered. A half step is written so that the solve
always runs along the direction the library handles in place, and the transposition
is folded into the preceding multiply (a transposed GEMM operand, or a kernel that
writes its output transposed). No explicit transpose copies are made.

With --profile N every solver runs one step inside an NVTX range "adi/<solver>"; the
one-time LU setup gets its own ranges "adi/<solver>:setup-getrf" / ":setup-getrs".
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark_cusolver import MODES, CusolverMathMode
from benchmark_pde import (
    ALPHA,
    adi_exact,
    adi_propagator,
    continuous_solution,
    error_metrics,
    grid_spacing,
    initial_condition,
)
from common import collect_env_info, condition_name, free_gpu_memory, log, summarize_times, time_gpu, write_json

SOLVERS = ("propagator", "propagator-lu", "dense-lu", "tridiag-thomas", "tridiag-cusparse", "tridiag-cusparse-pivot")
# Set by profile(): name of the NVTX range that wraps the timed steps (not the setup or warm-up).
_NVTX_RANGE: str | None = None

KERNELS = r"""
// w = (B v)^T or (v B)^T for the symmetric tridiagonal B = tridiag(bo, bd, bo):
// the 3-point stencil runs along `axis` of v and the result is written transposed.
extern "C" __global__
void apply_b_transposed(const double* __restrict__ v, double* __restrict__ w, const int n, const int axis,
                        const double bd, const double bo) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const long k = (long)i * n + j;
    const int p = axis == 0 ? i : j;
    const long s = axis == 0 ? n : 1;
    double off = 0.0;
    if (p > 0) off += v[k - s];
    if (p < n - 1) off += v[k + s];
    w[(long)j * n + i] = bd * v[k] + bo * off;
}

// Solve A x = d in place for every column of the C-ordered n x n array d (one thread per
// column, so the threads of a warp read neighbouring addresses). A = tridiag(lo, di, lo) is
// the same for all columns: inv[p] = 1 / (pivot of row p) and cp[p] = lo * inv[p] are
// precomputed on the host.
extern "C" __global__
void thomas_columns(double* __restrict__ d, const double* __restrict__ cp, const double* __restrict__ inv,
                    const int n, const double lo) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    double prev = d[col] * inv[0];
    d[col] = prev;
    for (int p = 1; p < n; ++p) {
        const long k = (long)p * n + col;
        prev = (d[k] - lo * prev) * inv[p];
        d[k] = prev;
    }
    for (int p = n - 2; p >= 0; --p) {
        const long k = (long)p * n + col;
        prev = d[k] - cp[p] * prev;
        d[k] = prev;
    }
}
"""


def stencil_coefficient(n: int, dt: float) -> float:
    """c = a / h^2, so that A = tridiag(-c, 1 + 2c, -c) and B = tridiag(c, 1 - 2c, c)."""
    return ALPHA * dt / 2 / grid_spacing(n) ** 2


def dense_tridiag(n: int, diag: float, off: float) -> np.ndarray:
    m = np.zeros((n, n))
    idx = np.arange(n)
    m[idx, idx] = diag
    m[idx[:-1], idx[1:]] = off
    m[idx[1:], idx[:-1]] = off
    return m


def thomas_coefficients(n: int, diag: float, off: float) -> tuple[np.ndarray, np.ndarray]:
    """Elimination coefficients of the Thomas algorithm for tridiag(off, diag, off)."""
    inv = np.empty(n)
    cp = np.empty(n)
    inv[0] = 1.0 / diag
    cp[0] = off * inv[0]
    for p in range(1, n):
        inv[p] = 1.0 / (diag - off * cp[p - 1])
        cp[p] = off * inv[p]
    return cp, inv


def follow_cublas_environment() -> dict:
    """Set the cuSOLVER math mode that matches the cuBLAS emulation environment variables."""
    if os.environ.get("CUBLAS_EMULATE_DOUBLE_PRECISION") == "1":
        mode = "emu-eager" if os.environ.get("CUBLAS_EMULATION_STRATEGY") == "eager" else "emu-performant"
    else:
        mode = "default"
    return {"mode": mode, **CusolverMathMode().set(*MODES[mode])}


def _in_setup_range(part: str, fn):
    """Run fn once; in profile mode inside an NVTX range "<solver range>:<part>"."""
    if _NVTX_RANGE is None:
        return fn()

    import cupy as cp
    from cupy.cuda import nvtx

    nvtx.RangePush(f"{_NVTX_RANGE}:{part}")
    out = fn()
    cp.cuda.Device().synchronize()
    nvtx.RangePop()
    return out


def _timed_integration(integrate, steps: int, warmup: int, repeats: int) -> list[float]:
    # Warm up separately so the timed run always starts from u0.
    time_gpu(lambda: integrate(1), warmup, 0)
    if _NVTX_RANGE is None:
        return time_gpu(lambda: integrate(steps), 0, repeats)

    import cupy as cp
    from cupy.cuda import nvtx

    nvtx.RangePush(_NVTX_RANGE)
    times = time_gpu(lambda: integrate(steps), 0, repeats)
    cp.cuda.Device().synchronize()
    nvtx.RangePop()
    return times


def run_propagator(n: int, dt: float, u0_h: np.ndarray, steps: int, warmup: int, repeats: int, from_lu: bool):
    """U <- M U M with the dense propagator (the adi-gemm scheme of benchmark_pde.py).

    With from_lu the propagator is not taken from the eigen-decomposition but computed on the
    GPU as M = A^-1 B (getrf + getrs with N right-hand sides), which is what a general dense
    problem would have to do. That setup is timed separately.
    """
    import cupy as cp
    import cupyx.scipy.linalg as cxl

    extra: dict = {}
    if from_lu:
        c = stencil_coefficient(n, dt)
        a_d = cp.asarray(dense_tridiag(n, 1.0 + 2.0 * c, -c))
        b_d = cp.asarray(dense_tridiag(n, 1.0 - 2.0 * c, c))

        def build():
            return cxl.lu_solve(cxl.lu_factor(a_d, check_finite=False), b_d, check_finite=False)

        extra["setup_getrf_getrs_s"] = float(np.median(time_gpu(build, 1, max(repeats, 3))))
        lu_piv = _in_setup_range("setup-getrf", lambda: cxl.lu_factor(a_d, check_finite=False))
        m = _in_setup_range("setup-getrs", lambda: cxl.lu_solve(lu_piv, b_d, check_finite=False))
        del lu_piv
        del a_d, b_d
        free_gpu_memory()
    else:
        m = cp.asarray(adi_propagator(n, dt))
    u0 = cp.asarray(u0_h)
    u = cp.empty_like(u0)
    tmp = cp.empty_like(u0)

    def integrate(n_steps: int) -> None:
        u[...] = u0
        for _ in range(n_steps):
            cp.matmul(m, u, out=tmp)
            cp.matmul(tmp, m, out=u)

    times = _timed_integration(integrate, steps, warmup, repeats)
    return times, cp.asnumpy(u), extra


def run_dense_lu(n: int, dt: float, u0_h: np.ndarray, steps: int, warmup: int, repeats: int):
    """getrf once, then 2 GEMM + 2 getrs per step, treating A and B as general dense matrices."""
    import cupy as cp
    import cupyx.scipy.linalg as cxl

    c = stencil_coefficient(n, dt)
    a_d = cp.asarray(dense_tridiag(n, 1.0 + 2.0 * c, -c))
    b_d = cp.asarray(dense_tridiag(n, 1.0 - 2.0 * c, c))
    setup = time_gpu(lambda: cxl.lu_factor(a_d, check_finite=False), 1, max(repeats, 3))
    lu_piv = _in_setup_range("setup-getrf", lambda: cxl.lu_factor(a_d, check_finite=False))
    del a_d

    u0 = cp.asarray(u0_h)
    u = cp.empty_like(u0)
    t = cp.empty_like(u0)

    def solve(x) -> None:
        # x.T is F-ordered, so getrs works in place on the buffer of x: x <- (A^-1 x^T)^T.
        cxl.lu_solve(lu_piv, x.T, overwrite_b=True, check_finite=False)

    def integrate(n_steps: int) -> None:
        u[...] = u0
        for _ in range(n_steps):
            cp.matmul(b_d, u.T, out=t)  # t = (U B)^T
            solve(t)  # t = (A^-1 U B)^T
            cp.matmul(b_d, t.T, out=u)  # u = B A^-1 U B
            solve(u)  # u = B A^-1 U B A^-1

    times = _timed_integration(integrate, steps, warmup, repeats)
    result = cp.asnumpy(u)
    # Where the time goes: the two GEMMs and the two triangular solves of one step, separately.
    phases = {
        "gemm_x2_s": float(
            np.median(time_gpu(lambda: (cp.matmul(b_d, u.T, out=t), cp.matmul(b_d, t.T, out=u)), 1, repeats))
        ),
        "getrs_x2_s": float(np.median(time_gpu(lambda: (solve(t), solve(u)), 1, repeats))),
    }
    return times, result, {"setup_getrf_s": float(np.median(setup)), "phases": phases}


def _tridiag_common(n: int, dt: float, u0_h: np.ndarray):
    import cupy as cp

    c = stencil_coefficient(n, dt)
    module = cp.RawModule(code=KERNELS)
    apply_b = module.get_function("apply_b_transposed")
    block = (32, 8, 1)
    grid = ((n + block[0] - 1) // block[0], (n + block[1] - 1) // block[1], 1)
    bd, bo = np.float64(1.0 - 2.0 * c), np.float64(c)

    def multiply(v, w, axis: int) -> None:
        apply_b(grid, block, (v, w, np.int32(n), np.int32(axis), bd, bo))

    u0 = cp.asarray(u0_h)
    return c, module, multiply, u0, cp.empty_like(u0), cp.empty_like(u0)


def run_tridiag_thomas(n: int, dt: float, u0_h: np.ndarray, steps: int, warmup: int, repeats: int):
    """Banded multiply + Thomas algorithm in custom CUDA kernels, one thread per grid line."""
    import cupy as cp

    c, module, multiply, u0, u, w = _tridiag_common(n, dt, u0_h)
    thomas = module.get_function("thomas_columns")
    cp_h, inv_h = thomas_coefficients(n, 1.0 + 2.0 * c, -c)
    cp_d, inv_d = cp.asarray(cp_h), cp.asarray(inv_h)
    block = (128, 1, 1)
    grid = ((n + block[0] - 1) // block[0], 1, 1)
    lo = np.float64(-c)

    def half(v, out) -> None:
        multiply(v, out, 0)  # out = (B v)^T
        thomas(grid, block, (out, cp_d, inv_d, np.int32(n), lo))  # out = A^-1 v^T B

    def integrate(n_steps: int) -> None:
        u[...] = u0
        for _ in range(n_steps):
            half(u, w)
            half(w, u)  # u = A^-1 (A^-1 U^T B)^T B = M U M

    times = _timed_integration(integrate, steps, warmup, repeats)
    return times, cp.asnumpy(u), {}


def run_tridiag_cusparse(n: int, dt: float, u0_h: np.ndarray, steps: int, warmup: int, repeats: int, pivot: bool):
    """Banded multiply + cuSPARSE gtsv2 / gtsv2_nopivot (every row of the array is a right-hand side)."""
    import cupy as cp
    from cupy.cuda import device
    from cupy_backends.cuda.libs import cusparse

    c, _, multiply, u0, u, w = _tridiag_common(n, dt, u0_h)
    handle = device.get_cusparse_handle()
    diag = cp.full(n, 1.0 + 2.0 * c)
    lower = cp.full(n, -c)
    upper = cp.full(n, -c)
    lower[0] = 0.0
    upper[-1] = 0.0
    if pivot:
        buffer_size, gtsv = cusparse.dgtsv2_bufferSizeExt, cusparse.dgtsv2
    else:
        buffer_size, gtsv = cusparse.dgtsv2_nopivot_bufferSizeExt, cusparse.dgtsv2_nopivot
    size = buffer_size(handle, n, n, lower.data.ptr, diag.data.ptr, upper.data.ptr, w.data.ptr, n)
    buffer = cp.empty(max(int(size), 1), dtype=cp.int8)

    def half(v, out) -> None:
        multiply(v, out, 1)  # out = (v B)^T
        # Column-major view of the C-ordered buffer: column r is row r, so this is out <- out A^-1.
        gtsv(handle, n, n, lower.data.ptr, diag.data.ptr, upper.data.ptr, out.data.ptr, n, buffer.data.ptr)

    def integrate(n_steps: int) -> None:
        u[...] = u0
        for _ in range(n_steps):
            half(u, w)
            half(w, u)  # u = B (B U^T A^-1)^T A^-1 = M U M

    times = _timed_integration(integrate, steps, warmup, repeats)
    untouched = bool((diag == 1.0 + 2.0 * c).all() and (lower[1:] == -c).all() and (upper[:-1] == -c).all())
    return times, cp.asnumpy(u), {"cusparse_buffer_bytes": int(size), "coefficients_untouched": untouched}


def runner(solver: str, n: int, args: argparse.Namespace):
    """Return a callable that runs `steps` steps and gives (times, result on host, extra record fields)."""
    u0 = initial_condition(n)
    steps, dt, wu, rep = args.steps, args.dt, args.warmup, args.repeats
    if solver in ("propagator", "propagator-lu"):
        return lambda: run_propagator(n, dt, u0, steps, wu, rep, from_lu=solver.endswith("lu"))
    if solver == "dense-lu":
        return lambda: run_dense_lu(n, dt, u0, steps, wu, rep)
    if solver == "tridiag-thomas":
        return lambda: run_tridiag_thomas(n, dt, u0, steps, wu, rep)
    if solver in ("tridiag-cusparse", "tridiag-cusparse-pivot"):
        return lambda: run_tridiag_cusparse(n, dt, u0, steps, wu, rep, pivot=solver.endswith("pivot"))
    raise ValueError(f"unknown solver: {solver}")


def run_case(solver: str, n: int, args: argparse.Namespace) -> dict:
    record: dict = {"solver": solver, "n": n, "steps": args.steps, "dt": args.dt}
    record["courant_number_r"] = ALPHA * args.dt / grid_spacing(n) ** 2
    log(f"{solver} n={n}: {args.steps} steps x {args.repeats} runs")
    try:
        times, u_gpu, extra = runner(solver, n, args)()
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        log(f"{solver} n={n}: FAILED ({exc!r})")
        record.update(status="failed", error=repr(exc))
        return record
    finally:
        free_gpu_memory()

    stats = summarize_times(times)
    exact = adi_exact(n, args.dt, args.steps)
    record.update(status="ok", times_s=times, **stats, time_per_step_s=stats["median_s"] / args.steps, **extra)
    record["vs_discrete_exact"] = error_metrics(u_gpu, exact)
    record["discretization_err_vs_continuous"] = error_metrics(exact, continuous_solution(n, args.steps * args.dt))
    log(
        f"{solver} n={n}: {record['time_per_step_s'] * 1e3:.3f} ms/step, "
        f"rel err vs discrete exact {record['vs_discrete_exact']['max_rel_err']:.3e}"
    )
    return record


def profile(n: int, args: argparse.Namespace) -> None:
    """One step per solver inside an NVTX range, for Nsight Systems."""
    global _NVTX_RANGE

    one_step = argparse.Namespace(**{**vars(args), "steps": 1, "warmup": 1, "repeats": 1})
    for solver in args.solvers:
        _NVTX_RANGE = f"adi/{solver}"
        runner(solver, n, one_step)()
        free_gpu_memory()
        log(f"profiled n={n} {solver}")
    _NVTX_RANGE = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", type=int, default=None, metavar="N", help="NVTX-annotated single steps at size N")
    p.add_argument("--output", default=None, help="default: results-adi/<condition>/adi.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cusolver_mode = follow_cublas_environment()
    if args.profile:
        profile(args.profile, args)
        return

    output = args.output or f"results-adi/{condition_name()}/adi.json"
    env = collect_env_info()
    log(f"cuBLAS {env['cublas_version']}, emu={env['emulation_env']}, cuSOLVER math mode {cusolver_mode}")
    payload = {
        "benchmark": "adi",
        "condition": condition_name(),
        "env": env,
        "cusolver_math_mode": cusolver_mode,
        "config": vars(args),
        "results": [],
    }
    for n in args.sizes:
        for solver in args.solvers:
            payload["results"].append(run_case(solver, n, args))
            write_json(output, payload)
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
