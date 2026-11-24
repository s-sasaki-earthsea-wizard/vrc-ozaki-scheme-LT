"""cuBLAS FP64 level-3 building blocks of a blocked factorization, with and without emulation.

A blocked LU / Cholesky spends most of its time in three kinds of updates:
a square GEMM-like product, a symmetric rank-k update (SYRK) and a triangular
solve with many right-hand sides (TRSM), plus GEMMs whose inner dimension is
the block size nb. This script times each of them on CuPy's cuBLAS handle, so
the emulation is switched by the cuBLAS environment variables of the
condition (cuda13-native / cuda13-emu / cuda13-emu-eager):

    gemm          C = A @ A                    (n x n x n)
    syrk          C = A A^T, upper triangle    (cublasDsyrk)
    trsm          X = L^-1 A, L lower          (cublasDtrsm, n right-hand sides)
    gemm_rank<k>  C = P P^T, P is n x k        (a trailing update with block size k)

With --profile each operation runs once inside an NVTX range "blas3:<op>" for
Nsight Systems, and timing is skipped.
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from benchmark_dgemm import make_matrix
from common import collect_env_info, condition_name, free_gpu_memory, log, summarize_times, time_gpu, write_json


def operations(n: int, ranks: list[int], phi: float, seed: int) -> tuple[dict[str, Callable[[], object]], dict]:
    """Build device inputs and return the callables to time, plus their flop counts."""
    import cupy as cp
    import cupyx.scipy.linalg as cxl
    from cupy import cublas

    rng = np.random.default_rng([seed, n, int(round(phi * 1000)), 3])
    a = cp.asarray(make_matrix(rng, n, phi))
    low = cp.tril(a) + n * cp.eye(n)  # well-conditioned lower triangle
    ops: dict[str, Callable[[], object]] = {
        "gemm": lambda: a @ a,
        "syrk": lambda: cublas.syrk("N", a),
        "trsm": lambda: cxl.solve_triangular(low, a, lower=True),
    }
    flops = {"gemm": 2.0 * n**3, "syrk": 1.0 * n**3, "trsm": 1.0 * n**3}
    for k in ranks:
        panel = cp.ascontiguousarray(a[:, :k])
        ops[f"gemm_rank{k}"] = lambda p=panel: p @ p.T
        flops[f"gemm_rank{k}"] = 2.0 * n * n * k
    return ops, flops


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--ranks", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--phi", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--profile", action="store_true", help="NVTX-annotated single calls, no timing")
    p.add_argument("--output", default=None, help="default: results-cusolver/<condition>/blas3.json")
    return p.parse_args()


def main() -> None:
    import cupy as cp
    from cupy.cuda import nvtx

    args = parse_args()
    ops, flops = operations(args.n, args.ranks, args.phi, args.seed)

    if args.profile:
        for name, fn in ops.items():
            fn()  # warm-up
            cp.cuda.Device().synchronize()
            nvtx.RangePush(f"blas3:{name}")
            fn()
            cp.cuda.Device().synchronize()
            nvtx.RangePop()
            free_gpu_memory()
            log(f"profiled n={args.n} {name}")
        return

    output = args.output or f"results-cusolver/{condition_name()}/blas3.json"
    env = collect_env_info()
    log(f"cuBLAS {env['cublas_version']}, emu={env['emulation_env']}")
    payload = {"benchmark": "blas3", "condition": condition_name(), "env": env, "config": vars(args), "results": []}
    for name, fn in ops.items():
        rec = {"n": args.n, "op": name, "flops": flops[name]}
        try:
            times = time_gpu(fn, args.warmup, args.repeats)
            stats = summarize_times(times)
            rec.update(status="ok", times_s=times, **stats, tflops_median=flops[name] / stats["median_s"] / 1e12)
            log(f"n={args.n} {name:14s} median {stats['median_s']:.4f} s, {rec['tflops_median']:.3f} TFLOPS")
        except Exception as exc:  # noqa: BLE001 - keep going with the next operation
            rec.update(status="failed", error=repr(exc))
            log(f"n={args.n} {name}: FAILED ({exc!r})")
        finally:
            free_gpu_memory()
        payload["results"].append(rec)
        write_json(output, payload)
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
