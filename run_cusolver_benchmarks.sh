#!/usr/bin/env bash
# cuSOLVER experiment: do the dense FP64 solvers (LU, Cholesky, QR, eigh) pick up FP64
# emulation (Ozaki scheme), and which cuBLAS level-3 building blocks do?
#
#   src/benchmark_cusolver.py  switches the math mode of CuPy's cuSOLVER handle in-process
#                              (cusolverDnSetMathMode). Run under cuda13-native and again
#                              under cuda13-emu, to show that the cuBLAS environment
#                              variables do not reach the cuBLAS handles inside cuSOLVER.
#   src/benchmark_blas3.py     cuBLAS GEMM / SYRK / TRSM / rank-k GEMM, switched by the
#                              cuBLAS environment variables of each condition.
#   nsys                       which kernels actually ran (Ozaki-II or native FP64).
#
# Environment variables:
#   RESULTS_DIR   output directory (default: results-cusolver, or results-quick/cusolver with QUICK=1)
#   SIZES         cuSOLVER sizes (default: "2048 4096 8192 12000")
#   QR_EIGH_MAX_N skip qr / eigh above this size (default: 12000). The 2026-11-22 run used
#                 8192; qr / eigh at 12000 were measured afterwards in a separate run:
#                   run_in cuda13-native python src/benchmark_cusolver.py --sizes 12000 \
#                     --routines qr eigh --qr-eigh-max-n 12000 \
#                     --output results-cusolver/cuda13-native/cusolver-qr-eigh-n12000.json
#                 src/report_cusolver.py merges such cusolver-*.json files.
#   BLAS3_N       size of the level-3 operations (default: 8192)
#   PROFILE_SIZES cuSOLVER sizes for the nsys kernel check (default: "2048 8192", empty = skip)
#   QUICK=1       smoke test with small sizes
#   SKIP_BUILD=1  do not (re)build the docker image
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/conditions.sh
source scripts/conditions.sh

SIZES="${SIZES:-2048 4096 8192 12000}"
QR_EIGH_MAX_N="${QR_EIGH_MAX_N:-12000}"
BLAS3_N="${BLAS3_N:-8192}"
PROFILE_SIZES="${PROFILE_SIZES-2048 8192}"
CUSOLVER_ARGS="--qr-eigh-max-n ${QR_EIGH_MAX_N}"
BLAS3_ARGS=""
if [[ "${QUICK:-0}" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-results-quick/cusolver}"
    SIZES="512 1024"
    BLAS3_N=1024
    PROFILE_SIZES=1024
    CUSOLVER_ARGS="${CUSOLVER_ARGS} --repeats 2 --repeats-large 1"
    BLAS3_ARGS="--repeats 2 --ranks 128 256 512"
fi
RESULTS_DIR="${RESULTS_DIR:-results-cusolver}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "building image: cuda13"
    docker compose build cuda13
fi

log "host GPU state"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.limit --format=csv

for cond in cuda13-native cuda13-emu cuda13-emu-eager; do
    out="${RESULTS_DIR}/${cond}"
    mkdir -p "${out}"
    log "=== ${cond}: environment check ==="
    run_in "${cond}" python src/common.py "${out}/env.json"

    log "=== ${cond}: cuBLAS level-3 (n=${BLAS3_N}) ==="
    # shellcheck disable=SC2086
    run_in "${cond}" python src/benchmark_blas3.py --n "${BLAS3_N}" --output "${out}/blas3.json" ${BLAS3_ARGS} \
        2>&1 | tee "${out}/blas3.log"

    # The cuSOLVER math modes are switched in-process; the eager cuBLAS condition adds nothing.
    if [[ "${cond}" != "cuda13-emu-eager" ]]; then
        log "=== ${cond}: cuSOLVER math modes (sizes ${SIZES}) ==="
        # shellcheck disable=SC2086
        run_in "${cond}" python src/benchmark_cusolver.py --sizes ${SIZES} --output "${out}/cusolver.json" \
            ${CUSOLVER_ARGS} 2>&1 | tee "${out}/cusolver.log"
    fi
done

if [[ -n "${PROFILE_SIZES}" ]]; then
    nsys_dir="${RESULTS_DIR}/nsys"
    mkdir -p "${nsys_dir}"
    # profile <condition> <report name> <command...>
    profile() {
        local cond="$1" rep="${nsys_dir}/$2"
        shift 2
        log "=== nsys: ${rep} (${cond}) ==="
        run_in "${cond}" bash -c "
            nsys profile -t cuda,nvtx --force-overwrite true -o '${rep}' $* > /dev/null &&
            nsys stats -q --report nvtx_kern_sum --format csv \
                --force-export true --force-overwrite true -o '${rep}' '${rep}.nsys-rep' > /dev/null &&
            rm -f '${rep}.sqlite'
        "
    }
    for n in ${PROFILE_SIZES}; do
        profile cuda13-native "cusolver_n${n}" \
            python src/benchmark_cusolver.py --profile "${n}" \
            --routines lu_solve cholesky qr eigh --modes default emu-performant emu-eager
    done
    for cond in cuda13-native cuda13-emu cuda13-emu-eager; do
        # shellcheck disable=SC2086
        profile "${cond}" "blas3_${cond}_n${BLAS3_N}" \
            python src/benchmark_blas3.py --profile --n "${BLAS3_N}" ${BLAS3_ARGS}
    done
fi

log "=== report ==="
run_in cuda13-native python src/report_cusolver.py --results-dir "${RESULTS_DIR}"
log "done: ${RESULTS_DIR}/report.md"
