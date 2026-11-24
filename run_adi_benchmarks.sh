#!/usr/bin/env bash
# ADI solver comparison: the same Peaceman-Rachford ADI step of the 2D heat equation with a
# dense propagator (GEMM), a dense LU solve (cuSOLVER) and tridiagonal solvers (custom
# Thomas kernel, cuSPARSE gtsv2), with FP64 emulation off and on. See src/benchmark_adi.py.
#
# The cuBLAS environment variables of each condition switch cuBLAS, and benchmark_adi.py
# sets the matching math mode on the cuSOLVER handle. The tridiagonal solvers use neither
# library, so they double as a control.
#
# Environment variables:
#   CONDITIONS    default: "cuda13-native cuda13-emu cuda13-emu-eager"
#   RESULTS_DIR   output directory (default: results-adi, or results-quick/adi with QUICK=1)
#   SIZES         default: "1024 2048 4096 8192"
#   PROFILE_N     size for the nsys kernel check (default: 8192, empty = skip)
#   ADI_ARGS      extra CLI arguments for src/benchmark_adi.py
#   QUICK=1       smoke test with small sizes
#   SKIP_BENCH=1  keep the existing adi.json files and redo only the nsys profiles and the report
#   SKIP_BUILD=1  do not (re)build the docker image
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/conditions.sh
source scripts/conditions.sh

CONDITIONS="${CONDITIONS:-cuda13-native cuda13-emu cuda13-emu-eager}"
SIZES="${SIZES:-1024 2048 4096 8192}"
PROFILE_N="${PROFILE_N-8192}"
ADI_ARGS="${ADI_ARGS:-}"
if [[ "${QUICK:-0}" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-results-quick/adi}"
    SIZES="256 512"
    PROFILE_N=512
    ADI_ARGS="--steps 5 --repeats 2 ${ADI_ARGS}"
fi
RESULTS_DIR="${RESULTS_DIR:-results-adi}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "building image: cuda13"
    docker compose build cuda13
fi

log "host GPU state"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.limit --format=csv

for cond in ${CONDITIONS}; do
    [[ "${SKIP_BENCH:-0}" == "1" ]] && continue
    out="${RESULTS_DIR}/${cond}"
    mkdir -p "${out}"
    log "=== ${cond}: environment check ==="
    run_in "${cond}" python src/common.py "${out}/env.json"

    log "=== ${cond}: ADI solvers (sizes ${SIZES}) ==="
    # shellcheck disable=SC2086
    run_in "${cond}" python src/benchmark_adi.py --sizes ${SIZES} --output "${out}/adi.json" ${ADI_ARGS} \
        2>&1 | tee "${out}/adi.log"
done

if [[ -n "${PROFILE_N}" ]]; then
    nsys_dir="${RESULTS_DIR}/nsys"
    mkdir -p "${nsys_dir}"
    for cond in ${CONDITIONS}; do
        rep="${nsys_dir}/adi_${cond}_n${PROFILE_N}"
        log "=== nsys: ${rep} ==="
        run_in "${cond}" bash -c "
            nsys profile -t cuda,nvtx --force-overwrite true -o '${rep}' \
                python src/benchmark_adi.py --profile ${PROFILE_N} > /dev/null &&
            nsys stats -q --report nvtx_kern_sum --format csv \
                --force-export true --force-overwrite true -o '${rep}' '${rep}.nsys-rep' > /dev/null &&
            rm -f '${rep}.sqlite'
        "
    done
fi

log "=== report ==="
run_in cuda13-native python src/report_adi.py --results-dir "${RESULTS_DIR}"
log "done: ${RESULTS_DIR}/report.md"
