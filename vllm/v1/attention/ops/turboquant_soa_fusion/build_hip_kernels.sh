#!/usr/bin/env bash
# Build the four HIP decode kernels used by the SoA fusion path into shared
# objects loadable via ctypes.CDLL from external_ops.py.
#
# These kernels were ported from jiangyon-amd/vllm-ISOquant @
# turboquant_attention_opt. The .hip sources are checked in; the .so files
# are not (they are gfx-target dependent).
#
# Usage:
#   bash build_hip_kernels.sh               # builds for gfx950 (MI350X/MI355X)
#   ARCH=gfx942 bash build_hip_kernels.sh   # builds for gfx942 (MI300X)
#
# Output filenames match the names expected by external_ops.py.
set -euo pipefail

cd "$(dirname "$0")"

ARCH=${ARCH:-gfx950}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

if [[ ! -x "${HIPCC}" ]]; then
    echo "ERROR: hipcc not found at ${HIPCC}. Set HIPCC env var." >&2
    exit 1
fi

# Sources (left) -> output .so (right). Output names MUST match the names
# external_ops.py looks for via Path(__file__).with_name(...).
declare -A SOURCES=(
    [hip_v3_scalar.hip]=hip_v3_scalar.so
    [hip_v3_mfma_qk.hip]=hip_v3_mfma_qk.so
    [hip_v3_flash_tq.hip]=hip_v3_flash_tq.so
    [soa_bf16q_pv_mfma_decode.hip]=soa_bf16q_pv_mfma_decode.so
    [soa_bf16q_pv_mfma_decode_gqa6.hip]=soa_bf16q_pv_mfma_decode_gqa6.so
)

echo "Building HIP kernels for arch=${ARCH} with ${HIPCC}"
echo

for src in "${!SOURCES[@]}"; do
    out=${SOURCES[$src]}
    echo "  ${src}  ->  ${out}"
    ${HIPCC} \
        -O3 \
        -shared -fPIC \
        --offload-arch="${ARCH}" \
        ${EXTRA_FLAGS} \
        "${src}" -o "${out}"
done

echo
echo "Done. Output files:"
ls -la *.so 2>/dev/null || echo "  (no .so produced)"
