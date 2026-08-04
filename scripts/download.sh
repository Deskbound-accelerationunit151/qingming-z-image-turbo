#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/models}"
BASE_DIR="${MODEL_ROOT}/z_image_turbo_ms"
GGUF_DIR="${MODEL_ROOT}/z_image_turbo_gguf"
TARGET="${1:-all}"

if ! command -v hf >/dev/null 2>&1; then
    python3 -m pip install -U huggingface_hub
fi

mkdir -p "${BASE_DIR}" "${GGUF_DIR}"

download_base() {
    hf download \
        Tongyi-MAI/Z-Image-Turbo \
        --local-dir "${BASE_DIR}"
}

download_q8() {
    hf download \
        leejet/Z-Image-Turbo-GGUF \
        z_image_turbo-Q8_0.gguf \
        --local-dir "${GGUF_DIR}"
}

download_q6k() {
    hf download \
        leejet/Z-Image-Turbo-GGUF \
        z_image_turbo-Q6_K.gguf \
        --local-dir "${GGUF_DIR}"
}

download_q5km() {
    hf download \
        jayn7/Z-Image-Turbo-GGUF \
        z_image_turbo-Q5_K_M.gguf \
        --local-dir "${GGUF_DIR}"
}

case "${TARGET}" in
    base)
        download_base
        ;;
    q8)
        download_base
        download_q8
        ;;
    q6k)
        download_base
        download_q6k
        ;;
    q5km)
        download_base
        download_q5km
        ;;
    all)
        download_base
        download_q8
        download_q6k
        download_q5km
        ;;
    *)
        echo "usage: $0 [base|q8|q6k|q5km|all]" >&2
        exit 2
        ;;
esac
