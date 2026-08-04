#!/usr/bin/env bash
#
# run_q5km_780m.sh — Run Z-Image-Turbo (Q5_K_M, phased-memory variant) on an
# AMD Radeon 780M (gfx1103) APU.
#
# Targets the devices/780m phased build (zimage_q5km_780m), which frees the
# Qwen text-encoder weights before the DiT phase (~halving sustained DiT-phase
# GTT, and a few % faster than the original build on this APU).
#
# Two prerequisites for clean, reliable runs on this iGPU. Both matter:
#
# 1. Setting up GTT
#    The 780M is a unified-memory APU with only a ~512 MiB VRAM carve-out; the
#    model (Q5_K_M weights + workspace ~16 GiB) must back onto system RAM via
#    the GTT pool. Ensure a large GTT and TTM page pool on the kernel cmdline,
#    e.g.:
#        amdgpu.gttsize=28672 ttm.pages_limit=6015590 ttm.page_pool_size=6015590
#    (edit /etc/default/grub, update-grub, reboot). The GFX ISA override is
#    applied automatically by this script:
#        export HSA_OVERRIDE_GFX_VERSION=11.0.0   # run gfx1100 build on gfx1103
#
# 2. Dropping to TTY for clean runs
#    Under a GNOME/desktop session the iGPU shares memory and compute with the
#    display compositor, which drives amdgpu page-pin failures
#    ("init_user_pages: Failed to get user pages: -1") and sporadic GPU resets.
#    For stable inference, drop to a text console and stop the GUI first:
#        Ctrl+Alt+F3                                       # switch to TTY, log in
#        sudo systemctl isolate multi-user.target          # stop GNOME/GDM
#        ./scripts/run_q5km_780m.sh "your prompt" out.ppm  # run
#        sudo systemctl isolate graphical.target           # restore desktop
#    Under TTY the run is silent (0 EPERM, 0 faults) and deterministic.
#
# The GGUF is staged on /dev/shm (see devices/780m/NOTES.md for why tmpfs is
# required on this APU), then removed on exit (including on error/interrupt).
#
# With NO arguments the script is interactive: it looks for the 780M binary in
# the current folder and, if present, asks for prompt, resolution and output
# name. If the binary is missing it tells you how to build it.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BINARY="${BINARY:-${ROOT}/zimage_q5km_780m}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/z_image_turbo_ms}"
GGUF_SRC="${GGUF_SRC:-${ROOT}/models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf}"
SIZE="${SIZE:-512x512}"
SEED="${SEED:-42}"
STEPS="${STEPS:-8}"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") "PROMPT" OUTPUT.ppm [options]
       $(basename "$0")                       # interactive (no args)

Arguments:
  "PROMPT"            Text prompt (required, quote it).
  OUTPUT.ppm          Output image path (required).

Options:
  --size WxH          Resolution: 512x512 | 576x1024 | 1024x1024 (default: ${SIZE}).
  --seed N            RNG seed (default: ${SEED}).
  --max-steps N       Diffusion steps (default: ${STEPS}).
  --binary FILE       Phased 780M binary (default: ${BINARY}).
  --model-dir DIR     Base model dir (default: ${MODEL_DIR}).
  --gguf FILE         Source GGUF to stage on tmpfs (default: ${GGUF_SRC}).

Env:
  HSA_OVERRIDE_GFX_VERSION  Set automatically to 11.0.0 for gfx1103 APUs.

The GGUF is copied to /dev/shm, the model runs from there, and the copy is
removed on exit (including on error or interrupt).
EOF
}

ARGC=$#
PROMPT=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)     PROMPT="$2"; shift 2;;
        --output)     OUTPUT="$2"; shift 2;;
        --size)       SIZE="$2"; shift 2;;
        --seed)       SEED="$2"; shift 2;;
        --max-steps)  STEPS="$2"; shift 2;;
        --binary)     BINARY="$2"; shift 2;;
        --model-dir)  MODEL_DIR="$2"; shift 2;;
        --gguf)       GGUF_SRC="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        --)           shift; break;;
        -*)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"
            elif [[ -z "$OUTPUT" ]]; then
                OUTPUT="$1"
            else
                echo "error: unexpected argument: $1" >&2
                exit 2
            fi
            shift
            ;;
    esac
done

# Interactive mode: invoked with no arguments at all.
if (( ARGC == 0 )); then
    BINARY_CANDIDATE="./zimage_q5km_780m"
    if [[ ! -x "$BINARY_CANDIDATE" ]]; then
        cat >&2 <<EOF
The 780M phased binary was not found in the current folder:
    $BINARY_CANDIDATE  (PWD=$(pwd))

Build it first from the repo root:

    hipcc --offload-arch=gfx1100 -x hip -std=c++17 -O3 -ffast-math -DNDEBUG \\
          -Wno-unused-function -Wno-unused-result \\
          devices/780m/q5km.cpp -o zimage_q5km_780m
EOF
        exit 1
    fi
    BINARY="$BINARY_CANDIDATE"

    echo "Interactive mode (780M phased binary: $BINARY)" >&2
    read -r -p "Prompt: " PROMPT || { echo; echo "aborted" >&2; exit 130; }
    if [[ -z "$PROMPT" ]]; then
        echo "error: prompt is required" >&2
        exit 2
    fi
    read -r -p "Resolution [512x512 | 576x1024 | 1024x1024] (default 512x512): " SIZE_IN \
        || { echo; echo "aborted" >&2; exit 130; }
    SIZE="${SIZE_IN:-512x512}"
    case "$SIZE" in
        512x512|576x1024|1024x1024) ;;
        *) echo "error: invalid resolution '$SIZE'" >&2; exit 2 ;;
    esac
    DEFAULT_OUT="outputs/q5km_${SIZE}.ppm"
    read -r -p "Output file (default ${DEFAULT_OUT}): " OUT_IN \
        || { echo; echo "aborted" >&2; exit 130; }
    OUTPUT="${OUT_IN:-$DEFAULT_OUT}"
fi

if [[ -z "$PROMPT" || -z "$OUTPUT" ]]; then
    echo "error: prompt and output file are required" >&2
    usage
    exit 2
fi

for f in "$BINARY" "$GGUF_SRC" "$MODEL_DIR"; do
    if [[ ! -e "$f" ]]; then
        echo "error: not found: $f" >&2
        exit 1
    fi
done

TMP_GGUF="/dev/shm/$(basename "$GGUF_SRC").$$"
cleanup() {
    if [[ -f "$TMP_GGUF" ]]; then
        rm -f "$TMP_GGUF"
        echo "[cleanup] removed $TMP_GGUF" >&2
    fi
}
trap cleanup EXIT INT TERM

SHM_FREE_MB=$(( $(stat -f -c "%a" /dev/shm) * $(stat -f -c "%S" /dev/shm) / 1024 / 1024 ))
GGUF_SIZE_MB=$(( $(stat -c "%s" "$GGUF_SRC") / 1024 / 1024 ))
if (( SHM_FREE_MB < GGUF_SIZE_MB )); then
    echo "error: /dev/shm has ${SHM_FREE_MB} MiB free, need ${GGUF_SIZE_MB} MiB for the GGUF" >&2
    exit 1
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

echo "[stage] copying GGUF (${GGUF_SIZE_MB} MiB) to $TMP_GGUF ..." >&2
cp -- "$GGUF_SRC" "$TMP_GGUF"

echo "[run] prompt=${PROMPT@Q} size=${SIZE} seed=${SEED} steps=${STEPS} binary=$BINARY" >&2
"$BINARY" 1 \
    --size "$SIZE" \
    --model-dir "$MODEL_DIR" \
    --dit-gguf "$TMP_GGUF" \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --max-steps "$STEPS" \
    --output "$OUTPUT"
