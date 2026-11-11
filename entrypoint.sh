#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

MODELS_DIR="${MODELS_DIR:-/models}"
GPU_COUNT="${GPU_COUNT:-2}"
CTX_SIZE="${CTX_SIZE:-262144}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-8192}"
UBATCH="${UBATCH:-8192}"
PORT="${PORT:-8016}"
ENABLE_MTP="${ENABLE_MTP:-0}"
ENABLE_VISION="${ENABLE_VISION:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
THREADS="${THREADS:-16}"

if [ "$GPU_COUNT" = "1" ]; then
    DEVICE_ARGS=(--device SYCL0)
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-131072}"
else
    DEVICE_ARGS=(--device "SYCL0,SYCL1" --split-mode tensor --tensor-split "1,1")
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-$CTX_SIZE}"
fi

download_if_needed() {
    local dest="$1" repo="$2" file="$3" rev="$4"
    if [ -f "$dest" ]; then
        return 0
    fi
    if [ "$SKIP_DOWNLOAD" = "1" ]; then
        echo "[entrypoint] missing $dest (SKIP_DOWNLOAD=1; place the file on the host model dir)"
        exit 1
    fi
    echo "[entrypoint] downloading $repo/$file @$rev ..."
    hf download "$repo" "$file" --revision "$rev" --local-dir "$MODELS_DIR"
}

verify_sha() {
    local dest="$1" digest="$2" label="$3"
    if [ -z "$digest" ]; then
        echo "[entrypoint] WARNING: no SHA-256 configured for $label, skipping verify"
        return 0
    fi
    echo "$digest  $dest" | sha256sum -c - || {
        echo "[entrypoint] SHA-256 mismatch for $label"
        exit 1
    }
}

TARGET="$MODELS_DIR/${MODEL_FILE:?MODEL_FILE is required}"
download_if_needed "$TARGET" "${MODEL_REPO:?MODEL_REPO is required}" "$MODEL_FILE" "${MODEL_REV:?MODEL_REV is required}"
verify_sha "$TARGET" "${MODEL_SHA256:?MODEL_SHA256 is required}" "$MODEL_FILE"

DRAFT_ARGS=()
if [ "$ENABLE_MTP" = "1" ]; then
    MTP_REPO="${MTP_REPO:-$MODEL_REPO}"
    MTP_REV="${MTP_REV:-$MODEL_REV}"
    MTP_PATH="$MODELS_DIR/${MTP_FILE:?MTP_FILE is required when ENABLE_MTP=1}"
    download_if_needed "$MTP_PATH" "$MTP_REPO" "$MTP_FILE" "$MTP_REV"
    verify_sha "$MTP_PATH" "${MTP_SHA256:-}" "$MTP_FILE"
    DRAFT_ARGS=(
        --spec-type draft-mtp
        --model-draft "$MTP_PATH"
        --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-5}"
        --spec-draft-threads "${DRAFT_THREADS:-16}"
        --spec-draft-poll 1
        --spec-draft-ngl "${DRAFT_NGL:-0}"
        --spec-draft-device "${DRAFT_DEVICE:-SYCL0}"
    )
fi

MM_ARGS=()
if [ "$ENABLE_VISION" = "1" ]; then
    MMPROJ_REPO="${MMPROJ_REPO:-$MODEL_REPO}"
    MMPROJ_REV="${MMPROJ_REV:-$MODEL_REV}"
    MMPROJ_PATH="$MODELS_DIR/${MMPROJ_FILE:?MMPROJ_FILE is required when ENABLE_VISION=1}"
    download_if_needed "$MMPROJ_PATH" "$MMPROJ_REPO" "$MMPROJ_FILE" "$MMPROJ_REV"
    verify_sha "$MMPROJ_PATH" "${MMPROJ_SHA256:-}" "$MMPROJ_FILE"
    MM_ARGS=(--no-mmproj-offload --mmproj "$MMPROJ_PATH")
fi

# Lab runtime environment (b70-optimization-lab qwen38-27b-q4km-tp2 config.env)
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1
export GGML_SYCL_COMM_SINGLE_KERNEL=1
export GGML_META_FUSE_ALLREDUCE_ADD=1
export GGML_META_FUSE_ALLREDUCE_ADD_RMS_MUL=1
export GGML_SYCL_COMM_FUSED_Q8=1
export GGML_SYCL_FUSED_SWIGLU_Q8=1
export GGML_SYCL_FUSED_ATTN_Q8=1
export GGML_SYCL_FUSED_GDN_Q8=1
export GGML_SYCL_FUSED_GDN_BETA_SIGMOID=1
export GGML_SYCL_FUSED_CONCAT_STATE=1
export GGML_SYCL_FUSED_GDN_STATE_IO=1
export GGML_SYCL_FUSED_CONV_STATE_IO=1
export GGML_SYCL_COMM_DIRECT_Q8=2
export GGML_SYCL_FUSED_ROPE_SET_ROWS=1
export GGML_SYCL_COMM_REDUCE_VEC4=1
export GGML_SYCL_FUSED_QK_NORM_ROPE=1
export GGML_SYCL_FUSED_CONV_SILU_L2=1
export GGML_SYCL_FUSE_EXT=31
# Quality guards (verified coherent on the JIT build):
# - FATTN_MMA=0: joint_matrix unavailable under oneAPI 2025.3 JIT
# - Q4K reorder-family OFF: expects the lab AOT build's reordered layout,
#   produces corrupted output on stock ggml-org weights under JIT
export GGML_SYCL_FATTN_MMA=0
export GGML_SYCL_MMQ_Q4K_REORDER=1
export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=1
export GGML_SYCL_FUSED_MMVQ_PAIR=0
export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=0
export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=0
export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=0
export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=0

exec /build/llama.cpp/build-sycl/bin/llama-server \
    --model "$TARGET" \
    "${MM_ARGS[@]}" \
    "${DRAFT_ARGS[@]}" \
    --host 0.0.0.0 --port "$PORT" \
    "${DEVICE_ARGS[@]}" \
    --gpu-layers 99 \
    --flash-attn on \
    --batch-size "$BATCH" \
    --ubatch-size "$UBATCH" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --cache-ram 0 \
    --ctx-checkpoints 0 \
    --fit off \
    --reasoning off \
    --threads "$THREADS" \
    --poll 50 \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --metrics
