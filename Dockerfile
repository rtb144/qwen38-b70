# Generic llama.cpp SYCL tensor-parallel server (default: Qwen3.8-27B on Intel Arc).
#
# Build:  docker compose build
# Run:    cp .env.example .env && docker compose up -d
#
# Source: mndodd/llama.cpp @ 4302fb5 + steveseguin/b70-optimization-lab TP2 stack
# (both patches SHA-256-verified against the lab's published digests).
#
# Pin is the Docker Hub manifest digest for
# intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04 (linux/amd64).
FROM intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04@sha256:79a5e333ff6f773793d124b78047001c51cbcd53035e5100313abf2f771af95a AS llama-src

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
        python3-pip python3-venv ffmpeg \
    && pip3 install --break-system-packages "huggingface_hub[cli]" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone -q https://github.com/mndodd/llama.cpp llama.cpp \
    && cd llama.cpp \
    && git checkout -q 4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126

COPY patches/ /patches/
# Digests from b70-optimization-lab (README: f21e9b55… and 0a278585…).
RUN echo "f21e9b557c3d024527ac98d5f189cf7ea72fa8c38a5faf2a22ee339fd1988998  /patches/tp2-full-stack.patch" | sha256sum -c - \
    && echo "0a27858525f6a402cf9c92d1b93daee0a80e2ffaef9137bf7bce784a549b58b6  /patches/q4k-increment.patch" | sha256sum -c - \
    && cd /build/llama.cpp \
    && git apply --check /patches/tp2-full-stack.patch \
    && git apply /patches/tp2-full-stack.patch \
    && git apply --check /patches/q4k-increment.patch \
    && git apply /patches/q4k-increment.patch \
    && git diff --check

# SYCL JIT configure only (AOT bmg_g31 needs oneAPI 2026.1.x whose UR runtime
# does not enumerate devices on public GPU stacks). Use:
#   docker build --target cmake-configure .
FROM llama-src AS cmake-configure
SHELL ["/bin/bash", "-c"]
RUN cd /build/llama.cpp \
    && . /opt/intel/oneapi/setvars.sh --force \
    && cmake -G "Unix Makefiles" -S . -B build-sycl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/cc \
        -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
        -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=OFF -DGGML_SYCL_DNN=OFF \
        -DGGML_SYCL_HOST_MEM_FALLBACK=OFF \
        -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON

FROM cmake-configure AS runtime
RUN cd /build/llama.cpp \
    && . /opt/intel/oneapi/setvars.sh --force \
    && cmake --build build-sycl --target llama-server -j"$(nproc)"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default model pins (SHA-256 verified at start). Override via compose/.env.
ENV MODEL_REPO=ggml-org/Qwen3.8-27B-GGUF \
    MODEL_REV=0669b98607d47046c7c2b3f801011d54a08cfccf \
    MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf \
    MODEL_SHA256=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34 \
    MTP_FILE=mtp-Qwen3.8-27B-Q4_0.gguf \
    MMPROJ_FILE=mmproj-Qwen3.8-27B-Q8_0.gguf \
    PORT=8016

EXPOSE 8016
ENTRYPOINT ["/entrypoint.sh"]
