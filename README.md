# qwen38-b70

One-command **llama.cpp SYCL** tensor-parallel server, defaulting to
Qwen3.8-27B on **two Intel Arc Pro B70 GPUs**, 262k context, OpenAI-compatible
API. Host paths, ports, render nodes, and model pins are set in `.env`.

```bash
cp .env.example .env          # or: cp examples/z8g4.env .env
# edit DRI_* / HOST_MODELS_DIR / HOST_PORT
docker compose up -d --build
```

On first start (unless `SKIP_DOWNLOAD=1`) it downloads the pinned,
SHA-256-verified GGUF into the host model directory, then serves on
**${HOST_PORT}** (default **8016**).

```bash
curl http://localhost:8016/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","max_tokens":64,
       "messages":[{"role":"user","content":"Hello"}]}'
```

## Configuration

Copy `.env.example` to `.env`. Compose interpolates that file for device
mounts and also injects the values into the container.

| Variable | Default | Meaning |
|----------|---------|---------|
| `HOST_MODELS_DIR` | `./models` | Host dir mounted at `/models` |
| `HOST_PORT` / `PORT` | `8016` | Published / listen port |
| `DRI_RENDER_0/1` | see `.env.example` | Host render nodes to pass (not all of `/dev/dri`) |
| `DRI_BYPATH_0/1` | see `.env.example` | Matching `/dev/dri/by-path/pci-*-render` entries |
| `GPU_COUNT` | `2` | `1` or `2` Arc GPUs (`1`: no TP, higher prefill) |
| `CTX_SIZE` | `262144` | Context on 2 GPUs |
| `CTX_SIZE_OVERRIDE` | blank | If set, wins; 1 GPU still defaults to 131072 when blank |
| `PARALLEL` | `1` | Concurrent request slots |
| `BATCH` / `UBATCH` | `8192` | Prefill batch; 8192 maximizes prefill |
| `ENABLE_MTP` | `0` | `1` = speculative decoding (mtp-Q4_0 draft) |
| `SPEC_DRAFT_N_MAX` | `5` | Draft tokens per step when MTP is on |
| `THREADS` | `16` | CPU threads for the target model |
| `ENABLE_VISION` | `0` | `1` = image input (mmproj Q8_0, CPU encoder) |
| `SKIP_DOWNLOAD` | `0` | `1` = require a local GGUF; no Hugging Face call |
| `MODEL_REPO` / `MODEL_REV` / `MODEL_FILE` / `MODEL_SHA256` | pinned Qwen3.8-27B Q4_K_M | Any GGUF; SHA-256 is required for the target |
| `MTP_*` / `MMPROJ_*` | same repo/rev, optional SHA | Draft / projector files |

Defaults match this README (`PARALLEL=1`, `ENABLE_MTP=0`). They are not the
old lab compose override (`PARALLEL=4`, `ENABLE_MTP=1`).

### GPU selection (do not pass all of `/dev/dri`)

On a mixed host (Intel Arc + NVIDIA + extra render nodes) the container must
see **only** the Arc cards you intend to use.

1. On the GPU host run `./scripts/find-intel-render-nodes.sh`.
2. Match PCI IDs (`lspci`, `xpu-smi discovery`) to
   `/dev/dri/by-path/pci-<addr>-render` → `../renderD*`.
3. Set `DRI_RENDER_*`, `DRI_RENDER_*_NAME`, `DRI_BYPATH_*`, and
   `DRI_BYPATH_*_NAME` in `.env`. The `*_NAME` values are the basename so the
   nodes keep the same names inside the container.

**`/dev/dri/by-path` must be mounted** (done in compose for the selected
entries). The Intel Level Zero driver enumerates GPUs through it;
`--device /dev/dri/renderD*` alone leaves adapters invisible.

If devices still do not enumerate, also pass the matching `pci-*-card` nodes
the same way (not enabled by default).

### ipc / devices

`ipc: host` is **not** set. This server is one process with two SYCL devices;
host IPC was a vLLM leftover. `shm_size` (default 8g) is kept for SYCL/USM
staging. The container is not privileged and does not bind the whole
`/dev/dri` tree.

### Local model, no network

Place `MODEL_FILE` in `HOST_MODELS_DIR` and set `SKIP_DOWNLOAD=1`. The
entrypoint still runs `sha256sum` against `MODEL_SHA256`.

## Running on z8g4

Host facts: Ubuntu, Docker, 2× Arc Pro B70 at **renderD129 / renderD130**
(PCI `0000:17:00.0` and `0000:9b:00.0`), plus an RTX 5060 Ti and other
render nodes. Ports **8010** (b70a coder), **8012** (b70b reviewer), and
**8014** (test) are taken. Models live under `/opt/models/<model-name>/`.
`xpu-smi` is installed; Level Zero 1.34.0.

Do this **on z8g4** (this repo does not SSH there):

```bash
git clone https://github.com/rtb144/qwen38-b70.git
cd qwen38-b70
cp examples/z8g4.env .env

# Confirm the B70 render nodes still match the env file
./scripts/find-intel-render-nodes.sh

# If the GGUF is already at /opt/models/Qwen3.8-27B-GGUF/, keep SKIP_DOWNLOAD=1.
# Otherwise set SKIP_DOWNLOAD=0 in .env for a first-time Hugging Face fetch.

docker compose build
docker compose up -d
docker compose ps
curl -sS http://127.0.0.1:8016/health

# Timings (prompt + generation tok/s) at a short prompt and ~32k, plus a
# needle-in-haystack retrieval check:
./scripts/bench-api.sh http://127.0.0.1:8016
```

To enable MTP later: set `ENABLE_MTP=1` in `.env` (and put the draft GGUF next
to the target, or allow download) and `docker compose up -d`.

## Measured performance (Q4_K_M, f16 KV)

### 2x B70, tensor parallel (`GPU_COUNT=2`)

| Prompt context | Prefill tok/s | TTFT    | Decode tok/s |
|----------------|--------------:|---------|-------------:|
| ~2.5k          | 955           | 2.1 s   | 51.1         |
| ~10k           | 881           | 9.1 s   | 50.9         |
| ~20k           | 801           | 20.0 s  | 51.1         |
| ~40k           | 694           | 46.2 s  | 49.7         |
| ~60k           | 563           | 106.6 s | 46.9         |
| ~160k          | 365           | 355.8 s | 42.0         |
| ~245k          | 275           | 727.3 s | 30.8         |

### 1x B70 (`GPU_COUNT=1`, max context 131,072)

| Prompt context | Prefill tok/s | TTFT    | Decode tok/s |
|----------------|--------------:|---------|-------------:|
| ~2.5k          | 1164          | 1.8 s   | 33.3         |
| ~10k           | 1018          | 7.9 s   | 33.4         |
| ~40k           | 785           | 40.8 s  | 33.4         |
| ~65k           | 625           | 96.1 s  | 31.9         |
| ~128k          | 426           | 282.1 s | 27.5         |

Single-GPU prefill **beats** TP2 at every context length (the all-reduce costs
more than the second card recovers); TP2 wins on decode and KV capacity.

### MTP speculative decoding (`ENABLE_MTP=1`)

| Config            | Task              | tok/s | Baseline | Draft acceptance |
|-------------------|-------------------|------:|---------:|-----------------:|
| 2x B70 (TP2)      | Easy (counting)   | 84.3  | 51.1     | 97.2%           |
| 2x B70 (TP2)      | Hard (random gen) | 49.0  | 51.1     | 37.5%           |
| 1x B70 (64k ctx)  | Easy (counting)   | 61.7  | 33.3     | 97.2%           |
| 1x B70 (64k ctx)  | Hard (random gen) | 28.2  | 33.3     | 29.6%           |

Note: on 1x B70 the draft model plus 131k context exceeds VRAM
(UR_RESULT_ERROR_DEVICE_LOST); 65536 is the validated MTP context on a single card.

Decode is nearly flat with context: only 16 of 64 layers are full attention
(Qwen3.8 hybrid GDN); the linear-attention layers are O(1) per token.

## What is inside

- **Source**: [mndodd/llama.cpp](https://github.com/mndodd/llama.cpp) @
  `4302fb5` + the
  [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab)
  full TP2 stack (`dp4a2` patch, decoded SHA-256
  `f21e9b55…`) + Q4K increment (`0a278585…`). Both digests verified against
  the lab's published values at image build.
- **Default model**: `ggml-org/Qwen3.8-27B-GGUF` @ `0669b986`,
  `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256 checked at every container start.
- **Base image**: `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04` pinned
  by manifest digest `sha256:79a5e333ff6f…`.
- **Build**: SYCL JIT via Intel oneAPI DPC++ 2025.3.3. AOT `bmg_g31` builds
  work under oneAPI 2026.1.1, but that runtime's UR level-zero adapter does not
  enumerate devices on any publicly available GPU userspace today; JIT is the
  verified-working path and reaches decode parity with the lab's promoted
  record (51 tok/s vs 49.7).

- The Q4K reorder-family fused kernels are disabled by default: they assume the
  lab AOT build's reordered tensor layout and corrupt output otherwise.
- Tested on: Arch kernel 7.1.8 (xe driver), 2x Arc Pro B70 (Battlemage G31),
  oneAPI 2025.3.3 container image.

One-off vLLM / sglang lab launchers live in [`experimental/`](experimental/)
(contents unchanged). The llama.cpp compose path does not use them.

## sglang XPU — not viable (2026-08)

sglang 0.5.13 XPU (`llm-scaler-sgl:bmg`) was tested as an alternative path
for Qwen3.8-27B on 2× B70. It loads and prefills, but the GDN decode kernel
produces NaN after ~4 decode tokens → all-`!!!` output at 0.44 tok/s.

Root cause: **SSM state drift** in
`fused_sigmoid_gating_delta_rule_update_kernel`. The fp32 state overflows to
NaN as aggressive decay values (`g` reaching -33 → `exp(g) ≈ 5e-15`) interact
with growing `h` across decode steps, producing `inf - inf = NaN` in the delta
rule update `v -= sum(h * k)`. Related to sglang issue
[#35150](https://github.com/sgl-project/sglang/issues/35150). The ratio-3
(48 V : 16 K heads) split is handled by the fallback path, but the numerical
instability is independent of the ratio.

Additionally, SGLang cannot run quantized models on XPU: GPTQ/AutoRound uses
CUDA-only `gptq_shuffle`/`gptq_gemm` kernels, AWQ-INT4 uses CUDA-only marlin
path, and FP8 block quantization has TP2 shape mismatches.

This repo (llama.cpp SYCL) is unaffected: `GGML_OP_GATED_DELTA_NET` enforces
fp32 at cache, op, and kernel levels, making the `!!!` state drift
structurally impossible.

Full investigation in [`docs/sglang-xpu-qwen38-analysis.md`](docs/sglang-xpu-qwen38-analysis.md).

## vLLM TP2 — 140.8 tok/s on 2× B70 (2026-08-20)

Real tensor parallelism (TP2) with MTP5 speculative decoding on the lab vLLM
image (`intel/llm-scaler-vllm:0.21.0-b3`) achieves **140.8 tok/s aggregate**
with 4 concurrent request streams, exceeding the 120 tok/s target.

### Results

| Config | GPU(s) | Decode tok/s | Coherent | Notes |
|--------|---------|-------------:|----------|-------|
| vLLM PIECEWISE MTP5 fp16 | 1× B70 | **90.1** hard | ✅ | Target 60 exceeded (+50%) |
| vLLM TP2 eager MTP5 fp16, 4 concurrent | 2× B70 | **140.8** aggregate | ✅ | Target 120 exceeded (+17%) |

### How to run TP2

Scripts are under `experimental/` (historical lab launchers; not used by
`docker compose`).

```bash
# 1. Copy the TP2 launch script to the host
scp experimental/lab_tp2_batch_mtp.sh HOST:/tmp/lab_tp2_batch_mtp.sh

# 2. Launch (auto-retries on intermittent XCCL failures)
scp experimental/launch_tp2_retry.sh HOST:/tmp/launch_tp2_retry.sh
ssh HOST 'bash /tmp/launch_tp2_retry.sh'

# 3. Benchmark
scp experimental/bench_concurrent_stable.sh HOST:/tmp/bench_concurrent_stable.sh
ssh HOST 'bash /tmp/bench_concurrent_stable.sh 4 5'
```

### Patches applied (all in the launch script, no image rebuild needed)

1. **mamba_utils.py** — `ctypes.c_int64(state.data_ptr()).value` fixes XPU
   pointer overflow (0xFFFF... range exceeds int64 max).
2. **gdn_linear_attn.py** — try/except around `self.out_proj.weight` in
   `_gdn_outproj_esimd_eligible()` for quantized `RowParallelLinear` in TP2.
3. **utils.py** — `non_blocking=False` in `CpuGpuBuffer.copy_to_gpu()` to
   avoid level_zero async copy staging buffer OOM.
4. **xpu_communicator.py** — catch XCCL `OUT_OF_RESOURCES` and fall back to
   CPU gloo `all_reduce` (tensor.cpu() → dist.all_reduce → tensor.xpu()).
   Only needed during profile_run (large prefill tensors); decode all_reduce
   works fine via XCCL (small single-token tensors).
5. **CLI flags** — `--skip-mm-profiling` (bypasses vision encoder all_reduce
   crash), `--enforce-eager` (avoids TP1 inductor compilation hang),
   `--tensor-parallel-size 2`, `--max-num-seqs 4`, `--gpu-memory-utilization 0.45`.
