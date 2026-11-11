# experimental/

One-off lab scripts from the original vLLM / sglang / hill-climb work on the
`omarchy` host. **Not used by `docker compose`.** File contents were not
rewritten; they still contain that lab's absolute paths.

Moved here from the repo root:

- `lab_tp2_*.sh`, `launch_tp2_retry.sh`, `launch-mtp*.sh`, `launch-test.sh`,
  `launch-vllm-dual.sh`, `mtp_breakable_entry.sh`, `vllm-b70-mtp-entrypoint.sh`
- `bench-vllm.sh`, `bench-vllm-lab.sh`, `bench-vllm-dual.sh`,
  `bench_concurrent_stable.sh`
- `hillclimb_iter.sh`, `HILLCLIMB.md`, `knob_queue.txt`
- `patch-mamba.sh`, `patch-mamba-uint64.sh`
- `vllm_gloo_kernels.py`, `vllm_src/`
- `check_cg_mode.py`, `check_gpu_mem.py`
- `test_ctypes.py`, `test_ptr_fix.py`, `test_sync_capture.py`,
  `test_triton_ptrmath.py`, `test_triton_uint64.py`

Removed as junk (not a script): `hillclimb_automation.log`.

The analysis notes in `docs/` stay put.
