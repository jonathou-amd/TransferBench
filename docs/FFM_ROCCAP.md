# FFM pre-silicon simulation and roccap capture

This branch (`ffm-sim-roccap-capture`) adds support for running TransferBench in **FFM
pre-silicon simulation** on a single physical GPU while modeling **N logical GPUs**,
then capturing **roccap `.cap` traces** plus matching **`heap_bases.json`** files for
FFM / aqltoolkit replay.

## Overview

When `TB_SIMULATION=1`, TransferBench:

- Allocates one **listener heap pool** per logical GPU (`NUM_GPU_DEVICES`) on physical device 0
- Sub-allocates src/dst buffers inside the pool for each logical GPU index
- Writes **`heap_bases.json`** listing each pool base address (Iris-style)
- Runs executors **serially** by default (`TB_SERIAL_EXECUTORS=1`) for clean roccap captures
- Supports **per-GPU capture** via `TB_CAPTURE_EXECUTOR=N` (used by `roccap_capture.py`)

### Files added or changed

| File | Purpose |
|------|---------|
| `src/client/Simulation.hpp` | Listener pools, heap JSON, capture executor filter |
| `src/header/TransferBench.hpp` | Sim memory, topology, serial executor loop |
| `src/client/Client.cpp` | Normal process exit so roccap can finalize caps |
| `src/client/Presets/AllToAll.hpp` | A2A direct pairs in sim (skip XGMI hop check) |
| `src/client/Presets/TdmSweep.hpp` | TDM sweep on sim-capable hardware |
| `src/header/tdmCopy.h` | TDM copy helpers (gfx1260) |
| `scripts/ffm_env.sh` | Container roccap 4.10 `PATH` / `LD_LIBRARY_PATH` |
| `scripts/roccap_capture.py` | Wrapper: one `.cap` + `_heap_bases.json` per logical GPU |

## Requirements

- **Singularity image:** `competes-arcadia-260806.sif` (roccap **4.10**, 256-CU FFM arcadia)
- **Do not** bind-mount host `libhsa` into the container (breaks FFM)
- Bind your workspace, e.g. `--bind /large_data:/large_data`

Avoid `customized-competes-gfx1260-260518.sif` for capture — roccap 4.8 finalize is broken on FFM.

## Setup (inside container)

```bash
# Host: start container
singularity shell --bind /large_data:/large_data /path/to/competes-arcadia-260806.sif

# Inside container
cd /path/to/TransferBench
source scripts/ffm_env.sh
GPU_TARGETS=gfx1260 DISABLE_POD_COMM=1 make -j
```

`ffm_env.sh` prepends `/workspace/rocplaycap-Linux/opt/rocm/bin` (bundled roccap 4.10).

## GFX all-to-all capture

Uses `GpuReduceKernel` with non-temporal loads/stores when `--gfx-temporal 3` is set.
Caps are **packet format 2.4** (roccap 4.10).

```bash
./scripts/roccap_capture.py \
  --num-gpu-devices 2 \
  --tb-sim-heap-bytes 512M \
  --num-iterations 1 \
  --num-warmups 0 \
  --gfx-block-size 1024 \
  --gfx-unroll 8 \
  --num-sub-exec 128 \
  --gfx-temporal 3 \
  --dispatchlog \
  --output-dir caps \
  a2a 128M
```

**Output (example):**

```
caps/transferbench_GBS_1024_UNROLL_8_NSE_128_TEMP_3_A2A_128M_2nproc_rank0.cap
caps/transferbench_GBS_1024_UNROLL_8_NSE_128_TEMP_3_A2A_128M_2nproc_rank0_heap_bases.json
caps/transferbench_GBS_1024_UNROLL_8_NSE_128_TEMP_3_A2A_128M_2nproc_rank1.cap
caps/transferbench_GBS_1024_UNROLL_8_NSE_128_TEMP_3_A2A_128M_2nproc_rank1_heap_bases.json
```

Validate a cap:

```bash
roccap extract --meta caps/transferbench_..._rank0.cap
```

## TDM all-to-all capture

Uses `GpuTdmKernel` and `tdm::tdmCopy` (tensor load/store, not `flat_load` NT modifiers).
`--gfx-block-size` maps to `TDM_BLOCK_SIZE`.

```bash
./scripts/roccap_capture.py \
  --use-tdm-exec \
  --num-gpu-devices 2 \
  --tb-sim-heap-bytes 512M \
  --num-iterations 1 \
  --num-warmups 0 \
  --gfx-block-size 256 \
  --num-sub-exec 128 \
  --dispatchlog \
  --output-dir caps \
  a2a 128M
```

**Output (example):**

```
caps/transferbench_tdm_GBS_256_NSE_128_A2A_128M_2nproc_rank0.cap
caps/transferbench_tdm_GBS_256_NSE_128_A2A_128M_2nproc_rank0_heap_bases.json
...
```

## SDMA (DMA) all-to-all capture

Uses the GPU **SDMA copy engine** via `hipMemcpyDeviceToDeviceNoCU` — not GFX or TDM
kernels. Default `--disp` is `0-` (captures SDMA traffic, not shader dispatches).

`GPU_MAX_HW_QUEUES` defaults to `NUM_GPU_DEVICES - 1` so all peer copies can run in
parallel (override with `--gpu-max-hw-queues`).

```bash
./scripts/roccap_capture.py \
  --use-dma-exec \
  --num-gpu-devices 8 \
  --tb-sim-heap-bytes 768M \
  --num-iterations 1 \
  --num-warmups 0 \
  --dispatchlog \
  --output-dir caps_mgpu \
  a2a 36M
```

**Output (example):**

```
caps_mgpu/transferbench_ffm_dma_write_GMQ_7_A2A_36M_8nproc_rank0.cap
caps_mgpu/transferbench_ffm_dma_write_GMQ_7_A2A_36M_8nproc_rank0_heap_bases.json
...
```

For remote-read (executor on DST GPU):

```bash
./scripts/roccap_capture.py \
  --use-remote-read 1 \
  --use-dma-exec \
  --num-gpu-devices 8 \
  --tb-sim-heap-bytes 768M \
  --num-iterations 1 \
  --num-warmups 0 \
  --dispatchlog \
  --output-dir caps_mgpu \
  a2a 36M
```

Batch run (6 configs, no 32-GPU) from the workspace parent directory:

```bash
/large_data/jonathou/transferbench_ffm_08042026/run_dma_sweep.sh
```

Log: `dma_sweep_run.log`

## Async-to/from-LDS all-to-all capture

Same `GpuTdmKernel` launch path as tensor TDM, but the copy body uses
`async::tdmCopy` (`global_load_async_to_lds` / `global_store_async_from_lds`)
instead of tensor TDM. One gfx1260 binary; select at runtime with `--use-async-exec`.

| Mode | Flag | Kernel / ISA |
|------|------|--------------|
| GMEM / GFX | (default) | `GpuReduceKernel` |
| Tensor TDM | `--use-tdm-exec` | `GpuTdmKernel` + tensor load/store |
| Async LDS | `--use-async-exec` | `GpuTdmKernel` + async-to/from-LDS |
| SDMA | `--use-dma-exec` | no shader dispatch |

`--gfx-unroll`, `--gfx-temporal`, and `--gfx-scope` reuse the VMEM knobs:
unroll selects the async hot-loop factor; temporal maps `GFX_TEMPORAL` 0–3 onto
RT/NT load and store hints; scope maps `GFX_SCOPE` 0–3 onto WGP/SE/DEV/SYS
(async defaults to SYS when unset).

```bash
./scripts/roccap_capture.py \
  --use-async-exec \
  --num-gpu-devices 2 \
  --tb-sim-heap-bytes 768M \
  --num-iterations 1 \
  --num-warmups 0 \
  --gfx-block-size 256 \
  --gfx-unroll 32 \
  --gfx-temporal 3 \
  --gfx-scope sys \
  --num-sub-exec 80 \
  --dispatchlog \
  --output-dir caps_mgpu \
  a2a 256M
```

Remote-read:

```bash
./scripts/roccap_capture.py \
  --use-remote-read 1 \
  --use-async-exec \
  ...
```

**Output (example):** `transferbench_ffm_async_write_GBS_256_UNROLL_32_NSE_80_TEMP_3_SCOPE_SYS_A2A_256M_2nproc_rank0.cap`

## Mixed TDM + VMEM (wave split)

Same transfer size `N` as pure TDM or pure VMEM. With `USE_TDM_EXEC=1` and
`MIX_TDM_WARPS=M` (`0 < M < nWaves`), each TG splits warps via a dedicated
`GpuTdmMixKernel` (separate from pure `GpuTdmKernel` / `GpuAsyncTdmKernel` so
VGPR budgets are not merged across unused paths):

- warps `[0, M)` → tensor TDM via `tdm::tdmCopyByTeam` on the first `K` floats
- warps `[M, nWaves)` → VMEM packed copy on the remaining `N-K` floats

`K = (N / nWaves) * M` (aligned down to 64 floats / 256B so TDM has no sub-row
tail; VMEM absorbs the remainder). `TDM_BLOCK_SIZE=256` → 8 waves;
`MIX_TDM_WARPS=2` → ~25% TDM / ~75% VMEM. Async path ignores mix (tensor-TDM only).
Default `--disp` for mix is `GpuTdmMixKernel/0-`.

```bash
./scripts/roccap_capture.py \
  --use-tdm-exec \
  --mix-tdm-warps 2 \
  --num-gpu-devices 2 \
  --tb-sim-heap-bytes 768M \
  --num-iterations 1 \
  --num-warmups 0 \
  --gfx-block-size 256 \
  --num-sub-exec 8 \
  --dispatchlog \
  --output-dir caps_mix \
  a2a 32M
```

**Output (example):** `transferbench_ffm_tdm_write_GBS_256_UNROLL_32_NSE_8_TEMP_3_MIX_TW_2_A2A_32M_2nproc_rank0.cap`
(Mixed names also include `UNROLL_*` / `TEMP_*` / `SCOPE_*` for the VMEM half when those flags are set.)

## Key environment variables

| Variable | Purpose |
|----------|---------|
| `TB_SIMULATION=1` | Enable listener-pool sim (set by wrapper) |
| `NUM_GPU_DEVICES` | Number of logical GPUs |
| `TB_SIM_HEAP_BYTES` | Bytes per listener pool (default 2G; use 512M+ for 128M transfers) |
| `TB_SERIAL_EXECUTORS=1` | One executor at a time (required for roccap) |
| `TB_CAPTURE_EXECUTOR=N` | Capture only logical GPU N (set by wrapper per cap) |
| `TB_HEAP_BASES_FILE` | Exact path for heap JSON (set by wrapper) |
| `GFX_*` / `TDM_*` | Standard TransferBench kernel tuning |
| `USE_DMA_EXEC=1` | SDMA executor (set by `--use-dma-exec`) |
| `HSA_ENABLE_SDMA=1` | Required for real SDMA (wrapper sets when using `--use-dma-exec`) |
| `GPU_MAX_HW_QUEUES` | Max parallel SDMA streams per GPU (auto `N-1` in wrapper) |
| `USE_HSA_DMA=1` | HSA async copy instead of hipMemcpy (optional `--use-hsa-dma`) |
| `USE_ASYNC_EXEC=1` | Async-to/from-LDS via `GpuTdmKernel` (set by `--use-async-exec`) |
| `USE_ASYNC_COPY=1` | Select async backend inside TDM executor (also set by `--use-async-exec`) |
| `MIX_TDM_WARPS=M` | With TDM: first M warps tensor-TDM, rest VMEM (same N; set by `--mix-tdm-warps`) |

Do **not** set `TB_ROCCAP_FAST_EXIT=1` — it truncates caps on FFM.

## Manual sim run (no roccap)

```bash
export TB_SIMULATION=1 NUM_GPU_DEVICES=2 TB_SIM_HEAP_BYTES=512M
export NUM_ITERATIONS=1 NUM_WARMUPS=0 A2A_DIRECT=1
./TransferBench a2a 128M
```
