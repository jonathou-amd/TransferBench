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

Do **not** set `TB_ROCCAP_FAST_EXIT=1` — it truncates caps on FFM.

## Manual sim run (no roccap)

```bash
export TB_SIMULATION=1 NUM_GPU_DEVICES=2 TB_SIM_HEAP_BYTES=512M
export NUM_ITERATIONS=1 NUM_WARMUPS=0 A2A_DIRECT=1
./TransferBench a2a 128M
```
