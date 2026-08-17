# TransferBench

TransferBench is a utility for benchmarking simultaneous copies between user-specified
CPU and GPU memory locations using CPUs/GPU kernels/DMA engines/NIC devices.

> [!NOTE]
> The published documentation is available at [TransferBench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `TransferBench/docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## FFM pre-silicon simulation and roccap capture

> Branch: `ffm-sim-roccap-capture` — [fork](https://github.com/jonathou-amd/TransferBench/tree/ffm-sim-roccap-capture) based on `candidate-1.70`.

Run TransferBench as **1 physical GPU / N logical GPUs** (listener heap pools on device 0),
export **`heap_bases.json`** for FFM replay, and capture **one roccap `.cap` per logical GPU**
via `scripts/roccap_capture.py`.

Full details: **[docs/FFM_ROCCAP.md](docs/FFM_ROCCAP.md)**

### Quick start (inside `competes-arcadia-260806.sif`)

```bash
singularity shell --bind /large_data:/large_data competes-arcadia-260806.sif

cd TransferBench
source scripts/ffm_env.sh
GPU_TARGETS=gfx1260 DISABLE_POD_COMM=1 make -j
```

Do **not** bind-mount host `libhsa`. Use container roccap 4.10 (via `ffm_env.sh`).

### GFX capture (GpuReduceKernel)

```bash
./scripts/roccap_capture.py \
  --num-gpu-devices 2 \
  --tb-sim-heap-bytes 512M \
  --gfx-block-size 1024 \
  --gfx-unroll 8 \
  --num-sub-exec 128 \
  --gfx-temporal 3 \
  --dispatchlog \
  --output-dir caps \
  a2a 128M
```

### TDM capture (GpuTdmKernel)

```bash
./scripts/roccap_capture.py \
  --use-tdm-exec \
  --num-gpu-devices 2 \
  --tb-sim-heap-bytes 512M \
  --gfx-block-size 256 \
  --num-sub-exec 128 \
  --dispatchlog \
  --output-dir caps \
  a2a 128M
```

Each run produces paired files under `caps/`:

- `<basename>_2nproc_rank<N>.cap`
- `<basename>_2nproc_rank<N>_heap_bases.json`
