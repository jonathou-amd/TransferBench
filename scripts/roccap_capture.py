#!/usr/bin/env python3
"""
Run TransferBench under roccap capture (Iris-style wrapper for FFM / pre-silicon).

Run inside the FFM singularity container (competes-arcadia-260806.sif):

  cd TransferBench-singularity
  source scripts/ffm_env.sh
  ./scripts/roccap_capture.py \\
    --num-gpu-devices 2 \\
    --tb-sim-heap-bytes 512M \\
    --gfx-block-size 1024 \\
    --gfx-unroll 8 \\
    --num-sub-exec 96 \\
    --gfx-temporal 3 \\
    a2a 128M

Uses the container's bundled roccap (4.10 via /workspace/rocplaycap-Linux).

Produces one .cap and matching heap_bases JSON per logical GPU (Iris-style pairing):
  transferbench_ffm_write_GBS_768_UNROLL_4_NSE_8_TEMP_3_A2A_32M_2nproc_rank0.cap
  transferbench_ffm_read_GBS_768_UNROLL_4_NSE_8_TEMP_3_A2A_32M_2nproc_rank0.cap
  transferbench_ffm_tdm_write_LDS_64K_GBS_256_NSE_64_A2A_32M_2nproc_rank0.cap
  transferbench_ffm_tdm_read_LDS_64K_GBS_256_NSE_64_A2A_32M_2nproc_rank0.cap
  transferbench_ffm_async_write_GBS_256_NSE_80_A2A_256M_2nproc_rank0.cap
  transferbench_ffm_async_read_GBS_256_NSE_80_A2A_256M_2nproc_rank0.cap
  transferbench_ffm_dma_write_GMQ_8_NSE_7_A2A_36M_8nproc_rank0.cap
  transferbench_ffm_dma_read_GMQ_8_NSE_7_A2A_36M_8nproc_rank0.cap

Basename prefix is transferbench_ffm; read/write from USE_REMOTE_READ (default write).
Modes: default=GMEM/GFX, --use-tdm-exec=tensor TDM, --use-async-exec=async-to/from-LDS,
--use-dma-exec=SDMA (default --disp 0-).
DMA uses SDMA (hipMemcpyDeviceToDeviceNoCU), not GFX kernels; default --disp is 0-.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_TB_BINARY = "TransferBench"
DEFAULT_DISP_GFX = "(GpuCopyKernel|GpuReduceKernel)/0-"
DEFAULT_HEAP_PREFIX = "a2a"  # legacy manual runs only; wrapper sets TB_HEAP_BASES_FILE
CAP_NAME_PREFIX = "transferbench_ffm"

# FFM singularity image layout (see /workspace/rocplaycap-Linux in container)
ROCCAP_BUNDLE = Path("/workspace/rocplaycap-Linux/opt/rocm")


def access_mode_tag(args: argparse.Namespace) -> str:
    """read = USE_REMOTE_READ=1 (remote read / local write); write = default."""
    remote_read = 0 if args.use_remote_read is None else args.use_remote_read
    return "read" if remote_read else "write"


def parse_size_bytes(raw: str) -> int:
    """Parse TransferBench-style sizes: 65536, 64K, 256M (K/M/G multiply by 1024)."""
    text = raw.strip()
    if not text:
        return 0
    unit = text[-1].upper()
    if unit in ("K", "M", "G"):
        try:
            val = int(text[:-1])
        except ValueError:
            sys.exit(f"Invalid size (expected integer or K/M/G suffix): {raw!r}")
    else:
        try:
            return int(text)
        except ValueError:
            sys.exit(f"Invalid size (expected integer or K/M/G suffix): {raw!r}")
    if unit == "G":
        val *= 1024
    if unit in ("G", "M"):
        val *= 1024
    if unit in ("G", "M", "K"):
        val *= 1024
    return val


def format_lds_basename_tag(raw: str, bytes_val: int) -> str:
    """Build LDS tag for cap basename, e.g. LDS_64K or LDS_MAX."""
    if bytes_val == 0:
        return "LDS_MAX"
    suffix = raw.strip().upper().rstrip("B")
    if suffix and suffix[-1] in "KMG":
        return f"LDS_{suffix}"
    for unit, scale in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if bytes_val >= scale and bytes_val % scale == 0:
            return f"LDS_{bytes_val // scale}{unit}"
    return f"LDS_{bytes_val}"


def parse_gfx_scope(value: str) -> int:
    """Parse GFX_SCOPE: -1/unset, 0..3, or wgp|se|dev|sys."""
    raw = value.strip().lower()
    aliases = {
        "-1": -1,
        "unset": -1,
        "default": -1,
        "0": 0,
        "wgp": 0,
        "cu": 0,
        "wavefront": 0,
        "1": 1,
        "se": 1,
        "workgroup": 1,
        "2": 2,
        "dev": 2,
        "device": 2,
        "agent": 2,
        "3": 3,
        "sys": 3,
        "system": 3,
    }
    if raw not in aliases:
        raise argparse.ArgumentTypeError(
            f"invalid --gfx-scope {value!r}; use -1/unset or 0..3 / wgp|se|dev|sys"
        )
    return aliases[raw]


def scope_basename_tag(scope: int) -> str:
    return {0: "WGP", 1: "SE", 2: "DEV", 3: "SYS"}.get(scope, str(scope))


def default_gpu_max_hw_queues(args: argparse.Namespace) -> int:
    """A2A_DIRECT needs one stream per peer GPU (N-1 parallel SDMA transfers)."""
    return max(1, args.num_gpu_devices - 1)


def build_capture_basename(args: argparse.Namespace) -> str:
    access = access_mode_tag(args)
    if args.use_dma_exec:
        parts = [f"{CAP_NAME_PREFIX}_dma_{access}"]
        gmq = args.gpu_max_hw_queues
        if gmq is None:
            gmq = default_gpu_max_hw_queues(args)
        parts.append(f"GMQ_{gmq}")
        if args.num_sub_exec is not None:
            parts.append(f"NSE_{args.num_sub_exec}")
        parts.append(f"{args.preset.upper()}_{args.transfer_size}")
        return "_".join(parts)

    # Async and tensor-TDM share GpuTdmKernel / LDS staging; distinguish in basename.
    if args.use_async_exec or args.use_tdm_exec:
        mode = "async" if args.use_async_exec else "tdm"
        parts = [f"{CAP_NAME_PREFIX}_{mode}_{access}"]
        if args.tdm_lds_bytes is not None:
            lds_raw = args.tdm_lds_bytes
            lds_val = parse_size_bytes(lds_raw)
            parts.append(format_lds_basename_tag(lds_raw, lds_val))
        if args.gfx_block_size is not None:
            parts.append(f"GBS_{args.gfx_block_size}")
        # Async hot-loop unroll / temporal reuse --gfx-unroll / --gfx-temporal (same as VMEM).
        if args.use_async_exec:
            unroll = 2 if args.gfx_unroll is None else args.gfx_unroll
            parts.append(f"UNROLL_{unroll}")
        if args.num_sub_exec is not None:
            parts.append(f"NSE_{args.num_sub_exec}")
        if args.use_async_exec and args.gfx_temporal is not None:
            parts.append(f"TEMP_{args.gfx_temporal}")
        if args.gfx_scope is not None and args.gfx_scope >= 0:
            parts.append(f"SCOPE_{scope_basename_tag(args.gfx_scope)}")
        parts.append(f"{args.preset.upper()}_{args.transfer_size}")
        return "_".join(parts)

    parts = [f"{CAP_NAME_PREFIX}_{access}"]
    if args.gfx_block_size is not None:
        parts.append(f"GBS_{args.gfx_block_size}")
    if args.gfx_unroll is not None:
        parts.append(f"UNROLL_{args.gfx_unroll}")
    if args.num_sub_exec is not None:
        parts.append(f"NSE_{args.num_sub_exec}")
    if args.gfx_temporal is not None:
        parts.append(f"TEMP_{args.gfx_temporal}")
    if args.gfx_scope is not None and args.gfx_scope >= 0:
        parts.append(f"SCOPE_{scope_basename_tag(args.gfx_scope)}")
    parts.append(f"{args.preset.upper()}_{args.transfer_size}")
    return "_".join(parts)


def build_cap_stem(basename: str, num_gpu_devices: int, gpu_id: int) -> str:
    """Suffix: {nproc}nproc_rank{rank} where nproc = NUM_GPU_DEVICES, rank = logical GPU index."""
    return f"{basename}_{num_gpu_devices}nproc_rank{gpu_id}"


def build_disp_filter(args: argparse.Namespace) -> str:
    if args.disp:
        return args.disp
    if args.kernel_regex:
        return f"{args.kernel_regex}/0-"
    if args.use_dma_exec:
        return "0-"
    if args.use_async_exec or args.use_tdm_exec:
        return "GpuTdmKernel/0-"
    return DEFAULT_DISP_GFX


def setup_ffm_paths() -> None:
    """Prefer the FFM-bundled roccap and its libraries when present."""
    roccap_bin = ROCCAP_BUNDLE / "bin" / "roccap"
    roccap_lib = ROCCAP_BUNDLE / "lib"
    if not roccap_bin.is_file():
        return

    path_parts = os.environ.get("PATH", "").split(":")
    if str(roccap_bin.parent) not in path_parts:
        os.environ["PATH"] = f"{roccap_bin.parent}:{os.environ.get('PATH', '')}"

    if roccap_lib.is_dir():
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        roccap_lib_str = str(roccap_lib)
        if roccap_lib_str not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{roccap_lib_str}:{ld}" if ld else roccap_lib_str


def apply_env(args: argparse.Namespace, heap_prefix: str) -> dict[str, str]:
    env = os.environ.copy()

    if args.tb_simulation:
        env["TB_SIMULATION"] = "1"
        env["TB_SERIAL_EXECUTORS"] = "1"

    if args.num_gpu_devices is not None:
        env["NUM_GPU_DEVICES"] = str(args.num_gpu_devices)
    if args.tb_sim_heap_bytes:
        env["TB_SIM_HEAP_BYTES"] = args.tb_sim_heap_bytes

    env["TB_SIM_GPU"] = str(args.tb_sim_gpu)
    env["TB_HEAP_BASES_PREFIX"] = heap_prefix

    if args.num_iterations is not None:
        env["NUM_ITERATIONS"] = str(args.num_iterations)
    if args.num_warmups is not None:
        env["NUM_WARMUPS"] = str(args.num_warmups)
    if args.gfx_block_size is not None:
        if args.use_async_exec or args.use_tdm_exec:
            env["TDM_BLOCK_SIZE"] = str(args.gfx_block_size)
        else:
            env["GFX_BLOCK_SIZE"] = str(args.gfx_block_size)
    if args.gfx_unroll is not None:
        env["GFX_UNROLL"] = str(args.gfx_unroll)
    elif args.use_async_exec:
        # a2a defaults GFX_UNROLL=2; set explicitly so async hot-loop unroll is deterministic.
        env["GFX_UNROLL"] = "2"
    if args.num_sub_exec is not None:
        env["NUM_SUB_EXEC"] = str(args.num_sub_exec)
    if args.gfx_temporal is not None:
        env["GFX_TEMPORAL"] = str(args.gfx_temporal)
    if args.gfx_scope is not None:
        env["GFX_SCOPE"] = str(args.gfx_scope)
    if args.a2a_direct is not None:
        env["A2A_DIRECT"] = str(args.a2a_direct)
    if args.tb_verbose:
        env["TB_VERBOSE"] = "1"
    else:
        env.pop("TB_VERBOSE", None)
    if args.use_dma_exec:
        env["USE_DMA_EXEC"] = "1"
        # FFM container often sets HSA_ENABLE_SDMA=0; without SDMA, DMA falls back to GFX blit kernels.
        env["HSA_ENABLE_SDMA"] = "1"
        gmq = args.gpu_max_hw_queues
        if gmq is None:
            gmq = default_gpu_max_hw_queues(args)
        env["GPU_MAX_HW_QUEUES"] = str(gmq)
    if args.use_hsa_dma:
        env["USE_HSA_DMA"] = "1"
    if args.use_async_exec:
        env["USE_ASYNC_EXEC"] = "1"
        env["USE_ASYNC_COPY"] = "1"
    if args.use_tdm_exec:
        env["USE_TDM_EXEC"] = "1"
    if args.tdm_lds_bytes is not None:
        env["TDM_LDS_BYTES"] = str(parse_size_bytes(args.tdm_lds_bytes))
    if args.use_remote_read is not None:
        env["USE_REMOTE_READ"] = str(args.use_remote_read)

    return env


def capture_env_for_gpu(base_env: dict[str, str], gpu_id: int, heap_bases_file: Path) -> dict[str, str]:
    env = base_env.copy()
    env["TB_CAPTURE_EXECUTOR"] = str(gpu_id)
    env["TB_HEAP_BASES_FILE"] = str(heap_bases_file)
    env.pop("TB_HEAP_BASES_PREFIX", None)
    env["TB_SERIAL_EXECUTORS"] = "1"
    env["ALWAYS_VALIDATE"] = "-1"
    return env


def resolve_tb_binary(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    found = shutil.which(raw)
    if found:
        return Path(found).resolve()
    candidate = REPO_ROOT / raw
    if candidate.is_file():
        return candidate.resolve()
    sys.exit(f"TransferBench binary not found: {raw} (also tried {candidate})")


def find_roccap() -> str:
    override = os.environ.get("ROCCAP")
    if override:
        if not Path(override).is_file():
            sys.exit(f"ROCCAP={override} is not a file")
        return override

    bundled = ROCCAP_BUNDLE / "bin" / "roccap"
    if bundled.is_file():
        return str(bundled)

    found = shutil.which("roccap")
    if found:
        return found

    sys.exit(
        "roccap not found. Run inside the FFM container, or set:\n"
        "  export PATH=/workspace/rocplaycap-Linux/opt/rocm/bin:$PATH\n"
        "  export LD_LIBRARY_PATH=/workspace/rocplaycap-Linux/opt/rocm/lib:$LD_LIBRARY_PATH"
    )


def describe_exit(rc: int) -> str:
    if rc >= 0:
        return f"exit code {rc}"
    sig = -rc
    try:
        return f"signal {sig} ({signal.Signals(sig).name})"
    except (ValueError, AttributeError):
        return f"signal {sig}"


def resolve_written_cap(requested: Path) -> Path:
    """roccap 4.10+ may write basename_0001.cap instead of basename.cap."""
    if requested.is_file() and requested.stat().st_size > 0:
        return requested

    stem = requested.name
    if stem.endswith(".cap"):
        stem = stem[:-4]
    parent = requested.parent
    suffix_matches = sorted(parent.glob(f"{stem}_*.cap"), key=lambda p: p.stat().st_mtime)
    for candidate in reversed(suffix_matches):
        if candidate.stat().st_size > 0:
            return candidate
    return requested


def validate_cap_file(cap_path: Path) -> tuple[bool, str]:
    """Return (ok, message). A non-empty file is not enough — roccap must finalize the tar."""
    if not cap_path.is_file() or cap_path.stat().st_size == 0:
        return False, "capture file is missing or empty"

    roccap_path = find_roccap()
    cap_path = cap_path.resolve()
    # roccap extract --meta writes roc_capture*.json into cwd; many prior captures in
    # the repo root trigger "Maximum number of filenames reached" even when the .cap is fine.
    validate_cwd = cap_path.parent
    try:
        completed = subprocess.run(
            [roccap_path, "extract", "--meta", cap_path.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=str(validate_cwd),
        )
    except OSError as exc:
        return False, f"roccap extract --meta failed to launch: {exc}"

    if completed.returncode == 0:
        return True, "roccap extract --meta succeeded"

    output = (completed.stdout or "").strip()
    if "Cannot locate metadata stream" in output:
        return False, "tar archive is truncated (metadata stream missing; roccap likely crashed during finalize)"
    if "Maximum number of filenames reached" in output:
        # Fallback: run from an empty temp dir (repo root may have many roc_capture_*.json).
        with tempfile.TemporaryDirectory(prefix="roccap_validate_") as tmp:
            try:
                retry = subprocess.run(
                    [roccap_path, "extract", "--meta", str(cap_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    cwd=tmp,
                )
            except OSError as exc:
                return False, f"roccap extract --meta failed to launch: {exc}"
            if retry.returncode == 0:
                return True, "roccap extract --meta succeeded (validated in temp dir)"
            output = (retry.stdout or "").strip() or output
    if output:
        return False, output.splitlines()[-1]
    return False, f"roccap extract --meta failed with exit code {completed.returncode}"


def run_command(argv: list[str], env: dict[str, str], cwd: Path, cap_path: Path | None = None) -> int:
    print(f"Executing: {' '.join(argv)}")
    sys.stdout.flush()
    try:
        completed = subprocess.run(argv, env=env, cwd=str(cwd), check=False)
    except OSError as exc:
        print(f"[ERROR] Failed to launch: {exc}", file=sys.stderr)
        return 127

    rc = completed.returncode
    actual_cap = resolve_written_cap(cap_path) if cap_path is not None else None
    if actual_cap is not None and actual_cap != cap_path:
        print(f"  roccap wrote: {actual_cap}", file=sys.stderr)

    if rc == 0:
        if actual_cap is not None:
            ok, msg = validate_cap_file(actual_cap)
            if ok:
                return 0
            print(f"[ERROR] Capture file failed validation: {msg}", file=sys.stderr)
            print(f"[ERROR] File: {actual_cap} ({actual_cap.stat().st_size} bytes)", file=sys.stderr)
            return 1
        return 0

    cap_ok = actual_cap is not None and actual_cap.is_file() and actual_cap.stat().st_size > 0
    if cap_ok:
        ok, msg = validate_cap_file(actual_cap)
        if ok:
            if rc == -signal.SIGSEGV:
                print(
                    f"[WARN] roccap exited with SIGSEGV during teardown, but capture validated OK "
                    f"({actual_cap.stat().st_size} bytes): {actual_cap}",
                    file=sys.stderr,
                )
            return 0
        print(f"[ERROR] Capture file failed validation: {msg}", file=sys.stderr)
        print(f"[ERROR] File: {actual_cap} ({actual_cap.stat().st_size} bytes)", file=sys.stderr)

    print(f"[ERROR] Command failed: {describe_exit(rc)}", file=sys.stderr)
    if cap_path is not None and cap_path.is_file() and cap_path.stat().st_size == 0:
        print(
            "[HINT] Capture file is empty. The --disp filter may not match what TransferBench "
            "launched. For GFX a2a with GFX_UNROLL>2, try:\n"
            "  --disp 'GpuReduceKernel/0-'\n"
            "  --disp '(GpuCopyKernel|GpuReduceKernel)/0-'\n"
            "For SDMA (--use-dma-exec), default --disp is '0-' (no GFX kernel dispatch).",
            file=sys.stderr,
        )
    if rc == -signal.SIGSEGV:
        print(
            "[HINT] Segfault during roccap teardown is common on FFM. If roccap extract --meta "
            "passes on the .cap, the capture is usable.\n"
            "       Truncated caps usually meant TransferBench called _exit(); rebuild after "
            "disabling TB_ROCCAP_FAST_EXIT.\n"
            "       Try: source scripts/ffm_env.sh\n"
            "       Then: ./scripts/roccap_capture.py --skip-roccap ...  (must succeed first)",
            file=sys.stderr,
        )
    return rc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture TransferBench under roccap with Iris-style naming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Positional PRESET SIZE example: a2a 32M",
    )

    parser.add_argument("preset", nargs="?", default="a2a")
    parser.add_argument("transfer_size", nargs="?", default="32M")

    parser.add_argument("--tb-binary", default=DEFAULT_TB_BINARY)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--name", default="", help="Override auto-generated .cap basename")
    parser.add_argument(
        "--tb-heap-bases-prefix",
        default="",
        help="Legacy: TB_HEAP_BASES_PREFIX for manual runs (wrapper uses TB_HEAP_BASES_FILE instead)",
    )

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--tb-simulation", action=argparse.BooleanOptionalAction, default=True)
    sim.add_argument("--num-gpu-devices", type=int, default=2)
    sim.add_argument("--tb-sim-heap-bytes", default="256M")
    sim.add_argument("--tb-sim-gpu", type=int, default=0)

    bench = parser.add_argument_group("benchmark env")
    bench.add_argument("--num-iterations", type=int, default=1)
    bench.add_argument("--num-warmups", type=int, default=0)
    bench.add_argument("--gfx-block-size", type=int, default=None)
    bench.add_argument(
        "--gfx-unroll",
        type=int,
        default=None,
        help="GFX_UNROLL for VMEM kernels, and async hot-loop unroll for --use-async-exec "
        "(legal: 1-8,16,32,48,56,64; async default 2).",
    )
    bench.add_argument("--num-sub-exec", type=int, default=None)
    bench.add_argument(
        "--gfx-temporal",
        type=int,
        default=None,
        help="GFX_TEMPORAL for VMEM kernels, and async load/store NT hints for --use-async-exec "
        "(0=none/RT, 1=NT load, 2=NT store, 3=both NT).",
    )
    bench.add_argument(
        "--gfx-scope",
        type=parse_gfx_scope,
        default=None,
        help="GFX_SCOPE for VMEM and async (0/wgp, 1/se, 2/dev, 3/sys). "
        "VMEM: unset keeps legacy loads/stores; set uses scoped global ops (dwordx4 via "
        "amdgcn builtins). Async: unset defaults to SYS.",
    )
    bench.add_argument("--a2a-direct", type=int, default=1, choices=[0, 1])
    bench.add_argument(
        "--use-remote-read",
        type=int,
        default=None,
        choices=[0, 1],
        help="0=write (local read, remote write; default), 1=read (remote read, local write). "
        "Affects USE_REMOTE_READ and cap basename (transferbench_ffm_write_* vs transferbench_ffm_read_*).",
    )
    bench.add_argument("--tb-verbose", action="store_true")
    bench.add_argument(
        "--use-dma-exec",
        action="store_true",
        help="Use GPU SDMA executor (USE_DMA_EXEC=1). Caps SDMA traffic via --disp 0- (no GFX kernel).",
    )
    bench.add_argument(
        "--gpu-max-hw-queues",
        type=int,
        default=None,
        metavar="N",
        help="GPU_MAX_HW_QUEUES for DMA/BMA parallelism. Default with --use-dma-exec: NUM_GPU_DEVICES-1.",
    )
    bench.add_argument(
        "--use-hsa-dma",
        action="store_true",
        help="Use hsa_amd_memory_async_copy instead of hipMemcpyAsync (USE_HSA_DMA=1).",
    )
    bench.add_argument(
        "--use-tdm-exec",
        action="store_true",
        help="Use tensor-TDM executor (USE_TDM_EXEC=1) via GpuTdmKernel + tdm::tdmCopy.",
    )
    bench.add_argument(
        "--use-async-exec",
        action="store_true",
        help="Use async-to/from-LDS copy backend (USE_ASYNC_EXEC=1) via same GpuTdmKernel + async::tdmCopy. "
        "Mutually exclusive with --use-tdm-exec / --use-dma-exec.",
    )
    bench.add_argument(
        "--tdm-lds-bytes",
        default=None,
        metavar="SIZE",
        help="TDM/async LDS staging bytes per threadblock (e.g. 64K, 65536, 0=device max). "
        "Sets TDM_LDS_BYTES and adds LDS_<SIZE> to TDM/async cap basenames.",
    )

    roccap = parser.add_argument_group("roccap")
    roccap.add_argument("-k", "--kernel-regex", default="")
    roccap.add_argument("--disp", default="")
    roccap.add_argument("--skip-roccap", action="store_true")
    roccap.add_argument(
        "--roccap-loglevel",
        default="info",
        choices=["fatal", "error", "warn", "info", "debug", "trace"],
        help="Use info unless debugging roccap itself (trace is very noisy)",
    )
    roccap.add_argument(
        "--capture-gpu",
        type=int,
        default=-1,
        help="Capture only this logical GPU index (default: all 0..num_gpu_devices-1)",
    )
    roccap.add_argument(
        "--dispatchlog",
        action="store_true",
        help="Write roccap --dispatchlog sidecar per GPU (.dispatch.log)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.use_dma_exec and (args.use_tdm_exec or args.use_async_exec):
        print("[WARN] Both --use-dma-exec and TDM/async set; using DMA", file=sys.stderr)
        args.use_tdm_exec = False
        args.use_async_exec = False
    elif args.use_async_exec and args.use_tdm_exec:
        print("[WARN] Both --use-async-exec and --use-tdm-exec set; using async", file=sys.stderr)
        args.use_tdm_exec = False

    setup_ffm_paths()
    os.chdir(REPO_ROOT)

    capture_basename = args.name or build_capture_basename(args)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_binary = resolve_tb_binary(args.tb_binary)
    base_env = apply_env(args, args.tb_heap_bases_prefix or DEFAULT_HEAP_PREFIX)
    disp_filter = build_disp_filter(args)

    gpu_ids = (
        [args.capture_gpu]
        if args.capture_gpu >= 0
        else list(range(args.num_gpu_devices))
    )

    print("TransferBench roccap capture (one .cap per logical GPU)")
    print(f"  cwd:        {REPO_ROOT}")
    print(f"  binary:     {tb_binary}")
    print(f"  args:       {args.preset} {args.transfer_size}")
    print(f"  basename:   {capture_basename}_<nproc>nproc_rank<N>.{{cap,heap_bases.json}}")
    print(f"  roccap:     {find_roccap() if not args.skip_roccap else '(skipped)'}")
    print(f"  --disp:     {disp_filter}")
    print(f"  gpu ids:    {gpu_ids}")

    tb_argv = [str(tb_binary), args.preset, args.transfer_size]
    overall_rc = 0

    for gpu_id in gpu_ids:
        cap_stem = build_cap_stem(capture_basename, args.num_gpu_devices, gpu_id)
        cap_path = output_dir / f"{cap_stem}.cap"
        heap_bases_path = output_dir / f"{cap_stem}_heap_bases.json"
        run_env = capture_env_for_gpu(base_env, gpu_id, heap_bases_path)

        print(f"\n=== Logical GPU {gpu_id} ===")
        print(f"  cap file:   {cap_path}")
        print(f"  heap json:  {heap_bases_path}")
        print("  env:")
        for key in sorted(k for k in run_env if k.startswith(("TB_", "NUM_", "GFX_", "TDM_", "A2A_", "USE_", "ALWAYS_"))):
            print(f"    {key}={run_env[key]}")

        if args.skip_roccap:
            rc = run_command(tb_argv, run_env, REPO_ROOT)
        else:
            roccap_path = find_roccap()
            roccap_argv = [
                roccap_path,
                "capture",
                "--loglevel",
                args.roccap_loglevel,
                "--file",
                str(cap_path),
                "--disp",
                disp_filter,
            ]
            if args.dispatchlog:
                dispatch_log = output_dir / f"{cap_stem}.dispatch.log"
                roccap_argv.extend(["--dispatchlog", str(dispatch_log)])
            roccap_argv.extend([str(tb_binary), args.preset, args.transfer_size])
            rc = run_command(roccap_argv, run_env, REPO_ROOT, cap_path=cap_path)

        if rc == 0 and not heap_bases_path.is_file():
            print(f"[ERROR] Missing heap_bases JSON: {heap_bases_path}", file=sys.stderr)
            rc = 1

        if rc != 0:
            overall_rc = rc

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
