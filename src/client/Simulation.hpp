/*
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

FFM / pre-silicon simulation support for TransferBench.

When TB_SIMULATION=1, logical GPU indices map to listener buffer regions on a
single physical GPU. GPU 0 is the active device; GPU 1..N-1 are local buffers
whose base addresses are exported for FFM/roccap (Iris-compatible JSON).
*/

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(__NVCC__)
#include <cuda_runtime.h>
#else
#include "hip/hip_runtime.h"
#endif

namespace TransferBench {
namespace Simulation {

namespace detail {

inline bool ParseBoolEnv(const char* name)
{
  const char* val = getenv(name);
  if (!val || !val[0]) return false;
  return !strcmp(val, "1") || !strcasecmp(val, "true") || !strcasecmp(val, "yes");
}

inline int ParseIntEnv(const char* name, int defaultValue)
{
  const char* val = getenv(name);
  if (!val || !val[0]) return defaultValue;
  return atoi(val);
}

inline size_t ParseSizeEnv(const char* name, size_t defaultValue)
{
  const char* val = getenv(name);
  if (!val || !val[0]) return defaultValue;
  size_t size = strtoull(val, nullptr, 10);
  char unit = val[strlen(val) - 1];
  switch (unit) {
  case 'G': case 'g': size *= 1024;
  case 'M': case 'm': size *= 1024;
  case 'K': case 'k': size *= 1024;
  default: break;
  }
  return size;
}

} // namespace detail

struct ListenerPool {
  void*  base     = nullptr;
  size_t capacity = 0;
  size_t offset   = 0;
};

inline bool& EnabledFlag()
{
  static bool enabled = false;
  return enabled;
}

inline int& PhysicalDeviceId()
{
  static int deviceId = 0;
  return deviceId;
}

inline int& SimGpuId()
{
  static int simGpu = 0;
  return simGpu;
}

inline int& NumLogicalGpus()
{
  static int numGpus = 0;
  return numGpus;
}

inline std::vector<ListenerPool>& Pools()
{
  static std::vector<ListenerPool> pools;
  return pools;
}

inline bool IsEnabled()
{
  return EnabledFlag();
}

// Run local GPU executors one at a time (required for roccap/FFM capture).
// Default: on in sim mode. Override with TB_SERIAL_EXECUTORS=0 or TB_PARALLEL_EXECUTORS=1.
inline bool SerialExecutors()
{
  if (!IsEnabled()) return detail::ParseBoolEnv("TB_SERIAL_EXECUTORS");
  if (detail::ParseBoolEnv("TB_PARALLEL_EXECUTORS")) return false;
  const char* val = getenv("TB_SERIAL_EXECUTORS");
  if (val && val[0]) return detail::ParseBoolEnv("TB_SERIAL_EXECUTORS");
  return true;
}

// When set (0..N-1), only run the matching logical GPU executor. Used for per-GPU roccap capture.
inline int CaptureExecutorIndex()
{
  return detail::ParseIntEnv("TB_CAPTURE_EXECUTOR", -1);
}

inline bool RocapCaptureMode()
{
  return detail::ParseBoolEnv("TB_ROCCAP_CAPTURE");
}

inline bool ShouldRunGpuExecutor(int exeIndex)
{
  int cap = CaptureExecutorIndex();
  if (cap < 0) return true;
  return exeIndex == cap;
}

inline int GetNumLogicalGpus()
{
  return NumLogicalGpus();
}

inline int GetSimGpu()
{
  return SimGpuId();
}

inline int GetPhysicalDeviceId()
{
  return PhysicalDeviceId();
}

inline bool OwnsPointer(void* ptr)
{
  if (!IsEnabled() || !ptr) return false;
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  for (auto const& pool : Pools()) {
    if (!pool.base) continue;
    auto base = reinterpret_cast<uintptr_t>(pool.base);
    if (addr >= base && addr < base + pool.capacity) return true;
  }
  return false;
}

inline void Shutdown();

inline void WriteHeapBasesJson()
{
  std::string path;
  const char* outFile = getenv("TB_HEAP_BASES_FILE");
  if (outFile && outFile[0]) {
    path = outFile;
  } else {
    const char* prefix = getenv("TB_HEAP_BASES_PREFIX");
    if (!prefix || !prefix[0]) prefix = "transferbench";
    path = std::string(prefix) + "_rank_" + std::to_string(SimGpuId()) + "_heap_bases.json";
  }

  std::ofstream out(path);
  if (!out) {
    fprintf(stderr, "[TransferBench][SIM] Failed to write %s\n", path.c_str());
    return;
  }

  int const jsonRank = CaptureExecutorIndex() >= 0 ? CaptureExecutorIndex() : SimGpuId();

  out << "{\n";
  out << "  \"rank\": " << jsonRank << ",\n";
  out << "  \"num_ranks\": " << NumLogicalGpus() << ",\n";
  out << "  \"sim_gpu\": " << SimGpuId() << ",\n";
  out << "  \"num_gpus\": " << NumLogicalGpus() << ",\n";
  out << "  \"physical_device\": " << PhysicalDeviceId() << ",\n";
  out << "  \"simulation\": true,\n";
  out << "  \"heap_bases\": [";
  for (int i = 0; i < NumLogicalGpus(); ++i) {
    if (i) out << ", ";
    out << "\"0x" << std::hex << reinterpret_cast<uintptr_t>(Pools()[i].base) << std::dec << "\"";
  }
  out << "]\n";
  out << "}\n";

  fprintf(stderr, "[TransferBench][SIM] Wrote listener heap bases to %s\n", path.c_str());
}

inline bool Init()
{
  if (!detail::ParseBoolEnv("TB_SIMULATION") && !detail::ParseBoolEnv("IRIS_SIMULATION")) {
    EnabledFlag() = false;
    return false;
  }

  int physicalCount = 0;
#if defined(__NVCC__)
  if (cudaGetDeviceCount(&physicalCount) != cudaSuccess) physicalCount = 0;
#else
  if (hipGetDeviceCount(&physicalCount) != hipSuccess) physicalCount = 0;
#endif
  if (physicalCount <= 0) {
    fprintf(stderr, "[TransferBench][SIM] No physical GPU found; simulation disabled\n");
    EnabledFlag() = false;
    return false;
  }

  PhysicalDeviceId() = 0;
  SimGpuId() = detail::ParseIntEnv("TB_SIM_GPU", detail::ParseIntEnv("LOCAL_RANK", 0));

  int numLogical = detail::ParseIntEnv("TB_SIM_GPUS", -1);
  if (numLogical <= 0) numLogical = detail::ParseIntEnv("NUM_GPU_DEVICES", -1);
  if (numLogical <= 0) numLogical = 2;
  NumLogicalGpus() = numLogical;

  size_t heapBytes = detail::ParseSizeEnv("TB_SIM_HEAP_BYTES", static_cast<size_t>(1) << 31);

#if defined(__NVCC__)
  cudaSetDevice(PhysicalDeviceId());
#else
  (void)hipSetDevice(PhysicalDeviceId());
#endif

  Pools().assign(NumLogicalGpus(), ListenerPool{});
  for (int i = 0; i < NumLogicalGpus(); ++i) {
    void* base = nullptr;
#if defined(__NVCC__)
    cudaError_t err = cudaMalloc(&base, heapBytes);
    if (err != cudaSuccess) {
#else
    hipError_t err = hipMalloc(&base, heapBytes);
    if (err != hipSuccess) {
#endif
      fprintf(stderr, "[TransferBench][SIM] hipMalloc failed for listener GPU %d (%zu bytes)\n", i, heapBytes);
      Shutdown();
      EnabledFlag() = false;
      return false;
    }
    Pools()[i].base     = base;
    Pools()[i].capacity = heapBytes;
    Pools()[i].offset   = 0;
#if defined(__NVCC__)
    cudaMemset(base, 0, heapBytes);
#else
    (void)hipMemset(base, 0, heapBytes);
#endif
  }

#if defined(__NVCC__)
  cudaDeviceSynchronize();
#else
  (void)hipDeviceSynchronize();
#endif

  EnabledFlag() = true;
  WriteHeapBasesJson();

  fprintf(stderr,
          "[TransferBench][SIM] Enabled: %d logical GPU(s) on physical device %d "
          "(%zu bytes per listener, sim_gpu=%d, serial_executors=%s",
          NumLogicalGpus(), PhysicalDeviceId(), heapBytes, SimGpuId(),
          SerialExecutors() ? "yes" : "no");
  if (CaptureExecutorIndex() >= 0)
    fprintf(stderr, ", capture_executor=GPU%d", CaptureExecutorIndex());
  if (RocapCaptureMode())
    fprintf(stderr, ", roccap_capture=1");
  fprintf(stderr, ")\n");
  return true;
}

inline void Shutdown()
{
  if (!IsEnabled()) return;
  for (auto& pool : Pools()) {
    if (pool.base) {
#if defined(__NVCC__)
      cudaFree(pool.base);
#else
      (void)hipFree(pool.base);
#endif
      pool.base = nullptr;
    }
    pool.offset = 0;
  }
  Pools().clear();
  EnabledFlag() = false;
}

inline hipError_t SetLogicalDevice(int logicalGpuIndex)
{
#if defined(__NVCC__)
  if (IsEnabled()) return cudaSetDevice(PhysicalDeviceId());
  return cudaSetDevice(logicalGpuIndex);
#else
  if (IsEnabled()) return hipSetDevice(PhysicalDeviceId());
  return hipSetDevice(logicalGpuIndex);
#endif
}

inline int PhysicalDeviceForQuery(int logicalGpuIndex)
{
  if (IsEnabled()) return PhysicalDeviceId();
  return logicalGpuIndex;
}

inline bool AllocateGpu(int logicalGpuIndex, size_t numBytes, void** memPtr)
{
  if (logicalGpuIndex < 0 || logicalGpuIndex >= NumLogicalGpus()) {
    fprintf(stderr,
            "[TransferBench][SIM] GPU index %d out of range [0, %d)\n",
            logicalGpuIndex, NumLogicalGpus());
    return false;
  }

  ListenerPool& pool = Pools()[logicalGpuIndex];
  size_t const alignment = 256;
  size_t alignedSize = (numBytes + alignment - 1) / alignment * alignment;

  if (pool.offset + alignedSize > pool.capacity) {
    fprintf(stderr,
            "[TransferBench][SIM] Listener pool GPU %d OOM (need %zu bytes, %zu free)\n",
            logicalGpuIndex, numBytes, pool.capacity - pool.offset);
    return false;
  }

  *memPtr = static_cast<char*>(pool.base) + pool.offset;
  pool.offset += alignedSize;
  return true;
}

} // namespace Simulation
} // namespace TransferBench
