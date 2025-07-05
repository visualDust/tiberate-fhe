#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

py::dict get_cuda_device_properties() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  py::dict result;

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    py::dict device_info;

    // Basic info
    device_info["name"] = std::string(prop.name);
    device_info["major"] = prop.major;
    device_info["minor"] = prop.minor;
    device_info["computeCapability"] =
        std::to_string(prop.major) + "." + std::to_string(prop.minor);

    // Memory sizes
    device_info["totalGlobalMem"] = static_cast<uint64_t>(prop.totalGlobalMem);
    device_info["sharedMemPerBlock"] = prop.sharedMemPerBlock;
    device_info["sharedMemPerMultiprocessor"] = prop.sharedMemPerMultiprocessor;
    device_info["totalConstMem"] = prop.totalConstMem;
    device_info["l2CacheSize"] = prop.l2CacheSize;
    device_info["memoryBusWidth"] = prop.memoryBusWidth;
    device_info["memoryClockRate"] = prop.memoryClockRate;
    device_info["memPitch"] = prop.memPitch;
    device_info["textureAlignment"] = prop.textureAlignment;
    device_info["texturePitchAlignment"] = prop.texturePitchAlignment;

    // Execution info
    device_info["multiProcessorCount"] = prop.multiProcessorCount;
    device_info["maxThreadsPerBlock"] = prop.maxThreadsPerBlock;
    device_info["maxThreadsPerMultiProcessor"] =
        prop.maxThreadsPerMultiProcessor;
    device_info["regsPerBlock"] = prop.regsPerBlock;
    device_info["regsPerMultiprocessor"] = prop.regsPerMultiprocessor;
    device_info["warpSize"] = prop.warpSize;
    device_info["clockRate"] = prop.clockRate;
    device_info["asyncEngineCount"] = prop.asyncEngineCount;

    // Launch configuration
    device_info["maxThreadsDim"] = std::vector<int>{
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]};
    device_info["maxGridSize"] = std::vector<int>{
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]};

    // Unified memory and managed memory support
    device_info["unifiedAddressing"] =
        static_cast<bool>(prop.unifiedAddressing);
    device_info["managedMemory"] = static_cast<bool>(prop.managedMemory);
    device_info["concurrentManagedAccess"] =
        static_cast<bool>(prop.concurrentManagedAccess);
    device_info["canMapHostMemory"] = static_cast<bool>(prop.canMapHostMemory);

    // Capabilities
    device_info["deviceOverlap"] = static_cast<bool>(prop.deviceOverlap);
    device_info["cooperativeLaunch"] =
        static_cast<bool>(prop.cooperativeLaunch);
    device_info["cooperativeMultiDeviceLaunch"] =
        static_cast<bool>(prop.cooperativeMultiDeviceLaunch);
    device_info["isMultiGpuBoard"] = static_cast<bool>(prop.isMultiGpuBoard);

    result[py::cast(i)] = device_info;
  }

  return result;
}

PYBIND11_MODULE(cuda_info, m) {
  m.def("get_cuda_device_properties",
        &get_cuda_device_properties,
        "Return detailed CUDA device properties");
}
