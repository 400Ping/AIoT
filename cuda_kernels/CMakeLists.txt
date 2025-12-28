cmake_minimum_required(VERSION 3.18)
project(CudaKernels LANGUAGES CXX CUDA)

# =================================================================
# 1. 核心修復：允許 VS2022 使用舊版 CUDA
# =================================================================
add_definitions(-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)

# =================================================================
# 2. 關鍵修復：編碼與巨集
# =================================================================
add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)

# 【修正點】只針對 C++ (.cpp) 檔案加上 /utf-8
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

# 【修正點】針對 CUDA (.cu) 檔案，透過 -Xcompiler 傳遞 /utf-8
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # 注意：使用 -Xcompiler=/utf-8 確保 NVCC 知道這是傳給 host compiler 的
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
endif()

# =================================================================
# 3. 專案設定
# =================================================================
# 為了保險起見，我們退回到 C++14 (CUDA 11.3 的原生支援標準)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

include_directories(src)

# 建立 Library
add_library(kernel_lib STATIC src/preprocess.cu src/postprocess.cu)

# 連結測試執行檔
add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE kernel_lib)

add_executable(test_postprocess tests/test_postprocess.cpp)
target_link_libraries(test_postprocess PRIVATE kernel_lib)

find_package(pybind11 REQUIRED)

pybind11_add_module(cuda_lib src/bindings.cpp)
target_link_libraries(cuda_lib PRIVATE kernel_lib)