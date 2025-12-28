#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
// [重要] 新增這行以使用 cudaDeviceSynchronize
#include <cuda_runtime.h> 
#include "preprocess.h"
#include "postprocess.h"

namespace py = pybind11;

// Wrapper for Preprocess
void py_preprocess_ptr(long long d_src_addr, long long d_dst_addr, int src_w, int src_h, int dst_w, int dst_h) {
    launch_preprocess(
        reinterpret_cast<unsigned char*>(d_src_addr),
        reinterpret_cast<float*>(d_dst_addr),
        src_w, src_h, dst_w, dst_h
    );
}

// Host 端直接餵 numpy array，並回傳 CHW 排列的 float32 numpy
pybind11::array_t<float> py_preprocess(pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> src, int dst_w, int dst_h) {
    auto buf = src.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw pybind11::value_error("預期輸入 shape 為 [H, W, 3]");
    }

    int src_h = static_cast<int>(buf.shape[0]);
    int src_w = static_cast<int>(buf.shape[1]);
    size_t src_bytes = static_cast<size_t>(src_w) * src_h * 3;
    size_t dst_bytes = static_cast<size_t>(dst_w) * dst_h * 3 * sizeof(float);

    unsigned char* d_src = nullptr;
    float* d_dst = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_src, src_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc d_src 失敗");
    }
    err = cudaMalloc(&d_dst, dst_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        throw std::runtime_error("cudaMalloc d_dst 失敗");
    }

    err = cudaMemcpy(d_src, buf.ptr, src_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        throw std::runtime_error("cudaMemcpy Host->Device 失敗");
    }

    launch_preprocess(d_src, d_dst, src_w, src_h, dst_w, dst_h);
    cudaDeviceSynchronize();

    // 回傳 shape [3, dst_h, dst_w] 的 CHW
    pybind11::array_t<float> output({3, dst_h, dst_w});
    auto out_buf = output.request();

    err = cudaMemcpy(out_buf.ptr, d_dst, dst_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy Device->Host 失敗");
    }

    return output;
}

// Wrapper for Postprocess
void py_postprocess_ptr(
    long long d_dets_addr,    
    long long d_heatmap_addr, 
    int num_boxes,            
    int img_w, int img_h,     
    float conf_thresh,        
    int target_class,         
    float decay_rate          
) {
    PostProcessParams params;
    params.img_w = img_w;
    params.img_h = img_h;
    params.conf_thresh = conf_thresh;
    params.target_class = target_class;

    launch_heatmap_generation(
        reinterpret_cast<float*>(d_dets_addr),
        reinterpret_cast<float*>(d_heatmap_addr),
        num_boxes,
        params,
        decay_rate
    );

    // 等待 GPU 完成
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(cuda_lib, m) {
    m.doc() = "CUDA Kernels via PyBind11";

    m.def("preprocess", &py_preprocess, "Preprocess numpy array and return CHW float32");
    m.def("preprocess_ptr", &py_preprocess_ptr, "Preprocess taking GPU pointers");
    m.def("postprocess_ptr", &py_postprocess_ptr, "Postprocess heatmap generation");
}
