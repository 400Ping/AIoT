#include <pybind11/pybind11.h>
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

    m.def("preprocess_ptr", &py_preprocess_ptr, "Preprocess taking GPU pointers");
    m.def("postprocess_ptr", &py_postprocess_ptr, "Postprocess heatmap generation");
}