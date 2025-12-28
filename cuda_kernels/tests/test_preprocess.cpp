#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "preprocess.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    // 1. 設定測試參數
    int src_w = 4;
    int src_h = 4;
    int dst_w = 2; // 縮小測試
    int dst_h = 2;
    
    // 模擬一張 4x4 的紅色圖片 (BGR 格式: 0, 0, 255)
    // 注意：如果是真實 YOLO 預處理，這裡需要考慮 BGR -> RGB
    std::vector<unsigned char> h_src(src_w * src_h * 3, 0);
    for(int i=0; i<src_w*src_h; i++) {
        h_src[i*3 + 0] = 0;   // B
        h_src[i*3 + 1] = 0;   // G
        h_src[i*3 + 2] = 255; // R
    }

    // 2. 準備 Device 記憶體
    unsigned char* d_src;
    float* d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, h_src.size()));
    CHECK_CUDA(cudaMalloc(&d_dst, dst_w * dst_h * 3 * sizeof(float)));

    // 3. 執行 Kernel
    CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), h_src.size(), cudaMemcpyHostToDevice));
    
    std::cout << "Running Preprocess Kernel..." << std::endl;
    launch_preprocess(d_src, d_dst, src_w, src_h, dst_w, dst_h);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 檢查結果
    std::vector<float> h_dst(dst_w * dst_h * 3);
    CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, dst_w * dst_h * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    // 驗證：縮小後的像素應該仍然接近紅色 (R channel ~= 1.0)
    // DST 是 CHW 排列
    // Channel 0 (R 假設你已經修復了 BGR->RGB，或者如果沒修復這裡是 B)
    // 讓我們檢查第一個像素的值
    int pixel_idx = 0;
    float ch0 = h_dst[pixel_idx]; // Plane 0
    float ch1 = h_dst[pixel_idx + dst_w * dst_h]; // Plane 1
    float ch2 = h_dst[pixel_idx + 2 * dst_w * dst_h]; // Plane 2

    std::cout << "Pixel(0,0) values: " << ch0 << ", " << ch1 << ", " << ch2 << std::endl;

    // 清理
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}