// cuda_kernels/tests/test_postprocess.cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "postprocess.h"

// 錯誤檢查巨集
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    // --- 1. 設定參數 ---
    PostProcessParams params;
    params.img_w = 20; 
    params.img_h = 20;
    params.conf_thresh = 0.5f;
    params.target_class = 0; 

    // --- 2. 準備資料 ---
    // 只有一個 Box 在中心 (10, 10)
    int num_boxes = 1;
    std::vector<float> h_dets = {
        0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0.0f 
    };

    size_t det_size = h_dets.size() * sizeof(float);
    size_t map_size = params.img_w * params.img_h * sizeof(float);

    // Device Memory
    float *d_dets, *d_heatmap;
    CHECK_CUDA(cudaMalloc(&d_dets, det_size));
    CHECK_CUDA(cudaMalloc(&d_heatmap, map_size));

    CHECK_CUDA(cudaMemcpy(d_dets, h_dets.data(), det_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_heatmap, 0, map_size)); // 初始為 0

    // --- 3. 測試 Step 1: 第一次生成 (Frame 1) ---
    std::cout << "Running Frame 1 (Generation)..." << std::endl;
    // 第一次 decay 沒影響 (因為底是 0)，加上一個 Box
    launch_heatmap_generation(d_dets, d_heatmap, num_boxes, params, 0.9f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 取回結果檢查 Frame 1 數值
    std::vector<float> h_map1(params.img_w * params.img_h);
    CHECK_CUDA(cudaMemcpy(h_map1.data(), d_heatmap, map_size, cudaMemcpyDeviceToHost));
    
    float center_val_1 = h_map1[10 * params.img_w + 10];
    std::cout << "Frame 1 Center Value: " << center_val_1 << std::endl;

    if (center_val_1 < 1.0f) {
        std::cerr << "[FAIL] Frame 1 value too low, heatmap not generated?" << std::endl;
        return 1;
    }

    // --- 4. 測試 Step 2: 第二次生成 (Frame 2 - 純 Decay) ---
    std::cout << "Running Frame 2 (Decay only)..." << std::endl;
    
    // 這裡我們傳入 num_boxes = 0，模擬沒有新偵測到的物件
    // 預期：舊的熱點應該會衰減 (乘上 decay_rate)
    float decay_rate = 0.5f; // 設誇張一點方便檢查
    launch_heatmap_generation(d_dets, d_heatmap, 0, params, decay_rate);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 取回結果檢查 Frame 2 數值
    std::vector<float> h_map2(params.img_w * params.img_h);
    CHECK_CUDA(cudaMemcpy(h_map2.data(), d_heatmap, map_size, cudaMemcpyDeviceToHost));

    float center_val_2 = h_map2[10 * params.img_w + 10];
    std::cout << "Frame 2 Center Value (after 0.5 decay): " << center_val_2 << std::endl;

    // --- 5. 驗證 ---
    // 容許一點點浮點誤差
    float expected = center_val_1 * decay_rate;
    if (abs(center_val_2 - expected) < 0.01f) {
        std::cout << "[PASS] Decay logic works correctly." << std::endl;
    } else {
        std::cout << "[FAIL] Decay failed. Expected " << expected << " but got " << center_val_2 << std::endl;
        return 1;
    }

    cudaFree(d_dets);
    cudaFree(d_heatmap);
    return 0;
}