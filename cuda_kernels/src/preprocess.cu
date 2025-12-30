#include "preprocess.h"
#include <cuda_runtime.h>

// [重要] 移除了 <math.h> 與 <algorithm> 以避開 VS2022 STL 衝突
// [重要] 移除了 #define NOMINMAX (CMakeLists.txt 已經定義了)

// mean/std 會放在 constant memory，避免每次呼叫都重新讀 host 端
__constant__ float c_mean[3] = {0.0f, 0.0f, 0.0f};
__constant__ float c_inv_std[3] = {1.0f, 1.0f, 1.0f};
__constant__ int c_norm_enabled = 0;  // 0: 關閉（預設）；1: 啟用 mean/std 正規化

__global__ void preprocess_kernel(
    const unsigned char* __restrict__ src, 
    float* __restrict__ dst, 
    int src_w, int src_h, 
    int dst_w, int dst_h,
    float scale, int ox, int oy
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // 1. 計算座標
    float src_x_f = (x - ox) / scale;
    float src_y_f = (y - oy) / scale;

    int dst_idx = y * dst_w + x;
    int channel_stride = dst_w * dst_h;

    const float PAD_COLOR = 114.0f / 255.0f;

    // 2. Padding 判定
    if (src_x_f < 0 || src_x_f >= src_w - 0.5f || src_y_f < 0 || src_y_f >= src_h - 0.5f) {
        dst[dst_idx] = PAD_COLOR;                    
        dst[dst_idx + channel_stride] = PAD_COLOR;   
        dst[dst_idx + 2 * channel_stride] = PAD_COLOR; 
        return;
    }

    // 3. 雙線性插值 (使用 CUDA 內建 floorf, min, max)
    int x_low = (int)floorf(src_x_f);
    int y_low = (int)floorf(src_y_f);
    
    int x_high = min(x_low + 1, src_w - 1);
    int y_high = min(y_low + 1, src_h - 1);
    
    x_low = max(0, x_low);
    y_low = max(0, y_low);

    float dx = src_x_f - x_low;
    float dy = src_y_f - y_low;
    
    float w1 = (1.0f - dx) * (1.0f - dy); 
    float w2 = dx * (1.0f - dy);          
    float w3 = (1.0f - dx) * dy;          
    float w4 = dx * dy;                   

    int src_pitch = src_w * 3;

    for (int c = 0; c < 3; c++) {
        // BGR 轉 RGB
        int src_c = 2 - c; 

        unsigned char v1 = src[y_low  * src_pitch + x_low  * 3 + src_c];
        unsigned char v2 = src[y_low  * src_pitch + x_high * 3 + src_c];
        unsigned char v3 = src[y_high * src_pitch + x_low  * 3 + src_c];
        unsigned char v4 = src[y_high * src_pitch + x_high * 3 + src_c];

        float val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;

        val = val / 255.0f;
        if (c_norm_enabled) {
            val = (val - c_mean[c]) * c_inv_std[c];
        }

        dst[dst_idx + c * channel_stride] = val;
    }
}

void set_preprocess_normalization(const float mean[3], const float std[3], bool enable) {
    float host_inv_std[3];
    for (int i = 0; i < 3; ++i) {
        // 避免除以 0
        host_inv_std[i] = (std[i] != 0.0f) ? (1.0f / std[i]) : 1.0f;
    }

    cudaMemcpyToSymbol(c_mean, mean, sizeof(float) * 3);
    cudaMemcpyToSymbol(c_inv_std, host_inv_std, sizeof(float) * 3);

    int flag = enable ? 1 : 0;
    cudaMemcpyToSymbol(c_norm_enabled, &flag, sizeof(int));
}

void launch_preprocess(const unsigned char* d_src, float* d_dst, int src_w, int src_h, int dst_w, int dst_h) {
    // 手動計算 min (取代 std::min)
    float scale_w = (float)dst_w / src_w;
    float scale_h = (float)dst_h / src_h;
    float scale = (scale_w < scale_h) ? scale_w : scale_h;

    // 手動實作 round: (int)(val + 0.5f)
    int new_unpad_w = (int)(src_w * scale + 0.5f);
    int new_unpad_h = (int)(src_h * scale + 0.5f);
    
    int ox = (dst_w - new_unpad_w) / 2;
    int oy = (dst_h - new_unpad_h) / 2;

    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block>>>(d_src, d_dst, src_w, src_h, dst_w, dst_h, scale, ox, oy);
}
