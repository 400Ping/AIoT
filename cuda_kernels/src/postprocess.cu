#include "postprocess.h"
#include <cuda_runtime.h>
#include <stdio.h> 

// [重要] 移除了 <cmath> 以避開衝突

// Kernel 1: Decay
__global__ void decay_kernel(float* heatmap, int size, float decay_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        heatmap[idx] *= decay_rate; 
    }
}

// Kernel 2: Heatmap
__global__ void heatmap_kernel(
    const float* __restrict__ dets, 
    float* __restrict__ heatmap,    
    int num_boxes,
    int width,
    int height,
    float conf_thresh,
    int target_class
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const float* box = &dets[idx * 6];
    float cx    = box[0];
    float cy    = box[1];
    float w     = box[2];
    float h     = box[3];
    float conf  = box[4];
    float cls   = box[5];

    if (conf < conf_thresh || (int)cls != target_class) return;

    float center_x = cx * width;
    float center_y = cy * height;
    float box_w    = w * width;
    float box_h    = h * height;

    // 使用 CUDA 內建 max/min (不需要 <algorithm> 或 <cmath>)
    int x_min = max(0, (int)(center_x - box_w / 2.0f));
    int y_min = max(0, (int)(center_y - box_h / 2.0f));
    int x_max = min(width - 1, (int)(center_x + box_w / 2.0f));
    int y_max = min(height - 1, (int)(center_y + box_h / 2.0f));

    float sigma_x = box_w / 4.0f;
    float sigma_y = box_h / 4.0f;
    
    sigma_x = max(1.0f, sigma_x);
    sigma_y = max(1.0f, sigma_y);

    for (int y = y_min; y <= y_max; y++) {
        float dy = y - center_y;
        float dy_sq = dy * dy;
        for (int x = x_min; x <= x_max; x++) {
            float dx = x - center_x;
            float dx_sq = dx * dx;

            // 使用 CUDA 內建 expf
            float weight = expf(-(dx_sq / (2 * sigma_x * sigma_x) + dy_sq / (2 * sigma_y * sigma_y)));
            
            atomicAdd(&heatmap[y * width + x], weight * 2.0f);
        }
    }
}

void launch_heatmap_generation(
    const float* d_detections, 
    float* d_heatmap, 
    int num_boxes, 
    PostProcessParams params,
    float decay_rate
) {
    cudaError_t err;

    int total_pixels = params.img_w * params.img_h;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    decay_kernel<<<blocks, threads>>>(d_heatmap, total_pixels, decay_rate);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Decay Kernel Error: %s\n", cudaGetErrorString(err));
        return;
    }

    if (num_boxes > 0) {
        int box_threads = 64; 
        int box_blocks = (num_boxes + box_threads - 1) / box_threads;

        heatmap_kernel<<<box_blocks, box_threads>>>(
            d_detections, 
            d_heatmap, 
            num_boxes, 
            params.img_w, params.img_h, 
            params.conf_thresh, 
            params.target_class
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Heatmap Kernel Error: %s\n", cudaGetErrorString(err));
        }
    }
}