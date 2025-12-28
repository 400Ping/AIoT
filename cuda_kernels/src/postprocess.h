// src/postprocess.h
#pragma once

struct PostProcessParams {
    int img_w;
    int img_h;
    float conf_thresh;
    int target_class;
};

void launch_heatmap_generation(
    const float* d_detections, 
    float* d_heatmap, 
    int num_boxes, 
    PostProcessParams params, 
    float decay_rate
);