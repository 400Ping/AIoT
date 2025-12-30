#pragma once
// 設定前處理的 mean/std 正規化；預設停用（等同除以 255）
void set_preprocess_normalization(const float mean[3], const float std[3], bool enable);

// 主要前處理入口：letterbox + 雙線性插值 + (選擇性) mean/std 正規化
void launch_preprocess(const unsigned char* d_src, float* d_dst, int src_w, int src_h, int dst_w, int dst_h);
