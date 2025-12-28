#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 cuda_kernels 的 preprocess/postprocess 加速 YOLO 推論。
流程：
1. 讀取攝影機畫面 -> 複製到 GPU
2. 透過 cuda_lib.preprocess_ptr 做 resize/normalize
3. ultralytics YOLO 在 GPU 上推論
4. 透過 cuda_lib.postprocess_ptr 把結果轉成 heatmap 疊圖展示
按 q 離開。
"""

from pathlib import Path
import sys
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config


def load_cuda_lib():
    """將 cuda_kernels/build* 加入 sys.path 並載入 cuda_lib。"""
    project_root = Path(__file__).resolve().parent.parent
    cuda_dir = project_root / "cuda_kernels"
    candidates = [
        cuda_dir / "build",
        cuda_dir / "build" / "Release",
        cuda_dir / "build" / "Debug",
        cuda_dir / "build" / "lib",
    ]
    added = []
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p))
            added.append(str(p))

    try:
        import cuda_lib  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            f"載入 cuda_lib 失敗，已加入的搜尋路徑：{added or '無'}；請先在 AIoT/cuda_kernels 下 CMake/編譯。"
        ) from exc

    return cuda_lib


class CudaHelmetDemo:
    def __init__(self, source=0, model_path=None, net_size=640, max_boxes=100):
        if not torch.cuda.is_available():
            raise RuntimeError("找不到 CUDA 裝置，請確認有 GPU 環境並安裝對應 CUDA。")

        self.cuda_lib = load_cuda_lib()

        self.device = torch.device("cuda")
        self.net_w = net_size
        self.net_h = net_size
        self.max_boxes = max_boxes

        model_path = model_path or config.MODEL_PATH
        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 source={source}")

        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 預先配置 GPU 記憶體
        self.d_src = torch.empty(
            (self.src_h, self.src_w, 3), dtype=torch.uint8, device=self.device
        )
        self.d_dst = torch.empty(
            (1, 3, self.net_h, self.net_w), dtype=torch.float32, device=self.device
        )
        self.d_dets = torch.zeros(
            (self.max_boxes, 6), dtype=torch.float32, device=self.device
        )
        self.d_heatmap = torch.zeros(
            (self.src_h, self.src_w), dtype=torch.float32, device=self.device
        )

    def preprocess(self, frame: np.ndarray):
        # Host -> Device
        self.d_src.copy_(torch.from_numpy(frame).to(self.device, non_blocking=True))

        # CUDA kernel resize/normalize
        self.cuda_lib.preprocess_ptr(
            self.d_src.data_ptr(),
            self.d_dst.data_ptr(),
            self.src_w,
            self.src_h,
            self.net_w,
            self.net_h,
        )

    def postprocess_heatmap(self, num_boxes: int, conf_thresh=0.5, target_class=0):
        self.cuda_lib.postprocess_ptr(
            self.d_dets.data_ptr(),
            self.d_heatmap.data_ptr(),
            num_boxes,
            self.src_w,
            self.src_h,
            float(conf_thresh),
            int(target_class),
            0.95,  # decay
        )
        return self.d_heatmap.cpu().numpy()

    def run(self):
        print("開始 CUDA Helmet Demo，按 q 離開。")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("讀取影像失敗，結束。")
                    break

                # 預處理到 GPU
                self.preprocess(frame)

                # YOLO 推論 (直接吃 GPU tensor)
                results = self.model(self.d_dst, verbose=False)
                num_boxes = 0
                if results and results[0].boxes is not None:
                    det = results[0].boxes
                    boxes = torch.cat(
                        [det.xywhn, det.conf.unsqueeze(1), det.cls.unsqueeze(1)], dim=1
                    )
                    num_boxes = min(len(boxes), self.max_boxes)
                    self.d_dets[:num_boxes] = boxes[:num_boxes]

                # 後處理 heatmap
                heatmap = self.postprocess_heatmap(num_boxes, conf_thresh=0.5, target_class=0)
                heatmap_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                display = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

                cv2.imshow("AIoT CUDA Helmet", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    demo = CudaHelmetDemo()
    demo.run()


if __name__ == "__main__":
    main()
