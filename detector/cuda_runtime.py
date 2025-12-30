#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可重用的 CUDA YOLO 前處理/推論工具，集中管理 cuda_lib 載入與 GPU 緩衝區。
"""

from pathlib import Path
import sys
from typing import Tuple, Dict, Any

import torch
from ultralytics import YOLO


def find_cuda_paths():
    project_root = Path(__file__).resolve().parent.parent
    cuda_dir = project_root / "cuda_kernels"
    candidates = [
        cuda_dir / "build",
        cuda_dir / "build" / "Release",
        cuda_dir / "build" / "Debug",
        cuda_dir / "build" / "lib",
    ]
    return [p for p in candidates if p.exists()]


def diagnose_cuda_lib():
    paths = find_cuda_paths()
    search_list = [str(p) for p in paths]
    print(f"[Info] 搜尋 cuda_lib 路徑：{search_list or '無'}")

    found_files = []
    for p in paths:
        for f in p.iterdir():
            if "cuda_lib" in f.name and (f.suffix in {".so", ".pyd"}):
                found_files.append(f)
                print(f"[Info] 發現檔案：{f}")

    try:
        import cuda_lib  # type: ignore

        print(f"[Success] 成功載入 cuda_lib: {cuda_lib.__file__}")
        return True
    except Exception as exc:
        print(f"[Error] 匯入 cuda_lib 失敗: {exc}")
        if not found_files:
            print("[Hint] 在搜尋路徑中找不到 cuda_lib 相關檔案，請先在 AIoT/cuda_kernels 編譯。")
        return False


def load_cuda_lib(verbose: bool = False):
    """將 cuda_kernels/build* 加入 sys.path 並載入 cuda_lib。"""
    added = []
    for p in find_cuda_paths():
        sys.path.insert(0, str(p))
        added.append(str(p))

    if verbose:
        print(f"[Info] 已加入搜尋路徑：{added or '無'}")

    try:
        import cuda_lib  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"載入 cuda_lib 失敗，已加入的搜尋路徑：{added or '無'}；請先在 AIoT/cuda_kernels 下 CMake/編譯。"
        ) from exc

    return cuda_lib


class CudaHelmetPipeline:
    """封裝 CUDA 前處理 + YOLO 推論 + 熱度圖後處理。"""

    def __init__(self, model_path: str, net_size: int = 640, max_boxes: int = 100, verbose: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("找不到 CUDA 裝置，請確認有 GPU 環境並安裝對應 CUDA。")

        self.cuda_lib = load_cuda_lib(verbose=verbose)
        self.device = torch.device("cuda")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.net_w = net_size
        self.net_h = net_size
        self.max_boxes = max_boxes

        # 會在第一次看到影像時配置
        self.src_w = None
        self.src_h = None
        self.d_src = None
        self.d_dst = None
        self.d_dets = None
        self.d_heatmap = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_buffers(self, frame):
        """根據來源影像大小配置 GPU 緩衝區。"""
        h, w, _ = frame.shape
        if self.src_w == w and self.src_h == h:
            return

        self.src_w = w
        self.src_h = h

        self.d_src = torch.empty((h, w, 3), dtype=torch.uint8, device=self.device)
        self.d_dst = torch.empty((1, 3, self.net_h, self.net_w), dtype=torch.float32, device=self.device)
        self.d_dets = torch.zeros((self.max_boxes, 6), dtype=torch.float32, device=self.device)
        self.d_heatmap = torch.zeros((h, w), dtype=torch.float32, device=self.device)

    def run_frame(self, frame, conf_thresh: float = 0.5, target_class: int = 0, decay: float = 0.95) -> Dict[str, Any]:
        """
        對單張影像做前處理 + YOLO 推論 + 熱度圖。
        回傳 dict: {heatmap, num_boxes, summary, detections}
        """
        if frame is None:
            raise ValueError("frame 不可為 None")

        self.ensure_buffers(frame)
        self._preprocess(frame)

        with torch.no_grad():
            results = self.model(self.d_dst, verbose=False)

        num_boxes = self._copy_detections(results)
        heatmap = self._postprocess_heatmap(num_boxes, conf_thresh, target_class, decay)

        summary = self._summarize_boxes(num_boxes)
        return {
            "heatmap": heatmap,
            "num_boxes": num_boxes,
            "summary": summary,
            "results": results,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _preprocess(self, frame):
        # Host -> Device
        self.d_src.copy_(torch.from_numpy(frame).to(self.device, non_blocking=True))
        self.cuda_lib.preprocess_ptr(
            self.d_src.data_ptr(),
            self.d_dst.data_ptr(),
            self.src_w,
            self.src_h,
            self.net_w,
            self.net_h,
        )

    def _copy_detections(self, results) -> int:
        if not results or results[0].boxes is None:
            return 0

        det = results[0].boxes
        boxes = torch.cat([det.xywhn, det.conf.unsqueeze(1), det.cls.unsqueeze(1)], dim=1)
        num_boxes = min(len(boxes), self.max_boxes)
        self.d_dets[:num_boxes] = boxes[:num_boxes]
        return num_boxes

    def _postprocess_heatmap(self, num_boxes: int, conf_thresh: float, target_class: int, decay: float):
        self.cuda_lib.postprocess_ptr(
            self.d_dets.data_ptr(),
            self.d_heatmap.data_ptr(),
            num_boxes,
            self.src_w,
            self.src_h,
            float(conf_thresh),
            int(target_class),
            float(decay),
        )
        return self.d_heatmap.cpu().numpy()

    def _summarize_boxes(self, num_boxes: int) -> Dict[str, Any]:
        if num_boxes == 0:
            return {"num_people": 0, "num_no_helmet": 0, "status": "safe"}

        boxes = self.d_dets[:num_boxes].detach().cpu()
        cls_col = boxes[:, 5].to(torch.int32)

        # Dataset mapping: 0 helmet, 1 person, 2 head (no helmet)
        num_people = int((cls_col == 1).sum().item())
        num_no_helmet = int((cls_col == 2).sum().item())
        status = "unsafe" if num_no_helmet > 0 else "safe"

        return {"num_people": num_people, "num_no_helmet": num_no_helmet, "status": status}
