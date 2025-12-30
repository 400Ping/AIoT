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
import argparse
import sys
import os
import cv2
import numpy as np
import torch
import termios
import tty
import time
import threading
import select

import config
from cuda_runtime import (
    CudaHelmetPipeline,
    diagnose_cuda_lib,
)

try:
    from motor_controller import MotorController
    MOTOR_AVAILABLE = True
except Exception as exc:  # pragma: no cover - runtime guard
    MotorController = None  # type: ignore
    MOTOR_AVAILABLE = False
    MOTOR_IMPORT_ERROR = exc


class CudaHelmetDemo:
    def __init__(self, source=0, model_path=None, net_size=640, max_boxes=100, enable_manual=True):
        self.pipeline = None
        self.stop_event = threading.Event()
        self.manual_thread = None
        if enable_manual:
            self.manual_thread = ManualControlThread(self.stop_event)

        self.device = torch.device("cuda")
        self.net_w = net_size
        self.net_h = net_size
        self.max_boxes = max_boxes

        model_path = model_path or config.MODEL_PATH
        if not Path(model_path).exists():
            # fallback to repo-local models/best.pt
            repo_model = Path(__file__).resolve().parent.parent / "models" / "best.pt"
            if repo_model.exists():
                print(f"[Warn] 模型檔不存在於 {model_path}，改用 {repo_model}")
                model_path = str(repo_model)
            else:
                raise FileNotFoundError(f"找不到模型檔：{model_path}，且預設 {repo_model} 也不存在")

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 source={source}，請使用 --source <index|video 路徑>")

        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pipeline = CudaHelmetPipeline(
            model_path=model_path,
            net_size=self.net_w,
            max_boxes=self.max_boxes,
        )
        self.model = self.pipeline.model
        # 先配置 buffer，必要時會自動調整
        dummy_frame = np.zeros((self.src_h, self.src_w, 3), dtype=np.uint8)
        self.pipeline.ensure_buffers(dummy_frame)

        if self.manual_thread:
            self.manual_thread.start()

    def run(self):
        print("開始 CUDA Helmet Demo，按 q 離開。")
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    print("讀取影像失敗，結束。")
                    break

                result = self.pipeline.run_frame(frame, conf_thresh=0.5, target_class=0, decay=0.95)
                heatmap = result["heatmap"]
                heatmap_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                display = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

                cv2.imshow("AIoT CUDA Helmet", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.manual_thread:
                self.stop_event.set()
                self.manual_thread.join(timeout=1.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDA Helmet demo + manual control")
    parser.add_argument(
        "--source",
        default=0,
        type=str,
        help="cv2.VideoCapture source (index or video path, default camera 0)",
    )
    parser.add_argument("--model", default=None, help="YOLO model path (default config.MODEL_PATH)")
    parser.add_argument("--net-size", type=int, default=640, help="network input size (square)")
    parser.add_argument("--max-boxes", type=int, default=100, help="max detections to keep on GPU")
    parser.add_argument("--diagnose", action="store_true", help="only檢查 cuda_lib 載入狀態後退出")
    parser.add_argument("--verbose", action="store_true", help="print search paths when loading cuda_lib")
    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="禁用 WASD 手動控制 (預設啟用，與偵測同時運行)",
    )
    return parser.parse_args(argv)


def poll_key(timeout=0.1):
    """非阻塞讀取單鍵，未按鍵時回傳 None。"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if not rlist:
            return None
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class ManualControlThread(threading.Thread):
    def __init__(self, stop_event):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.mc = MotorController() if MOTOR_AVAILABLE else None

    def run(self):
        if not self.mc:
            print(f"[Warn] 無法啟用手動控制：MotorController 載入失敗 ({MOTOR_IMPORT_ERROR})")
            return

        print(
            "W/A/S/D 控制小車，Space 停止，Q 離開：\n"
            "  w: 前進\n  s: 後退\n  a: 左轉\n  d: 右轉\n"
            "  space: 停止\n  q: 離開\n"
        )
        try:
            while not self.stop_event.is_set():
                key = poll_key()
                if not key:
                    continue
                key = key.lower()
                if key == "w":
                    self.mc.forward()
                elif key == "s":
                    self.mc.backward()
                elif key == "a":
                    self.mc.left()
                elif key == "d":
                    self.mc.right()
                elif key == " ":
                    self.mc.stop()
                elif key == "q":
                    self.mc.stop()
                    self.stop_event.set()
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.mc.stop()
        finally:
            self.mc.stop()


def main(argv=None):
    args = parse_args(argv)
    if args.diagnose:
        ok = diagnose_cuda_lib()
        sys.exit(0 if ok else 1)

    # 將數字字串轉成 int 以支援 camera index，其餘當成檔案路徑
    src_arg = int(args.source) if str(args.source).isdigit() else args.source

    demo = CudaHelmetDemo(
        source=src_arg,
        model_path=args.model,
        net_size=args.net_size,
        max_boxes=args.max_boxes,
        enable_manual=not args.no_manual,
    )
    if args.verbose:
        print(f"[Info] 使用 source={args.source}, model={demo.model.ckpt_path if hasattr(demo.model, 'ckpt_path') else args.model or config.MODEL_PATH}")
    demo.run()


if __name__ == "__main__":
    main()
