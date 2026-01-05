#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUDA 版安全帽偵測 Demo：
- 開啟攝影機，顯示畫面（含 heatmap 疊圖）
- YOLO 偵測 + 違規事件上報 (Flask) + LED/蜂鳴器 (若有硬體)
- 無需自走車馬達控制；在筆電 / GPU 平台也能跑
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import torch

import config
from cuda_runtime import CudaHelmetPipeline, diagnose_cuda_lib

try:
    from hardware import HardwareController

    HARDWARE_AVAILABLE = True
    HARDWARE_ERR = None
except Exception as exc:  # pragma: no cover - runtime guard
    HARDWARE_AVAILABLE = False
    HARDWARE_ERR = exc


class DummyHardware:
    """在非樹莓派環境下的簡易 stub。"""

    def trigger_alarm(self):
        print("[Hardware] trigger_alarm (stub)")

    def clear_alarm(self):
        print("[Hardware] clear_alarm (stub)")

    def cleanup(self):
        pass


def save_violation_image(frame):
    os.makedirs(config.IMG_SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_{ts}.jpg"
    path = os.path.join(config.IMG_SAVE_DIR, filename)
    cv2.imwrite(path, frame)
    image_url = f"/static/violations/{filename}"
    return path, image_url


def send_event(summary, image_url):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "camera_id": config.CAMERA_ID,
        "status": summary["status"],
        "num_people": summary["num_people"],
        "num_no_helmet": summary["num_no_helmet"],
        "image_url": image_url,
    }
    try:
        requests.post(
            f"{config.SERVER_URL}/api/events",
            json=payload,
            timeout=1.0,
        )
    except Exception as e:
        print(f"[WARN] Failed to send event to server: {e}")


class ViolationTracker:
    def __init__(self, threshold_sec: float):
        self.threshold_sec = threshold_sec
        self.prev_event_state = "safe"
        self.event_state = "safe"
        self.unsafe_start_time = None
        self.unsafe_duration = 0.0

    def update(self, status: str, now: float):
        if status == "unsafe":
            if self.unsafe_start_time is None:
                self.unsafe_start_time = now
                self.unsafe_duration = 0.0
            else:
                self.unsafe_duration = now - self.unsafe_start_time
        else:
            self.unsafe_start_time = None
            self.unsafe_duration = 0.0

        self.prev_event_state = self.event_state
        self.event_state = "unsafe" if self.unsafe_duration >= self.threshold_sec else "safe"

        triggered = self.prev_event_state == "safe" and self.event_state == "unsafe"
        return triggered, self.unsafe_duration


class CudaHelmetCameraDemo:
    def __init__(self, source, model_path, net_size, max_boxes, unsafe_threshold, verbose=False):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 source={source}")

        self.pipeline = CudaHelmetPipeline(
            model_path=model_path,
            net_size=net_size,
            max_boxes=max_boxes,
            verbose=verbose,
        )

        # 先根據目前攝影機設定配置 buffer
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dummy = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        self.pipeline.ensure_buffers(dummy)

        if HARDWARE_AVAILABLE:
            self.hardware = HardwareController()
        else:
            print(f"[Warn] 無法載入硬體控制 (LED/蜂鳴器)：{HARDWARE_ERR}")
            print("       將使用 stub，僅列印訊息。")
            self.hardware = DummyHardware()

        self.tracker = ViolationTracker(unsafe_threshold)

    def run(self):
        print("開始 CUDA Helmet Camera Demo，按 q 離開。")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("讀取影像失敗，結束。")
                    break

                result = self.pipeline.run_frame(frame, conf_thresh=0.5, target_class=2, decay=0.95)
                heatmap = result["heatmap"]
                summary = result["summary"]

                now = time.monotonic()
                triggered, unsafe_duration = self.tracker.update(summary["status"], now)

                if triggered:
                    print("=== NEW VIOLATION EVENT ===")
                    self.hardware.trigger_alarm()
                    _, image_url = save_violation_image(frame)
                    send_event(summary, image_url)

                # 疊熱度圖、顯示狀態
                heatmap_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                display = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

                status_text = (
                    f"Status: {summary['status']} | unsafe_duration: {unsafe_duration:.1f}s | "
                    f"no_helmet: {summary['num_no_helmet']}"
                )
                color = (0, 255, 0) if summary["status"] == "safe" else (0, 0, 255)
                cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                cv2.imshow("AIoT CUDA Helmet", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hardware.cleanup()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDA Helmet demo (無自走車)")
    parser.add_argument("--source", default=0, type=str, help="cv2.VideoCapture source (index or video path, default camera 0)")
    parser.add_argument("--model", default=None, help="YOLO model path (default config.MODEL_PATH)")
    parser.add_argument("--net-size", type=int, default=640, help="network input size (square)")
    parser.add_argument("--max-boxes", type=int, default=100, help="max detections to keep on GPU")
    parser.add_argument("--unsafe-threshold", type=float, default=3.0, help="連續 unsafe 秒數達到門檻才算違規事件")
    parser.add_argument("--diagnose", action="store_true", help="僅檢查 cuda_lib 載入後退出")
    parser.add_argument("--verbose", action="store_true", help="print search paths when loading cuda_lib")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.diagnose:
        ok = diagnose_cuda_lib()
        return 0 if ok else 1
    if not torch.cuda.is_available():
        print("[Error] 未偵測到 CUDA 裝置，無法執行 CUDA demo。")
        print("        Raspberry Pi 5 請改用 `python helmet_cam.py` 或 `python car_main.py --cpu`。")
        return 1

    src_arg = int(args.source) if str(args.source).isdigit() else args.source
    model_path = args.model or config.MODEL_PATH

    demo = CudaHelmetCameraDemo(
        source=src_arg,
        model_path=model_path,
        net_size=args.net_size,
        max_boxes=args.max_boxes,
        unsafe_threshold=args.unsafe_threshold,
        verbose=args.verbose,
    )
    demo.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
