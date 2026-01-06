#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from . import config
    from .ppe_detector import PPEDetector
    from .cuda_runtime import CudaHelmetPipeline, diagnose_cuda_lib
    from .event_utils import save_violation_image, send_event, ViolationTracker
except ImportError:  # allow running as a script from detector/
    import config  # type: ignore
    from ppe_detector import PPEDetector  # type: ignore
    from cuda_runtime import CudaHelmetPipeline, diagnose_cuda_lib  # type: ignore
    from event_utils import save_violation_image, send_event, ViolationTracker  # type: ignore

try:
    from .hardware import HardwareController

    HARDWARE_AVAILABLE = True
    HARDWARE_ERR = None
except Exception as exc:  # pragma: no cover - runtime guard
    try:
        from hardware import HardwareController  # type: ignore

        HARDWARE_AVAILABLE = True
        HARDWARE_ERR = None
    except Exception as exc2:  # pragma: no cover - runtime guard
        HARDWARE_AVAILABLE = False
        HARDWARE_ERR = exc2


class DummyHardware:
    """在非樹莓派環境下的簡易 stub。"""

    def trigger_alarm(self):
        print("[Hardware] trigger_alarm (stub)")

    def clear_alarm(self):
        print("[Hardware] clear_alarm (stub)")

    def cleanup(self):
        pass


def resolve_model_path(model_path):
    model_path = model_path or config.MODEL_PATH
    if not model_path:
        raise ValueError("MODEL_PATH is empty")
    model_path = str(model_path)
    if Path(model_path).exists():
        return model_path
    repo_model = Path(__file__).resolve().parent.parent / "models" / "best.pt"
    if repo_model.exists():
        print(f"[Warn] 模型檔不存在於 {model_path}，改用 {repo_model}")
        return str(repo_model)
    raise FileNotFoundError(f"找不到模型檔：{model_path}，且預設 {repo_model} 也不存在")


class HelmetCameraDemo:
    def __init__(self, mode, source, model_path, net_size, max_boxes, unsafe_threshold, verbose=False):
        self.mode = mode
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 source={source}")

        self.model_path = resolve_model_path(model_path)
        self.tracker = ViolationTracker(unsafe_threshold)

        if HARDWARE_AVAILABLE:
            self.hardware = HardwareController()
        else:
            print(f"[Warn] 無法載入硬體控制 (LED/蜂鳴器)：{HARDWARE_ERR}")
            print("       將使用 stub，僅列印訊息。")
            self.hardware = DummyHardware()

        self.pipeline = None
        self.detector = None

        if self.mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("未偵測到 CUDA 裝置，無法使用 GPU 模式。")
            self.pipeline = CudaHelmetPipeline(
                model_path=self.model_path,
                net_size=net_size,
                max_boxes=max_boxes,
                verbose=verbose,
            )
            src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dummy = np.zeros((src_h, src_w, 3), dtype=np.uint8)
            self.pipeline.ensure_buffers(dummy)
        else:
            self.detector = PPEDetector(model_path=self.model_path)

    def run(self):
        title = "AIoT Helmet (CUDA)" if self.mode == "cuda" else "AIoT Helmet (CPU)"
        print(f"開始 Helmet Demo，模式={self.mode}，按 q 離開。")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("讀取影像失敗，結束。")
                    break

                if self.mode == "cuda":
                    result = self.pipeline.run_frame(frame, conf_thresh=0.5, target_class=2, decay=0.95)
                    summary = result["summary"]
                    heatmap = result["heatmap"]
                    heatmap_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                    display = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                else:
                    summary = self.detector.analyze_frame(frame)
                    display = frame

                now = time.monotonic()
                triggered, unsafe_duration = self.tracker.update(summary["status"], now)

                if triggered:
                    print("=== NEW VIOLATION EVENT ===")
                    self.hardware.trigger_alarm()
                    _, image_url = save_violation_image(frame)
                    send_event(summary, image_url)

                status_text = (
                    f"Status: {summary['status']} | unsafe_duration: {unsafe_duration:.1f}s | "
                    f"no_helmet: {summary['num_no_helmet']}"
                )
                color = (0, 255, 0) if summary["status"] == "safe" else (0, 0, 255)
                cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                cv2.imshow(title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hardware.cleanup()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Helmet camera demo (CPU/CUDA)")
    parser.add_argument("--mode", choices=["cpu", "cuda"], default="cpu", help="偵測模式 (預設 cpu)")
    parser.add_argument("--cuda", action="store_true", help="等同於 --mode cuda")
    parser.add_argument("--source", default=0, type=str, help="cv2.VideoCapture source (index or video path, default camera 0)")
    parser.add_argument("--model", default=None, help="YOLO model path (default config.MODEL_PATH)")
    parser.add_argument("--net-size", type=int, default=640, help="network input size (square, CUDA only)")
    parser.add_argument("--max-boxes", type=int, default=100, help="max detections to keep on GPU (CUDA only)")
    parser.add_argument("--unsafe-threshold", type=float, default=10.0, help="連續 unsafe 秒數達到門檻才算違規事件")
    parser.add_argument("--diagnose", action="store_true", help="僅檢查 cuda_lib 載入後退出 (CUDA)")
    parser.add_argument("--verbose", action="store_true", help="print search paths when loading cuda_lib")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mode = "cuda" if args.cuda else args.mode
    if args.diagnose:
        ok = diagnose_cuda_lib()
        return 0 if ok else 1

    src_arg = int(args.source) if str(args.source).isdigit() else args.source

    demo = HelmetCameraDemo(
        mode=mode,
        source=src_arg,
        model_path=args.model,
        net_size=args.net_size,
        max_boxes=args.max_boxes,
        unsafe_threshold=args.unsafe_threshold,
        verbose=args.verbose,
    )
    demo.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
