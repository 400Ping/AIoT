#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time

from hardware import HardwareController
from ppe_detector import PPEDetector

# 連續多少秒都判定為 unsafe 才算「違規事件」
UNSAFE_THRESHOLD_SEC = 10.0  # demo 可以改成 3.0 比較快看到效果


def main():
    hardware = HardwareController()
    detector = PPEDetector()

    cap = cv2.VideoCapture(0)  # USB 攝影機 /dev/video0
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # 事件層級狀態
    prev_event_state = "safe"
    event_state = "safe"
    unsafe_start_time = None
    unsafe_duration = 0.0

    print("安全帽偵測 demo 開始，按 q 離開。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # 1. YOLO 偵測當下這一幀
            result = detector.analyze_frame(frame)
            instant_status = result["status"]  # "safe" / "unsafe"

            # 這就是你原本在 car_main 印的那句
            print(f"instant_status: {instant_status} | result: {result}")

            now = time.monotonic()

            # 2. 計算「連續 unsafe 的時間」
            if instant_status == "unsafe":
                if unsafe_start_time is None:
                    unsafe_start_time = now
                    unsafe_duration = 0.0
                else:
                    unsafe_duration = now - unsafe_start_time
            else:
                unsafe_start_time = None
                unsafe_duration = 0.0

            # 3. 根據 unsafe_duration 決定「事件層級」狀態
            prev_event_state = event_state
            if unsafe_duration >= UNSAFE_THRESHOLD_SEC:
                event_state = "unsafe"
            else:
                event_state = "safe"

            # 4. 第一次從 safe → unsafe，記一次違規事件 + 觸發警報 + 上報
            if prev_event_state == "safe" and event_state == "unsafe":
                print("=== NEW VIOLATION EVENT ===")
                # (a) 觸發警報（紅燈閃爍 + 蜂鳴器）
                hardware.trigger_alarm()
                # 解除警報由按鈕控制（hardware.py 裡的 button callback）

                # (b) 存截圖 + 上報到後端 (Flask /api/events)
                _, image_url = detector.save_violation_image(frame)
                detector.send_event(result, image_url)

            # 5. 在畫面上疊文字顯示狀態
            status_text = (
                f"Status: {instant_status} | unsafe_duration: {unsafe_duration:.1f}s"
            )
            color = (0, 255, 0) if instant_status == "safe" else (0, 0, 255)

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("PPE Helmet Demo", frame)

            # 在視窗中按 q 離開
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("Shutting down helmet demo...")
        cap.release()
        cv2.destroyAllWindows()
        hardware.cleanup()


if __name__ == "__main__":
    main()
