#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import threading
import readchar  # pip install readchar

from .hardware import HardwareController
from .ppe_detector import PPEDetector
from .motor_controller import MotorController

# 連續多少秒都判定為 unsafe 才算「違規事件」
UNSAFE_THRESHOLD_SEC = 10.0  # 可以改成 3.0 等看需求


def main():
    hardware = HardwareController()
    detector = PPEDetector()
    motors = MotorController(default_move_time=0.3)

    cap = cv2.VideoCapture(0)  # USB 攝影機 /dev/video0
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # 方案 B 相關狀態（事件層級）
    prev_event_state = "safe"
    event_state = "safe"
    unsafe_start_time = None
    unsafe_duration = 0.0

    running = True  # 控制整個程式是否繼續跑

    # ------------ 鍵盤控制 thread（WASD） ------------
    def keyboard_loop():
        nonlocal running
        print("WASD 控車，空白鍵停車，q 離開。")
        while running:
            ch = readchar.readkey()
            if ch.lower() == "w":
                print("[KEY] w → forward")
                motors.forward_continuous()
            elif ch.lower() == "s":
                print("[KEY] s → backward")
                motors.backward_continuous()
            elif ch.lower() == "a":
                print("[KEY] a → turn left")
                motors.turn_left_continuous()
            elif ch.lower() == "d":
                print("[KEY] d → turn right")
                motors.turn_right_continuous()
            elif ch == " ":
                print("[KEY] space → stop")
                motors.stop()
            elif ch.lower() == "q":
                print("[KEY] q → quit")
                running = False
                motors.stop()
                break

    kb_thread = threading.Thread(target=keyboard_loop, daemon=True)
    kb_thread.start()

    # ------------ 主迴圈：攝影機 + 偵測 + 事件判斷 ------------
    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # 1. YOLO 偵測當下這一幀
            result = detector.analyze_frame(frame)
            # result["status"] 是這一幀的瞬間 safe/unsafe
            instant_status = result["status"]
            print("instant_status:", instant_status, "| result:", result)

            now = time.monotonic()

            # 2. 用方案 B 計算「連續 unsafe 的時間」
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

            # 4. 當事件狀態第一次從 safe → unsafe，就記一次違規事件
            if prev_event_state == "safe" and event_state == "unsafe":
                print("=== NEW VIOLATION EVENT ===")
                # (a) 觸發警報（紅燈閃爍 + 蜂鳴器）
                hardware.trigger_alarm()
                # 解除警報由按鈕控制（hardware.py 裡的 button.when_pressed）

                # (b) 存截圖 + 上報到後端 (Flask /api/events)
                _, image_url = detector.save_violation_image(frame)
                detector.send_event(result, image_url)

            # 5. 避免燒 CPU，也讓時間累積比較穩定
            time.sleep(0.1)  # 大約每秒 10 幀

    finally:
        print("Shutting down...")
        running = False
        motors.stop()
        motors.cleanup()
        hardware.cleanup()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
