#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_controller.py

抽象化小車馬達控制，用來取代直接在程式裡操作 RPi.GPIO。

提供:
- forward(), backward(), turn_left(), turn_right(), stop()
- forward_continuous(), backward_continuous(), turn_left_continuous(), turn_right_continuous()

在 Raspberry Pi 以外的環境（例如 Mac）匯入時，會自動使用 Mock GPIO，不會噴錯。
"""

import time

try:
    import RPi.GPIO as GPIO
    _ON_PI = True
except ImportError:
    # 在 Mac / 非 Pi 環境時用假的 GPIO，方便你先開發 / 測試
    _ON_PI = False

    class _MockGPIO:
        BOARD = "BOARD"
        BCM = "BCM"
        OUT = "OUT"
        LOW = False
        HIGH = True

        def setmode(self, *args, **kwargs):
            print("[MockGPIO] setmode", args, kwargs)

        def setup(self, *args, **kwargs):
            print("[MockGPIO] setup", args, kwargs)

        def output(self, *args, **kwargs):
            print("[MockGPIO] output", args, kwargs)

        def cleanup(self, *args, **kwargs):
            print("[MockGPIO] cleanup", args, kwargs)

    GPIO = _MockGPIO()


class MotorController:
    """封裝左右兩側馬達的前進/後退控制。

    預設腳位對應 yachentw/yzucseiot 的 move_car.py：
    - Motor_R1_Pin = 16
    - Motor_R2_Pin = 18
    - Motor_L1_Pin = 11
    - Motor_L2_Pin = 13
    使用 GPIO.BOARD 編號。
    """

    def __init__(
        self,
        pin_r1: int = 23,
        pin_r2: int = 24,
        pin_l1: int = 17,
        pin_l2: int = 27,
        board_mode=GPIO.BOARD,
        default_move_time: float = 0.5,
        auto_cleanup: bool = False,
    ) -> None:
        """
        Args:
            pin_r1, pin_r2, pin_l1, pin_l2:
                馬達控制腳位，預設與原範例相同（BOARD 模式下）。
            board_mode:
                GPIO.BOARD 或 GPIO.BCM，預設用 BOARD。
            default_move_time:
                forward()/backward()/turn_xxx() 沒有指定 duration 時，
                持續動作的秒數。
            auto_cleanup:
                如果為 True，在物件被 GC 時嘗試呼叫 GPIO.cleanup()。
        """
        self.pin_r1 = pin_r1
        self.pin_r2 = pin_r2
        self.pin_l1 = pin_l1
        self.pin_l2 = pin_l2
        self.board_mode = board_mode
        self.default_move_time = default_move_time
        self.auto_cleanup = auto_cleanup

        self._initialized = False
        self._setup_pins()

    # ---------- 初始化 / 清理 ----------
    def _setup_pins(self) -> None:
        GPIO.setmode(self.board_mode)
        GPIO.setup(self.pin_r1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.pin_r2, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.pin_l1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.pin_l2, GPIO.OUT, initial=GPIO.LOW)
        self._initialized = True
        if not _ON_PI:
            print("[MotorController] Initialized in mock mode (non-Pi environment).")

    def cleanup(self) -> None:
        """釋放 GPIO 資源，在程式結束前呼叫。"""
        if not self._initialized:
            return
        self.stop()
        GPIO.cleanup()
        self._initialized = False

    def __del__(self):
        if self.auto_cleanup and self._initialized:
            try:
                self.cleanup()
            except Exception:
                pass

    # ---------- 低階輸出 ----------
    def _drive(
        self,
        r1: bool,
        r2: bool,
        l1: bool,
        l2: bool,
        duration: float | None = None,
        auto_stop: bool = True,
    ) -> None:
        GPIO.output(self.pin_r1, GPIO.HIGH if r1 else GPIO.LOW)
        GPIO.output(self.pin_r2, GPIO.HIGH if r2 else GPIO.LOW)
        GPIO.output(self.pin_l1, GPIO.HIGH if l1 else GPIO.LOW)
        GPIO.output(self.pin_l2, GPIO.HIGH if l2 else GPIO.LOW)

        if duration is not None and duration > 0:
            time.sleep(duration)
            if auto_stop:
                self.stop()

    # ---------- 高階 API（一次動一下） ----------
    def stop(self) -> None:
        """立即停止車子。"""
        GPIO.output(self.pin_r1, GPIO.LOW)
        GPIO.output(self.pin_r2, GPIO.LOW)
        GPIO.output(self.pin_l1, GPIO.LOW)
        GPIO.output(self.pin_l2, GPIO.LOW)

    def forward(self, duration: float | None = None) -> None:
        """前進指定秒數，預設 default_move_time。"""
        if duration is None:
            duration = self.default_move_time
        self._drive(True, False, True, False, duration=duration, auto_stop=True)

    def backward(self, duration: float | None = None) -> None:
        """後退指定秒數，預設 default_move_time。"""
        if duration is None:
            duration = self.default_move_time
        self._drive(False, True, False, True, duration=duration, auto_stop=True)

    def turn_right(self, duration: float | None = None) -> None:
        """原地向右轉。"""
        if duration is None:
            duration = self.default_move_time
        self._drive(False, False, True, False, duration=duration, auto_stop=True)

    def turn_left(self, duration: float | None = None) -> None:
        """原地向左轉。"""
        if duration is None:
            duration = self.default_move_time
        self._drive(True, False, False, False, duration=duration, auto_stop=True)

    # ---------- 連續模式（WASD 按住/切換用） ----------
    def forward_continuous(self) -> None:
        """持續前進（需手動 stop()）。"""
        self._drive(True, False, True, False, duration=None, auto_stop=False)

    def backward_continuous(self) -> None:
        """持續後退（需手動 stop()）。"""
        self._drive(False, True, False, True, duration=None, auto_stop=False)

    def turn_right_continuous(self) -> None:
        """持續右轉（需手動 stop()）。"""
        self._drive(False, False, True, False, duration=None, auto_stop=False)

    def turn_left_continuous(self) -> None:
        """持續左轉（需手動 stop()）。"""
        self._drive(True, False, False, False, duration=None, auto_stop=False)
