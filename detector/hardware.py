# detector/hardware.py
import threading
import time

from gpio_compat import GPIO

import config


class HardwareController:
    """用 RPi.GPIO 控制紅綠燈、蜂鳴器、按鈕。

    功能和之前用 gpiozero 的版本一致：
    - trigger_alarm(): 綠燈關、紅燈 + 蜂鳴器閃爍 / 嗶嗶叫
    - clear_alarm():   紅燈 / 蜂鳴器關、綠燈亮
    - 按鈕按下 => clear_alarm()
    """

    def __init__(self):
        # 使用 BCM 腳位編號，對應 config.RED_LED_PIN 等
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.red_pin = config.RED_LED_PIN
        self.green_pin = config.GREEN_LED_PIN
        self.buzzer_pin = config.BUZZER_PIN
        self.button_pin = config.BUTTON_PIN

        # 設定輸出腳位
        GPIO.setup(self.red_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.green_pin, GPIO.OUT, initial=GPIO.HIGH)  # 一開始綠燈亮
        GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)

        # 按鈕用內建上拉電阻，預設為 HIGH，按下變 LOW
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.alarm_active = False
        self._lock = threading.Lock()
        self._alarm_thread = None

        # 設定按鈕中斷，FALLING 觸發（HIGH -> LOW），加一點 debounce
        GPIO.add_event_detect(
            self.button_pin,
            GPIO.FALLING,
            callback=self._button_callback,
            bouncetime=200,
        )

    # ------------------------------------------------------------
    # 對外 API
    # ------------------------------------------------------------
    def trigger_alarm(self):
        """觸發警報：紅燈 / 蜂鳴器閃爍，綠燈關。"""
        with self._lock:
            if self.alarm_active:
                # 已經在警報狀態，就不要再開新的 thread
                return
            self.alarm_active = True

            # 綠燈關
            GPIO.output(self.green_pin, GPIO.LOW)

            # 開一個背景 thread 負責 blink
            self._alarm_thread = threading.Thread(
                target=self._alarm_worker, daemon=True
            )
            self._alarm_thread.start()

    def clear_alarm(self):
        """解除警報：紅燈 / 蜂鳴器關，綠燈亮。"""
        with self._lock:
            self.alarm_active = False

            # 把輸出全部歸零
            GPIO.output(self.red_pin, GPIO.LOW)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.output(self.green_pin, GPIO.HIGH)

    def cleanup(self):
        """程式結束前呼叫，釋放 GPIO 資源。"""
        try:
            GPIO.remove_event_detect(self.button_pin)
        except RuntimeError:
            # 如果沒設過 event detect，這裡會丟錯，忽略即可
            pass
        GPIO.cleanup()

    # ------------------------------------------------------------
    # 內部用函式
    # ------------------------------------------------------------
    def _alarm_worker(self):
        """在背景 thread 裡閃爍紅燈 / 蜂鳴器。"""
        try:
            while True:
                with self._lock:
                    if not self.alarm_active:
                        break

                # 開
                GPIO.output(self.red_pin, GPIO.HIGH)
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                time.sleep(0.2)

                # 關
                GPIO.output(self.red_pin, GPIO.LOW)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                time.sleep(0.2)
        finally:
            # 確保離開時關閉輸出（避免 thread 中途被打斷）
            GPIO.output(self.red_pin, GPIO.LOW)
            GPIO.output(self.buzzer_pin, GPIO.LOW)

    def _button_callback(self, channel):
        """按鈕中斷 callback：解除警報。"""
        # 再讀一次腳位，確保真的是按下（簡單防抖）
        if GPIO.input(self.button_pin) == GPIO.LOW:
            print("[Hardware] Button pressed -> clear alarm")
            self.clear_alarm()
