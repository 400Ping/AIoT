# detector/hardware.py
from gpiozero import LED, Buzzer, Button
from signal import pause
import threading
from . import config

class HardwareController:
    def __init__(self):
        self.red = LED(config.RED_LED_PIN)
        self.green = LED(config.GREEN_LED_PIN)
        self.buzzer = Buzzer(config.BUZZER_PIN)
        self.button = Button(config.BUTTON_PIN, pull_up=True)
        self.alarm_active = False
        self._lock = threading.Lock()

        # 初始狀態：綠燈亮，紅燈/蜂鳴器關
        self.green.on()
        self.red.off()
        self.buzzer.off()

        # 按鈕按下時解除警報
        self.button.when_pressed = self.clear_alarm

    def trigger_alarm(self):
        """觸發警報：紅燈閃爍 + 蜂鳴器鳴叫，綠燈關"""
        with self._lock:
            if self.alarm_active:
                return
            self.alarm_active = True

            self.green.off()
            # 紅燈與蜂鳴器閃爍
            self.red.blink(on_time=0.2, off_time=0.2)
            self.buzzer.beep(on_time=0.2, off_time=0.2)

    def clear_alarm(self):
        """解除警報：紅燈/蜂鳴器關，綠燈亮"""
        with self._lock:
            self.alarm_active = False
            self.red.off()
            self.buzzer.off()
            self.green.on()

    def cleanup(self):
        self.red.close()
        self.green.close()
        self.buzzer.close()
        self.button.close()
