# detector/motor_controller.py
import time
import RPi.GPIO as GPIO


class MotorController:
    """控制小車馬達的類別（L298N + 兩個 DC 馬達）

    - 預設使用 BCM 腳位編號
    - 若 GPIO mode 尚未設定，會設成 BCM
    - 若已經設定成其他 mode，則不再呼叫 setmode，只是印出警告
    """

    def __init__(
        self,
        pin_r1=23,
        pin_r2=24,
        pin_l1=17,
        pin_l2=27,
        board_mode=GPIO.BCM,
        default_move_time=0.3,
    ):
        # 右馬達兩腳
        self.pin_r1 = pin_r1
        self.pin_r2 = pin_r2
        # 左馬達兩腳
        self.pin_l1 = pin_l1
        self.pin_l2 = pin_l2

        self.board_mode = board_mode
        self.default_move_time = default_move_time

        self._setup_pins()

    def _setup_pins(self):
        """設定 GPIO 腳位方向，不去硬改已經存在的 mode。"""
        try:
            current_mode = GPIO.getmode()
        except RuntimeError:
            current_mode = None

        # getmode() 在 RPi.GPIO 中：
        # - 未設定時會是 GPIO.UNKNOWN（-1）
        # - 設成 BCM 時是 GPIO.BCM
        # - 設成 BOARD 時是 GPIO.BOARD 
        if current_mode in (None, GPIO.UNKNOWN):
            # 尚未設定過，這裡統一設為 BCM
            GPIO.setmode(self.board_mode)
        elif current_mode != self.board_mode:
            # 已經設定成別的模式，就不要再 setmode，避免 ValueError 
            print(
                f"[MotorController] GPIO mode already set to {current_mode}, "
                f"expected {self.board_mode}. Using existing mode."
            )

        # 設定輸出腳位
        for pin in (self.pin_r1, self.pin_r2, self.pin_l1, self.pin_l2):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    # ------------------ 基本動作 ------------------
    def stop(self):
        GPIO.output(self.pin_r1, GPIO.LOW)
        GPIO.output(self.pin_r2, GPIO.LOW)
        GPIO.output(self.pin_l1, GPIO.LOW)
        GPIO.output(self.pin_l2, GPIO.LOW)

    def forward(self, t=None):
        """雙馬達同向前進"""
        if t is None:
            t = self.default_move_time
        GPIO.output(self.pin_r1, GPIO.HIGH)
        GPIO.output(self.pin_r2, GPIO.LOW)
        GPIO.output(self.pin_l1, GPIO.HIGH)
        GPIO.output(self.pin_l2, GPIO.LOW)
        time.sleep(t)
        self.stop()

    def backward(self, t=None):
        """雙馬達倒退"""
        if t is None:
            t = self.default_move_time
        GPIO.output(self.pin_r1, GPIO.LOW)
        GPIO.output(self.pin_r2, GPIO.HIGH)
        GPIO.output(self.pin_l1, GPIO.LOW)
        GPIO.output(self.pin_l2, GPIO.HIGH)
        time.sleep(t)
        self.stop()

    def left(self, t=None):
        """原地左轉：右輪前進，左輪後退"""
        if t is None:
            t = self.default_move_time
        GPIO.output(self.pin_r1, GPIO.HIGH)
        GPIO.output(self.pin_r2, GPIO.LOW)
        GPIO.output(self.pin_l1, GPIO.LOW)
        GPIO.output(self.pin_l2, GPIO.HIGH)
        time.sleep(t)
        self.stop()

    def right(self, t=None):
        """原地右轉：右輪後退，左輪前進"""
        if t is None:
            t = self.default_move_time
        GPIO.output(self.pin_r1, GPIO.LOW)
        GPIO.output(self.pin_r2, GPIO.HIGH)
        GPIO.output(self.pin_l1, GPIO.HIGH)
        GPIO.output(self.pin_l2, GPIO.LOW)
        time.sleep(t)
        self.stop()
