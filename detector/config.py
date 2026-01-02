# detector/config.py
from pathlib import Path

SERVER_URL = "https://166608653535.ngrok-free.app"  # 如果 Flask 跑在 Pi 上就用 localhost
CAMERA_ID = "cam-01"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = "/Users/jiec/AIoT/models/best.pt"  # 你的 YOLO 安全帽模型
IMG_SAVE_DIR = str(BASE_DIR / "server" / "static" / "violations")

# GPIO 腳位 (BCM 模式)
RED_LED_PIN = 5    # 實體 pin 29
GREEN_LED_PIN = 6  # 實體 pin 31
BUZZER_PIN = 22    # 實體 pin 15（原本就可留）
BUTTON_PIN = 26    # 實體 pin 37
