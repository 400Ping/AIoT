# detector/config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best.pt"
DEFAULT_IMG_SAVE_DIR = BASE_DIR / "server" / "static" / "violations"

# 可用環境變數覆寫，方便在 Jetson / Pi / 筆電共用同一份專案
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5001")
CAMERA_ID = os.getenv("CAMERA_ID", "cam-01")
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
IMG_SAVE_DIR = os.getenv("IMG_SAVE_DIR", str(DEFAULT_IMG_SAVE_DIR))

# GPIO 腳位 (BCM 模式)
RED_LED_PIN = 5    # 實體 pin 29
GREEN_LED_PIN = 6  # 實體 pin 31
BUZZER_PIN = 22    # 實體 pin 15（原本就可留）
BUTTON_PIN = 26    # 實體 pin 37
