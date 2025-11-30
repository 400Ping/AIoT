# detector/config.py
SERVER_URL = "http://127.0.0.1:5000"  # 如果 Flask 跑在 Pi 上就用 localhost
CAMERA_ID = "car-01"

MODEL_PATH = "/home/pi/AIoT/models/best.pt"  # 你的 YOLO 安全帽模型
IMG_SAVE_DIR = "/home/pi/AIoT/server/static/violations"

# GPIO 腳位 (BCM 模式)
RED_LED_PIN = 5    # 實體 pin 29
GREEN_LED_PIN = 6  # 實體 pin 31
BUZZER_PIN = 22    # 實體 pin 15（原本就可留）
BUTTON_PIN = 26    # 實體 pin 37
