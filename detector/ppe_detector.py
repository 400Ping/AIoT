# detector/ppe_detector.py
import cv2
from datetime import datetime
import os
import requests

try:
    from . import config
    from .yolo_utils import load_yolo_model, yolo_class_ids, yolo_predict
except ImportError:  # allow running as a script from detector/
    import config  # type: ignore
    from yolo_utils import load_yolo_model, yolo_class_ids, yolo_predict  # type: ignore

# Hard Hat Dataset class mapping:
# 0: helmet, 1: person, 2: head (沒戴安全帽的頭)
CLASS_HELMET = 0
CLASS_PERSON = 1
CLASS_HEAD = 2

class PPEDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model, self.backend = load_yolo_model(model_path)

    def analyze_frame(self, frame):
        """
        回傳高階資訊：
        - num_people: 畫面中人數
        - num_no_helmet: 畫面中沒戴安全帽的人數 (head)
        - status: "safe" / "unsafe"
        """
        num_people = 0
        num_no_helmet = 0

        results = yolo_predict(self.model, self.backend, frame, imgsz=640, conf=0.5)
        for cls_id in yolo_class_ids(results, self.backend):
            if cls_id == CLASS_PERSON:
                num_people += 1
            elif cls_id == CLASS_HEAD:
                num_no_helmet += 1

        status = "unsafe" if num_no_helmet > 0 else "safe"
        return {
            "num_people": num_people,
            "num_no_helmet": num_no_helmet,
            "status": status,
        }

    def save_violation_image(self, frame):
        os.makedirs(config.IMG_SAVE_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{ts}.jpg"
        path = os.path.join(config.IMG_SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        # 給 Flask 用的 URL 路徑
        image_url = f"/static/violations/{filename}"
        return path, image_url

    def send_event(self, result, image_url):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": config.CAMERA_ID,
            "status": result["status"],
            "num_people": result["num_people"],
            "num_no_helmet": result["num_no_helmet"],
            "image_url": image_url,
        }
        try:
            requests.post(
                f"{config.SERVER_URL}/api/events",
                json=payload,
                timeout=1.0,
            )
        except Exception as e:
            print(f"[WARN] Failed to send event to server: {e}")
