# detector/ppe_detector.py
try:
    from . import config
    from .yolo_utils import load_yolo_model, yolo_class_ids, yolo_predict
    from .event_utils import save_violation_image as save_violation_image_impl
    from .event_utils import send_event as send_event_impl
except ImportError:  # allow running as a script from detector/
    import config  # type: ignore
    from yolo_utils import load_yolo_model, yolo_class_ids, yolo_predict  # type: ignore
    from event_utils import save_violation_image as save_violation_image_impl  # type: ignore
    from event_utils import send_event as send_event_impl  # type: ignore

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
        return save_violation_image_impl(frame)

    def send_event(self, result, image_url):
        return send_event_impl(result, image_url)
