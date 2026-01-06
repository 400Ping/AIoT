import os
from datetime import datetime

import requests

try:
    from . import config
except ImportError:  # allow running as a script from detector/
    import config  # type: ignore


def save_violation_image(frame, img_save_dir=None):
    save_dir = img_save_dir or config.IMG_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_{ts}.jpg"
    path = os.path.join(save_dir, filename)
    cv2 = _lazy_import_cv2()
    cv2.imwrite(path, frame)
    image_url = f"/static/violations/{filename}"
    return path, image_url


def send_event(summary, image_url, server_url=None, camera_id=None, timeout=1.0):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "camera_id": camera_id or config.CAMERA_ID,
        "status": summary["status"],
        "num_people": summary["num_people"],
        "num_no_helmet": summary["num_no_helmet"],
        "image_url": image_url,
    }
    try:
        requests.post(
            f"{server_url or config.SERVER_URL}/api/events",
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"[WARN] Failed to send event to server: {exc}")
    return payload


class ViolationTracker:
    def __init__(self, threshold_sec: float):
        self.threshold_sec = threshold_sec
        self.prev_event_state = "safe"
        self.event_state = "safe"
        self.unsafe_start_time = None
        self.unsafe_duration = 0.0

    def update(self, status: str, now: float):
        if status == "unsafe":
            if self.unsafe_start_time is None:
                self.unsafe_start_time = now
                self.unsafe_duration = 0.0
            else:
                self.unsafe_duration = now - self.unsafe_start_time
        else:
            self.unsafe_start_time = None
            self.unsafe_duration = 0.0

        self.prev_event_state = self.event_state
        self.event_state = "unsafe" if self.unsafe_duration >= self.threshold_sec else "safe"

        triggered = self.prev_event_state == "safe" and self.event_state == "unsafe"
        return triggered, self.unsafe_duration


def _lazy_import_cv2():
    import cv2  # local import to keep startup light in non-vision uses

    return cv2
