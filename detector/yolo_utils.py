# detector/yolo_utils.py
import os
from pathlib import Path

import torch


def _resolve_yolov5_repo():
    env_path = os.getenv("YOLOV5_REPO")
    if env_path:
        repo = Path(env_path).expanduser().resolve()
        if repo.exists():
            return str(repo)

    local_repo = Path(__file__).resolve().parent.parent / "yolov5"
    if local_repo.exists():
        return str(local_repo)
    return None


def load_yolo_model(model_path, device=None):
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        backend = "ultralytics"
    except Exception as exc:
        repo = _resolve_yolov5_repo()
        if not repo:
            raise RuntimeError(
                "Ultralytics 無法載入，且找不到本機 YOLOv5 repo。"
                "請設定 YOLOV5_REPO 指向 yolov5 目錄或改用 Python >= 3.7。"
            ) from exc
        model = torch.hub.load(repo, "custom", path=model_path, source="local")
        backend = "yolov5"

    if device is not None:
        try:
            model.to(device)
        except Exception:
            pass
    return model, backend


def yolo_predict(model, backend, frame, imgsz=640, conf=0.5, verbose=False):
    if backend == "ultralytics":
        return model(frame, imgsz=imgsz, conf=conf, verbose=verbose)

    if hasattr(model, "conf"):
        model.conf = conf
    return model(frame, size=imgsz)


def yolo_boxes_xywhn(results, backend, device=None):
    if backend == "ultralytics":
        if not results or results[0].boxes is None:
            return None
        det = results[0].boxes
        boxes = torch.cat(
            [det.xywhn, det.conf.unsqueeze(1), det.cls.unsqueeze(1)],
            dim=1,
        )
    else:
        xywhn = getattr(results, "xywhn", None)
        if xywhn is None or len(xywhn) == 0:
            return None
        boxes = xywhn[0]

    if boxes is None or boxes.numel() == 0:
        return None
    if device is not None:
        boxes = boxes.to(device)
    return boxes


def yolo_class_ids(results, backend):
    if backend == "ultralytics":
        if not results or results[0].boxes is None:
            return []
        return [int(box.cls[0]) for box in results[0].boxes]

    xywhn = getattr(results, "xywhn", None)
    if xywhn is None or len(xywhn) == 0:
        return []
    cls_col = xywhn[0][:, 5]
    return [int(x) for x in cls_col.tolist()]
