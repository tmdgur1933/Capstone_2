from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: BBox
    center: Point
    bottom_center: Point


@dataclass
class TrackedPerson:
    track_id: int
    source: str
    point: Point
    bbox: BBox
    confidence: float


class HeadDetector:
    def __init__(self, head_model_path: str, conf: float = 0.20, imgsz: int = 960, device: str = "cpu"):
        self.head_model_path = Path(head_model_path)
        if not self.head_model_path.exists():
            raise FileNotFoundError(f"Head model not found: {self.head_model_path}")
        self.model = YOLO(str(self.head_model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(source=frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections
        for box in results[0].boxes:
            xyxy_raw = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy_raw]
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            label = self.model.names.get(cls_id, "head") if hasattr(self.model, "names") else "head"
            detections.append(Detection(str(label), confidence, (x1, y1, x2, y2), (cx, cy), (cx, int(y2))))
        return detections


class ByteTrackPersonTracker:
    def __init__(self, body_model_path: str, conf: float = 0.25, imgsz: int = 960, device: str = "cpu", tracker_config: str = "bytetrack.yaml"):
        self.body_model_path = Path(body_model_path)
        if not self.body_model_path.exists():
            raise FileNotFoundError(f"Body model not found: {self.body_model_path}")
        self.model = YOLO(str(self.body_model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.tracker_config = tracker_config

    def update(self, frame) -> List[TrackedPerson]:
        results = self.model.track(source=frame, persist=True, tracker=self.tracker_config, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        tracked: List[TrackedPerson] = []
        if not results or results[0].boxes is None:
            return tracked
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0].detach().cpu().item())
            xyxy_raw = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy_raw]
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            cx = int((x1 + x2) / 2)
            tracked.append(TrackedPerson(track_id, "bytetrack_body", (cx, int(y2)), (x1, y1, x2, y2), confidence))
        return tracked
