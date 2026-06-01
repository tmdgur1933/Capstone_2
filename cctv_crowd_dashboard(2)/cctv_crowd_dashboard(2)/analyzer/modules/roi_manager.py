import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass
class ROIConfig:
    camera_id: str
    roi_name: str
    points: List[Point]


class ROIManager:
    """
    Polygon ROI 관리 모듈.

    points가 비어 있거나 3개 미만이면 전체 화면을 ROI로 본다.
    points가 3개 이상이면 polygon ROI로 판단한다.
    """

    def __init__(self, config: ROIConfig):
        self.config = config
        self.points = config.points

        if len(self.points) >= 3:
            self.polygon = np.array(self.points, dtype=np.int32)
        else:
            self.polygon = None

    @classmethod
    def from_json(cls, path: str):
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"ROI config not found: {path_obj}")

        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_points = data.get("points", [])
        points: List[Point] = []

        for p in raw_points:
            if len(p) != 2:
                continue
            points.append((int(p[0]), int(p[1])))

        config = ROIConfig(
            camera_id=data.get("camera_id", "cam_001"),
            roi_name=data.get("roi_name", "default_roi"),
            points=points,
        )

        return cls(config)

    @property
    def is_full_frame(self) -> bool:
        return self.polygon is None

    def contains_point(self, point: Sequence[int]) -> bool:
        """
        point가 ROI 내부에 있는지 판단.
        polygon이 없으면 전체 화면이므로 True.
        """
        if self.is_full_frame:
            return True

        x, y = int(point[0]), int(point[1])
        result = cv2.pointPolygonTest(self.polygon, (x, y), False)

        return result >= 0

    def filter_detections(self, detections, point_attr: str):
        """
        detections 중 ROI 안에 들어온 것만 반환.
        
        """
        filtered = []

        for det in detections:
            point = getattr(det, point_attr)
            if self.contains_point(point):
                filtered.append(det)

        return filtered

    def draw(self, frame):
        """
        영상 위에 ROI를 그린다.
        """
        if self.is_full_frame:
            cv2.putText(
                frame,
                "ROI: FULL FRAME",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return frame

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [self.polygon],
            color=(0, 255, 255),
        )

        alpha = 0.08
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.polylines(
            frame,
            [self.polygon],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return frame

    def to_dict(self):
        return {
            "camera_id": self.config.camera_id,
            "roi_name": self.config.roi_name,
            "points": [[x, y] for x, y in self.points],
            "is_full_frame": self.is_full_frame,
        }
