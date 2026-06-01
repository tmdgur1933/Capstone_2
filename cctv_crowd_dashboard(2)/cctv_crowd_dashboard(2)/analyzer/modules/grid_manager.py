from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2


Point = Tuple[int, int]


@dataclass
class GridCell:
    cell_id: str
    row: int
    col: int
    bbox: Tuple[int, int, int, int]
    count: int = 0
    status: str = "NORMAL"

    def contains_point(self, point: Sequence[int]) -> bool:
        x, y = int(point[0]), int(point[1])
        x1, y1, x2, y2 = self.bbox
        return x1 <= x < x2 and y1 <= y < y2

    def to_dict(self):
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "col": self.col,
            "bbox": list(self.bbox),
            "count": self.count,
            "status": self.status,
        }


class GridManager:
    """
    ROI 내부를 grid cell로 나누고 cell별 사람 수를 계산한다.
    """

    def __init__(self, cell_width: int = 80, cell_height: int = 80):
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.cells: List[GridCell] = []

    def build_cells(self, frame_width: int, frame_height: int, roi_manager):
        """
        ROI 내부에 포함되는 grid cell만 active cell로 생성한다.
        ROI가 full frame이면 전체 화면을 grid로 나눈다.
        """

        self.cells = []

        if roi_manager.is_full_frame:
            min_x, min_y = 0, 0
            max_x, max_y = frame_width, frame_height
        else:
            xs = [p[0] for p in roi_manager.points]
            ys = [p[1] for p in roi_manager.points]

            min_x = max(0, min(xs))
            min_y = max(0, min(ys))
            max_x = min(frame_width, max(xs))
            max_y = min(frame_height, max(ys))

        row = 0
        y = min_y

        while y < max_y:
            col = 0
            x = min_x

            while x < max_x:
                x1 = x
                y1 = y
                x2 = min(x + self.cell_width, frame_width)
                y2 = min(y + self.cell_height, frame_height)

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # polygon ROI일 경우 cell 중심점이 ROI 안에 있는 cell만 사용
                if roi_manager.contains_point(center):
                    cell_id = f"r{row}_c{col}"
                    self.cells.append(
                        GridCell(
                            cell_id=cell_id,
                            row=row,
                            col=col,
                            bbox=(x1, y1, x2, y2),
                        )
                    )

                x += self.cell_width
                col += 1

            y += self.cell_height
            row += 1

        print(f"[INFO] Active grid cells: {len(self.cells)}")

    def reset_counts(self):
        for cell in self.cells:
            cell.count = 0
            cell.status = "NORMAL"

    def count_points(self, points: List[Point]):
        """
        사람 대표 좌표들이 어느 grid cell에 들어가는지 계산한다.
        """

        self.reset_counts()

        for point in points:
            for cell in self.cells:
                if cell.contains_point(point):
                    cell.count += 1
                    break

    def get_count_map(self) -> Dict[str, int]:
        return {cell.cell_id: cell.count for cell in self.cells}

    def to_list(self):
        return [cell.to_dict() for cell in self.cells]

    def draw(self, frame):
        """
        grid cell과 count/status를 화면에 그림.
        """

        for cell in self.cells:
            x1, y1, x2, y2 = cell.bbox

            if cell.status == "DANGER":
                color = (0, 0, 255)
                thickness = 3
            elif cell.status == "WARNING":
                color = (0, 165, 255)
                thickness = 3
            else:
                color = (180, 180, 180)
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if cell.count > 0 or cell.status != "NORMAL":
                label = f"{cell.count}"
                if cell.status != "NORMAL":
                    label = f"{cell.status}:{cell.count}"

                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return frame