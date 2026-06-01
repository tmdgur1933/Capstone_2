from dataclasses import dataclass
from typing import Dict


@dataclass
class AlertSummary:
    overall_status: str
    warning_cell_count: int
    danger_cell_count: int
    max_cell_count: int



class AlertEngine:
    """
    Grid cell별 count를 보고 WARNING / DANGER를 판단한다.

    persist_frames:
    - 임계값을 한 프레임만 넘었다고 바로 경고하지 않고,
      연속 N프레임 이상 지속될 때만 경고 상태로 바꾼다.
    """

    def __init__(
        self,
        warning_threshold: int = 5,
        danger_threshold: int = 8,
        persist_frames: int = 5,
    ):
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.persist_frames = persist_frames

        self.streaks: Dict[str, int] = {}
        self.last_candidate_status: Dict[str, str] = {}

    def _candidate_status(self, count: int) -> str:
        if count >= self.danger_threshold:
            return "DANGER"
        if count >= self.warning_threshold:
            return "WARNING"
        return "NORMAL"

    def update_cells(self, cells) -> AlertSummary:
        warning_cell_count = 0
        danger_cell_count = 0
        max_cell_count = 0

        for cell in cells:
            count = cell.count
            max_cell_count = max(max_cell_count, count)

            candidate = self._candidate_status(count)

            if candidate == "NORMAL":
                self.streaks[cell.cell_id] = 0
                self.last_candidate_status[cell.cell_id] = "NORMAL"
                cell.status = "NORMAL"
                continue

            previous_candidate = self.last_candidate_status.get(cell.cell_id)

            if previous_candidate == candidate:
                self.streaks[cell.cell_id] = self.streaks.get(cell.cell_id, 0) + 1
            else:
                self.streaks[cell.cell_id] = 1
                self.last_candidate_status[cell.cell_id] = candidate

            if self.streaks[cell.cell_id] >= self.persist_frames:
                cell.status = candidate
            else:
                cell.status = "NORMAL"

            if cell.status == "DANGER":
                danger_cell_count += 1
            elif cell.status == "WARNING":
                warning_cell_count += 1

        if danger_cell_count > 0:
            overall_status = "DANGER"
        elif warning_cell_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "NORMAL"

        return AlertSummary(
            overall_status=overall_status,
            warning_cell_count=warning_cell_count,
            danger_cell_count=danger_cell_count,
            max_cell_count=max_cell_count,
        )