import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2

from modules.alert_engine import AlertEngine
from modules.backend_client import BackendClient
from modules.bytetrack_tracker import ByteTrackPersonTracker, HeadDetector
from modules.db_logger import DBLogger
from modules.grid_manager import GridManager
from modules.roi_manager import ROIManager
from setup_roi import run_roi_editor_on_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_roi_path(video_path: Path) -> Path:
    return PROJECT_ROOT / "analyzer" / "configs" / f"{video_path.stem}_roi.json"


def default_db_path() -> Path:
    return PROJECT_ROOT / "analyzer" / "outputs" / "analysis.db"


def default_tracker_config_path() -> str:
    custom_path = PROJECT_ROOT / "analyzer" / "configs" / "bytetrack_custom.yaml"
    if custom_path.exists():
        return str(custom_path)
    return "bytetrack.yaml"


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def load_alert_config(path: Path):
    default_config = {
        "cell_width": 80,
        "cell_height": 80,
        "warning_threshold": 5,
        "danger_threshold": 8,
        "persist_frames": 5,
    }
    if not path.exists():
        print(f"[WARN] Alert config not found. Use default: {default_config}")
        return default_config
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    config = default_config.copy()
    config.update(data)
    return config


class ROIFlowCounter:
    """
    Gate 없이 ROI 경계 통과만으로 IN/OUT을 계산한다.
    이전 프레임에서 ROI 밖, 현재 프레임에서 ROI 안이면 IN.
    이전 프레임에서 ROI 안, 현재 프레임에서 ROI 밖이면 OUT.
    """

    def __init__(self, cooldown_frames: int = 20, recent_window_frames: int = 150, warning_threshold: int = 5, danger_threshold: int = 10):
        self.cooldown_frames = cooldown_frames
        self.recent_window_frames = recent_window_frames
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.total_in = 0
        self.total_out = 0
        self.previous_inside = {}
        self.last_cross_frame = {}
        self.recent_events = deque()

    def update(self, tracked_persons, roi_manager: ROIManager, frame_index: int):
        events = []
        for person in tracked_persons:
            track_id = person.track_id
            curr_inside = roi_manager.contains_point(person.point)
            if track_id not in self.previous_inside:
                self.previous_inside[track_id] = curr_inside
                continue
            prev_inside = self.previous_inside[track_id]
            self.previous_inside[track_id] = curr_inside
            if prev_inside == curr_inside:
                continue
            last_frame = self.last_cross_frame.get(track_id, -999999)
            if frame_index - last_frame < self.cooldown_frames:
                continue
            if not prev_inside and curr_inside:
                count_type = "IN"
                self.total_in += 1
            elif prev_inside and not curr_inside:
                count_type = "OUT"
                self.total_out += 1
            else:
                continue
            self.last_cross_frame[track_id] = frame_index
            event = {"frame_index": frame_index, "track_id": track_id, "count_type": count_type}
            events.append(event)
            self.recent_events.append(event)
        while self.recent_events and frame_index - self.recent_events[0]["frame_index"] > self.recent_window_frames:
            self.recent_events.popleft()
        return events

    def get_summary(self, current_roi_person_count: int):
        recent_in = sum(1 for e in self.recent_events if e["count_type"] == "IN")
        recent_out = sum(1 for e in self.recent_events if e["count_type"] == "OUT")
        recent_diff = recent_in - recent_out
        abs_recent_diff = abs(recent_diff)
        if abs_recent_diff >= self.danger_threshold:
            flow_status = "DANGER"
        elif abs_recent_diff >= self.warning_threshold:
            flow_status = "WARNING"
        else:
            flow_status = "NORMAL"
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "stay_count": current_roi_person_count,
            "net_flow": self.total_in - self.total_out,
            "recent_in": recent_in,
            "recent_out": recent_out,
            "flow_imbalance": recent_diff,
            "flow_status": flow_status,
        }


def draw_track(frame, person, color=(255, 0, 0)):
    x1, y1, x2, y2 = person.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, f"ID:{person.track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_status_panel(frame, frame_index, roi_person_count, flow_summary, alert_summary):
    net_flow = flow_summary["net_flow"]
    recent_diff = flow_summary["flow_imbalance"]
    flow_status = flow_summary["flow_status"]
    if flow_status == "DANGER":
        flow_color = (0, 0, 255)
    elif flow_status == "WARNING":
        flow_color = (0, 165, 255)
    else:
        flow_color = (0, 255, 0)
    alert_status = alert_summary.overall_status
    if alert_status == "DANGER":
        alert_color = (0, 0, 255)
    elif alert_status == "WARNING":
        alert_color = (0, 165, 255)
    else:
        alert_color = (0, 255, 0)
    cv2.rectangle(frame, (12, 12), (790, 224), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (790, 224), (80, 80, 80), 1)
    cv2.putText(frame, f"Frame: {frame_index}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"ROI Persons: {roi_person_count}", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"IN: {flow_summary['total_in']} | OUT: {flow_summary['total_out']} | STAY: {flow_summary['stay_count']}", (25, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"DIFF: {net_flow:+d} | RECENT DIFF: {recent_diff:+d} | FLOW: {flow_status}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, flow_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"GRID ALERT: {alert_status} | WARNING: {alert_summary.warning_cell_count} | DANGER: {alert_summary.danger_cell_count}", (25, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.75, alert_color, 2, cv2.LINE_AA)


def read_first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame from video.")
    return frame


def run_interactive_roi_setup_if_needed(args, video_path: Path, roi_path: Path):
    if args.skip_interactive_setup:
        print("[INFO] Interactive setup skipped. Reusing saved ROI config.")
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI config not found: {roi_path}\nRun without --skip-interactive-setup to create it.")
        return
    print("[INFO] Interactive ROI setup mode enabled.")
    print("[INFO] First frame will open for ROI setup.")
    first_frame = read_first_frame(video_path)
    run_roi_editor_on_frame(frame=first_frame, video_path=video_path, output_path=roi_path, camera_id=args.camera_id, roi_name=f"{video_path.stem}_roi", max_display_width=args.max_display_width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-model", default=str(PROJECT_ROOT / "models" / "ccrv_head_v5.pt"))
    parser.add_argument("--body-model", default=str(PROJECT_ROOT / "models" / "body_wider_labeling.pt"))
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data" / "E05_008.mp4"))
    parser.add_argument("--camera-id", default="cam_001")
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--skip-interactive-setup", action="store_true", help="Reuse saved ROI config without opening setup window.")
    parser.add_argument("--roi", default=None)
    parser.add_argument("--alert-config", default=str(PROJECT_ROOT / "analyzer" / "configs" / "alert_config.json"))
    parser.add_argument("--db", default=None)
    parser.add_argument("--head-conf", type=float, default=0.20)
    parser.add_argument("--body-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-show", dest="show", action="store_false", default=True)
    parser.add_argument("--device", default="auto", help="auto, 0 for GPU, cpu for CPU.")
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--flow-recent-window", type=int, default=150)
    parser.add_argument("--flow-warning-threshold", type=int, default=5)
    parser.add_argument("--flow-danger-threshold", type=int, default=10)
    parser.add_argument("--cross-cooldown", type=int, default=20)
    parser.add_argument("--send-backend", action="store_true")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--backend-endpoint", default="/api/analysis/snapshot")
    parser.add_argument("--send-every-n-frames", type=int, default=10)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    roi_path = Path(args.roi) if args.roi else default_roi_path(video_path)
    alert_config_path = Path(args.alert_config)
    db_path = Path(args.db) if args.db else default_db_path()
    tracker_config = args.tracker_config if args.tracker_config else default_tracker_config_path()
    resolved_device = resolve_device(args.device)

    print(f"[INFO] Device requested: {args.device}")
    print(f"[INFO] Device selected : {resolved_device}")
    print(f"[INFO] Tracker config  : {tracker_config}")

    run_interactive_roi_setup_if_needed(args=args, video_path=video_path, roi_path=roi_path)

    print("[INFO] Loading ROI...")
    roi_manager = ROIManager.from_json(str(roi_path))
    print(f"[INFO] ROI config: {roi_path}")
    print(f"[INFO] ROI: {roi_manager.to_dict()}")
    print("[INFO] Loading alert config...")
    alert_config = load_alert_config(alert_config_path)
    print(f"[INFO] Alert config: {alert_config}")
    print("[INFO] Loading models...")
    print("[INFO] Body tracking: ByteTrack")
    print("[INFO] Flow counting: ROI boundary crossing")

    body_tracker = ByteTrackPersonTracker(body_model_path=args.body_model, conf=args.body_conf, imgsz=args.imgsz, device=resolved_device, tracker_config=tracker_config)
    head_detector = HeadDetector(head_model_path=args.head_model, conf=args.head_conf, imgsz=args.imgsz, device=resolved_device)
    flow_counter = ROIFlowCounter(cooldown_frames=args.cross_cooldown, recent_window_frames=args.flow_recent_window, warning_threshold=args.flow_warning_threshold, danger_threshold=args.flow_danger_threshold)

    backend_client = None
    if args.send_backend:
        backend_client = BackendClient(backend_url=args.backend_url, endpoint=args.backend_endpoint)
        print(f"[INFO] Backend sending enabled: {backend_client.url}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grid_manager = GridManager(cell_width=int(alert_config["cell_width"]), cell_height=int(alert_config["cell_height"]))
    grid_manager.build_cells(frame_width=width, frame_height=height, roi_manager=roi_manager)
    alert_engine = AlertEngine(warning_threshold=int(alert_config["warning_threshold"]), danger_threshold=int(alert_config["danger_threshold"]), persist_frames=int(alert_config["persist_frames"]))

    frame_index = 0
    start_time = time.time()
    print("[INFO] Start analyzing video with ByteTrack + ROI boundary flow...")
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Database: {db_path}")

    db_logger = DBLogger(db_path=db_path, video_name=video_path.stem)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        all_tracks = body_tracker.update(frame)
        all_heads = head_detector.detect(frame)
        roi_tracks = [person for person in all_tracks if roi_manager.contains_point(person.point)]
        roi_heads = roi_manager.filter_detections(all_heads, point_attr="center")

        flow_counter.update(tracked_persons=all_tracks, roi_manager=roi_manager, frame_index=frame_index)
        roi_person_count = max(len(roi_tracks), len(roi_heads))
        flow_summary = flow_counter.get_summary(current_roi_person_count=roi_person_count)

        person_points = [person.point for person in roi_tracks]
        grid_manager.count_points(person_points)
        alert_summary = alert_engine.update_cells(grid_manager.cells)

        db_logger.insert(frame_index=frame_index, in_count=flow_summary["total_in"], out_count=flow_summary["total_out"], roi_person_count=roi_person_count)
        minimal_payload = {"in_count": flow_summary["total_in"], "out_count": flow_summary["total_out"], "roi_person_count": roi_person_count}
        if backend_client is not None and frame_index % args.send_every_n_frames == 0:
            backend_client.send_snapshot(minimal_payload)

        if args.show:
            roi_manager.draw(frame)
            grid_manager.draw(frame)
            for person in roi_tracks:
                draw_track(frame, person, color=(255, 0, 0))
            for head in roi_heads:
                x1, y1, x2, y2 = head.xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            draw_status_panel(frame=frame, frame_index=frame_index, roi_person_count=roi_person_count, flow_summary=flow_summary, alert_summary=alert_summary)
            cv2.imshow("CCTV Crowd Analyzer - ByteTrack ROI Flow", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Stopped by user.")
                break
        if args.max_frames > 0 and frame_index >= args.max_frames:
            break
        if frame_index % 30 == 0:
            print(f"[INFO] frame={frame_index}, roi_persons={roi_person_count}, tracks={len(roi_tracks)}, heads={len(roi_heads)}, in={flow_summary['total_in']}, out={flow_summary['total_out']}, diff={flow_summary['net_flow']}, recent_diff={flow_summary['flow_imbalance']}, flow={flow_summary['flow_status']}, alert={alert_summary.overall_status}")

    db_logger.close()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print("[DONE] Analysis finished.")
    print(f"[DONE] Frames processed: {frame_index}")
    print(f"[DONE] Elapsed: {elapsed:.2f}s")
    print(f"[DONE] Database: {db_path}")


if __name__ == "__main__":
    main()
