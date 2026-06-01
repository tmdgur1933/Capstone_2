import argparse
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_roi_output_path(video_path: Path) -> Path:
    """
    영상별 ROI config 경로를 자동 생성한다.

    예:
    data/E05_008.mp4
    -> analyzer/configs/E05_008_roi.json
    """
    return PROJECT_ROOT / "analyzer" / "configs" / f"{video_path.stem}_roi.json"


class PolygonROIEditor:
    def __init__(
        self,
        frame,
        camera_id: str,
        roi_name: str,
        output_path: Path,
        max_display_width: int = 1280,
    ):
        self.original_frame = frame
        self.camera_id = camera_id
        self.roi_name = roi_name
        self.output_path = output_path
        self.points = []
        self.saved = False

        self.original_height, self.original_width = frame.shape[:2]

        self.scale = 1.0
        if self.original_width > max_display_width:
            self.scale = max_display_width / self.original_width

        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)

        self.window_name = "Step 1/2 - ROI Setup"

    def original_to_display(self, point):
        x, y = point
        return int(x * self.scale), int(y * self.scale)

    def display_to_original(self, point):
        x, y = point
        return int(x / self.scale), int(y / self.scale)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_point = self.display_to_original((x, y))
            self.points.append(original_point)
            print(f"[CLICK] ROI point {len(self.points)} = {original_point}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"[UNDO] removed ROI point = {removed}")

    def draw(self):
        display = cv2.resize(
            self.original_frame.copy(),
            (self.display_width, self.display_height),
        )

        display_points = [self.original_to_display(p) for p in self.points]

        for idx, point in enumerate(display_points):
            cv2.circle(display, point, 5, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(idx + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(display_points) >= 2:
            for i in range(len(display_points) - 1):
                cv2.line(display, display_points[i], display_points[i + 1], (0, 255, 255), 2)

        if len(display_points) >= 3:
            cv2.line(display, display_points[-1], display_points[0], (0, 255, 255), 2)

        guide_lines = [
            "STEP 1/2: ROI SETUP",
            "Left click: add polygon point",
            "Right click: undo last point",
            "R: reset",
            "Enter: save ROI",
            "ESC: cancel",
            "Points must follow boundary clockwise or counter-clockwise",
        ]

        y = 30
        for line in guide_lines:
            cv2.putText(
                display,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

        cv2.putText(
            display,
            f"Points: {len(self.points)}",
            (20, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return display

    def save(self):
        if len(self.points) < 3:
            raise ValueError("ROI needs at least 3 points.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "camera_id": self.camera_id,
            "roi_name": self.roi_name,
            "points": [[int(x), int(y)] for x, y in self.points],
        }

        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.saved = True
        print(f"[DONE] ROI saved: {self.output_path}")
        print(json.dumps(data, ensure_ascii=False, indent=2))

    def run(self) -> bool:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("[INFO] ROI editor started.")
        print("[INFO] Left click: add point")
        print("[INFO] Right click: undo last point")
        print("[INFO] R: reset")
        print("[INFO] Enter: save and continue")
        print("[INFO] ESC: cancel")

        while True:
            display = self.draw()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                print("[INFO] ROI setup canceled.")
                break

            if key == ord("r"):
                self.points = []
                print("[INFO] ROI reset.")

            if key == 13 or key == 10:
                try:
                    self.save()
                    break
                except ValueError as e:
                    print(f"[ERROR] {e}")

        cv2.destroyWindow(self.window_name)
        return self.saved


def run_roi_editor_on_frame(
    *,
    frame,
    video_path: Path,
    output_path: Path | None = None,
    camera_id: str = "cam_001",
    roi_name: str | None = None,
    max_display_width: int = 1280,
) -> Path:
    output = output_path if output_path else default_roi_output_path(video_path)
    name = roi_name if roi_name else f"{video_path.stem}_roi"

    print(f"[INFO] ROI output: {output}")

    editor = PolygonROIEditor(
        frame=frame,
        camera_id=camera_id,
        roi_name=name,
        output_path=output,
        max_display_width=max_display_width,
    )

    saved = editor.run()

    if not saved:
        raise RuntimeError("ROI setup was canceled.")

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "data" / "E05_008.mp4"),
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output ROI config. If omitted, uses analyzer/configs/{video_stem}_roi.json",
    )
    parser.add_argument("--camera-id", default="cam_001")
    parser.add_argument("--roi-name", default=None)
    parser.add_argument("--max-display-width", type=int, default=1280)

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(args.output) if args.output else default_roi_output_path(video_path)
    roi_name = args.roi_name if args.roi_name else f"{video_path.stem}_roi"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read first frame from video.")

    print(f"[INFO] Video: {video_path}")

    run_roi_editor_on_frame(
        frame=frame,
        video_path=video_path,
        output_path=output_path,
        camera_id=args.camera_id,
        roi_name=roi_name,
        max_display_width=args.max_display_width,
    )


if __name__ == "__main__":
    main()
