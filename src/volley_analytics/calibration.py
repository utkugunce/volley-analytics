"""Court calibration: 4-corner click → homography → pixel↔court mapping.

The homography maps pixel coordinates in the source video to real-world
court coordinates in meters on a standard 9 × 18 m FIVB court. Corner
order is TL, TR, BR, BL (top-left first, clockwise) in both spaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COURT_W = 9.0   # sideline-to-sideline (m)
COURT_H = 18.0  # endline-to-endline (m)


def default_court_corners() -> np.ndarray:
    """TL, TR, BR, BL on the 9 × 18 m court, in meters."""
    return np.array(
        [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_H], [0.0, COURT_H]],
        dtype=np.float64,
    )


@dataclass
class Calibration:
    H: np.ndarray            # 3 × 3 homography, pixel → court (m)
    image_corners: np.ndarray  # 4 × 2 pixel corners (TL, TR, BR, BL)
    court_corners: np.ndarray  # 4 × 2 court corners (m)

    def pixel_to_court(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def court_to_pixel(self, pts_m) -> np.ndarray:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "H": self.H.tolist(),
            "image_corners": self.image_corners.tolist(),
            "court_corners": self.court_corners.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(
            H=np.array(d["H"], dtype=np.float64),
            image_corners=np.array(d["image_corners"], dtype=np.float64),
            court_corners=np.array(d["court_corners"], dtype=np.float64),
        )


def compute_calibration(image_corners, court_corners=None) -> Calibration:
    """Build a Calibration from 4 pixel corners and 4 court corners.

    `image_corners` and `court_corners` must both be ordered
    TL, TR, BR, BL. `court_corners` defaults to a 9 × 18 m court.
    """
    image_corners = np.asarray(image_corners, dtype=np.float64)
    if image_corners.shape != (4, 2):
        raise ValueError(f"image_corners must be (4, 2), got {image_corners.shape}")
    if court_corners is None:
        court_corners = default_court_corners()
    else:
        court_corners = np.asarray(court_corners, dtype=np.float64)
        if court_corners.shape != (4, 2):
            raise ValueError(f"court_corners must be (4, 2), got {court_corners.shape}")

    H, _ = cv2.findHomography(image_corners, court_corners, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed; corners may be degenerate.")
    return Calibration(H=H, image_corners=image_corners, court_corners=court_corners)


def click_corners(frame, window_name: str = "calibration") -> np.ndarray:
    """Show `frame` in an OpenCV window and collect 4 clicks (TL, TR, BR, BL).

    Press ESC to abort. Run on a desktop session — does not work on a
    headless server.
    """
    points: list[tuple[int, int]] = []
    display = frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(window_name, display)

    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while len(points) < 4:
            if (cv2.waitKey(20) & 0xFF) == 27:  # ESC
                break
    finally:
        cv2.destroyWindow(window_name)

    if len(points) != 4:
        raise RuntimeError("Calibration aborted before 4 points were clicked.")
    return np.array(points, dtype=np.float64)
