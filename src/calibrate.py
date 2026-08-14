"""Interactive calibration: mouse-drag ROI + background capture.

Usage:
    uv run python -m src.calibrate [--config config.json] [--index N]

Controls (inside the "calibrate" window):
    drag left button   : draw the ROI box over the live feed
    c                  : capture background (median of N frames)
    r                  : reset ROI to centered default
    s                  : save ROI + background to disk, then exit
    q / ESC            : quit without saving
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera
from .config import load_or_default
from .detector import Detector, calibrate_background


@dataclass
class _Roi:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    drawing: bool = False
    done: bool = False


def _mouse(event, x, y, flags, param):
    roi: _Roi = param
    if event == cv2.EVENT_LBUTTONDOWN:
        roi.x, roi.y, roi.w, roi.h = x, y, 0, 0
        roi.drawing, roi.done = True, False
    elif event == cv2.EVENT_MOUSEMOVE and roi.drawing:
        roi.w, roi.h = x - roi.x, y - roi.y
    elif event == cv2.EVENT_LBUTTONUP:
        roi.drawing = False
        roi.done = True
        roi.w, roi.h = abs(roi.w), abs(roi.h)
        roi.x, roi.y = min(roi.x, roi.x + roi.w), min(roi.y, roi.y + roi.h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ROI and background")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--index", type=int, default=None, help="camera index")
    args = parser.parse_args()

    config = load_or_default(args.config or "config.json")
    if args.index is not None:
        config = _replace(config, camera_index=args.index)

    camera = Camera(config)
    detector = Detector(config)

    window = "calibrate"
    cv2.namedWindow(window)
    roi = _Roi()
    cv2.setMouseCallback(window, _mouse, roi)

    background = None
    print("Calibration started. Drag to draw ROI, 'c' to capture background.")
    print("Press 's' to save & exit, 'q'/'ESC' to quit without saving.")

    try:
        while True:
            bgr = camera.read_monitor()
            if bgr is None:
                continue

            overlay = bgr.copy()

            if roi.done and roi.w > 2 and roi.h > 2:
                cv2.rectangle(
                    overlay,
                    (roi.x, roi.y),
                    (roi.x + roi.w, roi.y + roi.h),
                    (0, 255, 0),
                    2,
                )

            status = []
            status.append(f"ROI: {roi.w}x{roi.h} @ ({roi.x},{roi.y})")
            status.append("BG: {}".format("OK" if background is not None else "none"))
            y = 20
            for line in status:
                cv2.putText(
                    overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
                y += 20

            cv2.imshow(window, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                cx, cy = config.monitor_width // 2, config.monitor_height // 2
                w, h = config.roi_width, config.roi_height
                roi.x, roi.y, roi.w, roi.h = cx - w // 2, cy - h // 2, w, h
                roi.done = True
                print("ROI reset to centered default.")
            if key == ord("c"):
                if not (roi.done and roi.w > 2 and roi.h > 2):
                    print("Draw an ROI first (drag with mouse).")
                    continue
                config = config.with_roi(roi.x, roi.y, roi.w, roi.h)
                detector = Detector(config)
                background = calibrate_background(detector, camera, config)
                detector.set_background(background)
                print(f"Background captured ({len(background.ravel())} px).")
            if key == ord("s"):
                if not (roi.done and roi.w > 2 and roi.h > 2):
                    print("Draw an ROI first (drag with mouse).")
                    continue
                if background is None:
                    print("Capture the background first (press 'c').")
                    continue
                config = config.with_roi(roi.x, roi.y, roi.w, roi.h)
                np.save(config.background_path, background)
                config.save()
                print("Saved background ->", config.background_path)
                print("Saved config   ->", config.config_path)
                break
    finally:
        cv2.destroyAllWindows()
        camera.release()


def _replace(config, **kwargs):
    from dataclasses import asdict

    d = asdict(config)
    d.update(kwargs)
    return type(config)(**d)


if __name__ == "__main__":
    main()
