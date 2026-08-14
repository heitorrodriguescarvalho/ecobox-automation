"""High-resolution capture and optional saving."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config


class Capture:
    def __init__(self, config: Config):
        self.config = config

    def grab(self, camera) -> np.ndarray:
        """Return the latest full-resolution BGR frame (the capture image)."""
        frame = camera.read_raw()
        if frame is None:
            raise RuntimeError("Could not read full-resolution frame")
        return frame

    def save(self, image: np.ndarray, filename: str) -> Path:
        """Persist a captured image under the captures directory."""
        directory = Path(self.config.captures_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        cv2.imwrite(str(path), image)
        return path


def prepare_classification_image(
    image: np.ndarray,
    config: Config,
    crop_roi: bool = True,
) -> bytes:
    """Prepare a full-res BGR frame for the vision API.

    Flow: crop the ROI (scaled from monitor-frame coordinates) -> resize
    preserving aspect ratio into the classification box -> JPEG bytes.

    Only the (small) cropped/resized data is copied, never the full frame.
    ``crop_roi=False`` uses the whole image (used by the ``--image`` test mode,
    where no calibrated ROI exists).
    """
    src = image
    if crop_roi:
        src = _crop_roi_scaled(image, config)

    tw, th = config.classification_image_width, config.classification_image_height
    sh, sw = src.shape[:2]
    if sw == 0 or sh == 0:
        raise RuntimeError("Empty image after ROI crop; check ROI calibration")

    scale = min(tw / sw, th / sh)
    new_w = max(1, round(sw * scale))
    new_h = max(1, round(sh * scale))
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    off_x = (tw - new_w) // 2
    off_y = (th - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    ok, buf = cv2.imencode(
        ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _crop_roi_scaled(image: np.ndarray, config: Config) -> np.ndarray:
    """Crop the monitor-space ROI scaled to the full-res frame coordinates."""
    x, y, w, h = config.roi
    img_h, img_w = image.shape[:2]
    scale_x = img_w / max(config.monitor_width, 1)
    scale_y = img_h / max(config.monitor_height, 1)
    rx = max(0, min(int(x * scale_x), img_w - 1))
    ry = max(0, min(int(y * scale_y), img_h - 1))
    rw = max(1, min(int(w * scale_x), img_w - rx))
    rh = max(1, min(int(h * scale_y), img_h - ry))
    return image[ry : ry + rh, rx : rx + rw]
