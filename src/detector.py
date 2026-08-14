"""Background-difference detector (no ML).

Each analyzed frame is compared against a calibrated background. The detector
is a pure function over one frame: it returns the fraction of ROI pixels that
changed and the difference mask (for debug rendering).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class Analysis:
    # Fraction of ROI pixels that differ from the background (object presence).
    changed_ratio: float
    # Fraction of ROI pixels that differ from the previous frame (motion).
    motion_ratio: float
    diff_mask: np.ndarray
    roi_gray: np.ndarray


class Detector:
    def __init__(self, config: Config, background: np.ndarray | None = None):
        self.config = config
        # Background stored as the blurred grayscale of the ROI at monitor res.
        self.background: np.ndarray | None = None
        if background is not None:
            self.set_background(background)
        self._morph_kernel = None
        if config.morph_ksize > 1:
            self._morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (config.morph_ksize, config.morph_ksize),
            )
        # Previous blurred ROI, used to compute the inter-frame motion ratio.
        self._prev_roi_blur: np.ndarray | None = None

    # --- Background management ------------------------------------------
    def set_background(self, background: np.ndarray) -> None:
        """Set the background from an ROI-cropped grayscale array.

        ``background`` must have the same shape as the current ROI
        (``(roi_height, roi_width)``). It is blurred before being stored.
        """
        expected = (self.config.roi_height, self.config.roi_width)
        if background.shape != expected:
            raise ValueError(
                f"Background shape {background.shape} != ROI shape {expected}. "
                "Recalibrate the background (ecobox --recalibrate)."
            )
        self.background = self._blur(background)
        self._prev_roi_blur = None

    def set_background_from_bgr(self, bgr_frame: np.ndarray) -> None:
        """Set the background from a full monitor-resolution BGR frame."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        self.set_background(self._crop(gray))

    @property
    def has_background(self) -> bool:
        return self.background is not None

    # --- Helpers ---------------------------------------------------------
    def _crop(self, gray: np.ndarray) -> np.ndarray:
        x, y, w, h = self.config.roi
        return gray[y : y + h, x : x + w]

    def _blur(self, gray: np.ndarray) -> np.ndarray:
        k = self.config.blur_ksize
        if k <= 1:
            return gray
        return cv2.GaussianBlur(gray, (k, k), 0)

    # --- Analysis --------------------------------------------------------
    def analyze(self, monitor_frame_bgr: np.ndarray) -> Analysis:
        """Compare current frame against background; return changed ratio."""
        gray = cv2.cvtColor(monitor_frame_bgr, cv2.COLOR_BGR2GRAY)
        roi = self._crop(gray)
        roi_blur = self._blur(roi)

        if self.background is None:
            self._prev_roi_blur = roi_blur
            return Analysis(
                changed_ratio=0.0, motion_ratio=0.0, diff_mask=roi * 0, roi_gray=roi
            )

        diff = cv2.absdiff(roi_blur, self.background)
        _, mask = cv2.threshold(
            diff, self.config.diff_threshold, 255, cv2.THRESH_BINARY
        )

        if self._morph_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        total = mask.size
        changed = int(cv2.countNonZero(mask))
        ratio = changed / total if total else 0.0

        motion_ratio = 0.0
        if self._prev_roi_blur is not None:
            motion_diff = cv2.absdiff(roi_blur, self._prev_roi_blur)
            _, motion_mask = cv2.threshold(
                motion_diff,
                self.config.motion_diff_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            if self._morph_kernel is not None:
                motion_mask = cv2.morphologyEx(
                    motion_mask, cv2.MORPH_OPEN, self._morph_kernel
                )
            moved = int(cv2.countNonZero(motion_mask))
            motion_ratio = moved / total if total else 0.0

        self._prev_roi_blur = roi_blur
        return Analysis(
            changed_ratio=ratio, motion_ratio=motion_ratio, diff_mask=mask, roi_gray=roi
        )


# --- Background calibration ------------------------------------------------
def calibrate_background(
    detector: Detector, camera, config: Config, frames: int | None = None
) -> np.ndarray:
    """Capture ``frames`` monitor frames and return their pixel-wise median.

    The returned array is the blurred grayscale ROI (i.e. exactly what the
    detector expects via ``set_background``).
    """
    n = frames or config.background_frames
    interval = config.background_sample_interval
    samples: list[np.ndarray] = []

    import time as _time

    while len(samples) < n:
        bgr = camera.read_monitor()
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        x, y, w, h = config.roi
        samples.append(gray[y : y + h, x : x + w])
        _time.sleep(interval)

    stacked = np.median(np.stack(samples), axis=0).astype(np.uint8)
    return detector._blur(stacked)
