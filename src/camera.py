"""Camera wrapper.

We keep a single VideoCapture handle opened at the full capture resolution.
Monitoring frames are produced by downscaling the current frame, avoiding
the cost and unreliability of re-opening / reconfiguring the camera to swap
resolutions on demand.

The camera only runs while frames are read; reading is done by the caller at
a throttled rate (see main.py), so CPU stays low.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config


class Camera:
    def __init__(self, config: Config):
        self.config = config
        self._cap = cv2.VideoCapture(config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {config.camera_index}")
        # Request full resolution; some cameras only honor closest supported.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
        # Keep only the newest frame in the driver buffer. When reading at a
        # low rate (2-5 FPS) the kernel buffer backfills with high-rate frames
        # and read() eventually returns stale frames, freezing the feed.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read_raw(self) -> np.ndarray | None:
        """Return the latest BGR frame at full capture resolution."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def read_monitor(self) -> np.ndarray | None:
        """Return a downscaled BGR frame at monitor resolution."""
        frame = self.read_raw()
        if frame is None:
            return None
        if (
            frame.shape[1] == self.config.monitor_width
            and frame.shape[0] == self.config.monitor_height
        ):
            return frame
        return cv2.resize(
            frame,
            (self.config.monitor_width, self.config.monitor_height),
            interpolation=cv2.INTER_AREA,
        )

    def release(self) -> None:
        self._cap.release()
