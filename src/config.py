"""Central configuration for the ecobox detection pipeline.

All tuning parameters live here (or in a JSON file that overrides the
defaults) so no magic numbers are scattered across the codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    # --- Camera ---
    camera_index: int = 1
    # Full-resolution the webcam is opened at; monitoring downsamples from here.
    capture_width: int = 1920
    capture_height: int = 1080

    # --- Monitoring resolution (low-res processing happens at this size) ---
    monitor_width: int = 640
    monitor_height: int = 360
    # Number of frames processed per second during monitoring.
    monitor_fps: float = 3.0

    # --- ROI (Region of Interest), in monitor-frame pixel coordinates ---
    # (x, y, width, height). Defaults to a centered box; set via calibration.
    roi_x: int = 160
    roi_y: int = 90
    roi_width: int = 320
    roi_height: int = 180

    # --- Detection ---
    # Per-pixel absolute difference threshold to consider a pixel "changed".
    diff_threshold: int = 25
    # Per-pixel absolute difference threshold for the inter-frame motion
    # metric (used to decide when the object has stopped moving).
    motion_diff_threshold: int = 10
    # Minimum fraction of ROI pixels that must change to report a difference.
    min_changed_ratio: float = 0.02
    # Number of consecutive positive frames to confirm an object.
    confirm_frames: int = 3
    # Gaussian blur kernel size (odd) applied before comparison.
    blur_ksize: int = 5
    # Morphology "open" kernel size (odd); 0 disables morphology.
    morph_ksize: int = 3

    # --- Stability ---
    # Maximum changed ratio per frame to consider the object "stable".
    stability_threshold: float = 0.005
    # Minimum stable duration (seconds) before capture is allowed.
    stability_time: float = 0.8

    # --- Background ---
    # Number of frames to average (median) when calibrating the background.
    background_frames: int = 15
    # Seconds to wait between background sample frames during calibration.
    background_sample_interval: float = 0.1
    # Directory where the calibrated background is stored.
    background_path: str = "background.npy"
    # Path to a saved config JSON (optional).
    config_path: str = "config.json"
    # Directory where captured images are saved.
    captures_dir: str = "captures"

    # --- Debug ---
    # When True, opens debug windows and overlays.
    debug: bool = False
    # Preview refresh rate in debug mode. Higher = fresher feed, more CPU.
    # Kept separate from monitor_fps so detection cadence stays low.
    debug_display_fps: float = 10.0
    # When True, show the image to be sent to Gemini and ask for user approval
    # before each request; also show the response afterwards (debugging).
    gemini_debug: bool = False

    # --- Gemini cloud classification ---
    # NOTE: the API key is NOT stored here. It is read from the
    # GEMINI_API_KEY environment variable (see src/gemini.py).
    gemini_model: str = "gemini-3.5-flash-lite"
    # Per-request timeout in seconds (connection + response).
    gemini_timeout: float = 30.0
    # Minimum confidence to accept a classification (0.0..1.0).
    classification_confidence_threshold: float = 0.85
    # Maximum number of classification attempts per object (1 = no retry).
    max_classification_attempts: int = 2
    # Seconds to wait between classification attempts (before re-capturing).
    classification_retry_delay: float = 1.0
    # Image sent to Gemini: ROI cropped, letterboxed into this box
    # (aspect ratio preserved).
    classification_image_width: int = 640
    classification_image_height: int = 640
    # JPEG quality for the image sent to Gemini (0..100, 75..85 recommended).
    jpeg_quality: int = 80

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return (self.roi_x, self.roi_y, self.roi_width, self.roi_height)

    def with_roi(self, x: int, y: int, w: int, h: int) -> Config:
        """Return a copy of this config with a new ROI."""
        d = asdict(self)
        d.update(roi_x=x, roi_y=y, roi_width=w, roi_height=h)
        return Config(**d)

    def save(self, path: Path | str | None = None) -> None:
        """Persist this config to JSON so calibration survives restarts."""
        target = Path(path) if path else Path(self.config_path)
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        # Do not persist the config_path itself recursively / irrelevant keys.
        data.pop("config_path", None)
        data.pop("debug", None)
        data.pop("background_path", None)
        data.pop("captures_dir", None)
        target.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: Path | str) -> Config:
        """Load a config from JSON, falling back to defaults for missing keys."""
        base = Config()
        data: dict[str, Any] = json.loads(Path(path).read_text())
        base_data = asdict(base)
        base_data.update({k: v for k, v in data.items() if k in base_data})
        return Config(**base_data)


def load_or_default(config_path: str) -> Config:
    """Load config from ``config_path`` if it exists, else defaults."""
    p = Path(config_path)
    if p.exists():
        return Config.load(p)
    return Config()
