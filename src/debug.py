"""Optional debug rendering. No-ops unless a window is created.

In normal (non-debug) mode main.py never calls into this module, so no
windows or UI resources are created.
"""

from __future__ import annotations

import cv2

from .detector import Analysis
from .state_machine import State


def render_debug(
    frame: cv2.Mat,
    background_gray: cv2.Mat,
    analysis: Analysis,
    state: State,
    roi,
    changed_ratio: float,
) -> cv2.Mat:
    """Compose a debug view of monitor frame + ROI + background + diff mask."""
    x, y, w, h = roi
    view = frame.copy()
    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)

    bg = (
        cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
        if background_gray is not None
        else None
    )

    text = (
        f"{state.name}  changed={changed_ratio:.3f}  "
        f"motion={analysis.motion_ratio:.3f}"
    )
    cv2.putText(view, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Place background and diff mask side-by-side under the frame.
    mask = cv2.cvtColor(analysis.diff_mask, cv2.COLOR_GRAY2BGR)
    if bg is None:
        bg = mask.copy()
    combined = cv2.hconcat([bg, mask])
    scale = min(1.0, frame.shape[1] / (combined.shape[1] or 1))
    if scale < 1.0:
        combined = cv2.resize(
            combined, (frame.shape[1], int(combined.shape[0] * scale))
        )
    canvas = cv2.vconcat([view, combined])
    return canvas


def show_debug(title: str, canvas: cv2.Mat) -> None:
    cv2.imshow(title, canvas)
    cv2.waitKey(1)
