"""Main pipeline runner.

Wires: Camera -> Detector -> StateMachine -> Capture -> GeminiClassifier.

Usage:
    uv run ecobox [--config config.json] [--index N] [--debug] [--recalibrate]
    uv run ecobox --image photo.jpg      # classify a local image (no webcam)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .camera import Camera
from .capture import Capture, prepare_classification_image
from .classification import ClassificationResult
from .config import Config, load_or_default
from .debug import render_debug, show_debug
from .detector import Detector
from .gemini import (
    ClassificationError,
    ClassificationRejectedError,
    GeminiClassifier,
)
from .state_machine import State, StateMachine

load_dotenv()

log = logging.getLogger("ecobox.main")

# Lightweight logs on state transitions.
STATE_LOGS = {
    State.OBJECT_DETECTED: "Object detected",
    State.READY_TO_CAPTURE: "Object stabilized",
    State.WAITING_FOR_EMPTY: "Waiting for bin to become empty",
}


def sort_waste(category: str, confidence: float) -> None:
    """Actuator interface — replace with servo/motor logic later.

    Called only for accepted classifications (confidence >= threshold).
    """
    log.info("sort_waste: %s (confidence=%.2f)", category, confidence)


class Pipeline:
    def __init__(self, config: Config, classifier):
        self.config = config
        self.classifier = classifier
        self.camera = Camera(config)
        self.detector = Detector(config)
        self.state_machine = StateMachine(config)
        self.capture = Capture(config)
        self._background = self._load_background()
        self._prev_state: State | None = None

    def _load_background(self) -> np.ndarray | None:
        path = Path(self.config.background_path)
        if path.exists():
            bg = np.load(str(path))
            self.detector.set_background(bg)
            return bg
        return None

    def calibrate_now(self) -> None:
        from .detector import calibrate_background

        bg = calibrate_background(self.detector, self.camera, self.config)
        np.save(self.config.background_path, bg)
        self.detector.set_background(bg)
        self._background = bg
        log.info("Background recalibrated -> %s", self.config.background_path)

    def _notify_state(self, state: State) -> None:
        if state is not self._prev_state:
            msg = STATE_LOGS.get(state)
            if msg:
                log.info(msg)
            self._prev_state = state

    def run(self) -> None:
        cfg = self.config
        if not self.detector.has_background:
            log.error("No background found. Calibrate first: uv run ecobox-calibrate")
            raise SystemExit(1)

        interval = 1.0 / max(cfg.monitor_fps, 0.001)
        display_interval = 1.0 / max(cfg.debug_display_fps, 0.001)
        last_analysis = 0.0
        last_display = 0.0

        if cfg.debug:
            cv2_window_ready(gemini_debug=cfg.gemini_debug)
            if cfg.gemini_debug:
                # Keep the window small and out of the way so the terminal
                # (where approval and logs happen) stays visible.
                cv2_resize_window("ecobox-debug", 384, 340)
                cv2_move_window("ecobox-debug", 0, 0)

        try:
            while True:
                now = time.monotonic()

                if not cfg.debug:
                    if now - last_analysis < interval:
                        time.sleep(0.01)
                        continue
                    last_analysis = now
                    bgr = self.camera.read_monitor()
                    if bgr is None:
                        continue
                    analysis = self.detector.analyze(bgr)
                    state = self.state_machine.update(analysis, now=now)
                    self._notify_state(state)
                    if state is State.READY_TO_CAPTURE:
                        self._on_capture_ready()
                    continue

                # Debug mode: read/display fresh frames at a higher rate so the
                # preview stays responsive, but only feed the state machine at
                # the detection cadence (preserves timing + keeps CPU low).
                if now - last_display < display_interval:
                    cv2_wait_key(1)
                    continue
                last_display = now

                bgr = self.camera.read_monitor()
                if bgr is None:
                    continue

                analysis = self.detector.analyze(bgr)
                if now - last_analysis >= interval:
                    last_analysis = now
                    state = self.state_machine.update(analysis, now=now)
                    self._notify_state(state)
                    if state is State.READY_TO_CAPTURE:
                        self._on_capture_ready()

                canvas = render_debug(
                    bgr,
                    self._background,
                    analysis,
                    self.state_machine.state,
                    cfg.roi,
                    analysis.changed_ratio,
                )
                show_debug("ecobox-debug", canvas)
                key = cv2_wait_key(1) & 0xFF
                if key == ord("r"):
                    self.calibrate_now()
                if key in (ord("q"), 27):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2_destroy_windows()
            self.camera.release()

    def _on_capture_ready(self) -> None:
        """READY_TO_CAPTURE: capture, classify (with retry), handle result."""
        cfg = self.config
        # READY_TO_CAPTURE -> CLASSIFYING: block new detections.
        self.state_machine.finish_capture()

        first_image = self.capture.grab(self.camera)
        if cfg.debug:
            self.capture.save(
                first_image, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

        result = self._classify_with_retry(first_image)
        self.state_machine.finish_classification()  # CLASSIFYING -> CLASSIFIED

        self._handle_result(result)
        self.state_machine.finish_processing()  # CLASSIFIED -> WAITING_FOR_EMPTY

    def _classify_with_retry(self, first_image: np.ndarray) -> ClassificationResult:
        cfg = self.config
        threshold = cfg.classification_confidence_threshold
        max_attempts = max(1, cfg.max_classification_attempts)

        image = first_image
        last: ClassificationResult = ClassificationResult(
            category="unknown",
            confidence=0.0,
            success=False,
            message="no attempts made",
        )

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                # Low confidence (or error) -> re-capture a fresh image.
                time.sleep(max(0.0, cfg.classification_retry_delay))
                image = self.capture.grab(self.camera)
                log.info("Re-capturing for retry %d/%d", attempt, max_attempts)

            prepared = prepare_classification_image(image, cfg)
            log.info("Sending image to Gemini (attempt %d/%d)", attempt, max_attempts)

            try:
                last = self.classifier.classify(prepared)
                last = ClassificationResult(
                    category=last.category,
                    confidence=last.confidence,
                    attempts=attempt,
                    success=True,
                )
                if last.confidence >= threshold:
                    return last
                log.info(
                    "Confidence below threshold: %s (%.2f)",
                    last.category,
                    last.confidence,
                )
            except ClassificationRejectedError:
                log.info("Sending image skipped by user (gemini_debug)")
                return ClassificationResult(
                    category="unknown",
                    confidence=0.0,
                    attempts=attempt,
                    success=False,
                    message="skipped by user",
                )
            except ClassificationError as e:
                log.error("Gemini request failed: %s (%s)", e.kind, e.message)
                last = ClassificationResult(
                    category="unknown",
                    confidence=0.0,
                    attempts=attempt,
                    success=False,
                    message=e.message,
                )

        # All attempts done and still not accepted -> unknown.
        if last.confidence < threshold:
            log.info(
                "Classification inconclusive after %d attempt(s); treating as unknown",
                max_attempts,
            )
            return ClassificationResult(
                category="unknown",
                confidence=0.0,
                attempts=max_attempts,
                success=False,
                message=f"best confidence {last.confidence:.2f} below threshold",
            )
        return last

    def _handle_result(self, result: ClassificationResult) -> None:
        if (
            result.success
            and result.confidence >= self.config.classification_confidence_threshold
        ):
            log.info(
                "Classification: %s (confidence=%.2f)",
                result.category,
                result.confidence,
            )
            sort_waste(result.category, result.confidence)
        else:
            log.info(
                "Classification: unknown (attempts=%d, success=%s)",
                result.attempts,
                result.success,
            )


def cv2_window_ready(gemini_debug: bool = False) -> None:
    import cv2

    flags = cv2.WINDOW_NORMAL if gemini_debug else cv2.WINDOW_AUTOSIZE
    cv2.namedWindow("ecobox-debug", flags)


def cv2_resize_window(name: str, w: int, h: int) -> None:
    import cv2

    try:
        cv2.resizeWindow(name, w, h)
    except cv2.error:
        pass


def cv2_move_window(name: str, x: int, y: int) -> None:
    import cv2

    try:
        cv2.moveWindow(name, x, y)
    except cv2.error:
        pass


def cv2_wait_key(delay: int) -> int:
    import cv2

    return cv2.waitKey(delay)


def cv2_destroy_windows() -> None:
    import cv2

    cv2.destroyAllWindows()


def _build_config(args) -> Config:
    config = load_or_default(args.config or "config.json")
    if args.index is not None:
        from dataclasses import asdict

        d = asdict(config)
        d["camera_index"] = args.index
        config = Config(**d)
    want_debug = args.debug or os.environ.get("ECOBOX_DEBUG")
    want_gd = (
        args.gemini_debug
        or os.environ.get("ECOBOX_GEMINI_DEBUG")
        or config.gemini_debug
    )
    if want_debug or want_gd:
        # gemini_debug implies the camera preview window, so the user can
        # watch the feed while deciding on the terminal.
        from dataclasses import asdict

        d = asdict(config)
        d["debug"] = True
        config = Config(**d)
    if want_gd:
        from dataclasses import asdict

        d = asdict(config)
        d["gemini_debug"] = True
        config = Config(**d)
    return config


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    if not debug:
        # Keep our logs light: silence SDK/HTTP chatter unless debugging.
        for name in ("google", "google_genai", "httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


def run_image_test(image_path: Path, config: Config) -> None:
    """Classify a local image directly (no webcam, no actuators)."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        log.error("Could not read image: %s", image_path)
        raise SystemExit(1)

    classifier = GeminiClassifier(config)
    jpeg = prepare_classification_image(image, config, crop_roi=False)
    log.info("Sending image to Gemini: %s", image_path)
    try:
        result = classifier.classify(jpeg)
    except ClassificationRejectedError:
        log.info("Sending image skipped by user (gemini_debug)")
        return
    except ClassificationError as e:
        log.error("Gemini request failed: %s (%s)", e.kind, e.message)
        raise SystemExit(2)
    log.info(
        "Classification: %s (confidence=%.2f, attempts=%d)",
        result.category,
        result.confidence,
        result.attempts,
    )
    print(f"{result.category}\t{result.confidence:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ecobox waste detection pipeline")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--index", type=int, default=None, help="camera index")
    parser.add_argument(
        "--debug", action="store_true", help="enable debug windows + verbose logs"
    )
    parser.add_argument(
        "--gemini-debug",
        action="store_true",
        help="save the image to captures/ and ask approval on the terminal "
        "before each Gemini request; show the response afterwards "
        "(also opens the camera preview)",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="recalibrate background before starting",
    )
    parser.add_argument(
        "--image",
        default=None,
        metavar="PATH",
        help="classify a local image with Gemini and exit (no webcam). "
        "The whole image is used (ROI not applied).",
    )
    args = parser.parse_args()

    # Verbose SDK logging only for an explicit --debug/ECOBOX_DEBUG, not when
    # gemini_debug alone implies the preview window.
    explicit_debug = args.debug or bool(os.environ.get("ECOBOX_DEBUG"))

    config = _build_config(args)
    _setup_logging(explicit_debug)

    if args.image:
        run_image_test(Path(args.image), config)
        return

    classifier = GeminiClassifier(config)
    pipeline = Pipeline(config, classifier)
    if args.recalibrate:
        pipeline.calibrate_now()
    pipeline.run()


if __name__ == "__main__":
    main()
