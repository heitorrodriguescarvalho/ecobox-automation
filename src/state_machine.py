"""State machine coordinating detection, stability, capture and processing.

States:
    EMPTY -> POSSIBLE_OBJECT -> OBJECT_DETECTED -> WAITING_FOR_STABILITY
          -> READY_TO_CAPTURE -> CLASSIFYING -> CLASSIFIED
          -> WAITING_FOR_EMPTY -> EMPTY

While CLASSIFYING the machine ignores analyses: no new captures, no new
detections, no separation until the classification is accepted.

The machine is driven by feeding it an ``Analysis`` (changed ratio) each tick
and advancing ``now()`` time.
"""

from __future__ import annotations

import enum
import time

from .config import Config
from .detector import Analysis


class State(enum.Enum):
    EMPTY = "EMPTY"
    POSSIBLE_OBJECT = "POSSIBLE_OBJECT"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    WAITING_FOR_STABILITY = "WAITING_FOR_STABILITY"
    READY_TO_CAPTURE = "READY_TO_CAPTURE"
    CLASSIFYING = "CLASSIFYING"
    CLASSIFIED = "CLASSIFIED"
    WAITING_FOR_EMPTY = "WAITING_FOR_EMPTY"


class StateMachine:
    def __init__(self, config: Config):
        self.config = config
        self.state = State.EMPTY
        self._confirm_count = 0
        self._stable_since: float | None = None

    def reset(self) -> None:
        self.__init__(self.config)

    def update(self, analysis: Analysis, now: float | None = None) -> State:
        now = now if now is not None else time.monotonic()
        ratio = analysis.changed_ratio
        motion = analysis.motion_ratio
        cfg = self.config

        if self.state is State.EMPTY:
            if ratio >= cfg.min_changed_ratio:
                self._confirm_count = 1
                self._set(State.POSSIBLE_OBJECT)
            return self.state

        if self.state is State.POSSIBLE_OBJECT:
            if ratio >= cfg.min_changed_ratio:
                self._confirm_count += 1
                if self._confirm_count >= cfg.confirm_frames:
                    self._confirm_count = 0
                    self._set(State.OBJECT_DETECTED)
            else:
                self._confirm_count = 0
                self._set(State.EMPTY)
            return self.state

        if self.state is State.OBJECT_DETECTED:
            self._stable_since = None
            self._set(State.WAITING_FOR_STABILITY)
            return self.state

        if self.state is State.WAITING_FOR_STABILITY:
            # Stability = the object is not moving between consecutive frames
            # (inter-frame motion below threshold), NOT similarity to the empty
            # background. Otherwise the system would only "stabilize" after the
            # object is removed.
            if motion <= cfg.stability_threshold:
                if self._stable_since is None:
                    self._stable_since = now
                elif now - self._stable_since >= cfg.stability_time:
                    self._stable_since = None
                    self._set(State.READY_TO_CAPTURE)
            else:
                self._stable_since = None
                self._set(State.OBJECT_DETECTED)
            return self.state

        if self.state is State.READY_TO_CAPTURE:
            # Capture is triggered externally via ``finish_capture()``; this
            # state just waits. Do not advance automatically here.
            return self.state

        if self.state in (State.CLASSIFYING, State.CLASSIFIED):
            # Waiting on the cloud API / result handling. Ignore analyses:
            # no new captures or detections while classifying.
            return self.state

        if self.state is State.WAITING_FOR_EMPTY:
            if ratio < cfg.min_changed_ratio:
                self._set(State.EMPTY)
            return self.state

        return self.state

    def finish_capture(self) -> None:
        """Called after the high-res frame was captured (starts classifying)."""
        if self.state is State.READY_TO_CAPTURE:
            self._set(State.CLASSIFYING)

    def finish_classification(self) -> None:
        """Called when classification (incl. retries) has finished."""
        if self.state is State.CLASSIFYING:
            self._set(State.CLASSIFIED)

    def finish_processing(self) -> None:
        """Called after the result was handled; waits for the bin to empty."""
        if self.state is State.CLASSIFIED:
            self._set(State.WAITING_FOR_EMPTY)

    def _set(self, state: State) -> None:
        self.state = state
