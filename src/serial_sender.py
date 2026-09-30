"""Serial actuator link — send a message per trash type when an object is detected.

The pipeline classifies each object (plastic / paper / metal / non-recyclable /
unknown) and this module translates the category into the bytes written to the
serial port (e.g. an Arduino driving servos/flaps).

Mapping object (category -> message)::

    TRASH_SERIAL_MESSAGES = {
        "plastic": "PLASTIC",
        "paper": "PAPER",
        "metal": "METAL",
        "non-recyclable": "NON_RECYCLABLE",
    }

``unknown`` (and anything unrecognized) is deliberately NOT in the map: it
falls back to the ``non-recyclable`` message so inconclusive objects are
sorted as non-recyclable instead of being dropped.

``SerialSender`` is a no-op when ``config.serial_enabled`` is False, so the
pipeline runs unchanged without hardware. When enabled, ``pyserial`` is
imported lazily (only on first open) and every failure is raised as
``SerialError`` so callers can log it without crashing the detection loop.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping

log = logging.getLogger("ecobox.serial")

# Object mapping trash type -> serial message (without line ending).
# Keep values short, uppercase, ASCII-only so any microcontroller can parse
# them with a simple line-based comparison.
TRASH_SERIAL_MESSAGES: dict[str, str] = {
    "plastic": "PLASTIC",
    "paper": "PAPER",
    "metal": "METAL",
    "non-recyclable": "NON_RECYCLABLE",
}

# Category whose message is reused when the type is unknown/unrecognized.
FALLBACK_CATEGORY = "non-recyclable"


class SerialError(RuntimeError):
    """A serial-port failure (missing lib, port busy, write error, ...)."""


def resolve_serial_message(
    category: str,
    messages: Mapping[str, str] | None = None,
) -> str:
    """Return the serial message for ``category``.

    ``unknown`` (or any value not in the map) resolves to the
    ``non-recyclable`` message, per the "unknown -> non-recyclable" rule.
    """
    table: Mapping[str, str] = messages if messages is not None else TRASH_SERIAL_MESSAGES
    key = (category or "").strip().lower()
    if key in table:
        return table[key]
    fallback = table.get(FALLBACK_CATEGORY, TRASH_SERIAL_MESSAGES[FALLBACK_CATEGORY])
    return fallback


class SerialSender:
    """Line-oriented serial writer, disabled by default (no-op)."""

    def __init__(self, config) -> None:
        self.config = config
        self._serial = None
        self._lock = threading.Lock()

    # --- State ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "serial_enabled", False))

    @property
    def is_open(self) -> bool:
        return self._serial is not None

    # --- Lifecycle -----------------------------------------------------
    def open(self):
        """Open the port if enabled. Safe to call repeatedly. Returns self."""
        if not self.enabled:
            return self
        with self._lock:
            if self._serial is not None:
                return self
            try:
                import serial  # type: ignore  # pyserial, lazy so tests w/o hw work
            except ImportError as e:
                raise SerialError(
                    "pyserial is not installed (run `uv sync` / `pip install pyserial`)"
                ) from e
            try:
                self._serial = serial.Serial(
                    port=self.config.serial_port,
                    baudrate=self.config.serial_baudrate,
                    timeout=self.config.serial_timeout,
                )
            except Exception as e:
                raise SerialError(
                    f"Could not open serial port {self.config.serial_port}: {e}"
                ) from e
            log.info(
                "Serial open: %s @ %d baud",
                self.config.serial_port,
                self.config.serial_baudrate,
            )
            return self

    def close(self) -> None:
        with self._lock:
            ser, self._serial = self._serial, None
        if ser is not None:
            try:
                ser.close()
            except Exception as e:  # noqa: BLE001 -- never crash shutdown on close
                log.warning("Error closing serial port: %s", e)

    # --- Sending -------------------------------------------------------
    def message_for(self, category: str) -> str:
        """Resolve ``category`` to its wire message (unknown -> non-recyclable)."""
        messages = getattr(self.config, "serial_messages", None) or None
        return resolve_serial_message(category, messages)

    def send(self, category: str) -> str | None:
        """Send the message for ``category``. Returns it, or None if disabled.

        Raises ``SerialError`` on failure so the pipeline can log it and keep
        detecting (serial must never kill the loop).
        """
        message = self.message_for(category)
        if not self.enabled:
            log.debug("Serial disabled: would send %r for %r", message, category)
            return None
        self.open()
        line = f"{message}{getattr(self.config, 'serial_line_ending', chr(10))}"
        data = line.encode("ascii", errors="replace")
        try:
            with self._lock:
                assert self._serial is not None
                self._serial.write(data)
                self._serial.flush()
        except Exception as e:
            raise SerialError(f"Serial write failed ({message!r}): {e}") from e
        log.info("Serial sent: %r for category %r", message, category)
        return message
