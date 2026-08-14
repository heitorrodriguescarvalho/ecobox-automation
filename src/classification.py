"""Classification result type shared by the whole system.

The detection pipeline and state machine only depend on this lightweight
dataclass. Concrete classifiers (e.g. ``GeminiClassifier`` in ``gemini.py``)
produce instances of it, so the rest of the system never touches the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    attempts: int = 1
    success: bool = True
    message: str = ""
