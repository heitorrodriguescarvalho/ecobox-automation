"""Gemini API classifier — cloud vision for waste classification.

This module is the ONLY place that talks to the Gemini REST API. The detector
and state machine know nothing about it; they consume ``ClassificationResult``
via the ``GeminiClassifier.classify(bytes)`` interface.

Uses only the standard library (``urllib``) — no ``google-genai``, no
``httpx``. The API key comes from the ``GEMINI_API_KEY`` environment variable
and is never logged.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .classification import ClassificationResult
from .config import Config

log = logging.getLogger("ecobox.gemini")

API_KEY_ENV = "GEMINI_API_KEY"
API_HOST = "https://generativelanguage.googleapis.com"

ALLOWED_CATEGORIES = frozenset(
    {"plastic", "paper", "metal", "non-recyclable", "unknown"}
)

PROMPT = """\
Você é um classificador de resíduos.

Analise a imagem e identifique o material do objeto principal que foi
colocado na lixeira.

Escolha exatamente uma das categorias:

plastic
paper
metal
non-recyclable
unknown

Não crie novas categorias.

Use "non-recyclable" quando você conseguir identificar o material, mas ele
não for plástico, papel nem metal (por exemplo: vidro, orgânico, tecido,
borracha).

Use "unknown" somente quando você NÃO conseguir identificar o material com
confiança.

Não descreva a imagem e não explique sua decisão."""

CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "category": {"type": "STRING", "enum": sorted(ALLOWED_CATEGORIES)},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["category", "confidence"],
}


class ClassificationError(RuntimeError):
    """A failed Gemini request, normalized so callers can classify/retry.

    ``kind`` is one of: auth, rate_limit, timeout, http, unavailable,
    invalid_response.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class ClassificationRejectedError(RuntimeError):
    """The user declined to send the image (``gemini_debug`` mode)."""


def _preview_and_approve(image: bytes, captures_dir: str) -> bool:
    """Save the prepared image for review and ask for approval on the terminal.

    No image window is opened: the JPEG is written to ``captures_dir`` so the
    user can open it in the file manager, then approve/reject on the terminal.
    """
    from datetime import datetime

    directory = Path(captures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"gemini_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = directory / filename
    path.write_bytes(image)
    log.info("Image captured for review -> %s", path)
    print("[gemini-debug] Open the image in the file manager to inspect it.")
    try:
        answer = input("[gemini-debug] Send this image to Gemini? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "s")


def _show_response(result: ClassificationResult, raw_text: str, usage: Any) -> None:
    """Log the Gemini response on the terminal."""
    log.info("Gemini response: %s (confidence=%.2f)", result.category, result.confidence)
    if raw_text:
        log.info("Gemini raw response: %s", raw_text)
    if usage:
        log.info("Gemini usageMetadata: %s", usage)


class GeminiClassifier:
    def __init__(self, config: Config):
        self.config = config
        self._api_key = os.environ.get(API_KEY_ENV)
        if not self._api_key:
            raise RuntimeError(
                f"Missing {API_KEY_ENV} environment variable. "
                f'Set it with: export {API_KEY_ENV}="..."'
            )

    # --- Public interface -------------------------------------------------
    def classify(self, image: bytes) -> ClassificationResult:
        """Classify JPEG-encoded ``image``.

        Raises ``ClassificationError`` on any API/validation failure so the
        caller can retry. Raises ``ClassificationRejectedError`` when the user
        declines to send it (``gemini_debug`` mode). Returns a validated
        ``ClassificationResult`` on success.
        """
        if self.config.gemini_debug and not _preview_and_approve(
            image, self.config.captures_dir
        ):
            raise ClassificationRejectedError("image skipped by user")

        payload = self._build_payload(image)
        raw_body = self._post(payload)

        result = self._parse_response(raw_body)
        if self.config.gemini_debug:
            _show_response(result, raw_body.get("raw_text", ""), raw_body.get("usage"))
        return result

    # --- Internals --------------------------------------------------------
    def _build_payload(self, image: bytes) -> dict[str, Any]:
        b64 = base64.b64encode(image).decode("ascii")
        return {
            "contents": [
                {
                    "parts": [
                        {"text": PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": CLASSIFICATION_SCHEMA,
            },
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_HOST}/v1beta/models/{self.config.gemini_model}:generateContent"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key or "",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.gemini_timeout
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise self._map_http_error(e.code) from e
        except TimeoutError as e:
            raise ClassificationError("timeout", "Gemini request timed out") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ClassificationError(
                    "timeout", "Gemini request timed out"
                ) from e
            raise ClassificationError(
                "network", f"Gemini request failed: URLError: {e.reason}"
            ) from e
        except Exception as e:
            raise ClassificationError(
                "network", f"Gemini request failed: {type(e).__name__}: {e}"
            ) from e

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ClassificationError(
                "invalid_response", "Gemini returned non-JSON envelope"
            ) from e
        if not isinstance(envelope, dict):
            raise ClassificationError(
                "invalid_response", "Gemini response envelope is not a JSON object"
            )

        try:
            text = envelope["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ClassificationError(
                "invalid_response", "Gemini response has no text part"
            ) from e

        return {"raw_text": text, "usage": envelope.get("usageMetadata")}

    def _map_http_error(self, code: int | None) -> ClassificationError:
        if code in (400, 401, 403):
            return ClassificationError("auth", f"authentication error (HTTP {code})")
        if code == 429:
            return ClassificationError("rate_limit", "rate limit exceeded (HTTP 429)")
        if code is not None and 500 <= code < 600:
            return ClassificationError(
                "unavailable", f"Gemini server error (HTTP {code})"
            )
        return ClassificationError("http", f"Gemini HTTP error (code {code})")

    def _parse_response(self, body: dict[str, Any]) -> ClassificationResult:
        text = body.get("raw_text")
        if not text:
            raise ClassificationError("invalid_response", "empty response from Gemini")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ClassificationError(
                "invalid_response", "Gemini returned non-JSON"
            ) from e

        if not isinstance(data, dict):
            raise ClassificationError(
                "invalid_response", "Gemini response is not a JSON object"
            )

        category = data.get("category")
        if category not in ALLOWED_CATEGORIES:
            raise ClassificationError(
                "invalid_response", f"Gemini returned invalid category: {category!r}"
            )

        try:
            confidence = float(data.get("confidence"))
        except (TypeError, ValueError):
            raise ClassificationError(
                "invalid_response",
                f"Gemini returned invalid confidence: {data.get('confidence')!r}",
            )
        if not 0.0 <= confidence <= 1.0:
            raise ClassificationError(
                "invalid_response",
                f"Gemini returned out-of-range confidence: {confidence}",
            )

        return ClassificationResult(category=category, confidence=confidence)
