"""Speech-to-text abstraction over the cluster's STT backends.

Two real services live in the cluster with different protocols and quirks
(see the enrich spec §4): ``whisper-http`` (OpenAI-compatible) and ``subgen``
(plain FastAPI, SRT output). This package exposes a clean async abstraction
with one adapter each, plus an ``STTClient`` that can race both and pick the
fastest responder.
"""

from tunabrain.stt.client import (
    STTBackend,
    STTClient,
    STTResult,
    STTSegment,
    SubgenBackend,
    WhisperHTTPBackend,
    build_stt_client,
)

__all__ = [
    "STTBackend",
    "STTClient",
    "STTResult",
    "STTSegment",
    "SubgenBackend",
    "WhisperHTTPBackend",
    "build_stt_client",
]
