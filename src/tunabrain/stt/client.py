"""Pluggable async STT client with two real backends.

The two cluster backends have meaningfully different wire shapes:

* ``whisper-http`` (basnijholt agent-cli-whisper) — OpenAI-compatible
  ``POST /v1/audio/transcriptions`` (multipart), ``GET /health`` probe. Only the
  ``turbo`` model is registered in the current deployment, so the model name is
  configurable and defaults to ``turbo`` (never ``large-v3``).
* ``subgen`` (mccloud/subgen) — plain FastAPI ``POST /asr`` (multipart),
  ``GET /status`` probe. Its ``output=json`` and ``word_timestamps`` params are
  broken in the current deployment, so we always request ``output=srt`` and
  parse the SRT ourselves.

Rather than a lowest-common-denominator client, each backend is a thin adapter
implementing :class:`STTBackend`. :class:`STTClient` selects one per request; in
``auto`` mode it probes both and uses whichever health endpoint responds first,
falling back to the other backend if the winner's transcription fails.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BackendName = Literal["whisper-http", "subgen", "auto"]

# Default probe timeout for 'auto' backend selection (spec §4.3: 2s probe).
_PROBE_TIMEOUT_SECONDS = 2.0


class STTSegment(BaseModel):
    """A single timed transcript segment."""

    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class STTResult(BaseModel):
    """The result of a transcription, normalised across backends."""

    text: str = Field(..., description="The full transcript as one string")
    segments: list[STTSegment] = Field(
        default_factory=list, description="Timed segments when the backend provides them"
    )
    language: str | None = Field(None, description="Detected/declared language (ISO-639-1)")
    duration_seconds: float = Field(0.0, description="Media duration in seconds, when known")


@runtime_checkable
class STTBackend(Protocol):
    """A single STT backend adapter."""

    name: str

    async def probe(self, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
        """Return True if the backend's health endpoint responds OK."""
        ...

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        timeout: float = 600.0,
    ) -> STTResult:
        """Transcribe ``audio`` bytes to an :class:`STTResult`."""
        ...


def _parse_srt(srt: str) -> list[STTSegment]:
    """Parse an SRT document into timed segments.

    Blocks are separated by blank lines; each block is an index line, a
    ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` timing line, and one or more text lines.
    Malformed blocks are skipped rather than raising.
    """
    segments: list[STTSegment] = []
    timing = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    for block in re.split(r"\n\s*\n", srt.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        match = None
        text_start = 0
        for idx, line in enumerate(lines):
            m = timing.search(line)
            if m:
                match = m
                text_start = idx + 1
                break
        if not match:
            continue
        sh, sm, ss, sms, eh, em, es, ems = (int(g) for g in match.groups())
        start = sh * 3600 + sm * 60 + ss + sms / 1000.0
        end = eh * 3600 + em * 60 + es + ems / 1000.0
        text = " ".join(lines[text_start:]).strip()
        if text:
            segments.append(STTSegment(start=start, end=end, text=text))
    return segments


class WhisperHTTPBackend:
    """OpenAI-compatible whisper-http adapter."""

    name = "whisper-http"

    def __init__(self, base_url: str, *, model: str = "turbo") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def probe(self, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug("whisper-http probe failed: %s", exc)
            return False

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        timeout: float = 600.0,
    ) -> STTResult:
        # verbose_json gives us segments + duration in one shot; we discard the
        # model internals (logprobs, token ids) and keep only text/timing.
        data: dict[str, str] = {"model": self._model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
        files = {"file": ("audio.wav", audio, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/transcriptions", data=data, files=files
            )
            resp.raise_for_status()
            payload = resp.json()
        segments = [
            STTSegment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)),
                       text=str(s.get("text", "")).strip())
            for s in payload.get("segments", [])
            if str(s.get("text", "")).strip()
        ]
        return STTResult(
            text=str(payload.get("text", "")).strip(),
            segments=segments,
            language=payload.get("language"),
            duration_seconds=float(payload.get("duration", 0.0) or 0.0),
        )


class SubgenBackend:
    """mccloud/subgen adapter. Always requests SRT and parses it locally."""

    name = "subgen"

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def probe(self, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self._base_url}/status")
                return resp.status_code == 200
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug("subgen probe failed: %s", exc)
            return False

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        timeout: float = 600.0,
    ) -> STTResult:
        # output=json and word_timestamps are broken in the current deployment,
        # so we always ask for srt and parse it ourselves (spec §4.2).
        data: dict[str, str] = {"task": "transcribe", "output": "srt"}
        if language:
            data["language"] = language
        files = {"audio_file": ("audio.wav", audio, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self._base_url}/asr", data=data, files=files)
            resp.raise_for_status()
            srt = resp.text
        segments = _parse_srt(srt)
        text = " ".join(seg.text for seg in segments).strip()
        duration = segments[-1].end if segments else 0.0
        return STTResult(
            text=text, segments=segments, language=language, duration_seconds=duration
        )


class STTClient:
    """Pluggable STT client. Picks a backend per request (or via env default)."""

    def __init__(
        self,
        whisper_url: str,
        subgen_url: str,
        default: str = "auto",
        *,
        whisper_model: str = "turbo",
        backends: dict[str, STTBackend] | None = None,
        probe_timeout: float = _PROBE_TIMEOUT_SECONDS,
    ) -> None:
        self.backends: dict[str, STTBackend] = backends or {
            "whisper-http": WhisperHTTPBackend(whisper_url, model=whisper_model),
            "subgen": SubgenBackend(subgen_url),
        }
        self.default = default
        self.probe_timeout = probe_timeout

    async def _race_probe(self) -> str:
        """Return the name of the backend whose health probe responds first.

        Falls back to the first configured backend if none respond.
        """
        names = list(self.backends)

        async def _probe(name: str) -> str:
            ok = await self.backends[name].probe(timeout=self.probe_timeout)
            if not ok:
                # Never "win" the race on a failed probe: block until cancelled.
                await asyncio.sleep(self.probe_timeout + 1)
            return name

        tasks = [asyncio.create_task(_probe(name)) for name in names]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=self.probe_timeout + 1
            )
            for task in pending:
                task.cancel()
            for task in done:
                if not task.cancelled() and task.exception() is None:
                    return task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
        logger.warning("No STT backend responded to probe; defaulting to %s", names[0])
        return names[0]

    async def transcribe(
        self,
        audio: bytes,
        *,
        backend: str | None = None,
        language: str | None = None,
        timeout: float = 600.0,
    ) -> tuple[STTResult, str]:
        """Transcribe ``audio``; return ``(result, backend_used)``.

        For an explicit backend, that backend is used directly. For ``auto`` (the
        default), both backends are probed and the fastest responder transcribes;
        if it fails, the other backend is tried before giving up.
        """
        choice = backend or self.default
        if choice != "auto":
            if choice not in self.backends:
                raise ValueError(f"Unknown STT backend: {choice!r}")
            adapter = self.backends[choice]
            return await adapter.transcribe(audio, language=language, timeout=timeout), adapter.name

        winner = await self._race_probe()
        order = [winner] + [name for name in self.backends if name != winner]
        last_exc: Exception | None = None
        for name in order:
            try:
                result = await self.backends[name].transcribe(
                    audio, language=language, timeout=timeout
                )
                return result, name
            except Exception as exc:
                logger.warning("STT backend %s failed (%s); trying next", name, exc)
                last_exc = exc
        raise RuntimeError(f"All STT backends failed; last error: {last_exc}")


def build_stt_client() -> STTClient:
    """Construct an :class:`STTClient` from environment-driven settings."""
    from tunabrain.config import get_settings

    settings = get_settings()
    return STTClient(
        whisper_url=settings.stt_whisper_url,
        subgen_url=settings.stt_subgen_url,
        default=settings.stt_default_backend,
        whisper_model=settings.stt_whisper_model,
    )
