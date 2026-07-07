"""Long-form enrichment: fetch -> audio -> STT -> keyframes -> categorize -> tags.

Grout's long-form media (documentaries, video essays, interviews, debates)
carries no reliable external metadata, so we synthesise grounding from the media
itself: transcribe the audio (via the pluggable STT client) and optionally
caption a handful of keyframes, then feed the combined summary into the existing
/categorize and /tags building blocks.

Every stage degrades gracefully — a failed or skipped stage is recorded as a
:class:`PipelineStageResult` and the pipeline continues with whatever grounding
it managed to gather (down to filename-only). The whole pipeline is bounded by a
hard timeout so a hung backend can never wedge a request indefinitely.

The module-level ``fetch_media`` / ``_download_url`` / ``extract_audio`` /
``extract_keyframes`` / ``caption_keyframes`` names are deliberate seams: they
are looked up on this module at call time so callers (and tests) can substitute
them without real network, ffmpeg, or GPU access.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import httpx

from tunabrain.api.models import (
    CostEstimate,
    EnrichLongFormRequest,
    EnrichLongFormResponse,
    MediaContext,
    MediaSource,
    PipelineStageResult,
)
from tunabrain.chains.categorization import categorize_media
from tunabrain.chains.tagging import generate_tags
from tunabrain.config import get_settings, is_debug_enabled
from tunabrain.keyframes.caption import caption_keyframes
from tunabrain.scheduling.cost import calculate_cost
from tunabrain.stt.audio import extract_audio
from tunabrain.stt.client import build_stt_client
from tunabrain.stt.keyframes import extract_keyframes

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT_SECONDS = 300.0


def _estimate_cost(llm_calls: int) -> CostEstimate:
    """Rough CostEstimate for a pipeline that made ``llm_calls`` LLM calls."""
    model = get_settings().llm_model
    prompt_tokens = 1500 * llm_calls
    completion_tokens = 500 * llm_calls
    cost_usd = calculate_cost(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return CostEstimate(
        estimated_cost_usd=cost_usd,
        llm_calls_used=llm_calls,
        estimated_tokens=f"~{prompt_tokens + completion_tokens:,}",
        model=model,
    )


def _stage(
    name: str,
    status: str,
    started: float,
    *,
    backend: str | None = None,
    detail: str | None = None,
) -> PipelineStageResult:
    return PipelineStageResult(
        stage=name,
        status=status,
        duration_seconds=round(perf_counter() - started, 3),
        backend=backend,
        detail=detail,
    )


async def _download_url(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``. Isolated so tests can stub the network."""
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes():
                    fh.write(chunk)


async def fetch_media(source: MediaSource, scratch_dir: Path) -> Path:
    """Resolve the media bytes to a local path in ``scratch_dir``.

    For ``file_id`` the file is expected to already be staged in the scratch
    space. For ``url`` it is downloaded there.
    """
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if source.file_id:
        path = scratch_dir / source.file_id
        if not path.exists():
            raise FileNotFoundError(f"staged media file not found: {path}")
        return path

    # URL path (MediaSource guarantees exactly one of url/file_id is set).
    name = Path(urlparse(source.url).path).name or "download"
    dest = scratch_dir / name
    await _download_url(source.url, dest)
    return dest


def _assemble_context(
    transcript: str, captions: list[str], max_transcript_chars: int
) -> MediaContext | None:
    """Combine transcript (capped) + keyframe captions into a grounding context.

    The capped text is what gets sent to (and echoed by) the LLM as
    ``context.summary`` — resolving to ``source='provided-summary'``. The full,
    untruncated transcript is returned separately on the response's top-level
    ``transcript`` field.
    """
    parts: list[str] = []
    if transcript:
        capped = transcript[:max_transcript_chars]
        parts.append(f"Transcript:\n{capped}")
    if captions:
        joined = "\n".join(f"- {c}" for c in captions)
        parts.append(f"Keyframe captions:\n{joined}")
    if not parts:
        return None
    return MediaContext(summary="\n\n".join(parts))


async def _run_pipeline(request: EnrichLongFormRequest, stt_client) -> EnrichLongFormResponse:
    opts = request.options
    debug = is_debug_enabled(request.debug)
    scratch_dir = Path(get_settings().scratch_dir)

    stages: list[PipelineStageResult] = []
    warnings: list[str] = []
    transcript = ""
    keyframe_captions: list[str] = []
    media_path: Path | None = None

    duration_seconds = (request.media.duration_minutes or 0) * 60

    # --- fetch ---
    started = perf_counter()
    try:
        media_path = await fetch_media(request.source, scratch_dir)
        stages.append(_stage("fetch", "success", started, detail=str(media_path)))
    except Exception as exc:
        logger.warning("Enrich long-form fetch failed: %s", exc)
        warnings.append(f"fetch failed: {exc}")
        stages.append(_stage("fetch", "failed", started, detail=str(exc)))

    # --- extract audio ---
    audio_bytes: bytes | None = None
    skip_stt = bool(duration_seconds) and duration_seconds < opts.skip_stt_below_seconds
    started = perf_counter()
    if media_path is None:
        stages.append(_stage("extract_audio", "skipped", started, detail="no media available"))
    elif skip_stt:
        stages.append(
            _stage("extract_audio", "skipped", started, detail="duration below STT threshold")
        )
    else:
        try:
            audio_path = await extract_audio(media_path)
            audio_bytes = Path(audio_path).read_bytes()
            stages.append(_stage("extract_audio", "success", started))
        except Exception as exc:
            logger.warning("Enrich long-form audio extraction failed: %s", exc)
            warnings.append(f"audio extraction failed: {exc}")
            stages.append(_stage("extract_audio", "failed", started, detail=str(exc)))

    # --- stt ---
    started = perf_counter()
    if audio_bytes is None:
        detail = "duration below STT threshold" if skip_stt else "no audio available"
        stages.append(_stage("stt", "skipped", started, detail=detail))
    else:
        client = stt_client or build_stt_client()
        try:
            result, backend = await client.transcribe(
                audio_bytes,
                backend=opts.stt_backend,
                timeout=float(opts.stt_timeout_seconds),
            )
            transcript = result.text
            stages.append(_stage("stt", "success", started, backend=backend))
        except Exception as exc:
            logger.warning("Enrich long-form STT failed: %s", exc)
            warnings.append(f"stt failed: {exc}")
            stages.append(_stage("stt", "failed", started, detail=str(exc)))

    # --- keyframes ---
    started = perf_counter()
    if media_path is None:
        stages.append(_stage("keyframes", "skipped", started, detail="no media available"))
    elif not opts.enable_keyframe_analysis:
        stages.append(_stage("keyframes", "skipped", started, detail="disabled by request"))
    else:
        try:
            frames = await extract_keyframes(
                media_path,
                opts.keyframe_count,
                duration_seconds=duration_seconds or None,
            )
            keyframe_captions = await caption_keyframes(frames)
            if keyframe_captions:
                stages.append(_stage("keyframes", "success", started))
            else:
                warnings.append("keyframe captioning produced no captions")
                stages.append(
                    _stage("keyframes", "warning", started, detail="no captions produced")
                )
        except Exception as exc:
            logger.warning("Enrich long-form keyframe analysis failed: %s", exc)
            warnings.append(f"keyframes failed: {exc}")
            stages.append(_stage("keyframes", "failed", started, detail=str(exc)))

    # --- assemble grounding context ---
    context = _assemble_context(transcript, keyframe_captions, opts.max_transcript_chars)

    # --- categorize ---
    started = perf_counter()
    dimensions = []
    resolved_context: MediaContext | None = context
    categorize_calls = 0
    try:
        categorization = await categorize_media(
            media=request.media,
            categories=request.categories,
            channels=request.channels,
            debug=debug,
            context=context,
        )
        dimensions = categorization.dimensions
        resolved_context = categorization.context
        categorize_calls = len(request.categories) + (1 if request.channels else 0)
        stages.append(_stage("categorize", "success", started))
    except Exception as exc:  # pragma: no cover - categorize is robust internally
        logger.warning("Enrich long-form categorize failed: %s", exc)
        warnings.append(f"categorize failed: {exc}")
        stages.append(_stage("categorize", "failed", started, detail=str(exc)))

    # --- tags ---
    started = perf_counter()
    tags: list[str] = []
    tags_calls = 0
    try:
        tags, tag_context = await generate_tags(
            request.media,
            request.existing_tags,
            debug=debug,
            context=resolved_context,
        )
        resolved_context = tag_context
        tags_calls = 1
        stages.append(_stage("tags", "success", started))
    except Exception as exc:  # pragma: no cover - tagging is robust internally
        logger.warning("Enrich long-form tags failed: %s", exc)
        warnings.append(f"tags failed: {exc}")
        stages.append(_stage("tags", "failed", started, detail=str(exc)))

    llm_calls = max(1, categorize_calls + tags_calls + len(keyframe_captions))
    return EnrichLongFormResponse(
        media=request.media,
        dimensions=dimensions,
        tags=tags,
        transcript=transcript,
        keyframe_captions=keyframe_captions,
        context=resolved_context,
        pipeline_stages=stages,
        cost_estimate=_estimate_cost(llm_calls),
        warnings=warnings,
    )


async def run_enrich_long_form(
    request: EnrichLongFormRequest,
    *,
    stt_client=None,
    hard_cap_seconds: float | None = None,
) -> EnrichLongFormResponse:
    """Run the long-form enrichment pipeline under a hard timeout.

    ``stt_client`` and ``hard_cap_seconds`` are injectable for testing; the route
    passes neither and relies on the env-derived defaults.
    """
    hard_cap = hard_cap_seconds if hard_cap_seconds is not None else get_settings().enrich_long_timeout
    logger.info(
        "Enrich long-form for title='%s' (hard cap %ss, stt=%s)",
        request.media.title,
        hard_cap,
        request.options.stt_backend,
    )
    try:
        return await asyncio.wait_for(_run_pipeline(request, stt_client), timeout=hard_cap)
    except asyncio.TimeoutError:
        logger.warning("Enrich long-form pipeline exceeded hard cap of %ss", hard_cap)
        return EnrichLongFormResponse(
            media=request.media,
            transcript="",
            cost_estimate=_estimate_cost(0),
            warnings=[f"pipeline aborted: exceeded hard timeout of {hard_cap}s"],
            pipeline_stages=[],
        )
