"""Unit tests for the /enrich/long-form pipeline and the STT abstraction.

The pipeline is exercised through the module-level seams
(``fetch_media`` / ``_download_url`` / ``extract_audio`` / ``extract_keyframes``
/ ``caption_keyframes`` / ``categorize_media`` / ``generate_tags``) so no real
network, ffmpeg, or GPU is needed. STT backend selection is tested directly
against ``STTClient`` with stub backends.
"""

from __future__ import annotations

import asyncio

import pytest

from tunabrain.api.models import (
    CategoryDefinition,
    DimensionSelection,
    EnrichLongFormOptions,
    EnrichLongFormRequest,
    MediaContext,
    MediaItem,
    MediaSource,
)
from tunabrain.chains import enrich_long
from tunabrain.chains.categorization import CategorizationResult
from tunabrain.stt.client import STTClient, STTResult, _parse_srt

# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #


def _media(duration_minutes: int = 20) -> MediaItem:
    return MediaItem(id="grout-long-1", title="video-essay-abc", duration_minutes=duration_minutes)


def _request(**opt_kwargs) -> EnrichLongFormRequest:
    return EnrichLongFormRequest(
        media=_media(),
        source=MediaSource(url="https://example.com/video.mp4"),
        categories={
            "audience": CategoryDefinition(description="Time-of-day", values=["daytime"]),
        },
        options=EnrichLongFormOptions(**opt_kwargs),
    )


@pytest.fixture
def stub_llm_stages(monkeypatch):
    """Stub categorize + tags so the pipeline never calls a real LLM."""
    seen: dict[str, object] = {}

    async def fake_categorize(*, media, categories, channels, debug, context):
        seen["categorize_context"] = context
        return CategorizationResult(
            dimensions=[DimensionSelection(dimension="audience", values=["daytime"])],
            channel_mappings=[],
            context=context or MediaContext(source="none"),
        )

    async def fake_generate_tags(media, existing_tags=None, *, debug=False, context=None):
        seen["tags_context"] = context
        return ["documentary", "long-form"], context or MediaContext(source="none")

    monkeypatch.setattr(enrich_long, "categorize_media", fake_categorize)
    monkeypatch.setattr(enrich_long, "generate_tags", fake_generate_tags)
    return seen


def _stub_media_io(monkeypatch, tmp_path, *, transcript="hello world transcript", captions=None):
    """Stub fetch/audio/keyframes so the pipeline has fake media artifacts."""
    captions = captions if captions is not None else ["a person speaking to camera"]

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-video")
    audio = tmp_path / "video.wav"
    audio.write_bytes(b"fake-audio")

    async def fake_fetch(source, scratch_dir):
        return video

    async def fake_extract_audio(path, **kwargs):
        return audio

    async def fake_extract_keyframes(path, count, **kwargs):
        return [tmp_path / f"frame-{i}.jpg" for i in range(count)]

    async def fake_caption(paths, **kwargs):
        return list(captions)

    class StubSTT:
        def __init__(self):
            self.calls = []

        async def transcribe(self, audio_bytes, *, backend=None, language=None, timeout=600.0):
            self.calls.append({"backend": backend, "timeout": timeout})
            return STTResult(text=transcript, duration_seconds=1200.0), backend or "whisper-http"

    monkeypatch.setattr(enrich_long, "fetch_media", fake_fetch)
    monkeypatch.setattr(enrich_long, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(enrich_long, "extract_keyframes", fake_extract_keyframes)
    monkeypatch.setattr(enrich_long, "caption_keyframes", fake_caption)
    return StubSTT()


def _stage(resp, name):
    return next((s for s in resp.pipeline_stages if s.stage == name), None)


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_enrich_long_form_fetches_from_url(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    async def fake_download(url, dest):
        recorded["url"] = url
        recorded["dest"] = dest
        dest.write_bytes(b"downloaded")

    monkeypatch.setattr(enrich_long, "_download_url", fake_download)

    source = MediaSource(url="https://example.com/path/clip.mp4")
    path = await enrich_long.fetch_media(source, tmp_path)

    assert recorded["url"] == "https://example.com/path/clip.mp4"
    assert path == tmp_path / "clip.mp4"
    assert path.read_bytes() == b"downloaded"


@pytest.mark.anyio
async def test_enrich_long_form_fetches_from_file_id(tmp_path):
    staged = tmp_path / "staged-media-id"
    staged.write_bytes(b"already here")

    source = MediaSource(file_id="staged-media-id")
    path = await enrich_long.fetch_media(source, tmp_path)

    assert path == staged
    assert path.read_bytes() == b"already here"


@pytest.mark.anyio
async def test_enrich_long_form_missing_file_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await enrich_long.fetch_media(MediaSource(file_id="nope"), tmp_path)


# --------------------------------------------------------------------------- #
# audio extraction (ffmpeg args)
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_enrich_long_form_extracts_audio(monkeypatch, tmp_path):
    from tunabrain.stt import audio as audio_module

    recorded: dict[str, object] = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(audio_module.asyncio, "create_subprocess_exec", fake_exec)

    video = tmp_path / "in.mp4"
    out = await audio_module.extract_audio(video)

    cmd = recorded["cmd"]
    assert cmd[0] == "ffmpeg"
    # Strip video, force mono 16kHz PCM WAV.
    assert "-vn" in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "pcm_s16le" in cmd
    assert str(video) in cmd
    assert out == video.with_suffix(".wav")


# --------------------------------------------------------------------------- #
# STT backend selection (STTClient)
# --------------------------------------------------------------------------- #


class _FakeBackend:
    def __init__(self, name, *, probe_ok=True, probe_delay=0.0, result_text="txt", fail=False):
        self.name = name
        self._probe_ok = probe_ok
        self._probe_delay = probe_delay
        self._result_text = result_text
        self._fail = fail
        self.transcribe_calls = 0

    async def probe(self, *, timeout=2.0):
        if self._probe_delay:
            await asyncio.sleep(self._probe_delay)
        return self._probe_ok

    async def transcribe(self, audio, *, language=None, timeout=600.0):
        self.transcribe_calls += 1
        if self._fail:
            raise RuntimeError(f"{self.name} failed")
        return STTResult(text=self._result_text, duration_seconds=1.0)


def _client_with(backends):
    client = STTClient("http://w", "http://s", default="auto")
    client.backends = {b.name: b for b in backends}
    client.probe_timeout = 0.2
    return client


@pytest.mark.anyio
async def test_enrich_long_form_uses_whisper_http_when_specified():
    whisper = _FakeBackend("whisper-http", result_text="from whisper")
    subgen = _FakeBackend("subgen", result_text="from subgen")
    client = _client_with([whisper, subgen])

    result, used = await client.transcribe(b"audio", backend="whisper-http")

    assert used == "whisper-http"
    assert result.text == "from whisper"
    assert whisper.transcribe_calls == 1
    assert subgen.transcribe_calls == 0


@pytest.mark.anyio
async def test_enrich_long_form_uses_subgen_when_specified():
    whisper = _FakeBackend("whisper-http")
    subgen = _FakeBackend("subgen", result_text="from subgen")
    client = _client_with([whisper, subgen])

    result, used = await client.transcribe(b"audio", backend="subgen")

    assert used == "subgen"
    assert result.text == "from subgen"
    assert subgen.transcribe_calls == 1
    assert whisper.transcribe_calls == 0


@pytest.mark.anyio
async def test_enrich_long_form_auto_picks_responded_first():
    # subgen probes instantly; whisper is slow -> subgen wins the race.
    whisper = _FakeBackend("whisper-http", probe_delay=0.2, result_text="from whisper")
    subgen = _FakeBackend("subgen", probe_delay=0.0, result_text="from subgen")
    client = _client_with([whisper, subgen])

    result, used = await client.transcribe(b"audio", backend="auto")

    assert used == "subgen"
    assert result.text == "from subgen"


@pytest.mark.anyio
async def test_enrich_long_form_auto_falls_back_when_winner_fails():
    # whisper wins the probe but fails to transcribe; subgen is the fallback.
    whisper = _FakeBackend("whisper-http", probe_delay=0.0, fail=True)
    subgen = _FakeBackend("subgen", probe_delay=0.1, result_text="from subgen")
    client = _client_with([whisper, subgen])

    result, used = await client.transcribe(b"audio", backend="auto")

    assert used == "subgen"
    assert result.text == "from subgen"


def test_subgen_srt_parsing():
    srt = (
        "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n"
        "2\n00:00:02,500 --> 00:00:05,000\nGeneral Kenobi.\n"
    )
    segments = _parse_srt(srt)
    assert [s.text for s in segments] == ["Hello there.", "General Kenobi."]
    assert segments[0].start == 0.0
    assert segments[1].end == 5.0


# --------------------------------------------------------------------------- #
# pipeline: STT skip / keyframes / context assembly
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_enrich_long_form_skips_stt_for_short_media(monkeypatch, tmp_path, stub_llm_stages):
    stub = _stub_media_io(monkeypatch, tmp_path, captions=[])
    # 1-minute media (60s) below a 120s threshold -> STT (and audio) skipped.
    req = EnrichLongFormRequest(
        media=MediaItem(id="x", title="short-clip", duration_minutes=1),
        source=MediaSource(url="https://example.com/v.mp4"),
        categories={"audience": CategoryDefinition(description="d", values=["daytime"])},
        options=EnrichLongFormOptions(skip_stt_below_seconds=120, enable_keyframe_analysis=False),
    )
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    assert _stage(resp, "extract_audio").status == "skipped"
    assert _stage(resp, "stt").status == "skipped"
    assert resp.transcript == ""
    # The STT client was never invoked.
    assert stub.calls == []
    # categorize still runs on filename-only grounding.
    assert [d.dimension for d in resp.dimensions] == ["audience"]


@pytest.mark.anyio
async def test_enrich_long_form_runs_stt_for_long_media(monkeypatch, tmp_path, stub_llm_stages):
    stub = _stub_media_io(monkeypatch, tmp_path, transcript="a full transcript", captions=[])
    req = _request(enable_keyframe_analysis=False)  # 20-minute media
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    assert _stage(resp, "stt").status == "success"
    assert resp.transcript == "a full transcript"
    assert len(stub.calls) == 1


@pytest.mark.anyio
async def test_enrich_long_form_extracts_keyframes_when_enabled(monkeypatch, tmp_path, stub_llm_stages):
    stub = _stub_media_io(monkeypatch, tmp_path, captions=["scene one", "scene two"])
    req = _request(enable_keyframe_analysis=True, keyframe_count=2)
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    assert resp.keyframe_captions == ["scene one", "scene two"]
    assert _stage(resp, "keyframes").status == "success"


@pytest.mark.anyio
async def test_enrich_long_form_skips_keyframes_when_disabled(monkeypatch, tmp_path, stub_llm_stages):
    stub = _stub_media_io(monkeypatch, tmp_path)
    req = _request(enable_keyframe_analysis=False)
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    assert resp.keyframe_captions == []
    assert _stage(resp, "keyframes").status == "skipped"


@pytest.mark.anyio
async def test_enrich_long_form_assembles_context_from_transcript_and_captions(
    monkeypatch, tmp_path, stub_llm_stages
):
    stub = _stub_media_io(
        monkeypatch, tmp_path, transcript="the moon landing", captions=["archival footage"]
    )
    req = _request()
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    ctx = stub_llm_stages["categorize_context"]
    assert ctx is not None
    assert "the moon landing" in ctx.summary
    assert "archival footage" in ctx.summary
    # Full transcript is echoed on the top-level field.
    assert resp.transcript == "the moon landing"


@pytest.mark.anyio
async def test_enrich_long_form_propagates_context_to_categorize_and_tags(
    monkeypatch, tmp_path, stub_llm_stages
):
    stub = _stub_media_io(monkeypatch, tmp_path, transcript="grounding text", captions=[])
    req = _request(enable_keyframe_analysis=False)
    await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    cat_ctx = stub_llm_stages["categorize_context"]
    tags_ctx = stub_llm_stages["tags_context"]
    assert "grounding text" in cat_ctx.summary
    # tags receives the context echoed by categorize (same grounding).
    assert tags_ctx is not None and "grounding text" in tags_ctx.summary


@pytest.mark.anyio
async def test_enrich_long_form_max_transcript_chars_caps_llm_context(
    monkeypatch, tmp_path, stub_llm_stages
):
    long_transcript = "x" * 20000
    stub = _stub_media_io(monkeypatch, tmp_path, transcript=long_transcript, captions=[])
    req = _request(enable_keyframe_analysis=False, max_transcript_chars=100)
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    ctx = stub_llm_stages["categorize_context"]
    # Sent-to-LLM summary is capped...
    assert len(ctx.summary) <= 100 + len("Transcript:\n")
    # ...but the full transcript is preserved on the response.
    assert resp.transcript == long_transcript


# --------------------------------------------------------------------------- #
# graceful degradation
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_enrich_long_form_handles_stt_failure_gracefully(monkeypatch, tmp_path, stub_llm_stages):
    _stub_media_io(monkeypatch, tmp_path, captions=[])

    class FailingSTT:
        async def transcribe(self, *a, **k):
            raise RuntimeError("stt down")

    req = _request(enable_keyframe_analysis=False)
    resp = await enrich_long.run_enrich_long_form(
        req, stt_client=FailingSTT(), hard_cap_seconds=10
    )

    assert any("stt failed" in w for w in resp.warnings)
    assert _stage(resp, "stt").status == "failed"
    # categorize still ran (filename-only grounding).
    assert [d.dimension for d in resp.dimensions] == ["audience"]


@pytest.mark.anyio
async def test_enrich_long_form_handles_keyframe_failure_gracefully(
    monkeypatch, tmp_path, stub_llm_stages
):
    stub = _stub_media_io(monkeypatch, tmp_path, transcript="transcript kept")

    async def boom_keyframes(path, count, **kwargs):
        raise RuntimeError("ffmpeg keyframe boom")

    monkeypatch.setattr(enrich_long, "extract_keyframes", boom_keyframes)

    req = _request()
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    assert any("keyframes failed" in w for w in resp.warnings)
    assert _stage(resp, "keyframes").status == "failed"
    # transcript-only context still used.
    assert resp.transcript == "transcript kept"


@pytest.mark.anyio
async def test_enrich_long_form_handles_fetch_failure_gracefully(
    monkeypatch, tmp_path, stub_llm_stages
):
    async def boom_fetch(source, scratch_dir):
        raise RuntimeError("404 not found")

    monkeypatch.setattr(enrich_long, "fetch_media", boom_fetch)

    req = _request()
    resp = await enrich_long.run_enrich_long_form(req, hard_cap_seconds=10)

    assert any("fetch failed" in w for w in resp.warnings)
    assert _stage(resp, "fetch").status == "failed"
    # Downstream media stages are skipped, but categorize/tags still run.
    assert _stage(resp, "extract_audio").status == "skipped"
    assert _stage(resp, "stt").status == "skipped"
    assert _stage(resp, "keyframes").status == "skipped"
    assert [d.dimension for d in resp.dimensions] == ["audience"]


@pytest.mark.anyio
async def test_enrich_long_form_handles_backend_timeout(monkeypatch, tmp_path, stub_llm_stages):
    _stub_media_io(monkeypatch, tmp_path, captions=[])

    class TimingOutSTT:
        async def transcribe(self, *a, **k):
            raise asyncio.TimeoutError()

    req = _request(enable_keyframe_analysis=False)
    resp = await enrich_long.run_enrich_long_form(
        req, stt_client=TimingOutSTT(), hard_cap_seconds=10
    )

    assert _stage(resp, "stt").status == "failed"
    assert any("stt failed" in w for w in resp.warnings)


@pytest.mark.anyio
async def test_enrich_long_form_returns_pipeline_stages(monkeypatch, tmp_path, stub_llm_stages):
    stub = _stub_media_io(monkeypatch, tmp_path)
    req = _request()
    resp = await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    reported = {s.stage for s in resp.pipeline_stages}
    assert reported == {"fetch", "extract_audio", "stt", "keyframes", "categorize", "tags"}
    for stage in resp.pipeline_stages:
        assert stage.status in {"success", "skipped", "warning", "failed"}
        assert stage.duration_seconds >= 0


@pytest.mark.anyio
async def test_enrich_long_form_enforces_timeout(monkeypatch, tmp_path, stub_llm_stages):
    async def slow_fetch(source, scratch_dir):
        await asyncio.sleep(5)
        return tmp_path / "video.mp4"

    monkeypatch.setattr(enrich_long, "fetch_media", slow_fetch)

    req = _request()
    resp = await enrich_long.run_enrich_long_form(req, hard_cap_seconds=0.1)

    # The pipeline aborted at the cap and returned a degraded response.
    assert any("exceeded hard timeout" in w for w in resp.warnings)
    assert resp.dimensions == []
    assert resp.transcript == ""


@pytest.mark.anyio
async def test_enrich_long_form_selects_backend_from_options(monkeypatch, tmp_path, stub_llm_stages):
    stub = _stub_media_io(monkeypatch, tmp_path)
    req = _request(stt_backend="subgen")
    await enrich_long.run_enrich_long_form(req, stt_client=stub, hard_cap_seconds=10)

    assert stub.calls[0]["backend"] == "subgen"
