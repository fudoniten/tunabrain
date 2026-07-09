"""Pydantic contract tests for the duration-aware scheduling additions to
tunabrain.scheduling.grid (Phase 2 of DURATION_AWARE_SCHEDULING.md,
tunarr-scheduler).
"""

from __future__ import annotations

from tunabrain.scheduling.grid import (
    CatalogProfile,
    RuntimeBucket,
    TagAggregate,
    TagRuntimeHistogram,
)


def test_runtime_bucket_accepts_null_max_minutes_for_open_ended_top_bucket():
    """Pseudovision's histogram always emits an open-ended top bucket (the old
    '90+min', now '210+min') with max_minutes = null. Before this fix,
    RuntimeBucket.max_minutes was a non-nullable `int`, so any real
    CatalogProfile — which always has *some* content in the open bucket —
    would fail pydantic validation the moment it reached Tunabrain's
    /propose-quarterly-grid request body. tunarr-scheduler's own contracts.clj
    had already special-cased this as nullable; this brings the upstream
    "authoritative" contract back in sync with it.
    """
    bucket = RuntimeBucket(label="210+min", min_minutes=210, max_minutes=None, item_count=3)
    assert bucket.max_minutes is None
    # And the closed-bucket case still works as before.
    closed = RuntimeBucket(label="90-105min", min_minutes=90, max_minutes=105, item_count=12)
    assert closed.max_minutes == 105


def test_tag_aggregate_round_trips():
    agg = TagAggregate(tag="genre:comedy", show_count=5, episode_count=120)
    assert agg.model_dump() == {
        "tag": "genre:comedy",
        "show_count": 5,
        "episode_count": 120,
    }


def test_tag_runtime_histogram_round_trips_nested_buckets():
    histo = TagRuntimeHistogram(
        tag="genre:movie",
        buckets=[
            RuntimeBucket(label="90-105min", min_minutes=90, max_minutes=105, item_count=12),
            RuntimeBucket(label="210+min", min_minutes=210, max_minutes=None, item_count=1),
        ],
    )
    dumped = histo.model_dump()
    assert dumped["tag"] == "genre:movie"
    assert len(dumped["buckets"]) == 2
    # _WireModel drops None values — the second bucket's max_minutes should be
    # entirely absent from the serialized form, not present as `null`.
    assert "max_minutes" not in dumped["buckets"][1]
    assert dumped["buckets"][0]["max_minutes"] == 105


def test_catalog_profile_carries_tag_aggregates_and_tag_runtime_histograms():
    """Both fields are new; a profile that omits them entirely (e.g. an older
    Pseudovision build) should still validate via the default empty list."""
    minimal = CatalogProfile(total_items=0, total_episodes=0)
    assert minimal.tag_aggregates == []
    assert minimal.tag_runtime_histograms == []

    full = CatalogProfile(
        total_items=10,
        total_episodes=8,
        movie_count=2,
        tag_aggregates=[TagAggregate(tag="channel:goldenreels", show_count=3, episode_count=40)],
        tag_runtime_histograms=[
            TagRuntimeHistogram(
                tag="genre:movie",
                buckets=[
                    RuntimeBucket(
                        label="90-105min", min_minutes=90, max_minutes=105, item_count=2
                    )
                ],
            )
        ],
    )
    assert full.tag_aggregates[0].tag == "channel:goldenreels"
    assert full.tag_runtime_histograms[0].buckets[0].item_count == 2
