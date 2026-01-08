from fastapi.testclient import TestClient

from tunabrain.app import create_app
import tunabrain.api.routes as routes
from tunabrain.api.models import TagAuditResult, TagDecision


def test_tag_governance_triage_endpoint(monkeypatch):
    client = TestClient(create_app())

    async def fake_triage(tags, *, target_limit=None, debug=False):  # pragma: no cover - simple stub
        return [
            TagDecision(
                tag="vampire_bat",
                action="merge",
                replacement="vampires",
                rationale="Too narrow; merge into broader vampire programming",
            )
        ]

    monkeypatch.setattr(routes, "triage_tags", fake_triage)

    payload = {
        "tags": [
            {"tag": "vampire_bat", "usage_count": 2, "example_titles": ["Bat Movie"]}
        ],
        "target_limit": 250,
    }

    response = client.post("/tag-governance/triage", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "decisions": [
            {
                "tag": "vampire_bat",
                "action": "merge",
                "replacement": "vampires",
                "rationale": "Too narrow; merge into broader vampire programming",
            }
        ]
    }


def test_tag_audit_endpoint(monkeypatch):
    client = TestClient(create_app())

    async def fake_audit(tags, *, debug=False):  # pragma: no cover - simple stub
        return [
            TagAuditResult(
                tag="ultra_specific_plot_detail",
                reason="Too detailed and specific for scheduling decisions",
            ),
            TagAuditResult(
                tag="obscure_reference",
                reason="Too obscure for general TV channel scheduling",
            ),
        ]

    monkeypatch.setattr(routes, "audit_tags", fake_audit)

    payload = {
        "tags": [
            "action",
            "comedy",
            "ultra_specific_plot_detail",
            "obscure_reference",
            "family_friendly",
        ]
    }

    response = client.post("/tags/audit", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "tags_to_delete": [
            {
                "tag": "ultra_specific_plot_detail",
                "reason": "Too detailed and specific for scheduling decisions",
            },
            {
                "tag": "obscure_reference",
                "reason": "Too obscure for general TV channel scheduling",
            },
        ]
    }
