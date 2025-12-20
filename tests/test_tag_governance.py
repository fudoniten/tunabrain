from fastapi.testclient import TestClient

from tunabrain.app import create_app
import tunabrain.api.routes as routes
from tunabrain.api.models import TagDecision


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
