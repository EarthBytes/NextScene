"""Tests for item and user catalog API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    app.state.serving = MagicMock()
    app.state.serving_error = None
    return TestClient(app)


def test_search_items_requires_no_params(client, monkeypatch):
    def fake_search(_db, **kwargs):
        assert kwargs["q"] is None
        return ([{"item_id": 1, "title": "Toy Story", "genres": ["Animation"], "year": 1995}], 1)

    monkeypatch.setattr("app.api.items.search_items", fake_search)
    response = client.get("/api/items/search")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["title"] == "Toy Story"


def test_get_item_not_found(client, monkeypatch):
    monkeypatch.setattr("app.api.items.get_item", lambda _db, _id: None)
    response = client.get("/api/items/999")
    assert response.status_code == 404


def test_user_history_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.users.load_user_history",
        lambda _db, user_id, limit=50: [
            {
                "item_id": 1,
                "title": "Toy Story",
                "genres": ["Animation"],
                "year": 1995,
                "interaction_id": 10,
                "type": "rating",
                "ts": "2024-01-01T00:00:00+00:00",
                "context_json": {"rating": 5},
            }
        ],
    )
    monkeypatch.setattr(
        "app.api.users.user_stats",
        lambda _db, user_id: {"user_id": user_id, "interaction_count": 1, "rating_count": 1},
    )

    response = client.get("/api/users/1/history")
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 1
    assert payload["history"][0]["type"] == "rating"
