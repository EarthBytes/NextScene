from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.recommendation_service import Recommendation, ServingContext


@pytest.fixture
def client_with_mock_service():
    mock_service = MagicMock()
    mock_service.model_version = "test-model@v1"
    mock_service.recommend.return_value = [
        Recommendation(item_id=42, title="Test Movie", score=0.91),
        Recommendation(item_id=7, title="Another Film", score=0.88),
    ]
    app.state.serving = ServingContext(
        service=mock_service,
        popularity_ranking=[42, 7],
        model_version="test-model@v1",
    )
    app.state.serving_error = None
    return TestClient(app)


def test_get_recommendations_returns_model_results(client_with_mock_service):
    response = client_with_mock_service.get("/api/recommendations", params={"user_id": 123, "k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 123
    assert payload["model_version"] == "test-model@v1"
    assert len(payload["recommendations"]) == 2
    assert payload["recommendations"][0]["item_id"] == 42
    assert payload["recommendations"][0]["title"] == "Test Movie"


def test_get_recommendations_ab_test_force_variant(client_with_mock_service, monkeypatch):
    from app.config import settings
    from app.services.recommendation_service import Recommendation

    monkeypatch.setattr(settings, "enable_ab_test", True)
    monkeypatch.setattr(
        "app.api.recommendations.popularity_recommendations",
        lambda *_args, **_kwargs: [Recommendation(item_id=7, title="Popular", score=1.0)],
    )

    response = client_with_mock_service.get(
        "/api/recommendations",
        params={"user_id": 123, "k": 2, "variant": "popularity"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["variant"] == "popularity"
    assert payload["experiment"] == "generative_vs_popularity"
    assert payload["recommendations"][0]["item_id"] == 7


def test_get_recommendations_validates_variant(client_with_mock_service):
    response = client_with_mock_service.get(
        "/api/recommendations",
        params={"user_id": 1, "k": 2, "variant": "invalid"},
    )
    assert response.status_code == 400


def test_get_recommendations_validates_k(client_with_mock_service):
    response = client_with_mock_service.get("/api/recommendations", params={"user_id": 1, "k": 0})
    assert response.status_code == 400
