from unittest.mock import MagicMock, patch

import pytest
from app.main import app
from app.services.recommendation_service import ServingContext
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_serving():
    mock_service = MagicMock()
    mock_service.model_version = "test-model@v1"
    mock_catalog = MagicMock()
    app.state.serving = ServingContext(
        service=mock_service,
        popularity_ranking=[42],
        model_version="test-model@v1",
        catalog_searcher=mock_catalog,
    )
    app.state.serving_error = None
    return TestClient(app)


def test_get_explanations_disabled(monkeypatch, client_with_serving):
    from app.config import settings

    monkeypatch.setattr(settings, "enable_explainability", False)
    response = client_with_serving.get("/api/explanations", params={"user_id": 1, "item_id": 42})
    assert response.status_code == 404


def test_get_explanations_returns_payload(client_with_serving):
    from app.services.explanation_service import ExplanationResult

    with patch("app.api.explanations.explain_recommendation") as explain:
        explain.return_value = ExplanationResult(
            user_id=1,
            item_id=42,
            title="Test Movie",
            retrieval_score=0.9,
            rank_score=0.8,
            features={"retrieval_score": 0.9},
            feature_importance={"retrieval_score": 1.0},
            history_length=5,
            variant="generative",
        )
        response = client_with_serving.get("/api/explanations", params={"user_id": 1, "item_id": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["item_id"] == 42
    assert payload["title"] == "Test Movie"
    assert payload["variant"] == "generative"
    assert payload["features"]["retrieval_score"] == 0.9
