from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.recommendation_service import ServingContext


def test_create_interaction_returns_201():
    mock_service = MagicMock()
    app.state.serving = ServingContext(
        service=mock_service,
        popularity_ranking=[42],
        model_version="test-model@v1",
    )
    app.state.serving_error = None
    client = TestClient(app)

    interaction = MagicMock()
    interaction.interaction_id = 99
    interaction.user_id = 1
    interaction.item_id = 42
    interaction.type = "rating"
    interaction.ts = datetime(2026, 9, 2, tzinfo=UTC)

    with patch("app.api.interactions.log_interaction", return_value=interaction):
        response = client.post(
            "/api/interactions",
            json={
                "user_id": 1,
                "item_id": 42,
                "type": "rating",
                "context_json": {"rating": 4.5},
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["interaction_id"] == 99
    assert payload["type"] == "rating"


def test_create_interaction_validates_type():
    mock_service = MagicMock()
    app.state.serving = ServingContext(
        service=mock_service,
        popularity_ranking=[42],
        model_version="test-model@v1",
    )
    client = TestClient(app)

    with patch(
        "app.api.interactions.log_interaction",
        side_effect=ValueError("Invalid interaction type 'bad'"),
    ):
        response = client.post(
            "/api/interactions",
            json={"user_id": 1, "item_id": 42, "type": "bad"},
        )
    assert response.status_code == 400
