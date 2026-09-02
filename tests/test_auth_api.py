"""Tests for authentication endpoints."""

from unittest.mock import patch

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_register_and_login(monkeypatch):
    calls = {"count": 0}

    def fake_register(db, *, username, password):
        from app.models.user import User

        calls["count"] += 1
        return User(
            id=99,
            username=username.lower(),
            password_hash="hashed",
            profile_json={"display_name": username},
        )

    def fake_authenticate(db, *, username, password):
        from app.models.user import User

        return User(
            id=99,
            username=username.lower(),
            password_hash="hashed",
            profile_json={"display_name": username},
        )

    monkeypatch.setattr("app.api.auth.register_user", fake_register)
    monkeypatch.setattr("app.api.auth.authenticate_user", fake_authenticate)
    monkeypatch.setattr("app.api.auth.create_access_token", lambda uid, uname: "test-token")

    register_response = client.post(
        "/api/auth/register",
        json={"username": "newuser", "password": "secret12"},
    )
    assert register_response.status_code == 201
    assert register_response.json()["access_token"] == "test-token"

    login_response = client.post(
        "/api/auth/login",
        json={"username": "newuser", "password": "secret12"},
    )
    assert login_response.status_code == 200
    assert login_response.json()["user"]["username"] == "newuser"


def test_register_preserves_display_name(monkeypatch):
    captured: dict = {}

    def fake_register(db, *, username, password):
        from app.models.user import User

        captured["username"] = username
        return User(
            id=1,
            username=username.lower(),
            password_hash="hashed",
            profile_json={"display_name": username},
        )

    monkeypatch.setattr("app.api.auth.register_user", fake_register)
    monkeypatch.setattr("app.api.auth.create_access_token", lambda uid, uname: "token")

    response = client.post(
        "/api/auth/register",
        json={"username": "CateJames", "password": "secret12"},
    )
    assert response.status_code == 201
    assert response.json()["user"]["username"] == "CateJames"
def test_delete_account():
    from app.api.deps import get_current_user
    from app.models.user import User

    def fake_get_user():
        return User(
            id=42,
            username="catejames",
            password_hash="hashed",
            profile_json={"display_name": "CateJames"},
        )

    deleted: dict = {}

    def fake_delete(db, user):
        deleted["user_id"] = user.id

    app.dependency_overrides[get_current_user] = fake_get_user
    try:
        with patch("app.api.auth.delete_user_account", fake_delete):
            response = client.delete(
                "/api/auth/me",
                headers={"Authorization": "Bearer test-token"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 204
    assert deleted["user_id"] == 42
