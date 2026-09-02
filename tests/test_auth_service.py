"""Tests for password hashing."""

from unittest.mock import MagicMock

from app.services.auth_service import delete_user_account, hash_password, register_user, verify_password


def test_register_preserves_display_name_casing():
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None

    user = register_user(session, username="CateJames", password="secret12")
    assert user.username == "catejames"
    assert user.profile_json["display_name"] == "CateJames"


def test_register_rejects_case_variant_username():
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = 42

    try:
        register_user(session, username="CATEJAMES", password="secret12")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "already taken" in str(exc).lower()


def test_delete_user_account_removes_interactions():
    session = MagicMock()
    user = MagicMock()
    user.id = 7
    user.username = "catejames"
    user.password_hash = "hashed"

    delete_user_account(session, user)

    session.execute.assert_called_once()
    session.delete.assert_called_once_with(user)
    session.commit.assert_called_once()
    hashed = hash_password("my-secret-password")
    assert hashed != "my-secret-password"
    assert verify_password("my-secret-password", hashed)
    assert not verify_password("wrong-password", hashed)


def test_hash_accepts_normal_length_passwords():
    password = "a" * 50
    hashed = hash_password(password)
    assert verify_password(password, hashed)
