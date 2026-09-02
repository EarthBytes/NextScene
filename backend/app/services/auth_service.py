"""Authentication helpers: password hashing and JWT tokens."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.interaction import Interaction
from app.models.user import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc


def user_display_name(user: User) -> str:
    profile = user.profile_json or {}
    display = profile.get("display_name")
    if isinstance(display, str) and display.strip():
        return display.strip()
    return user.username or ""


def register_user(session: Session, *, username: str, password: str) -> User:
    display_name = username.strip()
    normalized = display_name.lower()
    if len(normalized) < 3:
        raise ValueError("Username must be at least 3 characters")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    existing = session.execute(
        select(User.id).where(func.lower(User.username) == normalized)
    ).scalar_one_or_none()
    if existing is not None:
        raise ValueError("Username already taken")

    user = User(
        username=normalized,
        password_hash=hash_password(password),
        profile_json={"display_name": display_name},
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, *, username: str, password: str) -> User | None:
    normalized = username.strip().lower()
    user = session.execute(
        select(User).where(func.lower(User.username) == normalized)
    ).scalar_one_or_none()
    if user is None or user.password_hash is None:
        return None
    if not verify_password(password, user.password_hash):
        return None

    profile = dict(user.profile_json or {})
    display = profile.get("display_name")
    if not isinstance(display, str) or not display.strip():
        profile["display_name"] = user.username or normalized
        user.profile_json = profile
        session.commit()
        session.refresh(user)

    return user


def delete_user_account(session: Session, user: User) -> None:
    if user.username is None or user.password_hash is None:
        raise ValueError("Only app accounts can be deleted")

    session.execute(delete(Interaction).where(Interaction.user_id == user.id))
    session.delete(user)
    session.commit()
