from app.db.session import get_db
from app.models.user import User
from app.services.auth_service import decode_access_token
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = decode_access_token(token)
        user_id = int(payload["sub"])
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    user = db.get(User, user_id)
    if user is None or user.username is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
