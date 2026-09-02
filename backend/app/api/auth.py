from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    delete_user_account,
    register_user,
    user_display_name,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

router = APIRouter()


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: int
    username: str


@router.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(body: AuthRequest, db: Session = Depends(get_db)):
    try:
        user = register_user(db, username=body.username, password=body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_access_token(user.id, user_display_name(user))
    return AuthResponse(
        access_token=token,
        user={"id": user.id, "username": user_display_name(user)},
    )


@router.post("/auth/login", response_model=AuthResponse)
def login(body: AuthRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, username=body.username, password=body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user.id, user_display_name(user))
    return AuthResponse(
        access_token=token,
        user={"id": user.id, "username": user_display_name(user)},
    )


@router.get("/auth/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return UserResponse(id=current_user.id, username=user_display_name(current_user))


@router.delete("/auth/me", status_code=204)
def delete_account(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = current_user.id
        delete_user_account(db, current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    serving = getattr(request.app.state, "serving", None)
    if serving is not None and serving.user_cache is not None:
        serving.user_cache.invalidate(user_id)
        serving.user_cache.invalidate(1_000_000_000 + user_id)
