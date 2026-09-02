from app.db.session import get_db
from app.services.item_service import load_user_history, user_stats
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

router = APIRouter()


class HistoryEntry(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    year: int | None = None
    image_url: str | None = None
    poster_url: str | None = None
    imdb_id: str | None = None
    metadata_json: dict = Field(default_factory=dict)
    interaction_id: int
    type: str
    ts: str
    context_json: dict = Field(default_factory=dict)


class UserStatsResponse(BaseModel):
    user_id: int
    interaction_count: int
    rating_count: int


class UserHistoryResponse(BaseModel):
    user_id: int
    history: list[HistoryEntry]
    stats: UserStatsResponse


@router.get("/users/{user_id}/history", response_model=UserHistoryResponse)
def get_user_history(
    user_id: int,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    history = load_user_history(db, user_id, limit=limit)
    stats = user_stats(db, user_id)
    return UserHistoryResponse(user_id=user_id, history=history, stats=stats)
