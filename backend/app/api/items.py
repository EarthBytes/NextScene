from app.db.session import get_db
from app.services.item_service import get_item, list_genres, list_tags, search_items
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

router = APIRouter()


class ItemSummary(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    year: int | None = None
    image_url: str | None = None
    poster_url: str | None = None
    imdb_id: str | None = None
    metadata_json: dict = Field(default_factory=dict)


class ItemDetail(ItemSummary):
    description: str | None = None


class ItemSearchResponse(BaseModel):
    items: list[ItemSummary]
    total: int
    limit: int
    offset: int
    query: str | None = None
    genre: str | None = None
    genres: list[str] = Field(default_factory=list)
    tag: str | None = None


class GenreCount(BaseModel):
    genre: str
    count: int


class TagCount(BaseModel):
    tag: str
    count: int


@router.get("/items/search", response_model=ItemSearchResponse)
def search_catalog(
    q: str | None = Query(None, description="Title search query"),
    genre: str | None = Query(None, description="Single genre filter"),
    genres: list[str] | None = Query(None, description="Genre filters"),
    tag: str | None = Query(None, description="Tag search"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    active_genres = genres or ([genre] if genre else None)
    items, total = search_items(
        db,
        q=q,
        genres=active_genres,
        tag=tag,
        limit=limit,
        offset=offset,
    )
    return ItemSearchResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        query=q,
        genre=genre,
        genres=active_genres or [],
        tag=tag,
    )


@router.get("/items/genres", response_model=list[GenreCount])
def get_genres(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return list_genres(db, limit=limit)


@router.get("/items/tags", response_model=list[TagCount])
def get_tags(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return list_tags(db, limit=limit)


@router.get("/items/{item_id}", response_model=ItemDetail)
def get_item_detail(item_id: int, db: Session = Depends(get_db)):
    item = get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return item
