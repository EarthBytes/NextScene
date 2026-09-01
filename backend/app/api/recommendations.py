from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.services.recommendation_service import popularity_recommendations

router = APIRouter()


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User identifier")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations")


class RecommendationItem(BaseModel):
    item_id: int
    title: str | None = None
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    model_version: str = "not-trained"


def _recommendations_for_user(request: Request, db: Session, user_id: int, k: int) -> RecommendationResponse:
    serving = getattr(request.app.state, "serving", None)
    if serving is None:
        error = getattr(request.app.state, "serving_error", "serving context not loaded")
        raise HTTPException(status_code=503, detail=f"Recommendation service unavailable: {error}")

    if serving.service is not None:
        results = serving.service.recommend(db, user_id=user_id, k=k)
        model_version = serving.service.model_version
    elif settings.enable_fallback_recs:
        results = popularity_recommendations(
            db,
            user_id=user_id,
            k=k,
            popularity_ranking=serving.popularity_ranking,
        )
        model_version = serving.model_version
    else:
        raise HTTPException(status_code=503, detail="Model not loaded and fallback is disabled")

    return RecommendationResponse(
        user_id=user_id,
        recommendations=[
            RecommendationItem(item_id=rec.item_id, title=rec.title, score=rec.score)
            for rec in results
        ],
        model_version=model_version,
    )


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    request: Request,
    user_id: int,
    k: int = 10,
    db: Session = Depends(get_db),
):
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")
    return _recommendations_for_user(request, db, user_id=user_id, k=k)


@router.post("/recommendations", response_model=RecommendationResponse)
def post_recommendations(
    request: Request,
    body: RecommendationRequest,
    db: Session = Depends(get_db),
):
    return _recommendations_for_user(request, db, user_id=body.user_id, k=body.k)
