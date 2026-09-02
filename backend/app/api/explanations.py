from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.services.explanation_service import explain_recommendation

router = APIRouter()


class ExplanationResponse(BaseModel):
    user_id: int
    item_id: int
    title: str | None = None
    variant: str
    retrieval_score: float | None = None
    rank_score: float | None = None
    features: dict[str, float] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    history_length: int = 0


@router.get("/explanations", response_model=ExplanationResponse)
def get_explanation(
    request: Request,
    user_id: int,
    item_id: int,
    db: Session = Depends(get_db),
):
    if not settings.enable_explainability:
        raise HTTPException(status_code=404, detail="Explainability is disabled")

    serving = getattr(request.app.state, "serving", None)
    if serving is None or serving.service is None:
        error = getattr(request.app.state, "serving_error", "serving context not loaded")
        raise HTTPException(status_code=503, detail=f"Recommendation service unavailable: {error}")

    if serving.catalog_searcher is None:
        raise HTTPException(status_code=503, detail="Catalog searcher not available")

    try:
        result = explain_recommendation(
            db,
            serving.service,
            user_id=user_id,
            item_id=item_id,
            catalog_searcher=serving.catalog_searcher,
            user_cache=serving.user_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    return ExplanationResponse(
        user_id=result.user_id,
        item_id=result.item_id,
        title=result.title,
        variant=result.variant,
        retrieval_score=result.retrieval_score,
        rank_score=result.rank_score,
        features=result.features,
        feature_importance=result.feature_importance,
        history_length=result.history_length,
    )
