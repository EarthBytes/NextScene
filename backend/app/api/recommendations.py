from fastapi import APIRouter, Depends, HTTPException, Request
from prometheus_client import Counter
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.services.ranking_service import EXPERIMENT_NAME, assign_ab_variant
from app.services.recommendation_service import popularity_recommendations

router = APIRouter()

RECOMMENDATION_VARIANT = Counter(
    "recommendations_variant_total",
    "Recommendation responses by A/B experiment variant",
    ["experiment", "variant"],
)


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
    variant: str | None = None
    experiment: str | None = None


def _recommendations_for_user(
    request: Request,
    db: Session,
    user_id: int,
    k: int,
    *,
    force_variant: str | None = None,
) -> RecommendationResponse:
    serving = getattr(request.app.state, "serving", None)
    if serving is None:
        error = getattr(request.app.state, "serving_error", "serving context not loaded")
        raise HTTPException(status_code=503, detail=f"Recommendation service unavailable: {error}")

    variant: str | None = None
    experiment: str | None = None
    use_generative = serving.service is not None

    if settings.enable_ab_test and serving.service is not None:
        experiment = EXPERIMENT_NAME
        variant = force_variant or assign_ab_variant(
            user_id,
            generative_fraction=settings.ab_test_generative_fraction,
        )
        use_generative = variant == "generative"
        RECOMMENDATION_VARIANT.labels(experiment=experiment, variant=variant).inc()

    if use_generative and serving.service is not None:
        results = serving.service.recommend(db, user_id=user_id, k=k)
        model_version = serving.service.model_version
        if variant is None:
            variant = "generative"
    elif settings.enable_fallback_recs:
        results = popularity_recommendations(
            db,
            user_id=user_id,
            k=k,
            popularity_ranking=serving.popularity_ranking,
        )
        model_version = serving.model_version
        if variant is None:
            variant = "popularity"
    else:
        raise HTTPException(status_code=503, detail="Model not loaded and fallback is disabled")

    return RecommendationResponse(
        user_id=user_id,
        recommendations=[
            RecommendationItem(item_id=rec.item_id, title=rec.title, score=rec.score)
            for rec in results
        ],
        model_version=model_version,
        variant=variant,
        experiment=experiment,
    )


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    request: Request,
    user_id: int,
    k: int = 10,
    variant: str | None = None,
    db: Session = Depends(get_db),
):
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")
    if variant is not None and variant not in {"generative", "popularity"}:
        raise HTTPException(status_code=400, detail="variant must be 'generative' or 'popularity'")
    return _recommendations_for_user(request, db, user_id=user_id, k=k, force_variant=variant)


@router.post("/recommendations", response_model=RecommendationResponse)
def post_recommendations(
    request: Request,
    body: RecommendationRequest,
    db: Session = Depends(get_db),
):
    return _recommendations_for_user(request, db, user_id=body.user_id, k=body.k)
