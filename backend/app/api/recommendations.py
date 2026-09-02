from fastapi import APIRouter, Depends, HTTPException, Request
from prometheus_client import Counter, Histogram
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

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_pipeline_seconds",
    "Recommendation pipeline stage latency in seconds",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

INFERENCE_ERRORS = (RuntimeError, ValueError)


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
    latency_ms: float | None = None
    timing: dict[str, float] | None = None


def _record_timing(timing) -> None:
    for stage, ms in timing.as_dict().items():
        stage_name = stage.removesuffix("_ms")
        RECOMMENDATION_LATENCY.labels(stage=stage_name).observe(ms / 1000)


def _popularity_results(serving, db: Session, user_id: int, k: int):
    return popularity_recommendations(
        db,
        user_id=user_id,
        k=k,
        popularity_ranking=serving.popularity_ranking,
        user_cache=serving.user_cache,
    )


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
    timing = None

    if settings.enable_ab_test and serving.service is not None:
        experiment = EXPERIMENT_NAME
        variant = force_variant or assign_ab_variant(
            user_id,
            generative_fraction=settings.ab_test_generative_fraction,
        )
        use_generative = variant == "generative"
        RECOMMENDATION_VARIANT.labels(experiment=experiment, variant=variant).inc()

    if use_generative and serving.service is not None:
        try:
            results, timing = serving.service.recommend(db, user_id=user_id, k=k)
            model_version = serving.service.model_version
            if variant is None:
                variant = "generative"
        except INFERENCE_ERRORS:
            if not settings.enable_fallback_recs:
                raise
            results = _popularity_results(serving, db, user_id, k)
            model_version = serving.model_version
            variant = "popularity"
    elif settings.enable_fallback_recs:
        results = _popularity_results(serving, db, user_id, k)
        model_version = serving.model_version
        if variant is None:
            variant = "popularity"
    else:
        raise HTTPException(status_code=503, detail="Model not loaded and fallback is disabled")

    if timing is not None:
        _record_timing(timing)

    return RecommendationResponse(
        user_id=user_id,
        recommendations=[
            RecommendationItem(item_id=rec.item_id, title=rec.title, score=rec.score)
            for rec in results
        ],
        model_version=model_version,
        variant=variant,
        experiment=experiment,
        latency_ms=timing.total_ms if timing is not None else None,
        timing=timing.as_dict() if timing is not None else None,
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
