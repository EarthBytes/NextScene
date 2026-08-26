from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(user_id: int, k: int = 10):
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")

    return RecommendationResponse(
        user_id=user_id,
        recommendations=[],
        model_version="not-trained",
    )


@router.post("/recommendations", response_model=RecommendationResponse)
def post_recommendations(body: RecommendationRequest):
    return get_recommendations(user_id=body.user_id, k=body.k)
