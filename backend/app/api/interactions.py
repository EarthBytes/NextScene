from app.db.session import get_db
from app.services.interaction_service import log_interaction
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

router = APIRouter()


class InteractionRequest(BaseModel):
    user_id: int = Field(..., description="User identifier")
    item_id: int = Field(..., description="Item identifier")
    type: str = Field(..., description="Interaction type: view, rating, tag, click, purchase")
    context_json: dict = Field(default_factory=dict, description="Optional context payload")


class InteractionResponse(BaseModel):
    interaction_id: int
    user_id: int
    item_id: int
    type: str
    ts: str


@router.post("/interactions", response_model=InteractionResponse, status_code=201)
def create_interaction(
    request: Request,
    body: InteractionRequest,
    db: Session = Depends(get_db),
):
    serving = getattr(request.app.state, "serving", None)
    user_cache = serving.user_cache if serving is not None else None

    try:
        interaction = log_interaction(
            db,
            user_id=body.user_id,
            item_id=body.item_id,
            interaction_type=body.type,
            context_json=body.context_json,
            user_cache=user_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return InteractionResponse(
        interaction_id=int(interaction.interaction_id),
        user_id=interaction.user_id,
        item_id=interaction.item_id,
        type=interaction.type,
        ts=interaction.ts.isoformat(),
    )
