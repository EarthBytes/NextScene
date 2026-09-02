from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter()


@router.get("/health")
def health_check(request: Request, db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    serving = getattr(request.app.state, "serving", None)
    return {
        "status": "ok",
        "database": "connected",
        "model_loaded": serving is not None and serving.service is not None,
        "retrieval_mode": serving.retrieval_mode if serving is not None else None,
    }
