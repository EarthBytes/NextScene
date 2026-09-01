from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api import health, recommendations
from app.config import settings
from app.db.session import SessionLocal
from app.services.recommendation_service import try_load_serving_context


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = SessionLocal()
    try:
        app.state.serving = try_load_serving_context(session)
    except Exception as exc:
        app.state.serving = None
        app.state.serving_error = str(exc)
    finally:
        session.close()
    yield


app = FastAPI(
    title="Generative Recommendation System",
    description="Transformer + vector search + ranking for next-item prediction",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def root():
    return {
        "service": "generative-recsys",
        "version": "0.1.0",
        "docs": "/docs",
        "explainability": settings.enable_explainability,
        "model_loaded": getattr(app.state, "serving", None) is not None
        and app.state.serving.service is not None,
    }
