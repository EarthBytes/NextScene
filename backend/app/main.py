from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api import explanations, health, interactions, recommendations
from app.config import settings
from app.db.session import SessionLocal
from app.middleware.latency import LatencyLoggingMiddleware, latency_percentiles
from app.services.recommendation_service import try_load_serving_context, warmup_serving_context


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = SessionLocal()
    try:
        app.state.serving = try_load_serving_context(session)
        app.state.serving_error = None
        if app.state.serving is not None:
            warmup_serving_context(session, app.state.serving)
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

if settings.enable_latency_logging:
    app.add_middleware(LatencyLoggingMiddleware)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])
app.include_router(interactions.router, prefix="/api", tags=["interactions"])
app.include_router(explanations.router, prefix="/api", tags=["explanations"])

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def root():
    serving = getattr(app.state, "serving", None)
    return {
        "service": "generative-recsys",
        "version": "0.1.0",
        "docs": "/docs",
        "explainability": settings.enable_explainability,
        "model_loaded": serving is not None and serving.service is not None,
        "retrieval_mode": serving.retrieval_mode if serving is not None else None,
        "latency": latency_percentiles(),
    }
