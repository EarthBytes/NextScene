from contextlib import asynccontextmanager
from pathlib import Path

from app.api import auth, explanations, health, interactions, items, me, recommendations, users
from app.config import settings
from app.db.session import SessionLocal
from app.logging_config import configure_logging
from app.middleware.latency import LatencyLoggingMiddleware, latency_percentiles
from app.middleware.security import SecurityHeadersMiddleware
from app.services.recommendation_service import try_load_serving_context, warmup_serving_context
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

configure_logging()


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
    title="NextScene API",
    description="Personalized movie recommendations powered by transformer inference and vector search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(SecurityHeadersMiddleware)

if settings.enable_latency_logging:
    app.add_middleware(LatencyLoggingMiddleware)

cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(me.router, prefix="/api", tags=["me"])
app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])
app.include_router(interactions.router, prefix="/api", tags=["interactions"])
app.include_router(explanations.router, prefix="/api", tags=["explanations"])
app.include_router(items.router, prefix="/api", tags=["items"])
app.include_router(users.router, prefix="/api", tags=["users"])

posters_dir = Path(settings.posters_dir)
posters_dir.mkdir(parents=True, exist_ok=True)
app.mount("/posters", StaticFiles(directory=str(posters_dir)), name="posters")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def root():
    serving = getattr(app.state, "serving", None)
    return {
        "service": "nextscene",
        "version": "1.0.0",
        "docs": "/docs",
        "explainability": settings.enable_explainability,
        "model_loaded": serving is not None and serving.service is not None,
        "retrieval_mode": serving.retrieval_mode if serving is not None else None,
        "latency": latency_percentiles(),
    }
