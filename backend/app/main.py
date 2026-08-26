from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api import health, recommendations
from app.config import settings

app = FastAPI(
    title="Generative Recommendation System",
    description="Transformer + vector search + ranking for next-item prediction",
    version="0.1.0",
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
    }
