from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://recsys:recsys@localhost:5432/recsys"
    omdb_api_key: str = ""
    tmdb_api_key: str = ""
    transformer_model_path: str = "models/transformer-full-v2"
    ranking_model_path: str = "models/ranking"
    enable_ranking: bool = True
    ranking_candidate_pool_size: int = 50
    enable_ab_test: bool = False
    ab_test_generative_fraction: float = 0.5
    faiss_index_path: str = "data/faiss/items.index"
    sequences_cache_path: str = "data/sequences"
    posters_dir: str = "data/posters"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    enable_explainability: bool = True
    enable_fallback_recs: bool = True
    enable_faiss_serving: bool = False
    user_cache_ttl_seconds: int = 300
    user_cache_max_size: int = 1000
    enable_latency_logging: bool = True
    warmup_on_startup: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_format: str = "console"
    environment: str = "development"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080

    def validate_production(self) -> None:
        if self.environment != "production":
            return
        if self.jwt_secret == "dev-secret-change-in-production" or len(self.jwt_secret) < 32:
            raise ValueError("Set a strong JWT_SECRET (32+ chars) in production")


settings = Settings()
settings.validate_production()
