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
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
