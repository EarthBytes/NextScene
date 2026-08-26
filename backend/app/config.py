from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://recsys:recsys@localhost:5432/recsys"
    omdb_api_key: str = ""
    transformer_model_path: str = "models/transformer"
    ranking_model_path: str = "models/ranking"
    faiss_index_path: str = "data/faiss/items.index"
    enable_explainability: bool = True
    enable_fallback_recs: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
