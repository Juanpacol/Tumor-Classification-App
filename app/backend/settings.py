from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore", protected_namespaces=()
    )

    database_url: str = "postgresql://ether_user:ether_password@localhost:5432/ether_db"
    db_statement_cache_size: int | None = None
    storage_dir: str = "./storage"
    model_bundle_path: str | None = None
    meta_model_api_key: str | None = None
    meta_model_id: str = "muse-spark-1.3"
    cors_origins: str = "http://localhost:3000"

    @property
    def async_database_url(self) -> str:
        for prefix in ("postgresql://", "postgres://"):
            if self.database_url.startswith(prefix):
                return "postgresql+asyncpg://" + self.database_url.removeprefix(prefix)
        return self.database_url

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
