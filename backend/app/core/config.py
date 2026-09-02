from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VERA Backend"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./data/vera.db"

    class Config:
        env_file = ".env"

settings = Settings()
