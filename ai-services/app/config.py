import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_API_URL: str = "http://localhost:11434"  # Cambiar a tu servidor local
    # OLLAMA_API_KEY no es necesaria para el modelo local
    # OLLAMA_API_KEY: str = None

settings = Settings()