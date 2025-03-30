from fastapi import FastAPI
from app.routers.ollama_router import router as query_router

app = FastAPI(title="FastAPI AI API", description="API de ejemplo de consultas a los modelos Llama y OpenAi", version="1.0")

app.include_router(query_router, prefix="/api", tags=["query"])