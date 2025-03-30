from fastapi import APIRouter, Depends, HTTPException
import requests
from app.models.request_model import QueryRequest
from app.models.response_model import QueryResponse
from app.services.ollama_service import OllamaService


router = APIRouter()

@router.post("/simple-request", response_model=QueryResponse)
def simple_query_ollama(request: QueryRequest):
    try:
        return OllamaService.simple_query_api(request)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/request-with-history", response_model=QueryResponse)
def query_ollama(request: QueryRequest):
    try:
        return OllamaService.chat_api(request)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/request-with-langchain", response_model=QueryResponse)
def query_ollama_langchain(request: QueryRequest):
    try:
        # return OllamaService.chat_langchain(request)
        return OllamaService.chat_with_template(request)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))