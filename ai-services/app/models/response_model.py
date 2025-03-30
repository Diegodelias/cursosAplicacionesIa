from pydantic import BaseModel

class QueryResponse(BaseModel):
    result: str