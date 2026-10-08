from typing import List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    document_id: str
    filename: str
    chunks_created: int


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SourceChunk(BaseModel):
    document_id: str
    content: str
    similarity: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceChunk]


class NewSessionResponse(BaseModel):
    session_id: str


class ChatMessageOut(BaseModel):
    role: str
    content: str
    sources: Optional[list] = None
    created_at: Optional[str] = None
