from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class UploadResponse(BaseModel):
    chat_id: str
