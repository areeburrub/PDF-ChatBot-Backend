from fastapi import APIRouter, HTTPException
from schema.chat import ChatRequest
from services.chat_service import chat_service

router = APIRouter()


@router.post("/chat/{chat_id}")
async def chat(chat_id: str, chat_request: ChatRequest):
    try:
        response = chat_service.get_response(chat_id, chat_request.question)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
