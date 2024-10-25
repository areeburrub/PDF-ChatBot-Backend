import os
from fastapi import APIRouter, UploadFile, HTTPException
from services.pdf_service import pdf_processor
from schema.chat import UploadResponse

router = APIRouter()

UPLOAD_DIRECTORY = "./uploads"  # Define your upload directory


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile):
    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        # Save the uploaded file locally
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        chat_id = pdf_processor.process_pdf(file_path)
        return UploadResponse(chat_id=chat_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
