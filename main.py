import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.endpoints import upload, chat
from core.config import load_env

# Initialize environment variables
load_env()

app = FastAPI(title="PDF Chat API")

# Configure CORS
origins = [
    "http://localhost:3000",  # Your frontend URL
    "https://pdf-chat-bot-frontend.vercel.app",
    # Add any other allowed origins (frontend URLs) here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(upload.router, prefix="/v1")
app.include_router(chat.router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
