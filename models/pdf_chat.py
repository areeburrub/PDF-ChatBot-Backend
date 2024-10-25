# models/pdf_chat.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain_community.vectorstores import FAISS


@dataclass
class PDFChat:
    """
    Data class representing a PDF chat session.
    """

    knowledge_base: FAISS
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()


class ChatStorage:
    """
    Storage class for managing chat sessions.
    """

    def __init__(self):
        self._storage: Dict[str, PDFChat] = {}

    def get(self, chat_id: str) -> Optional[PDFChat]:
        """Get chat session by ID."""
        return self._storage.get(chat_id)

    def __setitem__(self, chat_id: str, chat: PDFChat):
        """Store chat session."""
        self._storage[chat_id] = chat

    def update(self, chat_id: str, chat: PDFChat):
        """Update existing chat session."""
        self._storage[chat_id] = chat

    def delete(self, chat_id: str):
        """Delete chat session."""
        self._storage.pop(chat_id, None)


# Create a singleton instance
chat_storage = ChatStorage()
