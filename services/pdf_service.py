from typing import Optional
import uuid
import os
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.text_splitter import split_text
from models.pdf_chat import PDFChat, chat_storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize the PDF processor with storage directories.

        Args:
            storage_dir: Base directory for storing embeddings and metadata
        """
        self.base_dir = Path(storage_dir)
        self.embeddings_dir = self.base_dir / "embeddings"
        self.metadata_dir = self.base_dir / "metadata"

        # Create storage directories if they don't exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings()

    def _save_knowledge_base(self, chat_id: str, knowledge_base: FAISS) -> None:
        """Save the FAISS knowledge base to disk."""
        try:
            embedding_path = self.embeddings_dir / f"{chat_id}.faiss"
            knowledge_base.save_local(str(embedding_path))
            logger.info(f"Saved knowledge base for chat ID: {chat_id}")

        except Exception as e:
            logger.error(f"Error saving knowledge base: {str(e)}")
            raise RuntimeError(f"Failed to save knowledge base: {str(e)}")

    def _load_knowledge_base(self, chat_id: str) -> Optional[FAISS]:
        """Load a previously saved knowledge base."""
        try:
            embedding_path = self.embeddings_dir / f"{chat_id}.faiss"

            if embedding_path.exists():
                knowledge_base = FAISS.load_local(
                    str(embedding_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Loaded knowledge base for chat ID: {chat_id}")
                return knowledge_base

            return None

        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise RuntimeError(f"Failed to load knowledge base: {str(e)}")

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                text = "".join(
                    page.extract_text()
                    for page in pdf_reader.pages
                    if page.extract_text()
                )
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def _create_knowledge_base(self, text: str) -> FAISS:
        """Create a new FAISS knowledge base from text."""
        try:
            chunks = split_text(text)
            knowledge_base = FAISS.from_texts(chunks, self.embeddings)
            return knowledge_base
        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            raise RuntimeError(f"Failed to create knowledge base: {str(e)}")

    def process_pdf(self, file_path: str) -> str:
        """
        Process a PDF file and create a chat session.

        Args:
            file_path: Path to the PDF file

        Returns:
            Chat ID for the session
        """
        try:
            # Generate a new chat ID
            chat_id = str(uuid.uuid4())

            # Extract text from PDF
            text = self._extract_text_from_pdf(file_path)

            # Create knowledge base
            knowledge_base = self._create_knowledge_base(text)

            # Save knowledge base
            self._save_knowledge_base(chat_id, knowledge_base)

            # Create PDF chat instance with initial metadata
            chat = PDFChat(
                knowledge_base=knowledge_base,
                metadata={
                    "file_path": str(file_path),
                    "created_at": datetime.utcnow().isoformat(),
                    "file_name": Path(file_path).name,
                },
            )

            # Store chat session
            chat_storage[chat_id] = chat

            logger.info(f"Successfully processed PDF and created chat ID: {chat_id}")
            return chat_id

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

    def get_or_create_knowledge_base(self, chat_id: str) -> Optional[FAISS]:
        """Retrieve an existing knowledge base or create a new one if needed."""
        try:
            # Try to load existing knowledge base
            knowledge_base = self._load_knowledge_base(chat_id)

            if knowledge_base:
                return knowledge_base

            # If chat exists but knowledge base doesn't, recreate it
            chat_data = chat_storage.get(chat_id)
            if chat_data and chat_data.metadata.get("file_path"):
                text = self._extract_text_from_pdf(chat_data.metadata["file_path"])
                knowledge_base = self._create_knowledge_base(text)
                self._save_knowledge_base(chat_id, knowledge_base)
                return knowledge_base

            return None

        except Exception as e:
            logger.error(f"Error retrieving knowledge base: {str(e)}")
            raise RuntimeError(f"Failed to retrieve knowledge base: {str(e)}")


# Create a singleton instance
pdf_processor = PDFProcessor()
