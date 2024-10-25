from typing import Dict, Any, TypedDict, Optional
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from services.pdf_service import pdf_processor
from models.pdf_chat import chat_storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    messages: list
    context: str


class ChatService:
    def __init__(self):
        self.memory = MemorySaver()
        self.model = ChatOpenAI()

    def _create_workflow(self, knowledge_base) -> StateGraph:
        """
        Creates a new workflow graph for processing chat messages.

        Args:
            knowledge_base: FAISS vector store for the current chat
        """
        workflow = StateGraph(state_schema=ChatState)

        # Create bound method with knowledge_base
        def search_with_kb(state):
            return self._search_chunks(state, knowledge_base)

        # Add nodes and edges
        workflow.add_node("search", search_with_kb)
        workflow.add_node("model", self._call_model)
        workflow.set_entry_point("search")
        workflow.add_edge("search", "model")
        workflow.set_finish_point("model")

        return workflow

    def _search_chunks(self, state: ChatState, knowledge_base) -> Dict[str, Any]:
        """
        Searches for relevant chunks in the knowledge base.

        Args:
            state: Current chat state
            knowledge_base: FAISS vector store to search in
        """
        try:
            user_question = state["messages"][-1].content
            # Add safety checks for knowledge base operations
            if not hasattr(knowledge_base, "similarity_search"):
                raise ValueError("Invalid knowledge base format")

            docs = knowledge_base.similarity_search(user_question)
            # Validate document format before processing
            if not all(hasattr(doc, "page_content") for doc in docs):
                raise ValueError("Invalid document format in knowledge base")

            pdf_context = " ".join([doc.page_content for doc in docs])
            return {"messages": state["messages"], "context": pdf_context}
        except Exception as e:
            logger.error(f"Error in search_chunks: {str(e)}")
            raise RuntimeError(f"Failed to search document chunks: {str(e)}")

    def _call_model(self, state: ChatState) -> Dict[str, Any]:
        """Generates a response using the language model."""
        try:
            # Validate state contents before processing
            if (
                not isinstance(state, dict)
                or "context" not in state
                or "messages" not in state
            ):
                raise ValueError("Invalid state format")

            pdf_context = state["context"]
            user_question = state["messages"][-1].content

            # Sanitize inputs before combining
            pdf_context = str(pdf_context)
            user_question = str(user_question)

            # Construct a prompt that includes context and question
            combined_input = (
                f"Context from PDF:\n{pdf_context}\n\n"
                f"User Question: {user_question}\n\n"
                f"Please provide a response based on the context above."
            )

            # Get model response
            response = self.model.invoke([HumanMessage(content=combined_input)])

            return {
                "messages": state["messages"] + [response],
                "context": state["context"],
            }
        except Exception as e:
            logger.error(f"Error in call_model: {str(e)}")
            raise RuntimeError(f"Failed to generate model response: {str(e)}")

    def get_response(self, chat_id: str, user_question: str) -> Dict[str, Any]:
        """
        Processes a user question and returns an AI response with chat history.

        Args:
            chat_id: Unique identifier for the chat session
            user_question: The user's question about the PDF

        Returns:
            Dict containing the AI response and updated chat history

        Raises:
            ValueError: If chat_id is not found
            RuntimeError: If processing fails
        """
        try:
            # Validate inputs
            if not isinstance(chat_id, str) or not chat_id.strip():
                raise ValueError("Invalid chat ID")
            if not isinstance(user_question, str) or not user_question.strip():
                raise ValueError("Invalid user question")

            # Retrieve chat data
            chat_data = chat_storage.get(chat_id)
            if not chat_data:
                raise ValueError(f"Chat session not found for ID: {chat_id}")

            # Get knowledge base
            knowledge_base = pdf_processor.get_or_create_knowledge_base(chat_id)
            if not knowledge_base:
                raise ValueError(f"Knowledge base not found for chat ID: {chat_id}")

            # Update chat's knowledge base reference
            chat_data.knowledge_base = knowledge_base

            # Create and compile workflow
            workflow = self._create_workflow(knowledge_base)
            app_instance = workflow.compile(checkpointer=self.memory)

            # Initialize state with the user's question
            initial_state = {
                "messages": [HumanMessage(content=user_question)],
                "context": "",
            }

            # Process the message through the workflow
            config = {"configurable": {"thread_id": chat_id}}
            last_event = None

            for event in app_instance.stream(
                initial_state, config, stream_mode="values"
            ):
                last_event = event

            if not last_event:
                raise RuntimeError("No response generated from the model")

            # Extract AI response
            ai_response = last_event["messages"][-1].content

            # Update chat history
            chat_data.messages.append(
                {
                    "user": user_question,
                    "ai": ai_response,
                    "timestamp": chat_data.get_timestamp(),
                }
            )

            # Save updated chat data
            chat_storage.update(chat_id, chat_data)

            return {"response": ai_response, "chat_history": chat_data.messages}

        except ValueError as e:
            logger.error(f"ValueError in get_response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise RuntimeError(f"Failed to process chat message: {str(e)}")


# Create a singleton instance
chat_service = ChatService()
