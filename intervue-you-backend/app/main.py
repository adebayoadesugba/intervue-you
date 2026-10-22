"""
main.py

FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
(from the project root, intervue-you-backend/)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_interview import router as interview_router
from app.api.routes_voice import router as voice_router
from app.rag import retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warms PyTorch, HuggingFace sentence-transformers, and ChromaDB on server startup
    to eliminate the 20-30s delay on the first interview question.
    """
    logger.info("Pre-warming embedding model and vector database...")
    try:
        # Trigger a lightweight lookup to initialize models into RAM
        retriever.get_relevant_questions(
            category="frontend",
            difficulty="junior",
            question_type="introduction",
            competency="communication",
            stack="core",
            query_text="prewarm",
            k=1,
        )
        logger.info("Embedding model & Vector DB are warm and ready!")
    except Exception as e:
        logger.warning(f"Startup pre-warming encountered a non-fatal note: {e}")

    yield


app = FastAPI(title="Intervue You", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during local development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(interview_router)
app.include_router(voice_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}