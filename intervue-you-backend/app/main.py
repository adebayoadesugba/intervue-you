"""
main.py

FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
(from the project root, intervue-you-backend/)
"""

from fastapi import FastAPI

from app.api.routes_interview import router as interview_router
from app.api.routes_voice import router as voice_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Intervue You")
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