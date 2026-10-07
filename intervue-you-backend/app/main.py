"""
main.py

FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
(from the project root, intervue-you-backend/)
"""

from fastapi import FastAPI

from app.api.routes_interview import router as interview_router

app = FastAPI(title="Intervue You")
app.include_router(interview_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
