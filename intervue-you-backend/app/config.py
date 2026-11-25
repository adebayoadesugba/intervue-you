"""
config.py

Centralized configuration for system paths, database locations, and embedding models.
Prevents vector DB path divergence across different execution directories.
"""

from pathlib import Path

# Absolute path resolution for the backend directory root
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Shared absolute vector store location
DB_PATH = str(BACKEND_DIR / "vector_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ALLOWED_QUESTION_TYPES = {"introduction", "behavioral", "situational", "resume_based", "technical"}
ALLOWED_COMPETENCIES = {"communication", "teamwork", "problem_solving", "technical_depth"}