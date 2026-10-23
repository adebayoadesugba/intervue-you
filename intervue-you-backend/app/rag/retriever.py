"""
retriever.py

Queries the Chroma vector store with strict 5-variable filtering.
Uses absolute DB_PATH from app.config to guarantee working directory independence.
Shuffles unranked sampling pools to prevent duplicate question sequences across candidates.
"""

import logging
import random
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import DB_PATH, EMBEDDING_MODEL
from app.models.session import InterviewSession

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def _get_vectorstore() -> Chroma:
    """Lazy initializer for vector store using absolute DB_PATH."""
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=_embeddings)
    return _vectorstore


def _doc_to_question(doc: Document) -> dict:
    """Convert a retrieved Document back into a plain question dict."""
    ref_points = doc.metadata.get("reference_points", "")
    parsed_refs = [r.strip() for r in ref_points.split(" | ") if r.strip()] if ref_points else []

    return {
        "id": doc.metadata.get("content_hash", doc.page_content),
        "question": doc.page_content,
        "category": doc.metadata.get("display_category", doc.metadata.get("category", "")),
        "difficulty": doc.metadata.get("display_difficulty", doc.metadata.get("difficulty", "")),
        "stack": doc.metadata.get("display_stack", doc.metadata.get("stack", "")),
        "question_type": doc.metadata.get("question_type", "technical"),
        "competency": doc.metadata.get("competency", "technical_depth"),
        "reference_points": parsed_refs,
        "source_file": doc.metadata.get("source_file", ""),
    }


def get_relevant_questions(
    category: str,
    difficulty: str,
    question_type: str,
    competency: str,
    stack: str = "core",
    query_text: str = "",
    exclude_ids: Optional[list[str]] = None,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve candidate questions matching strict blueprint constraints.
    Searches both the specific role AND the "general" shared bank.
    Shuffles unranked matches to vary questions across candidate sessions.
    """
    exclude_ids = exclude_ids or []
    store = _get_vectorstore()

    allowed_categories = [category.lower(), "general"]
    allowed_difficulties = [difficulty.lower(), "any"]
    allowed_stacks = [stack.lower(), "core", "none", "universal"]

    # --- STRICT VARIABLE FILTER CONSTRUCTION ---
    filter_conditions = [
        {"category": {"$in": allowed_categories}},
        {"difficulty": {"$in": allowed_difficulties}},
        {"question_type": {"$eq": question_type.lower()}},
        {"stack": {"$in": allowed_stacks}},
    ]

    if competency:
        filter_conditions.append({"competency": {"$eq": competency.lower()}})

    filter_dict = {
        "$and": filter_conditions
    }

    # --- 1. SEMANTIC SEARCH WITH FALLBACK ---
    if query_text.strip():
        fetch_k = max(20, k + len(exclude_ids) + 5)
        try:
            raw_results = store.similarity_search(query_text, k=fetch_k, filter=filter_dict)
            questions = [_doc_to_question(doc) for doc in raw_results]
            filtered = [q for q in questions if q["id"] not in exclude_ids]
            if filtered:
                return filtered[:k]
        except Exception as e:
            print(f"⚠️ Vector search error in ChromaDB: {e}. Falling back to unranked sampling.")

    # --- 2. UNRANKED SAMPLING WITH FALLBACK ---
    try:
        raw_get = store.get(where=filter_dict, limit=50)
    except Exception as e:
        print(f"⚠️ Filtered store.get failed: {e}. Attempting unconstrained retrieval.")
        try:
            raw_get = store.get(limit=50)
        except Exception as e2:
            print(f"⚠️ Store retrieval failed completely: {e2}.")
            raw_get = {}

    docs = raw_get.get("documents") or []
    metas = raw_get.get("metadatas") or []
    raw_results = [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]

    questions = [_doc_to_question(doc) for doc in raw_results]
    filtered = [q for q in questions if q["id"] not in exclude_ids]

    random.shuffle(filtered)
    return filtered[:k]

def verify_session_agenda_fillable(session: InterviewSession) -> tuple[bool, str]:
    """
    Pre-flight check: Verifies ChromaDB has enough questions to fill every 
    slot in the generated Blueprint Agenda before the interview starts.
    """
    for slot in session.agenda:
        if slot.question_type == "resume_based":
            continue
            
        matches = get_relevant_questions(
            category=session.category,
            difficulty=session.difficulty,
            question_type=slot.question_type,
            competency=slot.competency,
            stack=slot.stack,
            exclude_ids=session.user_asked_question_ids,
            k=1
        )
        
        if not matches:
            error_msg = f"Missing question bank data for Slot {slot.slot_id} | Type: '{slot.question_type}' | Competency: '{slot.competency}' | Stack: '{slot.stack}'."
            logger.error(f"Pre-flight failed: {error_msg}")
            return False, error_msg

    return True, "Session agenda verified."