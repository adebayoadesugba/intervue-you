"""
retriever.py

Queries the Chroma vector store built by embed_questions.py.
Filters by category + difficulty first (hard filter), then performs
semantic search within that filtered pool, excluding any questions
already asked in the current session.
"""

from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DB_NAME = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy-loaded singletons
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def _get_vectorstore() -> Chroma:
    """Lazy initializer for vector store and embeddings model."""
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=_embeddings)
    return _vectorstore


def _doc_to_question(doc: Document) -> dict:
    """Convert a retrieved Document back into a plain question dict."""
    ref_points = doc.metadata.get("reference_points", "")
    parsed_refs = [r.strip() for r in ref_points.split(" | ") if r.strip()] if ref_points else []

    return {
        "question": doc.page_content,
        # Return display string if available; fallback to stored category
        "category": doc.metadata.get("display_category", doc.metadata.get("category", "")),
        "subcategory": doc.metadata.get("display_subcategory", doc.metadata.get("subcategory", "")),
        "difficulty": doc.metadata.get("display_difficulty", doc.metadata.get("difficulty", "")),
        "question_type": doc.metadata.get("question_type", "technical"),
        "reference_points": parsed_refs,
        "source_file": doc.metadata.get("source_file", ""),
    }


def get_relevant_questions(
    category: str,
    difficulty: str,
    query_text: str = "",
    exclude_questions: Optional[list[str]] = None,
    k: int = 10,
) -> list[dict]:
    """
    Retrieve up to k candidate questions matching category + difficulty,
    optionally semantically ranked by query_text, excluding already asked items.
    """
    exclude_questions = exclude_questions or []
    store = _get_vectorstore()

    # Hard metadata filter
    filter_dict = {
        "$and": [
            {"category": {"$eq": category}},
            {"difficulty": {"$eq": difficulty}},
        ]
    }

    fetch_k = k + len(exclude_questions) + 2

    if query_text.strip():
        # Semantic search on filtered pool
        results = store.similarity_search(
            query_text, k=fetch_k, filter=filter_dict
        )
    else:
        # Fallback for session initializers / unranked sampling
        raw_get = store.get(where=filter_dict, limit=fetch_k)
        docs = raw_get.get("documents") or []
        metas = raw_get.get("metadatas") or []

        results = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(docs, metas)
        ]

    # Convert to standard dict structures and filter exclusions
    questions = [_doc_to_question(doc) for doc in results]
    filtered_questions = [q for q in questions if q["question"] not in exclude_questions]

    return filtered_questions[:k]


if __name__ == "__main__":
    # Test retrieval
    results = get_relevant_questions(category="frontend", difficulty="junior")
    print(f"Retrieved {len(results)} questions:\n")
    for r in results:
        print(f"- {r['question']} [{r['subcategory']}]")
        print(f"  Reference Points: {r['reference_points']}")