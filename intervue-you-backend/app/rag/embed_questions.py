"""
embed_questions.py
 
Walks the data/question_bank/ folder, loads every JSON question file,
and embeds all questions into a local Chroma vector store.
 
Run this once to build the store, and again any time you add or edit
question files.
intervue-you-backend
 
Usage:
    python embed_questions.py
"""

import glob
import hashlib
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pathlib import Path
QUESTION_BANK_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "knowledge-base" / "question_bank" / "**" / "*.json")
DB_NAME = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 500


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions() -> dict[str, Document]:
    docs = {}

    for file_path in glob.glob(QUESTION_BANK_PATH, recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        for q in questions:
            text = q["question"].strip()
            refs = q.get("reference_points", [])

            # Original values for UI display
            orig_cat = q.get("category", "General").strip()
            orig_sub = q.get("subcategory", "").strip()
            orig_diff = q.get("difficulty", "Unspecified").strip()
            orig_stack = q.get("stack", "core").strip()

            metadata = {
                # Normalized metadata for case-insensitive Chroma filtering
                "category": orig_cat.lower(),
                "subcategory": orig_sub.lower(),
                "difficulty": orig_diff.lower(),
                "stack": orig_stack.lower(),
                "question_type": q.get("question_type", "technical").strip().lower(),
                "reference_points": " | ".join(refs) if isinstance(refs, list) else str(refs),
                "source_file": file_path,
                # Preserved original strings for clean display output
                "display_category": orig_cat,
                "display_subcategory": orig_sub,
                "display_difficulty": orig_diff,
                "display_stack": orig_stack,
            }

            # Hash covers content and metadata
            payload = {"text": text, **metadata}
            metadata["content_hash"] = sha(json.dumps(payload, sort_keys=True))

            doc_id = sha(f"{file_path}::{text}")
            docs[doc_id] = Document(page_content=text, metadata=metadata)

    return docs


def main():
    current = load_questions()
    if not current:
        print("No questions found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

    existing = store.get(include=["metadatas"]) or {}
    existing_ids = existing.get("ids") or []
    existing_metas = existing.get("metadatas") or []

    existing_hashes = {
        doc_id: (meta or {}).get("content_hash")
        for doc_id, meta in zip(existing_ids, existing_metas)
    }

    # 1. Purge stale
    stale = [i for i in existing_hashes if i not in current]
    if stale:
        for start in range(0, len(stale), BATCH_SIZE):
            store.delete(ids=stale[start : start + BATCH_SIZE])

    # 2. Upsert changed / new
    to_upsert = [
        i for i, d in current.items()
        if existing_hashes.get(i) != d.metadata["content_hash"]
    ]
    if to_upsert:
        for start in range(0, len(to_upsert), BATCH_SIZE):
            batch_ids = to_upsert[start : start + BATCH_SIZE]
            store.add_documents(
                documents=[current[i] for i in batch_ids],
                ids=batch_ids,
            )

    print(f"Sync complete. Active: {len(current)} | Embedded: {len(to_upsert)} | Removed: {len(stale)}")


if __name__ == "__main__":
    main()