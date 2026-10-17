"""
embed_questions.py
 
Walks the data/question_bank/ folder, loads every JSON question file,
and embeds all questions into a local Chroma vector store.
 
Run this once to build the store, and again any time you add or edit
question files.

Usage:
    python embed_questions.py
"""

import glob
import hashlib
import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

QUESTION_BANK_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "knowledge-base" / "question_bank" / "**" / "*.json")
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DB_NAME = str(BACKEND_DIR / "vector_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 500

ALLOWED_QUESTION_TYPES = {"introduction", "behavioral", "situational", "resume_based", "technical"}
ALLOWED_COMPETENCIES = {"communication", "teamwork", "problem_solving", "technical_depth"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions() -> dict[str, Document]:
    docs = {}
    warnings_count = 0

    for file_path in glob.glob(QUESTION_BANK_PATH, recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        for idx, q in enumerate(questions):
            text = q.get("question", "").strip()
            if not text:
                continue

            refs = q.get("reference_points", [])

            # --- Validation Guardrails ---
            q_type = q.get("question_type", "technical").strip().lower()
            competency = q.get("competency", "").strip().lower()

            # 1. Disallow unknown question_types
            if q_type not in ALLOWED_QUESTION_TYPES:
                print(f"❌ ERROR [{file_path} item {idx}]: Invalid question_type '{q_type}'. Must be one of {ALLOWED_QUESTION_TYPES}")
                sys.exit(1)

            # 2. Require competency for behavioral questions
            if q_type == "behavioral" and not competency:
                print(f"❌ ERROR [{file_path} item {idx}]: Behavioral questions MUST explicitly state a competency (e.g. 'teamwork').")
                sys.exit(1)

            # 3. Fallback handling for missing non-behavioral competencies
            if not competency:
                warnings_count += 1
                competency = "communication" if q_type == "introduction" else "technical_depth"
                print(f"⚠️ WARNING [{file_path} item {idx}]: Missing competency. Defaulted to '{competency}'.")

            # 4. Disallow unknown competencies
            if competency not in ALLOWED_COMPETENCIES:
                print(f"❌ ERROR [{file_path} item {idx}]: Invalid competency '{competency}'. Must be one of {ALLOWED_COMPETENCIES}")
                sys.exit(1)

            # --- Original values for UI display ---
            orig_cat = q.get("category", "General").strip()
            orig_sub = q.get("subcategory", "").strip()
            orig_diff = q.get("difficulty", "Unspecified").strip()
            orig_stack = q.get("stack", "core").strip()

            metadata = {
                # Normalized metadata for strict Chroma filtering
                "category": orig_cat.lower(),
                "subcategory": orig_sub.lower(),
                "difficulty": orig_diff.lower(),
                "stack": orig_stack.lower(),
                "question_type": q_type,
                "competency": competency,
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

    if warnings_count > 0:
        print(f"⚠️ Validation finished with {warnings_count} fallback warnings. Please update legacy JSONs.")

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