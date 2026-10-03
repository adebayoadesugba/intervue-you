"""
embed_questions.py

Walks the knowledge_base/question_bank/ folder, loads every JSON question file,
and embeds all questions into a local Chroma vector store.

Run this once to build the store, and again any time you add or edit
question files.

Usage:
    python embed_questions.py
"""

import glob
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Adjust this path if you run the script from a different working directory.
# Expected layout: knowledge-base/question_bank/<category>/<difficulty>.json

QUESTION_BANK_PATH = "knowledge-base/question_bank/**/*.json"
DB_NAME = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, local, small, fast, and good enough for our use case

def load_questions() -> list[Document]:
    """Load every question JSON file and turn each question into a Document."""
    documents = []
    files = glob.glob(QUESTION_BANK_PATH, recursive=True)
    print(f"Found {len(files)} question bank files")

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        for q in questions:
            # Chroma metadata values must be str / int / float / bool —
            # a list like reference_points has to be flattened into a string.
            # Split it back out on " | " wherever you read metadata later.
            metadata = {
                "category": q["category"],
                "subcategory": q.get("subcategory", ""),
                "difficulty": q["difficulty"],
                "question_type": q.get("question_type", "technical"),
                "reference_points": " | ".join(q.get("reference_points", [])),
                "source_file": file_path,
            }

            documents.append(Document(page_content=q["question"], metadata=metadata))
            print(documents)

    return documents


def main():
    documents = load_questions()
    print(f"Loaded {len(documents)} questions total")

    if not documents:
        print("No questions found — check that knowledge-base/question_bank/ exists and has .json files in it.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    print(f"Embedded and stored {len(documents)} questions in '{DB_NAME}/'")

    # Quick sanity check: how many questions landed in each category/difficulty
    breakdown = {}
    for doc in documents:
        key = (doc.metadata["category"], doc.metadata["difficulty"])
        breakdown[key] = breakdown.get(key, 0) + 1

    print("\nBreakdown by category / difficulty:")
    for (category, difficulty), count in sorted(breakdown.items()):
        print(f"  {category:<15} {difficulty:<10} {count}")


if __name__ == "__main__":
    main()
