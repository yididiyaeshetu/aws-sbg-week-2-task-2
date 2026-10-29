"""
Week 2: FAQ semantic search with Pandas + ChromaDB

Pipeline:
  1. Load a CSV/JSON of FAQs with Pandas
  2. Clean the text and chunk it
  3. Embed and store the chunks in Chroma
  4. Query it and print the top 3 matches

Install:  pip install pandas chromadb
Run:      python faq_search.py
          python faq_search.py faqs.json "how do I reset my password?"

Expected columns in the input file: question, answer
"""

import re
import sys

import chromadb
import pandas as pd

DATA_PATH = "faqs.csv"          # or "faqs.json"
COLLECTION_NAME = "faqs"
CHUNK_SIZE = 80                 # max words per chunk
CHUNK_OVERLAP = 15              # words shared between neighbouring chunks
TOP_K = 3


# ---------------------------------------------------------------- 1. LOAD
def load_faqs(path: str) -> pd.DataFrame:
    """Load FAQs from a CSV or JSON file into a DataFrame."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .json")

    # Normalise column names so 'Question ' and 'question' both work
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {"question", "answer"} - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing column(s): {missing}")

    print(f"Loaded {len(df)} rows from {path}")
    return df[["question", "answer"]]


# --------------------------------------------------------------- 2. CLEAN
def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, trim."""
    text = re.sub(r"<[^>]+>", " ", str(text))   # remove HTML tags
    text = re.sub(r"\s+", " ", text)            # collapse whitespace/newlines
    return text.strip()


def clean_faqs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and duplicates, then clean every text field."""
    before = len(df)
    df = df.dropna(subset=["question", "answer"]).copy()

    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)

    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    print(f"Cleaning: {before} rows -> {len(df)} rows")
    return df


# --------------------------------------------------------------- 3. CHUNK
def chunk_words(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks of at most `size` words."""
    words = text.split()
    if len(words) <= size:
        return [text]

    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(df: pd.DataFrame):
    """
    Turn each FAQ into one or more chunks.
    Each chunk keeps the question as a prefix so it stays searchable
    even when a long answer is split up.
    """
    ids, documents, metadatas = [], [], []

    for faq_id, row in df.iterrows():
        for i, piece in enumerate(chunk_words(row["answer"])):
            ids.append(f"faq{faq_id}_chunk{i}")
            documents.append(f"Q: {row['question']} A: {piece}")
            metadatas.append({
                "faq_id": int(faq_id),
                "chunk_index": i,
                "question": row["question"],
            })

    print(f"Created {len(documents)} chunks")
    return ids, documents, metadatas


# -------------------------------------------------- 4. EMBED + STORE (Chroma)
def build_collection(ids, documents, metadatas):
    """
    Store chunks in Chroma. With no embedding function specified, Chroma uses
    its default model (all-MiniLM-L6-v2) to embed the documents automatically.
    """
    client = chromadb.PersistentClient(path="./chroma_db")

    # Start fresh each run so re-running doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance
    )
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Stored {collection.count()} chunks in Chroma")
    return collection


# ------------------------------------------------------------------ 5. QUERY
def search(collection, query: str, top_k: int = TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)

    print(f"\nQuery: {query!r}")
    print("=" * 60)

    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]),
        start=1,
    ):
        similarity = 1 - dist   # cosine distance -> similarity
        print(f"\n#{rank}  (similarity: {similarity:.3f})")
        print(f"Question: {meta['question']}")
        print(f"Match:    {doc}")


# ------------------------------------------------------------------- MAIN
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    query = sys.argv[2] if len(sys.argv) > 2 else "How can I change my password?"

    df = load_faqs(path)
    df = clean_faqs(df)
    ids, documents, metadatas = build_chunks(df)
    collection = build_collection(ids, documents, metadatas)
    search(collection, query)


if __name__ == "__main__":
    main()
