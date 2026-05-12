"""
indexer.py
Embeds parsed DITA chunks using OpenAI embeddings and stores in ChromaDB.
"""

import os
import chromadb
from openai import OpenAI
from typing import List, Dict

COLLECTION_NAME = "dita_docs"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_chroma_client(persist_dir: str = "./chroma_store"):
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(chroma_client):
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def index_chunks(chunks: List[Dict], persist_dir: str = "./chroma_store"):
    """Embed and index all chunks into ChromaDB."""
    chroma_client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(chroma_client)

    # Clear existing data for clean re-index
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"Cleared {len(existing['ids'])} existing documents")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "topic_id": c["topic_id"],
            "topic_title": c["topic_title"],
            "section_title": c["section_title"],
            "source_file": c["source_file"],
            "tags": c["tags"],
            "topic_shortdesc": c["topic_shortdesc"],
        } for c in chunks]
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB")
    return len(chunks)
