"""RAG system — document processing, vector search, and LLM response generation."""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()


class DocumentProcessor:
    """Process and chunk documents for RAG."""

    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_text_files(self, directory: str) -> List[Document]:
        """Load text files from directory."""
        documents = []
        data_dir = Path(directory)

        if not data_dir.exists():
            print(f"Directory {directory} does not exist")
            return documents

        for file_path in data_dir.glob("*.txt"):
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": "text",
                    },
                ))
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({"chunk_id": i, "total_chunks": len(chunks)})
                chunked_docs.append(chunk)

        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

    def process_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Complete document processing pipeline: load → chunk → embed."""
        documents = self.load_text_files(directory)
        if not documents:
            print("No documents found!")
            return []

        chunks = self.chunk_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.get_embeddings(texts)

        return [
            {"content": chunk.page_content, "metadata": chunk.metadata, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings)
        ]


class VectorStore:
    """In-memory vector store with vectorized cosine similarity search."""

    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.documents: List[Dict[str, Any]] = []
        # Normalised embeddings stored as a 2-D numpy array for fast batch search.
        self._embeddings_norm: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, processed_docs: List[Dict[str, Any]]):
        """Add processed documents to the store."""
        new_embeddings = []
        for doc in processed_docs:
            self.documents.append({"content": doc["content"], "metadata": doc["metadata"]})
            new_embeddings.append(doc["embedding"])

        # Pre-normalise so search is a single matrix multiply.
        raw = np.array(new_embeddings, dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalised = raw / np.where(norms > 0, norms, 1.0)

        self._embeddings_norm = (
            np.vstack([self._embeddings_norm, normalised])
            if self._embeddings_norm is not None
            else normalised
        )
        print(f"Added {len(processed_docs)} documents to vector store")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Return a normalised embedding vector for the query."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query],
            )
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k most relevant documents for the query."""
        if not self.documents or self._embeddings_norm is None:
            return []

        query_vec = self._get_query_embedding(query)
        if query_vec is None:
            return []

        # Vectorised cosine similarity (dot product of pre-normalised vectors).
        similarities = self._embeddings_norm @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "similarity_score": float(similarities[idx]),
            }
            for idx in top_indices
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, filepath: str):
        """Persist vector store to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        raw_embeddings = (
            self._embeddings_norm.tolist() if self._embeddings_norm is not None else []
        )
        with open(filepath, "w") as f:
            json.dump({"documents": self.documents, "embeddings": raw_embeddings}, f, indent=2)
        print(f"Vector store saved to {filepath}")

    def load_from_file(self, filepath: str):
        """Load vector store from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.documents = data["documents"]
        raw = np.array(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self._embeddings_norm = raw / np.where(norms > 0, norms, 1.0)

        print(f"Vector store loaded: {len(self.documents)} documents")


class RAGSystem:
    """End-to-end RAG system: retrieve relevant chunks, then generate a grounded answer."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = VectorStore(self.client)
        self.document_processor = DocumentProcessor(self.client)
        self.conversation_memory: Dict[str, List[Dict[str, str]]] = {}

        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """Load an existing vector store, or build one from the data directory."""
        try:
            self.vector_store.load_from_file("models/vector_store.json")
            print("✅ Loaded existing vector store")
        except Exception:
            print("📁 No existing vector store — building from data/raw ...")
            self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Build and save a new vector store from raw documents."""
        try:
            processed_docs = self.document_processor.process_documents("data/raw")
            if processed_docs:
                self.vector_store.add_documents(processed_docs)
                self.vector_store.save_to_file("models/vector_store.json")
                print("✅ Vector store created and saved")
            else:
                print("⚠️ No documents found to process")
        except Exception as e:
            print(f"⚠️ Could not initialize vector store: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, directory_path: str) -> int:
        """Ingest new documents into the RAG system."""
        processed_docs = self.document_processor.process_documents(directory_path)
        if processed_docs:
            self.vector_store.add_documents(processed_docs)
            self.vector_store.save_to_file("models/vector_store.json")
        return len(processed_docs)

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        return self.conversation_memory.get(conversation_id, [])

    def _update_memory(self, conversation_id: str, user_msg: str, assistant_msg: str):
        """Append a turn to conversation memory, keeping the last 10 exchanges."""
        history = self.conversation_memory.setdefault(conversation_id, [])
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
        if len(history) > 20:
            self.conversation_memory[conversation_id] = history[-20:]

    def _build_prompt(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
    ) -> str:
        context = "\n\n".join(
            f"[Source {i + 1}: {r['metadata'].get('filename', 'unknown')}]\n{r['content']}"
            for i, r in enumerate(search_results)
        )

        conversation_context = ""
        if conversation_history:
            conversation_context = "\n".join(
                f"{m['role'].title()}: {m['content']}" for m in conversation_history[-6:]
            )

        return f"""You are a helpful AI assistant that answers questions based on the provided context documents.

CONTEXT DOCUMENTS:
{context}

CONVERSATION HISTORY:
{conversation_context}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Answer using ONLY information from the context documents.
2. If the context is insufficient, say so clearly.
3. Cite the specific sources you use.
4. Be conversational and refer to previous messages if relevant.
5. Keep your answer concise but comprehensive.

ANSWER:"""

    def query(self, question: str, conversation_id: str = "default") -> Dict[str, Any]:
        """Retrieve relevant chunks and return a grounded LLM response."""
        try:
            search_results = self.vector_store.search(question, top_k=3)

            if not search_results:
                return {
                    "answer": "No relevant documents found. Please add documents first.",
                    "sources": [],
                    "confidence": 0.0,
                }

            history = self.get_conversation_history(conversation_id)
            prompt = self._build_prompt(question, search_results, history)

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )

            answer = response.choices[0].message.content
            sources = [r["metadata"].get("filename", "unknown") for r in search_results]
            confidence = min(
                sum(r["similarity_score"] for r in search_results) / len(search_results) * 1.2,
                1.0,
            )

            self._update_memory(conversation_id, question, answer)

            return {"answer": answer, "sources": sources, "confidence": confidence}

        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {
                "answer": f"Error processing your question: {e}",
                "sources": [],
                "confidence": 0.0,
            }


def test_rag_system():
    """Smoke-test the complete RAG system."""
    print("🚀 Testing RAG System...")
    rag = RAGSystem()

    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What's the difference between machine learning and deep learning?",
    ]

    for query in test_queries:
        print(f"\n🤔 Question: {query}")
        result = rag.query(query, "test_user")
        print(f"🤖 Answer: {result['answer']}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print("-" * 50)

    return rag


if __name__ == "__main__":
    test_rag_system()
