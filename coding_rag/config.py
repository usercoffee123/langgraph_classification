from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    backend: str
    repo_path: Path
    persist_dir: Path
    database_url: str
    collection_name: str
    embedding_model: str
    embedding_batch_size: int
    chat_model: str
    top_k: int


    @staticmethod
    def from_env() -> "Settings":
        repo = Path(os.getenv("RAG_REPO_PATH", ".")).resolve()
        persist = Path(os.getenv("RAG_PERSIST_DIR", ".rag/chroma")).resolve()
        database_url = os.getenv("DATABASE_URL", "").strip()
        backend = os.getenv("RAG_BACKEND", "postgres" if database_url else "chroma").strip().lower()
        default_embedding_model = "text-embedding-3-small" if backend == "postgres" else "text-embedding-3-large"
        return Settings(
            backend=backend,
            repo_path=repo,
            persist_dir=persist,
            database_url=database_url,
            collection_name=os.getenv("RAG_COLLECTION", "codebase_chunks"),
            embedding_model=os.getenv("RAG_EMBED_MODEL", default_embedding_model),
            embedding_batch_size=int(os.getenv("RAG_EMBED_BATCH_SIZE", "50")),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
            top_k=int(os.getenv("RAG_TOP_K", "8")),
        )
