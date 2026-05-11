from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable, List, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .ingest import build_documents, split_documents


PYTHON_SYMBOL_PATTERNS = [
    ("class", re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")),
]

RUST_SYMBOL_PATTERNS = [
    ("function", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")),
    ("struct", re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("enum", re.compile(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("impl", re.compile(r"^\s*impl\s+([A-Za-z_][A-Za-z0-9_]*)")),
]


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


class PostgresRAGStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            chunk_size=settings.embedding_batch_size,
        )

    def _connect(self):
        if not self.settings.database_url:
            raise RuntimeError("DATABASE_URL is required for the postgres backend")
        conn = psycopg.connect(self.settings.database_url, row_factory=dict_row)
        register_vector(conn)
        return conn

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_") or "unknown"

    def _ensure_schema(self, embedding_dim: int) -> None:
        dim = int(embedding_dim)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS code_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        repo_path TEXT NOT NULL,
                        path TEXT NOT NULL,
                        source TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        language TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dim}) NOT NULL,
                        tsv tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(content, ''))
                        ) STORED
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS symbols (
                        id BIGSERIAL PRIMARY KEY,
                        repo_path TEXT NOT NULL,
                        path TEXT NOT NULL,
                        symbol_name TEXT NOT NULL,
                        symbol_type TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        signature TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        tsv tsvector GENERATED ALWAYS AS (
                            to_tsvector(
                                'english',
                                coalesce(symbol_name, '') || ' ' || coalesce(symbol_type, '') || ' ' || coalesce(signature, '')
                            )
                        ) STORED
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commits (
                        id BIGSERIAL PRIMARY KEY,
                        repo_path TEXT NOT NULL,
                        commit_hash TEXT NOT NULL,
                        author TEXT NOT NULL,
                        commit_date TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        tsv tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(commit_hash, '') || ' ' || coalesce(author, '') || ' ' || coalesce(message, ''))
                        ) STORED
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_path ON code_chunks (repo_path)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_repo_path ON symbols (repo_path)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_commits_repo_path ON commits (repo_path)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_code_chunks_tsv ON code_chunks USING GIN (tsv)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_tsv ON symbols USING GIN (tsv)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_commits_tsv ON commits USING GIN (tsv)")
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding
                    ON code_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            conn.commit()

    @staticmethod
    def _chunk_metadata(doc: Document, chunk_index: int) -> dict:
        return {
            **doc.metadata,
            "chunk_index": chunk_index,
        }

    @staticmethod
    def _symbol_patterns(language: str):
        if language == "rs":
            return RUST_SYMBOL_PATTERNS
        return PYTHON_SYMBOL_PATTERNS

    def _extract_symbols(self, docs: Sequence[Document]) -> List[dict]:
        symbols: List[dict] = []
        for doc in docs:
            if doc.metadata.get("kind") != "file":
                continue
            language = doc.metadata.get("language", "")
            path = doc.metadata.get("path", doc.metadata.get("source", "unknown"))
            lines = doc.page_content.splitlines()
            patterns = self._symbol_patterns(language)
            for line_number, line in enumerate(lines, start=1):
                for symbol_type, pattern in patterns:
                    match = pattern.match(line)
                    if not match:
                        continue
                    symbol_name = match.group(1)
                    symbols.append(
                        {
                            "repo_path": str(self.settings.repo_path),
                            "path": path,
                            "symbol_name": symbol_name,
                            "symbol_type": symbol_type,
                            "start_line": line_number,
                            "end_line": line_number,
                            "signature": line.strip(),
                            "metadata": {
                                "language": language,
                                "source": path,
                            },
                        }
                    )
        return symbols

    def index_repository(self) -> int:
        docs = build_documents(self.settings)
        file_docs = [doc for doc in docs if doc.metadata.get("kind") == "file"]
        commit_docs = [doc for doc in docs if doc.metadata.get("kind") == "git_commit"]
        chunks = split_documents(file_docs)

        if not chunks and not commit_docs:
            return 0

        if chunks:
            first_embedding = self.embeddings.embed_query(chunks[0].page_content)
            self._ensure_schema(len(first_embedding))
        else:
            # Fallback dimension for empty code repositories.
            self._ensure_schema(len(self.embeddings.embed_query("")))

        chunk_vectors = self.embeddings.embed_documents([chunk.page_content for chunk in chunks]) if chunks else []
        commit_vectors = self.embeddings.embed_documents([doc.page_content for doc in commit_docs]) if commit_docs else []

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM code_chunks WHERE repo_path = %s", (str(self.settings.repo_path),))
                cur.execute("DELETE FROM symbols WHERE repo_path = %s", (str(self.settings.repo_path),))
                cur.execute("DELETE FROM commits WHERE repo_path = %s", (str(self.settings.repo_path),))

                for chunk_index, (chunk, embedding) in enumerate(zip(chunks, chunk_vectors), start=1):
                    metadata = self._chunk_metadata(chunk, chunk_index)
                    cur.execute(
                        """
                        INSERT INTO code_chunks (
                            repo_path, path, source, kind, language, chunk_index, content, metadata, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(self.settings.repo_path),
                            str(metadata.get("path", metadata.get("source", "unknown"))),
                            str(metadata.get("source", "unknown")),
                            str(metadata.get("kind", "file")),
                            str(metadata.get("language", "")),
                            int(chunk_index),
                            chunk.page_content,
                            Jsonb(metadata),
                            embedding,
                        ),
                    )

                symbols = self._extract_symbols(file_docs)
                for symbol in symbols:
                    cur.execute(
                        """
                        INSERT INTO symbols (
                            repo_path, path, symbol_name, symbol_type, start_line, end_line, signature, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            symbol["repo_path"],
                            symbol["path"],
                            symbol["symbol_name"],
                            symbol["symbol_type"],
                            symbol["start_line"],
                            symbol["end_line"],
                            symbol["signature"],
                            Jsonb(symbol["metadata"]),
                        ),
                    )

                for doc, embedding in zip(commit_docs, commit_vectors):
                    commit_hash = str(doc.metadata.get("sha", ""))
                    if not commit_hash:
                        continue
                    cur.execute(
                        """
                        INSERT INTO commits (
                            repo_path, commit_hash, author, commit_date, message, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(self.settings.repo_path),
                            commit_hash,
                            str(doc.metadata.get("author", "")),
                            str(doc.metadata.get("date", "")),
                            doc.page_content,
                            Jsonb(dict(doc.metadata)),
                        ),
                    )
            conn.commit()

        return len(chunks)

    def _row_to_document(self, row: dict, kind: str) -> Document:
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "source": row.get("source", row.get("path", "unknown")),
                "path": row.get("path", row.get("source", "unknown")),
                "kind": kind,
            }
        )
        if row.get("symbol_name"):
            metadata["symbol_name"] = row["symbol_name"]
            metadata["symbol_type"] = row.get("symbol_type", "")
            metadata["start_line"] = row.get("start_line", 0)
            metadata["end_line"] = row.get("end_line", 0)
        if row.get("commit_hash"):
            metadata["sha"] = row["commit_hash"]
            metadata["author"] = row.get("author", "")
            metadata["date"] = row.get("commit_date", "")
        return Document(page_content=row.get("content") or row.get("message") or row.get("signature") or "", metadata=metadata)

    def _semantic_search(self, conn, query_embedding: Sequence[float], k: int) -> List[SearchHit]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT path, source, kind, language, content, metadata, (1 - (embedding <=> (%s::vector))) AS score
                FROM code_chunks
                WHERE repo_path = %s
                ORDER BY embedding <=> (%s::vector)
                LIMIT %s
                """,
                (query_embedding, str(self.settings.repo_path), query_embedding, k),
            )
            rows = cur.fetchall()
        return [SearchHit(self._row_to_document(row, "file"), float(row.get("score") or 0.0)) for row in rows]

    def _keyword_search(self, conn, query: str, k: int) -> List[SearchHit]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT path, source, kind, language, content, metadata,
                       ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
                FROM code_chunks
                WHERE repo_path = %s
                  AND tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, str(self.settings.repo_path), query, k),
            )
            rows = cur.fetchall()
        return [SearchHit(self._row_to_document(row, "file"), float(row.get("score") or 0.0)) for row in rows]

    def _symbol_search(self, conn, query: str, k: int) -> List[SearchHit]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT path, symbol_name, symbol_type, start_line, end_line, signature, metadata,
                       ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
                FROM symbols
                WHERE repo_path = %s
                  AND tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC, start_line ASC
                LIMIT %s
                """,
                (query, str(self.settings.repo_path), query, k),
            )
            rows = cur.fetchall()
        return [SearchHit(self._row_to_document(row, "symbol"), float(row.get("score") or 0.0) + 0.25) for row in rows]

    def _git_search(self, conn, query: str, k: int) -> List[SearchHit]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT commit_hash, author, commit_date, message, metadata,
                       ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
                FROM commits
                WHERE repo_path = %s
                  AND tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC, commit_date DESC
                LIMIT %s
                """,
                (query, str(self.settings.repo_path), query, k),
            )
            rows = cur.fetchall()
        return [SearchHit(self._row_to_document(row, "git_commit"), float(row.get("score") or 0.0) + 0.1) for row in rows]

    @staticmethod
    def _fuse_hits(*result_sets: Iterable[SearchHit], limit: int = 8) -> List[Document]:
        scores: dict[str, dict] = {}
        for result_set in result_sets:
            for rank, hit in enumerate(result_set):
                doc = hit.document
                metadata = doc.metadata
                key = "|".join(
                    [
                        str(metadata.get("kind", "")),
                        str(metadata.get("path", metadata.get("source", ""))),
                        str(metadata.get("symbol_name", "")),
                        str(metadata.get("sha", metadata.get("commit_hash", ""))),
                        str(hash(doc.page_content)),
                    ]
                )
                item = scores.setdefault(key, {"doc": doc, "score": 0.0})
                item["doc"] = doc
                item["score"] += hit.score + (1.0 / (rank + 1))
        ranked = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
        return [item["doc"] for item in ranked[:limit]]

    def search(self, query: str, query_type: str = "architecture", k: int | None = None) -> List[Document]:
        k = k or self.settings.top_k
        expanded_k = max(k * 3, 10)
        with self._connect() as conn:
            query_embedding = self.embeddings.embed_query(query)
            semantic = self._semantic_search(conn, query_embedding, expanded_k)
            keyword = self._keyword_search(conn, query, expanded_k)
            symbol = self._symbol_search(conn, query, expanded_k)
            git = self._git_search(conn, query, expanded_k)

        if query_type == "symbol_lookup":
            return self._fuse_hits(symbol, semantic, keyword, git, limit=k)
        if query_type == "change_history":
            return self._fuse_hits(git, keyword, semantic, symbol, limit=k)
        if query_type == "migration_plan":
            return self._fuse_hits(semantic, keyword, symbol, git, limit=k)
        return self._fuse_hits(semantic, keyword, symbol, git, limit=k)


class PostgresHybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = PostgresRAGStore(settings)

    def search(self, query: str, query_type: str = "architecture", k: int | None = None) -> List[Document]:
        return self.store.search(query=query, query_type=query_type, k=k)


def index_postgres_repo(settings: Settings) -> int:
    store = PostgresRAGStore(settings)
    return store.index_repository()
