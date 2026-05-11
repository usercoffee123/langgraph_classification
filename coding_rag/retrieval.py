from __future__ import annotations

from collections import defaultdict
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .ingest import build_documents, split_documents
from .postgres_backend import PostgresHybridRetriever


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.backend
        self.postgres_retriever = PostgresHybridRetriever(settings) if settings.backend == "postgres" else None
        embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.persist_dir),
        )

        source_docs = split_documents(build_documents(settings))
        self.bm25 = BM25Retriever.from_documents(source_docs)
        self.bm25.k = max(settings.top_k * 3, 10)

    def search(self, query: str, k: int | None = None) -> List[Document]:
        k = k or self.settings.top_k

        if self.postgres_retriever is not None:
            query_type = "architecture"
            lowered = query.lower()
            if any(token in lowered for token in ["where", "symbol", "function", "class", "defined"]):
                query_type = "symbol_lookup"
            elif any(token in lowered for token in ["why", "architecture", "exist", "purpose"]):
                query_type = "architecture"
            elif any(token in lowered for token in ["migrate", "migration", "plan"]):
                query_type = "migration_plan"
            elif any(token in lowered for token in ["commit", "changed", "history", "why was"]):
                query_type = "change_history"
            return self.postgres_retriever.search(query, query_type=query_type, k=k)

        semantic = self.vectorstore.similarity_search(query, k=max(k * 3, 10))
        keyword = self.bm25.invoke(query)

        scored = defaultdict(lambda: {"doc": None, "score": 0.0})

        for rank, doc in enumerate(semantic):
            key = f"{doc.metadata.get('source','')}-{hash(doc.page_content)}"
            scored[key]["doc"] = doc
            scored[key]["score"] += 1.0 / (rank + 1)

        for rank, doc in enumerate(keyword):
            key = f"{doc.metadata.get('source','')}-{hash(doc.page_content)}"
            scored[key]["doc"] = doc
            scored[key]["score"] += 1.0 / (rank + 1)

        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[:k] if item["doc"] is not None]
