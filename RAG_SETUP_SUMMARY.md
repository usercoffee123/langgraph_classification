# Coding RAG Setup Summary

Date: 2026-05-11

## Goal
Set up a Coding Assistant RAG that can index a local repository, retrieve relevant context, and answer engineering questions.

## What Was Added

- Core coding RAG package with ingestion, retrieval, and graph orchestration.
- Local vector mode using Chroma.
- Postgres plus pgvector mode with multi-strategy retrieval in one database.
- Ephemeral Docker script for Postgres startup and teardown.

## Files Added

- coding_rag/postgres_backend.py
- coding_rag/__init__.py
- coding_rag/config.py
- coding_rag/ingest.py
- coding_rag/retrieval.py
- coding_rag/graph.py
- build_index.py
- ask_rag.py
- scripts/with_pgvector.sh

## Files Updated

- requirements.txt
- .env.example

## Retrieval Model Implemented

Single datastore approach with Postgres plus pgvector:

1. Semantic retrieval
- pgvector nearest-neighbor search over embedded code chunks.

2. Keyword retrieval
- Postgres full-text search ranking over chunk content.

3. Symbol retrieval
- Structured symbols table populated from lightweight source parsing.

4. Git-aware retrieval
- Commits table indexed from local git history and searchable with full-text.

Result sets are fused and reranked before answer synthesis.

## Important Fixes Applied During Setup

- Added non-git repository fallback so indexing does not fail outside a git repo.
- Added file size and directory exclusions to avoid indexing giant data and build artifacts.
- Added embedding batch-size control to reduce OpenAI rate-limit spikes.
- Fixed script handling for command prefixes like RAG_REPO_PATH=... .
- Fixed Postgres vector dimension issue by defaulting Postgres backend to text-embedding-3-small.
- Fixed SQL vector operator binding with explicit cast to vector.
- Added Docker volume persistence so ephemeral containers can share indexed data across runs.

## How To Run (Chroma Local Mode)

From the langchain project directory:

    RAG_BACKEND=chroma RAG_REPO_PATH=/Users/ccastorena/Documents/repos/profiling .venv/bin/python build_index.py
    RAG_BACKEND=chroma RAG_REPO_PATH=/Users/ccastorena/Documents/repos/profiling .venv/bin/python ask_rag.py

## How To Run (Postgres Ephemeral Docker Mode)

From the langchain project directory:

    scripts/with_pgvector.sh RAG_REPO_PATH=/Users/ccastorena/Documents/repos/profiling .venv/bin/python build_index.py
    scripts/with_pgvector.sh RAG_REPO_PATH=/Users/ccastorena/Documents/repos/profiling .venv/bin/python ask_rag.py

## Optional: Persist Postgres Data In Home Directory

You can store Postgres data in a host folder instead of a Docker-managed named volume by setting DATA_DIR.

Example using ~/rag_pg_data:

    DATA_DIR=~/rag_pg_data scripts/with_pgvector.sh RAG_REPO_PATH=/Users/ccastorena/Documents/repos/profiling .venv/bin/python build_index.py
    DATA_DIR=~/rag_pg_data scripts/with_pgvector.sh RAG_REPO_PATH=/Users/ccastorena/Documents/repos/profiling .venv/bin/python ask_rag.py

Important: Use the same DATA_DIR value on later runs so the same index is reused.

## What Was Verified

- Docker availability verified.
- Profiling repo indexed successfully in Postgres mode.
- Query execution succeeded end-to-end and returned repository-specific answers with citations.

## Notes

- If you switch repositories, rerun build_index.py for the new RAG_REPO_PATH.
- Chroma mode is simplest for local use with no database server.
- Postgres mode is best when you want semantic, keyword, symbol, and git-aware retrieval from one database.
- The Docker wrapper also supports DATA_DIR for host-path persistence and VOLUME_NAME for Docker named-volume persistence.
