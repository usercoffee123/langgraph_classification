from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import subprocess

from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


EXCLUDE_DIRS = {
    ".git",
    ".rag",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

MAX_FILE_BYTES = 2 * 1024 * 1024

ALLOWED_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".sql",
    ".sh",
}


def discover_files(repo_path: Path) -> Iterable[Path]:
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def get_file_git_metadata(repo: Repo | None, file_path: Path) -> dict:
    if repo is None:
        return {"last_commit": "", "last_author": "", "last_modified": ""}

    rel = file_path.relative_to(Path(repo.working_tree_dir))
    rel_str = str(rel)

    try:
        commits = list(repo.iter_commits(paths=rel_str, max_count=1))
        if commits:
            c = commits[0]
            return {
                "last_commit": c.hexsha,
                "last_author": c.author.name,
                "last_modified": str(c.committed_datetime),
            }
    except Exception:
        pass

    return {"last_commit": "", "last_author": "", "last_modified": ""}


def get_recent_commit_summary(repo_path: Path, limit: int = 200) -> List[Document]:
    cmd = [
        "git",
        "-C",
        str(repo_path),
        "log",
        f"--max-count={limit}",
        "--pretty=format:%H\t%an\t%ad\t%s",
        "--date=short",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return []

    docs: List[Document] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", maxsplit=3)
        if len(parts) != 4:
            continue
        sha, author, date, subject = parts
        docs.append(
            Document(
                page_content=f"Commit {sha}: {subject}",
                metadata={
                    "source": "git_log",
                    "sha": sha,
                    "author": author,
                    "date": date,
                    "kind": "git_commit",
                },
            )
        )
    return docs


def build_documents(settings: Settings) -> List[Document]:
    try:
        repo: Repo | None = Repo(settings.repo_path)
    except (InvalidGitRepositoryError, NoSuchPathError):
        repo = None

    docs: List[Document] = []

    for file_path in discover_files(settings.repo_path):
        text = safe_read(file_path)
        if not text.strip():
            continue

        rel = str(file_path.relative_to(settings.repo_path))
        ext = file_path.suffix.lower().lstrip(".")
        git_meta = get_file_git_metadata(repo, file_path)

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": rel,
                    "path": rel,
                    "language": ext,
                    "kind": "file",
                    **git_meta,
                },
            )
        )

    if repo is not None:
        docs.extend(get_recent_commit_summary(settings.repo_path, limit=300))
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\nclass ", "\ndef ", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(settings: Settings) -> int:
    settings.persist_dir.mkdir(parents=True, exist_ok=True)

    docs = build_documents(settings)
    chunks = split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        chunk_size=settings.embedding_batch_size,
    )

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir),
    )

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir),
    )

    vectorstore.add_documents(chunks)
    return len(chunks)
