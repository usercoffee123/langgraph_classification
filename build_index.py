from dotenv import load_dotenv

from coding_rag.config import Settings
from coding_rag.ingest import ingest
from coding_rag.postgres_backend import index_postgres_repo


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    if settings.backend == "postgres":
        count = index_postgres_repo(settings)
    else:
        count = ingest(settings)
    print(f"Indexed {count} chunks from {settings.repo_path}")
    if settings.backend == "postgres":
        print(f"Persisted Postgres index to {settings.database_url}")
    else:
        print(f"Persisted Chroma collection to {settings.persist_dir}")


if __name__ == "__main__":
    main()
