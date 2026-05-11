from dotenv import load_dotenv

from coding_rag.config import Settings
from coding_rag.graph import ask


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()

    print("Coding Assistant RAG")
    print("Type your question. Ctrl+C to exit.\n")

    while True:
        try:
            q = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye")
            break

        if not q:
            continue

        answer = ask(settings, q)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()
