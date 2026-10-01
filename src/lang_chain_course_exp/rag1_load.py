from pathlib import Path
from langchain_core.documents import Document

DATA_FILE = Path("mediumblog1.txt")


def load_text_file(file_path: Path) -> Document:
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path.resolve()}")

    text = file_path.read_text(encoding="utf-8")

    # [L2] "Create documents": Document(page_content=..., metadata={...})
    document = Document(
        page_content=text,
        metadata={"source": file_path.name},
    )
    return document


def main() -> None:
    document = load_text_file(DATA_FILE)

    print("Type:", type(document).__name__)
    print("Metadata:", document.metadata)
    print("Number of characters:", len(document.page_content))
    print("-----")
    print("First 300 characters:")
    print(document.page_content[:300])


if __name__ == "__main__":
    main()