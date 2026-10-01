# rag2_split.py
# Splits the loaded Document into chunks and shows how chunk_size changes the result.
#
# Docs used:
#   [L2] https://docs.langchain.com/oss/python/langchain/knowledge-base
#        section: "Load and split a PDF" (splitting part)

from pathlib import Path

# [L2] from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Our own function from Part 1, reused (not copied)
from lang_chain_course_exp.rag1_load import load_text_file

DATA_FILE = Path("mediumblog1.txt")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def split_document(document, chunk_size, chunk_overlap):
    # [L2] RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # [L2] splitter.split_documents([...]) takes a LIST of Documents
    chunks = splitter.split_documents([document])
    return chunks


def main() -> None:
    document = load_text_file(DATA_FILE)

    big_chunks = split_document(document, chunk_size=4000, chunk_overlap=CHUNK_OVERLAP)
    print("With chunk_size=4000 ->", len(big_chunks), "chunk(s)")

    chunks = split_document(document, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"With chunk_size={CHUNK_SIZE} ->", len(chunks), "chunks")
    print()

    chunk_number = 1
    for chunk in chunks:
        print(f"--- Chunk {chunk_number} ---")
        print("Length:", len(chunk.page_content))
        print("Metadata:", chunk.metadata)
        print("Starts with:", chunk.page_content[:80])
        chunk_number = chunk_number + 1


if __name__ == "__main__":
    main()