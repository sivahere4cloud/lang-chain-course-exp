# rag4_ingest.py
# Ingestion: load -> split -> embed -> store in Pinecone. Safe to re-run.
#
# Docs used:
#   [L2] https://docs.langchain.com/oss/python/integrations/vectorstores/pinecone
#        sections: "Initialization", "Manage vector store -> Add items to vector store"

import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# [L2] "Initialization": from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from lang_chain_course_exp.rag1_load import load_text_file
from lang_chain_course_exp.rag2_split import split_document

load_dotenv()

DATA_FILE = Path("mediumblog1.txt")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "medium-analyzer"


def make_chunk_ids(chunks):
    # NOT from docs: the docs use random uuid4() IDs.
    # Fixed IDs like "mediumblog1.txt-chunk-1" make re-runs overwrite, not duplicate.
    ids = []
    number = 1
    for chunk in chunks:
        source = chunk.metadata["source"]
        ids.append(f"{source}-chunk-{number}")
        number = number + 1
    return ids


def main() -> None:
    # 1) Load and split (Parts 1 and 2)
    document = load_text_file(DATA_FILE)
    chunks = split_document(document, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print("Chunks to store:", len(chunks))

    # 2) Embeddings client (Part 3) -- no API call yet
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 3) Connect to the existing index (Part 0 created it)
    # [L2] "Initialization": index = pc.Index(index_name)
    pc = Pinecone()
    index = pc.Index(INDEX_NAME)

    stats_before = index.describe_index_stats()
    print("Vectors in index BEFORE:", stats_before.total_vector_count)

    # 4) Wrap the index in LangChain's vector store
    # [L2] "Initialization": PineconeVectorStore(index=index, embedding=embeddings)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # 5) Embed + store all chunks
    # [L2] "Add items to vector store": vector_store.add_documents(documents=..., ids=...)
    ids = make_chunk_ids(chunks)
    stored_ids = vector_store.add_documents(documents=chunks, ids=ids)
    print("Stored:", len(stored_ids), "records")
    print("First 3 IDs:", stored_ids[:3])

    # 6) Pinecone updates its counts shortly after a write, so wait a moment
    time.sleep(5)
    stats_after = index.describe_index_stats()
    print("Vectors in index AFTER:", stats_after.total_vector_count)


if __name__ == "__main__":
    main()