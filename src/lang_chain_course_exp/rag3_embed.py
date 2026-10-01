# rag3_embed.py
# Embeds the chunks and a question, then scores each chunk against the question.
#
# Docs used:
#   [L2] https://docs.langchain.com/oss/python/langchain/knowledge-base
#        section: "Generate embeddings"

import math
from pathlib import Path

from dotenv import load_dotenv

# [L2] from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from lang_chain_course_exp.rag1_load import load_text_file
from lang_chain_course_exp.rag2_split import split_document

load_dotenv()

DATA_FILE = Path("mediumblog1.txt")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"
QUESTION = "What techniques are used for fast similarity search?"


def cosine_similarity(vector_a, vector_b):
    dot_product = 0.0
    length_a = 0.0
    length_b = 0.0
    for i in range(len(vector_a)):
        dot_product = dot_product + vector_a[i] * vector_b[i]
        length_a = length_a + vector_a[i] * vector_a[i]
        length_b = length_b + vector_b[i] * vector_b[i]
    return dot_product / (math.sqrt(length_a) * math.sqrt(length_b))


def main() -> None:
    document = load_text_file(DATA_FILE)
    chunks = split_document(document, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # [L2] embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 1) Embed the question (one text)
    question_vector = embeddings.embed_query(QUESTION)
    print("Question vector length:", len(question_vector))
    print("First 5 numbers:", question_vector[:5])
    print()

    # 2) Embed all chunks in ONE call (a list of texts)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.page_content)
    chunk_vectors = embeddings.embed_documents(chunk_texts)
    print("Number of chunk vectors:", len(chunk_vectors))
    print()

    # 3) Score every chunk against the question
    best_score = -1.0
    best_number = 0
    for i in range(len(chunks)):
        score = cosine_similarity(question_vector, chunk_vectors[i])
        print(f"Chunk {i + 1}: score {score:.3f} | {chunks[i].page_content[:50]!r}")
        if score > best_score:
            best_score = score
            best_number = i + 1

    print()
    print(f"Best match: chunk {best_number} (score {best_score:.3f})")


if __name__ == "__main__":
    main()