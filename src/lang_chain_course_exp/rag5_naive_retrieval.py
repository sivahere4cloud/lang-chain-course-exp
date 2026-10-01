# rag5_naive_retrieval.py
# Naive RAG: retrieve chunks from Pinecone, put them in a prompt, ask the LLM.
# Compares an answer WITHOUT retrieval vs WITH retrieval.
#
# Docs used:
#   [L2] https://docs.langchain.com/oss/python/langchain/knowledge-base   -> "Use retrievers"
#   [L2] https://docs.langchain.com/oss/python/integrations/vectorstores/pinecone
#        -> "Query vector store" -> "Query by turning into retriever"
#   [L2] https://docs.langchain.com/oss/python/langchain/retrieval        -> "2-step RAG"

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"   # MUST be the same model used at ingestion
INDEX_NAME = "medium-analyzer"
CHAT_MODEL = "gpt-5-mini"
TOP_K = 3
QUESTION = "Which conferences have hosted competitions on vector search?"

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below.\n"
    "If the context does not contain the answer, say \"I don't know based on the document.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    context = ""
    for doc in docs:
        context = context + doc.page_content + "\n\n"
    return context


def main() -> None:
    llm = ChatOpenAI(model=CHAT_MODEL)

    # --- 1) WITHOUT retrieval: the LLM answers from its training only ---
    raw_answer = llm.invoke(QUESTION)
    print("=== WITHOUT retrieval ===")
    print(raw_answer.content)
    print()

    # --- 2) Connect to the vector store (same setup as Part 4) ---
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    index = Pinecone().Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # [L2] "Use retrievers": retriever = vector_store.as_retriever(search_kwargs={"k": ...})
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # --- 3) RETRIEVE: question -> top-k chunks ---
    docs = retriever.invoke(QUESTION)
    print(f"=== Retrieved {len(docs)} chunks ===")
    for doc in docs:
        print("-", doc.metadata.get("source"), "|", doc.page_content[:70].replace("\n", " "))
    print()

    # --- 4) AUGMENT: put the chunks into the prompt ---
    context = format_docs(docs)
    prompt_value = RAG_PROMPT.invoke({"context": context, "question": QUESTION})

    # --- 5) GENERATE: the LLM answers from the context ---
    rag_answer = llm.invoke(prompt_value)
    print("=== WITH retrieval ===")
    print(rag_answer.content)


if __name__ == "__main__":
    main()


    """
    QUESTION ──► retriever.invoke ──► 3 chunks ──► format_docs ──► RAG_PROMPT ──► llm.invoke ──► answer
            (step 1: RETRIEVE)                (AUGMENT)                     (step 2: GENERATE)
    """