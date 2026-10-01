# rag6_lcel_chain.py
# 2-step RAG as ONE LCEL chain: retrieve -> format -> prompt -> LLM -> str.
#
# Docs used:
#   [L4] https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough
#   [L4] https://reference.langchain.com/python/langchain-core/runnables
#   [L2] https://docs.langchain.com/oss/python/langchain/knowledge-base  -> "Use retrievers"

from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from lang_chain_course_exp.rag5_naive_retrieval import RAG_PROMPT, format_docs

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "medium-analyzer"
CHAT_MODEL = "gpt-5-mini"
TOP_K = 3
QUESTION = "Which conferences have hosted competitions on vector search?"


def build_rag_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    index = Pinecone().Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=CHAT_MODEL)

    # Step A: question -> chunks -> one context string
    retrieval_part = itemgetter("question") | retriever | RunnableLambda(format_docs)

    # Step B: keep {"question"}, ADD "context", then prompt -> llm -> plain text
    rag_chain = (
        RunnablePassthrough.assign(context=retrieval_part)
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main() -> None:
    rag_chain = build_rag_chain()

    # 1) invoke: wait for the whole answer
    answer = rag_chain.invoke({"question": QUESTION})
    print("=== invoke ===")
    print(type(answer).__name__, "|", answer)
    print()

    # 2) stream: print the answer piece by piece as it is generated
    print("=== stream ===")
    for piece in rag_chain.stream({"question": QUESTION}):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()