# rag6a_lcel_basics.py
# The 4 LCEL building blocks, shown one at a time. No API calls, no cost.
#
# Docs used:
#   [L4] https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough
#   [L4] https://reference.langchain.com/python/langchain-core/runnables

from operator import itemgetter

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def fake_retriever(question):
    # Stands in for the real retriever in 6b -- returns text instead of Documents
    return f"[3 chunks about: {question}]"


def main() -> None:
    # --- Block 1: StrOutputParser -- AIMessage in, str out ---
    parser = StrOutputParser()
    message = AIMessage(content="SISAP and NeurIPS.")
    text = parser.invoke(message)
    print("1) StrOutputParser:", type(message).__name__, "->", type(text).__name__, "|", text)

    # --- Block 2: itemgetter -- dict in, one value out ---
    get_question = itemgetter("question")
    value = get_question({"question": "Which conferences?"})
    print("2) itemgetter:", value)

    # --- Block 3: | joins steps -- output of left becomes input of right ---
    question_to_chunks = get_question | RunnableLambda(fake_retriever)
    chunks = question_to_chunks.invoke({"question": "Which conferences?"})
    print("3) itemgetter | fake_retriever:", chunks)

    # --- Block 4: RunnablePassthrough.assign -- keep the dict, ADD a key ---
    add_context = RunnablePassthrough.assign(context=question_to_chunks)
    result = add_context.invoke({"question": "Which conferences?"})
    print("4) assign:", result)


if __name__ == "__main__":
    main()
    