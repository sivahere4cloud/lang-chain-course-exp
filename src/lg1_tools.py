# lg1_tools.py
# The agent's tools and the LLM with tools bound. Tested here WITHOUT a graph.
#
# Course: react.py (branch project/ReAct-Agent-Function-Calling)
# Docs:   [L2] https://docs.langchain.com/oss/python/langgraph/quickstart -> "Define tools and model"

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

MODEL = "openai:gpt-5-mini"


@tool
def triple(num: float) -> float:
    """Multiply a number by 3.

    Args:
        num: The number to triple.

    Returns:
        The number multiplied by 3.
    """
    return float(num) * 3


TOOLS = [TavilySearch(max_results=1), triple]

LLM_WITH_TOOLS = init_chat_model(MODEL).bind_tools(TOOLS)


def main() -> None:
    # 1) What tools does the LLM know about?
    for t in TOOLS:
        print("Tool:", t.name)
    print()

    # 2) Run our own tool directly (no LLM)
    print("triple(4) =", triple.invoke({"num": 4}))
    print()

    # 3) Ask the LLM ONE time -- it should ask for a tool, not answer
    question = "What is the temperature in Tokyo? List it and then triple it"
    ai_message = LLM_WITH_TOOLS.invoke(question)
    print("Content:", repr(ai_message.content))
    for tool_call in ai_message.tool_calls:
        print("Tool call ->", tool_call["name"], tool_call["args"])


if __name__ == "__main__":
    main()