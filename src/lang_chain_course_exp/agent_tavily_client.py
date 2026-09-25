from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.

    Args:
        query: The query to search for.

    Returns:
        The search result.
    """
    print(f"Searching for {query}")
    response = tavily.search(query=query)
    return str(response)


def main() -> None:
    llm = ChatOpenAI(model="gpt-5-mini")
    tools = [search]
    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in London?")]}
    )

    for message in result["messages"]:
        print("-----")
        print("Message type:", type(message).__name__)

        if type(message).__name__ == "AIMessage":
            print("Tool calls:", message.tool_calls)
            print("Content:", message.content)

        if type(message).__name__ == "ToolMessage":
            print("Tool result length:", len(message.content))


if __name__ == "__main__":
    main()