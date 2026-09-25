from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


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
    return "Tokyo weather is sunny"


def main() -> None:
    llm = ChatOpenAI(model="gpt-5-mini")
    tools = [search]
    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )

    print("Type of result:", type(result))
    print("Keys:", result.keys())

    for message in result["messages"]:
        print("-----")
        print("Message type:", type(message).__name__)
        print("Content:", message.content)

        if type(message).__name__ == "AIMessage":
            print("Tool calls:", message.tool_calls)

        if type(message).__name__ == "ToolMessage":
            print("Answers tool call id:", message.tool_call_id)

if __name__ == "__main__":
    main()