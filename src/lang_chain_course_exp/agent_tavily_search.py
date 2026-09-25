from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


def main()-> None:
    search_tool = TavilySearch()

    print("Tool name:", search_tool.name)
    print("Tool args:", list(search_tool.args.keys()))

    llm = ChatOpenAI(model="gpt-5-mini")
    tools = [search_tool]
    agent = create_agent(model=llm, tools=tools)

    question = (
        "Search for 3 job postings for an AI engineer using LangChain "
        "in London on LinkedIn and list their details."
    )
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    for message in result["messages"]:
        print("-----")
        print("Message type:", type(message).__name__)

        if type(message).__name__ == "AIMessage":
            for tool_call in message.tool_calls:
                print("Tool call args:", tool_call["args"])
            print("Content:", message.content)

        if type(message).__name__ == "ToolMessage":
            print("Tool result length:", len(message.content))


if __name__ == "__main__":
    main()