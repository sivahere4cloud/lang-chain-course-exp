from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """A source used by the agent to answer."""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """The agent's final answer, with its sources."""

    answer: str = Field(description="The agent's answer to the query")
    sources: list[Source] = Field(description="List of sources used to generate the answer")


def main() -> None:
    llm = ChatOpenAI(model="gpt-5-mini")
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    question = (
        "Search for 3 job postings for an AI engineer using LangChain "
        "in London on LinkedIn and list their details."
    )
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    print("Keys:", result.keys())

    structured = result["structured_response"]
    print("Type:", type(structured).__name__)
    print("-----")
    print("Answer:", structured.answer)
    print("-----")
    print("Number of sources:", len(structured.sources))

    for source in structured.sources:
        print("Source:", source.url)


if __name__ == "__main__":
    main()