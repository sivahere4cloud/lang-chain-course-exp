from dotenv import load_dotenv
import os
load_dotenv()


def main() -> None:
    print("Hello from search-agent!")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    langsmith_project = os.environ.get("LANGSMITH_PROJECT")

    print("Tavily key found:", tavily_key is not None)
    print("OpenAI key found:", openai_key is not None)
    print("LangSmith project:", langsmith_project)

if __name__ == "__main__":
    main()