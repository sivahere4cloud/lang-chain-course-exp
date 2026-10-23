# lg2_nodes.py
# The two nodes of the ReAct graph, each tested on its own (no graph yet).
#
# Course: nodes.py (branch project/ReAct-Agent-Function-Calling)
# Docs:   [L2] https://docs.langchain.com/oss/python/langgraph/quickstart
#              "3. Define model node", "4. Define tool node"

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState

from lang_chain_course_exp.lg1_tools import LLM_WITH_TOOLS, TOOLS

SYSTEM_MESSAGE = "You are a helpful assistant that can use tools to answer questions."


# --- Node 1: agent_reason (= "3. Define model node") ---
def run_agent_reasoning(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    for message in state["messages"]:
        messages.append(message)

    response = LLM_WITH_TOOLS.invoke(messages)
    return {"messages": [response]}


# --- Node 2: act (= "4. Define tool node") ---
def run_tools(state: MessagesState) -> dict:
    tools_by_name = {}
    for t in TOOLS:
        tools_by_name[t.name] = t

    last_message = state["messages"][-1]

    results = []
    for tool_call in last_message.tool_calls:
        tool_to_run = tools_by_name[tool_call["name"]]
        observation = tool_to_run.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": results}


def main() -> None:
    question = "What is the temperature in Tokyo? List it and then triple it"

    # Test node 1: give it a state with just the question
    state = {"messages": [HumanMessage(content=question)]}
    update_1 = run_agent_reasoning(state)
    ai_message = update_1["messages"][0]
    print("Node 1 returned:", type(ai_message).__name__)
    print("  tool_calls:", ai_message.tool_calls)
    print()

    # Merge the update by hand (what the add_messages reducer will do in the graph)
    state["messages"].append(ai_message)

    # Test node 2: give it the state whose LAST message has the tool call
    update_2 = run_tools(state)
    for tool_message in update_2["messages"]:
        print("Node 2 returned:", type(tool_message).__name__)
        print("  answers call id:", tool_message.tool_call_id)
        print("  content (first 200 chars):", tool_message.content[:200])


if __name__ == "__main__":
    main()