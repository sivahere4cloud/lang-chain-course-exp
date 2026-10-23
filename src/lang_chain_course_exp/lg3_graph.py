# lg3_graph.py
# The ReAct agent as a LangGraph graph: state + nodes + edges -> compile -> draw -> invoke.
#
# Course: main.py (branch project/ReAct-Agent-Function-Calling)
# Docs:   [L2] https://docs.langchain.com/oss/python/langgraph/quickstart
#              "5. Define logic to determine whether to end", "6. Build agent and compile"
#         [L2] https://docs.langchain.com/oss/python/langgraph/graph-api
#              "Edges -> Conditional edges", "Visualization"

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from lang_chain_course_exp.lg2_nodes import run_agent_reasoning, run_tools

AGENT_REASON = "agent_reason"
ACT = "act"
QUESTION = "What is the temperature in Tokyo? List it and then triple it"


# --- Router for the conditional edge (= "5. Define logic to determine whether to end") ---
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return ACT
    return END


# --- Build the graph (= "6. Build agent and compile") ---
def build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node(AGENT_REASON, run_agent_reasoning)   # box 1
    graph.add_node(ACT, run_tools)                       # box 2

    graph.add_edge(START, AGENT_REASON)                  # __start__ -> agent_reason
    graph.add_conditional_edges(AGENT_REASON, should_continue, [ACT, END])  # dotted arrows
    graph.add_edge(ACT, AGENT_REASON)                    # act -> agent_reason (the loop)

    return graph.compile()


def main() -> None:
    app = build_graph()

    # 1) Draw the graph as Mermaid text (works offline)
    print(app.get_graph().draw_mermaid())

    # 2) Save it as a picture (uses an online service, so it may fail offline)
    try:
        app.get_graph().draw_mermaid_png(output_file_path="flow.png")
        print("Saved flow.png")
    except Exception as error:
        print("Could not draw flow.png:", error)

    # 3) Run the graph
    result = app.invoke({"messages": [HumanMessage(content=QUESTION)]})

    print("\n=== Every message in the final state ===")
    for message in result["messages"]:
        print("-", type(message).__name__, "| tool_calls:", getattr(message, "tool_calls", None))

    print("\n=== Final answer ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()