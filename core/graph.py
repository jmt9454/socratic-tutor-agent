from dotenv import load_dotenv
load_dotenv()

import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import our defined state and nodes
from state import AgentState
from nodes import router_node, planner_node, inquisitor_node, evaluator_node, arc_planner_node


def create_graph():
    """
    Creates, configures, and compiles the stateful agent graph.
    """
    print(AgentState)
    # Initialize the graph
    graph_builder = StateGraph(AgentState)

    # Add the nodes
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("planner_node", planner_node)
    graph_builder.add_node("arc_planner_node", arc_planner_node)
    graph_builder.add_node("inquisitor_node", inquisitor_node)
    graph_builder.add_node("evaluator_node", evaluator_node)

    def after_router_node(state: AgentState):
        if not state.get("remaining_topics"):
            return "planner_node"
        return "evaluator_node"

    def after_planner_node(state: AgentState):
        if state.get("remaining_topics") == []:
            return "END"
        return "arc_planner_node"

    def after_evaluator_node(state: AgentState):
        if state.get("remaining_learning_outcomes") == []:
            return "planner_node"
        return "arc_planner_node"

    def after_arc_planner_node(state: AgentState):
        return "inquisitor_node"

    def after_inquisitor_node(state: AgentState):
        return "END"

    # Define the edges
    graph_builder.set_entry_point("router_node")
    graph_builder.add_conditional_edges("router_node", after_router_node, {"planner_node":"planner_node","evaluator_node":"evaluator_node"})
    graph_builder.add_conditional_edges("planner_node", after_planner_node, {"arc_planner_node":"arc_planner_node", "END": END})
    graph_builder.add_conditional_edges("arc_planner_node", after_arc_planner_node, {"inquisitor_node":"inquisitor_node"})
    graph_builder.add_conditional_edges("inquisitor_node", after_inquisitor_node, {"END":END})
    graph_builder.add_conditional_edges("evaluator_node", after_evaluator_node, {"planner_node":"planner_node","arc_planner_node":"arc_planner_node"})

    # Checkpointer
    # The threads.db file will be created in the root directory

    # Compile the graph
    return graph_builder


# Create the final, usable app object
app = create_graph()

# Print the Mermaid code
png = app.compile().get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
