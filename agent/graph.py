# agent/graph.py
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import our defined state and nodes
from .state import AgentState
from .nodes import call_model


def create_graph():
    """
    Creates, configures, and compiles the stateful agent graph.
    """
    # Initialize the graph
    graph_builder = StateGraph(AgentState)

    # Add the nodes
    graph_builder.add_node("call_model", call_model)
    # graph_builder.add_node("another_node", another_node) # Example

    # Define the edges
    graph_builder.set_entry_point("call_model")
    graph_builder.add_edge("call_model", END)
    # ... add conditional edges, tool logic, etc. here ...

    # Add the Checkpointer
    # The .db file will be created in the root directory
    db_path = 'threads.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)

    memory = SqliteSaver(conn)

    # Compile the graph
    return graph_builder.compile(checkpointer=memory)


# Create the final, usable app object
app = create_graph()
