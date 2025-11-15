# agent/nodes.py
from langchain_openai import ChatOpenAI
from .state import AgentState  # Import state from the same package

# --- Model Initialization ---
# It's common to initialize your model here so all nodes can share it.
# Or, you could pass it into your node functions if you need more flexibility.
model = ChatOpenAI(model="gpt-4o-mini")

# --- Node Functions ---


def call_model(state: AgentState):
    """
    A simple node that calls the LLM with the current message history.
    """
    messages = state["messages"]
    response = model.invoke(messages)

    # Always return a dictionary matching a part of the AgentState
    return {"messages": [response]}


def another_node(state: AgentState):
    """
    (Example of a future node)
    This node could call a tool, update a counter, etc.
    """
    # ... logic for this node ...
    print("--- Executing Another Node ---")
    return {}  # This node doesn't add messages
