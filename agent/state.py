# agent/state.py
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The shared state for the agent, acting as its "memory."

    Attributes:
        messages: The list of messages in the conversation,
                  managed by `add_messages` to append new messages.
    """

    messages: Annotated[list[AnyMessage], add_messages]
