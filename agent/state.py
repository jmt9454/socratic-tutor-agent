# agent/state.py
from typing import Optional, TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

def add_string_list(
    left: list[str] | None, right: list[str] | None
) -> list[str]:
    """Adds two lists of strings together."""
    return (left or []) + (right or [])

class AgentState(TypedDict):
    """
    Attributes:
        messages: The list of messages in the conversation,
                  managed by `add_messages` to append new messages.
        current_topic: The current topic being discussed.
        current_rubric: The rubric for evaluating understanding of the current topic.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    internal_monologue: Annotated[list[AnyMessage], add_string_list]
    overall_goal: str
    learning_outcomes: dict[str, list[str]]
    remaining_topics: Optional[list[str]]
    completed_topics: Optional[Annotated[list[str], add_string_list]]
    remaining_learning_outcomes: Optional[list[str]]