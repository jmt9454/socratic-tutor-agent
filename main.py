# main.py

import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Import the compiled app from our package
from agent.graph import app #noqa: E402


def run_conversation():
    """
    Example of running a stateful conversation.
    """
    # A 'thread_id' is the "key" to a specific conversation.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": 'bd2a82a0-6d1a-4c5d-a0b8-24ad6a7b020a'}}

    print(f"--- Starting New Conversation (ID: {thread_id}) ---")

    # --- First Turn ---
    print("\n[USER]: Hello, my name is Alex.")
    user_input = {"messages": [HumanMessage(content="Hello, my name is Alex.")]}
    response = app.invoke(user_input, config=config)
    print(f"[AGENT]: {response['messages'][-1].content}")

    # --- Second Turn ---
    print("\n[USER]: What is my name?")
    user_input = {"messages": [HumanMessage(content="What is my name?")]}
    response = app.invoke(user_input, config=config)
    print(f"[AGENT]: {response['messages'][-1].content}")

    print("\n--- Conversation Finished ---")


if __name__ == "__main__":
    run_conversation()
