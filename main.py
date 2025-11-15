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

    print(f"--- Starting New Conversation (ID: {'bd2a82a0-6d1a-4c5d-a0b8-24ad6a7b020a'}) ---")

    # --- Learning Outcomes ---
    overall_goal = "Nested Loops"
    learning_outcomes = {
        "Variables & Assignment": [
            "1. Understand what a 'variable' is and how it acts as a label for a piece of data in memory.",
            "2. Grasp *why* programmers use variables (like making code easier to read and manage).",
            "3. Know how to write an 'assignment' statement and spot the three parts: the `name`, the `=` operator, and the `value`.",
            "4. Learn the 'do's and 'don'ts' of naming variables in Python (like using `snake_case`)."
        ],
        "Core Data Types": [
            "1. Understand the concept of a 'data type' (and why it's so important for what you can *do* with a variable).",
            "2. Be able to spot and describe the four core types: `int` (whole numbers), `float` (decimals), `str` (text), and `bool` (True/False).",
            "3. Learn how (and why) you'd 'cast' or convert a value from one type to another (like turning the text `'5'` into the number `5`).",
            "4. Know how to use the `type()` function to ask Python what data type a variable is."
        ],
        "Iteration with 'For' Loops": [
            "1. Grasp the core idea of 'iteration' (looping) and why it's a programmer's best tool for automation.",
            "2. Understand what an 'iterable' is (Hint: it's anything you can loop over, like a list or a string).",
            "3. Know how to write a `for` loop to process items in a `list`, `string`, or a `range()`.",
            "4. Be able to explain the role of the 'loop variable' (the temporary name you give to each item as you loop)."
        ],
        "Nested Loops": [
            "1. Understand what a 'nested loop' is (simply a loop inside another loop).",
            "2. Be able to trace the 'execution flow'—how the 'inner' loop runs all its cycles for *each single cycle* of the 'outer' loop.",
            "3. See *why* nested loops are the perfect tool for working with 2D data (like a grid, matrix, or a list of lists).",
            "4. Know how to write a nested loop to access every single item in a 2D list."
        ]
    }
    
    initial_state = {
    "overall_goal": overall_goal,
    "learning_outcomes": learning_outcomes
    }

    response = app.invoke(initial_state, config=config)
    print(f"\n[AGENT]: {response['messages'][-1].content}")
    while True:
        try:
            # Get user input from the console
            human_message = input("\n[USER]: ")

            # Check for an exit command
            if human_message.lower() in ["quit", "exit"]:
                print("\n--- Conversation Finished ---")
                break

            # Prepare the input for the graph
            # Note: We only pass the *new* message. The checkpointer
            # will handle loading all the old ones.
            user_input = {
                "messages": [HumanMessage(content=human_message)],
                **initial_state  # Pass the static curriculum info
            }

            # Invoke the app
            response = app.invoke(user_input, config=config)

            # Print the agent's last response
            print(f"\n[AGENT]: {response['messages'][-1].content}")

        except KeyboardInterrupt:
            print("\n\n--- Conversation Interrupted ---")
            break
        except Exception as e:
            print(f"\n[ERROR]: An error occurred: {e}")
            break


if __name__ == "__main__":
    run_conversation()
