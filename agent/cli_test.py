# cli_test.py

import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Import the compiled app from our package
from graph import app #noqa: E402


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
            "Variables & Concepts": [
                "1. Understand what a **'variable'** is and how it acts as a **label or placeholder** for a piece of data stored in memory.",
                "2. Grasp ***why*** programmers use variables (e.g., to make code reusable, easier to read, and manage complex values).",
                "3. Understand the concept of **'assignment'** as the action of storing a value into a variable.",
                "4. Recognize that **naming conventions** are essential for writing professional and clean code."
            ],
            "Core Data Types": [
                "1. Understand the fundamental concept of a **'data type'** and its importance in defining what kind of data a variable holds (numbers, text, etc.).",
                "2. Be able to describe the four core types based on their content: **Integers** (whole numbers), **Floats** (numbers with decimal parts), **Strings** (text/characters), and **Booleans** (True/False states).",
                "3. Understand the theoretical process of **'type casting'** or **conversion**—the idea of changing a value's data type to use it in a different context (e.g., treating a number as text, or vice-versa)."
            ],
            "Iteration & Loops": [
                "1. Grasp the core idea of **'iteration'** (looping) and understand *why* it is the primary method for **automation** and performing repetitive tasks efficiently.",
                "2. Define an **'iterable'** as any structure or collection of data that can be processed one item at a time (e.g., a list of items, a sequence of characters in a word).",
                "3. Understand the concept of a **'loop variable'** as the temporary name given to the current item being processed during an iteration."
            ],
            "Nested Structures": [
                "1. Understand the theoretical concept of a **'nested loop'**—simply a loop contained entirely within the body of another loop.",
                "2. Understand the relationship between the loops: the **'inner' loop completes all its cycles** for every single cycle of the **'outer' loop**.",
                "3. Recognize ***why*** nested loops are necessary for working with **two-dimensional (2D) data** (data organized in rows and columns, like a grid or matrix)."
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
