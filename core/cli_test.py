import asyncio
import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- 1. IMPORTS FOR ASYNC ---
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import create_graph

async def run_conversation():
    """
    Example of running a stateful conversation asynchronously.
    """
    thread_id = 'bd2a82a0-6d1a-4c5d-a0b8-24ad6a7b020ab'
    config = {"configurable": {"thread_id": thread_id}}

    print(f"--- Starting Conversation (ID: {thread_id}) ---")

    # --- Learning Outcomes ---
    overall_goal = "Introductory Asymptotic Notation"
    learning_outcomes = {
       "1. The Measurement Problem (Why Stopwatches Fail)": [
            "1. Understand that measuring code execution with physical time (seconds/milliseconds) is unreliable due to differences in computer hardware and background processes.",
            "2. Grasp the concept of **'input size'** (usually denoted as the variable $n$) and recognize that true efficiency is measured by observing how performance changes as $n$ grows.",
            "3. Shift the analytical perspective from 'how fast does this run?' to **'how many operations does this code take?'**."
        ],
        "2. Counting Operations & Rate of Growth": [
            "1. Be able to identify basic, single-step operations in code (e.g., variable assignment, basic arithmetic, true/false comparisons).",
            "2. Understand the concept of **'rate of growth'** as the direct relationship between the input size ($n$) and the total number of operations performed.",
            "3. Recognize that as data sets become massive (scaling toward infinity), the rate of growth becomes the only metric that truly matters."
        ],
        "3. Best, Worst, and Average Cases": [
            "1. Understand that an algorithm's performance can change based on the *actual data* it receives (e.g., searching for a name and finding it on the first try vs. the very last try).",
            "2. Differentiate conceptually between the Best Case (lucky scenario), Average Case (typical scenario), and Worst Case (unlucky scenario).",
            "3. Grasp ***why*** programmers primarily focus on the **Worst Case**: to guarantee the algorithm will never perform worse than a specific, predictable bound."
        ],
        "4. Big O Notation (The Core Rules)": [
            "1. Define **'Big O Notation'** as the standardized mathematical vocabulary used by engineers to describe an algorithm's worst-case time or space complexity.",
            "2. Understand the rule of **'dropping constants'**: recognize that $O(2n)$ or $O(n + 5)$ is simplified to $O(n)$ because static numbers don't significantly impact the trajectory of massive growth.",
            "3. Understand the rule of **'dropping non-dominant terms'**: recognize that in an equation like $O(n^2 + n)$, the $n^2$ dominates the growth rate as $n$ scales, simplifying the final notation to $O(n^2)$."
        ],
        "5. The Foundational Complexity Classes": [
            "1. Identify **Constant Time $O(1)$**: operations that take the exact same amount of time regardless of how large the data gets (e.g., looking up an item in a list by its exact index position).",
            "2. Identify **Linear Time $O(n)$**: operations where the time required scales 1:1 with the data (e.g., using a single loop to check every item in a list one by one).",
            "3. Identify **Quadratic Time $O(n^2)$**: operations where time scales exponentially with data, typically recognized by nested structures (e.g., an inner loop running entirely for every step of an outer loop)."
        ]
    }
    
    initial_state = {
        "overall_goal": overall_goal,
        "learning_outcomes": learning_outcomes
    }

    # --- 2. ASYNC DATABASE CONTEXT ---
    # We must connect to the database within an async context manager
    async with AsyncSqliteSaver.from_conn_string("threads.db") as checkpointer:
        
        # Compile the graph with the active checkpointer
        workflow = create_graph()
        app = workflow.compile(checkpointer=checkpointer)

        # First Call (Initial State)
        # Use 'ainvoke' (Async Invoke)
        print("Initializing...")
        response = await app.ainvoke(initial_state, config=config)
        print(f"\n[AGENT]: {response['messages'][-1].content}")

        while True:
            try:
                # Get user input (Note: input() is blocking, but acceptable for a simple CLI test)
                human_message = input("\n[USER]: ")

                if human_message.lower() in ["quit", "exit"]:
                    print("\n--- Conversation Finished ---")
                    break

                user_input = {
                    "messages": [HumanMessage(content=human_message)],
                    **initial_state 
                }

                # Invoke the app asynchronously
                response = await app.ainvoke(user_input, config=config)

                print(f"\n[AGENT]: {response['messages'][-1].content}")

            except KeyboardInterrupt:
                print("\n\n--- Conversation Interrupted ---")
                break
            except Exception as e:
                print(f"\n[ERROR]: An error occurred: {e}")
                break

if __name__ == "__main__":
    # --- 3. RUN ASYNC LOOP ---
    asyncio.run(run_conversation())