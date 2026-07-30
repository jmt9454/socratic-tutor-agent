import argparse
import asyncio
import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- 1. IMPORTS FOR ASYNC ---
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import create_graph

async def run_conversation(thread_id: str | None = None):
    """
    Example of running a stateful conversation asynchronously.
    A fresh thread_id is generated unless one is supplied (to resume a prior thread).
    """
    resuming = thread_id is not None
    if not resuming:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"--- {'Resuming' if resuming else 'Starting'} Conversation (ID: {thread_id}) ---")

    # --- Learning Outcomes ---
    overall_goal = "Study Checklist: Recognizing Phishing and Deceptive Email Tactics"

    learning_outcomes = {
        "1. Spotting Deceptive Links": [
            "1. Typosquatting",
            "2. Homograph attacks",
            "3. URL Shorteners",
            "4. Methods of double-checking links"
            "5. Open Redirects",
        ],
        "2. Phishing Variants and Social Engineering": [
            "1. Phishing",
            "2. Spear Phishing",
            "3. Whaling",
            "4. Smishing",
            "3. Pretexting",
            "4. Tactics for social engineering (fear, urgency, etc.)",
        ],
        "3_Dangerous_Attachments": [
                "1. Double extensions",
                "2. Malicious Macros"
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
    parser = argparse.ArgumentParser(description="CLI test for the tutor graph.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Resume an existing thread by ID (default: generate a fresh one).",
    )
    args = parser.parse_args()

    # --- 3. RUN ASYNC LOOP ---
    asyncio.run(run_conversation(thread_id=args.thread_id))