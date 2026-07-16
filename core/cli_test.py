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
    overall_goal = "Recognizing Phishing and Deceptive Email Tactics"
    learning_outcomes = {
        "1. Spotting Deceptive Links": [
            "1. What a 'homograph' attack is — and how a URL can look exactly right yet lead somewhere else entirely.",
            "2. What 'typosquatting' is, and why one wrong letter can land you on an attacker's site.",
            "3. What shortened links (`bit.ly/...`) change about what you can tell from a URL.",
            "4. What an 'open redirect' vulnerability is, and why a link starting at a trusted site isn't automatically safe.",
            "5. How to check where a link really goes before clicking — and why the visible link text isn't enough."
        ],
        "2. Phishing Variants and Social Engineering": [
            "1. The difference between mass phishing, 'spear phishing', and 'whaling' — and who each one targets.",
            "2. The difference between 'smishing' and 'vishing' — and the channel each one uses to reach you.",
            "3. What 'pretexting' is, and the kinds of roles attackers impersonate to pull it off.",
            "4. The psychological pressure tactics that make these scams work on people."
        ],
        "3. Email Authentication (SPF / DKIM / DMARC)": [
            "1. What an 'SPF' record does — and how its job differs from an MX record's.",
            "2. What 'DKIM' adds to an email, and what it does — and doesn't — protect.",
            "3. What 'DMARC' does when the other two checks fail.",
            "4. Where SPF, DKIM, and DMARC records actually live."
        ],
        "4. Reading Sender Information": [
            "1. Which part of an email's sender information is easiest to fake — and why you can't trust it alone.",
            "2. How the sender you see displayed can differ from what the raw headers record.",
            "3. What the 'Return-Path' field is actually for.",
            "4. What a mismatch between visible and hidden sender fields can tell you."
        ],
        "5. Dangerous Attachments": [
            "1. How 'double extensions' disguise dangerous files as harmless ones.",
            "2. The operating-system behavior that makes the double-extension trick work.",
            "3. Whether a familiar-looking file type means an attachment is safe.",
            "4. Whether a small file size tells you anything about attachment safety."
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