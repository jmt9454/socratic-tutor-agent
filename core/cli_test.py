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
            "1. Know that attackers use lookalike characters from other alphabets to disguise URLs presented to a user — a 'homograph' attack.",
            "2. Recognize how users can misspell domains and arrive at a different site('typosquatting') and how shortened links can disguise malicious sites.",
            "3. Understand that attackers can bounce links through trusted sites via redirects ('open redirect' vulnerabilities).",
            "4. Always check a link's true destination (e.g., by hovering) before clicking — the displayed text and the actual address can differ."
        ],
        "2. Phishing Variants and Social Engineering": [
            "1. Distinguish phishing by target: mass email blasts vs. targeted attacks on individuals ('spear phishing') vs. attacks on executives ('whaling').",
            "2. Distinguish phishing by channel: email vs. SMS ('smishing') vs. voice calls ('vishing').",
            "3. Recognize fabricated scenarios ('pretexting'), such as impersonating support staff, vendors, or leadership.",
            "4. Identify the pressure tactics that power these scenarios: urgency, authority, fear."
        ],
        "3. Email Authentication (SPF / DKIM / DMARC)": [
            "1. SPF (Sender Policy Framework) lists which servers may send email for a domain.",
            "2. DKIM (DomainKeys Identified Mail) adds a cryptographic signature to prove the message wasn't forged.",
            "3. DMARC (Domain-based Message Authentication, Reporting, and Conformance) tells receiving servers what to do when the other two checks fail.",
            "4. Know where these live: they're published as DNS records, alongside routing records like MX (Mail Exchange)."
        ],
        "4. Reading Sender Information": [
            "1. The visible 'From' address is trivially forgeable — never trust it alone.",
            "2. Hidden header fields can differ from the visible sender.",
            "3. The bounce/return address ('Return-Path') is one such hidden field.",
            "4. Mismatches between visible and hidden sender fields can reveal spoofing."
        ],
        "5. Dangerous Attachments": [
            "1. Watch for disguised executables using layered 'double extensions' (e.g., 'photo.jpg.exe' style tricks).",
            "2. These tricks exploit operating system settings that hide known extensions.",
            "3. A familiar-looking file type doesn't make an attachment safe.",
            "4. Small file size doesn't make an attachment safe either."
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