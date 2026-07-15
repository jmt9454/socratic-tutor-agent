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
    overall_goal = "Introductory Asymptotic Notation"
    learning_outcomes = {
        "1. Deceptive Links and URL Manipulation": [
            "1. Understand the mechanics of **Homograph attacks** and how attackers use visually similar characters from different alphabets to spoof legitimate URLs.",
            "2. Identify the concept of **Typosquatting** and how attackers exploit common domain misspellings to trap inattentive users.",
            "3. Recognize the security purpose of hovering over hyperlinks to reveal the true destination, guarding against mismatches between display text and actual web addresses.",
            "4. Comprehend how **URL shorteners** can be weaponized to obscure malicious destinations from victims.",
            "5. Define an **Open Redirect vulnerability** and explain how attackers use trusted domains to silently forward users to malicious sites."
        ],
        "2. Social Engineering and Phishing Variations": [
            "1. Differentiate between mass phishing and **Spear Phishing**, which utilizes targeted intelligence against specific individuals or organizations.",
            "2. Identify **Whaling** as a specialized phishing campaign aimed strictly at high-ranking executives or privileged users.",
            "3. Understand **Smishing** as the application of social engineering tactics through Short Message Service (SMS) channels.",
            "4. Grasp the concept of **Pretexting**, where attackers fabricate elaborate scenarios (e.g., posing as IT support) to manipulate victims into divulging credentials."
        ],
        "3. Domain and Email Authentication Standards": [
            "1. Understand the role of **SPF (Sender Policy Framework)** DNS records in specifying which mail servers are explicitly authorized to send emails on behalf of a domain.",
            "2. Comprehend how **DKIM (DomainKeys Identified Mail)** mitigates spoofing by attaching a cryptographic digital signature to the email header.",
            "3. Define the purpose of **DMARC (Domain-based Message Authentication, Reporting, and Conformance)** in instructing receiving servers on how to handle messages that fail SPF or DKIM checks."
        ],
        "4. Anatomy of an Email and Header Spoofing": [
            "1. Recognize the **'From' address** field as a primary target for manipulation because it is easily spoofed to mimic trusted sources.",
            "2. Understand the technical function of the **'Return-Path'** header, which indicates where automated bounce messages should be routed, distinct from the visible sender."
        ],
        "5. Malicious Payloads and Attachments": [
            "1. Identify common technical indicators of malicious email attachments, particularly the use of **double extensions** (e.g., 'invoice.pdf.exe') designed to trick users into executing malware."
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