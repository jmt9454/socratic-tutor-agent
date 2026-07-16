import os
import secrets

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Import the Async Saver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
load_dotenv()
from graph import create_graph

# --- 0. Runtime Config (env-driven for Docker) ---
# Where the checkpointer database lives. In Docker this points at a volume
# (e.g., /data/threads.db) so student progress survives container restarts.
THREADS_DB_PATH = os.getenv("THREADS_DB_PATH", "threads.db")

# Optional shared secret. If TUTOR_API_KEY is set (non-empty), every /chat
# request must carry a matching X-API-Key header. If unset, auth is disabled
# (network isolation is then the only access control).
TUTOR_API_KEY = os.getenv("TUTOR_API_KEY", "")


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if not TUTOR_API_KEY:
        return  # auth disabled by config
    if not x_api_key or not secrets.compare_digest(x_api_key, TUTOR_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# --- 1. Global State ---
# This will hold our compiled graph once the server starts
app_graph = None

# --- 2. Lifespan Manager (Database Setup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs BEFORE the server starts receiving requests.
    It connects to the DB and compiles the graph.
    """
    print(f"--- Starting up: Connecting to Async Database ({THREADS_DB_PATH}) ---")
    print(f"--- API key auth: {'ENABLED' if TUTOR_API_KEY else 'DISABLED (no TUTOR_API_KEY set)'} ---")
    async with AsyncSqliteSaver.from_conn_string(THREADS_DB_PATH) as checkpointer:
        # 1. Create the workflow
        workflow = create_graph()
        
        # 2. Compile it with the ASYNC checkpointer
        global app_graph
        app_graph = workflow.compile(checkpointer=checkpointer)
        
        # 3. Yield control to the application
        yield
    
    print("--- Shutting down: Closing Database ---")

# --- 3. Static Data (Curriculum) ---
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

# --- 4. Models ---
class ChatInput(BaseModel):
    message: str
    thread_id: str

class ChatOutput(BaseModel):
    response: str

# --- 5. Setup FastAPI with Lifespan ---
app = FastAPI(
    title="Tutor Agent API",
    lifespan=lifespan # <--- Link the lifespan manager here
)

@app.get("/health")
async def health():
    """Lets the OpenWebUI pipe (or curl) verify the backend is up and the graph compiled."""
    return {"status": "ok", "graph_ready": app_graph is not None}


@app.post("/chat", operation_id="chat_with_tutor", dependencies=[Depends(verify_api_key)])
async def chat_endpoint(payload: ChatInput):
    # Ensure the graph is loaded
    if app_graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    config = {"configurable": {"thread_id": payload.thread_id}}
    
    graph_input = {
        "messages": [HumanMessage(content=payload.message)],
        "overall_goal": overall_goal,
        "learning_outcomes": learning_outcomes
    }

    async def generate():
        try:
            async for event in app_graph.astream_events(graph_input, config, version="v2"):
                kind = event["event"]
                
                # 1. IDENTIFY THE NODE
                # We check which node is currently doing the work
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                
                # 2. FILTER: ONLY STREAM THE "INQUISITOR"
                # We ignore 'evaluator_node', 'planner_node', etc.
                if kind == "on_chat_model_stream" and node_name == "inquisitor_node":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
                        
        except Exception as e:
            print(f"Stream Error: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)