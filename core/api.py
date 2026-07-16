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
        "1. How lookalike URLs deceive: 'homograph' attacks use foreign characters that look identical, while 'typosquatting' registers plain misspellings — and how the two tricks differ.",
        "2. How links hide their true destination: shortened links (`bit.ly/...`) show you nothing, and an 'open redirect' can start at a trusted site yet bounce you somewhere malicious.",
        "3. Why a link's visible text proves nothing about where it goes — and how to reveal the true destination (e.g., by hovering) before clicking."
    ],
    "2. Phishing Variants and Social Engineering": [
        "1. Who phishing targets: mass email blasts vs. 'spear phishing' aimed at specific individuals vs. 'whaling' aimed at executives.",
        "2. How phishing reaches you beyond email — 'smishing' (SMS) and 'vishing' (voice calls) — and how 'pretexting' (a fabricated role like IT support) plus pressure tactics (urgency, authority, fear) make these work."
    ],
    "3. Email Authentication (SPF / DKIM / DMARC)": [
        "1. What an 'SPF' record does vs. what 'DKIM' adds — who may send for a domain vs. proof the message wasn't forged — and how SPF's job differs from an MX record's.",
        "2. What 'DMARC' does when SPF or DKIM fail — and where all three records actually live."
    ],
    "4. Reading Sender Information": [
        "1. Why the visible 'From' address is the easiest thing to fake, how hidden fields like 'Return-Path' (where bounce messages go) can differ from it — and what a mismatch tells you."
    ],
    "5. Dangerous Attachments": [
        "1. How 'double extensions' (e.g., `photo.jpg.exe`) disguise executables, the OS behavior that hides the trick — and why a familiar file type or small size never means safe."
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