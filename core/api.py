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