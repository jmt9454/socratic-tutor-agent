import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Import the Async Saver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
load_dotenv()
# Import your workflow builder (NOT the compiled app)
from graph import create_graph

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
    print("--- 🚀 Starting up: Connecting to Async Database ---")
    async with AsyncSqliteSaver.from_conn_string("threads.db") as checkpointer:
        # 1. Create the workflow
        workflow = create_graph()
        
        # 2. Compile it with the ASYNC checkpointer
        global app_graph
        app_graph = workflow.compile(checkpointer=checkpointer)
        
        # 3. Yield control to the application
        yield
    
    print("--- Shutting down: Closing Database ---")

# --- 3. Static Data (Curriculum) ---
OVERALL_GOAL = "Nested Loops"
LEARNING_OUTCOMES =     learning_outcomes = {
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

# --- 4. Models ---
class ChatInput(BaseModel):
    message: str
    thread_id: str

class ChatOutput(BaseModel):
    response: str

# --- 5. Setup FastAPI with Lifespan ---
app = FastAPI(
    title="Tutor Agent API",
    version="1.5.0",
    lifespan=lifespan # <--- Link the lifespan manager here
)

@app.post("/chat", operation_id="chat_with_tutor")
async def chat_endpoint(payload: ChatInput):
    # Ensure the graph is loaded
    if app_graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    config = {"configurable": {"thread_id": payload.thread_id}}
    
    graph_input = {
        "messages": [HumanMessage(content=payload.message)],
        "overall_goal": OVERALL_GOAL,
        "learning_outcomes": LEARNING_OUTCOMES
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