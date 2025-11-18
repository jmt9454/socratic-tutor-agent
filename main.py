# main.py

from fastapi import FastAPI
import uvicorn
from api.v1.endpoints import router as api_router

# Initialize the FastAPI application
app = FastAPI(
    title="Beginner Theory Learning API",
    description="An API service to deliver learning theory outcomes.",
    version="1.0.0"
)

# Include the router containing all V1 endpoints
app.include_router(api_router, prefix="/api/v1")

# --- Optional: Define a root health check endpoint ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running. Access endpoints via /api/v1"}

if __name__ == "__main__":
    # To run the server, execute this from your terminal:
    # uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)