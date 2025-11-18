import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import upload, query



# --- Startup and Shutdown Events ---
@asynccontextmanager
async def lifespan(app):
    # Startup actions
    print("Starting up the Rag-based Data Assistant Backend...")
    yield
    # Shutdown actions
    print("Shutting down the Rag-based Data Assistant Backend...")

app = FastAPI(
    lifespan=lifespan,
    title = "Rag-based Data Assistant API",
    description = "An API for uploading documents and querying them using RAG techniques.",
    version = "1.0.0"
)

# cors settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# including api routes
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])

# --- Root Route ---
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Rag-based Data Assistant API. Use /api/v1/upload to upload documents and /api/v1/ask to query them.",
        "docs": "/docs",
        "status": "API is running"
    }

# --- Health Check Route ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "The Rag-based Data Assistant API is operational."
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)