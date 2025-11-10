import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from app.core.llm import get_llm_response
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Request schema
class QueryRequest(BaseModel):
    query: str
    db_path: str

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Handle user query by retrieving relevant chunks from the FAISS index
    and generating a response using a language model.
    """
    # Load the FAISS index
    index_file = os.path.join(request.db_path, "faiss_index.bin")
    metadata_file = os.path.join(request.db_path, "metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        raise HTTPException(status_code=404, detail="FAISS index or metadata not found.")

    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embedding for the query
    query_embedding = model.encode([request.query])

    # Search the index for similar chunks
    k = 5  # number of nearest neighbors to retrieve
    distances, indices = index.search(np.array(query_embedding), k)

    # Retrieve the relevant chunks using the indices
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(metadata):
            relevant_chunks.append(metadata[idx]['text'])

    # Combine relevant chunks into a single context
    context = "\n".join(relevant_chunks)
    prompt = f"You are an AI assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {request.query}\nAnswer:"

    # Generate response using the language model
    response = get_llm_response(prompt)

    return {
        "query": request.query,
        "response": response,
        "retrieved_chunks": relevant_chunks
        }