from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
# langchain imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List
# other imports
from app.core.llm import get_llm_response

router = APIRouter()

# Request schema
class QueryRequest(BaseModel):
    query: str
    db_path: str

# embedding model
embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Handle user query by retrieving relevant chunks from the FAISS index
    and generating a response using a language model.
    """
    # Load the FAISS index
    #index_file = os.path.join(request.db_path, "faiss_index.bin")

    try:
        vectorstore = FAISS.load_local(
            folder_path=request.db_path,
            index_name="faiss_index.bin",
            embeddings= embed_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {e}")

    # Search the index for similar chunks
    k = 3  # number of nearest neighbors to retrieve
    retrirever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Retrieve the relevant chunks using the indices
    retrieved_docs: List[Document] = retrirever.invoke(request.query)
    relevant_chunks = []
    for doc in retrieved_docs:
        relevant_chunks.append(doc.page_content)

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