import os
from typing import List
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into chunks of specified size with overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# clean the chunks
def clean_chunk(chunk: str) -> str:
    """Remove stop words and extra whitespace from a text chunk."""
    words = chunk.split()
    cleaned_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    cleaned_chunk = " ".join(cleaned_words)
    return cleaned_chunk

# generate and store embeddings
def generate_embedding(text: str, file_id: str, db_path: str):
    """
    Generate embeddings for the given text and store them in a FAISS index.
    Each chunk is associated with the provided file_id.
    """
    # Load pre-trained SentenceTransformer model
    #model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Chunk the text
    chunks = chunk_text(text)

    # Clean the chunks
    cleaned_chunks = [
        Document(
            page_content = clean_chunk(chunk),
            metadata={'file_id': file_id}
        )
        for chunk in chunks
    ]

    # Generate embeddings
    print(f"Generating embeddings for {len(cleaned_chunks)} chunks...")

    index_path = os.path.join(db_path, "faiss_index.bin")
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}...")
        vectorstore = FAISS.load_local(
            folder_path=db_path,
            index_name="faiss_index.bin",
            embeddings=model,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"Creating new FAISS index at {index_path}...")
        vectorstore = FAISS.from_documents(documents=cleaned_chunks, embedding=model)

    # Save the updated index and metadata
    vectorstore.save_local(
        folder_path=db_path,
        index_name="faiss_index.bin"
    )
    print(f"Faiss index saved successfully at {index_path}.")
