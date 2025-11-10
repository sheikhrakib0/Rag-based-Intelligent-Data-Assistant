import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
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
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Chunk the text
    chunks = chunk_text(text)

    # Clean the chunks
    cleaned_chunks = [clean_chunk(chunk) for chunk in chunks]

    # Generate embeddings
    print(f"Generating embeddings for {len(cleaned_chunks)} chunks...")
    embeddings = model.encode(cleaned_chunks, show_progress_bar=True)

    # Initialize or load FAISS index
    dimension = embeddings.shape[1]
    index_file = os.path.join(db_path, "faiss_index.bin")
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        with open(os.path.join(db_path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(dimension)
        metadata = []

    # Add embeddings to the index and store metadata
    start_id = index.ntotal
    index.add(embeddings) # type: ignore
    for i in range(len(cleaned_chunks)):
        metadata.append({
            "file_id": file_id,
            "chunk_id": start_id + i,
            "text": cleaned_chunks[i]
        })

    # Save the updated index and metadata
    faiss.write_index(index, index_file)
    with open(os.path.join(db_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)