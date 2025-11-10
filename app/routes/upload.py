import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException
from ..core.file_parser import extract_text_from_file
from ..core.embeddings import generate_embedding


# initializing the router
router = APIRouter()

# directory to save uploaded files
UPLOAD_DIRECTORY = "data/uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, extract its text, generate embeddings, and store them.
    """
    try:
        # 1. Save the file locally
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}_{file.filename}")

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2. Extract text from the file
        extracted_text = extract_text_from_file(file_path)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from the uploaded file.")
        
        # 3. Generate embeddings for the extracted text and store them
        db_path = "data/vector_db"
        os.makedirs(db_path, exist_ok=True)
        generate_embedding(extracted_text, file_id, db_path)

        return {
            "message": "File uploaded and processed successfully.", "file_id": file_id,
            "filename": file.filename,
            "stored_in": file_path
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file upload: {str(e)}")