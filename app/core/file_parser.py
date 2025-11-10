import os
import pandas as pd
import pymupdf # PyMyPDF for PDF parsing
import fitz  # PyMuPDF

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a given file based on its type.
    Supports .txt and .pdf files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    file_ext = os.path.splitext(file_path)[1].lower()


    if file_ext == ".txt":
        return extract_text_from_txt(file_path)
    elif file_ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_ext == ".csv":
        return extract_text_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a .txt file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading .txt file: {str(e)}")
    return clean_text(text)



def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a .pdf file using PyMuPDF."""
    pages_text = []
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                pages_text.append(page.get_text("text"))
    except Exception as e:
        raise RuntimeError(f"Error reading .pdf file: {str(e)}")
    text = "\n".join(pages_text)
    return clean_text(text)

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from a .csv file by concatenating all cell values."""
    text = ""
    try:
        df = pd.read_csv(file_path)
        text = " ".join(df.astype(str).values.flatten())
    except Exception as e:
        raise RuntimeError(f"Error reading .csv file: {str(e)}")
    return clean_text(text)

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Basic cleaning: remove extra whitespace
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()