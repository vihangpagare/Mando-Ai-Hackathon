import argparse
import os
import shutil
import json
import pandas as pd
from typing import List
from PIL import Image
import pytesseract
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders import (
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
PUBLIC_DATA_PATH = "data/public"
PRIVATE_DATA_PATH = "data/private"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents() -> List[Document]:
    document_list = []
    
    for data_path in [PUBLIC_DATA_PATH, PRIVATE_DATA_PATH]:
        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                try:
                    if file_ext == ".pdf":
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    elif file_ext in (".png", ".jpg", ".jpeg"):
                        text = perform_ocr(file_path)
                        docs = [Document(page_content=text, 
                                      metadata={"source": file_path, "page": 0})]
                    elif file_ext == ".docx":
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                    elif file_ext == ".pptx":
                        loader = UnstructuredPowerPointLoader(file_path)
                        docs = loader.load()
                    elif file_ext == ".xlsx":
                        loader = UnstructuredExcelLoader(file_path)
                        docs = loader.load()
                    elif file_ext == ".txt":
                        loader = TextLoader(file_path)
                        docs = loader.load()
                    elif file_ext == ".csv":
                        docs = [load_csv(file_path)]
                    elif file_ext == ".json":
                        docs = [load_json(file_path)]
                    else:
                        continue

                    # Normalize metadata
                    for doc in docs:
                        doc.metadata.setdefault("page", 0)
                        doc.metadata["source"] = file_path
                    
                    document_list.extend(docs)

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return document_list

def perform_ocr(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR failed for {image_path}: {str(e)}")
        return ""

def load_csv(file_path: str) -> Document:
    try:
        df = pd.read_csv(file_path)
        return Document(
            page_content=df.to_csv(index=False),
            metadata={"source": file_path, "page": 0}
        )
    except Exception as e:
        print(f"Error loading CSV {file_path}: {str(e)}")
        return Document(page_content="", metadata={"source": file_path})

def load_json(file_path: str) -> Document:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return Document(
            page_content=json.dumps(data, indent=2),
            metadata={"source": file_path, "page": 0}
        )
    except Exception as e:
        print(f"Error loading JSON {file_path}: {str(e)}")
        return Document(page_content="", metadata={"source": file_path})

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get()["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids 
                 if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new documents")
        db.add_documents(
            new_chunks,
            ids=[chunk.metadata["id"] for chunk in new_chunks]
        )
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
