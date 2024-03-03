;==========================================
; Title:  ChromaDB client
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import os
import chromadb
from chromadb.config import Settings


base_directory = r"C:\Users\Solmaz"


relative_directory = ".cache"

persist_directory = os.path.join(base_directory, relative_directory)

Create the ChromaDB client with the updated persist directory
#chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="/content/.."
    )
)

collection.add(
    documents=pdf_subset["title"][:100].tolist(),
    metadatas=[{"topic": topic} for topic in pdf_subset["topic"][:100].tolist()],
    ids=[f"id{x}" for x in range(100)],
)
