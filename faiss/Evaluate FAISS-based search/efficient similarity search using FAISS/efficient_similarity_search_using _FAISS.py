;==========================================
; Title: efficient similarity search using FAISS
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;========================================== 
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

index_faiss = faiss.IndexFlatIP(dim)
index_faiss.add(title_embeddings)
