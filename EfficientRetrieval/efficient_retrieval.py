;==========================================
; Title: efficient_retrieval
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

cache_dir = r"C:\Users\Solmaz\.cache"
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)


pdf_subset.reset_index(inplace=True)
pdf_subset.rename(columns={'index': 'id'}, inplace=True)

# Encode
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())

# Create an index for efficient similarity search
pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.index.values).flatten().astype("int")

content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)


