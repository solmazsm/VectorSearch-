;==========================================
; Title: multi-vector search
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, InputExample
import faiss

# Function to perform multi-vector search
def multi_vector_search(queries, index, pdf_to_index, k=3):
    results = []
    for query_text in queries:
        # Encode the query using the model
        query_vector = model.encode(query_text)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Perform similarity search
        top_k = index.search(query_vector, k)

        # Retrieve and process results
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].tolist()
        docs = pdf_to_index.loc[ids]
        docs["similarities"] = similarities
        results.append(docs)
    return results
