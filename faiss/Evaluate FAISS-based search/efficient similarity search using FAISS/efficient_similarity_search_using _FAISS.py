index_faiss = faiss.IndexFlatIP(dim)
index_faiss.add(title_embeddings)
