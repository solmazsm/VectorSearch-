# multi-vector search
def multi_vector_search(queries, index, pdf_to_index, k=3):
    results = []
    for query_text in queries:
        # Encode the query using the model
        query_vector = model.encode([query_text])
        faiss.normalize_L2(query_vector)

        # Perform similarity search
        top_k = index.search(query_vector, k)

        # Retrieve 
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].tolist()
        docs = pdf_to_index.loc[ids]
        docs["similarities"] = similarities
        results.append(docs)
    return results
