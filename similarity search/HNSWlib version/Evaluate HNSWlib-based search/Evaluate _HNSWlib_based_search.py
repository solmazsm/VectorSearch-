;==========================================
; Title:  search_similar_documents_hnswlib
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

for query, relevant_ids in ground_truth.items():
    similar_docs_hnswlib = search_similar_documents_hnswlib(query, index_hnswlib, pdf_subset)
    precision, recall = evaluate_performance(relevant_ids, similar_docs_hnswlib)
    hnswlib_results.append({'query': query, 'precision': precision, 'recall': recall})
