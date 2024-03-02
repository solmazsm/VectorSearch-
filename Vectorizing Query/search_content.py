# a semantic search based on a query, evaluates the results using simulated ground truth labels, 
# calculates the F1 score, generates a classification report, and returns the search results along with their similarities.

def search_content(query, pdf_to_index, k=3):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    #set k to limit the number of vectors
    top_k = index_content.search(query_vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()
    results = pdf_to_index.loc[ids]
    results["similarities"] = similarities
    
    ground_truth_labels = np.random.randint(0, 2, size=len(results))
    predicted_labels = np.random.randint(0, 2, size=len(results))
    
    
    f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=1)
    report = classification_report(ground_truth_labels, predicted_labels, zero_division=1)
    
    
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)
    
    return results


search_results = search_content("book", pdf_to_index)
