grouped_results = results_df.groupby(["index_dimension", "similarity_threshold", "pretrained_model"]).mean().reset_index()

# Identify combinations with the highest precision, recall, and lowest query time
best_precision = grouped_results[grouped_results["precision"] == grouped_results["precision"].max()]
best_recall = grouped_results[grouped_results["recall"] == grouped_results["recall"].max()]
fastest_query_time = grouped_results[grouped_results["query_time"] == grouped_results["query_time"].min()]

print("Best Precision:")
print(best_precision)
print("\nBest Recall:")
print(best_recall)
print("\nFastest Query Time:")
print(fastest_query_time)
