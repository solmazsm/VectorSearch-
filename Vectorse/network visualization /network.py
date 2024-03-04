;==========================================
; Title: multi-vector search
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Vectorize 
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, lowercase=True)
doc_term_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(doc_term_matrix)


threshold = 0.5 

# graph
G = nx.Graph()

#nodes
for i in range(len(documents)):
    G.add_node(i, content=documents[i])

# Add edges based on similarity threshold
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if similarity_matrix[i][j] >= threshold:
            G.add_edge(i, j, weight=similarity_matrix[i][j])

