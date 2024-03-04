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



plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=30, edge_color='gray', with_labels=False)


plt.title("Similarity Network of News Articles")
plt.xlabel("Content")
plt.ylabel("Similarity")
plt.text(-0.1, -0.1, "Nodes represent news articles.\nEdges indicate high similarity \nscores above a threshold.")
plt.show()
