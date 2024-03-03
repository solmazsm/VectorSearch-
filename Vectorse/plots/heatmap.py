;==========================================
; Title: heatmap
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# Plot heatmap for FAISS (NewsCatcher)
sns.heatmap(similarity_matrix_faiss, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=axs[0])
axs[0].set_title('NewsCatcher Similarity Heatmap (FAISS)')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('NewsCatcher')
sns.heatmap(similarity_matrix_hnswlib, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=axs[1])
axs[1].set_title('NewsCatcher Similarity Heatmap (HNSWlib)')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('NewsCatcher')

# Plot heatmap for FAISS (All the News)
sns.heatmap(similarity_matrix_faiss, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=axs[2])
axs[2].set_title('All the News Similarity Heatmap (FAISS)')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('All the News')
sns.heatmap(similarity_matrix_hnswlib, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=axs[3])
axs[3].set_title('All the News Similarity Heatmap (HNSWlib)')
axs[3].set_xlabel('Index')
axs[3].set_ylabel('All the News')

plt.tight_layout()
plt.savefig('heatmaps.svg', format='svg')
plt.show()
