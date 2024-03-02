import matplotlib.pyplot as plt
import seaborn as sns




fig, axes = plt.subplots(1, 3, figsize=(18, 6))


sns.scatterplot(x='index_dimension', y='Precision', hue='pretrained_model', style='index_type', data=results_df, ax=axes[0])
axes[0].set_title('Precision vs. Index Dimension')
axes[0].legend(loc='upper left') 


sns.scatterplot(x='similarity_threshold', y='Recall', hue='pretrained_model', style='index_type', data=results_df, ax=axes[1])
axes[1].set_title('Recall vs. Similarity Threshold')
axes[1].legend(loc='upper left') 

sns.scatterplot(x='index_dimension', y='Query Time', hue='pretrained_model', style='index_type', data=results_df, ax=axes[2])
axes[2].set_title('Query Time vs. Index Dimension')
axes[2].legend(loc='upper left') 

plt.tight_layout()
plt.savefig('hyperparameter_analysis.svg', format='svg', bbox_inches='tight')
plt.show()
