import numpy as np
import matplotlib.pyplot as plt


# Get query times
query_times = results_df['query_time'].values


sorted_query_times = np.sort(query_times)

#cumulative distribution function
ecdf = np.arange(1, len(sorted_query_times) + 1) / len(sorted_query_times)


plt.figure(figsize=(10, 6))  
plt.plot(sorted_query_times, ecdf, marker='.', linestyle='-', color='b', label='ECDF')  
plt.xlabel('Query Time (seconds)', fontsize=12)  
plt.ylabel('Cumulative Probability', fontsize=12)  
plt.title('Empirical Cumulative Distribution Function (ECDF) of Query Times', fontsize=14) 
plt.grid(True)  
plt.legend(fontsize=12, facecolor='white') 
plt.tight_layout()

plt.savefig('ecdf_plot.svg', format='svg')
plt.show()  
