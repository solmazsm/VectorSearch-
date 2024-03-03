import matplotlib.pyplot as plt


topic_distribution.plot(kind='bar')
plt.xlabel('Topic')
plt.ylabel('Number of Samples')
plt.title('Distribution of Topics in the Dataset')
plt.show()
