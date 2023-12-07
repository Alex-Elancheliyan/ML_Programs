import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#Creating a random data set with 4 groups
data, _ = make_blobs(n_samples=150, centers=4, random_state=42)

#Giving the K values
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

#Plotting the values (clusters)
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
plt.title('K-Means Clustering')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
plt.legend()
plt.show()