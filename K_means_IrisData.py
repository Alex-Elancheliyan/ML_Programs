import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Loading the iris dataset from Sk_learn and splitting the data into data and target.
iris = load_iris()
data = iris.data
target = iris.target
data_standardized = StandardScaler().fit_transform(data)

#Specifying the K values (i.e no of clusters we wanted) and fit the model.
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_standardized)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_standardized)

#Plotting the values
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()