"""
Created on Sun Oct  4 12:04:39 2020

K MEANS IS AN UNSUPERVISED ALGORITHM
Only in this project we can compare its results to the 'real' clusters
(because we created the data artifically)

@author: Marc
"""
# Importing matplotlib for visualization
import matplotlib.pyplot as plt

# Variables
centers = 5
clusters = 2

# Create artificial data
from sklearn.datasets import make_blobs
data = make_blobs(n_samples = 500, n_features = 2, centers = centers, cluster_std = 1)

# K Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = clusters)
kmeans.fit(data[0])

# Visualize the results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10, 20))
ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c = kmeans.labels_, cmap = 'rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c = data[1], cmap = 'rainbow')
file_name = 'original' + str(centers) + '_vs_kmeans' + str(clusters) + '.png'
plt.savefig(file_name)
