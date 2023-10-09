# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\M GNANESHWARI\Downloads\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
dataset,X
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
#scipy is an open source python libray which contain tools to do clusterning and build the dendogram. 
#we are not going to import whole scipy we are importing only scipy which related to the cluster and hierarachy

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
dendrogram
#we are going to build the dendogram with only one line of code 
#linkage is one of the hieararchicla clustering algorithm & you have to build the linkage on X 
#ward method actually try to minimise the variance on each cluster & in k-means we minimise the sum of squared 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# YOU CAN IMPLETE HEAR FIND ELBOW METOD 

"""
What is Sklearn cluster?
It stands for “Density-based spatial clustering of applications with noise”. This algorithm is based on the intuitive notion of “clusters” & “noise” that clusters are dense regions of the lower density in the data space, separated by lower density regions of data points. Scikit-learn have sklearn. cluster.
"""
# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
hc
y_hc = hc.fit_predict(X)
y_hc
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# PLEASE COMPARE BOTH K-MEANS CLUSTERING vs HIERARCHICAL CLUSTERING
# ASO 


