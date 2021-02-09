
# importing the iris dataset from sklearn
from sklearn import datasets
iris = datasets.load_iris()

# Splititng data into dependent and independent variables
X = iris.data[:, :]

# Importing CPA to reduce X to 3 main components
from sklearn.decomposition import PCA
X_red = PCA(n_components=3).fit_transform(X)

# Importing matplot plt and 3d plot methods
import matplotlib.pyplot as plt

# Setting the colors for the graphs
colors = 10 * ['r.','g.','b.','c.','m.']


'''
PLOTITNG 3 MAIN COMPONENTS - NO CLUSTERING APPLIED
'''
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection='3d')
ax.scatter3D(X_red[:,0], X_red[:,1], X_red[:,2])
my_cmap = plt.get_cmap('hsv') # Creating color map
sctt = ax.scatter3D(X_red[:,0], X_red[:,1], X_red[:,2],
                    alpha = 0.8,c = (X_red[:,0] + X_red[:,1] + X_red[:,2]))
plt.title("Iris 3 Main Components - no clustering applied", fontweight ='bold')
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
plt.show()


# Importing the MeanShift algorithm and numpy for the labels
import numpy as np
from sklearn.cluster import MeanShift
ms = MeanShift()


'''
PLOTTING THE 3 MAIN COMPONENTS WITH THE MEANSHIFT CLUSTER
'''
ms.fit(X_red)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))
print('No. of clusters:', n_clusters)
fig = plt.figure(figsize = (10 ,7))
ax = plt.axes(projection='3d')
ax.scatter3D(X_red[:,0], X_red[:,1], X_red[:,2])
ax.scatter3D(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], 
             marker="x", s=100, linewidths=5)
plt.title('Iris - 3 Main Components: MSCluster', fontweight ='bold')
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
plt.show()


'''
PLOTTING THE 2 MAIN COMPONENTS - NO CLUSTERING APPLIED
'''
X_small = PCA(n_components=2).fit_transform(X)
plt.scatter(X_small[:,0], X[:,1])
plt.title('Iris 2 Main Components - no clustering applied', fontweight ='bold')
plt.xlabel('X-axis', fontweight ='bold') 
plt.ylabel('Y-axis', fontweight ='bold') 
plt.show()


'''
PLOTTING THE 2 MAIN COMPONENTS WITH THE MEANSHIFT CLUSTER
'''
ms.fit(X_small)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))
print('No. of clusters:', n_clusters)
plt.scatter(X_small[:,0], X_small[:,1])
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
            marker="x", s=100, linewidths=5)
plt.title('Iris - 2 Main Components: MSCluster', fontweight ='bold')
plt.xlabel('X-axis', fontweight ='bold') 
plt.ylabel('Y-axis', fontweight ='bold') 
plt.show()












