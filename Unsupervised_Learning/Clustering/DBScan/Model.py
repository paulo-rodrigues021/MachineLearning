
import pandas as pd
base = pd.read_csv('nursery.data', header = None)

## Checking for all the data types available in the data set
# for column in base.columns:
#     print(column, '\n')
#     print(base[column].unique())

# Label encoding all the object datatype value
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for column in base.columns:
    base[column] = encoder.fit_transform(base[column])

# Getting the 2 columns of interest
X = base.iloc[:, [3,6]].values

# importing scaler to transform the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

## testing for the eps and min_sample best parameters
# from sklearn.cluster import DBSCAN
# import numpy as np
# eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# samples = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# for ep in eps:
#     for samp in samples:
#         dbscan = DBSCAN(eps = ep, min_samples = samp)
#         predictions = dbscan.fit_predict(X)
#         uniques, quantity = np.unique(predictions, return_counts=True)
#         print('======================')
#         print('Samples: ', samp)
#         print('Eps: ', ep, '\n')
#         print('Uniques: ', uniques, '\n')
#         print('Quantity: ', quantity)
        
# For columsn 6,8
config_list = [{"eps": 0.8, "samp": 2},
              {"eps":0.4, "samp": 15}]

# For columns 3,6
#  eps: 0.9, samples: 2

# importing the model, getting the number of clusters and sizes
from sklearn.cluster import DBSCAN
import numpy as np
dbscan = DBSCAN(eps = 0.9, min_samples = 2)
predictions = dbscan.fit_predict(X)
uni, count = np.unique(predictions, return_counts=True)

# ploting the result
import matplotlib.pyplot as plt
plt.scatter(X[predictions == 0, 0], X[predictions == 0, 1], c='red',label='Cluster 1')
plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1],  c='blue',label='Cluster 2')
plt.scatter(X[predictions == 2, 0], X[predictions == 2, 1], c='orange',label='Cluster 3')
# plt.scatter(X[predictions == 3, 0], X[predictions == 3, 1], s=100, c='green',label='Cluster 4')
# plt.legend()
plt.show()
