
# importing the data (csv) format
import pandas as pd
base = pd.read_csv('credit_card_clients.csv', header = 1)

# Creating a new column for credit card bill amount based on the monthly data given
base['Bill_Total'] = base['BILL_AMT1'] + base['BILL_AMT2'] + \
                    base['BILL_AMT3'] + base['BILL_AMT4'] + \
                    base['BILL_AMT5'] + base['BILL_AMT6']

# Creatnig a new columns for the amount paid from the bill based on the monthly data given
base['Pmt_Total'] = base['PAY_AMT1'] + base['PAY_AMT2'] + \
                    base['PAY_AMT3'] + base['PAY_AMT4'] + \
                    base['PAY_AMT5'] + base['PAY_AMT6']

# Checking the relationship of the 3 main variabels (in amount)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(base[['LIMIT_BAL']], base[['Bill_Total']], base[['Pmt_Total']])
plt.show()

# Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = base.iloc[:, [1,25,26]].values
X = scaler.fit_transform(X)

# Checking the for best number of n_clusters (K) given the curve of Elbow Method
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 21):
    # random_state=0 to make sure the same values are used for the evaluation of K
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('ELBOW METHOD FOR K')
plt.show()

# Predicting the clusters based on the best number of K found (n_clusters = 8)
kmeans = KMeans(n_clusters = 8, random_state = 0)
predictions = kmeans.fit_predict(X)

