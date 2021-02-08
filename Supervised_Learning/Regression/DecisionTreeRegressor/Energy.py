
import pandas as pd
base = pd.read_csv('energydata_complete.csv')

# droping the date column because it will not be part of our model
base = base.drop(columns = 'date')

# getting to know some information on data
base.isnull().sum()
base.dtypes
base.describe()
corr = base.corr()

# getting the independent and dependent features
y = base['Appliances']
base = base.drop(columns = 'Appliances')
X = base

# Plotting the data to visually notice any variables' correlation to the objective
# import matplotlib.pyplot as plt
# for column in X.columns:
#     plt.scatter(base[column], y)
#     plt.xlabel(str(column))
#     plt.ylabel('Appliance')
#     plt.show()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Splitting data base into trainning and testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 20)


# importing sklearn's PCA to possibly reduce features. In the below lines we will know how many feat. to use
from sklearn.decomposition import PCA
pca = PCA(n_components = 20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Using these lines to know how many features to use. We can conclue 20 is a good number
components = pca.explained_variance_
components = pca.explained_variance_ratio_
components = pca.explained_variance_ratio_.cumsum()

# Importing the regressor
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

score_train = reg.score(X_train, y_train)
print('Trainning score: ', round(score_train * 100, 0))

reg.fit(X_test, y_test)
score_test = reg.score(X_test, y_test)
print('Testing score: ', round(score_test * 100, 0))

predictions = reg.predict(X_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error: ', mae)

# checking for the accuracy visually
import matplotlib.pyplot as plt
plt.scatter(y_test, predictions)

# checking for errors
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

# pandas confusion matrix
actual = pd.Series(y_test)
# predicted = pd.Series(predictions)
# df_confusion = pd.crosstab(actual, predicted)


