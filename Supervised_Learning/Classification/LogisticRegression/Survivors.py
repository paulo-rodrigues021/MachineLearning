
import pandas as pd
base = pd.read_csv('ThoraricSurgery.arff', header=None)

columns_names = ['DGN', 'PRE4', 'PRE5',
                'PRE6', 'PRE7', 'PRE8',
                'PRE9', 'PRE10', 'PRE11',
                'PRE14', 'PRE17', 'PRE19',
                'PRE25', 'PRE30', 'PRE32',
                'AGE', 'Risk1Y']

# Adding the columns' names
base.columns = columns_names

# Checking the data base
base.dtypes

## Checking for unexpected values in data base
# for i in base.columns:
#     print(i)
#     print(base[i].unique(),'\n')

# Getting some shape of the numerical data
base['PRE4'].plot(kind = 'density')
base['PRE5'].plot(kind = 'density')
base['AGE'].plot(kind= 'density')

# Checking for outliers
import matplotlib.pyplot as plt
plt.boxplot(base['PRE4'])
plt.boxplot(base['PRE5'])
plt.boxplot(base['AGE'])


# Substituting the outliers for np.nan values
import numpy as np
base['PRE4'].mask(base['PRE4'] >= 6, np.nan, inplace=True)
base['AGE'].mask(base['AGE'] < 40, np.nan, inplace=True)


# Substituting the np.nan values for each columns' mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
base[['PRE4']] = imputer.fit_transform(base[['PRE4']])
base[['AGE']] = imputer.fit_transform(base[['AGE']])


# Standard Scaling all the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
base[['PRE4']] = scaler.fit_transform(base[['PRE4']])
base[['PRE5']] = scaler.fit_transform(base[['PRE5']])
base[['AGE']] = scaler.fit_transform(base[['AGE']])

# setting X and y
X = base.iloc[:, :17].values
y = base.iloc[:, 16:17].values


# transofrming all string/object data into numerical
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
columns = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
for i in columns:
    X[:, i] = encoder.fit_transform(X[:, i])
y = encoder.fit_transform(y.ravel())


# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.5, random_state=42)


# Importing the model
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train, y_train)
score_train = reg.score(X_train, y_train)


# Checking for the models' accuracy and loss
from sklearn.metrics import accuracy_score, hamming_loss
pred = reg.predict(X_test)
score_test = accuracy_score(y_test, pred)
ham_loss = hamming_loss(y_test, pred)


















