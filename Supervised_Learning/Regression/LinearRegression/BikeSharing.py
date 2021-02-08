
'''
LINEAR REGRESSION MODEL
'''
print('LINEAR REGRESSION MODEL','\n')

# importing pandas do read the data file in csv
import pandas as pd
base = pd.read_csv('bike_sharing_day.csv')

correlations = base.corr()

# Dropping the useless and redundant features by sight
base = base.drop(columns = ['dteday', 'casual', 'registered'])

# setting X and y values without the columns names
X = base.iloc[:, :len(base.columns)-1].values
y = base.iloc[:, len(base.columns)-1].values


# splitting the data into train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 42)


# Importing the linear model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# Trainning the model
reg.fit(X_train, y_train)

# Getting to know the score of the training model
score_train = reg.score(X_train, y_train)
print('Linear Regression score train: ', round(score_train * 100, 4))

# getting to know the error per data
predictions = reg.predict(X_test)
result = abs(y_test - predictions)

# Getting to know the score of the test model
score_test = reg.score(X_test, y_test)
print('Linear Regression score test: ', round(score_test * 100, 4))

# Importing the evaluation metrics and taking the mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error: ', round(mae, 0))

# Printing the intercept an dcoefficients
intercept = reg.intercept_
coef = reg.coef_
print('===================================================','\n')

