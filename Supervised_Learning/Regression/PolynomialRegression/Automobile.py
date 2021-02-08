
# using pandas to read the .data file which has no column names
import pandas as pd
base = pd.read_csv('imports-85.data', header = None)

# setting the list of columns' names from the database website
columns_names = [
                    'sym', 'nor_los', 'make', 'fuel', 'asp', 'numd', 'body', 'dwheels', 'englo', 'whebase', 
                    'len', 'wid', 'height', 'curwhei', 'engty', 'numcyl', 'engsize', 'fuelsys', 'bore', 'stroke', 
                    'com_rat', 'horpwr', 'peak', 'citympg', 'higmpg', 'price'
                ]

base.columns = columns_names

# creating an index so we can easly drop lines
base.index = [x for x in range(1, len(base.values)+1)]

# getting some statistics on the data
base.describe()
base.dtypes

# checking for missing null values
base.isnull().sum()

# after identifying the missing values as '?' we delete the lines that we cant feature engineer
drop_list = []
columns = ['make', 'fuel', 'asp', 'numd', 'body', 'dwheels', 'englo', 'engty', 'numcyl', 'fuelsys', 'price']

# looping throught all the columns and checking for '?' values, adding them to a list to be droped
for column in columns:
    missing = base[base[column] == '?']  
      
    for item in missing.index:
        drop_list.append(item)

for line in drop_list:
    base = base.drop(line)


# replacing '?' to NaN values, so we can use the simple imputer
import numpy as np
base = base.replace(to_replace = '?', value = np.nan)
    

# replacing the '?' values with the column's mean where it is possible
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

columns = ['nor_los', 'bore', 'stroke', 'horpwr', 'peak']
for column in columns:
    base[[column]] = imputer.fit_transform(base[[column]])
    
# transforming the future y axis from object to string and then to integer type
base['price'] = base['price'].astype(str).astype(int)

# Categorical boolean mask
categorical_feature_mask = base.dtypes == object

# filter categorical columns using mask and turn it into a list
categorical_cols = base.columns[categorical_feature_mask].tolist()


from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()

# using label encoder to transform objs into into integers
for column in categorical_cols:
    base[column] = lab_enc.fit_transform(base[column])


# spliting the data base into the features and the objetive
X = base.iloc[:, 0:len(base.columns)-1].values
y = base.iloc[:, len(base.columns)-1].values


# standard scaling all the features (now, they're all numeric)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)

test_list = [.15, .20, .25, .30, .35]
degree_list = [2,3,4]
for tests in test_list:
    for degree in degree_list:
    
        print('Test size: ', tests)
        print('Degree: ', degree)
        
        # Splitting the data into train and test samples
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tests, random_state = 15)
        
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree = degree)
        X_poly = poly.fit_transform(X_train)
        
        reg = LinearRegression()
        # Trainning and getting the score of the training dataset
        reg.fit(X_train, y_train)
        score_train = reg.score(X_train, y_train)
        
        print('Trainning score: ', round(score_train * 100, 0))
        
        # Testing with the test dataset and getting the score
        reg.fit(X_test, y_test)
        score_test = reg.score(X_test, y_test)
        
        print('Testing score: ', round(score_test*100, 0))
        
        predictions = reg.predict(X_test)
        
        # Importing the evaluation metrics and taking the mean absolute error
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test, predictions)
        print('Mean Absolute Error: ', round(mae, 0))
        print()
        print('==========================')











