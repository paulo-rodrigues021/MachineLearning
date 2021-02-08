
import pandas as pd
base = pd.read_csv('LasVegas.csv', sep=';')

# Changing the column's order so then the Score can be the last one. It makes easier to manipulate the dataframe
base = base[['User country', 'Nr. reviews', 'Nr. hotel reviews', 'Helpful votes',
       'Period of stay', 'Traveler type', 'Pool', 'Gym',
       'Tennis court', 'Spa', 'Casino', 'Free internet', 'Hotel name',
       'Hotel stars', 'Nr. rooms', 'User continent', 'Member years',
       'Review month', 'Review weekday', 'Score']]

# Checking the data types
base.dtypes
## Checking for outliers
import matplotlib.pyplot as plt
# creating a categorical boolean mask
numerical_feature_mask = base.dtypes == float
numerical_cols = base.columns[numerical_feature_mask].tolist()
# Printing all the boxplots to identify the outliers
for column in numerical_cols:
    plt.title(column)
    plt.boxplot(base[column])
    plt.show()

# Substituting all outliers for nan values
import numpy as np
base['Member years'].mask(base['Member years'] < 0, np.nan, inplace=True)
base['Nr. hotel reviews'].mask(base['Nr. hotel reviews'] > 40, np.nan, inplace=True)
base['Helpful votes'].mask(base['Helpful votes'] > 180, np.nan, inplace=True)
base['Nr. reviews'].mask(base['Nr. reviews'] > 200, np.nan, inplace=True)

# Getting the data base cleaner from more outliers after the first ouliters were droped
base['Member years'].mask(base['Member years'] > 12, np.nan, inplace=True)
base['Nr. hotel reviews'].mask(base['Nr. hotel reviews'] > 25, np.nan, inplace=True)
base['Helpful votes'].mask(base['Helpful votes'] > 70, np.nan, inplace=True)

# Substituting the nan values with the column's mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
columns = ['Member years', 'Nr. hotel reviews', 'Helpful votes', 'Nr. reviews']
for column in columns:
    base[[column]] = imputer.fit_transform(base[[column]])

# Remove all columns that can disturb the dependent features' transformations
base = base.drop(columns = ['User country', 'Period of stay','Hotel name'])
classes = base.iloc[:,16:17].astype(str)
base = base.iloc[:, :16]

base['Hotel stars'] = base['Hotel stars'].replace(to_replace = '3,5', value = '3.5')
base['Hotel stars'] = base['Hotel stars'].replace(to_replace = '4,5', value = '4.5')
base['Hotel stars'] = base['Hotel stars'].astype(float)

# Splitting data into X and y
X = base.iloc[:,:].values
y = classes.iloc[:,:].values


# Splitting data into train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=42)


# Importing libraries for the ColumnTransformer method to run the data preprocessing methods
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
encoder = OneHotEncoder()
scaler = StandardScaler()

# The Hotel stars column in in the slcaer paramter bc the numbers' size is important
ct = ColumnTransformer([
                        ('onehot_encoder', encoder, [3,4,5,6,7,8,9,11,12,14,15]),
                        ('scaler', scaler, [0,1,2,10,13])],
                        remainder='passthrough')
X_train = ct.fit_transform(X_train).toarray()
X_test = ct.fit_transform(X_test).toarray()

    
# Using PCA to reduce the feature's quantity
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)

pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

components = pca.explained_variance_ratio_.cumsum()


# Importing the model, training and evaluating
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train.ravel())
score_train = classifier.score(X_train, y_train)
predictions = classifier.predict(X_test)
score_test = classifier.score(y_test, predictions)

print('No. of components: ', len(components))
print('Components: ', components)
print()
print('Training score: ', round(score_train * 100, 0))
print('Testing score: ', round(score_test * 100, 0))


## Testing for a known classification
# Defining the features' values
predict = [[11, 4, 13,	'Friends', 
            'NO',	'YES', 'NO', 'NO', 'YES', 'YES',
            3.0, 3773, 'North America', 9, 'January', 'Thursday']]

# transforming using the ct object
predict = ct.transform(predict).toarray() 
# result is the result for the prediction        
result = classifier.predict(pca.transform(predict))

# Getting the classes and the classes probabilities
classes = classifier.classes_
classes_prob = classifier.class_prior_
