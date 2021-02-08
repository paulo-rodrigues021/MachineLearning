
import pandas as pd
base = pd.read_csv('breast-cancer.data', sep=',', header=None)

columns_names = ['Rec_NoRec', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast',
                 'breast_squad', 'irradiant']

# Adding names to the columns
base.columns = columns_names

# # checking all the values
# for column in base.columns:
#     uniques = base[column].unique()
#     print(str(column))
#     print(uniques)
#     print()
    

X = base.iloc[:,1:10].values
y = base.iloc[:, 0:1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)

# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()

# label encoding all the columns
for i in range(9):
    X_train[:,i] = encoder.fit_transform(X_train[:,i])
    X_test[:,i] = encoder.fit_transform(X_test[:,i])
    
# label encoding the target variable
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

y_train = y_train.astype(int).reshape(-1,1)
y_test = y_test.astype(int).reshape(-1,1)

# importing the model and printing the training and testing scores to evaluate the model
from sklearn.tree import DecisionTreeClassifier
clas = DecisionTreeClassifier()
clas.fit(X_train, y_train.ravel())
print("Features' importance", clas.feature_importances_)
score_train = clas.score(X_train, y_train)
print('Training score: ', round(score_train * 100, 0))


predictions = clas.predict(X_test)
score_test = clas.score(y_test, predictions)
print('Test score: ', round(score_test * 100, 0))


