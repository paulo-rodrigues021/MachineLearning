
import pandas as pd
base = pd.read_csv('crx.data', header=None)

base.isnull().sum()

# for column in base.columns:
#     print('Columns: ', column)
#     print(base[column].unique())
#     print()
    
delete_rows = [0,1,3,4,5,6,8,9,11,12,13,15]
for i in delete_rows:
    indexes = base[base[i] == "?"]
    indexes = indexes.index.tolist()
    
    base = base.drop(indexes)
    
import numpy as np
base = base.replace(to_replace="?", value=np.nan)


from sklearn.impute import SimpleImputer
transform_columns = [2,7,10,14]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
for item in transform_columns:
    base[[item]] = imputer.fit_transform(base[[item]])

# Categorical boolean mask
categorical_feature_mask = base.dtypes == object

# filter categorical columns using mask and turn it into a list
categorical_cols = base.columns[categorical_feature_mask].tolist()


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for column in categorical_cols:
    base[column] = encoder.fit_transform(base[column])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# columns = [2,7,10,14]
X = base.iloc[:, :15].values
y = base.iloc[:, 15:16].values

X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20, random_state=50)


from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=1)
model.fit(X_train, y_train.ravel())
score_train = model.score(X_train, y_train)
print('Train score: ', round(score_train * 100, 0))


from sklearn.metrics import confusion_matrix, accuracy_score
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
matrix = confusion_matrix(y_test, pred)
print('Test score: ', round(accuracy * 100, 0))

