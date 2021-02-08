
import pandas as pd
base = pd.read_csv('e-shopping.csv', sep=';')

# Dropping all the useless columns
columns = ['year', 'month', 'day', 'session ID', 'country']
for column in columns:
    base = base.drop(columns = column)


from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
base.iloc[:, 3:4] = lab_enc.fit_transform(base.iloc[:, 3:4])

# separating the X and Y
y = base.iloc[:, 0:1].values
X = base.iloc[:, 1:].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
scaler = StandardScaler()
oh_enc = OneHotEncoder()


ct = ColumnTransformer([
    ('scaler', scaler, [7]),
    ('oh_enc', OneHotEncoder(), [0,1,2,3,4,5,7,8])
    ])

X = ct.fit_transform(X)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 20)

from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Importing the model
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 100)

reg.fit(X_train, y_train)
score_train = reg.score(X_train, y_train)

print('Trainning score: ', round(score_train * 100, 2))
