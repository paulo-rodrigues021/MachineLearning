
import pandas as pd
base = pd.read_csv('iris.data', header=None)

base.head()
base.dtypes
base.describe()


import seaborn as sns
sns.heatmap(base.corr())


X = base.iloc[:, :4].values
y = base.iloc[:, 4].values


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
score_train = model.score(X_train, y_train)

pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
score_acc = accuracy_score(y_test, pred)
matrix = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)
