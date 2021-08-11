import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("KNN.xlsx")
print(df)

X = df.iloc[: , :-1].values
Y = df.iloc[: , 2].values

X_train , X_test, Y_train , Y_test = train_test_split(X,Y, test_size = 0.20)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train , Y_train)

y_pred = knn.predict(X_test)

print(y_pred)
print(X_test)
