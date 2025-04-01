import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    df_data = pd.read_csv('./DataSets/heart.csv')
    print(df_data['target'].describe())

    X = df_data.drop(['target'] , axis=1)
    y = df_data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print(f'Accuracy KNN: {accuracy_score(y_test, knn_pred)}')

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print(f'Accuracy Bagging: {accuracy_score(y_test, bag_pred)}')