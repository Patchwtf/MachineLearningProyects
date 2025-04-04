import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv('./DataSets/heart.csv')
    print(df_heart.head(5))

    df_features = df_heart.drop('target', axis=1, inplace=False) #* X
    df_target = df_heart['target'] #* y

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.30, random_state=42)

    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(X_train_kpca, y_train)
    print(f'Score KPCA: {logistic.score(X_test_kpca, y_test)}')
