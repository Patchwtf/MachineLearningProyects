import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, IncrementalPCA

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

    print(X_train.shape)
    print(y_train.shape)

    #* Por defecto el n_componentes = min(n_muestras, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()
    
    logistic = LogisticRegression(solver='lbfgs')

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    logistic.fit(X_train_pca, y_train)
    print(f'SCORE PCA: {logistic.score(X_test_pca, y_test)}')

    X_train_ipca = ipca.transform(X_train)
    X_test_ipca = ipca.transform(X_test)
    logistic.fit(X_train_ipca, y_train)
    print(f'SCORE IPCA: {logistic.score(X_test_ipca, y_test)}')