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

    df_features = df_heart.drop('target', ax=1) #* X
    df_target = df_heart['target'] #* y

    df_features = StandardScaler().fit_transform(df_features)
    df_target = StandardScaler().fit_transform(df_target)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, train_size=0.30, random_state=42)