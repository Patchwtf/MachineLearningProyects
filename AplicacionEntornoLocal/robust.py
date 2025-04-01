import pandas as pd
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__=="__main__":
    df_data = pd.read_csv('./DataSets/felicidad_outliers.csv')
    print(df_data.head(5))

    X = df_data.drop(['country', 'score'] , axis=1)
    y = df_data['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'Huber': HuberRegressor(epsilon=1.35)}