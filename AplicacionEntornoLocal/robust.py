import pandas as pd
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter("ignore")

if __name__=="__main__":
    df_data = pd.read_csv('./DataSets/felicidad_outliers.csv')
    print(df_data.head(5))

    X = df_data.drop(['country', 'score'] , axis=1)
    y = df_data['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'Hubber': HuberRegressor(epsilon=1.35)}
    
    for name, estimador in estimadores.items():
       estimador.fit(X_train, y_train)
       predicciones = estimador.predict(X_test)
       mse = mean_squared_error(y_test, predicciones)
       print('='*64, f'\nMSE {name}: {mse}')