import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    df_data = pd.read_csv("./DataSets/felicidad.csv")
    print(df_data.head(2))
    X = df_data.drop(columns=['country', 'rank','score'], axis=1)
    y = df_data['score']
    reg = RandomForestRegressor()
    parametros = {
        'n_estimators': range(4,16),
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': range(2, 10),
    }

    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_squared_error').fit(X, y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))