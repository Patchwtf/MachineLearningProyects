import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

if __name__ == "__main__":
    df_data = pd.read_csv("./DataSets/felicidad.csv")
    X = df_data.drop(['country', 'score'], axis=1)
    y = df_data['score']

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for train, test in kf.split(X):
        print(f'{train}\n{test}\n', '='*80)