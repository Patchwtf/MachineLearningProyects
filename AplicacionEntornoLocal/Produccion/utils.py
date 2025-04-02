import pandas as pd
import joblib

class Utils:
    def load_from_csv(self, path):
        return pd.read_csv(path)
    
    def load_from_sql(self, path):
        pass
    
    def features_target(self, data, drop_cols, y):
        X = data.drop(drop_cols, axis=1)
        y = data[y]
        return X, y
    
    def model_export(self, clf, score):
        print(f"Best score: {score}")
        joblib.dump(clf, f'./models/best_model-{score}.pkl')