import pandas as pd
from sklearn.cluster import MeanShift

if __name__ == "__main__":
    df_data = pd.read_csv("./DataSets/candy.csv")
    print(df_data.head(5))

    X = df_data.drop('competitorname', axis=1)
    
    meanshift = MeanShift().fit(X)
    print(f'{max(meanshift.labels_)}')
    print('='*80)
    print(meanshift.cluster_centers_)
    
    df_data['meanshift'] = meanshift.labels_
    print('='*80)
    print(df_data.head(5))