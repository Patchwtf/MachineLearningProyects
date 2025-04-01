import pandas as pd
from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    df_data = pd.read_csv("./DataSets/candy.csv")
    print(df_data.head(5))

    X = df_data.drop('competitorname', axis=1)
    mkmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print(f'Total de centros:\t{len(mkmeans.cluster_centers_)}')
    print('='*80)
    print(f'{mkmeans.predict(X)}')

    df_data['group'] = mkmeans.predict(X)
    print('='*80)
    print(df_data.head(5))