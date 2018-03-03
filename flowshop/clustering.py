import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler


def cluster(file_name, variance_percent=.95, clusters=10):
    df = pd.read_csv('../resources/'+file_name, index_col='run_id')
    total = df['total']

    # normalized_df = (df - df.min())/(df.max() - df.min())
    # standarized_df = (df - df.mean())/df.std()

    pca = PCA(variance_percent)
    x = StandardScaler().fit_transform(df.values.astype(float))
    principalComponents = pca.fit_transform(x)

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(principalComponents)

    # print(estimator.labels_)
    # print(estimator.cluster_centers_)

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, principalComponents)
    run_ids = total.index.tolist()
    centroid_runs = [run_ids[index] for index in closest]
    # print(closest)
    # print(centroid_runs)

    total = total.to_frame().assign(cluster=kmeans.labels_)

    idx = total.groupby(['cluster'], sort=True)['total'].transform(min) == total['total']
    best_in_clusters = total[idx].sort_values(by=['cluster'])
    best_in_cluster_runs = best_in_clusters.index.values.tolist()

    return centroid_runs + best_in_cluster_runs
