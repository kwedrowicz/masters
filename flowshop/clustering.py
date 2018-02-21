import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../resources/flowshop_best.csv', index_col='run_id')
total = df['total']

# normalized_df = (df - df.min())/(df.max() - df.min())
# standarized_df = (df - df.mean())/df.std()

pca = PCA(.95)
x = StandardScaler().fit_transform(df.values)
principalComponents = pca.fit_transform(x)

kmeans = KMeans(n_clusters=10)
kmeans.fit(principalComponents)

# print(estimator.labels_)
# print(estimator.cluster_centers_)

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, principalComponents)
run_ids = total.index.tolist()
centroid_runs = [run_ids[index] for index in closest]
print(closest)
print(centroid_runs)


total = total.to_frame().assign(cluster=kmeans.labels_)

idx = total.groupby(['cluster'], sort=True)['total'].transform(max) == total['total']
best_in_clusters = total[idx].sort_values(by=['cluster'])

print(best_in_clusters)
