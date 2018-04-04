import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from flowshop.conditional_print import print_if


def cluster(file_name, variance_percent=.95, clusters=10, plots=False):

    print_if("Clustering {} for {} clusters with {} variance".format(file_name, clusters, variance_percent), boolean=True)

    df = pd.read_csv('../resources/'+file_name, index_col='run_id')
    total = df['total']

    # normalized_df = (df - df.min())/(df.max() - df.min())
    # standarized_df = (df - df.mean())/df.std()

    pca = PCA(variance_percent)
    x = StandardScaler().fit_transform(df.values.astype(float))
    principalComponents = pca.fit_transform(x)

    # print(principalComponents)
    print_if("PCA reduced to {} dimensions".format(principalComponents.shape[1]), boolean=True)

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(principalComponents)

    # print(kmeans.labels_)
    print_if("RunId - clusted id tuples: ", boolean=True)
    print_if(list(zip(total.index.tolist(), kmeans.labels_)), boolean=True)
    # print(estimator.cluster_centers_)
    if plots:
        plt_x = principalComponents[:, 0]
        if principalComponents.shape[1] > 1:
            plt_y = principalComponents[:, 1]
        else:
            plt_y = [1] * principalComponents.shape[0]

        plt.scatter(plt_x, plt_y)
        plt.title(file_name + " PCA")
        plt.show()

        plt.scatter(plt_x, plt_y, c=kmeans.labels_)
        plt.title(file_name + " clustered")
        plt.show()

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
