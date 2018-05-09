import pandas as pd
from sklearn import cluster as cl
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from flowshop.conditional_print import print_if
from flowshop.draw_plot import draw_plots


def cluster(file_path, variance_percent=.95, clusters=10, plots=True, algorithm='KMeans', bests=True, value_tags=True):
    print_if("Clustering {} for {} clusters with {} variance".format(file_path, clusters, variance_percent),
             boolean=True)

    df = pd.read_csv(file_path, index_col='run_id')
    total = df['total']

    # normalized_df = (df - df.min())/(df.max() - df.min())
    # standarized_df = (df - df.mean())/df.std()

    pca = PCA(variance_percent)
    x = StandardScaler().fit_transform(df.values.astype(float))
    principalComponents = pca.fit_transform(x)

    bandwidth = cl.estimate_bandwidth(principalComponents, quantile=.2)

    # print(principalComponents)
    print_if("PCA reduced to {} dimensions".format(principalComponents.shape[1]), boolean=True)

    # create estimator objects
    kmeans = cl.KMeans(n_clusters=clusters)
    spectral = cl.SpectralClustering(n_clusters=clusters)
    # ward = cl.AgglomerativeClustering(n_clusters=clusters)
    meanshift = cl.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # dbscan = cl.DBSCAN()
    birch = cl.Birch(n_clusters=clusters)
    guassian = mixture.GaussianMixture(n_components=clusters, covariance_type='full')

    clustering_algorithms = (
        ('KMeans', kmeans),
        ('MeanShift', meanshift),
        # ('SpectralClustering', spectral),
        # ('Ward', ward),
        # ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', guassian)
    )

    chosenEstimator = None

    for name, estimator, in clustering_algorithms:
        if name == algorithm:
            chosenEstimator = estimator
            estimator.fit(principalComponents)
            break

    # print(estimator.labels_)
    print_if("RunId - clusted id tuples: ", boolean=True)


    if hasattr(chosenEstimator, 'labels_'):
        labels = chosenEstimator.labels_.astype(pd.np.int)
    else:
        labels = chosenEstimator.predict(principalComponents)

    print_if(list(zip(total.index.tolist(), labels)), boolean=True)
    # print(estimator.cluster_centers_)

    if hasattr(chosenEstimator, 'cluster_centers_'):
        means = chosenEstimator.cluster_centers_
    elif hasattr(chosenEstimator, 'means_'):
        means = chosenEstimator.means_
    elif hasattr(chosenEstimator, 'subcluster_centers_'):
        means = chosenEstimator.subcluster_centers_
    elif hasattr(chosenEstimator, 'core_sample_indices_'):
        means = chosenEstimator.core_sample_indices_
    else:
        raise TypeError("DUPA")

    closest, _ = pairwise_distances_argmin_min(means, principalComponents)
    run_ids = total.index.tolist()
    centroid_runs = [run_ids[index] for index in closest]
    # print(closest)
    # print(centroid_runs)

    total = total.to_frame().assign(cluster=labels)

    if bests:
        idx = total.groupby(['cluster'], sort=True)['total'].transform(min) == total['total']
        best_in_clusters = total[idx].sort_values(by=['cluster'])
        best_in_cluster_runs = best_in_clusters.index.values.tolist()
        clusters_representatives = centroid_runs + best_in_cluster_runs
    else:
        clusters_representatives = centroid_runs

    if plots:
        draw_plots(principalComponents, chosenEstimator, file_path, labels, total, clusters_representatives, value_tags=value_tags)

    return clusters_representatives
