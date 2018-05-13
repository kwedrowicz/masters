import pandas
from sklearn import cluster as cl
from sklearn import mixture
from sklearn.metrics import pairwise_distances_argmin_min

from flowshop.functions.conditional_print import print_if


def cluster_by_estimator(data, estimator_name, clusters):
    bandwidth = cl.estimate_bandwidth(data, quantile=.2)

    # create estimator objects
    kmeans = cl.KMeans(n_clusters=clusters)
    spectral = cl.SpectralClustering(n_clusters=clusters)
    ward = cl.AgglomerativeClustering(n_clusters=clusters)
    meanshift = cl.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    dbscan = cl.DBSCAN()
    birch = cl.Birch(n_clusters=clusters)
    guassian = mixture.GaussianMixture(n_components=clusters, covariance_type='full')

    clustering_algorithms = (
        ('KMeans', kmeans),
        ('MeanShift', meanshift),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', guassian)
    )

    chosen_estimator = None

    for name, estimator, in clustering_algorithms:
        if name == estimator_name:
            chosen_estimator = estimator
            estimator.fit(data)
            break

    if chosen_estimator is None:
        raise TypeError('No such estimator')

    return chosen_estimator


def get_labels(estimator, data):
    if hasattr(estimator, 'labels_'):
        return estimator.labels_.astype(pandas.np.int)
    else:
        return estimator.predict(data)


def get_representatives(estimator, data, ids, labels, printable):
    if hasattr(estimator, 'cluster_centers_'):
        means = estimator.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(means, data)
        centroid_runs = [int(ids[index]) for index in closest]
    else:
        df = pandas.DataFrame(data, index=ids)
        grouped = df.assign(labels=labels).groupby('labels')
        medians = grouped.median()
        centroid_runs = []
        for index, row in medians.iterrows():
            closest, _ = pairwise_distances_argmin_min([row.values.tolist() + [index]], grouped.get_group(index))
            centroid_runs.append(grouped.get_group(index).iloc[[closest[0]]].index.tolist()[0])
    print_if("Centroid runs: {}".format(centroid_runs), boolean=printable)

    return centroid_runs
