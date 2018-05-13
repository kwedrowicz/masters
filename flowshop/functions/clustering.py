from sklearn import cluster as cl
from sklearn import mixture


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
