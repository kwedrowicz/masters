import sys
import pandas
from sklearn import cluster as cl
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

from flowshop.conditional_print import print_if
from flowshop.draw_plot import draw_plots


def main(argv):
    variance = .8
    clusters = 10
    source = "resources/csv/space.csv"
    printable = True

    print_if("Params:", boolean=printable)
    print_if("Source: {}".format(source), boolean=printable)
    print_if("Variance percent: {}".format(variance), boolean=printable)
    print_if("Clusters: {}".format(clusters), boolean=printable)

    # MAPPING START
    runs = pandas.read_csv(source, sep=',', index_col='run_id', dtype='float')
    runs = runs.fillna(0).astype(float)
    print_if("Generated pivoted table", boolean=printable)

    zero_rows_count = runs.shape[0]
    runs = runs[(runs.T != 0).any()]
    non_zero_rows_count = runs.shape[0]

    print_if("Removed {} zero rows from {} rows".format(zero_rows_count - non_zero_rows_count, zero_rows_count),
             boolean=printable)

    all_rows = runs.shape[0]
    deduplicated_runs = runs.drop_duplicates()
    deduplicated_rows = deduplicated_runs.shape[0]
    print_if("Removed {} duplicates from {} rows".format(all_rows - deduplicated_rows, all_rows), boolean=printable)
    # MAPPING END

    # CLUSTERING START

    pca = PCA(variance)
    x = StandardScaler().fit_transform(deduplicated_runs.values.astype(float))
    principalComponents = pca.fit_transform(x)
    print_if("PCA reduced to {} dimensions".format(principalComponents.shape[1]), boolean=True)
    estimator = cl.KMeans(n_clusters=clusters)
    estimator.fit(principalComponents)
    print_if("RunId - clusted id tuples: ", boolean=True)
    run_ids = deduplicated_runs.index.astype(int).tolist()
    print_if(list(zip(run_ids, estimator.labels_)), boolean=True)
    means = estimator.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(means, principalComponents)
    centroid_runs = [int(run_ids[index]) for index in closest]
    print_if("Centroid runs: {}".format(centroid_runs), boolean=True)
    # CLUSTERING END

    # PLOTTING START
    draw_plots(principalComponents, estimator, source, estimator.labels_, [], centroid_runs, value_tags=False)
    # PLOTTING END


if __name__ == "__main__":
    main(sys.argv[1:])
