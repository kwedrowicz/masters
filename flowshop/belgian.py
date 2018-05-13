from optparse import OptionParser
import pandas
from sklearn.metrics import pairwise_distances_argmin_min

from flowshop.functions.clustering import cluster_by_estimator
from flowshop.functions.conditional_print import print_if
from flowshop.functions.draw_plot import draw_plots
from flowshop.functions.pca import pca_reduce


def main():
    options = get_options_from_command()

    print_params(options)
    runs = prepare_data(options)
    cluster(runs, options)


def get_options_from_command():
    parser = OptionParser()
    parser.add_option("-s", "--source", dest="source", type="string", default="resources/csv/space.csv",
                      help="path to source file")
    parser.add_option("-v", "--variance", dest="variance", type="float", default=.8, help="variance for PCA")
    parser.add_option("-c", "--clusters", dest="clusters", type="int", default=10, help="number of clusters")
    parser.add_option("-p", "--printable", dest="printable", action="store_true", default=False,
                      help="print additional info during executing")
    parser.add_option("-e", "--estimator", dest="estimator", default="KMeans", help="estimator for clustering")

    (options, args) = parser.parse_args()

    return options


def print_params(options):
    print_if("Params:", boolean=options.printable)
    print_if("Source: {}".format(options.source), boolean=options.printable)
    print_if("Variance percent: {}".format(options.variance), boolean=options.printable)
    print_if("Clusters: {}".format(options.clusters), boolean=options.printable)


def prepare_data(options):
    runs = pandas.read_csv(options.source, sep=',', index_col='run_id', dtype='float')
    runs = runs.fillna(0).astype(float)
    print_if("Generated pivoted table", boolean=options.printable)

    zero_rows_count = runs.shape[0]
    runs = runs[(runs.T != 0).any()]
    non_zero_rows_count = runs.shape[0]

    print_if("Removed {} zero rows from {} rows".format(zero_rows_count - non_zero_rows_count, zero_rows_count),
             boolean=options.printable)

    all_rows = runs.shape[0]
    deduplicated_runs = runs.drop_duplicates()
    deduplicated_rows = deduplicated_runs.shape[0]
    print_if("Removed {} duplicates from {} rows".format(all_rows - deduplicated_rows, all_rows),
             boolean=options.printable)

    return deduplicated_runs


def cluster(runs, options):
    principal_components = pca_reduce(runs, options.variance)

    estimator = cluster_by_estimator(principal_components, options.estimator, options.clusters)

    if hasattr(estimator, 'labels_'):
        labels = estimator.labels_.astype(pandas.np.int)
    else:
        labels = estimator.predict(principal_components)

    print_if("RunId - cluster id tuples: ", boolean=True)
    run_ids = runs.index.astype(int).tolist()
    print_if(list(zip(run_ids, labels)), boolean=True)

    total = pandas.DataFrame({'labels': labels}, index=run_ids)

    if hasattr(estimator, 'cluster_centers_'):
        means = estimator.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(means, principal_components)
        centroid_runs = [int(run_ids[index]) for index in closest]
    else:
        df = pandas.DataFrame(principal_components, index=run_ids)
        grouped = df.assign(labels=labels).groupby('labels')
        medians = grouped.median()
        centroid_runs = []
        for index, row in medians.iterrows():
            closest, _ = pairwise_distances_argmin_min([row.values.tolist() + [index]], grouped.get_group(index))
            centroid_runs.append(grouped.get_group(index).iloc[[closest[0]]].index.tolist()[0])

    print_if("Centroid runs: {}".format(centroid_runs), boolean=True)
    draw_plots(principal_components, estimator, labels, 'Belgian',  total, centroid_runs,
               value_tags=False)

    return centroid_runs


if __name__ == "__main__":
    main()
