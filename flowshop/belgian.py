import pandas
from flowshop.functions.clustering import cluster_by_estimator, get_labels, get_representatives
from flowshop.functions.common_options import common_options
from flowshop.functions.conditional_print import print_if
from flowshop.functions.data_preparations import remove_zero_rows, remove_duplicates
from flowshop.functions.draw_plot import draw_plots, draw_representatives_bars
from flowshop.functions.pca import pca_reduce
from flowshop.functions.print_params import print_params


def main():
    options = get_options_from_command()

    print_params(options)
    runs = prepare_data(options)
    cluster(runs, options)


def get_options_from_command():
    parser = common_options()

    (options, args) = parser.parse_args()

    return options


def prepare_data(options):
    runs = pandas.read_csv(options.source, sep=',', index_col='run_id', dtype='float')
    runs = runs.fillna(0).astype(float)
    print_if("Loaded data", boolean=options.printable)

    runs = remove_zero_rows(runs, options.printable)
    runs = remove_duplicates(runs, options.printable)

    return runs


def cluster(runs, options):

    runs_total = runs[['total']]
    runs = runs.drop(columns=['total'])

    principal_components = pca_reduce(runs, options.variance)

    estimator = cluster_by_estimator(principal_components, options.estimator, options.clusters)
    labels = get_labels(estimator, principal_components)

    print_if("RunId - cluster id tuples: ", boolean=True)
    run_ids = runs.index.astype(int).tolist()
    print_if(list(zip(run_ids, labels)), boolean=True)

    total = pandas.DataFrame({'labels': labels, 'total': runs_total.total}, index=run_ids)
    idx = total.groupby(['labels'], sort=True)['total'].transform(min) == total['total']
    best_in_clusters = total[idx].sort_values(by=['labels']).drop_duplicates()
    best_in_cluster_runs = best_in_clusters.index.values.tolist()

    centroid_runs = get_representatives(estimator, principal_components, run_ids, labels, options.printable)

    representatives = centroid_runs + best_in_cluster_runs

    draw_plots(principal_components, estimator, labels, 'Metoda belgijska',  total, representatives,
               value_tags=True)
    draw_representatives_bars(representatives, total, 'Metoda belgijska', estimator)

    return centroid_runs


if __name__ == "__main__":
    main()
