import pandas
from flowshop.functions.clustering import cluster_by_estimator, get_labels, get_representatives
from flowshop.functions.common_options import common_options
from flowshop.functions.conditional_print import print_if
from flowshop.functions.data_preparations import remove_zero_rows, remove_duplicates
from flowshop.functions.draw_plot import draw_plots
from flowshop.functions.pca import pca_reduce
from flowshop.functions.print_params import print_params
from flowshop.functions.set_divide import set_divide


def main():
    options = get_options_from_command()

    print_params(options)
    runs = prepare_data(options)
    bests, worsts = set_divide(runs, options.quantile, options.printable)

    best_representatives = cluster(bests, options, 'Bests')
    worst_representatives = cluster(worsts, options, 'Worsts')

    all_representatives = sorted(best_representatives + worst_representatives)
    print_if("All representatives: {}".format(all_representatives), options.printable)


def get_options_from_command():
    parser = common_options()
    parser.add_option("-q", "--quantile", dest="quantile", type="float", default=.3,
                      help='quantile to divide data to best and worst sets')

    (options, args) = parser.parse_args()

    return options


def prepare_data(options):
    df = pandas.read_csv(options.source, sep=';', usecols=['run_id', 'instance_id', 'score'],
                         dtype='float')
    runs = df.pivot(index='run_id', columns='instance_id', values='score')
    runs = runs.fillna(0).astype(float)
    print_if("Generated pivoted table", boolean=options.printable)

    runs = remove_zero_rows(runs, options.printable)
    runs = remove_duplicates(runs, options.printable)

    return runs


def cluster(runs, options, title):
    principal_components = pca_reduce(runs, options.variance)

    estimator = cluster_by_estimator(principal_components, options.estimator, options.clusters)

    labels = get_labels(estimator, principal_components)

    print_if("RunId - cluster id tuples: ", boolean=True)
    run_ids = runs.index.astype(int).tolist()
    print_if(list(zip(run_ids, labels)), boolean=True)

    total = pandas.DataFrame({'labels': labels, 'total': runs.total}, index=run_ids)
    idx = total.groupby(['labels'], sort=True)['total'].transform(min) == total['total']
    best_in_clusters = total[idx].sort_values(by=['labels']).drop_duplicates()
    best_in_cluster_runs = best_in_clusters.index.values.tolist()

    centroid_runs = get_representatives(estimator, principal_components, run_ids, labels, options.printable)
    print_if("Best runs: {}".format(best_in_cluster_runs), boolean=True)
    representatives = centroid_runs + best_in_cluster_runs
    draw_plots(principal_components, estimator, labels, title, total, representatives, value_tags=True)

    return representatives


if __name__ == "__main__":
    main()
