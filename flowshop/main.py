import getopt
import sys
from os import path

from flowshop.clustering import cluster
from flowshop.conditional_print import print_if
from flowshop.mapping import prepare_runs


def main(argv):
    quantile = .7
    variance = .95
    clusters = 5
    printable = False
    try:
        opts, args = getopt.getopt(argv, "hpq:v:c:", ["quantile=", "variance=", "clusters="])
    except getopt.GetoptError:
        print('main.py --quantile .7 --variance .95 --clusters 5')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('main.py --quantile .7 --variance .95 --clusters 5')
            sys.exit()
        elif opt == '-p':
            printable = True
        elif opt == '--quantile':
            quantile = float(arg)
        elif opt == '--variance':
            variance = float(arg)
        elif opt == '--clusters':
            clusters = int(arg)
        else:
            raise IOError('Option not found')

    print_if("Params:", boolean=printable)
    print_if("Quantile: ", quantile, boolean=printable)
    print_if("Variance percent: ", variance, boolean=printable)
    print_if("Clusters: ", clusters, boolean=printable)

    prepare_runs('../resources/csv/space.csv', quantile, printable, flattened=True)
    best_run_ids = cluster(path.abspath('../resources/csv/flowshop_best.csv'), variance, clusters)
    worst_run_ids = cluster(path.abspath('../resources/csv/flowshop_worst.csv'), variance, clusters)

    run_ids = sorted(best_run_ids+worst_run_ids)

    print(run_ids)


if __name__ == "__main__":
    main(sys.argv[1:])
