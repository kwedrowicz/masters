from flowshop.functions.conditional_print import print_if


def print_params(options):
    print_if("Params:", boolean=options.printable)
    print_if("Source: {}".format(options.source), boolean=options.printable)
    print_if("Variance percent: {}".format(options.variance), boolean=options.printable)
    print_if("Clusters: {}".format(options.clusters), boolean=options.printable)
    print_if("Estimator: {}".format(options.estimator), boolean=options.printable)
