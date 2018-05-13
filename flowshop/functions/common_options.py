from optparse import OptionParser


def common_options():
    parser = OptionParser()
    parser.add_option("-s", "--source", dest="source", type="string", default="resources/csv/flowshop_raw.csv",
                      help="path to source file")
    parser.add_option("-v", "--variance", dest="variance", type="float", default=.8, help="variance for PCA")
    parser.add_option("-c", "--clusters", dest="clusters", type="int", default=5, help="number of clusters")
    parser.add_option("-p", "--printable", dest="printable", action="store_true", default=False,
                      help="print additional info during executing")
    parser.add_option("-e", "--estimator", dest="estimator", default="KMeans", help="estimator for clustering")

    return parser
