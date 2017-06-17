import pandas
from numpy import array
from scipy.cluster.vq import kmeans, whiten

import parameters
from functions import map_runs, get_closest_run_to_mean, map_dict_to_list

df = pandas.read_csv('resources/out.csv', sep=';', usecols=parameters.cols)
runs = map_runs(df)

sorted_keys = sorted(runs.keys())
dictListed = map_dict_to_list(runs)

features = array(dictListed)
whitened = whiten(features)
means = kmeans(whitened, parameters.k)

for mean in means[0]:
    print get_closest_run_to_mean(mean, whitened, sorted_keys)
